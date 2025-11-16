"""
RL Policy Optimizer: manages the policy network and coordinates optimization
"""

import torch
import torch.nn.functional as F
from tqdm import trange
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.optim as optim
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.interface import BasePromptOptimizer
from prompt_optimization.optimizers import (
    ContinuousPromptOptimizer,
    ContinuousPromptOptimizerWithProjection,
    DiscretePromptOptimizer
)

class LengthPolicyOptimizer:
    """RL optimizer that learns prompt length policy using REINFORCE."""
    
    def __init__(self, agent: PromptRLAgent):
        self.agent = agent
        self.emb_dim = agent.model.get_input_embeddings().weight.shape[1]
        
        # Simple policy network: state -> action probs
        self.state_dim = 4  # [length, likelihood, step_ratio, improvement]
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Actions: 0=remove, 1=keep, 2=add
        ).to(agent.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
    
    def _prepare_completions(self, target_completions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare completion tokens as batched tensors"""
        completion_tokens_list = [self.agent.tokenizer.encode(t, add_special_tokens=False) for t in target_completions]
        max_comp_len = max(len(ct) for ct in completion_tokens_list) if completion_tokens_list else 0
        pad_id = getattr(self.agent.tokenizer, 'pad_token_id', 0)
        completion_tokens_batch = torch.tensor([
            ct + [pad_id] * (max_comp_len - len(ct)) for ct in completion_tokens_list
        ], dtype=torch.long, device=self.agent.device)
        completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=self.agent.device)
        return completion_tokens_batch, completion_lengths
    
    def optimize_prompts_batch(self, target_completions: List[str], episodes: int = 3,
                               steps_per_episode: int = 50, initial_prompt_length: int = 32,
                               lr_embeddings: float = 0.01, alpha: float = 1.0, beta: float = 0.1,
                               mode: str = "continuous") -> Tuple[List[torch.Tensor], List[float], List[dict]]:
        """Unified batch optimization using pluggable optimizer interface"""
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []
        
        completion_tokens_batch, completion_lengths = self._prepare_completions(target_completions)
        
        # Create optimizer based on mode
        max_prompt_len = initial_prompt_length * 2  # Allow growth
        if mode == "continuous":
            optimizer: BasePromptOptimizer = ContinuousPromptOptimizer(
                self.agent, initial_prompt_length, max_prompt_len, B, lr_embeddings
            )
        elif mode == "continuous_proj":
            # Continuous with projection regularization
            projection_weight = getattr(self, 'projection_weight', 0.1)
            distance_metric = getattr(self, 'distance_metric', 'l2')
            optimizer: BasePromptOptimizer = ContinuousPromptOptimizerWithProjection(
                self.agent, initial_prompt_length, max_prompt_len, B, lr_embeddings,
                projection_weight=projection_weight, distance_metric=distance_metric
            )
        else:  # discrete
            optimizer: BasePromptOptimizer = DiscretePromptOptimizer(
                self.agent, initial_prompt_length, max_prompt_len, B, lr_embeddings
            )
        
        # Initialize prompts
        prompt_data, lengths = optimizer.initialize_prompts()
        
        best_rewards = torch.full((B,), float('-inf'), dtype=torch.float32, device=device)
        best_prompts: List[Optional[torch.Tensor]] = [None] * B
        
        traces = []
        
        for episode in trange(episodes, desc="Episodes"):
            episode_rewards = []
            episode_log_probs = []
            episode_states = []
            
            step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
            for step in step_bar:
                # Inner optimization step (e.g., gradient updates, GCG replacements)
                prompt_data, likelihoods = optimizer.inner_optimization_step(
                    prompt_data, lengths, completion_tokens_batch, completion_lengths, step
                )
                
                # Compute states for policy
                step_ratio = step / steps_per_episode
                states = torch.stack([
                    lengths.float() / initial_prompt_length,  # normalized length
                    likelihoods,  # current likelihood
                    torch.full((B,), step_ratio, device=device),  # step ratio
                    torch.zeros(B, device=device)  # improvement (simplified)
                ], dim=1)  # [B, 4]
                
                # Policy forward pass
                action_logits = self.policy_net(states)  # [B, 3]
                action_probs = F.softmax(action_logits, dim=-1)
                actions = torch.multinomial(action_probs, 1).squeeze(-1)  # [B]
                if B == 1:
                    log_probs = F.log_softmax(action_logits, dim=-1)[0, actions].unsqueeze(0)
                else:
                    log_probs = F.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(-1)
                
                # Apply length actions
                prompt_data, lengths = optimizer.apply_length_action(prompt_data, lengths, actions)
                
                # Compute rewards
                rewards = alpha * likelihoods - beta * lengths.float()
                
                # Update best
                for i in range(B):
                    if rewards[i] > best_rewards[i]:
                        best_rewards[i] = rewards[i]
                        best_prompts[i] = optimizer.clone_prompt(prompt_data, i, lengths[i].item())
                
                episode_rewards.append(rewards)
                episode_log_probs.append(log_probs)
                episode_states.append(states)
            
            # Policy update (REINFORCE)
            rewards_tensor = torch.stack(episode_rewards)  # [T, B]
            log_probs_tensor = torch.stack(episode_log_probs)  # [T, B]
            
            # Compute returns
            returns = torch.zeros_like(rewards_tensor)
            next_return = torch.zeros(B, device=device)
            for t in reversed(range(steps_per_episode)):
                next_return = rewards_tensor[t] + 0.99 * next_return
                returns[t] = next_return
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy loss
            policy_loss = -(log_probs_tensor * returns.detach()).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            traces.append({
                'episode': episode,
                'rewards': [float(r) for r in rewards_tensor[-1]],
                'lengths': [int(l) for l in lengths]
            })
        
        # Convert best prompts to tokens using optimizer's to_tokens method
        # This is mode-agnostic - each optimizer handles its own conversion
        final_prompts = []
        projection_losses = []
        
        # Prepare batch data for to_tokens
        # Check if we have embeddings (2D) or tokens (1D) by inspecting first non-None prompt
        is_embeddings = None
        max_len = 0
        for bp in best_prompts:
            if bp is not None and bp.numel() > 0:
                max_len = max(max_len, bp.shape[0])
                if is_embeddings is None:
                    is_embeddings = len(bp.shape) > 1
        
        if max_len > 0:
            # Create batch tensor - shape depends on whether we have embeddings or tokens
            if is_embeddings:
                prompt_data_batch = torch.zeros(B, max_len, optimizer.emb_dim, device=device)
            else:
                prompt_data_batch = torch.zeros(B, max_len, dtype=torch.long, device=device)
            lengths_batch = torch.zeros(B, dtype=torch.long, device=device)
            
            for i in range(B):
                if best_prompts[i] is not None:
                    length = best_prompts[i].shape[0]
                    if length > 0:
                        prompt_data_batch[i, :length] = best_prompts[i]
                        lengths_batch[i] = length
            
            # Use optimizer's to_tokens method (mode-agnostic)
            tokens_batch = optimizer.to_tokens(prompt_data_batch, lengths_batch)
            
            # Extract individual prompts and compute projection loss if applicable
            for i in range(B):
                length = lengths_batch[i].item()
                if length > 0:
                    final_prompts.append(tokens_batch[i, :length])
                    # Compute projection loss for continuous modes (embeddings -> tokens)
                    if is_embeddings and best_prompts[i] is not None:
                        # Use optimizer's method if available, otherwise compute directly
                        if hasattr(optimizer, '_compute_projection_loss'):
                            proj_loss = optimizer._compute_projection_loss(best_prompts[i])
                        else:
                            # Compute projection loss directly for continuous mode
                            embedding_layer = self.agent.model.get_input_embeddings()
                            vocab_embeds = embedding_layer.weight.detach()
                            distances = torch.cdist(best_prompts[i], vocab_embeds)
                            proj_loss = distances.min(dim=-1)[0].mean()
                        projection_losses.append(proj_loss.item() if isinstance(proj_loss, torch.Tensor) else proj_loss)
                    else:
                        projection_losses.append(0.0)  # No projection for discrete (already tokens)
                else:
                    final_prompts.append(torch.tensor([], dtype=torch.long, device=device))
                    projection_losses.append(float('inf'))
        else:
            for i in range(B):
                final_prompts.append(torch.tensor([], dtype=torch.long, device=device))
                projection_losses.append(float('inf'))
        
        # Store projection losses in traces for analysis
        if projection_losses:
            for trace in traces:
                trace['projection_loss'] = projection_losses[0] if projection_losses else 0.0
        
        return final_prompts, [float(r) for r in best_rewards], traces

