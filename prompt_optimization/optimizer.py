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
    
    def _apply_length_action_with_prefix(self, optimizer: BasePromptOptimizer, prompt_data: torch.Tensor,
                                        lengths: torch.Tensor, actions: torch.Tensor,
                                        prefix_tokens: torch.Tensor, prefix_lengths: torch.Tensor,
                                        attention_mask_offset: torch.Tensor, max_prefix_size: int,
                                        pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply length actions and handle prefix shifting when tokens are deleted.
        When a suffix token is deleted, it moves to the prefix and attention mask shifts.
        """
        device = prompt_data.device
        B = prompt_data.shape[0]
        
        # Handle prefix shifting for delete actions (action=0) BEFORE applying length action
        delete_mask = (actions == 0) & (lengths > 0)
        
        if delete_mask.any():
            # For items that will delete a token, capture the first suffix token and move to prefix
            # Convert suffix to tokens if needed (for continuous modes)
            if len(prompt_data.shape) == 3:  # embeddings [B, L, D]
                # Project to tokens for shifting
                embedding_layer = self.agent.model.get_input_embeddings()
                vocab_embeds = embedding_layer.weight.detach()  # [vocab_size, D]
                
                # Find closest tokens for first suffix position
                first_suffix_embeds = prompt_data[:, 0, :]  # [B, D]
                # Compute distances: [B, vocab_size]
                distances = torch.cdist(first_suffix_embeds, vocab_embeds)  # [B, vocab_size]
                first_suffix_tokens = distances.argmin(dim=-1)  # [B]
            else:  # already tokens [B, L]
                first_suffix_tokens = prompt_data[:, 0]  # [B]
            
            # Shift prefix: move first suffix token to end of prefix
            for i in range(B):
                if delete_mask[i] and prefix_lengths[i] < max_prefix_size:
                    # Add token to prefix
                    prefix_tokens[i, prefix_lengths[i]] = first_suffix_tokens[i]
                    prefix_lengths[i] += 1
                    attention_mask_offset[i] += 1
        
        # Apply standard length action (this will remove the token from suffix)
        prompt_data, lengths = optimizer.apply_length_action(prompt_data, lengths, actions)
        
        return prompt_data, lengths, prefix_tokens, prefix_lengths, attention_mask_offset
    
    def optimize_prompts_batch(self, target_completions: List[str], episodes: int = 3,
                               steps_per_episode: int = 50, initial_prompt_length: int = 32,
                               lr_embeddings: float = 0.01, alpha: float = 1.0, beta: float = 0.1,
                               mode: str = "continuous", batch_size: int = 64) -> Tuple[List[torch.Tensor], List[float], List[dict]]:
        """
        Unified batch optimization using pluggable optimizer interface.
        Processes prompts in batches of batch_size (default 64) for parallelization.
        """
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []
        
        # Process in batches of batch_size
        all_final_prompts = []
        all_rewards = []
        all_traces = []
        
        for batch_start in range(0, B, batch_size):
            batch_end = min(batch_start + batch_size, B)
            batch_completions = target_completions[batch_start:batch_end]
            batch_B = len(batch_completions)
            
            completion_tokens_batch, completion_lengths = self._prepare_completions(batch_completions)
            
            # Create optimizer based on mode
            max_prompt_len = initial_prompt_length * 2  # Allow growth
            if mode == "continuous":
                optimizer: BasePromptOptimizer = ContinuousPromptOptimizer(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings
                )
            elif mode == "continuous_proj":
                # Continuous with projection regularization
                projection_weight = getattr(self, 'projection_weight', 0.1)
                distance_metric = getattr(self, 'distance_metric', 'l2')
                optimizer: BasePromptOptimizer = ContinuousPromptOptimizerWithProjection(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings,
                    projection_weight=projection_weight, distance_metric=distance_metric
                )
            else:  # discrete
                optimizer: BasePromptOptimizer = DiscretePromptOptimizer(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings
                )
            
            # Initialize prompts (suffix)
            prompt_data, lengths = optimizer.initialize_prompts()
            
            # Initialize prefix tokens: empty initially, grows as we delete suffix tokens
            # Max prefix size = initial_prompt_length (can grow up to original suffix size)
            max_prefix_size = initial_prompt_length
            pad_id = getattr(self.agent.tokenizer, 'pad_token_id', 0)
            prefix_tokens = torch.full((batch_B, max_prefix_size), pad_id, dtype=torch.long, device=device)
            prefix_lengths = torch.zeros(batch_B, dtype=torch.long, device=device)
            
            # Track attention mask offsets (how much to mask at start due to deleted tokens)
            attention_mask_offset = torch.zeros(batch_B, dtype=torch.long, device=device)
            
            best_rewards = torch.full((batch_B,), float('-inf'), dtype=torch.float32, device=device)
            best_prompts: List[Optional[torch.Tensor]] = [None] * batch_B
            
            traces = []
        
            for episode in trange(episodes, desc=f"Episodes (batch {batch_start//batch_size + 1})"):
                episode_rewards = []
                episode_log_probs = []
                episode_states = []
                
                step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
                for step in step_bar:
                    # Inner optimization step (e.g., gradient updates, GCG replacements)
                    # Pass prefix info for proper likelihood computation
                    prompt_data, likelihoods = optimizer.inner_optimization_step(
                        prompt_data, lengths, completion_tokens_batch, completion_lengths, step,
                        prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths
                    )
                    
                    # Compute states for policy
                    step_ratio = step / steps_per_episode
                    states = torch.stack([
                        lengths.float() / initial_prompt_length,  # normalized length
                        likelihoods,  # current likelihood
                        torch.full((batch_B,), step_ratio, device=device),  # step ratio
                        torch.zeros(batch_B, device=device)  # improvement (simplified)
                    ], dim=1)  # [batch_B, 4]
                    
                    # Policy forward pass
                    action_logits = self.policy_net(states)  # [batch_B, 3]
                    action_probs = F.softmax(action_logits, dim=-1)
                    actions = torch.multinomial(action_probs, 1).squeeze(-1)  # [batch_B]
                    if batch_B == 1:
                        log_probs = F.log_softmax(action_logits, dim=-1)[0, actions].unsqueeze(0)
                    else:
                        log_probs = F.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(-1)
                    
                    # Apply length actions and handle prefix shifting
                    prompt_data, lengths, prefix_tokens, prefix_lengths, attention_mask_offset = self._apply_length_action_with_prefix(
                        optimizer, prompt_data, lengths, actions, prefix_tokens, prefix_lengths, 
                        attention_mask_offset, max_prefix_size, pad_id
                    )
                
                    # Compute rewards
                    rewards = alpha * likelihoods - beta * lengths.float()
                    
                    # Vectorized best update: only update where reward improved
                    improve_mask = rewards > best_rewards
                    if improve_mask.any():
                        best_rewards = torch.where(improve_mask, rewards, best_rewards)
                        # Update best prompts for improved items
                        for i in torch.nonzero(improve_mask, as_tuple=False).squeeze(-1).tolist():
                            best_prompts[i] = optimizer.clone_prompt(prompt_data, i, lengths[i].item())
                    
                    episode_rewards.append(rewards)
                    episode_log_probs.append(log_probs)
                    episode_states.append(states)
                
                # Policy update (REINFORCE)
                rewards_tensor = torch.stack(episode_rewards)  # [T, batch_B]
                log_probs_tensor = torch.stack(episode_log_probs)  # [T, batch_B]
                
                # Compute returns
                returns = torch.zeros_like(rewards_tensor)
                next_return = torch.zeros(batch_B, device=device)
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
            valid_indices = []
            for i, bp in enumerate(best_prompts):
                if bp is not None and bp.numel() > 0:
                    max_len = max(max_len, bp.shape[0])
                    if is_embeddings is None:
                        is_embeddings = len(bp.shape) > 1
                    valid_indices.append(i)
            
            if max_len > 0 and valid_indices:
                # Create batch tensor - shape depends on whether we have embeddings or tokens
                if is_embeddings:
                    prompt_data_batch = torch.zeros(batch_B, max_len, optimizer.emb_dim, device=device)
                else:
                    prompt_data_batch = torch.zeros(batch_B, max_len, dtype=torch.long, device=device)
                lengths_batch = torch.zeros(batch_B, dtype=torch.long, device=device)
                
                # Vectorized batch creation: copy valid prompts
                for i in valid_indices:
                    bp = best_prompts[i]
                    length = bp.shape[0]
                    if length > 0:
                        prompt_data_batch[i, :length] = bp
                        lengths_batch[i] = length
                
                # Use optimizer's to_tokens method (mode-agnostic)
                tokens_batch = optimizer.to_tokens(prompt_data_batch, lengths_batch)
                
                # Vectorized extraction: extract all prompts at once
                lengths_list = lengths_batch.tolist()
                batch_final_prompts = [tokens_batch[i, :lengths_list[i]] if lengths_list[i] > 0 
                               else torch.tensor([], dtype=torch.long, device=device) 
                               for i in range(batch_B)]
                
                # Compute projection losses (vectorized where possible)
                batch_projection_losses = []
                for i in range(batch_B):
                    if lengths_list[i] > 0:
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
                            batch_projection_losses.append(proj_loss.item() if isinstance(proj_loss, torch.Tensor) else proj_loss)
                        else:
                            batch_projection_losses.append(0.0)  # No projection for discrete (already tokens)
                    else:
                        batch_projection_losses.append(float('inf'))
            else:
                # No valid prompts: return empty tensors
                batch_final_prompts = [torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_B)]
                batch_projection_losses = [float('inf')] * batch_B
            
            # Store projection losses in traces for analysis
            if batch_projection_losses:
                for trace in traces:
                    trace['projection_loss'] = batch_projection_losses[0] if batch_projection_losses else 0.0
            
            # Accumulate results from this batch
            all_final_prompts.extend(batch_final_prompts)
            all_rewards.extend([float(r) for r in best_rewards])
            all_traces.extend(traces)
        
        return all_final_prompts, all_rewards, all_traces

