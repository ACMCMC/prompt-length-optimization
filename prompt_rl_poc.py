"""
Proof of Concept: RL-based Prompt Length Optimization
Given a target completion, find the shortest prompt that maximizes P(completion | prompt)
"""

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import deque
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import csv
import matplotlib.pyplot as plt
from enum import IntEnum

class LengthActions(IntEnum):
    REMOVE_LAST = 0
    KEEP = 1
    RETRACT = 2


class PromptRLAgent:
    def __init__(self, model_name="EleutherAI/pythia-410m"):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name)
        # device handling (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Vocabulary for random token sampling
        self.vocab_size = len(self.tokenizer)
        
    def get_completion_likelihood(self, prompt_tokens: List[int], completion_tokens: List[int]) -> float:
        """Calculate log P(completion | prompt)"""
        if not prompt_tokens:
            # If no prompt, just use BOS token
            input_ids = [self.tokenizer.bos_token_id] + completion_tokens
        else:
            input_ids = prompt_tokens + completion_tokens
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, labels=input_tensor)
            logits = outputs.logits
            
            # Calculate likelihood only for completion tokens
            prompt_len = len(prompt_tokens) if prompt_tokens else 1  # Account for BOS

            completion_logits = logits[0, prompt_len-1:-1]  # Shift for next token prediction
            completion_targets = torch.tensor(completion_tokens, dtype=torch.long, device=self.device)
            # Get log probabilities
            log_probs = F.log_softmax(completion_logits, dim=-1)
            token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze()
            
            return float(token_log_probs.sum())
    
    def calculate_reward(self, prompt_tokens: List[int], completion_tokens: List[int], 
                        alpha=1.0, beta=0.1, length_ratio_penalty: float = 0.0,
                        reference_length: Optional[int] = None) -> float:
        """Calculate reward: α * log P(completion | prompt) minus length penalties."""
        likelihood = self.get_completion_likelihood(prompt_tokens, completion_tokens)
        prompt_length = len(prompt_tokens)
        total_penalty = beta * prompt_length
        if reference_length is None or reference_length <= 0:
            ref_len = len(completion_tokens) if completion_tokens else prompt_length or 1
        else:
            ref_len = reference_length
        if length_ratio_penalty > 0.0:
            denom = ref_len if ref_len is not None else max(1, prompt_length)
            total_penalty += length_ratio_penalty * (prompt_length / denom)
        return alpha * likelihood - total_penalty
    
    def get_random_token(self) -> int:
        """Sample a random token from vocabulary (excluding special tokens)"""
        # Avoid special tokens like PAD, EOS, BOS
        special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 
                         self.tokenizer.bos_token_id}
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in special_tokens:
                return token

    def get_token_embedding(self, token_id: int) -> torch.Tensor:
        """Return embedding vector for a token id (detached)."""
        embed_layer = self.model.get_input_embeddings()
        return embed_layer(torch.tensor([token_id], device=self.device)).squeeze(0).detach()

    def get_top_token_candidates(self, prompt_tokens: List[int], top_k: int = 5) -> List[int]:
        """
        Get top-k next-token candidates given the current prompt tokens.
        Returns token ids sorted by probability.
        """
        if not prompt_tokens:
            # Start from BOS to keep logits well-defined
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long, device=self.device)
        else:
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[0, -1]
            probs = F.softmax(next_token_logits, dim=-1)
            topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))

        return topk.indices.tolist()

class LengthPolicyOptimizer:
    def __init__(self, agent: PromptRLAgent, state_embed_dim: int = 16):
        self.agent = agent
        self.emb_dim = self.agent.model.get_input_embeddings().weight.shape[1]
        print("Embedding dimension:", self.emb_dim)
        self.state_embed_dim = state_embed_dim
        self.scalar_feature_size = 11  # number of scalar features in state tensor
        self.actions = list(LengthActions)
        self.action_dim = len(self.actions)
        
        # Encoders to keep context compact
        self.prompt_summary_proj = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_embed_dim)
        ).to(self.agent.device)
        self.token_summary_proj = nn.Sequential(
            nn.Linear(self.emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.state_embed_dim)
        ).to(self.agent.device)
        
        policy_input_dim = self.scalar_feature_size + self.action_dim + 2 * self.state_embed_dim
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        ).to(self.agent.device)
        
        # Optimizers
        self.prompt_embeddings: Optional[nn.Parameter] = None
        self.prompt_optimizer: Optional[optim.Optimizer] = None
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.reward_baseline: Optional[float] = None
        self.baseline_momentum: float = 0.9
        self.default_entropy_coef: float = 0.01
        
        # Hyperparameters
        self.embedding_opt_steps = 5
        self.max_prompt_length_multiplier = 2
        self.max_prompt_cap = 128
        self.last_action: LengthActions = LengthActions.KEEP
        
        # RL tracking
        self.episode_rewards: List[float] = []
        self.episode_actions: List[int] = []
        self.episode_log_probs: List[torch.Tensor] = []
        self.episode_states: List[torch.Tensor] = []
        
        # History tracking
        self.loss_history: List[float] = []
        self.length_history: List[int] = []
        self.likelihood_history: List[float] = []
        self.action_history: List[int] = []
        self.trace: List[Dict] = []
    
    def optimize_prompt(self, target_completion: str, episodes=100, steps_per_episode=50,
                       initial_prompt_length=32, lr_embeddings=0.01, lr_policy=3e-4,
                       alpha=1.0, beta=0.1, length_ratio_penalty: float = 0.0,
                       entropy_coef: Optional[float] = None, log_every=10,
                       max_prompt_length: Optional[int] = None,
                       embedding_opt_steps: int = 5, top_k_tokens: int = 5,
                       baseline_momentum: Optional[float] = None) -> Tuple[List[int], float, List[float]]:
        """
        Train policy to learn optimal length adjustments while optimizing embeddings
        Policy learns when to remove/keep/add tokens based on current state
        """
        completion_tokens = self.agent.tokenizer.encode(target_completion, add_special_tokens=False)
        
        print(f"Target completion: '{target_completion}'")
        print(f"Training policy for {episodes} episodes, {steps_per_episode} steps each")
        print(f"Starting length: {initial_prompt_length}, α={alpha}, β={beta}")
        print(f"Length ratio penalty: {length_ratio_penalty}, entropy coef: {entropy_coef}")
        print("-" * 60)

        best_overall_reward = float('-inf')
        best_prompt = None
        best_overall_likelihood = float('-inf')
        self.reward_baseline = None
        
        # Reset trace for this optimization
        self.trace = []
        global_step = 0
        self.embedding_opt_steps = max(1, embedding_opt_steps)
        max_length_limit = max_prompt_length or min(
            max(initial_prompt_length * self.max_prompt_length_multiplier, initial_prompt_length + 4),
            self.max_prompt_cap
        )
        # Update policy learning rate if provided
        for g in self.policy_optimizer.param_groups:
            g['lr'] = lr_policy
        if baseline_momentum is not None:
            self.baseline_momentum = baseline_momentum
        entropy_coef = self.default_entropy_coef if entropy_coef is None else entropy_coef
        
        reference_length = max(1, len(completion_tokens)) if completion_tokens else max(1, initial_prompt_length)

        for episode in range(episodes):
            # Reset for new episode
            current_length = initial_prompt_length
            self.prompt_embeddings = self._initialize_prompt_embeddings(
                completion_tokens,
                initial_prompt_length
            )
            self._refresh_prompt_optimizer(lr_embeddings)
            # stack for retract functionality
            retract_stack = []  # store removed embeddings (in order of removal)
            self.last_action = LengthActions.KEEP
            
            episode_rewards = []
            episode_log_probs = []
            episode_actions = []
            episode_entropies = []
            
            # Enhanced tracking for optimization momentum
            likelihood_history = deque(maxlen=10)  # Track recent likelihoods for momentum calculation
            
            for step in range(steps_per_episode):
                # Optimize embeddings for a few steps to get optimization momentum
                print("episode :", episode, " step :", step)
                recent_likelihoods = []
                grad_norm = 0.0
                for emb_step in range(self.embedding_opt_steps):
                    self.prompt_optimizer.zero_grad()
                    likelihood = self._get_likelihood_from_embeddings(self.prompt_embeddings, completion_tokens)
                    recent_likelihoods.append(float(likelihood))
                    emb_loss = -likelihood
                    emb_loss.backward()
                    if self.prompt_embeddings.grad is not None:
                        grad_norm = float(self.prompt_embeddings.grad.norm().item())
                    self.prompt_optimizer.step()
                
                # Calculate optimization momentum features
                current_likelihood = recent_likelihoods[-1]
                likelihood_history.append(current_likelihood)
                
                # Calculate improvement rate (recent trend)
                if len(recent_likelihoods) >= 3:
                    improvement_rate = recent_likelihoods[-1] - recent_likelihoods[0]  # Change over 5 steps
                    # Detect diminishing returns
                    recent_improvements = [recent_likelihoods[i] - recent_likelihoods[i-1] 
                                         for i in range(1, len(recent_likelihoods))]
                    improvement_trend = np.mean(recent_improvements[-3:]) if len(recent_improvements) >= 3 else 0
                else:
                    improvement_rate = 0
                    improvement_trend = 0
                
                # Steps since significant improvement (threshold-based)
                steps_since_improvement = 0
                improvement_threshold = 0.1
                for i, past_likelihood in enumerate(reversed(list(likelihood_history))):
                    if current_likelihood - past_likelihood > improvement_threshold:
                        break
                    steps_since_improvement = i + 1
                
                # Get current state for policy with enhanced features (NO direct embedding access)
                discrete_prompt = self._embeddings_to_tokens(self.prompt_embeddings.detach())
                print("discrete_prompt : ", discrete_prompt)
                context_stats = self._get_prompt_context_stats(
                    discrete_prompt,
                    completion_tokens,
                    top_k=top_k_tokens
                )
                # print("context_stats : ", context_stats)
                state_features = self._build_state_tensor(
                    current_length=current_length,
                    current_likelihood=current_likelihood,
                    improvement_rate=improvement_rate,
                    improvement_trend=improvement_trend,
                    steps_since_improvement=steps_since_improvement,
                    grad_norm=grad_norm,
                    step=step,
                    steps_per_episode=steps_per_episode,
                    context_stats=context_stats,
                    max_length_limit=max_length_limit
                )
                
                print("state_features : ", state_features)
                # Policy decision
                policy_logits = self.policy_net(state_features)
                policy_probs = F.softmax(policy_logits, dim=-1)
                print("policy_probs : ", policy_probs)
                policy_dist = torch.distributions.Categorical(policy_probs)
                action = policy_dist.sample()
                print("action : ", action)
                log_prob = policy_dist.log_prob(action)
                entropy = policy_dist.entropy()
                
                # Execute action: 0=REMOVE, 1=KEEP, 2=RETRACT
                action_val = int(action.item())
                action_enum = LengthActions(action_val)
                print("action_enum : ", action_enum)
                new_length = current_length

                if action_enum == LengthActions.REMOVE_LAST and current_length > 1:
                    # Save last embedding to stack then remove
                    last_emb = self.prompt_embeddings[-1].clone().detach()
                    retract_stack.append(last_emb)
                    new_length = current_length - 1
                    self.prompt_embeddings = nn.Parameter(
                        self.prompt_embeddings[:-1].clone().detach().requires_grad_(True)
                    )
                    self._refresh_prompt_optimizer(lr_embeddings)
                elif action_enum == LengthActions.RETRACT and retract_stack:
                    # Restore last removed embedding
                    restored = retract_stack.pop()
                    restored = restored.unsqueeze(0)
                    self.prompt_embeddings = nn.Parameter(
                        torch.cat([self.prompt_embeddings.detach(), restored], dim=0).requires_grad_(True)
                    )
                    new_length = current_length + 1
                    self._refresh_prompt_optimizer(lr_embeddings)
                # action_val == 1 KEEP or invalid RETRACT (no stack) -> no change

                current_length = new_length
                self.last_action = action_enum
                
                # Calculate base reward
                with torch.no_grad():
                    final_likelihood = self._get_likelihood_from_embeddings(self.prompt_embeddings, completion_tokens)
                    length_ratio = current_length / reference_length
                    base_reward = (alpha * final_likelihood
                                   - beta * current_length
                                   - length_ratio_penalty * length_ratio)
                
                # Natural learning rewards (no artificial exploration penalties)
                discovery_bonus = 0
                if action_enum == LengthActions.REMOVE_LAST:
                    # Small bonus for compression attempts (encourages exploration)
                    discovery_bonus += 0.1
                
                # Efficiency bonus: reward good likelihood with fewer tokens
                if current_length < reference_length and final_likelihood > -1.0:
                    efficiency_bonus = (reference_length - current_length) * 0.05
                    discovery_bonus += efficiency_bonus
                
                reward = base_reward + discovery_bonus
                
                # Store for policy update (convert to float to avoid tensor issues)
                episode_rewards.append(float(reward))
                episode_log_probs.append(log_prob)
                episode_entropies.append(entropy)
                episode_actions.append(action_val)
                
                # Track best
                if reward > best_overall_reward:
                    best_overall_reward = float(reward)  # Convert to Python float
                    best_prompt = self._embeddings_to_tokens(self.prompt_embeddings.detach())
                
                # Logging with momentum info
                self.likelihood_history.append(float(current_likelihood))
                self.length_history.append(current_length)
                self.action_history.append(action_val)
                
                # Track best likelihood
                if current_likelihood > best_overall_likelihood:
                    best_overall_likelihood = float(current_likelihood)
                
                # Add to structured trace
                self.trace.append({
                    'step': global_step,
                    'episode': episode,
                    'likelihood': float(current_likelihood),
                    'best_likelihood': float(best_overall_likelihood),
                    'length': current_length,
                    'action': action_val,
                    'action_name': action_enum.name,
                    'reward': float(reward),
                    'grad_norm': grad_norm,
                    'context_entropy': context_stats.get('entropy', 0.0),
                    'target_prob': context_stats.get('target_prob', 0.0),
                    'improved': reward >= best_overall_reward
                })
                global_step += 1
                
                # Logging
                if log_every > 0 and episode % log_every == 0 and step % 10 == 0:
                    print(f"Ep {episode:3d} Step {step:2d}: action={action_enum.name} length={current_length} "
                          f"likelihood={current_likelihood:.3f} improv_rate={improvement_rate:.4f} "
                          f"since_improv={steps_since_improvement} reward={reward:.3f} "
                          f"entropy={context_stats.get('entropy', 0.0):.3f}")
            
            # Policy update at end of episode (REINFORCE)
            episode_return = sum(episode_rewards)
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + G  
                returns.insert(0, G)
            
            returns = torch.tensor(returns, device=self.agent.device, dtype=torch.float32)
            mean_return = returns.mean().item() if returns.numel() > 0 else 0.0
            baseline_val = self.reward_baseline if self.reward_baseline is not None else mean_return
            advantages = returns - baseline_val
            if returns.numel() > 0:
                updated_baseline = self.baseline_momentum * baseline_val + (1 - self.baseline_momentum) * mean_return
                self.reward_baseline = updated_baseline
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            policy_loss = 0.0
            for log_prob, adv, ent in zip(episode_log_probs, advantages, episode_entropies):
                policy_loss = policy_loss - log_prob * adv - entropy_coef * ent
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            if log_every > 0 and episode % log_every == 0:
                avg_length = sum(self.length_history[-steps_per_episode:]) / steps_per_episode
                print(f"Episode {episode}: return={episode_return:.2f} avg_length={avg_length:.1f} "
                      f"best_reward={best_overall_reward:.3f}")
        
        # Final discrete refinement to squeeze length without policy involvement
        if best_prompt:
            refined_prompt, refined_reward = self._refine_prompt(
                best_prompt,
                completion_tokens,
                alpha,
                beta,
                length_ratio_penalty,
                reference_length
            )
            if refined_reward > best_overall_reward:
                best_overall_reward = refined_reward
                best_prompt = refined_prompt
                best_overall_likelihood = self.agent.get_completion_likelihood(best_prompt, completion_tokens)
        
        return best_prompt, float(best_overall_reward), self.trace
    
    def _initialize_prompt_embeddings(self, completion_tokens: List[int], initial_length: int) -> nn.Parameter:
        """Heuristic initialization that seeds embeddings from completion tokens with noise."""
        device = self.agent.device
        seed_tokens = completion_tokens[:initial_length] if completion_tokens else []
        if len(seed_tokens) < initial_length:
            deficit = initial_length - len(seed_tokens)
            seed_tokens.extend(self.agent.get_random_token() for _ in range(deficit))
        elif len(seed_tokens) > initial_length:
            seed_tokens = seed_tokens[:initial_length]
        
        if not seed_tokens:
            seed_tokens = [self.agent.get_random_token() for _ in range(max(1, initial_length))]
        
        embed_layer = self.agent.model.get_input_embeddings()
        token_tensor = torch.tensor(seed_tokens, dtype=torch.long, device=device)
        embeddings = embed_layer(token_tensor).detach()
        
        # Add small noise to encourage exploration
        noise_scale = 0.01
        embeddings = embeddings + noise_scale * torch.randn_like(embeddings)
        embeddings = embeddings[:initial_length]
        
        self.initial_seed_tokens = seed_tokens  # useful for debugging / analysis
        return nn.Parameter(embeddings.clone().detach().requires_grad_(True))
    
    def _refresh_prompt_optimizer(self, lr_embeddings: float) -> None:
        self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
    
    def _get_prompt_context_stats(self, prompt_tokens: List[int], completion_tokens: List[int],
                                  top_k: int = 5) -> Dict[str, Optional[float]]:
        """Compute contextual statistics for the current discrete prompt."""
        device = self.agent.device
        tokenizer = self.agent.tokenizer
        if tokenizer.bos_token_id is not None:
            bos_id = tokenizer.bos_token_id
        elif tokenizer.eos_token_id is not None:
            bos_id = tokenizer.eos_token_id
        else:
            bos_id = 0
        
        if prompt_tokens:
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        else:
            input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = self.agent.model(input_ids)
            next_logits = outputs.logits[0, -1]
            probs = F.softmax(next_logits, dim=-1)
            log_probs = torch.log(probs + 1e-12)
            entropy = float(-(probs * log_probs).sum().item())
            topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        
        completion_first = completion_tokens[0] if completion_tokens else None
        target_prob = float(probs[completion_first].item()) if completion_first is not None else 0.0
        top_tokens = topk.indices.tolist()
        top_token_embedding = None
        if top_tokens:
            top_token_embedding = self.agent.get_token_embedding(top_tokens[0])
        
        overlap_ratio = 0.0
        if prompt_tokens and completion_tokens:
            overlap_ratio = len(set(prompt_tokens) & set(completion_tokens)) / max(1, len(prompt_tokens))
        
        return {
            'entropy': entropy,
            'target_prob': target_prob,
            'top_tokens': top_tokens,
            'top_token_embedding': top_token_embedding,
            'overlap_ratio': overlap_ratio,
            'target_token_id': completion_first
        }
    
    def _build_state_tensor(self, *, current_length: int, current_likelihood: float,
                            improvement_rate: float, improvement_trend: float,
                            steps_since_improvement: int, grad_norm: float,
                            step: int, steps_per_episode: int,
                            context_stats: Dict[str, Optional[float]],
                            max_length_limit: int) -> torch.Tensor:
        """Assemble state tensor combining scalars, last action, and context projections."""
        device = self.agent.device
        normalized_length = current_length / max(1, max_length_limit)
        scalar_values = [
            float(current_length),
            float(normalized_length),
            float(current_likelihood),
            float(improvement_rate),
            float(improvement_trend),
            float(steps_since_improvement),
            float(grad_norm),
            float(context_stats.get('entropy', 0.0) or 0.0),
            float(context_stats.get('target_prob', 0.0) or 0.0),
            float(context_stats.get('overlap_ratio', 0.0) or 0.0),
            float(step / max(1, steps_per_episode))
        ]
        scalar_tensor = torch.tensor(scalar_values, device=device, dtype=torch.float32)
        
        action_one_hot = torch.zeros(self.action_dim, device=device, dtype=torch.float32)
        action_index = int(self.last_action)
        action_one_hot[action_index] = 1.0
        
        if current_length > 0:
            prompt_mean = self.prompt_embeddings.detach().mean(dim=0)
        else:
            prompt_mean = torch.zeros(self.emb_dim, device=device)
        prompt_features = self.prompt_summary_proj(prompt_mean)
        
        top_token_embedding = context_stats.get('top_token_embedding')
        if top_token_embedding is not None:
            token_features = self.token_summary_proj(top_token_embedding.to(device))
        else:
            token_features = torch.zeros(self.state_embed_dim, device=device)
        
        return torch.cat([scalar_tensor, action_one_hot, prompt_features, token_features], dim=0)
    
    def _refine_prompt(self, prompt_tokens: Optional[List[int]], completion_tokens: List[int],
                       alpha: float, beta: float, length_ratio_penalty: float,
                       reference_length: int) -> Tuple[Optional[List[int]], float]:
        """Greedy discrete refinement to remove redundant tokens."""
        if not prompt_tokens:
            return prompt_tokens, float('-inf')
        
        best_prompt = prompt_tokens[:]
        best_reward = self.agent.calculate_reward(
            best_prompt,
            completion_tokens,
            alpha,
            beta,
            length_ratio_penalty,
            reference_length
        )
        improved = True
        while improved and len(best_prompt) > 1:
            improved = False
            for idx in range(len(best_prompt)):
                candidate = best_prompt[:idx] + best_prompt[idx+1:]
                candidate_reward = self.agent.calculate_reward(
                    candidate,
                    completion_tokens,
                    alpha,
                    beta,
                    length_ratio_penalty,
                    reference_length
                )
                if candidate_reward >= best_reward:
                    best_reward = candidate_reward
                    best_prompt = candidate
                    improved = True
                    break
        return best_prompt, best_reward
    
    def _get_likelihood_from_embeddings(self, prompt_embeds: torch.Tensor, completion_tokens: List[int]) -> torch.Tensor:
        """Calculate log P(completion | continuous prompt embeddings)"""
        # Convert completion tokens to embeddings
        completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=self.agent.device)
        completion_embeds = self.agent.model.get_input_embeddings()(completion_tensor)
        
        # Concatenate prompt embeddings + completion embeddings
        full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=0).unsqueeze(0)  # [1, seq_len, emb_dim]
        
        # Forward pass through transformer
        # We need to bypass the embedding layer and feed embeddings directly
        outputs = self.agent.model.gpt_neox(inputs_embeds=full_embeds)
        hidden_states = outputs.last_hidden_state
        
        # Get logits for completion positions only
        logits = self.agent.model.embed_out(hidden_states)  # [1, seq_len, vocab_size]
        prompt_len = prompt_embeds.shape[0]
        completion_logits = logits[0, prompt_len-1:-1]  # Shift for next token prediction
        
        # Calculate log probabilities for target completion tokens
        log_probs = F.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
        
        return target_log_probs.sum()
    
    def _embeddings_to_tokens(self, embeddings: torch.Tensor) -> List[int]:
        """Convert embeddings back to discrete tokens by finding nearest neighbors"""
        # Get all vocabulary embeddings
        vocab_embeds = self.agent.model.get_input_embeddings().weight  # [vocab_size, emb_dim]
        
        tokens = []
        for emb in embeddings:
            # Find closest token embedding
            distances = torch.norm(vocab_embeds - emb.unsqueeze(0), dim=1)
            closest_token = int(torch.argmin(distances).item())
            tokens.append(closest_token)
        
        return tokens

# Main function removed - use train.py and eval.py instead

if __name__ == "__main__":
    print("This module contains the core classes PromptRLAgent and LengthPolicyOptimizer.")
    print("Use train.py to train a model and eval.py to evaluate.")
    print("Example:")
    print("  python train.py --episodes 100")
    print("  python eval.py --test_prompt 'Your test text here'")
