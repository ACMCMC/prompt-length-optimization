"""
RL Policy Optimizer: manages the policy network and coordinates optimization
"""

import torch
import torch.nn.functional as F
from tqdm import trange
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.optim as optim
import logging
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.interface import BasePromptOptimizer
from prompt_optimization.model_inputs import ModelBatchedInput
from prompt_optimization.optimizers import (
    ContinuousPromptOptimizer,
    ContinuousPromptOptimizerWithProjection,
    DiscretePromptOptimizer
)

logger = logging.getLogger(__name__)

class LengthPolicyOptimizer:
    """RL optimizer that learns prompt length policy using REINFORCE or PPO."""
    
    def __init__(self, agent: PromptRLAgent, epsilon: float, epsilon_decay: float, epsilon_min: float,
                 entropy_coef: float, temperature: float,
                 use_ppo: bool, ppo_clip: float, ppo_epochs: int,
                 ppo_gamma: float, ppo_gae_lambda: float, ppo_value_coef: float,
                 policy_hidden_size: int, value_init_bias: float, value_init_gain: float,
                 max_grad_norm: float):
        self.agent = agent
        self.emb_dim = agent.model.get_input_embeddings().weight.shape[1]
        
        # Simple policy network: state -> action probs
        self.state_dim = 2  # [length, likelihood]
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, 3)  # Actions: 0=remove, 1=keep, 2=add
        ).to(agent.device)
        
        # Value network for PPO (estimates state values)
        # Initialize output layer to predict values around typical return scale
        # This helps the network start in the right range
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, 1)  # Single value output
        ).to(agent.device)
        
        # Initialize value network output layer to predict values around typical return scale
        # This prevents initial value loss from being extremely large
        with torch.no_grad():
            # Initialize last layer bias and weights from config
            if len(self.value_net) > 0:
                last_layer = self.value_net[-1]
                if isinstance(last_layer, nn.Linear):
                    last_layer.bias.fill_(value_init_bias)
                    nn.init.xavier_uniform_(last_layer.weight, gain=value_init_gain)
        
        self.max_grad_norm = max_grad_norm
        
        # Use shared optimizer for both networks (PPO) or separate (REINFORCE)
        self.use_ppo = use_ppo
        if use_ppo:
            # Shared optimizer for policy and value networks
            self.optimizer = optim.Adam(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                lr=3e-4
            )
        else:
            # Separate optimizer for policy only (REINFORCE)
            self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        
        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_epsilon = epsilon
        
        # Entropy bonus for exploration (encourages diverse action distributions)
        self.entropy_coef = entropy_coef
        
        # Temperature for softmax (higher = more exploration)
        self.temperature = temperature
        
        # PPO-specific parameters
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_gamma = ppo_gamma
        self.ppo_gae_lambda = ppo_gae_lambda
        self.ppo_value_coef = ppo_value_coef
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: [T, batch_B] tensor of rewards
            values: [T, batch_B] tensor of value estimates
            device: torch device
            
        Returns:
            returns: [T, batch_B] tensor of returns (value targets)
            advantages: [T, batch_B] tensor of advantages
        """
        T, batch_B = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Compute next values (for terminal state, next_value = 0)
        next_values = torch.cat([values[1:], torch.zeros(1, batch_B, device=device)], dim=0)
        
        # Compute TD errors: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        deltas = rewards + self.ppo_gamma * next_values - values
        
        # Compute GAE advantages: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.ppo_gamma * self.ppo_gae_lambda * gae
            advantages[t] = gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        return returns, advantages
    
    def _apply_length_action_with_prefix(self, optimizer: BasePromptOptimizer, prompt_data: torch.Tensor,
                                        lengths: torch.Tensor, actions: torch.Tensor,
                                        model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply length actions to suffix only.
        Prefix and completion are immutable - only suffix tokens/embeddings and attention mask can change.
        """
        # Handle remove actions (action=0): remove token from suffix by setting attention mask to 0
        remove_mask = (actions == 0) & (lengths > 0)
        if remove_mask.any():
            remove_indices = torch.nonzero(remove_mask, as_tuple=False).squeeze(-1)
            model_input.remove_suffix_token(remove_indices)
        
        # Handle add actions (action=2): add new token to suffix by setting attention mask to 1
        add_mask = (actions == 2) & (lengths < model_input.max_suffix_len)
        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            model_input.add_suffix_token(add_indices)
        
        # Apply standard length action (initializes new positions if needed)
        prompt_data, lengths = optimizer.apply_length_action(prompt_data, lengths, actions)
        
        # Update suffix in ModelBatchedInput (tokens/embeddings)
        # Note: model_input.mode is normalized to 'continuous' for continuous_proj
        if model_input.mode == 'continuous':
            model_input.update_suffix_embeddings(prompt_data)
        else:
            model_input.update_suffix_tokens(prompt_data)
        
        # Update lengths from attention mask
        lengths = model_input.suffix_attention_mask.sum(dim=1)
        
        return prompt_data, lengths
    
    def optimize_prompts_batch(self, target_completions: List[str], episodes: int,
                               steps_per_episode: int, initial_prompt_length: int,
                               lr_embeddings: float, alpha: float, beta: float,
                               mode: str, batch_size: int,
                               max_suffix_len: int, init_len: int,
                               wandb_log_fn=None, global_step_offset: int = 0) -> Tuple[List[torch.Tensor], List[float], List[dict], List[dict]]:
        """
        Unified batch optimization using pluggable optimizer interface.
        Processes prompts in batches of batch_size (default 64) for parallelization.
        """
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], [], []
        
        # Process in batches of batch_size
        all_final_prompts = []
        all_rewards = []
        all_traces = []
        all_policy_metrics = []  # Track policy training metrics
        
        for batch_start in range(0, B, batch_size):
            batch_end = min(batch_start + batch_size, B)
            batch_completions = target_completions[batch_start:batch_end]
            batch_B = len(batch_completions)
            
            # Create optimizer based on mode
            max_prompt_len = max_suffix_len  # Suffix size is fixed to max_suffix_len from config
            if mode == "continuous":
                optimizer: BasePromptOptimizer = ContinuousPromptOptimizer(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings,
                    max_suffix_len=max_suffix_len, init_len=init_len
                )
            elif mode == "continuous_proj":
                # Continuous with projection regularization
                projection_weight = getattr(self, 'projection_weight', None)
                if projection_weight is None:
                    raise ValueError("projection_weight must be set via set_optimization_params before using continuous_proj mode")
                distance_metric = getattr(self, 'distance_metric', None)
                if distance_metric is None:
                    raise ValueError("distance_metric must be set via set_optimization_params before using continuous_proj mode")
                optimizer: BasePromptOptimizer = ContinuousPromptOptimizerWithProjection(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings,
                    projection_weight=projection_weight, distance_metric=distance_metric,
                    max_suffix_len=max_suffix_len, init_len=init_len
                )
            else:  # discrete
                optimizer: BasePromptOptimizer = DiscretePromptOptimizer(
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings,
                    max_suffix_len=max_suffix_len, init_len=init_len
                )
            
            best_rewards = torch.full((batch_B,), float('-inf'), dtype=torch.float32, device=device)
            best_likelihoods = torch.full((batch_B,), float('-inf'), dtype=torch.float32, device=device)
            best_prompts: List[Optional[torch.Tensor]] = [None] * batch_B
            
            traces = []
            batch_policy_metrics = []  # Track policy metrics for this batch
        
            for episode in trange(episodes, desc=f"Episodes (batch {batch_start//batch_size + 1})"):
                # Create fresh ModelBatchedInput for this episode
                # Prefix texts are empty initially (will grow as tokens are deleted)
                prefix_texts = [''] * batch_B
                model_input = ModelBatchedInput(
                    prefix_texts=prefix_texts,
                    completion_texts=batch_completions,
                    tokenizer=self.agent.tokenizer,
                    device=self.agent.device,
                    embedding_layer=self.agent.model.get_input_embeddings(),
                    max_suffix_len=max_suffix_len,
                    init_len=init_len,
                    mode=mode
                )
                
                # Initialize prompts (suffix) using ModelBatchedInput
                prompt_data, lengths = optimizer.initialize_prompts(model_input)
                
                # Update model_input with initial suffix
                if mode in ['continuous', 'continuous_proj']:
                    model_input.update_suffix_embeddings(prompt_data)
                else:
                    model_input.update_suffix_tokens(prompt_data)
                
                episode_rewards = []
                episode_likelihoods = []
                episode_log_probs = []
                episode_states = []
                episode_action_probs = []  # Store for entropy
                episode_actions = []  # Store actions for PPO importance sampling computation
                
                logger.info(f"Starting optimization: Episode {episode+1}/{episodes}, Batch {batch_start//batch_size + 1}, {steps_per_episode} steps")
                
                # Set policy to eval mode during episode (no gradients, only inference)
                self.policy_net.eval()
                if self.use_ppo:
                    self.value_net.eval()
                
                step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
                for step in step_bar:
                    # ===== PROMPT OPTIMIZATION ONLY (no policy updates) =====
                    # Inner optimization step: optimize prompt embeddings/tokens only
                    # This does NOT update the policy network
                    if step == 0 or (step + 1) % 10 == 0 or step == steps_per_episode - 1:
                        logger.info(f"  Optimization step {step+1}/{steps_per_episode} (Episode {episode+1}, Batch {batch_start//batch_size + 1})")
                    
                    prompt_data, likelihoods = optimizer.inner_optimization_step(
                        prompt_data, lengths, step, model_input
                    )
                    
                    # Compute states for policy (inference only, no gradients)
                    states = torch.stack([
                        lengths.float() / initial_prompt_length,  # normalized length
                        likelihoods,  # current likelihood
                    ], dim=1)  # [batch_B, 2]
                    
                    # Policy forward pass (INFERENCE ONLY - no gradients, no updates)
                    # Policy network is in eval mode and we're only collecting data
                    with torch.no_grad():
                        action_logits = self.policy_net(states)  # [batch_B, 3]
                    # Compute action probabilities (detached, no gradients)
                    action_probs = F.softmax(action_logits / self.temperature, dim=-1).detach()
                    
                    # Epsilon-greedy action selection (batched)
                    explore_mask = torch.rand(batch_B, device=device) < self.current_epsilon
                    # Random exploration: uniform over 3 actions
                    random_actions = torch.randint(0, 3, (batch_B,), device=device)
                    # Exploitation: sample from policy
                    policy_actions = torch.multinomial(action_probs, 1).squeeze(-1)  # [batch_B]
                    # Combine: use random actions where explore_mask is True, policy actions otherwise
                    actions = torch.where(explore_mask, random_actions, policy_actions)
                    
                    # Compute log_probs: uniform for random actions, policy log_probs for exploitation
                    uniform_log_prob = torch.log(torch.tensor(1.0 / 3.0, device=device))
                    policy_log_probs = F.log_softmax(action_logits / self.temperature, dim=-1)
                    if batch_B == 1:
                        policy_log_probs_selected = policy_log_probs[0, actions].unsqueeze(0)
                    else:
                        policy_log_probs_selected = policy_log_probs.gather(1, actions.unsqueeze(1)).squeeze(-1)
                    # Use uniform log_prob for exploration, policy log_prob for exploitation
                    log_probs = torch.where(explore_mask, 
                                           torch.full((batch_B,), uniform_log_prob, device=device),
                                           policy_log_probs_selected)
                    
                    # Apply length actions and handle prefix shifting
                    prompt_data, lengths = self._apply_length_action_with_prefix(
                        optimizer, prompt_data, lengths, actions, model_input
                    )
                    
                    # Update lengths based on suffix attention mask
                    # Count active suffix positions
                    active_suffix_counts = model_input.suffix_attention_mask.sum(dim=1)  # [B]
                    lengths = active_suffix_counts
                
                    # Compute rewards: both likelihood and length are negative, less negative = better
                    # Likelihoods are negative (log probabilities): less negative = better (e.g., -50 > -80)
                    # Lengths are converted to negative: shorter = less negative = better (e.g., -20 > -60)
                    # Reward = alpha * likelihood - beta * length
                    # Both terms are negative, so higher (less negative) reward = better performance
                    rewards = alpha * likelihoods - beta * lengths.float()
                    
                    # Vectorized best update: only update where reward improved
                    improve_mask = rewards > best_rewards
                    if improve_mask.any():
                        best_rewards = torch.where(improve_mask, rewards, best_rewards)
                        best_likelihoods = torch.where(improve_mask, likelihoods, best_likelihoods)
                        # Update best prompts for improved items
                        for i in torch.nonzero(improve_mask, as_tuple=False).squeeze(-1).tolist():
                            best_prompts[i] = optimizer.clone_prompt(prompt_data, i, lengths[i].item())
                    
                    episode_rewards.append(rewards)
                    episode_likelihoods.append(likelihoods)
                    episode_log_probs.append(log_probs)
                    episode_states.append(states)
                    episode_action_probs.append(action_probs)  # Store for entropy
                    episode_actions.append(actions)  # Store actions for PPO
                    
                    # Store step-level trace for plotting
                    local_step = episode * steps_per_episode + step
                    global_step = global_step_offset + local_step  # Truly global step across all batches
                    # Convert to Python floats, handling NaN/Inf
                    likelihoods_list = []
                    for l in likelihoods:
                        l_val = float(l.item()) if torch.is_tensor(l) else float(l)
                        # Check if value is NaN or Inf, if so set to 0.0
                        if not (isinstance(l_val, (int, float)) and l_val == l_val and l_val != float('inf') and l_val != float('-inf')):
                            l_val = 0.0
                        likelihoods_list.append(l_val)
                    
                    best_likelihoods_list = []
                    for l in best_likelihoods:
                        l_val = float(l.item()) if torch.is_tensor(l) else float(l)
                        # Check if value is NaN or Inf, if so set to 0.0
                        if not (isinstance(l_val, (int, float)) and l_val == l_val and l_val != float('inf') and l_val != float('-inf')):
                            l_val = 0.0
                        best_likelihoods_list.append(l_val)
                    
                    traces.append({
                        'episode': episode,
                        'step': global_step,  # Global step across all episodes
                        'rewards': [float(r) for r in rewards],
                        'likelihoods': likelihoods_list,
                        'best_likelihoods': best_likelihoods_list,
                        'lengths': [int(l) for l in lengths]
                    })
                    
                    # Log step-level metrics to wandb in real-time
                    if wandb_log_fn is not None:
                        # Compute batch averages for logging
                        avg_reward = rewards.mean().item()
                        avg_likelihood = likelihoods.mean().item()
                        avg_length = lengths.float().mean().item()
                        avg_best_likelihood = best_likelihoods.float().mean().item()
                        
                        # Action distribution
                        action_counts = torch.bincount(actions, minlength=3)
                        action_probs_step = action_counts.float() / batch_B
                        
                        step_log_dict = {
                            'step/avg_reward': avg_reward,
                            'step/avg_likelihood': avg_likelihood,
                            'step/avg_length': avg_length,
                            'step/avg_best_likelihood': avg_best_likelihood,
                            'step/action_prob_decrease': action_probs_step[0].item(),
                            'step/action_prob_keep': action_probs_step[1].item(),
                            'step/action_prob_increase': action_probs_step[2].item(),
                            'step/epsilon': self.current_epsilon,
                            'episode': episode,
                            'step_in_episode': step,
                            'global_step': global_step,
                            'batch_idx': batch_start // batch_size
                        }
                        wandb_log_fn(step_log_dict, step=global_step)
                
                # ===== POLICY UPDATE (only after episode completes) =====
                # Now we update the policy network using collected episode data
                # Set to training mode for gradient computation
                self.policy_net.train()
                if self.use_ppo:
                    self.value_net.train()
                
                rewards_tensor = torch.stack(episode_rewards)  # [T, batch_B]
                log_probs_tensor = torch.stack(episode_log_probs)  # [T, batch_B]
                states_tensor = torch.stack(episode_states)  # [T, batch_B, state_dim]
                action_probs_tensor = torch.stack(episode_action_probs)  # [T, batch_B, 3]
                actions_tensor = torch.stack(episode_actions)  # [T, batch_B]
                
                if self.use_ppo:
                    # PPO update with multiple epochs
                    # Compute values for all states (old policy)
                    with torch.no_grad():
                        old_values = self.value_net(states_tensor).squeeze(-1)  # [T, batch_B]
                    
                    # Compute returns and advantages using GAE
                    returns, advantages = self._compute_gae(rewards_tensor, old_values, device)
                    
                    # Normalize returns for stable value learning (store stats for denormalization)
                    returns_mean = returns.mean()
                    returns_std = returns.std() + 1e-8
                    returns_normalized = (returns - returns_mean) / returns_std
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Store old log probs for importance sampling ratio
                    old_log_probs = log_probs_tensor.detach()
                    
                    # Multiple PPO epochs
                    final_policy_loss = None
                    final_value_loss = None
                    final_entropy = None
                    for epoch in range(self.ppo_epochs):
                        # Recompute log probs and values with current policy
                        action_logits = self.policy_net(states_tensor)  # [T, batch_B, 3]
                        new_action_probs = F.softmax(action_logits / self.temperature, dim=-1)
                        new_log_probs = F.log_softmax(action_logits / self.temperature, dim=-1)
                        
                        # Get log probs for the actions that were actually taken
                        new_log_probs_selected = new_log_probs.gather(2, actions_tensor.unsqueeze(-1)).squeeze(-1)  # [T, batch_B]
                        
                        # Compute importance sampling ratio
                        ratio = torch.exp(new_log_probs_selected - old_log_probs)  # [T, batch_B]
                        
                        # Compute clipped policy loss
                        policy_loss_1 = ratio * advantages
                        policy_loss_2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
                        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                        
                        # Value loss (use normalized returns for stable learning)
                        new_values = self.value_net(states_tensor).squeeze(-1)  # [T, batch_B]
                        # Normalize new values to match normalized returns
                        new_values_normalized = (new_values - returns_mean) / returns_std
                        # Compute loss on normalized scale (much smaller, more stable)
                        value_loss = F.mse_loss(new_values_normalized, returns_normalized)
                        # Scale back to original scale for logging (multiply by std^2)
                        value_loss_scaled = value_loss * (returns_std ** 2)
                        
                        # Entropy bonus
                        entropy = -(new_action_probs * torch.log(new_action_probs + 1e-8)).sum(dim=-1).mean()
                        entropy_bonus = entropy * self.entropy_coef
                        
                        # Total loss (use normalized value loss for training, but log scaled version)
                        total_loss = policy_loss + self.ppo_value_coef * value_loss - entropy_bonus
                        
                        # Update
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                            max_norm=self.max_grad_norm
                        )
                        self.optimizer.step()
                        
                        # Store final metrics from last epoch (use scaled loss for logging)
                        final_policy_loss = policy_loss.item()
                        final_value_loss = value_loss_scaled.item()  # Use scaled loss for logging
                        final_entropy = entropy.item()
                    
                    # Track policy metrics for PPO (after all epochs)
                    avg_reward = rewards_tensor.mean().item()
                    avg_return = returns.mean().item()
                    avg_advantage = advantages.mean().item()
                    batch_policy_metrics.append({
                        'episode': episode,
                        'batch_idx': batch_start // batch_size,
                        'avg_reward': avg_reward,
                        'avg_return': avg_return,
                        'avg_advantage': avg_advantage,
                        'policy_loss': final_policy_loss,
                        'value_loss': final_value_loss,
                        'entropy': final_entropy,
                        'epsilon': self.current_epsilon
                    })
                else:
                    # REINFORCE update
                    # Compute returns
                    returns = torch.zeros_like(rewards_tensor)
                    next_return = torch.zeros(batch_B, device=device)
                    for t in reversed(range(steps_per_episode)):
                        next_return = rewards_tensor[t] + self.ppo_gamma * next_return
                        returns[t] = next_return
                    
                    # Normalize returns
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                    
                    # Compute entropy bonus (encourages exploration)
                    # Entropy: -sum(p * log(p)) for each action distribution
                    entropy = -(action_probs_tensor * torch.log(action_probs_tensor + 1e-8)).sum(dim=-1)  # [T, batch_B]
                    entropy_bonus = entropy.mean() * self.entropy_coef
                    
                    # Policy loss with entropy bonus
                    policy_loss = -(log_probs_tensor * returns.detach()).mean() - entropy_bonus
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()
                    
                    # Track policy metrics for REINFORCE
                    avg_reward = rewards_tensor.mean().item()
                    avg_return = returns.mean().item()
                    policy_loss_val = policy_loss.item()
                    entropy_val = entropy.mean().item()
                    batch_policy_metrics.append({
                        'episode': episode,
                        'batch_idx': batch_start // batch_size,
                        'avg_reward': avg_reward,
                        'avg_return': avg_return,
                        'policy_loss': policy_loss_val,
                        'entropy': entropy_val,
                        'value_loss': 0.0,  # Not applicable for REINFORCE
                        'avg_advantage': 0.0,  # Not applicable for REINFORCE
                        'epsilon': self.current_epsilon
                    })
                
                # Decay epsilon after each episode
                self.current_epsilon = max(self.epsilon_min, self.current_epsilon * self.epsilon_decay)
            
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
            # Extend traces once per prompt in the batch (traces has batch-level data, so we need one copy per prompt)
            all_traces.extend([traces] * batch_B)
            # Accumulate policy metrics
            all_policy_metrics.extend(batch_policy_metrics)
        
        return all_final_prompts, all_rewards, all_traces, all_policy_metrics

