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
    """RL optimizer that learns prompt length policy using GRPO (Group Relative Policy Optimization)."""
    
    def __init__(self, agent: PromptRLAgent, epsilon: float, epsilon_decay: float, epsilon_min: float,
                 entropy_coef: float, temperature: float,
                 grpo_clip: float, grpo_epochs: int,
                 grpo_gamma: float, grpo_gae_lambda: float, grpo_value_coef: float,
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
        
        # Value network for GRPO (estimates state values)
        # Initialize output layer to predict values around typical return scale
        # This helps the network start in the right range
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, 1)  # Single value output
        ).to(agent.device)
        
        # Initialize value network with standard initialization (no bias preset)
        # Let the network learn the value scale naturally
        with torch.no_grad():
            # Initialize last layer with standard initialization
            if len(self.value_net) > 0:
                last_layer = self.value_net[-1]
                if isinstance(last_layer, nn.Linear):
                    # Use standard initialization (bias starts at 0, weights from xavier)
                    nn.init.xavier_uniform_(last_layer.weight, gain=value_init_gain)
        
        self.max_grad_norm = max_grad_norm
        
        # Shared optimizer for policy and value networks (GRPO)
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=3e-4
        )
        
        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.current_epsilon = epsilon
        
        # Entropy bonus for exploration (encourages diverse action distributions)
        self.entropy_coef = entropy_coef
        
        # Temperature for softmax (higher = more exploration)
        self.temperature = temperature
        
        # GRPO-specific parameters
        self.grpo_clip = grpo_clip
        self.grpo_epochs = grpo_epochs
        self.grpo_gamma = grpo_gamma
        self.grpo_gae_lambda = grpo_gae_lambda
        self.grpo_value_coef = grpo_value_coef
    
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
        deltas = rewards + self.grpo_gamma * next_values - values
        
        # Compute GAE advantages: A_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.grpo_gamma * self.grpo_gae_lambda * gae
            advantages[t] = gae
        
        # Returns are advantages + values
        returns = advantages + values
        
        return returns, advantages
    
    def _compute_grpo_advantages(self, rewards: torch.Tensor, values: torch.Tensor, 
                                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) with global normalization (GRPO).
        Advantages are normalized across all trajectories in the batch, making the policy
        learn from relative performance rather than absolute reward values.
        
        Args:
            rewards: [T, batch_B] tensor of rewards
            values: [T, batch_B] tensor of value estimates
            device: torch device
            
        Returns:
            returns: [T, batch_B] tensor of returns (value targets)
            advantages: [T, batch_B] tensor of advantages (globally normalized)
        """
        # Compute GAE for all trajectories
        returns, advantages = self._compute_gae(rewards, values, device)
        
        # Global normalization: normalize advantages across entire batch
        # This ensures best trajectories get highest advantages regardless of absolute reward scale
        advantages_mean = advantages.mean()
        advantages_std = advantages.std() + 1e-8
        advantages = (advantages - advantages_mean) / advantages_std
        
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
                               wandb_log_fn=None, global_step_offset: int = 0,
                               rollouts_per_prompt: int = 1) -> Tuple[List[torch.Tensor], List[float], List[dict], List[dict]]:
        """
        Unified batch optimization using pluggable optimizer interface.
        Processes prompts in batches of batch_size (default 64) for parallelization.
        
        Args:
            rollouts_per_prompt: Number of trajectories (rollouts) to run per prompt.
                                Total batch size = num_prompts * rollouts_per_prompt.
                                Advantages are computed per prompt across its rollouts.
        """
        device = self.agent.device
        num_prompts = len(target_completions)
        if num_prompts == 0:
            return [], [], [], []
        
        # Expand batch: duplicate each prompt rollouts_per_prompt times
        # This creates multiple trajectories per prompt for proper advantage computation
        expanded_completions = []
        prompt_indices = []  # Track which prompt each rollout belongs to
        for prompt_idx, completion in enumerate(target_completions):
            for rollout_idx in range(rollouts_per_prompt):
                expanded_completions.append(completion)
                prompt_indices.append(prompt_idx)
        
        B = len(expanded_completions)  # Total batch size = num_prompts * rollouts_per_prompt
        prompt_indices_tensor = torch.tensor(prompt_indices, device=device)  # [B]
        
        # Process in batches, ensuring we process complete groups of rollouts per prompt
        # batch_size refers to the number of prompts to process, not the number of rollouts
        all_final_prompts = []
        all_rewards = []
        all_traces = []
        all_policy_metrics = []  # Track policy training metrics
        
        # Process prompts in groups of batch_size (each prompt has rollouts_per_prompt rollouts)
        num_prompts_per_batch = batch_size  # Number of prompts per processing batch
        for prompt_batch_start in range(0, num_prompts, num_prompts_per_batch):
            prompt_batch_end = min(prompt_batch_start + num_prompts_per_batch, num_prompts)
            num_prompts_in_batch = prompt_batch_end - prompt_batch_start
            
            # Get all rollouts for these prompts (each prompt has rollouts_per_prompt rollouts)
            rollout_batch_start = prompt_batch_start * rollouts_per_prompt
            rollout_batch_end = prompt_batch_end * rollouts_per_prompt
            batch_completions = expanded_completions[rollout_batch_start:rollout_batch_end]
            batch_prompt_indices = prompt_indices_tensor[rollout_batch_start:rollout_batch_end]  # Track prompt indices for this batch
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
        
            # Store batch_prompt_indices for use in policy update (clone to avoid modifying original)
            batch_prompt_indices_for_episode = batch_prompt_indices.clone()
            
            batch_idx = prompt_batch_start // num_prompts_per_batch
            for episode in trange(episodes, desc=f"Episodes (batch {batch_idx + 1})"):
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
                episode_actions = []  # Store actions for GRPO importance sampling computation
                
                logger.info(f"Starting optimization: Episode {episode+1}/{episodes}, Batch {batch_idx + 1}, {steps_per_episode} steps")
                
                # Set policy to eval mode during episode (no gradients, only inference)
                self.policy_net.eval()
                self.value_net.eval()
                
                step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
                for step in step_bar:
                    # ===== PROMPT OPTIMIZATION ONLY (no policy updates) =====
                    # Inner optimization step: optimize prompt embeddings/tokens only
                    # This does NOT update the policy network
                    if step == 0 or (step + 1) % 10 == 0 or step == steps_per_episode - 1:
                        logger.info(f"  Optimization step {step+1}/{steps_per_episode} (Episode {episode+1}, Batch {batch_idx + 1})")
                    
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
                            'batch_idx': batch_idx
                        }
                        
                        # Log min_distances for continuous_proj mode
                        if hasattr(optimizer, 'min_distances'):
                            step_log_dict['step/avg_min_distance'] = optimizer.min_distances
                        wandb_log_fn(step_log_dict, step=global_step)
                
                # ===== POLICY UPDATE (only after episode completes) =====
                # Now we update the policy network using collected episode data
                # Set to training mode for gradient computation
                self.policy_net.train()
                self.value_net.train()
                
                rewards_tensor = torch.stack(episode_rewards)  # [T, batch_B]
                log_probs_tensor = torch.stack(episode_log_probs)  # [T, batch_B]
                states_tensor = torch.stack(episode_states)  # [T, batch_B, state_dim]
                action_probs_tensor = torch.stack(episode_action_probs)  # [T, batch_B, 3]
                actions_tensor = torch.stack(episode_actions)  # [T, batch_B]
                
                # GRPO update with multiple epochs
                # Compute values for all states (old policy)
                with torch.no_grad():
                    old_values = self.value_net(states_tensor).squeeze(-1)  # [T, batch_B]
                
                # Compute returns and advantages using GAE with global normalization (GRPO)
                # Advantages are normalized across all trajectories, making policy learn from relative performance
                returns, advantages = self._compute_grpo_advantages(
                    rewards_tensor, old_values, device
                )
                
                # Normalize returns for stable value learning (store stats for denormalization)
                # Note: advantages are already globally normalized in _compute_grpo_advantages
                returns_mean = returns.mean()
                returns_std = returns.std() + 1e-8
                returns_normalized = (returns - returns_mean) / returns_std
                
                # Store old log probs for importance sampling ratio
                old_log_probs = log_probs_tensor.detach()
                
                # Multiple GRPO epochs
                final_policy_loss = None
                final_value_loss = None
                final_entropy = None
                for epoch in range(self.grpo_epochs):
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
                    policy_loss_2 = torch.clamp(ratio, 1.0 - self.grpo_clip, 1.0 + self.grpo_clip) * advantages
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
                    total_loss = policy_loss + self.grpo_value_coef * value_loss - entropy_bonus
                    
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
                
                # Track policy metrics for GRPO (after all epochs)
                avg_reward = rewards_tensor.mean().item()
                avg_return = returns.mean().item()
                avg_advantage = advantages.mean().item()  # Should be ~0 after normalization
                std_advantage = advantages.std().item()  # Should be ~1 after normalization
                
                # Monitor value predictions vs returns to verify learning
                # Use final values from last epoch (after updates)
                with torch.no_grad():
                    final_values = self.value_net(states_tensor).squeeze(-1)  # [T, batch_B]
                avg_value_pred = final_values.mean().item()
                avg_return_actual = returns.mean().item()
                value_pred_error = (final_values - returns).abs().mean().item()  # Mean absolute error
                
                batch_policy_metrics.append({
                    'episode': episode,
                    'batch_idx': batch_idx,
                    'avg_reward': avg_reward,
                    'avg_return': avg_return,
                    'avg_advantage': avg_advantage,
                    'std_advantage': std_advantage,
                    'policy_loss': final_policy_loss,
                    'value_loss': final_value_loss,
                    'entropy': final_entropy,
                    'epsilon': self.current_epsilon,
                    'avg_value_pred': avg_value_pred,
                    'value_pred_error': value_pred_error
                })
                
                # Decay epsilon after each episode
                self.current_epsilon = max(self.epsilon_min, self.current_epsilon * self.epsilon_decay)
            
            # Aggregate best results per prompt (across its rollouts) before converting to tokens
            # We have batch_B rollouts, but only num_prompts_in_batch unique prompts
            # Note: batch_prompt_indices_for_episode contains indices relative to the original target_completions list
            # We need to map them to local batch indices (0, 1, 2, ...) for this processing batch
            unique_prompts_in_batch = torch.unique(batch_prompt_indices_for_episode, sorted=True)
            num_prompts_in_batch = len(unique_prompts_in_batch)
            
            # Map original prompt indices to local batch indices (0, 1, 2, ...)
            # The original indices might be [10, 11, 12] but we want [0, 1, 2] for batch_prompts
            # Since unique_prompts_in_batch is sorted, we can use enumerate
            prompt_idx_to_local = {int(prompt_idx.item()): local_idx 
                                   for local_idx, prompt_idx in enumerate(unique_prompts_in_batch)}
            
            batch_best_rewards_per_prompt = torch.full((num_prompts_in_batch,), float('-inf'), dtype=torch.float32, device=device)
            batch_best_prompts_per_prompt: List[Optional[torch.Tensor]] = [None] * num_prompts_in_batch
            
            # Group rollouts by prompt and find best for each prompt
            for original_prompt_idx in unique_prompts_in_batch:
                local_prompt_idx = prompt_idx_to_local[int(original_prompt_idx.item())]
                # Find all rollouts for this prompt
                mask = (batch_prompt_indices_for_episode == original_prompt_idx)
                prompt_best_rewards = best_rewards[mask]
                prompt_best_idx = prompt_best_rewards.argmax().item()
                # Get the global index in the batch
                global_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                best_rollout_idx = global_indices[prompt_best_idx].item()
                
                batch_best_rewards_per_prompt[local_prompt_idx] = best_rewards[best_rollout_idx]
                batch_best_prompts_per_prompt[local_prompt_idx] = best_prompts[best_rollout_idx]
            
            # Convert best prompts to tokens (one per original prompt)
            batch_final_prompts = []
            if batch_best_prompts_per_prompt and any(bp is not None for bp in batch_best_prompts_per_prompt):
                # Prepare batch data for to_tokens
                is_embeddings = None
                max_len = 0
                valid_indices = []
                for i, bp in enumerate(batch_best_prompts_per_prompt):
                    if bp is not None and bp.numel() > 0:
                        max_len = max(max_len, bp.shape[0])
                        if is_embeddings is None:
                            is_embeddings = len(bp.shape) > 1
                        valid_indices.append(i)
                
                if max_len > 0 and valid_indices:
                    # Create batch tensor
                    if is_embeddings:
                        prompt_data_batch = torch.zeros(num_prompts_in_batch, max_len, optimizer.emb_dim, device=device)
                    else:
                        prompt_data_batch = torch.zeros(num_prompts_in_batch, max_len, dtype=torch.long, device=device)
                    lengths_batch = torch.zeros(num_prompts_in_batch, dtype=torch.long, device=device)
                    
                    # Copy valid prompts
                    for i in valid_indices:
                        bp = batch_best_prompts_per_prompt[i]
                        length = bp.shape[0]
                        if length > 0:
                            prompt_data_batch[i, :length] = bp
                            lengths_batch[i] = length
                    
                    # Use optimizer's to_tokens method
                    tokens_batch = optimizer.to_tokens(prompt_data_batch, lengths_batch)
                    
                    # Extract prompts
                    lengths_list = lengths_batch.tolist()
                    batch_final_prompts = [tokens_batch[i, :lengths_list[i]] if lengths_list[i] > 0 
                                   else torch.tensor([], dtype=torch.long, device=device) 
                                   for i in range(num_prompts_in_batch)]
                else:
                    batch_final_prompts = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_prompts_in_batch)]
            else:
                batch_final_prompts = [torch.tensor([], dtype=torch.long, device=device) for _ in range(num_prompts_in_batch)]
            
            # Accumulate results from this batch (one per original prompt)
            all_final_prompts.extend(batch_final_prompts)
            all_rewards.extend([float(r) for r in batch_best_rewards_per_prompt])
            # Extend traces once per prompt in the batch
            all_traces.extend([traces] * num_prompts_in_batch)
            # Accumulate policy metrics
            all_policy_metrics.extend(batch_policy_metrics)
        
        return all_final_prompts, all_rewards, all_traces, all_policy_metrics

