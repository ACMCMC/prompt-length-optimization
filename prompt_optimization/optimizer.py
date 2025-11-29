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


# ============================================================================
# Complex Neural Network Architectures for Policy/Value Networks
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with layer normalization for stable training."""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: x + f(x)
        return self.activation(x + self.dropout(self.layers(x)))


class FeatureEncoder(nn.Module):
    """
    Encodes different groups of state features separately before combining.
    This allows the network to learn specialized representations for:
    - Length features (efficiency-related)
    - Performance features (likelihood-related)
    - Temporal features (episode progress)
    - History features (recent action patterns)
    """
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 64):
        super().__init__()
        # Feature group indices (matching state vector layout):
        # 0: len_norm_scaled (length)
        # 1: ll_norm (performance)
        # 2: delta_ll_norm (performance)
        # 3: best_ll_norm (performance)
        # 4: step_ratio (temporal)
        # 5: ll_per_token_norm (length/efficiency)
        # 6: steps_remaining_norm (temporal)
        # 7: recent_add_scaled (history)
        # 8: recent_remove_scaled (history)
        
        # Separate encoders for feature groups
        self.length_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # len_norm, ll_per_token
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        self.performance_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),  # ll, delta_ll, best_ll
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # step_ratio, steps_remaining
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),  # recent_add, recent_remove
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
        )
        
        # Combine all encoded features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split features into groups
        length_feats = torch.stack([x[:, 0], x[:, 5]], dim=1)      # len_norm, ll_per_token
        perf_feats = torch.stack([x[:, 1], x[:, 2], x[:, 3]], dim=1)  # ll, delta_ll, best_ll
        temporal_feats = torch.stack([x[:, 4], x[:, 6]], dim=1)    # step_ratio, steps_remaining
        history_feats = torch.stack([x[:, 7], x[:, 8]], dim=1)     # recent_add, recent_remove
        
        # Encode each group
        length_enc = self.length_encoder(length_feats)
        perf_enc = self.performance_encoder(perf_feats)
        temporal_enc = self.temporal_encoder(temporal_feats)
        history_enc = self.history_encoder(history_feats)
        
        # Concatenate and combine
        combined = torch.cat([length_enc, perf_enc, temporal_enc, history_enc], dim=1)
        return self.combiner(combined)


class ResidualPolicyNetwork(nn.Module):
    """
    Complex policy network with:
    - Feature-group encoding
    - Multiple residual blocks
    - Layer normalization
    - Dropout for regularization
    
    Architecture: 9 → FeatureEncoder(64) → 256 → ResBlock → ResBlock → 128 → 64 → 3
    Total params: ~100K (vs ~9K in simple MLP)
    """
    
    def __init__(self, state_dim: int = 9, action_dim: int = 3, 
                 hidden_dim: int = 256, num_residual_blocks: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Feature encoder (specialized encoding for different feature groups)
        self.feature_encoder = FeatureEncoder(state_dim, hidden_dim=64)
        
        # Input projection from encoded features
        self.input_proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        # Output head with gradual dimension reduction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, action_dim),
        )
        
        # Initialize output layer with small weights for stable initial policy
        nn.init.orthogonal_(self.output_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.output_head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode features by group
        x = self.feature_encoder(x)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output action logits
        return self.output_head(x)


class ResidualValueNetwork(nn.Module):
    """
    Complex value network with similar architecture to policy network.
    Outputs a single scalar value estimate.
    
    Architecture: 9 → FeatureEncoder(64) → 256 → ResBlock → ResBlock → 128 → 64 → 1
    """
    
    def __init__(self, state_dim: int = 9, hidden_dim: int = 256, 
                 num_residual_blocks: int = 2, dropout: float = 0.1):
        super().__init__()
        
        # Feature encoder (shared architecture with policy)
        self.feature_encoder = FeatureEncoder(state_dim, hidden_dim=64)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_residual_blocks)
        ])
        
        # Value output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        
        # Initialize output layer for reasonable initial value estimates
        nn.init.orthogonal_(self.output_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.output_head[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode features
        x = self.feature_encoder(x)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output value estimate
        return self.output_head(x)


# ============================================================================
# Main Optimizer Class
# ============================================================================

class LengthPolicyOptimizer:
    """RL optimizer that learns prompt length policy using PPO (or REINFORCE fallback)."""
    
    def __init__(self, agent: PromptRLAgent, reward_cfg=None, use_complex_network: bool = False):
        # reward_cfg is accepted for backward compatibility with the GCG branch; it is not used here.
        self.agent = agent
        self.emb_dim = agent.model.get_input_embeddings().weight.shape[1]
        # Default exploration params to keep logging fields defined
        self.epsilon = 0.0
        self.epsilon_decay = 1.0
        self.epsilon_min = 0.0
        self.current_epsilon = 0.0
        self.temperature = 1.0
        
        # Curriculum exploration parameters
        self.exploration_start = 0.5   # Initial probability of forced random action
        self.exploration_end = 0.05    # Final exploration probability
        self.exploration_decay_episodes = 50  # Episodes to decay exploration
        self.total_episodes_trained = 0  # Track across all prompts
        
        # Reward shaping parameters
        self.ll_threshold = -10.0     # LL above this is "good enough" - hard cap on ADD actions
        self.length_bonus_scale = 2.0 # Extra bonus for short lengths when LL is good
        
        # NEW: Efficiency-based reward mode
        self.reward_mode = 'standard'  # 'standard', 'efficiency', or 'hybrid'
        self.efficiency_alpha = 1.0   # Weight for LL per token term
        self.hard_cap_ll = -8.0       # Hard cap: block ADD actions when LL exceeds this
        
        # Action history for state representation (track last N actions per item in batch)
        self.action_history_len = 5
        
        # Enhanced policy network: state -> action probs
        # State features: [len_norm, ll, delta_ll, best_ll, step_ratio, 
        #                  ll_per_token, steps_remaining_norm, recent_add_ratio, recent_remove_ratio]
        self.state_dim = 9
        
        # Network architecture selection - can be passed from config
        self.use_complex_network = use_complex_network
        
        if self.use_complex_network:
            # Complex policy network with residual connections and layer normalization
            self.policy_net = ResidualPolicyNetwork(self.state_dim, 3).to(agent.device)
            self.value_net = ResidualValueNetwork(self.state_dim).to(agent.device)
        else:
            # Simple MLP - faster learning, more responsive to gradients
            self.policy_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),  # Tanh instead of ReLU for bounded gradients
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 3)
            ).to(agent.device)
            self.value_net = nn.Sequential(
                nn.Linear(self.state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            ).to(agent.device)
            # Initialize output layer with larger weights so initial policy isn't uniform
            nn.init.orthogonal_(self.policy_net[-1].weight, gain=1.0)  # gain=1.0, not 0.01
            nn.init.zeros_(self.policy_net[-1].bias)
        
        # Print network info
        policy_params = sum(p.numel() for p in self.policy_net.parameters())
        value_params = sum(p.numel() for p in self.value_net.parameters())
        print(f"[Network] {'Complex' if self.use_complex_network else 'Simple'} architecture")
        print(f"[Network] Policy params: {policy_params:,}, Value params: {value_params:,}")
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)  # Increased from 3e-4
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)   # Increased from 3e-4
    
    def get_exploration_prob(self, episode: int = None, total_episodes: int = None) -> float:
        """
        Get current exploration probability based on curriculum schedule.
        
        Uses a two-level decay:
        1. Within-prompt decay: decays from start to end over the episodes of current prompt
        2. Cross-prompt decay: further reduces exploration as more prompts are trained
        
        Args:
            episode: Current episode index within this prompt (0-indexed)
            total_episodes: Total episodes for this prompt
        """
        # If episode info provided, use within-prompt decay
        if episode is not None and total_episodes is not None and total_episodes > 0:
            # Within-prompt progress (0 to 1)
            within_progress = episode / total_episodes
            
            # Cross-prompt decay factor (reduces overall exploration as training progresses)
            # After exploration_decay_episodes total, cross_factor goes to 0
            if self.total_episodes_trained >= self.exploration_decay_episodes:
                cross_factor = 0.0
            else:
                cross_factor = 1.0 - (self.total_episodes_trained / self.exploration_decay_episodes)
            
            # Combine: start high, decay within prompt, and also decay across prompts
            # exploration = end + (start - end) * (1 - within_progress) * cross_factor
            base_range = self.exploration_start - self.exploration_end
            exploration = self.exploration_end + base_range * (1.0 - within_progress) * cross_factor
            return max(self.exploration_end, exploration)
        
        # Fallback: use total episodes only (legacy behavior)
        if self.total_episodes_trained >= self.exploration_decay_episodes:
            return self.exploration_end
        progress = self.total_episodes_trained / self.exploration_decay_episodes
        return self.exploration_start + progress * (self.exploration_end - self.exploration_start)
    
    def compute_shaped_reward(self, likelihoods: torch.Tensor, lengths: torch.Tensor, 
                               initial_prompt_length: int, alpha: float, beta: float,
                               actions: torch.Tensor = None, prev_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Compute shaped reward that encourages shorter lengths when LL is 'good enough'.
        
        IMPORTANT: Rewards are normalized to roughly [-1, +1] range to prevent
        exploding value loss in PPO. Raw LL values (-30 to -10) are scaled.
        
        NEW: Includes immediate action-based rewards to differentiate actions:
        - REMOVE (action=0) when LL > threshold: bonus
        - ADD (action=2) when LL > threshold: penalty
        - This gives the policy clear signal about which actions are good
        
        Supports three reward modes:
        1. 'standard': R = α * LL_norm - β * length_norm (original)
        2. 'efficiency': R = α * (LL / length)_norm - β * length_norm (LL per token)
        3. 'hybrid': Combines efficiency with bonus for short lengths when LL is good
        """
        length_norm = lengths.float() / max(1.0, float(initial_prompt_length))
        
        # Avoid division by zero
        safe_lengths = lengths.float().clamp(min=1.0)
        
        # Normalization constants for LL (typical range: -35 to -5)
        LL_MIN = -35.0  # Bad likelihood
        LL_MAX = -5.0   # Good likelihood
        LL_RANGE = LL_MAX - LL_MIN  # = 30
        
        if self.reward_mode == 'efficiency':
            ll_per_token = likelihoods / safe_lengths
            ll_per_token_norm = (ll_per_token + 0.6) * 2.0
            base_reward = alpha * ll_per_token_norm
            length_penalty = beta * (length_norm - 1.0).clamp(min=0)
            length_bonus = beta * (1.0 - length_norm).clamp(min=0)
            reward = base_reward - length_penalty + length_bonus
            
        elif self.reward_mode == 'hybrid':
            ll_per_token = likelihoods / safe_lengths
            ll_per_token_norm = (ll_per_token + 0.6) * 2.0
            base_reward = alpha * ll_per_token_norm
            length_penalty = beta * (length_norm - 1.0).clamp(min=0)
            length_bonus = beta * (1.0 - length_norm).clamp(min=0)
            good_ll_mask = likelihoods > self.ll_threshold
            extra_bonus = torch.zeros_like(likelihoods)
            if good_ll_mask.any():
                under_ratio = (1.0 - length_norm).clamp(min=0)
                extra_bonus[good_ll_mask] = self.length_bonus_scale * under_ratio[good_ll_mask]
            reward = base_reward - length_penalty + length_bonus + extra_bonus
            
        else:  # 'standard' mode
            ll_normalized = (likelihoods - LL_MIN) / LL_RANGE * 2.0 - 1.0
            base_reward = alpha * ll_normalized
            length_penalty = beta * (length_norm - 1.0).clamp(min=0)
            length_bonus = beta * (1.0 - length_norm).clamp(min=0)
            good_ll_mask = likelihoods > self.ll_threshold
            extra_bonus = torch.zeros_like(likelihoods)
            if good_ll_mask.any():
                extra_bonus[good_ll_mask] = self.length_bonus_scale * (1.0 - length_norm[good_ll_mask]).clamp(min=0)
            reward = base_reward - length_penalty + length_bonus + extra_bonus
        
        # ============ NEW: ACTION-BASED REWARD SHAPING ============
        # Give immediate feedback based on the action taken
        # This helps the policy learn which actions are good in which states
        if actions is not None:
            action_bonus = torch.zeros_like(reward)
            good_ll_mask = likelihoods > self.ll_threshold
            
            # When LL is good enough:
            # - REMOVE (action=0): reward for making prompt shorter
            # - KEEP (action=1): small reward for not making it longer
            # - ADD (action=2): penalty for making it longer unnecessarily
            remove_mask = (actions == 0) & good_ll_mask
            keep_mask = (actions == 1) & good_ll_mask
            add_mask = (actions == 2) & good_ll_mask
            
            action_bonus[remove_mask] = 0.5   # Bonus for removing when LL is good
            action_bonus[keep_mask] = 0.1     # Small bonus for keeping when LL is good
            action_bonus[add_mask] = -0.3     # Penalty for adding when LL is good
            
            # When LL is bad (below threshold):
            # - ADD might help, so no penalty
            # - REMOVE might hurt, so no bonus
            bad_ll_mask = ~good_ll_mask
            add_bad_mask = (actions == 2) & bad_ll_mask
            action_bonus[add_bad_mask] = 0.1  # Small encouragement to add when LL is bad
            
            reward = reward + action_bonus
        
        return reward
    
    def should_block_add(self, likelihoods: torch.Tensor) -> torch.Tensor:
        """
        Hard cap: return mask indicating which items should NOT be allowed to ADD.
        When LL exceeds hard_cap_ll, block ADD actions to prevent further growth.
        
        Returns:
            Boolean tensor [B] where True = block ADD action
        """
        return likelihoods > self.hard_cap_ll
    
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
                               mode: str = "continuous", batch_size: int = 64,
                               use_ppo: bool = True, ppo_epochs: int = 4, ppo_clip: float = 0.2,
                               gamma: float = 0.99, gae_lambda: float = 0.95,
                               value_coef: float = 0.5, entropy_coef: float = 0.01,
                               max_suffix_len: int = 64, init_len: int = 32,
                               wandb_log_fn=None, global_step_offset: int = 0,
                               log_prompt_indices: Optional[List[int]] = None,
                               gcg_top_k: int = 16, gcg_candidate_size: int = 32,
                               gcg_steps_per_action: int = 1,
                               base_prompts: Optional[List[str]] = None) -> Tuple[List[torch.Tensor], List[float], List[dict], List[dict]]:
        """
        Unified batch optimization using pluggable optimizer interface.
        Processes prompts in batches of batch_size (default 64) for parallelization.
        """
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], [], []
        if log_prompt_indices is None:
            log_prompt_indices = []
        
        # Process in batches of batch_size
        all_final_prompts = []
        all_rewards = []
        all_traces = []
        all_policy_metrics = []
        
        for batch_start in range(0, B, batch_size):
            batch_end = min(batch_start + batch_size, B)
            batch_completions = target_completions[batch_start:batch_end]
            batch_bases = base_prompts[batch_start:batch_end] if base_prompts is not None else None
            batch_B = len(batch_completions)
            
            completion_tokens_batch, completion_lengths = self._prepare_completions(batch_completions)
            # Prepare fixed prefixes (base prompts) if provided
            pad_id = getattr(self.agent.tokenizer, 'pad_token_id', 0)
            if batch_bases is not None:
                base_tokenized = [self.agent.tokenizer.encode(t or "", add_special_tokens=False) for t in batch_bases]
                max_prefix_len = max((len(t) for t in base_tokenized), default=0)
                prefix_tokens = torch.full((batch_B, max_prefix_len), pad_id, dtype=torch.long, device=device)
                prefix_lengths = torch.zeros(batch_B, dtype=torch.long, device=device)
                for i, toks in enumerate(base_tokenized):
                    if not toks:
                        continue
                    prefix_tokens[i, :len(toks)] = torch.tensor(toks, device=device)
                    prefix_lengths[i] = len(toks)
            else:
                prefix_tokens = torch.empty(batch_B, 0, dtype=torch.long, device=device)
                prefix_lengths = torch.zeros(batch_B, dtype=torch.long, device=device)

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
                    self.agent, initial_prompt_length, max_prompt_len, batch_B, lr_embeddings,
                    top_k=gcg_top_k, candidate_size=gcg_candidate_size,
                    gcg_steps_per_action=gcg_steps_per_action
                )
            
            # Initialize prompts (suffix)
            prompt_data, lengths = optimizer.initialize_prompts()
            
            # Initialize prefix tokens: empty initially, grows as we delete suffix tokens
            # Max prefix size = initial_prompt_length (can grow up to original suffix size)
            max_prefix_size = prefix_tokens.shape[1] if prefix_tokens.numel() > 0 else initial_prompt_length
            if prefix_tokens.shape[1] < max_prefix_size:
                # pad to max_prefix_size for uniform shape
                extra = max_prefix_size - prefix_tokens.shape[1]
                prefix_tokens = torch.cat(
                    [prefix_tokens, torch.full((batch_B, extra), pad_id, dtype=torch.long, device=device)],
                    dim=1
                )
            
            # Track attention mask offsets (how much to mask at start due to deleted tokens)
            attention_mask_offset = torch.zeros(batch_B, dtype=torch.long, device=device)
            
            best_rewards = torch.full((batch_B,), float('-inf'), dtype=torch.float32, device=device)
            best_prompts: List[Optional[torch.Tensor]] = [None] * batch_B
            
            traces = []
            batch_policy_metrics = []
        
            for episode in trange(episodes, desc=f"Episodes (batch {batch_start//batch_size + 1})"):
                # Reset prompts and length state each episode (fresh start)
                prompt_data, lengths = optimizer.initialize_prompts()
                attention_mask_offset = torch.zeros(batch_B, dtype=torch.long, device=device)
                prev_ll = torch.zeros(batch_B, device=device)
                best_ll = torch.full((batch_B,), float("-inf"), device=device)
                
                # Initialize action history for this episode [batch_B, action_history_len]
                action_history = torch.ones(batch_B, self.action_history_len, dtype=torch.long, device=device)  # Start with KEEP

                episode_rewards = []
                episode_log_probs = []
                episode_states = []
                episode_actions = []
                
                # Get current exploration probability for curriculum (scales within prompt's episodes)
                exploration_prob = self.get_exploration_prob(episode=episode, total_episodes=episodes)
                
                step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
                for step in step_bar:
                    global_step = global_step_offset + episode * steps_per_episode + step
                    # Inner optimization step (e.g., gradient updates, GCG replacements)
                    # Pass prefix info for proper likelihood computation
                    prompt_data, likelihoods = optimizer.inner_optimization_step(
                        prompt_data, lengths, completion_tokens_batch, completion_lengths, step,
                        prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths
                    )
                    
                    # Compute enhanced states for policy
                    step_ratio = step / steps_per_episode
                    steps_remaining_norm = (steps_per_episode - step) / steps_per_episode
                    len_norm = lengths.float() / initial_prompt_length
                    delta_ll = likelihoods - prev_ll
                    best_ll = torch.maximum(best_ll, likelihoods)
                    
                    # Efficiency metric: LL per token (higher = more efficient)
                    # Clamp lengths to minimum of 1 to avoid division by zero
                    safe_lengths = lengths.float().clamp(min=1.0)
                    ll_per_token = likelihoods / safe_lengths
                    
                    # Sanitize likelihoods and derived values to prevent NaN/inf in states
                    # Replace -inf with a large negative value, +inf with large positive, NaN with 0
                    likelihoods_safe = torch.where(torch.isinf(likelihoods) | torch.isnan(likelihoods),
                                                   torch.full_like(likelihoods, -50.0), likelihoods)
                    delta_ll_safe = torch.where(torch.isinf(delta_ll) | torch.isnan(delta_ll),
                                                torch.zeros_like(delta_ll), delta_ll)
                    best_ll_safe = torch.where(torch.isinf(best_ll) | torch.isnan(best_ll),
                                               torch.full_like(best_ll, -50.0), best_ll)
                    ll_per_token_safe = torch.where(torch.isinf(ll_per_token) | torch.isnan(ll_per_token),
                                                    torch.full_like(ll_per_token, -5.0), ll_per_token)
                    
                    # ========== NORMALIZE STATE FEATURES TO [-1, +1] RANGE ==========
                    # This helps the policy/value networks learn more effectively
                    
                    # len_norm: already in [0, ~2], normalize to [-1, 1] centered at 1.0 (init_len)
                    len_norm_scaled = (len_norm - 1.0)  # Now centered at 0: negative = shorter, positive = longer
                    
                    # likelihood: typical range [-50, 0], normalize to [-1, 1]
                    # Map -50 -> -1, 0 -> +1
                    LL_MIN, LL_MAX = -50.0, 0.0
                    ll_norm = (likelihoods_safe - LL_MIN) / (LL_MAX - LL_MIN) * 2.0 - 1.0
                    ll_norm = ll_norm.clamp(-1.0, 1.0)
                    
                    # delta_ll: typical range [-10, +10], normalize to [-1, 1]
                    delta_ll_norm = (delta_ll_safe / 10.0).clamp(-1.0, 1.0)
                    
                    # best_ll: same normalization as likelihood
                    best_ll_norm = (best_ll_safe - LL_MIN) / (LL_MAX - LL_MIN) * 2.0 - 1.0
                    best_ll_norm = best_ll_norm.clamp(-1.0, 1.0)
                    
                    # ll_per_token: typical range [-5, 0] for reasonable prompts, normalize
                    # Map -5 -> -1, 0 -> +1
                    ll_per_token_norm = (ll_per_token_safe / 2.5 + 1.0).clamp(-1.0, 1.0)
                    
                    # Action history features: ratio of recent ADD (2) and REMOVE (0) actions
                    recent_add_ratio = (action_history == 2).float().mean(dim=1)
                    recent_remove_ratio = (action_history == 0).float().mean(dim=1)
                    
                    # Scale ratios to [-1, 1] centered at 0.5
                    recent_add_scaled = (recent_add_ratio - 0.5) * 2.0
                    recent_remove_scaled = (recent_remove_ratio - 0.5) * 2.0
                    
                    states = torch.stack([
                        len_norm_scaled,        # 0: normalized length (centered at init_len)
                        ll_norm,                # 1: current likelihood (normalized)
                        delta_ll_norm,          # 2: change from previous step (normalized)
                        best_ll_norm,           # 3: running best LL (normalized)
                        torch.full((batch_B,), step_ratio * 2.0 - 1.0, device=device),  # 4: step ratio [-1, 1]
                        ll_per_token_norm,      # 5: efficiency (normalized)
                        torch.full((batch_B,), steps_remaining_norm * 2.0 - 1.0, device=device),  # 6: time budget [-1, 1]
                        recent_add_scaled,      # 7: recent ADD frequency [-1, 1]
                        recent_remove_scaled,   # 8: recent REMOVE frequency [-1, 1]
                    ], dim=1)  # [batch_B, state_dim=9]
                    prev_ll = likelihoods_safe.detach()
                    print("states:", states)
                    # Policy forward pass
                    action_logits = self.policy_net(states)  # [batch_B, 3]
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    # Curriculum exploration: force random actions early in training
                    # Always log the exploration probability (not whether this step was random)
                    self.current_epsilon = exploration_prob
                    if exploration_prob > 0 and torch.rand(1).item() < exploration_prob:
                        # Forced exploration: sample uniformly random action
                        actions = torch.randint(0, 3, (batch_B,), device=device)
                    else:
                        actions = torch.multinomial(action_probs, 1).squeeze(-1)  # [batch_B]
                    
                    print("actions:", actions)
                    print("action_probs:", action_probs)
                    print("exploration_prob:", exploration_prob)
                    # Mask "add" when at or above max_prompt_len to prevent runaway growth
                    if hasattr(optimizer, "max_prompt_len"):
                        add_mask = (actions == 2) & (lengths >= optimizer.max_prompt_len)
                        if add_mask.any():
                            actions = actions.clone()
                            actions[add_mask] = 1  # convert to KEEP
                    
                    # MINIMUM LENGTH: Block REMOVE actions when at minimum length (1 token)
                    # This prevents prompts from becoming empty (which causes -inf likelihood)
                    MIN_PROMPT_LEN = 2  # Minimum prompt length to maintain
                    remove_blocked = (actions == 0) & (lengths <= MIN_PROMPT_LEN)
                    if remove_blocked.any():
                        actions = actions.clone()
                        actions[remove_blocked] = 1  # Convert to KEEP
                    
                    # HARD CAP: Block ADD actions when LL is already good enough
                    # This prevents the policy from learning "ADD = better LL = higher reward"
                    hard_cap_mask = self.should_block_add(likelihoods_safe)
                    add_blocked = (actions == 2) & hard_cap_mask
                    if add_blocked.any():
                        actions = actions.clone()
                        # When LL is good and trying to ADD, convert to KEEP (not REMOVE to avoid over-shrinking)
                        actions[add_blocked] = 1  # Force KEEP instead of REMOVE
                        
                    if batch_B == 1:
                        log_probs = F.log_softmax(action_logits, dim=-1)[0, actions].unsqueeze(0)
                    else:
                        log_probs = F.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(-1)
                    episode_actions.append(actions)
                    
                    # Update action history (shift left and add new action)
                    action_history = torch.cat([action_history[:, 1:], actions.unsqueeze(1)], dim=1)
                    
                    # Apply length actions and handle prefix shifting
                    prompt_data, lengths, prefix_tokens, prefix_lengths, attention_mask_offset = self._apply_length_action_with_prefix(
                        optimizer, prompt_data, lengths, actions, prefix_tokens, prefix_lengths, 
                        attention_mask_offset, max_prefix_size, pad_id
                    )
                
                    # Compute shaped rewards (with bonus for short lengths when LL is good)
                    # Use sanitized likelihoods to avoid NaN rewards
                    # Pass actions for action-based reward shaping
                    rewards = self.compute_shaped_reward(likelihoods_safe, lengths, initial_prompt_length, alpha, beta, actions=actions)
                    
                    # Sanitize rewards to prevent NaN/inf propagation
                    rewards = torch.where(torch.isinf(rewards) | torch.isnan(rewards),
                                         torch.full_like(rewards, -10.0), rewards)
                    
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

                    # Optional wandb logging hook (step-level)
                    if wandb_log_fn is not None:
                        avg_reward = rewards.mean().item()
                        avg_likelihood = likelihoods.mean().item()
                        avg_length = lengths.float().mean().item()
                        avg_best_likelihood = best_rewards.mean().item()
                        action_counts = torch.bincount(actions, minlength=3)
                        action_probs_step = action_counts.float() / batch_B
                        step_log = {
                            'step/avg_reward': avg_reward,
                            'step/avg_likelihood': avg_likelihood,
                            'step/avg_length': avg_length,
                            'step/min_length': lengths.min().item(),
                            'step/max_length': lengths.max().item(),
                            'step/avg_best_reward': avg_best_likelihood,
                            'step/action_prob_decrease': action_probs_step[0].item(),
                            'step/action_prob_keep': action_probs_step[1].item(),
                            'step/action_prob_increase': action_probs_step[2].item(),
                            'step/epsilon': self.current_epsilon,
                            'episode': episode,
                            'step_in_episode': step,
                            'global_step': global_step,
                            'batch_idx': batch_start // batch_size
                        }
                        for idx in log_prompt_indices:
                            if 0 <= idx < batch_B:
                                step_log.update({
                                    f"prompt/{idx}/reward": rewards[idx].item(),
                                    f"prompt/{idx}/likelihood": likelihoods[idx].item(),
                                    f"prompt/{idx}/length": int(lengths[idx].item())
                                })
                        try:
                            wandb_log_fn(step_log)
                        except Exception:
                            pass
                
                # Policy update (PPO by default)
                rewards_tensor = torch.stack(episode_rewards)  # [T, batch_B]
                log_probs_tensor = torch.stack(episode_log_probs)  # [T, batch_B]
                states_tensor = torch.stack(episode_states)  # [T, batch_B, state_dim]
                T = states_tensor.shape[0]

                states_flat = states_tensor.view(T * batch_B, self.state_dim)
                values_flat = self.value_net(states_flat).squeeze()
                values_tensor = values_flat.view(T, batch_B)

                last_policy_loss = 0.0
                last_value_loss = 0.0
                last_entropy = 0.0

                if use_ppo:
                    advantages = torch.zeros_like(rewards_tensor, device=device)
                    last_gae = torch.zeros(batch_B, dtype=torch.float32, device=device)
                    next_value = torch.zeros(batch_B, dtype=torch.float32, device=device)
                    for t in reversed(range(T)):
                        delta = rewards_tensor[t] + gamma * next_value - values_tensor[t]
                        last_gae = delta + gamma * gae_lambda * last_gae
                        advantages[t] = last_gae
                        next_value = values_tensor[t]

                    returns_tensor = advantages + values_tensor
                    advantages_flat = advantages.view(-1)
                    returns_flat = returns_tensor.view(-1)
                    # Detach to avoid backpropagating through time/reuse in multiple PPO epochs
                    advantages_flat = advantages_flat.detach()
                    returns_flat = returns_flat.detach()
                    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8 + 1e-12)

                    old_log_probs_flat = log_probs_tensor.view(-1).detach()
                    actions_flat = torch.stack(episode_actions).view(-1)

                    for _ in range(max(1, ppo_epochs)):
                        policy_logits = self.policy_net(states_flat)
                        new_log_probs = F.log_softmax(policy_logits, dim=-1).gather(1, actions_flat.unsqueeze(1)).squeeze(1)
                        entropy = -(F.softmax(policy_logits, dim=-1) * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()

                        ratios = torch.exp(new_log_probs - old_log_probs_flat)
                        surr1 = ratios * advantages_flat
                        surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_flat
                        policy_loss = -torch.mean(torch.min(surr1, surr2))

                        value_preds = self.value_net(states_flat).squeeze()
                        value_loss = F.mse_loss(value_preds, returns_flat)

                        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                        last_policy_loss = policy_loss.item()
                        last_value_loss = value_loss.item()
                        last_entropy = entropy.item()
                        self.policy_optimizer.zero_grad()
                        self.value_optimizer.zero_grad()
                        total_loss.backward()
                        self.policy_optimizer.step()
                        self.value_optimizer.step()
                else:
                    # Fallback REINFORCE with value baseline
                    returns = torch.zeros_like(rewards_tensor)
                    next_return = torch.zeros(batch_B, device=device)
                    for t in reversed(range(T)):
                        next_return = rewards_tensor[t] + gamma * next_return
                        returns[t] = next_return
                    returns_flat = returns.view(-1)
                    advantages = returns_flat - values_flat.detach()
                    policy_loss = -(log_probs_tensor.view(-1) * advantages.detach()).mean()
                    value_loss = F.mse_loss(values_flat, returns_flat)
                    total_loss = policy_loss + value_coef * value_loss
                    last_policy_loss = policy_loss.item()
                    last_value_loss = value_loss.item()
                    last_entropy = 0.0
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
                
                traces.append({
                    'episode': episode,
                    'rewards': [float(r) for r in rewards_tensor[-1]],
                    'likelihoods': [float(ll) for ll in likelihoods],  # Final likelihoods at end of episode
                    'lengths': [int(l) for l in lengths]
                })
                
                # Increment total episodes trained (for curriculum exploration decay)
                self.total_episodes_trained += 1

                # Collect policy metrics for this episode
                batch_policy_metrics.append({
                    'episode': episode,
                    'batch_idx': batch_start // batch_size,
                    'avg_reward': float(rewards_tensor.mean().item()),
                    'avg_return': float(returns_tensor.mean().item() if use_ppo else returns.mean().item()),
                    'avg_advantage': float(advantages_flat.mean().item()) if use_ppo else 0.0,
                    'policy_loss': float(last_policy_loss),
                    'value_loss': float(last_value_loss),
                    'entropy': float(last_entropy),
                    'epsilon': exploration_prob,  # Log exploration probability
                    'step': global_step_offset + episode * steps_per_episode
                })

                # Optional wandb logging hook
                if wandb_log_fn is not None:
                    try:
                        wandb_log_fn({
                            'policy/avg_reward': float(rewards_tensor.mean().item()),
                            'policy/avg_return': float(returns_tensor.mean().item() if use_ppo else returns.mean().item()),
                            'policy/avg_advantage': float(advantages_flat.mean().item()) if use_ppo else 0.0,
                            'policy/policy_loss': float(last_policy_loss),
                            'policy/value_loss': float(last_value_loss),
                            'policy/entropy': float(last_entropy),
                            'policy/exploration_prob': exploration_prob,  # Curriculum exploration
                            'policy/episode': episode,
                            'policy/batch_idx': batch_start // batch_size
                        })
                        # Length histogram per episode for visibility
                        try:
                            import wandb as _wandb  # type: ignore
                            wandb_log_fn({
                                'episode/length_hist': _wandb.Histogram(lengths.detach().cpu().numpy()),
                                'episode': episode,
                                'batch_idx': batch_start // batch_size
                            })
                        except Exception:
                            pass
                    except Exception:
                        pass
            
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
            all_policy_metrics.extend(batch_policy_metrics)
        
        return all_final_prompts, all_rewards, all_traces, all_policy_metrics
