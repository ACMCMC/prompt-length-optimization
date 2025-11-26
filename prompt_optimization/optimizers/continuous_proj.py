"""
Continuous prompt optimization with projection regularization.
Optimizes both likelihood and distance to nearest vocabulary token.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from ..interface import BasePromptOptimizer
from ..model_inputs import ModelBatchedInput

class ContinuousPromptOptimizerWithProjection(BasePromptOptimizer):
    """
    Optimizes prompts in continuous embedding space with regularization
    to minimize distance to nearest vocabulary token.
    This ensures embeddings can be projected to tokens with minimal information loss.
    """
    
    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, 
                 projection_weight: float, distance_metric: str,
                 max_suffix_len: int, init_len: int):
        """
        Args:
            agent: PromptRLAgent instance
            initial_prompt_length: Starting prompt length
            max_prompt_len: Maximum prompt length (for suffix, should equal max_suffix_len)
            batch_size: Batch size
            lr_embeddings: Learning rate for embeddings
            projection_weight: Weight for projection regularization term (default 0.1)
            distance_metric: "l2" or "dot" for distance computation
            max_suffix_len: Maximum suffix length from config (fixed size for batched suffix embeddings)
            init_len: Initial number of suffix positions with attention mask = 1 from config
        """
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings, max_suffix_len, init_len)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.D = self.emb_dim
        self.projection_weight = float(projection_weight)
        self.distance_metric = str(distance_metric)
        
        # Pre-compute vocabulary embeddings for efficiency
        with torch.no_grad():
            self.vocab_embeds = self.embedding_layer.weight.detach()  # [vocab_size, D]
        
        # Initialize prompt embeddings as learnable parameters (will be initialized with BOS via ModelBatchedInput)
        # Suffix size is fixed to max_suffix_len from config
        self.prompt_embeds = nn.Parameter(
            torch.zeros(batch_size, self.max_suffix_len, self.emb_dim, device=self.device)
        )
        self.prompt_optimizer = optim.Adam([self.prompt_embeds], lr=lr_embeddings)
    
    def initialize_prompts(self, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS embeddings from ModelBatchedInput."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length, 
                           dtype=torch.long, device=self.device)
        # Reset embeddings to BOS token embedding from ModelBatchedInput
        with torch.no_grad():
            self.prompt_embeds.data = model_input.initialize_suffix_embeddings()
        return self.prompt_embeds, lengths
    
    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       model_input: ModelBatchedInput, requires_grad: bool = False) -> torch.Tensor:
        """Compute likelihoods from embeddings using ModelBatchedInput."""
        # Update model_input with current suffix embeddings
        model_input.update_suffix_embeddings(prompt_data)
        return self.agent.get_likelihoods_batch(model_input, requires_grad=requires_grad)
    
    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens by modifying lengths and initializing new positions."""
        # Vectorized length updates: remove (action=0) and add (action=2)
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_suffix_len)
        
        updated_lengths = lengths - remove_mask.long() + add_mask.long()
        
        # For add (action=2): initialize new position (will be set to BOS when updated via model_input)
        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            # Find first inactive position for each item
            for idx in add_indices:
                current_active = lengths[idx].item()
                if current_active < self.max_suffix_len:
                    # Initialize with zero, will be set to BOS when updated via model_input
                    prompt_data.data[idx, current_active].zero_()
        
        return prompt_data, updated_lengths
    
    def _compute_projection_loss(self, embeds: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute projection loss (mean distance to nearest vocab token).
        
        Args:
            embeds: [B, L, D] or [L, D] tensor of embeddings
            mask: [B, L] optional mask indicating which positions to include (1 = active, 0 = inactive)
                  If embeds is 2D, mask should be [L] or None
            
        Returns:
            loss: Scalar loss (mean distance to nearest vocab token, only over masked positions)
        """
        # Handle 2D input (single prompt)
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)  # [1, L, D]
            if mask is not None and mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [1, L]
        
        B, L, D = embeds.shape
        
        # Flatten for batch processing
        flat_embeds = embeds.reshape(B * L, D)  # [B*L, D]
        
        if self.distance_metric == "l2":
            # L2 distance: ||e - v||^2
            distances = torch.cdist(flat_embeds, self.vocab_embeds)  # [B*L, vocab_size]
            min_distances = distances.min(dim=-1)[0]  # [B*L]
        else:  # dot product
            # Negative dot product (maximize similarity = minimize negative dot)
            # Normalize embeddings first
            flat_embeds_norm = torch.nn.functional.normalize(flat_embeds, p=2, dim=-1)
            vocab_embeds_norm = torch.nn.functional.normalize(self.vocab_embeds, p=2, dim=-1)
            similarities = torch.matmul(flat_embeds_norm, vocab_embeds_norm.t())  # [B*L, vocab_size]
            max_similarities = similarities.max(dim=-1)[0]  # [B*L]
            # Convert to distance: 1 - similarity (since similarity is in [-1, 1])
            min_distances = 1.0 - max_similarities
        
        # Reshape to [B, L] and apply mask if provided
        min_distances = min_distances.reshape(B, L)  # [B, L]
        
        if mask is not None:
            # Only compute loss over active positions (where mask == 1)
            masked_distances = min_distances * mask  # [B, L]
            num_active = mask.sum()  # Total number of active positions
            if num_active > 0:
                loss = masked_distances.sum() / num_active
            else:
                loss = torch.tensor(0.0, device=embeds.device, requires_grad=True)
        else:
            # No mask: compute mean over all positions
            loss = min_distances.mean()
        
        return loss
    
    def _compute_projection_loss_per_prompt(self, embeds: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute per-prompt projection loss (mean distance to nearest vocab token per prompt).
        
        Args:
            embeds: [B, L, D] tensor of embeddings
            lengths: [B] tensor of actual lengths
            
        Returns:
            losses: [B] tensor of per-prompt projection losses
        """
        B, L, D = embeds.shape
        max_len = lengths.max().item()
        
        if max_len == 0:
            return torch.zeros(B, device=self.device)
        
        # Extract active embeddings only
        active_embeds = embeds[:, :max_len]  # [B, max_len, D]
        
        # Flatten for batch processing
        flat_embeds = active_embeds.view(B * max_len, D)  # [B*max_len, D]
        
        if self.distance_metric == "l2":
            distances = torch.cdist(flat_embeds, self.vocab_embeds)  # [B*max_len, vocab_size]
            min_distances = distances.min(dim=-1)[0]  # [B*max_len]
        else:  # dot product
            flat_embeds_norm = torch.nn.functional.normalize(flat_embeds, p=2, dim=-1)
            vocab_embeds_norm = torch.nn.functional.normalize(self.vocab_embeds, p=2, dim=-1)
            similarities = torch.matmul(flat_embeds_norm, vocab_embeds_norm.t())  # [B*max_len, vocab_size]
            max_similarities = similarities.max(dim=-1)[0]  # [B*max_len]
            min_distances = 1.0 - max_similarities
        
        # Reshape and compute mean per prompt, masking invalid positions
        min_distances = min_distances.view(B, max_len)  # [B, max_len]
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(-1)
        masked_distances = torch.where(length_mask, min_distances, torch.zeros_like(min_distances))
        per_prompt_losses = masked_distances.sum(dim=-1) / lengths.float().clamp(min=1.0)  # [B]
        
        return per_prompt_losses
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               step: int, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient descent optimization step to maximize likelihood while minimizing projection loss.
        """
        max_active_len = lengths.max().item()
        if max_active_len == 0:
            return prompt_data, torch.zeros(prompt_data.shape[0], dtype=torch.float32, device=prompt_data.device)
        
        # Ensure embeddings require gradients
        if not prompt_data.requires_grad:
            prompt_data.requires_grad_(True)
        
        # Update model_input with current suffix embeddings
        model_input.update_suffix_embeddings(prompt_data)
        
        # Compute likelihoods with gradients
        likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=True)
        
        # Compute projection loss with gradients enabled (only on active positions)
        # Use the suffix attention mask from model_input to determine which positions are active
        # This preserves the connection to the original suffix positions and avoids indexing issues
        suffix_mask = model_input.suffix_attention_mask  # [B, max_suffix_len]
        proj_loss = self._compute_projection_loss(prompt_data, mask=suffix_mask)
        
        # Combined loss: negative likelihood (maximize) + projection loss (minimize)
        loss = -likelihoods.mean() + self.projection_weight * proj_loss
        
        # Gradient descent step
        self.prompt_optimizer.zero_grad()
        loss.backward()
        self.prompt_optimizer.step()
        
        # Get final likelihoods (detached for return)
        # Update model_input with the updated parameter (after optimizer.step())
        # Use the parameter directly - it's been updated by the optimizer
        with torch.no_grad():
            model_input.update_suffix_embeddings(prompt_data)
            final_likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        return prompt_data, final_likelihoods
    
    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Project embeddings to nearest token IDs (vectorized, memory-efficient)."""
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        
        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)
        
        # Extract active embeddings only (masked by lengths)
        active_embeds = prompt_data[:, :max_len].detach()  # [B, max_len, D]
        D = active_embeds.shape[-1]
        
        # Flatten for memory-efficient batch processing: [B*max_len, D]
        flat_embeds = active_embeds.view(B * max_len, D)
        
        # Compute distances: [B*max_len, vocab_size]
        # This is more memory-efficient than [B, max_len, vocab_size]
        if self.distance_metric == "l2":
            distances = torch.cdist(flat_embeds, self.vocab_embeds)  # [B*max_len, vocab_size]
        else:  # dot product
            flat_embeds_norm = torch.nn.functional.normalize(flat_embeds, p=2, dim=-1)
            vocab_embeds_norm = torch.nn.functional.normalize(self.vocab_embeds, p=2, dim=-1)
            similarities = torch.matmul(flat_embeds_norm, vocab_embeds_norm.t())  # [B*max_len, vocab_size]
            # Convert similarity to distance (negate since we want to minimize distance = maximize similarity)
            distances = -similarities
        
        # Find nearest token: [B*max_len]
        tokens_flat = distances.argmin(dim=-1)
        
        # Reshape back to [B, max_len]
        tokens = tokens_flat.view(B, max_len)
        
        # Mask invalid positions (beyond actual length)
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))
        
        return tokens
    
    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's embeddings."""
        return prompt_data[idx, :length].clone().detach()

