"""
Continuous prompt optimization with projection regularization.
Optimizes both likelihood and distance to nearest vocabulary token.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from ..interface import BasePromptOptimizer

class ContinuousPromptOptimizerWithProjection(BasePromptOptimizer):
    """
    Optimizes prompts in continuous embedding space with regularization
    to minimize distance to nearest vocabulary token.
    This ensures embeddings can be projected to tokens with minimal information loss.
    """
    
    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, 
                 projection_weight: float = 0.1, distance_metric: str = "l2"):
        """
        Args:
            agent: PromptRLAgent instance
            initial_prompt_length: Starting prompt length
            max_prompt_len: Maximum prompt length
            batch_size: Batch size
            lr_embeddings: Learning rate for embeddings
            projection_weight: Weight for projection regularization term (default 0.1)
            distance_metric: "l2" or "dot" for distance computation
        """
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.D = self.emb_dim
        self.projection_weight = projection_weight
        self.distance_metric = distance_metric
        
        # Pre-compute vocabulary embeddings for efficiency
        with torch.no_grad():
            self.vocab_embeds = self.embedding_layer.weight.detach()  # [vocab_size, D]
        
        # Initialize prompt embeddings as learnable parameters
        self.prompt_embeds = nn.Parameter(
            torch.randn(batch_size, max_prompt_len, self.D, device=self.device) * 0.1
        )
        self.prompt_optimizer = optim.Adam([self.prompt_embeds], lr=lr_embeddings)
    
    def initialize_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with random embeddings."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length, 
                           dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.prompt_embeds.data = torch.randn(
                self.batch_size, self.max_prompt_len, self.D, device=self.device
            ) * 0.1
        return self.prompt_embeds, lengths
    
    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                       requires_grad: bool = False) -> torch.Tensor:
        """Compute likelihoods from embeddings."""
        max_active_len = lengths.max().item()
        active_embeds = prompt_data[:, :max_active_len]
        return self.agent.get_likelihoods_batch(
            active_embeds, completion_tokens, completion_lengths, requires_grad=requires_grad
        )
    
    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens by modifying lengths and initializing new positions."""
        updated_lengths = lengths.clone()
        for i in range(self.batch_size):
            action = actions[i].item()
            if action == 0 and lengths[i] > 0:  # remove
                updated_lengths[i] -= 1
            elif action == 2 and lengths[i] < self.max_prompt_len:  # add
                prompt_data.data[i, lengths[i]] = torch.randn(1, self.D, device=self.device) * 0.1
                updated_lengths[i] += 1
        return prompt_data, updated_lengths
    
    def _compute_projection_loss(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to nearest vocabulary token for each embedding.
        
        Args:
            embeds: [B, L, D] or [L, D] tensor of embeddings
            
        Returns:
            loss: Scalar loss (mean distance to nearest vocab token)
        """
        if embeds.dim() == 2:
            embeds = embeds.unsqueeze(0)  # [1, L, D]
        
        B, L, D = embeds.shape
        
        # Flatten for batch processing
        flat_embeds = embeds.view(B * L, D)  # [B*L, D]
        
        if self.distance_metric == "l2":
            # L2 distance: ||e - v||^2
            distances = torch.cdist(flat_embeds, self.vocab_embeds)  # [B*L, vocab_size]
            min_distances = distances.min(dim=-1)[0]  # [B*L]
            loss = min_distances.mean()
        else:  # dot product
            # Negative dot product (maximize similarity = minimize negative dot)
            # Normalize embeddings first
            flat_embeds_norm = torch.nn.functional.normalize(flat_embeds, p=2, dim=-1)
            vocab_embeds_norm = torch.nn.functional.normalize(self.vocab_embeds, p=2, dim=-1)
            similarities = torch.matmul(flat_embeds_norm, vocab_embeds_norm.t())  # [B*L, vocab_size]
            max_similarities = similarities.max(dim=-1)[0]  # [B*L]
            # Convert to distance: 1 - similarity (since similarity is in [-1, 1])
            min_distances = 1.0 - max_similarities
            loss = min_distances.mean()
        
        return loss
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                               step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize embeddings for both likelihood and proximity to vocabulary tokens.
        Uses gradient-free random walk with acceptance, considering both objectives.
        """
        max_active_len = lengths.max().item()
        active_embeds = prompt_data[:, :max_active_len].clone()  # Clone to avoid view issues
        
        # Get baseline likelihoods and projection loss
        with torch.no_grad():
            base_likelihoods = self.agent.get_likelihoods_batch(
                active_embeds, completion_tokens, completion_lengths, requires_grad=False
            )
            base_proj_loss = self._compute_projection_loss(active_embeds)
            # Combined objective: maximize likelihood, minimize projection distance
            base_objective = base_likelihoods.mean() - self.projection_weight * base_proj_loss
        
        # Try small random perturbations (accept if combined objective improves)
        for _ in range(3):
            noise = torch.randn_like(active_embeds) * 0.01
            test_embeds = active_embeds + noise
            with torch.no_grad():
                test_likelihoods = self.agent.get_likelihoods_batch(
                    test_embeds, completion_tokens, completion_lengths, requires_grad=False
                )
                test_proj_loss = self._compute_projection_loss(test_embeds)
                test_objective = test_likelihoods.mean() - self.projection_weight * test_proj_loss
            
            # Accept improvements in combined objective
            for i in range(self.batch_size):
                # Individual objective for this prompt
                base_obj_i = base_likelihoods[i] - self.projection_weight * base_proj_loss
                test_obj_i = test_likelihoods[i] - self.projection_weight * test_proj_loss
                
                if test_obj_i > base_obj_i:
                    active_embeds[i] = test_embeds[i]
                    base_likelihoods[i] = test_likelihoods[i]
                    base_objective = test_objective
        
        # Copy back to prompt_data
        prompt_data.data[:, :max_active_len] = active_embeds
        
        return prompt_data, base_likelihoods
    
    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Project embeddings to nearest token IDs."""
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        tokens = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        
        for i in range(B):
            length = lengths[i].item()
            if length > 0:
                embeds = prompt_data[i, :length].detach()  # [length, D]
                distances = torch.cdist(embeds, self.vocab_embeds)  # [length, vocab_size]
                token_ids = distances.argmin(dim=-1)  # [length]
                tokens[i, :length] = token_ids
        
        return tokens
    
    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's embeddings."""
        return prompt_data[idx, :length].clone().detach()

