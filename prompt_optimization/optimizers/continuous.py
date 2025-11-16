"""
Continuous prompt optimization: optimizes embeddings directly, then projects to tokens.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from ..interface import BasePromptOptimizer

class ContinuousPromptOptimizer(BasePromptOptimizer):
    """Optimizes prompts in continuous embedding space, then projects to tokens."""
    
    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.D = self.emb_dim
        
        # Initialize prompt embeddings as learnable parameters
        self.prompt_embeds = nn.Parameter(
            torch.randn(batch_size, max_prompt_len, self.D, device=self.device) * 0.1
        )
        self.prompt_optimizer = optim.Adam([self.prompt_embeds], lr=lr_embeddings)
    
    def initialize_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with random embeddings."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length, 
                           dtype=torch.long, device=self.device)
        # Reset embeddings to small random values
        with torch.no_grad():
            self.prompt_embeds.data = torch.randn(self.batch_size, self.max_prompt_len, self.D, device=self.device) * 0.1
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
                # Initialize new position with small random embedding
                prompt_data.data[i, lengths[i]] = torch.randn(1, self.D, device=self.device) * 0.1
                updated_lengths[i] += 1
        return prompt_data, updated_lengths
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                               step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient-free random walk with acceptance.
        This avoids CUDA memory issues from gradient computation.
        """
        max_active_len = lengths.max().item()
        active_embeds = prompt_data[:, :max_active_len].clone()  # Clone to avoid view issues
        
        # Get baseline likelihoods
        with torch.no_grad():
            base_likelihoods = self.agent.get_likelihoods_batch(
                active_embeds, completion_tokens, completion_lengths, requires_grad=False
            )
        
        # Try small random perturbations (accept if better)
        for _ in range(3):
            noise = torch.randn_like(active_embeds) * 0.01
            test_embeds = active_embeds + noise
            with torch.no_grad():
                test_likelihoods = self.agent.get_likelihoods_batch(
                    test_embeds, completion_tokens, completion_lengths, requires_grad=False
                )
            # Accept improvements
            for i in range(self.batch_size):
                if test_likelihoods[i] > base_likelihoods[i]:
                    active_embeds[i] = test_embeds[i]
                    base_likelihoods[i] = test_likelihoods[i]
        
        # Copy back to prompt_data
        prompt_data.data[:, :max_active_len] = active_embeds
        
        return prompt_data, base_likelihoods
    
    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Project embeddings to nearest token IDs."""
        vocab_embeds = self.embedding_layer.weight.detach()  # [vocab_size, D]
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        tokens = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        
        for i in range(B):
            length = lengths[i].item()
            if length > 0:
                embeds = prompt_data[i, :length].detach()  # [length, D]
                distances = torch.cdist(embeds, vocab_embeds)  # [length, vocab_size]
                token_ids = distances.argmin(dim=-1)  # [length]
                tokens[i, :length] = token_ids
        
        return tokens
    
    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's embeddings."""
        return prompt_data[idx, :length].clone().detach()

