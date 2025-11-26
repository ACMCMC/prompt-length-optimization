"""
Discrete prompt optimization: placeholder for GCG-based implementation.
Your research colleague will implement this.
"""

import torch
import random
from typing import Tuple
from ..interface import BasePromptOptimizer
from ..model_inputs import ModelBatchedInput

class DiscretePromptOptimizer(BasePromptOptimizer):
    """
    Placeholder for discrete/GCG optimization.
    TODO: Implement GCG-based token optimization here.
    """
    
    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, max_suffix_len: int, init_len: int):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings, max_suffix_len, init_len)
        self.embedding_layer = agent.model.get_input_embeddings()
        
        # Initialize with zeros (will be initialized with BOS via ModelBatchedInput)
        self.prompt_tokens = torch.zeros(
            (batch_size, max_prompt_len), dtype=torch.long, device=self.device
        )
    
    def initialize_prompts(self, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS tokens from ModelBatchedInput."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length,
                           dtype=torch.long, device=self.device)
        # Reset to BOS tokens from ModelBatchedInput
        self.prompt_tokens = model_input.initialize_suffix_tokens()
        return self.prompt_tokens, lengths
    
    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       model_input: ModelBatchedInput, requires_grad: bool = False) -> torch.Tensor:
        """Compute likelihoods from tokens using ModelBatchedInput."""
        # Update model_input with current suffix tokens
        model_input.update_suffix_tokens(prompt_data)
        return self.agent.get_likelihoods_batch(model_input, requires_grad=requires_grad)
    
    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens."""
        # Vectorized length updates: remove (action=0) and add (action=2)
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_prompt_len)
        
        updated_lengths = lengths - remove_mask.long() + add_mask.long()
        
        # Vectorized initialization of new positions
        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            add_positions = lengths[add_indices]
            new_tokens = torch.tensor(
                [self.agent.get_random_token() for _ in range(len(add_indices))],
                dtype=torch.long, device=self.device
            )
            prompt_data[add_indices, add_positions] = new_tokens
        
        return prompt_data, updated_lengths
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               step: int, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Implement GCG optimization here.
        This is a placeholder that does minimal token replacement.
        """
        # Update model_input with current suffix tokens
        model_input.update_suffix_tokens(prompt_data)
        likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        # Minimal optimization: only every 3rd step, limited items
        if step % 3 == 0:
            num_to_optimize = min(8, self.batch_size)
            indices = random.sample(range(self.batch_size), num_to_optimize) if self.batch_size > num_to_optimize else list(range(self.batch_size))
            max_active_len = lengths.max().item()
            active_tokens = prompt_data[:, :max_active_len]
            
            for i in indices:
                if lengths[i] > 0:
                    best_ll = likelihoods[i].item()
                    best_tokens = active_tokens[i].clone()
                    for _ in range(2):
                        pos = random.randint(0, lengths[i].item() - 1)
                        candidate = self.agent.get_random_token()
                        test_tokens = best_tokens.clone()
                        test_tokens[pos] = candidate
                        # Create test prompt_data with updated token
                        test_prompt_data = prompt_data.clone()
                        test_prompt_data[i, pos] = candidate
                        
                        # Update model_input temporarily for test
                        model_input.update_suffix_tokens(test_prompt_data)
                        test_ll = self.agent.get_likelihoods_batch(model_input, requires_grad=False)[i].item()
                        
                        if test_ll > best_ll:
                            best_ll = test_ll
                            best_tokens = test_tokens
                            prompt_data[i, :lengths[i]] = best_tokens
        
        # Update model_input with final suffix tokens
        model_input.update_suffix_tokens(prompt_data)
        final_likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        return prompt_data, final_likelihoods
    
    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Tokens are already token IDs, just pad/trim to max length (vectorized)."""
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        
        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)
        
        # Extract active tokens and pad with zeros
        tokens = prompt_data[:, :max_len].clone()  # [B, max_len]
        
        # Mask invalid positions (beyond actual length)
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))
        
        return tokens
    
    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's tokens."""
        return prompt_data[idx, :length].clone()

