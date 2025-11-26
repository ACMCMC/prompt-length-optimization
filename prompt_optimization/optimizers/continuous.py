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
                 batch_size: int, lr_embeddings: float, max_suffix_len: int = 64, init_len: int = 32):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings, max_suffix_len, init_len)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.D = self.emb_dim
        
        # Get BOS token embedding for initialization
        bos_token_id = agent.tokenizer.bos_token_id if agent.tokenizer.bos_token_id is not None else agent.tokenizer.eos_token_id
        with torch.no_grad():
            self.bos_embed = self.embedding_layer(torch.tensor([bos_token_id], device=self.device)).squeeze(0)
        
        # Initialize prompt embeddings as learnable parameters (all BOS)
        # Suffix size is fixed to max_suffix_len from config
        self.prompt_embeds = nn.Parameter(
            self.bos_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, self.max_suffix_len, -1).clone()
        )
        self.prompt_optimizer = optim.Adam([self.prompt_embeds], lr=lr_embeddings)
    
    def initialize_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS embeddings."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length, 
                           dtype=torch.long, device=self.device)
        # Reset embeddings to BOS token embedding
        with torch.no_grad():
            self.prompt_embeds.data = self.bos_embed.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.max_suffix_len, -1).clone()
        return self.prompt_embeds, lengths
    
    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                       requires_grad: bool = False, prefix_tokens: torch.Tensor = None,
                       prefix_lengths: torch.Tensor = None) -> torch.Tensor:
        """Compute likelihoods from embeddings using attention masks for inactive positions."""
        B, max_len, D = prompt_data.shape
        
        # Create attention mask for suffix: 1 for active positions (0 to length[i]), 0 for inactive
        suffix_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i in range(B):
            if lengths[i] > 0:
                suffix_attention_mask[i, :lengths[i]] = 1
        
        # Pass full embeddings with attention mask
        return self.agent.get_likelihoods_batch(
            prompt_data, completion_tokens, completion_lengths, requires_grad=requires_grad,
            prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths,
            suffix_attention_mask=suffix_attention_mask,
            max_suffix_len=self.max_suffix_len, init_len=self.init_len
        )
    
    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens by modifying lengths and initializing new positions."""
        # Vectorized length updates: remove (action=0) and add (action=2)
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_suffix_len)
        
        updated_lengths = lengths - remove_mask.long() + add_mask.long()
        
        # Vectorized initialization of new positions with BOS token embedding
        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            add_positions = lengths[add_indices]
            # Use BOS token embedding for new tokens (expand to match batch size)
            new_embeds = self.bos_embed.unsqueeze(0).expand(len(add_indices), -1).clone()
            prompt_data.data[add_indices, add_positions] = new_embeds
        
        return prompt_data, updated_lengths
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                               step: int, prefix_tokens: torch.Tensor = None,
                               prefix_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient descent optimization step to maximize likelihood.
        """
        # Ensure embeddings require gradients
        if not prompt_data.requires_grad:
            prompt_data.requires_grad_(True)
        
        # Use full embeddings (all positions initialized to BOS)
        # Attention mask will handle inactive positions
        B, max_len, D = prompt_data.shape
        
        # Create attention mask for suffix: 1 for active positions, 0 for inactive
        B, max_len, D = prompt_data.shape
        suffix_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for i in range(B):
            if lengths[i] > 0:
                suffix_attention_mask[i, :lengths[i]] = 1
        
        # Compute likelihoods with gradients enabled, using full embeddings
        likelihoods = self.agent.get_likelihoods_batch(
            prompt_data, completion_tokens, completion_lengths, requires_grad=True,
            prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths,
            suffix_attention_mask=suffix_attention_mask,
            max_suffix_len=self.max_suffix_len, init_len=self.init_len
        )
        
        # Loss: negative likelihood (we want to maximize likelihood = minimize negative likelihood)
        loss = -likelihoods.mean()
        
        # Gradient descent step
        self.prompt_optimizer.zero_grad()
        loss.backward()
        self.prompt_optimizer.step()
        
        # Get final likelihoods (detached for return)
        with torch.no_grad():
            B, max_len, D = prompt_data.shape
            suffix_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
            for i in range(B):
                if lengths[i] > 0:
                    suffix_attention_mask[i, :lengths[i]] = 1
            final_likelihoods = self.agent.get_likelihoods_batch(
                prompt_data, completion_tokens, completion_lengths, requires_grad=False,
                prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths,
                suffix_attention_mask=suffix_attention_mask,
                max_suffix_len=self.max_suffix_len, init_len=self.init_len
            )
        
        return prompt_data, final_likelihoods
    
    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Project embeddings to nearest token IDs (vectorized)."""
        vocab_embeds = self.embedding_layer.weight.detach()  # [vocab_size, D]
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        
        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)
        
        # Extract active embeddings only (masked by lengths)
        active_embeds = prompt_data[:, :max_len].detach()  # [B, max_len, D]
        
        # Batched distance computation: [B, max_len, vocab_size]
        distances = torch.cdist(active_embeds, vocab_embeds)
        
        # Batched argmin: [B, max_len]
        tokens = distances.argmin(dim=-1)
        
        # Mask invalid positions (beyond actual length)
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))
        
        return tokens
    
    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's embeddings."""
        return prompt_data[idx, :length].clone().detach()

