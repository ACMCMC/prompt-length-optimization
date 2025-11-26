"""
Discrete prompt optimization using a batched GCG-like update.
Ported from the GCG_atharv_working branch (prompt_rl_poc.py) to fit the current optimizer interface.
"""

import torch
import random
from typing import Tuple, List, Optional
from ..interface import BasePromptOptimizer


class DiscretePromptOptimizer(BasePromptOptimizer):
    """
    Discrete/GCG optimizer: works in token space and uses gradient information to propose replacements.
    """

    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, max_suffix_len: int = None, init_len: int = None):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings)
        self.embedding_layer = agent.model.get_input_embeddings()
        # Initialize with random non-special tokens
        self.prompt_tokens = torch.zeros(batch_size, max_prompt_len, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            for j in range(max_prompt_len):
                self.prompt_tokens[i, j] = self.agent.get_random_token()

    def initialize_prompts(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize prompts with random valid tokens."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length, dtype=torch.long, device=self.device)
        for i in range(self.batch_size):
            for j in range(self.max_prompt_len):
                self.prompt_tokens[i, j] = self.agent.get_random_token()
        return self.prompt_tokens, lengths

    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                       requires_grad: bool = False, prefix_tokens: torch.Tensor = None,
                       prefix_lengths: torch.Tensor = None) -> torch.Tensor:
        """Compute likelihoods from tokens."""
        max_active_len = lengths.max().item()
        active_tokens = prompt_data[:, :max_active_len]
        prompt_embeds = self.embedding_layer(active_tokens)
        return self.agent.get_likelihoods_batch(
            prompt_embeds, completion_tokens, completion_lengths, requires_grad=requires_grad,
            prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths
        )

    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens (random token on add)."""
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_prompt_len)

        updated_lengths = lengths - remove_mask.long() + add_mask.long()

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
                               completion_tokens: torch.Tensor, completion_lengths: torch.Tensor,
                               step: int, prefix_tokens: torch.Tensor = None,
                               prefix_lengths: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched discrete optimization inspired by GCG: compute gradients to propose replacements.
        """
        device = self.device
        B = prompt_data.shape[0]
        max_active_len = lengths.max().item()
        active_tokens = prompt_data[:, :max_active_len]
        embedding_layer = self.embedding_layer
        vocab_embeds = embedding_layer.weight.detach()
        vocab_size = vocab_embeds.shape[0]

        # Build prompt embeddings requiring grad
        prompt_embeds = embedding_layer(active_tokens).detach().requires_grad_(True)  # [B, L, D]

        # Compute likelihoods and gradients
        likelihoods = self.agent.get_likelihoods_batch(
            prompt_embeds, completion_tokens, completion_lengths, requires_grad=True,
            prefix_tokens=prefix_tokens, prefix_lengths=prefix_lengths
        )
        loss = -likelihoods.sum()
        self.agent.model.zero_grad(set_to_none=True)
        loss.backward()

        grads = prompt_embeds.grad.detach()  # [B, L, D]

        # Vectorized candidate proposal: top-k per position
        top_k = min(16, vocab_size)
        grads_flat = (-grads).reshape(B * max_active_len, -1)
        scores = torch.matmul(grads_flat, vocab_embeds.t()).reshape(B, max_active_len, vocab_size)
        if self.agent.special_token_ids:
            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
            mask[list(self.agent.special_token_ids)] = True
            scores[..., mask] = float('-inf')
        _, topk_idx = torch.topk(scores, k=top_k, dim=-1)  # [B, L, top_k]

        best_tokens_batch = active_tokens.clone()
        best_ll_batch = likelihoods.clone().detach()

        # For each position, try best candidate token and keep if improves likelihood
        for b in range(B):
            L = lengths[b].item()
            if L == 0:
                continue
            for pos in range(L):
                candidates = topk_idx[b, pos]
                for cand in candidates:
                    cand_token = int(cand.item())
                    if cand_token == int(active_tokens[b, pos].item()):
                        continue
                    test_tokens = best_tokens_batch[b].clone()
                    test_tokens[pos] = cand_token
                    test_embeds = embedding_layer(test_tokens.unsqueeze(0))
                    test_ll = self.agent.get_likelihoods_batch(
                        test_embeds, completion_tokens[b:b+1],
                        completion_lengths[b:b+1],
                        requires_grad=False,
                        prefix_tokens=prefix_tokens[b:b+1] if prefix_tokens is not None else None,
                        prefix_lengths=prefix_lengths[b:b+1] if prefix_lengths is not None else None
                    )[0]
                    if test_ll.item() > best_ll_batch[b].item():
                        best_ll_batch[b] = test_ll
                        best_tokens_batch[b] = test_tokens

        # Update prompt_data with improved tokens
        prompt_data[:, :max_active_len] = best_tokens_batch
        likelihoods = best_ll_batch
        return prompt_data, likelihoods

    def to_tokens(self, prompt_data: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Tokens are already token IDs, just pad/trim to max length."""
        B = prompt_data.shape[0]
        max_len = lengths.max().item()
        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)
        tokens = prompt_data[:, :max_len].clone()
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))
        return tokens

    def clone_prompt(self, prompt_data: torch.Tensor, idx: int, length: int) -> torch.Tensor:
        """Clone a single prompt's tokens."""
        return prompt_data[idx, :length].clone()
