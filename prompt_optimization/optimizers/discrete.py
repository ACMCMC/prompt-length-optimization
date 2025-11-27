"""
Discrete prompt optimization using the official GCG-style gradient + sampling update.
"""

import torch
from typing import Tuple, List, Optional
from ..interface import BasePromptOptimizer
from prompt_optimization.gcg_official import token_gradients, sample_control


class DiscretePromptOptimizer(BasePromptOptimizer):
    """
    Discrete/GCG optimizer: works in token space and uses gradient information to propose replacements.
    """

    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, max_suffix_len: int = None, init_len: int = None,
                 top_k: int = 16, candidate_size: int = 32):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.top_k = max(1, int(top_k))
        self.candidate_size = max(1, int(candidate_size))
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
        Batched discrete optimization using the official GCG gradient + sampling update.
        """
        device = self.device
        B = prompt_data.shape[0]
        max_active_len = lengths.max().item()
        active_tokens = prompt_data[:, :max_active_len]
        embedding_layer = self.embedding_layer

        best_tokens_batch = active_tokens.clone()
        best_ll_batch = torch.full((B,), float("-inf"), device=device)

        # Evaluate each example separately (closest to reference implementation)
        for b in range(B):
            L = lengths[b].item()
            comp_len = completion_lengths[b].item()
            if L == 0 or comp_len == 0:
                continue

            control_tokens = active_tokens[b, :L]
            comp_tokens = completion_tokens[b, :comp_len]

            # Build full sequence: control + completion
            input_ids = torch.cat([control_tokens, comp_tokens], dim=0)
            control_slice = slice(0, L)
            target_slice = slice(L, L + comp_len)
            loss_slice = slice(L - 1, L - 1 + comp_len)

            # Compute gradients using official token_gradients
            self.agent.model.zero_grad(set_to_none=True)
            grad = token_gradients(self.agent.model, input_ids, control_slice, target_slice, loss_slice)

            # Sample candidate controls
            not_allowed = torch.tensor(list(self.agent.special_token_ids), device=device) if self.agent.special_token_ids else None
            candidates = sample_control(
                control_tokens,
                grad,
                batch_size=self.candidate_size,
                topk=self.top_k,
                temp=1,
                not_allowed_tokens=not_allowed
            )

            # Score candidates with batched likelihood for efficiency
            cand_embeds = embedding_layer(candidates)  # [K, L, D]
            comp_tokens_batch = comp_tokens.unsqueeze(0).expand(candidates.shape[0], -1)
            comp_lengths_batch = torch.full((candidates.shape[0],), comp_len, device=device, dtype=torch.long)

            ll_batch = self.agent.get_likelihoods_batch(
                cand_embeds,
                comp_tokens_batch,
                comp_lengths_batch,
                requires_grad=False,
                prefix_tokens=None,
                prefix_lengths=None
            )

            if ll_batch.numel() > 0:
                best_idx = torch.argmax(ll_batch)
                best_tokens_batch[b, :L] = candidates[best_idx]
                best_ll_batch[b] = ll_batch[best_idx]
            else:
                # fallback to original tokens
                orig_embeds = embedding_layer(control_tokens.unsqueeze(0))
                orig_ll = self.agent.get_likelihoods_batch(
                    orig_embeds,
                    comp_tokens.unsqueeze(0),
                    torch.tensor([comp_len], device=device),
                    requires_grad=False
                )[0]
                best_ll_batch[b] = orig_ll

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
