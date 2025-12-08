"""
RL Agent for prompt optimization: handles model interactions and likelihood computation
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM


class PromptRLAgent:
    """Agent that interacts with language model for prompt optimization."""

    def __init__(self, model_name="EleutherAI/pythia-70m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.special_token_ids = {
            tok
            for tok in [
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.bos_token_id,
            ]
            if tok is not None
        }
        self.vocab_size = len(self.tokenizer)

    def _project_embeddings_to_tokens(
        self, embeds: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Project embeddings to nearest token IDs.

        Args:
            embeds: [B, L, D] tensor of embeddings
            mask: [B, L] attention mask (1 = active, 0 = inactive)

        Returns:
            token_ids: [B, L] tensor of nearest token IDs
        """
        B, L, D = embeds.shape
        embedding_layer = self.model.get_input_embeddings()
        vocab_embeds = embedding_layer.weight.detach()  # [vocab_size, D]

        # Flatten for batch processing
        flat_embeds = embeds.view(B * L, D)  # [B*L, D]

        # Compute L2 distances to all vocabulary embeddings
        distances = torch.cdist(flat_embeds, vocab_embeds)  # [B*L, vocab_size]

        # Find nearest token ID for each embedding
        token_ids_flat = distances.argmin(dim=-1)  # [B*L]

        # Reshape back to [B, L]
        token_ids = token_ids_flat.view(B, L)

        # Mask inactive positions (set to pad_id)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        token_ids = torch.where(
            mask.bool(), token_ids, torch.full_like(token_ids, pad_id)
        )

        return token_ids

    def get_likelihoods_batch(
        self, model_input, requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Batched likelihood computation using ModelBatchedInput.

        For continuous_proj mode:
        - During optimization (requires_grad=True): uses embeddings directly to preserve gradients
        - For reward computation (requires_grad=False): projects suffix embeddings to nearest token IDs
        For continuous mode: uses embeddings directly.
        For discrete mode: uses token IDs directly.

        Args:
            model_input: ModelBatchedInput instance with all inputs
            requires_grad: Whether to enable gradients
        """
        device = self.device

        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:

            # For continuous_proj mode:
            # - During optimization (requires_grad=True): use embeddings directly to preserve gradients
            # - For reward computation (requires_grad=False): project to tokens for accurate likelihood
            if model_input.original_mode == "continuous_proj" and not requires_grad:
                # Project suffix embeddings to nearest token IDs (for reward computation only)
                suffix_embeds = model_input.suffix_embeddings  # [B, max_suffix_len, D]
                suffix_mask = model_input.suffix_attention_mask  # [B, max_suffix_len]
                suffix_token_ids = self._project_embeddings_to_tokens(
                    suffix_embeds, suffix_mask
                )  # [B, max_suffix_len]

                # Temporarily store projected suffix tokens in model_input for concatenation
                # (similar to how discrete mode works)
                original_suffix_input_ids = model_input.suffix_input_ids
                model_input.suffix_input_ids = suffix_token_ids

                # Use the same concatenation logic as discrete mode
                input_ids, attention_mask = (
                    model_input.get_model_input_ids_and_attention_mask()
                )

                # Restore original suffix_input_ids (in case it's used elsewhere)
                model_input.suffix_input_ids = original_suffix_input_ids

                # Use token-based forward pass (like discrete mode)
                inputs_embeds = self.model.get_input_embeddings()(input_ids)

                # Forward pass with attention mask
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [B, seq_len, vocab]
            elif model_input.mode == "continuous" or (
                model_input.original_mode == "continuous_proj" and requires_grad
            ):
                inputs_embeds, attention_mask, suffix_mask = (
                    model_input.get_model_input_embeds_and_attention_mask()
                )
                # Prefix and completion embeddings are already detached in get_model_input_embeds_and_attention_mask
                # Only suffix embeddings have gradients
                # Forward pass with attention mask
                outputs = self.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [B, seq_len, vocab]
            elif model_input.mode == "discrete":
                # For discrete mode, we need embeddings for forward pass
                input_ids, attention_mask = (
                    model_input.get_model_input_ids_and_attention_mask()
                )

                # Forward pass with attention mask
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits  # [B, seq_len, vocab]
            else:
                raise ValueError(f"Invalid mode: {model_input.mode}")

        token_log_probs, comp_mask = self._compute_completion_token_log_probs(
            logits, model_input
        )
        masked_log_probs = torch.where(
            comp_mask, token_log_probs, torch.zeros_like(token_log_probs)
        )
        return masked_log_probs.sum(dim=-1)

    def compute_completion_log_probs(
        self, logits: torch.Tensor, model_input
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Public helper to expose completion token log probabilities for downstream users.

        Returns:
            token_log_probs: [B, max_completion_len_active]
            completion_mask: boolean mask indicating valid tokens
        """
        return self._compute_completion_token_log_probs(logits, model_input)

    def _compute_completion_token_log_probs(
        self, logits: torch.Tensor, model_input
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather log probabilities for completion tokens given logits over the packed sequence.
        """
        batch_size = logits.size(0)
        completion_mask = model_input.completion_attention_mask.bool()

        completion_lengths = completion_mask.sum(dim=1)
        max_completion_len = (
            int(completion_lengths.max().item()) if completion_lengths.numel() > 0 else 0
        )

        token_log_probs = logits.new_zeros((batch_size, max_completion_len))
        trimmed_mask = torch.zeros(
            (batch_size, max_completion_len), dtype=torch.bool, device=logits.device
        )
        completion_tokens = model_input.completion_input_ids
        completion_start_pos = model_input.get_completion_start_pos()

        for idx in range(batch_size):
            length = int(completion_lengths[idx].item())
            start = int(completion_start_pos[idx].item())
            end = start - 1 + length
            slice_logits = logits[idx, start - 1 : end, :]
            log_probs = F.log_softmax(slice_logits, dim=-1)
            tokens = completion_tokens[idx][completion_mask[idx]]
            gathered = log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
            token_log_probs[idx, :length] = gathered
            trimmed_mask[idx, :length] = True

        return token_log_probs, trimmed_mask

    def get_random_token(self) -> int:
        """
        Get a random token ID from the vocabulary, excluding special tokens.

        Returns:
            Random token ID (int)
        """
        # Sample from vocabulary, excluding special tokens
        while True:
            token_id = random.randint(0, self.vocab_size - 1)
            if token_id not in self.special_token_ids:
                return token_id
