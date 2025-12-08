"""
ModelBatchedInput: Container for batched model inputs (prefix, suffix, completion).
Handles tokenization, embedding computation, and attention mask management.
"""

from __future__ import annotations

import torch
from typing import List, Tuple, Optional


class ModelBatchedInput:
    """
    Container for batched model inputs with prefix, suffix, and completion.

    For continuous mode: stores embeddings for prefix/completion, updates suffix embeddings.
    For discrete mode: stores input_ids only, updates suffix tokens.
    """

    def __init__(
        self,
        prefix_texts: List[str],
        completion_texts: List[str],
        tokenizer,
        device: torch.device,
        embedding_layer: torch.nn.Module,
        max_suffix_len: int,
        init_len: int,
        mode: str,
    ):
        """
        Initialize batched model inputs.

        Args:
            prefix_texts: List of prefix text strings (can be empty strings)
            completion_texts: List of completion text strings
            tokenizer: Tokenizer instance (for encoding text to tokens)
            device: torch.device for tensor placement
            embedding_layer: Embedding layer (for converting tokens to embeddings)
            max_suffix_len: Maximum suffix length (fixed size)
            init_len: Initial number of active suffix positions
            mode: 'continuous', 'continuous_proj', or 'discrete'
        """
        assert mode in [
            "continuous",
            "continuous_proj",
            "discrete",
        ], f"mode must be 'continuous', 'continuous_proj', or 'discrete', got {mode}"
        # Normalize mode: continuous_proj uses embeddings like continuous
        self.mode = "continuous" if mode == "continuous_proj" else mode
        self.original_mode = mode  # Keep original for reference
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_layer = embedding_layer
        self.batch_size = len(completion_texts)
        self.max_suffix_len = max_suffix_len
        self.init_len = init_len
        self.pad_id = getattr(tokenizer, "pad_token_id", 0)
        self.vocab_size = len(tokenizer)
        self.special_token_ids = {
            tok
            for tok in [
                self.pad_id,
                tokenizer.eos_token_id,
                tokenizer.bos_token_id,
            ]
            if tok is not None
        }
        self._cached_suffix_start_pos: Optional[torch.Tensor] = None
        self._cached_completion_start_pos: Optional[torch.Tensor] = None
        self._cached_total_lengths: Optional[torch.Tensor] = None

        # Tokenize prefix and completion
        self._tokenize_prefix(prefix_texts)
        self._tokenize_completion(completion_texts)

        # Initialize suffix
        self._initialize_suffix()

        # Compute embeddings for continuous mode (including continuous_proj)
        if self.mode == "continuous":
            self._compute_embeddings()
        else:
            self.prefix_embeddings = None
            self.completion_embeddings = None
            self.suffix_embeddings = None

    def _invalidate_cached_positions(self):
        """Clear cached start/length tensors when sequence layout changes."""
        self._cached_suffix_start_pos = None
        self._cached_completion_start_pos = None
        self._cached_total_lengths = None

    def _tokenize_prefix(self, prefix_texts: List[str]):
        """Tokenize prefix texts and create attention masks using tokenizer batching."""
        # Save original padding side and set to left for prefix
        original_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
        self.tokenizer.padding_side = 'left'
        
        # Batch tokenize with padding
        tokenized = self.tokenizer(
            prefix_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        self.prefix_input_ids = tokenized["input_ids"].to(self.device)
        self.prefix_attention_mask = tokenized["attention_mask"].to(self.device)
        self._invalidate_cached_positions()

    def _tokenize_completion(self, completion_texts: List[str]):
        """Tokenize completion texts and create attention masks using tokenizer batching."""
        # Save original padding side and set to right for completion
        original_padding_side = getattr(self.tokenizer, 'padding_side', 'right')
        self.tokenizer.padding_side = 'right'
        
        # Use tokenizer's batch processing with right padding
        tokenized = self.tokenizer(
            completion_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            truncation=False,
        )
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        self.completion_input_ids = tokenized["input_ids"].to(self.device)
        self.completion_attention_mask = tokenized["attention_mask"].to(self.device)
        self._invalidate_cached_positions()

    def _initialize_suffix(self):
        """Initialize suffix with BOS tokens and attention mask."""
        bos_token_id = self._get_bos_token_id()

        # Initialize suffix input_ids to BOS tokens (inactive positions stay BOS)
        self.suffix_input_ids = torch.full(
            (self.batch_size, self.max_suffix_len),
            bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        # Initialize suffix attention mask: first init_len positions active
        self.suffix_attention_mask = torch.zeros(
            self.batch_size, self.max_suffix_len, dtype=torch.long, device=self.device
        )
        self.suffix_attention_mask[:, : self.init_len] = 1

        # Replace active positions with random tokens to avoid identical BOS initialization
        if self.init_len > 0:
            num_active = self.batch_size * self.init_len
            random_tokens = self._sample_random_tokens(num_active).view(
                self.batch_size, self.init_len
            )
            self.suffix_input_ids[:, : self.init_len] = random_tokens
        self._invalidate_cached_positions()

    def _get_bos_token_id(self) -> int:
        """Get BOS token ID, falling back to EOS if BOS is not available."""
        bos_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if bos_token_id is None:
            bos_token_id = 0
        return bos_token_id

    def _sample_random_tokens(self, count: int) -> torch.Tensor:
        """Sample random token IDs excluding known special tokens."""
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        random_tokens = torch.randint(
            0, self.vocab_size, (count,), device=self.device, dtype=torch.long
        )
        if self.special_token_ids:
            special_tensor = torch.tensor(
                list(self.special_token_ids), device=self.device, dtype=torch.long
            )
            invalid_mask = torch.isin(random_tokens, special_tensor)
            while invalid_mask.any():
                num_invalid = invalid_mask.sum().item()
                random_tokens[invalid_mask] = torch.randint(
                    0, self.vocab_size, (num_invalid,), device=self.device, dtype=torch.long
                )
                invalid_mask = torch.isin(random_tokens, special_tensor)
        return random_tokens

    def get_bos_embedding(self) -> torch.Tensor:
        """Get BOS token embedding for initialization (continuous mode)."""
        bos_token_id = self._get_bos_token_id()
        with torch.no_grad():
            bos_embed = self.embedding_layer(
                torch.tensor([bos_token_id], device=self.device)
            ).squeeze(0)
        return bos_embed

    def initialize_suffix_embeddings(self) -> torch.Tensor:
        """
        Initialize suffix embeddings with BOS embeddings (continuous mode).
        Returns initialized embeddings [B, max_suffix_len, D].
        """
        assert self.mode in [
            "continuous",
            "continuous_proj",
        ], "initialize_suffix_embeddings only for continuous modes"
        bos_embed = self.get_bos_embedding()
        suffix_embeds = (
            bos_embed.unsqueeze(0)
            .unsqueeze(0)
            .expand(self.batch_size, self.max_suffix_len, -1)
            .clone()
        )
        return suffix_embeds

    def initialize_suffix_tokens(self) -> torch.Tensor:
        """
        Initialize suffix tokens with BOS tokens (discrete mode).
        Returns initialized tokens [B, max_suffix_len].
        """
        assert (
            self.mode == "discrete"
        ), "initialize_suffix_tokens only for discrete mode"
        bos_token_id = self._get_bos_token_id()
        suffix_tokens = torch.full(
            (self.batch_size, self.max_suffix_len),
            bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        if self.init_len > 0:
            num_active = self.batch_size * self.init_len
            random_tokens = self._sample_random_tokens(num_active).view(
                self.batch_size, self.init_len
            )
            suffix_tokens[:, : self.init_len] = random_tokens
        return suffix_tokens

    def _compute_embeddings(self):
        """Compute embeddings for prefix, completion, and suffix (continuous mode only)."""
        assert self.mode in [
            "continuous",
            "continuous_proj",
        ], "Embeddings only computed for continuous modes"

        # Compute prefix embeddings
        self.prefix_embeddings = self.embedding_layer(
            self.prefix_input_ids
        )  # [B, max_prefix_len, D]

        # Compute completion embeddings
        self.completion_embeddings = self.embedding_layer(
            self.completion_input_ids
        )  # [B, max_completion_len, D]

        # Compute suffix embeddings from suffix_input_ids
        self.suffix_embeddings = self.embedding_layer(
            self.suffix_input_ids
        )  # [B, max_suffix_len, D]

    def add_suffix_token(self, batch_indices: torch.Tensor):
        """
        Add a suffix token by setting the next inactive position's attention mask to 1.

        Args:
            batch_indices: Tensor of batch indices [N] where to add tokens
        """
        for idx in batch_indices:
            # Find first 0 in attention mask
            mask = self.suffix_attention_mask[idx]
            first_zero = (mask == 0).nonzero(as_tuple=True)[0]
            if len(first_zero) > 0:
                pos = first_zero[0].item()
                self.suffix_attention_mask[idx, pos] = 1
        self._invalidate_cached_positions()

    def remove_suffix_token(self, batch_indices: torch.Tensor):
        """
        Remove a suffix token by setting the last active position's attention mask to 0.

        Args:
            batch_indices: Tensor of batch indices [N] where to remove tokens
        """
        for idx in batch_indices:
            # Find last 1 in attention mask
            mask = self.suffix_attention_mask[idx]
            last_one = (mask == 1).nonzero(as_tuple=True)[0]
            if len(last_one) > 0:
                pos = last_one[-1].item()
                self.suffix_attention_mask[idx, pos] = 0
        self._invalidate_cached_positions()

    def update_suffix_embeddings(self, suffix_embeds: torch.Tensor):
        """
        Update suffix embeddings (continuous mode only).
        Initializes new positions (where attention mask is 1 but embedding is zero) with BOS.

        Args:
            suffix_embeds: New suffix embeddings [B, max_suffix_len, D]
        """
        assert self.mode in [
            "continuous",
            "continuous_proj",
        ], "update_suffix_embeddings only for continuous modes"
        assert suffix_embeds.shape == (
            self.batch_size,
            self.max_suffix_len,
            self.embedding_layer.weight.shape[1],
        ), f"Expected shape {(self.batch_size, self.max_suffix_len, self.embedding_layer.weight.shape[1])}, got {suffix_embeds.shape}"

        # Initialize new positions (where attention mask is 1 but embedding is zero) with BOS
        # Do this before storing the reference to avoid in-place operations on the computation graph
        bos_embed = self.get_bos_embedding()
        with torch.no_grad():
            for i in range(self.batch_size):
                for j in range(self.max_suffix_len):
                    if self.suffix_attention_mask[i, j] == 1:
                        # Check if embedding is zero (newly added position)
                        if suffix_embeds[i, j].abs().sum().item() < 1e-6:
                            # Initialize with BOS embedding (in-place on the parameter)
                            suffix_embeds.data[i, j] = bos_embed.clone()

        # Store reference to the embeddings (this is the parameter tensor)
        # Note: We store the reference so get_model_input_embeds_and_attention_mask can use it
        # The parameter tensor itself is fine to reuse across forward/backward passes
        self.suffix_embeddings = suffix_embeds
        self._invalidate_cached_positions()

    def update_suffix_tokens(self, suffix_tokens: torch.Tensor):
        """
        Update suffix tokens (discrete mode only).
        Only initializes positions that are newly active (were 0 in old mask, are 1 in new mask) with BOS.
        Does not overwrite tokens that are already set (non-zero).

        Args:
            suffix_tokens: New suffix token IDs [B, max_suffix_len]
        """
        assert self.mode == "discrete", "update_suffix_tokens only for discrete mode"
        assert suffix_tokens.shape == (
            self.batch_size,
            self.max_suffix_len,
        ), f"Expected shape {(self.batch_size, self.max_suffix_len)}, got {suffix_tokens.shape}"

        # Only initialize positions that are newly active (attention mask is 1) but token is still zero
        # This handles the case where a new position was just added but hasn't been set yet
        # We don't want to overwrite tokens that were already set by apply_length_action
        bos_token_id = self._get_bos_token_id()
        for i in range(self.batch_size):
            for j in range(self.max_suffix_len):
                # Only set BOS if: position is active (mask=1) AND token is zero (not yet set)
                # This means it's a newly added position that hasn't been initialized yet
                if (
                    self.suffix_attention_mask[i, j] == 1
                    and suffix_tokens[i, j].item() == 0
                ):
                    suffix_tokens[i, j] = bos_token_id

        self.suffix_input_ids = suffix_tokens
        self._invalidate_cached_positions()

    def get_model_input_embeds_and_attention_mask(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get concatenated embeddings and attention mask for model input (continuous mode only).

        Returns:
            inputs_embeds: Concatenated embeddings [B, seq_len, D]
            attention_mask: Concatenated attention mask [B, seq_len]
            suffix_mask: Mask indicating which positions are suffix (for gradient computation) [B, seq_len]
        """
        assert self.mode in [
            "continuous",
            "continuous_proj",
        ], "get_model_input_embeds_and_attention_mask only for continuous modes"
        assert (
            self.prefix_embeddings is not None
        ), "Embeddings not computed (should be computed at init for continuous mode)"
        assert (
            self.suffix_embeddings is not None
        ), "Suffix embeddings must be set via update_suffix_embeddings"

        prefix_mask = self.prefix_attention_mask
        completion_mask = self.completion_attention_mask

        prefix_lengths = prefix_mask.sum(dim=1)
        suffix_lengths = self.suffix_attention_mask.sum(dim=1)
        completion_lengths = completion_mask.sum(dim=1)
        total_lengths = prefix_lengths + suffix_lengths + completion_lengths
        max_len = int(total_lengths.max().item())
        batch_size = self.batch_size
        embed_dim = self.embedding_layer.weight.shape[1]

        inputs_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=self.prefix_embeddings.dtype,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        suffix_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        suffix_start_pos = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        completion_start_pos = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )

        for idx in range(batch_size):
            pos = 0
            prefix_len = int(prefix_lengths[idx].item())
            if prefix_len > 0:
                prefix_tokens = self.prefix_embeddings[idx][
                    prefix_mask[idx].bool()
                ].detach()
                inputs_embeds[idx, pos : pos + prefix_len] = prefix_tokens
                attention_mask[idx, pos : pos + prefix_len] = 1
                pos += prefix_len

            suffix_start_pos[idx] = pos
            suffix_len = int(suffix_lengths[idx].item())
            if suffix_len > 0:
                suffix_embeds = self.suffix_embeddings[idx, :suffix_len]
                inputs_embeds[idx, pos : pos + suffix_len] = suffix_embeds
                attention_mask[idx, pos : pos + suffix_len] = 1
                suffix_mask[idx, pos : pos + suffix_len] = 1
                pos += suffix_len

            completion_start_pos[idx] = pos
            completion_len = int(completion_lengths[idx].item())
            if completion_len > 0:
                completion_tokens = self.completion_embeddings[idx][
                    completion_mask[idx].bool()
                ].detach()
                inputs_embeds[idx, pos : pos + completion_len] = completion_tokens
                attention_mask[idx, pos : pos + completion_len] = 1

        self._cached_suffix_start_pos = suffix_start_pos
        self._cached_completion_start_pos = completion_start_pos
        self._cached_total_lengths = total_lengths

        return inputs_embeds, attention_mask, suffix_mask

    def _build_compact_token_inputs(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        pad_value: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Pack prefix, active suffix, and completion tokens into a contiguous sequence."""
        device = prefix_ids.device
        batch_size = prefix_ids.size(0)
        prefix_lengths = prefix_mask.sum(dim=1)
        suffix_lengths = suffix_mask.sum(dim=1)
        completion_lengths = completion_mask.sum(dim=1)
        total_lengths = prefix_lengths + suffix_lengths + completion_lengths
        max_len = int(total_lengths.max().item()) if batch_size > 0 else 0

        if max_len == 0:
            input_ids = torch.empty(
                batch_size, 0, dtype=torch.long, device=device
            )
            attention_mask = torch.empty(
                batch_size, 0, dtype=torch.long, device=device
            )
            suffix_start = torch.zeros(batch_size, dtype=torch.long, device=device)
            completion_start = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )
            return (
                input_ids,
                attention_mask,
                suffix_start,
                completion_start,
                total_lengths,
            )

        input_ids = torch.full(
            (batch_size, max_len),
            pad_value,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=device
        )
        suffix_start_pos = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        completion_start_pos = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

        for idx in range(batch_size):
            pos = 0
            prefix_len = int(prefix_lengths[idx].item())
            if prefix_len > 0:
                prefix_tokens = prefix_ids[idx][prefix_mask[idx].bool()]
                input_ids[idx, pos : pos + prefix_len] = prefix_tokens
                attention_mask[idx, pos : pos + prefix_len] = 1
                pos += prefix_len

            suffix_start_pos[idx] = pos
            suffix_len = int(suffix_lengths[idx].item())
            if suffix_len > 0:
                suffix_tokens = suffix_ids[idx][suffix_mask[idx].bool()]
                input_ids[idx, pos : pos + suffix_len] = suffix_tokens
                attention_mask[idx, pos : pos + suffix_len] = 1
                pos += suffix_len

            completion_start_pos[idx] = pos
            completion_len = int(completion_lengths[idx].item())
            if completion_len > 0:
                completion_tokens = completion_ids[idx][
                    completion_mask[idx].bool()
                ]
                input_ids[idx, pos : pos + completion_len] = completion_tokens
                attention_mask[idx, pos : pos + completion_len] = 1

        return (
            input_ids,
            attention_mask,
            suffix_start_pos,
            completion_start_pos,
            total_lengths,
        )

    def get_model_input_ids_and_attention_mask(
        self,
        suffix_tokens_override: Optional[torch.Tensor] = None,
        completion_tokens_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get concatenated input IDs and attention mask for model input (discrete mode only).
        Also used by continuous_proj mode after projecting suffix embeddings to tokens.

        Returns:
            input_ids: Concatenated token IDs [B, seq_len]
            attention_mask: Concatenated attention mask [B, seq_len]
        """
        assert self.mode == "discrete" or self.original_mode == "continuous_proj"

        suffix_ids = (
            suffix_tokens_override
            if suffix_tokens_override is not None
            else self.suffix_input_ids
        )
        completion_ids = (
            completion_tokens_override
            if completion_tokens_override is not None
            else self.completion_input_ids
        )

        (
            input_ids,
            attention_mask,
            suffix_start,
            completion_start,
            total_lengths,
        ) = self._build_compact_token_inputs(
            self.prefix_input_ids,
            self.prefix_attention_mask,
            suffix_ids,
            self.suffix_attention_mask,
            completion_ids,
            self.completion_attention_mask,
            self.pad_id,
        )

        if suffix_tokens_override is None and completion_tokens_override is None:
            self._cached_suffix_start_pos = suffix_start
            self._cached_completion_start_pos = completion_start
            self._cached_total_lengths = total_lengths

        return input_ids, attention_mask

    def get_suffix_mask_in_fully_batched_input(self) -> torch.Tensor:
        """
        Get suffix mask (not the attention mask, but the mask of the suffix positions) in the fully batched input.
        This has the shape of the concatenated prefix, suffix and completion.
        """
        return torch.cat(
            [
                torch.zeros_like(self.prefix_attention_mask),
                self.suffix_attention_mask,  # This is the mask of the suffix positions
                torch.zeros_like(self.completion_attention_mask),
            ],
            dim=1,
        )

    def get_suffix_start_pos(self) -> torch.Tensor:
        """
        Get suffix start positions (per example) in the compact input.
        """
        if self._cached_suffix_start_pos is None:
            self.get_model_input_ids_and_attention_mask()
        return self._cached_suffix_start_pos

    def get_completion_start_pos(self) -> torch.Tensor:
        """
        Get completion start positions (per example) in the compact input.
        """
        if self._cached_completion_start_pos is None:
            self.get_model_input_ids_and_attention_mask()
        return self._cached_completion_start_pos
