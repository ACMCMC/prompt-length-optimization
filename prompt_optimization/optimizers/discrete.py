"""
Discrete prompt optimization using GCG (Greedy Coordinate Gradient) algorithm.
Based on nanoGCG implementation: https://github.com/GraySwanAI/nanoGCG
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from ..interface import BasePromptOptimizer
from ..model_inputs import ModelBatchedInput
import logging


logging.basicConfig(level=logging.INFO)

class DiscretePromptOptimizer(BasePromptOptimizer):
    """
    GCG-based discrete prompt optimization.

    Uses gradient-based candidate sampling: computes gradients w.r.t. token embeddings,
    selects top-k tokens based on gradients, then tests candidates in batches.
    """

    def __init__(
        self,
        agent,
        initial_prompt_length: int,
        max_prompt_len: int,
        batch_size: int,
        lr_embeddings: float,
        max_suffix_len: int,
        init_len: int,
        gcg_steps: int,
        gcg_top_k: int,
        gcg_batch_size: int,
        gcg_max_batch_size: int,
    ):
        super().__init__(
            agent,
            initial_prompt_length,
            max_prompt_len,
            batch_size,
            lr_embeddings,
            max_suffix_len,
            init_len,
        )
        self.embedding_layer = agent.model.get_input_embeddings()
        self.gcg_steps = gcg_steps  # GCG iterations per optimization step
        self.gcg_top_k = gcg_top_k  # Top-k tokens to sample from based on gradients
        self.gcg_batch_size = (
            gcg_batch_size  # Number of candidates to test per position
        )
        self.gcg_max_batch_size = (
            gcg_max_batch_size  # Maximum batch size for forward passes
        )

    def initialize_prompts(
        self, model_input: ModelBatchedInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS tokens from ModelBatchedInput."""
        lengths = torch.full(
            (self.batch_size,),
            self.initial_prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        prompt_tokens = model_input.initialize_suffix_tokens()
        return prompt_tokens, lengths

    def get_likelihoods(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        model_input: ModelBatchedInput,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute likelihoods from tokens using ModelBatchedInput."""
        model_input.update_suffix_tokens(prompt_data)
        return self.agent.get_likelihoods_batch(
            model_input, requires_grad=requires_grad
        )
    
    def apply_length_action(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens."""
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_prompt_len)

        updated_lengths = lengths - remove_mask.long() + add_mask.long()

        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            add_positions = lengths[add_indices].long()  # Convert to int for indexing
            new_tokens = torch.tensor(
                [self.agent.get_random_token() for _ in range(len(add_indices))],
                dtype=torch.long,
                device=self.device,
            )
            prompt_data[add_indices, add_positions] = new_tokens

        return prompt_data, updated_lengths
    
    def _compute_gradients(
        self,
        best_tokens: torch.Tensor,
        model_input: ModelBatchedInput,
    ) -> torch.Tensor:
        """
        Compute gradients of loss w.r.t. token embeddings for active suffix positions.

        Args:
            best_tokens: Current suffix tokens [B, max_suffix_len]
            model_input: ModelBatchedInput instance
            suffix_mask: Active positions mask [B, max_suffix_len] (1=active, 0=inactive)

        Returns:
            gradients: Gradients w.r.t. vocab logits [B, max_suffix_len, vocab_size]
        """
        batch_size = best_tokens.shape[0]
        vocab_size = self.embedding_layer.weight.shape[0]
        embed_dim = self.embedding_layer.weight.shape[1]
        dtype = self.embedding_layer.weight.dtype

        suffix_one_hot = torch.zeros(
            batch_size,
            self.max_suffix_len,
            vocab_size,
            device=self.device,
            dtype=dtype,
        )
        suffix_one_hot.scatter_(2, best_tokens.unsqueeze(-1), 1.0)
        suffix_one_hot.requires_grad_(True)
        suffix_one_hot.retain_grad()

        prefix_mask = model_input.prefix_attention_mask.bool()
        suffix_mask = model_input.suffix_attention_mask.bool()
        completion_mask = model_input.completion_attention_mask.bool()
        prefix_lengths = prefix_mask.sum(dim=1)
        suffix_lengths = suffix_mask.sum(dim=1)
        completion_lengths = completion_mask.sum(dim=1)
        total_lengths = prefix_lengths + suffix_lengths + completion_lengths
        max_len = int(total_lengths.max().item()) if batch_size > 0 else 0

        inputs_embeds = torch.zeros(
            (batch_size, max_len, embed_dim),
            dtype=dtype,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long, device=self.device
        )
        completion_start_pos = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )

        prefix_embeds = self.embedding_layer(model_input.prefix_input_ids).detach()
        completion_embeds = self.embedding_layer(
            model_input.completion_input_ids
        ).detach()

        for idx in range(batch_size):
            pos = 0
            prefix_len = int(prefix_lengths[idx].item())
            if prefix_len > 0:
                prefix_segment = prefix_embeds[idx][prefix_mask[idx]]
                inputs_embeds[idx, pos : pos + prefix_len] = prefix_segment
                attention_mask[idx, pos : pos + prefix_len] = 1
                pos += prefix_len

            suffix_len = int(suffix_lengths[idx].item())
            if suffix_len > 0:
                suffix_segment = torch.matmul(
                    suffix_one_hot[idx, :suffix_len], self.embedding_layer.weight
                )
                inputs_embeds[idx, pos : pos + suffix_len] = suffix_segment
                attention_mask[idx, pos : pos + suffix_len] = 1
                pos += suffix_len

            completion_start_pos[idx] = pos
            completion_len = int(completion_lengths[idx].item())
            if completion_len > 0:
                completion_segment = completion_embeds[idx][completion_mask[idx]]
                inputs_embeds[idx, pos : pos + completion_len] = completion_segment
                attention_mask[idx, pos : pos + completion_len] = 1

        position_ids = ModelBatchedInput._compute_position_ids(attention_mask)

        outputs = self.agent.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = outputs.logits  # [B, seq_len, vocab]

        token_log_probs, comp_mask = self._completion_token_log_probs_from_logits(
            logits,
            completion_start_pos,
            model_input.completion_input_ids,
            completion_mask,
        )
        masked_log_probs = torch.where(
            comp_mask, token_log_probs, torch.zeros_like(token_log_probs)
        )
        loss = -masked_log_probs.sum(dim=-1).mean()

        # Backward to get gradients
        loss.backward()
        gradients = suffix_one_hot.grad.clone()  # [B, max_suffix_len, vocab_size]

        # Clear grad reference on one-hot tensor
        suffix_one_hot.grad = None

        # Zero-out gradients for inactive suffix positions (attention mask = 0)
        inactive_mask = (model_input.suffix_attention_mask == 0).unsqueeze(-1)
        if inactive_mask.any():
            gradients = gradients.masked_fill(inactive_mask, 0.0)

        # Normalize gradients per position (avoid divide-by-zero)
        grad_norm = gradients.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        gradients = gradients / grad_norm

        # Clear gradients and intermediate tensors to free memory
        del (loss, outputs, logits, inputs_embeds, attention_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gradients

    def _sample_candidates_from_grad(
        self,
        prompt_data: torch.Tensor,
        gradients: torch.Tensor,
        suffix_mask: torch.Tensor,
        n_replace: int = 1,
    ) -> Tuple[torch.Tensor, list]:
        """
        Sample candidate sequences from gradients (following nanoGCG's sample_ids_from_grad).

        Generates search_width total candidate sequences (not per position).
        Each candidate updates n_replace randomly selected positions.

        Args:
            prompt_data: Current suffix tokens [B, max_suffix_len]
            gradients: Gradients w.r.t. vocab logits [B, max_suffix_len, vocab_size]
            suffix_mask: Active positions mask [B, max_suffix_len]
            n_replace: Number of positions to update per candidate (default 1)

        Returns:
            candidate_sequences: Full candidate sequences [search_width * B, max_suffix_len]
            update_info: List of (prompt_idx, candidate_idx, pos, new_token) for each update
        """
        # Get top-k tokens for each position: [B, max_suffix_len, top_k]
        topk_ids = (
            (-gradients).topk(self.gcg_top_k, dim=-1).indices
        )  # [B, max_suffix_len, top_k]

        search_width = (
            self.gcg_batch_size
        )  # Total number of candidate sequences per prompt

        # For each prompt, generate search_width candidate sequences
        all_candidate_sequences = []
        all_update_info = []

        actual_batch_size = prompt_data.shape[0]
        for prompt_idx in range(actual_batch_size):
            active_positions = torch.nonzero(
                suffix_mask[prompt_idx] == 1, as_tuple=False
            ).squeeze(-1)

            if active_positions.numel() == 0:
                continue

            num_active = active_positions.shape[0]

            # Start from current sequence
            original_sequence = prompt_data[prompt_idx : prompt_idx + 1]
            candidate_sequences = original_sequence.repeat(search_width, 1)

            # Deterministically spaced positions (similar to reference implementation)
            if search_width == 1:
                position_indices = torch.zeros(
                    1, dtype=torch.long, device=self.device
                )
            else:
                position_indices = torch.linspace(
                    0,
                    num_active - 1,
                    steps=search_width,
                    device=self.device,
                    dtype=torch.long,
                )
            selected_positions = active_positions[position_indices]  # [search_width]

            # For each candidate, sample replacement token from top-k of that position
            topk_for_prompt = topk_ids[prompt_idx]  # [max_suffix_len, top_k]
            random_indices = torch.randint(
                0, self.gcg_top_k, (search_width, 1), device=self.device
            )
            candidate_tokens = torch.gather(
                topk_for_prompt[selected_positions], 1, random_indices
            ).squeeze(-1)

            candidate_sequences.scatter_(
                1, selected_positions.unsqueeze(-1), candidate_tokens.unsqueeze(-1)
            )

            all_candidate_sequences.append(candidate_sequences)

            for cand_idx in range(search_width):
                pos = selected_positions[cand_idx].item()
                new_token = candidate_tokens[cand_idx].item()
                all_update_info.append((prompt_idx, cand_idx, pos, new_token))

        if len(all_candidate_sequences) == 0:
            return (
                torch.empty(
                    0, self.max_suffix_len, dtype=torch.long, device=self.device
                ),
                [],
            )

        candidate_sequences_tensor = torch.cat(
            all_candidate_sequences, dim=0
        )  # [search_width * num_prompts, max_suffix_len]
        return candidate_sequences_tensor, all_update_info

    def _test_candidates_batch(
        self,
        candidate_sequences: torch.Tensor,
        update_info: list,
        model_input: ModelBatchedInput,
    ) -> torch.Tensor:
        """
        Test candidate sequences in batches and return losses.

        Args:
            candidate_sequences: Full candidate sequences [search_width * B, max_suffix_len]
            update_info: List of (prompt_idx, candidate_idx, pos, new_token) for each update
            model_input: ModelBatchedInput instance

        Returns:
            losses: Losses for each candidate [search_width * B] (negative log likelihood)
        """
        total_candidates = candidate_sequences.shape[0]
        if total_candidates == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        base_input_ids, base_attention_mask = model_input.get_model_input_ids_and_attention_mask()
        suffix_start_all = model_input.get_suffix_start_pos()
        suffix_lengths_all = model_input.suffix_attention_mask.sum(dim=1).long()
        completion_start_all = model_input.get_completion_start_pos()
        completion_mask_all = model_input.completion_attention_mask.bool()

        all_losses = []

        # Process in chunks to respect max_batch_size
        for chunk_start in range(0, total_candidates, self.gcg_max_batch_size):
            chunk_end = min(chunk_start + self.gcg_max_batch_size, total_candidates)
            chunk_sequences = candidate_sequences[
                chunk_start:chunk_end
            ]  # [chunk_size, max_suffix_len]
            chunk_size = chunk_end - chunk_start

            # Get prompt indices for this chunk (each candidate corresponds to a prompt)
            # update_info format: (prompt_idx, candidate_idx, pos, new_token)
            # We need to map candidate_idx to the actual index in candidate_sequences
            # candidate_sequences is organized as: [prompt0_cand0, prompt0_cand1, ..., prompt1_cand0, ...]
            # So for candidate at index i, prompt_idx = i // search_width, candidate_idx = i % search_width
            search_width = self.gcg_batch_size
            prompt_indices = (
                torch.arange(chunk_start, chunk_end, device=self.device) // search_width
            ).long()

            # Build candidate input_ids by replacing suffix portion
            candidate_input_ids = base_input_ids[
                prompt_indices
            ].clone()  # [chunk_size, seq_len]
            candidate_attention_mask = base_attention_mask[
                prompt_indices
            ].clone()  # [chunk_size, seq_len]
            candidate_position_ids = ModelBatchedInput._compute_position_ids(
                candidate_attention_mask
            )

            suffix_start_batch = suffix_start_all[prompt_indices]
            suffix_lengths_batch = suffix_lengths_all[prompt_indices]
            for row in range(chunk_size):
                length = int(suffix_lengths_batch[row].item())
                if length == 0:
                    continue
                start = int(suffix_start_batch[row].item())
                candidate_input_ids[row, start : start + length] = chunk_sequences[
                    row, :length
                ]

            # Get completion data for each candidate
            candidate_completion_input_ids = model_input.completion_input_ids[
                prompt_indices
            ]  # [chunk_size, max_completion_len]

            # Forward pass
            with torch.no_grad():
                inputs_embeds = self.embedding_layer(candidate_input_ids)
                outputs = self.agent.model.gpt_neox(
                    inputs_embeds=inputs_embeds,
                    attention_mask=candidate_attention_mask,
                    position_ids=candidate_position_ids,
                )
                hidden_states = outputs.last_hidden_state
                logits = self.agent.model.embed_out(hidden_states)

                comp_start_batch = completion_start_all[prompt_indices]
                comp_mask_batch = completion_mask_all[prompt_indices]
                token_log_probs, comp_mask = self._completion_token_log_probs_from_logits(
                    logits,
                    comp_start_batch,
                    candidate_completion_input_ids,
                    comp_mask_batch,
                )
                masked_log_probs = torch.where(
                    comp_mask, token_log_probs, torch.zeros_like(token_log_probs)
                )
                token_counts = comp_mask.sum(dim=-1).clamp_min(1)
                losses = -masked_log_probs.sum(dim=-1) / token_counts

            all_losses.append(losses.cpu())  # Move to CPU to free GPU memory

            # Clear intermediate tensors
            del (
                inputs_embeds,
                outputs,
                hidden_states,
                logits,
                token_log_probs,
                masked_log_probs,
                losses,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_losses, dim=0).to(
            self.device
        )  # [total_candidates] (move back to device)

    def inner_optimization_step(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        step: int,
        model_input: ModelBatchedInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GCG optimization step: gradient-based candidate sampling and batched testing.

        Following nanoGCG algorithm:
        1. Compute gradients w.r.t. token embeddings
        2. Sample candidates from top-k tokens based on gradients
        3. Test candidates in batches
        4. Update best tokens
        5. Repeat for gcg_steps iterations
        """
        max_active_len = lengths.max().item()
        if max_active_len == 0:
            model_input.update_suffix_tokens(prompt_data)
            return prompt_data, self.agent.get_likelihoods_batch(
                model_input, requires_grad=False
            )

        # Get suffix mask to identify active positions
        suffix_mask = model_input.suffix_attention_mask

        # Track best likelihoods and tokens
        model_input.update_suffix_tokens(prompt_data)
        best_lls = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        initial_lls = best_lls.clone()  # For monotonicity check at the end
        best_tokens = prompt_data.clone()

        # GCG iterations
        gcg_iter_bar = tqdm(range(self.gcg_steps), desc="GCG", leave=False)
        for gcg_iter in gcg_iter_bar:
            # Step 1: Compute gradients w.r.t. token embeddings
            gradients = self._compute_gradients(
                best_tokens, model_input
            )  # [B, max_suffix_len, emb_dim]

            # Step 2: Sample candidates from gradients
            candidate_sequences, update_info = self._sample_candidates_from_grad(
                best_tokens, gradients, suffix_mask, n_replace=1
            )

            if candidate_sequences.shape[0] == 0:
                break

            # Step 3: Test candidates in batches
            losses = self._test_candidates_batch(
                candidate_sequences, update_info, model_input
            )  # [search_width * B]

            # Step 4: Pick best candidate per prompt
            # candidate_sequences is organized as: [prompt0_cand0, prompt0_cand1, ..., prompt1_cand0, ...]
            # So we reshape losses to [B, search_width] and pick best per prompt
            search_width = self.gcg_batch_size
            num_prompts = losses.shape[0] // search_width
            losses_reshaped = losses.view(
                num_prompts, search_width
            )  # [B, search_width]

            # Get best candidate index for each prompt (min loss = best)
            best_indices = losses_reshaped.argmin(dim=1)  # [B]

            # Update best tokens with best candidate sequences
            # Store old tokens and old likelihoods to track changes for logging
            old_tokens = best_tokens.clone()
            old_best_lls = best_lls.clone()

            for prompt_idx in range(num_prompts):
                best_candidate_idx = (
                    prompt_idx * search_width + best_indices[prompt_idx].item()
                )
                best_tokens[prompt_idx] = candidate_sequences[best_candidate_idx]

            # Recompute likelihoods with updated tokens
            model_input.update_suffix_tokens(best_tokens)
            current_lls = self.agent.get_likelihoods_batch(
                model_input, requires_grad=False
            )

            # Only keep improvements (restore previous tokens for non-improving prompts)
            improve_mask = current_lls > best_lls
            if improve_mask.any():
                best_lls = torch.where(improve_mask, current_lls, best_lls)
            best_tokens = torch.where(
                improve_mask.view(-1, 1), best_tokens, old_tokens
            )
            model_input.update_suffix_tokens(best_tokens)

            # Debug log: token replacements and likelihood changes (reusing computed values)
            for prompt_idx in range(num_prompts):
                if improve_mask[prompt_idx]:
                    # Find which positions changed (reusing already computed tokens)
                    changed_positions = (
                        (old_tokens[prompt_idx] != best_tokens[prompt_idx])
                        .nonzero(as_tuple=False)
                        .squeeze(-1)
                    )
                    if len(changed_positions.shape) == 0:
                        # Single position changed
                        changed_positions = changed_positions.unsqueeze(0)
                    if len(changed_positions) > 0:
                        for pos_tensor in changed_positions:
                            pos = (
                                pos_tensor.item()
                                if torch.is_tensor(pos_tensor)
                                else pos_tensor
                            )
                            old_token = old_tokens[prompt_idx, pos].item()
                            new_token = best_tokens[prompt_idx, pos].item()
                            old_token_str = self.agent.tokenizer.decode(
                                [old_token], skip_special_tokens=True
                            )
                            new_token_str = self.agent.tokenizer.decode(
                                [new_token], skip_special_tokens=True
                            )
                            # Reuse computed likelihoods: change = current_lls - old_best_lls
                            ll_change = (
                                current_lls[prompt_idx].item()
                                - old_best_lls[prompt_idx].item()
                            )
                            logging.debug(
                                f"GCG iter {gcg_iter+1} prompt {prompt_idx+1}: pos {pos} '{old_token_str}' -> '{new_token_str}', Δll={ll_change:.4f}"
                            )

            # Update progress bar
            gcg_iter_bar.set_postfix({"avg_ll": f"{best_lls.mean().item():.4f}"})

        # Final update
        prompt_data = best_tokens.clone()
        model_input.update_suffix_tokens(prompt_data)
        final_likelihoods = self.agent.get_likelihoods_batch(
            model_input, requires_grad=False
        )

        # Monotonicity check: warn if final likelihoods are worse than initial
        # This should not typically happen if GCG is behaving as a greedy ascent step.
        ll_deltas = final_likelihoods - initial_lls
        if (ll_deltas < -1e-6).any():
            num_decreased = (ll_deltas < 0).sum().item()
            min_delta = ll_deltas.min().item()
            logging.warning(
                f"Warning: GCG decreased likelihood for {num_decreased} prompts in this step (min Δll={min_delta:.4f})."
            )
            
        # Sanity check: check that all the tokens in the non-active positions are BOS
        # It's possible that after we increased length, and then decreased it, the non-active positions are not BOS. So just in case if that happens we raise a warning. But we don't fail.
        if not (prompt_data[suffix_mask == 0] == self.agent.tokenizer.bos_token_id).all():
            logging.warning("Warning: Tokens in non-active positions are not BOS")

        return prompt_data, final_likelihoods

    def to_tokens(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Tokens are already token IDs, just pad/trim to max length."""
        B = prompt_data.shape[0]
        max_len = lengths.max().item()

        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)

        tokens = prompt_data[:, :max_len].clone()
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))

        return tokens
    
    def clone_prompt(
        self, prompt_data: torch.Tensor, idx: int, length: int
    ) -> torch.Tensor:
        """Clone a single prompt's tokens."""
        return prompt_data[idx, :length].clone()

    def _completion_token_log_probs_from_logits(
        self,
        logits: torch.Tensor,
        completion_start_pos: torch.Tensor,
        completion_tokens: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather per-token log probabilities for completion tokens given logits.
        Returns (token_log_probs, mask).
        """
        batch_size = logits.size(0)
        if batch_size == 0:
            return logits.new_zeros((0, 0)), completion_mask[:, :0]

        completion_lengths = completion_mask.sum(dim=1)
        max_len = (
            int(completion_lengths.max().item()) if completion_lengths.numel() > 0 else 0
        )
        if max_len == 0:
            return logits.new_zeros((batch_size, 0)), completion_mask[:, :0]

        token_log_probs = logits.new_zeros((batch_size, max_len))
        trimmed_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.bool, device=logits.device
        )

        for idx in range(batch_size):
            length = int(completion_lengths[idx].item())
            if length == 0:
                continue
            start = int(completion_start_pos[idx].item())
            end = start - 1 + length
            slice_logits = logits[idx, start - 1 : end, :]
            log_probs = F.log_softmax(slice_logits, dim=-1)
            tokens = completion_tokens[idx][completion_mask[idx]]
            gathered = log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
            token_log_probs[idx, :length] = gathered
            trimmed_mask[idx, :length] = True

        return token_log_probs, trimmed_mask
