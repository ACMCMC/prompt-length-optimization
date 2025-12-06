"""
Discrete prompt optimization using the official GCG (Greedy Coordinate Gradient) algorithm.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm
from ..interface import BasePromptOptimizer
from ..model_inputs import ModelBatchedInput
import logging


logger = logging.getLogger(__name__)


def _get_embedding_layer(model):
    """Return the model input embedding layer."""
    return model.get_input_embeddings()


def _get_embedding_matrix(model):
    """Return the embedding weight matrix."""
    return _get_embedding_layer(model).weight


def _get_embeddings(model, input_ids):
    """Lookup embeddings for given token ids."""
    return _get_embedding_layer(model)(input_ids)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Compute gradients of the loss w.r.t. the coordinates (control tokens).
    """
    model_device = next(model.parameters()).device
    embed_weights = _get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model_device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model_device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    embeds = _get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, : input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop :, :],
        ],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = F.cross_entropy(logits[0, loss_slice, :], targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    return grad


def sample_control(
    control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None
):
    """
    Sample candidate control sequences following the official GCG sampling rule.
    """
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float("inf")

    topk = min(topk, grad.shape[1])
    top_indices = (-grad).topk(topk, dim=1).indices  # [L, topk]
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    pos_choices = torch.randint(0, len(control_toks), (batch_size,), device=grad.device)
    tok_choices = torch.randint(0, topk, (batch_size,), device=grad.device)
    new_token_val = top_indices[pos_choices, tok_choices].unsqueeze(-1)
    new_token_pos = pos_choices.unsqueeze(-1)
    new_control_toks = original_control_toks.scatter_(1, new_token_pos, new_token_val)
    return new_control_toks

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
        self.not_allowed_tokens = (
            torch.tensor(
                sorted(agent.special_token_ids),
                dtype=torch.long,
                device=self.device,
            )
            if agent.special_token_ids
            else None
        )
        self.last_first_token_prob = None
        self.latest_metrics: dict = {}

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
    
    def _trim_prefix_tokens(
        self, model_input: ModelBatchedInput, idx: int
    ) -> torch.Tensor:
        """Return prefix tokens (without padding) for a single example."""
        mask = model_input.prefix_attention_mask[idx].bool()
        tokens = model_input.prefix_input_ids[idx]
        return tokens[mask]

    def _trim_completion_tokens(
        self, model_input: ModelBatchedInput, idx: int
    ) -> torch.Tensor:
        """Return completion tokens (without padding) for a single example."""
        mask = model_input.completion_attention_mask[idx].bool()
        tokens = model_input.completion_input_ids[idx]
        return tokens[mask]

    def _compute_prompt_likelihoods(
        self, model_input: ModelBatchedInput
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the language model on the current batched inputs and return:
        - Total log-likelihood per prompt
        - Per-token log probabilities for completion tokens
        - Completion token mask
        """
        input_ids, attention_mask = model_input.get_model_input_ids_and_attention_mask()
        with torch.no_grad():
            outputs = self.agent.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        completion_start_pos = model_input.get_completion_start_pos()
        completion_tokens = model_input.completion_input_ids
        completion_mask = model_input.completion_attention_mask.bool()
        token_log_probs, comp_mask = self._completion_token_log_probs_from_logits(
            logits, completion_start_pos, completion_tokens, completion_mask
        )
        # Alignment check: ensure the first completion token prob matches manual extraction
        first_mask = comp_mask[:, 0]
        helper_probs = token_log_probs[first_mask, 0]
        batch_indices = torch.nonzero(first_mask, as_tuple=False).squeeze(-1)
        manual_probs = []
        for idx in batch_indices:
            start = int(completion_start_pos[idx].item())
            slice_logits = logits[idx, start - 1, :]
            log_probs = F.log_softmax(slice_logits, dim=-1)
            token_id = completion_tokens[idx][completion_mask[idx]][0]
            manual_probs.append(log_probs[token_id].exp().item())
        manual_probs_tensor = torch.tensor(
            manual_probs, device=self.device, dtype=helper_probs.dtype
        )
        diff = torch.max(torch.abs(manual_probs_tensor - helper_probs.exp()))
        if diff > 1e-5:
            logger.warning(
                "First-token prob mismatch detected (max diff=%.6f). Manual=%s Helper=%s",
                diff.item(),
                manual_probs_tensor.tolist(),
                helper_probs.exp().tolist(),
        )
        masked_log_probs = torch.where(
            comp_mask, token_log_probs, torch.zeros_like(token_log_probs)
        )
        likelihoods = masked_log_probs.sum(dim=-1)
        avg_per_token_logprob = masked_log_probs.sum() / comp_mask.sum().clamp(min=1)
        return likelihoods, token_log_probs, comp_mask, avg_per_token_logprob

    def _test_candidates_batch(
        self,
        candidate_sequences: torch.Tensor,
        candidate_prompt_indices: torch.Tensor,
        model_input: ModelBatchedInput,
    ) -> torch.Tensor:
        """
        Test candidate sequences in batches and return losses.

        Args:
            candidate_sequences: Full candidate sequences [search_width * B, max_suffix_len]
            candidate_prompt_indices: Prompt index for each candidate [total_candidates]
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
        prompt_indices_tensor = candidate_prompt_indices.to(self.device)

        # Process in chunks to respect max_batch_size
        for chunk_start in range(0, total_candidates, self.gcg_max_batch_size):
            chunk_end = min(chunk_start + self.gcg_max_batch_size, total_candidates)
            chunk_sequences = candidate_sequences[
                chunk_start:chunk_end
            ]  # [chunk_size, max_suffix_len]
            chunk_size = chunk_end - chunk_start
            prompt_indices = prompt_indices_tensor[chunk_start:chunk_end]
            # Build candidate input_ids by replacing suffix portion
            candidate_input_ids = base_input_ids[
                prompt_indices
            ].clone()  # [chunk_size, seq_len]
            candidate_attention_mask = base_attention_mask[
                prompt_indices
            ].clone()  # [chunk_size, seq_len]

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

        batch_size = prompt_data.shape[0]
        suffix_mask = model_input.suffix_attention_mask

        # Track best likelihoods and tokens
        model_input.update_suffix_tokens(prompt_data)
        best_lls, _, _, _ = self._compute_prompt_likelihoods(model_input)
        initial_lls = best_lls.clone()  # For monotonicity check at the end
        best_tokens = prompt_data.clone()
        accumulated_ll_delta = torch.zeros(1, device=self.device)

        self.last_first_token_prob = None
        self.last_avg_token_logprob = None
        gcg_iter_bar = tqdm(range(self.gcg_steps), desc="GCG", leave=False)
        self.latest_metrics = {}  # reset metrics for this run
        for gcg_iter in gcg_iter_bar:
            # Ensure latest tokens are reflected in model_input before scoring candidates
            model_input.update_suffix_tokens(best_tokens)

            candidate_sequences = []
            candidate_prompt_indices = []
            active_prompt_indices = []

            for prompt_idx in range(batch_size):
                length = int(lengths[prompt_idx].item())
                completion_len = int(
                    model_input.completion_attention_mask[prompt_idx].sum().item()
                )
                if length == 0 or completion_len == 0:
                    continue

                prefix_tokens = self._trim_prefix_tokens(model_input, prompt_idx)
                completion_tokens = self._trim_completion_tokens(model_input, prompt_idx)
                control_tokens = best_tokens[prompt_idx, :length].detach()

                if completion_tokens.numel() == 0:
                    continue

                input_ids = torch.cat(
                    [prefix_tokens, control_tokens, completion_tokens], dim=0
                )
                pref_len = prefix_tokens.shape[0]
                control_slice = slice(pref_len, pref_len + length)
                target_slice = slice(
                    pref_len + length, pref_len + length + completion_tokens.shape[0]
                )
                loss_slice = slice(
                    pref_len + length - 1,
                    pref_len + length - 1 + completion_tokens.shape[0],
                )

                self.agent.model.zero_grad(set_to_none=True)
                grad = token_gradients(
                    self.agent.model, input_ids, control_slice, target_slice, loss_slice
                )  # [length, vocab]

                candidates = sample_control(
                    control_tokens,
                    grad,
                    batch_size=self.gcg_batch_size,
                    topk=self.gcg_top_k,
                    temp=1,
                    not_allowed_tokens=self.not_allowed_tokens,
                )

                padded_candidates = best_tokens[prompt_idx : prompt_idx + 1].repeat(
                    self.gcg_batch_size, 1
                )
                padded_candidates[:, :length] = candidates

                candidate_sequences.append(padded_candidates)
                candidate_prompt_indices.extend([prompt_idx] * self.gcg_batch_size)
                active_prompt_indices.append(prompt_idx)

            if len(candidate_sequences) == 0:
                break

            candidate_tensor = torch.cat(candidate_sequences, dim=0)
            candidate_prompt_indices_tensor = torch.tensor(
                candidate_prompt_indices, dtype=torch.long, device=self.device
            )

            losses = self._test_candidates_batch(
                candidate_tensor, candidate_prompt_indices_tensor, model_input
            )

            search_width = self.gcg_batch_size
            num_active = len(active_prompt_indices)
            losses_reshaped = losses.view(num_active, search_width)
            best_indices = losses_reshaped.argmin(dim=1)

            old_tokens = best_tokens.clone()
            old_best_lls = best_lls.clone()

            for block_idx, prompt_idx in enumerate(active_prompt_indices):
                best_candidate_idx = (
                    block_idx * search_width + best_indices[block_idx].item()
                )
                best_tokens[prompt_idx] = candidate_tensor[best_candidate_idx]

            model_input.update_suffix_tokens(best_tokens)
            (
                current_lls,
                token_log_probs,
                comp_mask,
                avg_log_prob_batch,
            ) = self._compute_prompt_likelihoods(model_input)

            # Debug metrics: average per-token log prob and first-token probability (in [0, 1])
            avg_log_prob = float("nan")
            total_tokens = comp_mask.sum().item()
            if total_tokens > 0:
                valid_token_log_probs = token_log_probs[comp_mask]
                avg_log_prob = valid_token_log_probs.mean().item()

            avg_first_prob = float("nan")
            if comp_mask.shape[1] > 0:
                first_mask = comp_mask[:, 0]
                if first_mask.any():
                    first_probs = token_log_probs[first_mask, 0].exp()
                    avg_first_prob = first_probs.mean().item()

            self.last_first_token_prob = avg_first_prob
            self.last_avg_token_logprob = avg_log_prob_batch.item()
            logger.info(
                "GCG iter %d: avg per-token logprob=%.4f, avg first completion prob=%.4f",
                gcg_iter + 1,
                self.last_avg_token_logprob,
                avg_first_prob,
            )
            self.latest_metrics["avg_first_completion_prob"] = avg_first_prob

            improve_mask = current_lls > best_lls
            if improve_mask.any():
                best_lls = torch.where(improve_mask, current_lls, best_lls)
            best_tokens = torch.where(
                improve_mask.view(-1, 1), best_tokens, old_tokens
            )
            model_input.update_suffix_tokens(best_tokens)

            step_delta = torch.where(
                improve_mask, current_lls - old_best_lls, torch.zeros_like(current_lls)
            )
            accumulated_ll_delta += step_delta.sum()

            for block_idx, prompt_idx in enumerate(active_prompt_indices):
                if not improve_mask[prompt_idx]:
                    continue
                changed_positions = (
                    (old_tokens[prompt_idx] != best_tokens[prompt_idx])
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                if changed_positions.ndim == 0:
                    changed_positions = changed_positions.unsqueeze(0)
                for pos_tensor in changed_positions:
                    pos = (
                        pos_tensor.item() if torch.is_tensor(pos_tensor) else pos_tensor
                    )
                    old_token = old_tokens[prompt_idx, pos].item()
                    new_token = best_tokens[prompt_idx, pos].item()
                    old_token_str = self.agent.tokenizer.decode(
                        [old_token], skip_special_tokens=True
                    )
                    new_token_str = self.agent.tokenizer.decode(
                        [new_token], skip_special_tokens=True
                    )
                    ll_change = (
                        current_lls[prompt_idx].item()
                        - old_best_lls[prompt_idx].item()
                    )
                    logger.debug(
                        f"GCG iter {gcg_iter+1} prompt {prompt_idx+1}: pos {pos} '{old_token_str}' -> '{new_token_str}', Δll={ll_change:.4f}"
                    )

            gcg_iter_bar.set_postfix({"avg_ll": f"{best_lls.mean().item():.4f}"})

        # Final update
        prompt_data = best_tokens.clone()
        inactive_mask = (suffix_mask == 0).bool()
        if inactive_mask.any():
            # Ensure masked-out suffix positions always hold BOS-equivalent tokens
            bos_token_id = self.agent.tokenizer.bos_token_id
            if bos_token_id is None:
                bos_token_id = self.agent.tokenizer.pad_token_id
            if bos_token_id is None:
                bos_token_id = 0
            prompt_data[inactive_mask] = bos_token_id

        model_input.update_suffix_tokens(prompt_data)
        final_likelihoods, _, _, _ = self._compute_prompt_likelihoods(model_input)

        # Monotonicity check: warn if final likelihoods are worse than initial
        # This should not typically happen if GCG is behaving as a greedy ascent step.
        ll_deltas = final_likelihoods - initial_lls
        total_final_delta = ll_deltas.sum()
        if not torch.allclose(
            accumulated_ll_delta, total_final_delta, atol=1e-6, rtol=1e-6
        ):
            raise AssertionError(
                f"Accumulated LL delta {accumulated_ll_delta.item():.6f} "
                f"does not match final improvement {total_final_delta.item():.6f}"
            )
        if (ll_deltas < -1e-6).any():
            num_decreased = (ll_deltas < 0).sum().item()
            min_delta = ll_deltas.min().item()
            logger.warning(
                f"Warning: GCG decreased likelihood for {num_decreased} prompts in this step (min Δll={min_delta:.4f})."
            )
            
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
