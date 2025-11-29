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


class DiscretePromptOptimizer(BasePromptOptimizer):
    """
    GCG-based discrete prompt optimization.
    
    Uses gradient-based candidate sampling: computes gradients w.r.t. token embeddings,
    selects top-k tokens based on gradients, then tests candidates in batches.
    """
    
    def __init__(self, agent, initial_prompt_length: int, max_prompt_len: int,
                 batch_size: int, lr_embeddings: float, max_suffix_len: int, init_len: int,
                 gcg_steps: int, gcg_top_k: int, gcg_batch_size: int, gcg_max_batch_size: int):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings, max_suffix_len, init_len)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.gcg_steps = gcg_steps  # GCG iterations per optimization step
        self.gcg_top_k = gcg_top_k  # Top-k tokens to sample from based on gradients
        self.gcg_batch_size = gcg_batch_size  # Number of candidates to test per position
        self.gcg_max_batch_size = gcg_max_batch_size  # Maximum batch size for forward passes
    
    def initialize_prompts(self, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS tokens from ModelBatchedInput."""
        lengths = torch.full((self.batch_size,), self.initial_prompt_length,
                           dtype=torch.long, device=self.device)
        prompt_tokens = model_input.initialize_suffix_tokens()
        return prompt_tokens, lengths
    
    def get_likelihoods(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                       model_input: ModelBatchedInput, requires_grad: bool = False) -> torch.Tensor:
        """Compute likelihoods from tokens using ModelBatchedInput."""
        model_input.update_suffix_tokens(prompt_data)
        return self.agent.get_likelihoods_batch(model_input, requires_grad=requires_grad)
    
    def apply_length_action(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                           actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens."""
        remove_mask = (actions == 0) & (lengths > 0)
        add_mask = (actions == 2) & (lengths < self.max_prompt_len)
        
        updated_lengths = lengths - remove_mask.long() + add_mask.long()
        
        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            add_positions = lengths[add_indices].long()  # Convert to int for indexing
            new_tokens = torch.tensor(
                [self.agent.get_random_token() for _ in range(len(add_indices))],
                dtype=torch.long, device=self.device
            )
            prompt_data[add_indices, add_positions] = new_tokens
        
        return prompt_data, updated_lengths
    
    def _compute_gradients(self, prompt_data: torch.Tensor, model_input: ModelBatchedInput,
                          suffix_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of loss w.r.t. token embeddings for active suffix positions.
        
        Args:
            prompt_data: Current suffix tokens [B, max_suffix_len]
            model_input: ModelBatchedInput instance
            suffix_mask: Active positions mask [B, max_suffix_len] (1=active, 0=inactive)
        
        Returns:
            gradients: Gradients w.r.t. embeddings [B, max_suffix_len, emb_dim]
        """
        # Convert tokens to embeddings with gradients enabled
        suffix_embeds = self.embedding_layer(prompt_data)  # [B, max_suffix_len, emb_dim]
        suffix_embeds.requires_grad_(True)
        suffix_embeds.retain_grad()  # Required for non-leaf tensors to retain gradients
        
        # Temporarily update model_input with gradient-enabled embeddings
        # We need to manually construct the forward pass since discrete mode uses tokens
        # For gradient computation, we'll use embeddings directly
        original_mode = model_input.mode
        original_suffix_embeds = model_input.suffix_embeddings if hasattr(model_input, 'suffix_embeddings') else None
        
        # Store embeddings in model_input (temporarily switch to continuous mode logic)
        model_input.suffix_embeddings = suffix_embeds
        
        # Get concatenated embeddings and attention mask
        # We'll manually construct this to match get_model_input_embeds_and_attention_mask logic
        max_len = max(model_input.max_prefix_len, model_input.max_completion_len) if (model_input.max_prefix_len > 0 or model_input.max_completion_len > 0) else model_input.max_suffix_len
        
        # Pad prefix embeddings (left padding)
        if model_input.max_prefix_len > 0:
            prefix_embeds = self.embedding_layer(model_input.prefix_input_ids)  # [B, max_prefix_len, D]
            prefix_embeds_padded = torch.zeros(model_input.batch_size, max_len, self.emb_dim, device=self.device)
            prefix_embeds_padded[:, max_len - model_input.max_prefix_len:] = prefix_embeds
            prefix_mask_padded = torch.zeros(model_input.batch_size, max_len, dtype=torch.long, device=self.device)
            prefix_mask_padded[:, max_len - model_input.max_prefix_len:] = model_input.prefix_attention_mask
        else:
            prefix_embeds_padded = torch.zeros(model_input.batch_size, max_len, self.emb_dim, device=self.device)
            prefix_mask_padded = torch.zeros(model_input.batch_size, max_len, dtype=torch.long, device=self.device)
        
        # Pad completion embeddings (right padding)
        if model_input.max_completion_len > 0:
            completion_embeds = self.embedding_layer(model_input.completion_input_ids)  # [B, max_completion_len, D]
            completion_embeds_padded = torch.zeros(model_input.batch_size, max_len, self.emb_dim, device=self.device)
            completion_embeds_padded[:, :model_input.max_completion_len] = completion_embeds
            completion_mask_padded = torch.zeros(model_input.batch_size, max_len, dtype=torch.long, device=self.device)
            completion_mask_padded[:, :model_input.max_completion_len] = model_input.completion_attention_mask
        else:
            completion_embeds_padded = torch.zeros(model_input.batch_size, max_len, self.emb_dim, device=self.device)
            completion_mask_padded = torch.zeros(model_input.batch_size, max_len, dtype=torch.long, device=self.device)
        
        # Concatenate: prefix (padded) + suffix + completion (padded)
        inputs_embeds = torch.cat([prefix_embeds_padded, suffix_embeds, completion_embeds_padded], dim=1)  # [B, seq_len, D]
        attention_mask = torch.cat([prefix_mask_padded, suffix_mask, completion_mask_padded], dim=1)  # [B, seq_len]
        
        completion_start_pos = max_len + model_input.max_suffix_len
        
        # Forward pass with gradients
        outputs = self.agent.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden]
        logits = self.agent.model.embed_out(hidden_states)  # [B, seq_len, vocab]
        
        # Compute loss (negative log likelihood of completion)
        max_comp_len = model_input.completion_lengths.max().item()
        comp_logits = logits[:, completion_start_pos-1:completion_start_pos-1+max_comp_len, :]  # [B, max_comp_len, vocab]
        comp_tokens = model_input.completion_input_ids[:, :max_comp_len]  # [B, max_comp_len]
        
        log_probs = F.log_softmax(comp_logits, dim=-1)
        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, max_comp_len]
        
        comp_mask = torch.arange(max_comp_len, device=self.device).unsqueeze(0) < model_input.completion_lengths.unsqueeze(-1)
        masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
        loss = -masked_log_probs.sum(dim=-1).mean()  # Negative log likelihood (mean over batch)
        
        # Backward to get gradients
        loss.backward()
        gradients = suffix_embeds.grad.clone()  # [B, max_suffix_len, emb_dim]
        
        # Clear gradients and intermediate tensors to free memory
        if suffix_embeds.grad is not None:
            suffix_embeds.grad = None
        del loss, inputs_embeds, outputs, hidden_states, logits, comp_logits, comp_tokens, log_probs, token_log_probs, masked_log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Restore original state
        if original_suffix_embeds is not None:
            model_input.suffix_embeddings = original_suffix_embeds
        else:
            delattr(model_input, 'suffix_embeddings')
        
        return gradients
    
    def _sample_candidates_from_grad(self, prompt_data: torch.Tensor, gradients: torch.Tensor,
                                     suffix_mask: torch.Tensor, n_replace: int = 1) -> Tuple[torch.Tensor, list]:
        """
        Sample candidate sequences from gradients (following nanoGCG's sample_ids_from_grad).
        
        Generates search_width total candidate sequences (not per position).
        Each candidate updates n_replace randomly selected positions.
        
        Args:
            prompt_data: Current suffix tokens [B, max_suffix_len]
            gradients: Gradients w.r.t. embeddings [B, max_suffix_len, emb_dim]
            suffix_mask: Active positions mask [B, max_suffix_len]
            n_replace: Number of positions to update per candidate (default 1)
        
        Returns:
            candidate_sequences: Full candidate sequences [search_width * B, max_suffix_len]
            update_info: List of (prompt_idx, candidate_idx, pos, new_token) for each update
        """
        vocab_size = self.embedding_layer.weight.shape[0]
        embedding_weights = self.embedding_layer.weight  # [vocab_size, emb_dim]
        
        # Project gradients onto vocabulary space: grad @ W^T
        # This gives us gradient w.r.t. each possible token at each position
        grad_proj = gradients @ embedding_weights.T  # [B, max_suffix_len, vocab_size]
        
        # Get top-k tokens for each position: [B, max_suffix_len, top_k]
        topk_ids = (-grad_proj).topk(self.gcg_top_k, dim=-1).indices  # [B, max_suffix_len, top_k]
        
        search_width = self.gcg_batch_size  # Total number of candidate sequences per prompt
        
        # For each prompt, generate search_width candidate sequences
        all_candidate_sequences = []
        all_update_info = []
        
        # Use actual batch size from prompt_data (may be subset)
        actual_batch_size = prompt_data.shape[0]
        for prompt_idx in range(actual_batch_size):
            # Get active positions for this prompt
            active_positions = torch.nonzero(suffix_mask[prompt_idx] == 1, as_tuple=False).squeeze(-1)  # [num_active]
            
            if len(active_positions) == 0:
                continue
            
            num_active = len(active_positions)
            
            # Start with original sequence, repeat for all candidates
            original_sequence = prompt_data[prompt_idx:prompt_idx+1]  # [1, max_suffix_len]
            candidate_sequences = original_sequence.repeat(search_width, 1)  # [search_width, max_suffix_len]
            
            # For each candidate, randomly select n_replace positions to update
            # sampled_ids_pos: [search_width, n_replace] indices into active_positions
            # We use argsort of random values to randomly sample without replacement
            random_vals = torch.rand((search_width, num_active), device=self.device)
            sampled_pos_indices = torch.argsort(random_vals, dim=-1)[..., :n_replace]  # [search_width, n_replace]
            sampled_active_pos = active_positions[sampled_pos_indices]  # [search_width, n_replace]
            
            # For each candidate and each position to update, sample a token from top-k
            # sampled_ids_val: [search_width, n_replace] token IDs
            # topk_ids[prompt_idx] is [max_suffix_len, top_k]
            # sampled_active_pos is [search_width, n_replace] - indices into max_suffix_len
            # We need to gather from topk_ids for each position
            topk_for_positions = topk_ids[prompt_idx][sampled_active_pos]  # [search_width, n_replace, top_k]
            random_indices = torch.randint(0, self.gcg_top_k, (search_width, n_replace, 1), device=self.device)
            candidate_tokens = torch.gather(topk_for_positions, 2, random_indices).squeeze(2)  # [search_width, n_replace]
            
            # Update candidate sequences using scatter
            # scatter_(dim, index, src) where index and src have same shape
            candidate_sequences.scatter_(1, sampled_active_pos, candidate_tokens)
            
            all_candidate_sequences.append(candidate_sequences)
            
            # Store update info: (prompt_idx, candidate_idx, pos, new_token)
            for cand_idx in range(search_width):
                for replace_idx in range(n_replace):
                    pos = sampled_active_pos[cand_idx, replace_idx].item()
                    new_token = candidate_tokens[cand_idx, replace_idx].item()
                    all_update_info.append((prompt_idx, cand_idx, pos, new_token))
        
        if len(all_candidate_sequences) == 0:
            return torch.empty(0, self.max_suffix_len, dtype=torch.long, device=self.device), []
        
        candidate_sequences_tensor = torch.cat(all_candidate_sequences, dim=0)  # [search_width * B, max_suffix_len]
        return candidate_sequences_tensor, all_update_info
    
    def _test_candidates_batch(self, candidate_sequences: torch.Tensor, update_info: list,
                              model_input: ModelBatchedInput) -> torch.Tensor:
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
        
        # Get base concatenated input_ids structure from ModelBatchedInput
        base_input_ids, base_attention_mask, completion_start_pos = model_input.get_model_input_ids_and_attention_mask()
        max_len = max(model_input.max_prefix_len, model_input.max_completion_len) if (model_input.max_prefix_len > 0 or model_input.max_completion_len > 0) else model_input.max_suffix_len
        suffix_start_pos = max_len
        
        all_losses = []
        
        # Process in chunks to respect max_batch_size
        for chunk_start in range(0, total_candidates, self.gcg_max_batch_size):
            chunk_end = min(chunk_start + self.gcg_max_batch_size, total_candidates)
            chunk_sequences = candidate_sequences[chunk_start:chunk_end]  # [chunk_size, max_suffix_len]
            chunk_size = chunk_end - chunk_start
            
            # Get prompt indices for this chunk (each candidate corresponds to a prompt)
            # update_info format: (prompt_idx, candidate_idx, pos, new_token)
            # We need to map candidate_idx to the actual index in candidate_sequences
            # candidate_sequences is organized as: [prompt0_cand0, prompt0_cand1, ..., prompt1_cand0, ...]
            # So for candidate at index i, prompt_idx = i // search_width, candidate_idx = i % search_width
            search_width = self.gcg_batch_size
            prompt_indices = (torch.arange(chunk_start, chunk_end, device=self.device) // search_width).long()
            
            # Build candidate input_ids by replacing suffix portion
            candidate_input_ids = base_input_ids[prompt_indices].clone()  # [chunk_size, seq_len]
            candidate_attention_mask = base_attention_mask[prompt_indices].clone()  # [chunk_size, seq_len]
            
            # Replace suffix tokens with candidate sequences
            candidate_input_ids[:, suffix_start_pos:suffix_start_pos + self.max_suffix_len] = chunk_sequences
            
            # Get completion data for each candidate
            candidate_completion_input_ids = model_input.completion_input_ids[prompt_indices]  # [chunk_size, max_completion_len]
            candidate_completion_lengths = model_input.completion_lengths[prompt_indices]  # [chunk_size]
            
            # Forward pass
            with torch.no_grad():
                inputs_embeds = self.embedding_layer(candidate_input_ids)
                outputs = self.agent.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=candidate_attention_mask)
                hidden_states = outputs.last_hidden_state
                logits = self.agent.model.embed_out(hidden_states)
                
                # Compute losses (negative log likelihood)
                max_comp_len = candidate_completion_lengths.max().item()
                comp_logits = logits[:, completion_start_pos-1:completion_start_pos-1+max_comp_len, :]  # [chunk_size, max_comp_len, vocab]
                comp_tokens = candidate_completion_input_ids[:, :max_comp_len]  # [chunk_size, max_comp_len]
                
                log_probs = F.log_softmax(comp_logits, dim=-1)
                token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [chunk_size, max_comp_len]
                
                comp_mask = torch.arange(max_comp_len, device=self.device).unsqueeze(0) < candidate_completion_lengths.unsqueeze(-1)
                masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
                losses = -masked_log_probs.sum(dim=-1)  # [chunk_size] (negative log likelihood)
            
            all_losses.append(losses.cpu())  # Move to CPU to free GPU memory
            
            # Clear intermediate tensors
            del inputs_embeds, outputs, hidden_states, logits, comp_logits, comp_tokens, log_probs, token_log_probs, masked_log_probs, losses
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return torch.cat(all_losses, dim=0).to(self.device)  # [total_candidates] (move back to device)
    
    def inner_optimization_step(self, prompt_data: torch.Tensor, lengths: torch.Tensor,
                               step: int, model_input: ModelBatchedInput) -> Tuple[torch.Tensor, torch.Tensor]:
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
            return prompt_data, self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        # Get suffix mask to identify active positions
        suffix_mask = model_input.suffix_attention_mask  # [B, max_suffix_len]
        
        # Track best likelihoods and tokens
        model_input.update_suffix_tokens(prompt_data)
        best_lls = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        initial_lls = best_lls.clone()  # For monotonicity check at the end
        best_tokens = prompt_data.clone()
        
        # GCG iterations
        gcg_iter_bar = tqdm(range(self.gcg_steps), desc="GCG", leave=False)
        for gcg_iter in gcg_iter_bar:
            # Step 1: Compute gradients w.r.t. token embeddings
            gradients = self._compute_gradients(best_tokens, model_input, suffix_mask)  # [B, max_suffix_len, emb_dim]
            
            # Step 2: Sample candidates from gradients
            candidate_sequences, update_info = self._sample_candidates_from_grad(best_tokens, gradients, suffix_mask, n_replace=1)
            
            if candidate_sequences.shape[0] == 0:
                break
            
            # Step 3: Test candidates in batches
            losses = self._test_candidates_batch(candidate_sequences, update_info, model_input)  # [search_width * B]
            
            # Step 4: Pick best candidate per prompt
            # candidate_sequences is organized as: [prompt0_cand0, prompt0_cand1, ..., prompt1_cand0, ...]
            # So we reshape losses to [B, search_width] and pick best per prompt
            search_width = self.gcg_batch_size
            num_prompts = losses.shape[0] // search_width
            losses_reshaped = losses.view(num_prompts, search_width)  # [B, search_width]
            
            # Get best candidate index for each prompt (min loss = best)
            best_indices = losses_reshaped.argmin(dim=1)  # [B]
            
            # Update best tokens with best candidate sequences
            # Store old tokens and old likelihoods to track changes for logging
            old_tokens = best_tokens.clone()
            old_best_lls = best_lls.clone()
            
            for prompt_idx in range(num_prompts):
                best_candidate_idx = prompt_idx * search_width + best_indices[prompt_idx].item()
                best_tokens[prompt_idx] = candidate_sequences[best_candidate_idx]
            
            # Recompute likelihoods with updated tokens
            model_input.update_suffix_tokens(best_tokens)
            current_lls = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
            
            # Only keep improvements
            improve_mask = current_lls > best_lls
            best_lls = torch.where(improve_mask, current_lls, best_lls)
            
            # Debug log: token replacements and likelihood changes (reusing computed values)
            for prompt_idx in range(num_prompts):
                if improve_mask[prompt_idx]:
                    # Find which positions changed (reusing already computed tokens)
                    changed_positions = (old_tokens[prompt_idx] != best_tokens[prompt_idx]).nonzero(as_tuple=False).squeeze(-1)
                    if len(changed_positions.shape) == 0:
                        # Single position changed
                        changed_positions = changed_positions.unsqueeze(0)
                    if len(changed_positions) > 0:
                        for pos_tensor in changed_positions:
                            pos = pos_tensor.item() if torch.is_tensor(pos_tensor) else pos_tensor
                            old_token = old_tokens[prompt_idx, pos].item()
                            new_token = best_tokens[prompt_idx, pos].item()
                            old_token_str = self.agent.tokenizer.decode([old_token], skip_special_tokens=True)
                            new_token_str = self.agent.tokenizer.decode([new_token], skip_special_tokens=True)
                            # Reuse computed likelihoods: change = current_lls - old_best_lls
                            ll_change = current_lls[prompt_idx].item() - old_best_lls[prompt_idx].item()
                            print(f"GCG iter {gcg_iter+1} prompt {prompt_idx+1}: pos {pos} '{old_token_str}' -> '{new_token_str}', Δll={ll_change:.4f}")
            
            # Update progress bar
            gcg_iter_bar.set_postfix({'avg_ll': f'{best_lls.mean().item():.4f}'})
        
        # Final update
        prompt_data = best_tokens.clone()
        model_input.update_suffix_tokens(prompt_data)
        final_likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        # Monotonicity check: warn if final likelihoods are worse than initial
        # This should not typically happen if GCG is behaving as a greedy ascent step.
        ll_deltas = final_likelihoods - initial_lls
        if (ll_deltas < -1e-6).any():
            num_decreased = (ll_deltas < 0).sum().item()
            min_delta = ll_deltas.min().item()
            print(f"Warning: GCG decreased likelihood for {num_decreased} prompts in this step (min Δll={min_delta:.4f}).")
        
        return prompt_data, final_likelihoods
    
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
