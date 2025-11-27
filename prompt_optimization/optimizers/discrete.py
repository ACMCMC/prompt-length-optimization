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
                 batch_size: int, lr_embeddings: float, max_suffix_len: int, init_len: int,
                 gcg_steps: int, gcg_top_k: int, gcg_batch_size: int):
        super().__init__(agent, initial_prompt_length, max_prompt_len, batch_size, lr_embeddings, max_suffix_len, init_len)
        self.embedding_layer = agent.model.get_input_embeddings()
        self.gcg_steps = gcg_steps  # Steps per element
        self.gcg_top_k = gcg_top_k
        self.gcg_batch_size = gcg_batch_size
        
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
        GCG optimization with parallel position testing (following llm-attacks implementation).
        
        For each prompt, tests multiple positions in parallel, each with gcg_batch_size candidates.
        Total candidates tested per iteration: num_positions * gcg_batch_size (all in parallel).
        Repeats for gcg_steps iterations.
        """
        max_active_len = lengths.max().item()
        if max_active_len == 0:
            model_input.update_suffix_tokens(prompt_data)
            return prompt_data, self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        # Get current likelihoods
        model_input.update_suffix_tokens(prompt_data)
        likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=False)
        
        # GCG optimization: for each prompt, test all positions in parallel
        for i in range(self.batch_size):
            if lengths[i] == 0:
                continue
            
            num_elements = lengths[i].item()
            best_ll = likelihoods[i].item()
            best_tokens = prompt_data[i, :num_elements].clone()
            
            # Repeat for gcg_steps iterations
            for gcg_iter in range(self.gcg_steps):
                # Process positions one at a time to avoid OOM
                # For each position, test candidates in smaller batches to avoid OOM
                # Limit batch size to avoid memory issues
                max_candidates_per_batch = min(16, self.gcg_batch_size)  # Test at most 16 candidates at a time
                positions_to_test = list(range(num_elements))
                
                # Process each position sequentially
                for pos in positions_to_test:
                    # Test candidates for this position in batches
                    for candidate_batch_start in range(0, self.gcg_batch_size, max_candidates_per_batch):
                        candidate_batch_end = min(candidate_batch_start + max_candidates_per_batch, self.gcg_batch_size)
                        num_candidates_in_batch = candidate_batch_end - candidate_batch_start
                        total_candidates = num_candidates_in_batch
                
                        # Generate candidates for this batch: [num_candidates_in_batch]
                        candidates = torch.tensor(
                            [self.agent.get_random_token() for _ in range(num_candidates_in_batch)],
                            dtype=torch.long, device=self.device
                        )  # [num_candidates_in_batch]
                    
                        # Create test batch: duplicate prompt for each candidate
                        # Shape: [num_candidates_in_batch, max_prompt_len]
                        test_prompt_batch = prompt_data[i:i+1].expand(total_candidates, -1).clone()
                        
                        # Set candidates at position pos
                        test_prompt_batch[:, pos] = candidates
                        
                        # Prepare inputs for batch processing
                        prefix_input_ids = model_input.prefix_input_ids[i:i+1].expand(total_candidates, -1)
                        prefix_attention_mask = model_input.prefix_attention_mask[i:i+1].expand(total_candidates, -1)
                        completion_input_ids = model_input.completion_input_ids[i:i+1].expand(total_candidates, -1)
                        completion_attention_mask = model_input.completion_attention_mask[i:i+1].expand(total_candidates, -1)
                        completion_lengths = model_input.completion_lengths[i:i+1].expand(total_candidates)
                        suffix_mask = model_input.suffix_attention_mask[i:i+1].expand(total_candidates, -1)
                    
                        # Match ModelBatchedInput padding logic
                        prefix_len = prefix_input_ids.shape[1]
                        comp_len = completion_input_ids.shape[1]
                        max_len = max(prefix_len, comp_len) if (prefix_len > 0 or comp_len > 0) else model_input.max_suffix_len
                        pad_id = model_input.pad_id
                        
                        # Pad prefix (left padding)
                        if prefix_len > 0:
                            prefix_ids_padded = torch.full((total_candidates, max_len), pad_id, dtype=torch.long, device=self.device)
                            prefix_ids_padded[:, max_len - prefix_len:] = prefix_input_ids
                            prefix_mask_padded = torch.zeros(total_candidates, max_len, dtype=torch.long, device=self.device)
                            prefix_mask_padded[:, max_len - prefix_len:] = prefix_attention_mask
                        else:
                            prefix_ids_padded = torch.full((total_candidates, max_len), pad_id, dtype=torch.long, device=self.device)
                            prefix_mask_padded = torch.zeros(total_candidates, max_len, dtype=torch.long, device=self.device)
                        
                        # Pad completion (right padding)
                        if comp_len > 0:
                            completion_ids_padded = torch.full((total_candidates, max_len), pad_id, dtype=torch.long, device=self.device)
                            completion_ids_padded[:, :comp_len] = completion_input_ids
                            completion_mask_padded = torch.zeros(total_candidates, max_len, dtype=torch.long, device=self.device)
                            completion_mask_padded[:, :comp_len] = completion_attention_mask
                        else:
                            completion_ids_padded = torch.full((total_candidates, max_len), pad_id, dtype=torch.long, device=self.device)
                            completion_mask_padded = torch.zeros(total_candidates, max_len, dtype=torch.long, device=self.device)
                        
                        # Concatenate: prefix (padded) + suffix + completion (padded)
                        full_input_ids = torch.cat([prefix_ids_padded, test_prompt_batch, completion_ids_padded], dim=1)
                        full_attention_mask = torch.cat([prefix_mask_padded, suffix_mask, completion_mask_padded], dim=1)
                        
                        # Completion starts after prefix (padded) and suffix
                        completion_start_pos = max_len + model_input.max_suffix_len
                        
                        # Get embeddings
                        inputs_embeds = self.embedding_layer(full_input_ids)  # [total_candidates, total_len, D]
                        
                        # Forward pass
                        with torch.no_grad():
                            outputs = self.agent.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask)
                            hidden_states = outputs.last_hidden_state  # [total_candidates, total_len, hidden]
                            logits = self.agent.model.embed_out(hidden_states)  # [total_candidates, total_len, vocab]
                        
                        # Extract logits for completion positions
                        max_comp_len = completion_lengths.max().item()
                        comp_logits = logits[:, completion_start_pos-1:completion_start_pos-1+max_comp_len, :]  # [total_candidates, max_comp_len, vocab]
                        
                        # Extract completion tokens
                        comp_tokens = completion_input_ids[:, :max_comp_len]  # [total_candidates, max_comp_len]
                        
                        # Compute log probabilities
                        import torch.nn.functional as F
                        log_probs = F.log_softmax(comp_logits, dim=-1)
                        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [total_candidates, max_comp_len]
                        
                        # Mask invalid positions
                        comp_mask = torch.arange(max_comp_len, device=self.device).unsqueeze(0) < completion_lengths.unsqueeze(-1)
                        masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
                        
                        # Sum over completion length
                        candidate_lls_batch = masked_log_probs.sum(dim=-1)  # [total_candidates]
                        
                        # Track best candidate across all batches for this position
                        if candidate_batch_start == 0:
                            best_candidate_ll_for_pos = candidate_lls_batch.max().item()
                            best_candidate_idx_for_pos = candidate_lls_batch.argmax().item()
                            best_candidate_token_for_pos = candidates[best_candidate_idx_for_pos].item()
                        else:
                            # Compare with previous batches
                            current_best_ll = candidate_lls_batch.max().item()
                            if current_best_ll > best_candidate_ll_for_pos:
                                best_candidate_ll_for_pos = current_best_ll
                                best_candidate_idx_for_pos = candidate_lls_batch.argmax().item()
                                best_candidate_token_for_pos = candidates[best_candidate_idx_for_pos].item()
                        
                        # Clear cache to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Apply best candidate for this position if it's better than current best
                    if best_candidate_ll_for_pos > best_ll:
                        best_ll = best_candidate_ll_for_pos
                        # Update the token at this position
                        prompt_data[i, pos] = best_candidate_token_for_pos
                        best_tokens = prompt_data[i, :num_elements].clone()
                        # Update model_input with the improvement
                        model_input.update_suffix_tokens(prompt_data)
        
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

