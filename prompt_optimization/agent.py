"""
RL Agent for prompt optimization: handles model interactions and likelihood computation
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import random

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
            tok for tok in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
            if tok is not None
        }
        self.vocab_size = len(self.tokenizer)
    
    def get_likelihoods_batch(self, model_input, requires_grad: bool = False) -> torch.Tensor:
        """
        Batched likelihood computation using ModelBatchedInput.
        
        Args:
            model_input: ModelBatchedInput instance with all inputs
            requires_grad: Whether to enable gradients
        """
        B = model_input.batch_size
        device = self.device
        max_comp_actual = model_input.completion_lengths.max().item()
        
        if max_comp_actual == 0:
            return torch.zeros(B, dtype=torch.float32, device=device, requires_grad=requires_grad)
        
        # Get concatenated inputs from ModelBatchedInput
        if model_input.mode == 'continuous':
            inputs_embeds, attention_mask, suffix_mask, completion_start_pos = model_input.get_model_input_embeds_and_attention_mask()
            # Prefix and completion embeddings are already detached in get_model_input_embeds_and_attention_mask
            # Only suffix embeddings have gradients
        else:
            # For discrete mode, we need embeddings for forward pass
            input_ids, attention_mask, completion_start_pos = model_input.get_model_input_ids_and_attention_mask()
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Forward pass with attention mask
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden]
            logits = self.model.embed_out(hidden_states)  # [B, seq_len, vocab]
        
        # Extract logits for completion positions (before each completion token)
        # Completion starts at completion_start_pos, so we extract from completion_start_pos-1 to completion_start_pos-1+max_comp_actual
        comp_logits = logits[:, completion_start_pos-1:completion_start_pos-1+max_comp_actual, :]  # [B, max_comp_actual, vocab]
        
        # Extract completion tokens (only actual tokens, not padded)
        comp_tokens = model_input.completion_input_ids[:, :max_comp_actual]
        
        # Compute log probabilities
        log_probs = F.log_softmax(comp_logits, dim=-1)
        
        # Gather token log probs
        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, max_comp_actual]
        
        # Mask invalid positions
        comp_mask = torch.arange(max_comp_actual, device=device).unsqueeze(0) < model_input.completion_lengths.unsqueeze(-1)
        masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
        
        # Sum over completion length
        likelihoods = masked_log_probs.sum(dim=-1)  # [B]
        
        return likelihoods
    
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
    