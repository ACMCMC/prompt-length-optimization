"""
RL Agent for prompt optimization: handles model interactions and likelihood computation
"""

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
    
    def get_likelihoods_batch(self, prompt_embeds: torch.Tensor, completion_tokens: torch.Tensor, 
                             completion_lengths: torch.Tensor, requires_grad: bool = False,
                             prefix_tokens: torch.Tensor = None, prefix_lengths: torch.Tensor = None,
                             attention_mask: torch.Tensor = None, suffix_attention_mask: torch.Tensor = None,
                             max_suffix_len: int = 64, init_len: int = 32) -> torch.Tensor:
        """
        Batched likelihood computation with prefix + suffix structure.
        
        Structure: [prefix (max_len, padded left)] + [suffix (max_suffix_len, all BOS)] + [completion (max_len, padded right)]
        
        Args:
            prompt_embeds: Suffix embeddings [B, max_suffix_len, D] (fixed size, all positions initialized to BOS)
            completion_tokens: Completion tokens [B, max_comp]
            completion_lengths: Actual completion lengths [B]
            prefix_tokens: Prefix tokens [B, max_prefix] (optional, defaults to empty)
            prefix_lengths: Actual prefix lengths [B] (optional)
            attention_mask: Attention mask [B, max_seq] (optional, auto-generated if None)
            suffix_attention_mask: Suffix attention mask [B, max_suffix_len] (optional, auto-generated if None)
            requires_grad: Whether to enable gradients
            max_suffix_len: Maximum suffix length from config (default: 64)
            init_len: Initial number of suffix positions with attention mask = 1 from config (default: 32)
        """
        B, L_suffix, D = prompt_embeds.shape
        device = self.device
        embedding_layer = self.model.get_input_embeddings()
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        pad_embed = embedding_layer.weight[pad_id] if pad_id is not None else torch.zeros(D, device=device)
        if not requires_grad:
            pad_embed = pad_embed.detach()
        
        # Validate suffix size matches config
        if L_suffix != max_suffix_len:
            raise ValueError(f"Suffix size must be {max_suffix_len} (from config), got {L_suffix}")
        
        # Handle prefix: default to empty
        if prefix_tokens is None:
            prefix_tokens = torch.empty(B, 0, dtype=torch.long, device=device)
            prefix_lengths = torch.zeros(B, dtype=torch.long, device=device)
        else:
            # Ensure batch size matches
            if prefix_tokens.shape[0] != B:
                raise ValueError(f"prefix_tokens batch size {prefix_tokens.shape[0]} doesn't match prompt_embeds batch size {B}")
            if prefix_lengths is None:
                prefix_lengths = torch.full((B,), prefix_tokens.shape[1], dtype=torch.long, device=device)
            elif prefix_lengths.shape[0] != B:
                raise ValueError(f"prefix_lengths batch size {prefix_lengths.shape[0]} doesn't match prompt_embeds batch size {B}")
        
        max_prefix = prefix_tokens.shape[1] if prefix_tokens.numel() > 0 else 0
        max_comp = completion_tokens.shape[1]
        
        # Ensure completion_tokens batch size matches
        if completion_tokens.shape[0] != B:
            raise ValueError(f"completion_tokens batch size {completion_tokens.shape[0]} doesn't match prompt_embeds batch size {B}")
        if completion_lengths.shape[0] != B:
            raise ValueError(f"completion_lengths batch size {completion_lengths.shape[0]} doesn't match prompt_embeds batch size {B}")
        
        # max_len is the maximum of prefix and completion sizes (tokenizer will pad to this)
        max_len = max(max_prefix, max_comp) if (max_prefix > 0 or max_comp > 0) else max_suffix_len
        
        # ===== CONSTRUCT THREE SEPARATE TENSORS =====
        
        # 1. Prefix: [B, max_len, D] (tokenizer handles left padding)
        prefix_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_len, 1).to(device)
        prefix_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        if max_prefix > 0:
            # Pad prefix tokens to max_len (left padding)
            padded_prefix_tokens = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            for i in range(B):
                prefix_len = prefix_lengths[i].item()
                if prefix_len > 0:
                    # Left padding: pad tokens go on the left, actual tokens on the right
                    padded_prefix_tokens[i, max_len - prefix_len:] = prefix_tokens[i, :prefix_len]
                    # Attention mask: 1 for actual tokens (right side), 0 for padding (left side)
                    prefix_attention_mask[i, max_len - prefix_len:] = 1
            
            # Embed padded tokens
            prefix_embeds = embedding_layer(padded_prefix_tokens)  # [B, max_len, D]
        else:
            # No prefix, all padding
            prefix_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        # 2. Suffix: [B, max_suffix_len, D] (fixed size from config, all positions initialized to BOS, learnable)
        # prompt_embeds is already [B, max_suffix_len, D] with all BOS
        suffix_embeds = prompt_embeds  # [B, max_suffix_len, D]
        suffix_size = max_suffix_len
        
        # Suffix attention mask: use provided mask or create default (first init_len positions active)
        if suffix_attention_mask is not None:
            suffix_attention_mask_tensor = suffix_attention_mask  # [B, max_suffix_len]
        else:
            # Default: first init_len positions active (from YAML config)
            suffix_attention_mask_tensor = torch.zeros(B, max_suffix_len, dtype=torch.long, device=device)
            suffix_attention_mask_tensor[:, :init_len] = 1  # First init_len positions enabled
        
        # 3. Completion: [B, max_len, D] (tokenizer handles right padding)
        completion_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_len, 1).to(device)
        completion_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        if max_comp > 0:
            # Pad completion tokens to max_len (right padding)
            padded_completion_tokens = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            for i in range(B):
                comp_len = completion_lengths[i].item()
                if comp_len > 0:
                    # Right padding: actual tokens go on the left, pad tokens on the right
                    padded_completion_tokens[i, :comp_len] = completion_tokens[i, :comp_len]
                    # Attention mask: 1 for actual tokens (left side), 0 for padding (right side)
                    completion_attention_mask[i, :comp_len] = 1
            
            # Embed padded tokens
            completion_embeds = embedding_layer(padded_completion_tokens)  # [B, max_len, D]
        else:
            # No completion, all padding
            completion_attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        
        # ===== CONCATENATE ALL THREE TENSORS =====
        inputs_embeds = torch.cat([prefix_embeds, suffix_embeds, completion_embeds], dim=1)  # [B, prefix_size + suffix_size + completion_size, D]
        attention_mask = torch.cat([prefix_attention_mask, suffix_attention_mask_tensor, completion_attention_mask], dim=1)  # [B, prefix_size + suffix_size + completion_size]
        
        # Position offsets for likelihood computation
        pos_prefix_end = max_len
        pos_suffix_end = pos_prefix_end + suffix_size  # max_len + 64
        pos_comp_end = pos_suffix_end + max_len  # max_len + 64 + max_len
        
        # Forward pass with attention mask
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, max_seq, hidden]
            logits = self.model.embed_out(hidden_states)  # [B, max_seq, vocab]
        
        # Vectorized likelihood computation
        max_comp_actual = completion_lengths.max().item()
        if max_comp_actual == 0:
            return torch.zeros(B, dtype=torch.float32, device=device, requires_grad=requires_grad)
        
        # Extract logits for completion positions (before each completion token)
        # We need to extract logits at positions corresponding to actual completion tokens
        # Completion starts at pos_suffix_end, so we extract from pos_suffix_end-1 to pos_suffix_end-1+max_comp_actual
        comp_logits = logits[:, pos_suffix_end-1:pos_suffix_end-1+max_comp_actual, :]  # [B, max_comp_actual, vocab]
        
        # Extract completion tokens (only actual tokens, not padded)
        comp_tokens = completion_tokens[:, :max_comp_actual]
        
        # Compute log probabilities
        log_probs = F.log_softmax(comp_logits, dim=-1)
        
        # Gather token log probs
        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, max_comp_actual]
        
        # Mask invalid positions
        comp_mask = torch.arange(max_comp_actual, device=device).unsqueeze(0) < completion_lengths.unsqueeze(-1)
        masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
        
        # Sum over completion length
        likelihoods = masked_log_probs.sum(dim=-1)  # [B]
        
        return likelihoods
    
    def get_random_token(self) -> int:
        """Get a random token ID (excluding special tokens)."""
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in self.special_token_ids:
                return token

