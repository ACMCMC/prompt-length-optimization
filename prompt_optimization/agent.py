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
                             attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Batched likelihood computation with left-padded prefix + suffix + right-padded completion.
        
        Structure: [left_pad (for prefix)] + [prefix] + [suffix] + [completion] + [right_pad (for completion)]
        
        This maintains fixed total sequence length regardless of actual prefix/completion lengths.
        Attention mask is 0 for padding positions, 1 for real tokens.
        
        Args:
            prompt_embeds: Suffix embeddings [B, L_suffix, D]
            completion_tokens: Completion tokens [B, max_comp]
            completion_lengths: Actual completion lengths [B]
            prefix_tokens: Prefix tokens [B, max_prefix] (optional, defaults to empty)
            prefix_lengths: Actual prefix lengths [B] (optional)
            attention_mask: Attention mask [B, max_seq] (optional, auto-generated if None)
            requires_grad: Whether to enable gradients
        """
        B, L_suffix, D = prompt_embeds.shape
        device = self.device
        embedding_layer = self.model.get_input_embeddings()
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        pad_embed = embedding_layer.weight[pad_id] if pad_id is not None else torch.zeros(D, device=device)
        if not requires_grad:
            pad_embed = pad_embed.detach()
        
        # Handle prefix: default to empty
        if prefix_tokens is None:
            prefix_tokens = torch.empty(B, 0, dtype=torch.long, device=device)
            prefix_lengths = torch.zeros(B, dtype=torch.long, device=device)
        elif prefix_lengths is None:
            prefix_lengths = torch.full((B,), prefix_tokens.shape[1], dtype=torch.long, device=device)
        
        max_prefix = prefix_tokens.shape[1] if prefix_tokens.numel() > 0 else 0
        max_comp = completion_tokens.shape[1]
        print("max_comp :", max_comp)
        print("max_prefix :", max_prefix)
        print("L_suffix :", L_suffix)
        # Build full sequence structure with LEFT padding for prefix and RIGHT padding for completion
        # Structure: [left_pad] + [prefix] + [suffix] + [completion] + [right_pad]
        # Total fixed length = max_prefix + L_suffix + max_comp
        max_seq = max_prefix + L_suffix + max_comp
        print("max_seq :", max_seq)
        # Initialize everything with padding embeddings
        inputs_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_seq, 1).clone().to(device)
        
        # Initialize attention mask to 0 (masked out / padding)
        attn_mask = torch.zeros(B, max_seq, dtype=torch.long, device=device)
        
        # Position boundaries (fixed for all items in batch)
        # [0 ... max_prefix) = left_pad + prefix region
        # [max_prefix ... max_prefix + L_suffix) = suffix region
        # [max_prefix + L_suffix ... max_seq) = completion + right_pad region
        pos_prefix_region_end = max_prefix
        pos_suffix_start = max_prefix
        pos_suffix_end = max_prefix + L_suffix
        pos_comp_start = pos_suffix_end
        
        # Fill prefix with LEFT padding
        # For each item i: left_pad goes [0, max_prefix - prefix_lengths[i])
        #                  real prefix goes [max_prefix - prefix_lengths[i], max_prefix)
        if max_prefix > 0:
            prefix_embeds = embedding_layer(prefix_tokens)  # [B, max_prefix, D]
            for i in range(B):
                plen = prefix_lengths[i].item()
                if plen > 0:
                    # Left-align the padding, right-align the actual prefix
                    left_pad_len = max_prefix - plen
                    # Copy actual prefix tokens to the right side of prefix region
                    inputs_embeds[i, left_pad_len:max_prefix, :] = prefix_embeds[i, :plen, :]
                    # Set attention mask to 1 for actual prefix tokens
                    attn_mask[i, left_pad_len:max_prefix] = 1
                # If plen == 0, entire prefix region stays as padding with attn_mask = 0
        
        # Fill suffix (always fully used, no padding in suffix)
        inputs_embeds[:, pos_suffix_start:pos_suffix_end, :] = prompt_embeds
        attn_mask[:, pos_suffix_start:pos_suffix_end] = 1  # Suffix is always attended to
        
        # Fill completion with RIGHT padding
        # For each item i: real completion goes [pos_comp_start, pos_comp_start + completion_lengths[i])
        #                  right_pad goes [pos_comp_start + completion_lengths[i], max_seq)
        comp_embeds = embedding_layer(completion_tokens)  # [B, max_comp, D]
        for i in range(B):
            clen = completion_lengths[i].item()
            if clen > 0:
                # Copy actual completion tokens to the left side of completion region
                inputs_embeds[i, pos_comp_start:pos_comp_start + clen, :] = comp_embeds[i, :clen, :]
                # Set attention mask to 1 for actual completion tokens
                attn_mask[i, pos_comp_start:pos_comp_start + clen] = 1
            # If clen == 0, entire completion region stays as padding with attn_mask = 0
        
        # Use provided attention mask if given, otherwise use auto-generated one
        if attention_mask is not None:
            attn_mask = attention_mask
        
        # Forward pass with attention mask
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=attn_mask)
            hidden_states = outputs.last_hidden_state  # [B, max_seq, hidden]
            logits = self.model.embed_out(hidden_states)  # [B, max_seq, vocab]

        
        # Vectorized likelihood computation
        max_comp_actual = completion_lengths.max().item()
        if max_comp_actual == 0:
            return torch.zeros(B, dtype=torch.float32, device=device, requires_grad=requires_grad)
        
        # Extract logits for completion positions
        # The logit at position (pos_comp_start - 1) predicts the first completion token
        # The logit at position (pos_comp_start + k - 1) predicts completion token k
        comp_logits = logits[:, pos_comp_start - 1:pos_comp_start - 1 + max_comp_actual, :]  # [B, max_comp_actual, vocab]
        # Extract completion tokens
        comp_tokens = completion_tokens[:, :max_comp_actual]
        print("comp tokens : ", comp_tokens)
        
        # Compute log probabilities
        log_probs = F.log_softmax(comp_logits, dim=-1)
        
        # Gather token log probs
        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, max_comp_actual]
        
        # Mask invalid positions (beyond actual completion length)
        comp_mask = torch.arange(max_comp_actual, device=device).unsqueeze(0) < completion_lengths.unsqueeze(-1)
        masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
        
        # Sum over completion length to get total log likelihood
        likelihoods = masked_log_probs.sum(dim=-1)  # [B]
        print("Likelihoods shape:", likelihoods.shape)
        print("likelihood : ",likelihoods)
        return likelihoods
    
    def get_random_token(self) -> int:
        """Get a random token ID (excluding special tokens)."""
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in self.special_token_ids:
                return token

