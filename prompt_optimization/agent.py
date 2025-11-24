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
        Batched likelihood computation with prefix + suffix structure.
        
        Structure: [padding_left] + [prefix] + [suffix] + [completion] + [padding_right]
        
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
        
        # Padding sizes: 32 on left, variable on right
        padding_left = 32
        padding_right = max(32, max_comp)  # At least 32, or max completion length
        
        # Build full sequence structure
        max_seq = padding_left + max_prefix + L_suffix + max_comp + padding_right
        
        # Initialize with padding
        inputs_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_seq, 1).to(device)
        
        # Position offsets
        pos_pad_left = padding_left
        pos_prefix_end = pos_pad_left + max_prefix
        pos_suffix_end = pos_prefix_end + L_suffix
        pos_comp_end = pos_suffix_end + max_comp
        
        # Fill prefix (if exists)
        if max_prefix > 0:
            prefix_embeds = embedding_layer(prefix_tokens)  # [B, max_prefix, D]
            prefix_mask = torch.arange(max_prefix, device=device).unsqueeze(0) < prefix_lengths.unsqueeze(-1)  # [B, max_prefix]
            prefix_mask_expanded = prefix_mask.unsqueeze(-1).expand(-1, -1, D)
            # Ensure slice size matches prefix_embeds size
            slice_size = pos_prefix_end - pos_pad_left
            if slice_size != max_prefix:
                raise ValueError(f"Prefix slice size {slice_size} doesn't match max_prefix {max_prefix} (pos_pad_left={pos_pad_left}, pos_prefix_end={pos_prefix_end})")
            inputs_embeds[:, pos_pad_left:pos_prefix_end, :] = torch.where(
                prefix_mask_expanded,
                prefix_embeds,
                pad_embed.unsqueeze(0).unsqueeze(0).expand(B, max_prefix, -1)
            )
        
        # Fill suffix (prompt_embeds)
        suffix_slice_size = pos_suffix_end - pos_prefix_end
        if suffix_slice_size != L_suffix:
            raise ValueError(f"Suffix slice size {suffix_slice_size} doesn't match L_suffix {L_suffix} (pos_prefix_end={pos_prefix_end}, pos_suffix_end={pos_suffix_end})")
        inputs_embeds[:, pos_prefix_end:pos_suffix_end, :] = prompt_embeds
        
        # Fill completion
        comp_embeds = embedding_layer(completion_tokens)  # [B, max_comp, D]
        comp_mask = torch.arange(max_comp, device=device).unsqueeze(0) < completion_lengths.unsqueeze(-1)  # [B, max_comp]
        comp_mask_expanded = comp_mask.unsqueeze(-1).expand(-1, -1, D)
        # Ensure slice size matches completion size
        comp_slice_size = pos_comp_end - pos_suffix_end
        if comp_slice_size != max_comp:
            raise ValueError(f"Completion slice size {comp_slice_size} doesn't match max_comp {max_comp} (pos_suffix_end={pos_suffix_end}, pos_comp_end={pos_comp_end})")
        inputs_embeds[:, pos_suffix_end:pos_comp_end, :] = torch.where(
            comp_mask_expanded,
            comp_embeds,
            pad_embed.unsqueeze(0).unsqueeze(0).expand(B, max_comp, -1)
        )
        
        # Build attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.zeros(B, max_seq, dtype=torch.long, device=device)
            # Mask: 1 for valid tokens, 0 for padding
            for i in range(B):
                # Left padding: all 0 (masked)
                # Prefix: 1 for valid prefix tokens
                prefix_start = pos_pad_left
                prefix_end = prefix_start + prefix_lengths[i].item()
                attention_mask[i, prefix_start:prefix_end] = 1
                # Suffix: all 1 (all valid)
                attention_mask[i, pos_prefix_end:pos_suffix_end] = 1
                # Completion: 1 for valid completion tokens
                comp_start = pos_suffix_end
                comp_end = comp_start + completion_lengths[i].item()
                attention_mask[i, comp_start:comp_end] = 1
        
        # Forward pass with attention mask
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.gpt_neox(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, max_seq, hidden]
            logits = self.model.embed_out(hidden_states)  # [B, max_seq, vocab]
        
        # Vectorized likelihood computation
        max_comp = completion_lengths.max().item()
        if max_comp == 0:
            return torch.zeros(B, dtype=torch.float32, device=device, requires_grad=requires_grad)
        
        # Extract logits for completion positions (before each completion token)
        comp_logits = logits[:, pos_suffix_end-1:pos_suffix_end-1+max_comp, :]  # [B, max_comp, vocab]
        
        # Extract completion tokens
        comp_tokens = completion_tokens[:, :max_comp]
        
        # Compute log probabilities
        log_probs = F.log_softmax(comp_logits, dim=-1)
        
        # Gather token log probs
        token_log_probs = log_probs.gather(2, comp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, max_comp]
        
        # Mask invalid positions
        comp_mask = torch.arange(max_comp, device=device).unsqueeze(0) < completion_lengths.unsqueeze(-1)
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

