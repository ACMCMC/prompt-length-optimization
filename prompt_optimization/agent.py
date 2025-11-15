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
                             completion_lengths: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """Batched likelihood computation: [B, L, D] embeddings -> [B] likelihoods"""
        B, L, D = prompt_embeds.shape
        device = self.device
        embedding_layer = self.model.get_input_embeddings()
        
        # Get completion embeddings
        comp_embeds = embedding_layer(completion_tokens)  # [B, max_comp, D]
        
        # Build full sequence: [prompt_embeds, comp_embeds] for each batch item
        max_comp = completion_tokens.shape[1]
        max_full = L + max_comp
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
        pad_embed = embedding_layer.weight[pad_id] if pad_id is not None else torch.zeros(D, device=device)
        if not requires_grad:
            pad_embed = pad_embed.detach()
        
        inputs_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_full, 1).to(device)
        for i in range(B):
            inputs_embeds[i, :L, :] = prompt_embeds[i]
            comp_len = completion_lengths[i].item()
            if comp_len > 0:
                inputs_embeds[i, L:L+comp_len, :] = comp_embeds[i, :comp_len]
        
        # Forward pass
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            outputs = self.model.gpt_neox(inputs_embeds=inputs_embeds)
            hidden_states = outputs.last_hidden_state  # [B, max_full, hidden]
            logits = self.model.embed_out(hidden_states)  # [B, max_full, vocab]
        
        # Compute likelihoods for each batch item
        likelihood_list = []
        for i in range(B):
            comp_len = completion_lengths[i].item()
            if comp_len > 0:
                comp_logits = logits[i, L-1:L-1+comp_len]  # [comp_len, vocab]
                comp_tokens = completion_tokens[i, :comp_len]  # [comp_len]
                log_probs = F.log_softmax(comp_logits, dim=-1)
                token_log_probs = log_probs.gather(1, comp_tokens.unsqueeze(1)).squeeze()
                likelihood_list.append(token_log_probs.sum())
            else:
                likelihood_list.append(torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=requires_grad))
        
        return torch.stack(likelihood_list)
    
    def get_random_token(self) -> int:
        """Get a random token ID (excluding special tokens)."""
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in self.special_token_ids:
                return token

