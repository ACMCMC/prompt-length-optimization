"""
POC: RL-based Prompt Length Optimization
Find shortest prompts that maximize P(completion | prompt) using batched torch operations
"""

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import trange
import random
from typing import List, Tuple, Optional
import torch.nn as nn
import torch.optim as optim

class PromptRLAgent:
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
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in self.special_token_ids:
                return token

class LengthPolicyOptimizer:
    def __init__(self, agent: PromptRLAgent):
        self.agent = agent
        self.emb_dim = agent.model.get_input_embeddings().weight.shape[1]
        
        # Simple policy network: state -> action probs
        self.state_dim = 4  # [length, likelihood, step_ratio, improvement]
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Actions: 0=remove, 1=keep, 2=add
        ).to(agent.device)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
    
    def _prepare_completions(self, target_completions: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare completion tokens as batched tensors"""
        completion_tokens_list = [self.agent.tokenizer.encode(t, add_special_tokens=False) for t in target_completions]
        max_comp_len = max(len(ct) for ct in completion_tokens_list) if completion_tokens_list else 0
        pad_id = getattr(self.agent.tokenizer, 'pad_token_id', 0)
        completion_tokens_batch = torch.tensor([
            ct + [pad_id] * (max_comp_len - len(ct)) for ct in completion_tokens_list
        ], dtype=torch.long, device=self.agent.device)
        completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=self.agent.device)
        return completion_tokens_batch, completion_lengths
    
    def _embeddings_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to nearest token IDs"""
        embedding_layer = self.agent.model.get_input_embeddings()
        vocab_embeds = embedding_layer.weight.detach()  # [vocab_size, D]
        D = embeddings.shape[-1]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        distances = torch.cdist(embeddings, vocab_embeds)  # [L, vocab_size]
        token_ids = distances.argmin(dim=-1)  # [L]
        return token_ids.squeeze()
    
    def optimize_prompts_batch(self, target_completions: List[str], episodes: int = 3,
                               steps_per_episode: int = 50, initial_prompt_length: int = 32,
                               lr_embeddings: float = 0.01, alpha: float = 1.0, beta: float = 0.1,
                               mode: str = "continuous") -> Tuple[List[torch.Tensor], List[float], List[dict]]:
        """Unified batch optimization: continuous or discrete mode"""
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []
        
        completion_tokens_batch, completion_lengths = self._prepare_completions(target_completions)
        embedding_layer = self.agent.model.get_input_embeddings()
        D = self.emb_dim
        L = initial_prompt_length
        
        # Initialize prompts with fixed max length (simpler for batching)
        max_prompt_len = initial_prompt_length * 2  # Allow growth
        if mode == "continuous":
            prompt_embeds = nn.Parameter(torch.randn(B, max_prompt_len, D, device=device) * 0.1)
            prompt_optimizer = optim.Adam([prompt_embeds], lr=lr_embeddings)
            lengths = torch.full((B,), L, dtype=torch.long, device=device)
        else:  # discrete
            prompt_tokens = torch.tensor([
                [self.agent.get_random_token() for _ in range(max_prompt_len)] for _ in range(B)
            ], dtype=torch.long, device=device)
            lengths = torch.full((B,), L, dtype=torch.long, device=device)
        
        best_rewards = torch.full((B,), float('-inf'), dtype=torch.float32, device=device)
        best_prompts: List[Optional[torch.Tensor]] = [None] * B
        
        traces = []
        
        for episode in trange(episodes, desc="Episodes"):
            episode_rewards = []
            episode_log_probs = []
            episode_states = []
            
            step_bar = trange(steps_per_episode, desc=f"Episode {episode+1}", leave=False) if episodes > 1 else range(steps_per_episode)
            for step in step_bar:
                # Get current likelihoods (use only active length, pad to max for batching)
                max_active_len = lengths.max().item()
                if mode == "continuous":
                    active_embeds = prompt_embeds[:, :max_active_len]
                    # Simple gradient-free optimization: random walk with acceptance
                    with torch.no_grad():
                        base_likelihoods = self.agent.get_likelihoods_batch(active_embeds, completion_tokens_batch, completion_lengths, requires_grad=False)
                    # Try small random perturbations
                    for _ in range(3):
                        noise = torch.randn_like(active_embeds) * 0.01
                        test_embeds = active_embeds + noise
                        with torch.no_grad():
                            test_likelihoods = self.agent.get_likelihoods_batch(test_embeds, completion_tokens_batch, completion_lengths, requires_grad=False)
                        # Accept if better
                        for i in range(B):
                            if test_likelihoods[i] > base_likelihoods[i]:
                                prompt_embeds.data[i, :max_active_len] = test_embeds[i]
                                base_likelihoods[i] = test_likelihoods[i]
                    likelihoods = base_likelihoods
                else:  # discrete
                    active_tokens = prompt_tokens[:, :max_active_len]
                    prompt_embeds_current = embedding_layer(active_tokens)
                    likelihoods = self.agent.get_likelihoods_batch(prompt_embeds_current, completion_tokens_batch, completion_lengths, requires_grad=False)
                    # Simple GCG: try random token replacements (batched, limited for speed)
                    # Only optimize a few items per step to speed up
                    if step % 3 == 0:  # Only optimize every 3rd step
                        num_to_optimize = min(8, B)  # Optimize max 8 items per step
                        indices = random.sample(range(B), num_to_optimize) if B > num_to_optimize else list(range(B))
                        for i in indices:
                            if lengths[i] > 0:
                                best_ll = likelihoods[i].item()
                                best_tokens = prompt_tokens[i, :max_active_len].clone()
                                # Try fewer replacements for speed
                                for _ in range(2):  # Only 2 attempts
                                    pos = random.randint(0, lengths[i].item() - 1)
                                    candidate = self.agent.get_random_token()
                                    test_tokens = best_tokens.clone()
                                    test_tokens[pos] = candidate
                                    test_embeds = embedding_layer(test_tokens).unsqueeze(0)
                                    test_ll = self.agent.get_likelihoods_batch(test_embeds, completion_tokens_batch[i:i+1], completion_lengths[i:i+1], requires_grad=False)[0]
                                    if test_ll.item() > best_ll:
                                        best_ll = test_ll.item()
                                        best_tokens = test_tokens
                                prompt_tokens[i, :max_active_len] = best_tokens
                                likelihoods[i] = torch.tensor(best_ll, device=device)
                
                # Compute states for policy
                step_ratio = step / steps_per_episode
                states = torch.stack([
                    lengths.float() / initial_prompt_length,  # normalized length
                    likelihoods,  # current likelihood
                    torch.full((B,), step_ratio, device=device),  # step ratio
                    torch.zeros(B, device=device)  # improvement (simplified)
                ], dim=1)  # [B, 4]
                
                # Policy forward pass
                action_logits = self.policy_net(states)  # [B, 3]
                action_probs = F.softmax(action_logits, dim=-1)
                actions = torch.multinomial(action_probs, 1).squeeze()  # [B]
                log_probs = F.log_softmax(action_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze()
                
                # Apply actions: update lengths and prompts
                for i in range(B):
                    action = actions[i].item()
                    if action == 0 and lengths[i] > 0:  # remove
                        lengths[i] -= 1
                    elif action == 2 and lengths[i] < max_prompt_len:  # add
                        if mode == "continuous":
                            prompt_embeds.data[i, lengths[i]] = torch.randn(1, D, device=device) * 0.1
                        else:
                            prompt_tokens[i, lengths[i]] = self.agent.get_random_token()
                        lengths[i] += 1
                
                # Compute rewards
                rewards = alpha * likelihoods - beta * lengths.float()
                
                # Update best
                for i in range(B):
                    if rewards[i] > best_rewards[i]:
                        best_rewards[i] = rewards[i]
                        if mode == "continuous":
                            best_prompts[i] = self._embeddings_to_tokens(prompt_embeds[i, :lengths[i]].detach())
                        else:
                            best_prompts[i] = prompt_tokens[i, :lengths[i]].clone()
                
                episode_rewards.append(rewards)
                episode_log_probs.append(log_probs)
                episode_states.append(states)
            
            # Policy update (REINFORCE)
            rewards_tensor = torch.stack(episode_rewards)  # [T, B]
            log_probs_tensor = torch.stack(episode_log_probs)  # [T, B]
            
            # Compute returns
            returns = torch.zeros_like(rewards_tensor)
            next_return = torch.zeros(B, device=device)
            for t in reversed(range(steps_per_episode)):
                next_return = rewards_tensor[t] + 0.99 * next_return
                returns[t] = next_return
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Policy loss
            policy_loss = -(log_probs_tensor * returns.detach()).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            traces.append({
                'episode': episode,
                'rewards': [float(r) for r in rewards_tensor[-1]],
                'lengths': [int(l) for l in lengths]
            })
        
        # Convert best prompts
        final_prompts = []
        for i in range(B):
            if best_prompts[i] is not None:
                final_prompts.append(best_prompts[i])
            else:
                final_prompts.append(torch.tensor([], dtype=torch.long, device=device))
        
        return final_prompts, [float(r) for r in best_rewards], traces
