"""
Proof of Concept: RL-based Prompt Length Optimization
Given a target completion, find the shortest prompt that maximizes P(completion | prompt)
"""

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from tqdm import trange
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from collections import deque
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import csv
import matplotlib.pyplot as plt

class PromptRLAgent:
    def __init__(self, model_name="EleutherAI/pythia-410m"):
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPTNeoXForCausalLM.from_pretrained(model_name)
        # device handling (CUDA, MPS, or CPU)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Track special tokens for sampling/masking
        self.special_token_ids = {
            tok for tok in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.tokenizer.bos_token_id]
            if tok is not None
        }
        
        # Vocabulary for random token sampling
        self.vocab_size = len(self.tokenizer)
        
    def get_completion_likelihood(self, prompt_tokens: List[int], completion_tokens: List[int]) -> float:
        """Calculate log P(completion | prompt)"""
        if not prompt_tokens:
            # If no prompt, just use BOS token
            input_ids = [self.tokenizer.bos_token_id] + completion_tokens
        else:
            input_ids = prompt_tokens + completion_tokens
            
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor, labels=input_tensor)
            logits = outputs.logits
            
            # Calculate likelihood only for completion tokens
            prompt_len = len(prompt_tokens) if prompt_tokens else 1  # Account for BOS
            completion_logits = logits[0, prompt_len-1:-1]  # Shift for next token prediction
            completion_targets = torch.tensor(completion_tokens, dtype=torch.long, device=self.device)
            
            # Get log probabilities
            log_probs = F.log_softmax(completion_logits, dim=-1)
            token_log_probs = log_probs.gather(1, completion_targets.unsqueeze(1)).squeeze()
            return float(token_log_probs.sum())
    
    def calculate_reward(self, prompt_tokens: List[int], completion_tokens: List[int], 
                        alpha=1.0, beta=0.1) -> float:
        """Calculate reward: α * log P(completion | prompt) - β * prompt_length"""
        likelihood = self.get_completion_likelihood(prompt_tokens, completion_tokens)
        length_penalty = len(prompt_tokens)
        return alpha * likelihood - beta * length_penalty
    
    def get_random_token(self) -> int:
        """Sample a random token from vocabulary (excluding special tokens)"""
        # Avoid special tokens like PAD, EOS, BOS
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in self.special_token_ids:
                return token

class LengthPolicyOptimizer:
    def __init__(self, agent: PromptRLAgent):
        self.agent = agent
        self.emb_dim = self.agent.model.get_input_embeddings().weight.shape[1]
        
    # Policy network that decides length changes based on current state
    # Actions: 0=REMOVE, 1=KEEP, 2=ADD (append a data-driven token)
        # State: augmented scalar features capturing length, likelihood trends, and token importance
        self.state_dim = 10  # [len, ll, step_ratio, improv_rate, steps_since_improv, min, mean, max, weakest_idx_norm, tail_avg]
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        ).to(self.agent.device)
        
        # Optimizers
        self.prompt_embeddings = None
        self.prompt_optimizer = None
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        
        # RL tracking
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_states = []
        
        # History tracking
        self.loss_history = []
        self.length_history = []
        self.likelihood_history = []
        self.action_history = []
        self.trace = []  # Structured trace for plotting
        
    def optimize_prompt(self, target_completion: str, episodes=100, steps_per_episode=50,
                       initial_prompt_length=32, lr_embeddings=0.01, lr_policy=3e-4,
                       alpha=1.0, beta=0.1, log_every=10, optimization_mode: str = "continuous",
                       gcg_top_k: int = 16, gcg_batch_size: int = 32, gcg_steps: int = 5,
                       base_prompt: Optional[str] = None) -> Tuple[List[int], float, List[float]]:
        """Train the length-adjustment policy while optimizing the prompt.

        When ``optimization_mode`` is ``"continuous"`` (default) the prompt is optimized in
        continuous embedding space. When it is ``"discrete"`` the update uses Greedy
        Coordinate Gradient (GCG) in token space as described in
        https://arxiv.org/pdf/2307.15043.
        """

        mode = optimization_mode.lower()
        if mode not in {"continuous", "discrete"}:
            raise ValueError(
                f"Unsupported optimization_mode='{optimization_mode}'. Use 'continuous' or 'discrete'."
            )
        use_continuous = mode == "continuous"

        completion_tokens = self.agent.tokenizer.encode(target_completion, add_special_tokens=False)

        # If a base prompt is provided, tokenize and cache its embeddings; we will only
        # optimize a suffix appended to this base prompt.
        if base_prompt is not None:
            if isinstance(base_prompt, str):
                base_tokens = self.agent.tokenizer.encode(base_prompt, add_special_tokens=False)
            else:
                base_tokens = list(base_prompt)
        else:
            base_tokens = []

        # store on the optimizer so helper methods can include the base when building
        # full embeddings/tokens
        self.current_base_tokens: List[int] = base_tokens
        if base_tokens:
            with torch.no_grad():
                base_tensor = torch.tensor(base_tokens, dtype=torch.long, device=self.agent.device)
                self.current_base_embeds = self.agent.model.get_input_embeddings()(base_tensor).detach()
        else:
            self.current_base_embeds = None

        print(f"Target completion: '{target_completion}'")
        print(f"Training policy for {episodes} episodes, {steps_per_episode} steps each")
        print(f"Starting length: {initial_prompt_length}, α={alpha}, β={beta}, mode={mode}")
        if mode == "discrete":
            print(f"GCG settings -> top_k={gcg_top_k}, batch_size={gcg_batch_size}, steps={gcg_steps}")
        print("-" * 60)

        best_overall_reward = float('-inf')
        best_prompt: Optional[List[int]] = None
        best_overall_likelihood = float('-inf')

        last_prompt_tokens: Optional[List[int]] = None
        last_prompt_embeds: Optional[torch.Tensor] = None

        self.trace = []
        global_step = 0

        for episode in range(episodes):
            current_length = initial_prompt_length
            if use_continuous:
                # suffix embeddings (we optimize only suffix)
                self.prompt_embeddings = nn.Parameter(
                    torch.randn(current_length, self.emb_dim, device=self.agent.device) * 0.1
                )
                self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
            else:
                # suffix tokens only
                prompt_tokens = [self.agent.get_random_token() for _ in range(current_length)]
            episode_rewards: List[float] = []
            episode_log_probs: List[torch.Tensor] = []
            episode_actions: List[int] = []
            likelihood_history = deque(maxlen=10)
            initial_likelihood: Optional[float] = None
            final_likelihood_last: Optional[float] = None
            best_episode_prompt_tokens: Optional[List[int]] = None
            best_episode_likelihood = float('-inf')

            step_bar = trange(
                steps_per_episode,
                desc=f"Episode {episode+1}/{episodes}",
                leave=False
            )
            for step in step_bar:
                # --- inner optimization / scoring ---
                if use_continuous:
                    recent_likelihoods = self._optimize_continuous_prompt(completion_tokens, steps=5)
                    if recent_likelihoods:
                        current_likelihood = float(recent_likelihoods[-1])
                    else:
                        # include base embeddings (if any) when scoring
                        if getattr(self, 'current_base_embeds', None) is not None:
                            combined_embeds = torch.cat([self.current_base_embeds, self.prompt_embeddings.detach()], dim=0)
                        else:
                            combined_embeds = self.prompt_embeddings.detach()
                        current_likelihood = float(self._get_likelihood_from_embeddings(combined_embeds, completion_tokens))
                else:
                    prompt_tokens, recent_likelihoods = self._run_gcg_updates(
                        prompt_tokens,
                        completion_tokens,
                        steps=max(1, gcg_steps),
                        top_k=gcg_top_k,
                        batch_size=gcg_batch_size
                    )
                    if recent_likelihoods:
                        current_likelihood = float(recent_likelihoods[-1])
                    else:
                        combined_tokens = (self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []) + prompt_tokens
                        current_likelihood = self.agent.get_completion_likelihood(combined_tokens, completion_tokens)
                likelihood_history.append(current_likelihood)
                if initial_likelihood is None:
                    initial_likelihood = float(current_likelihood)

                if len(recent_likelihoods) >= 3:
                    improvement_rate = float(recent_likelihoods[-1] - recent_likelihoods[0])
                    recent_improvements = [
                        float(recent_likelihoods[i] - recent_likelihoods[i - 1])
                        for i in range(1, len(recent_likelihoods))
                    ]
                    improvement_trend = float(np.mean(recent_improvements[-3:])) if len(recent_improvements) >= 3 else 0.0
                else:
                    improvement_rate = 0.0
                    improvement_trend = 0.0

                steps_since_improvement = 0
                improvement_threshold = 0.1
                for i, past_likelihood in enumerate(reversed(list(likelihood_history))):
                    if current_likelihood - past_likelihood > improvement_threshold:
                        break
                    steps_since_improvement = i + 1

                importance_stats = self._compute_importance_stats(
                    use_continuous=use_continuous,
                    completion_tokens=completion_tokens,
                    prompt_embeddings=self.prompt_embeddings if use_continuous else None,
                    prompt_tokens=prompt_tokens if not use_continuous else None
                )

                with torch.no_grad():
                    state_vector = [
                        current_length,
                        current_likelihood,
                        step / steps_per_episode,
                        improvement_rate,
                        steps_since_improvement,
                        importance_stats['min_score'],
                        importance_stats['mean_score'],
                        importance_stats['max_score'],
                        importance_stats['weakest_pos_norm'],
                        importance_stats['tail_avg_score']
                    ]
                    state_features = torch.tensor(state_vector, device=self.agent.device, dtype=torch.float32)

                policy_logits = self.policy_net(state_features)
                policy_probs = F.softmax(policy_logits, dim=-1)
                policy_dist = torch.distributions.Categorical(policy_probs)
                action = policy_dist.sample()
                print("action =", action)
                log_prob = policy_dist.log_prob(action)

                action_val = int(action.item())
                new_length = current_length

                if action_val == 0 and current_length > 1:
                    if use_continuous:
                        self.prompt_embeddings = nn.Parameter(
                            self.prompt_embeddings[:-1].clone().detach().requires_grad_(True)
                        )
                        self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
                    else:
                        prompt_tokens.pop()
                    new_length = current_length - 1
                elif action_val == 2:
                    if use_continuous:
                        current_embeds = self.prompt_embeddings.detach() if self.prompt_embeddings is not None else None
                        # Combine fixed base embeddings (if present) with current suffix embeddings
                        if getattr(self, 'current_base_embeds', None) is not None:
                            if current_embeds is not None and current_embeds.numel() > 0:
                                combined_base = torch.cat([self.current_base_embeds, current_embeds], dim=0)
                            else:
                                combined_base = self.current_base_embeds
                        else:
                            combined_base = current_embeds
                        new_token_embed = self._initialize_continuous_append(
                            base_embeds=combined_base,
                            completion_tokens=completion_tokens
                        )
                        append_embed = new_token_embed.unsqueeze(0)
                        if current_embeds is None or current_embeds.shape[0] == 0:
                            updated = append_embed
                        else:
                            updated = torch.cat([current_embeds, append_embed], dim=0)
                        self.prompt_embeddings = nn.Parameter(updated.clone().detach().requires_grad_(True))
                        self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
                    else:
                        new_token = self._select_append_token(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            top_k=gcg_top_k
                        )
                        prompt_tokens.append(int(new_token))
                    new_length = current_length + 1

                current_length = new_length

                if use_continuous:
                    with torch.no_grad():
                        if getattr(self, 'current_base_embeds', None) is not None:
                            combined_embeds = torch.cat([self.current_base_embeds, self.prompt_embeddings.detach()], dim=0)
                        else:
                            combined_embeds = self.prompt_embeddings.detach()
                        final_likelihood = float(self._get_likelihood_from_embeddings(combined_embeds, completion_tokens))
                else:
                    # combine base + suffix tokens when scoring
                    combined_tokens = (self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []) + prompt_tokens
                    final_likelihood = self.agent.get_completion_likelihood(combined_tokens, completion_tokens)
                final_likelihood_last = float(final_likelihood)
                if final_likelihood_last > best_episode_likelihood:
                    best_episode_likelihood = final_likelihood_last
                    if use_continuous:
                        # Build best prompt as a list of token ids (base ids + suffix ids)
                        base_ids = self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []
                        suffix_ids = self._embeddings_to_tokens(self.prompt_embeddings.detach()) if getattr(self, 'prompt_embeddings', None) is not None else []
                        best_episode_prompt_tokens = base_ids + suffix_ids
                    else:
                        best_episode_prompt_tokens = (self.current_base_tokens if self.current_base_tokens else []) + prompt_tokens.copy()

                base_reward = alpha * final_likelihood - beta * current_length
                # print("final likelihood:", final_likelihood)
                # print("current length:", current_length)
                # print("base reward:", base_reward)
                discovery_bonus = 0.0
                if action_val == 0:
                    discovery_bonus += 0.1

                if current_length < initial_prompt_length and final_likelihood > -1.0:
                    discovery_bonus += (initial_prompt_length - current_length) * 0.05

                reward = base_reward + discovery_bonus

                episode_rewards.append(float(reward))
                episode_log_probs.append(log_prob)
                episode_actions.append(action_val)

                is_best = reward > best_overall_reward
                if is_best:
                    best_overall_reward = float(reward)
                    if use_continuous:
                        best_prompt = (self.current_base_tokens if self.current_base_tokens else []) + self._embeddings_to_tokens(self.prompt_embeddings.detach())
                    else:
                        best_prompt = (self.current_base_tokens if self.current_base_tokens else []) + prompt_tokens.copy()

                self.likelihood_history.append(float(current_likelihood))
                self.length_history.append(current_length)
                self.action_history.append(action_val)

                if current_likelihood > best_overall_likelihood:
                    best_overall_likelihood = float(current_likelihood)

                self.trace.append({
                    'step': global_step,
                    'episode': episode,
                    'likelihood': float(current_likelihood),
                    'best_likelihood': float(best_overall_likelihood),
                    'length': current_length,
                    'action': action_val,
                    'reward': float(reward),
                    'improved': is_best,
                    'importance_min': float(importance_stats['min_score']),
                    'importance_mean': float(importance_stats['mean_score']),
                    'importance_max': float(importance_stats['max_score'])
                })
                global_step += 1

                step_bar.set_postfix({
                    'step': step,
                    'len': current_length,
                    'll': f"{final_likelihood_last:.2f}",
                    'reward': f"{reward:.2f}"
                })

                if log_every > 0 and episode % log_every == 0 and step % 10 == 0:
                    print(
                        f"Ep {episode:3d} Step {step:2d}: action={action_val} length={current_length} "
                        f"likelihood={current_likelihood:.3f} improv_rate={improvement_rate:.4f} "
                        f"since_improv={steps_since_improvement} reward={reward:.3f}"
                    )

            if use_continuous:
                last_prompt_embeds = self.prompt_embeddings.detach().clone()
            else:
                last_prompt_tokens = prompt_tokens.copy()

            episode_return = sum(episode_rewards)
            returns = []
            G = 0.0
            for r in reversed(episode_rewards):
                G = r + G
                returns.insert(0, G)

            returns = torch.tensor(returns, device=self.agent.device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            policy_loss = 0.0
            for log_prob, ret in zip(episode_log_probs, returns):
                policy_loss = policy_loss - log_prob * ret

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            if log_every > 0 and episode % log_every == 0:
                recent_lengths = self.length_history[-steps_per_episode:]
                avg_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else current_length
                print(
                    f"Episode {episode}: return={episode_return:.2f} avg_length={avg_length:.1f} "
                    f"best_reward={best_overall_reward:.3f}"
                )

            # For episode summary, always compute the full prompt (base + suffix).
            # Ensure we define `episode_tokens` regardless of logging frequency so the
            # later summary printing does not reference an uninitialized variable.
            if use_continuous:
                episode_suffix_tokens = self._embeddings_to_tokens(self.prompt_embeddings.detach()) if getattr(self, 'prompt_embeddings', None) is not None else []
            else:
                # prompt_tokens should exist for discrete mode, but guard defensively
                episode_suffix_tokens = prompt_tokens.copy() if 'prompt_tokens' in locals() and prompt_tokens is not None else []

            episode_tokens = (self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []) + episode_suffix_tokens
            episode_text = self.agent.tokenizer.decode(episode_tokens, skip_special_tokens=True) if episode_tokens else ""
            print(f"Episode {episode}: final prompt text=\"{episode_text}\"")
            final_ll = final_likelihood_last if final_likelihood_last is not None else 0.0
            final_avg = final_ll / max(len(episode_tokens), 1)
            print(
                f"Episode {episode}: log P(target | final prompt)={final_ll:.3f} "
                f"avg log prob per token={final_avg:.3f}"
            )

            if best_episode_prompt_tokens is not None:
                # best_episode_prompt_tokens should be a list of token ids; decode them for display
                try:
                    best_episode_text = self.agent.tokenizer.decode(best_episode_prompt_tokens, skip_special_tokens=True)
                except Exception:
                    # Fallback: join string representations if decode fails
                    best_episode_text = "".join(str(t) for t in best_episode_prompt_tokens)
                best_avg = best_episode_likelihood / max(len(best_episode_prompt_tokens), 1)
                print(f"Episode {episode}: best prompt text=\"{best_episode_text}\"")
                print(
                    f"Episode {episode}: best log P(target | prompt)={best_episode_likelihood:.3f} "
                    f"best avg log prob per token={best_avg:.3f}"
                )

        # clear base cache
        self.current_base_tokens = []
        self.current_base_embeds = None

        if best_prompt is None:
            if use_continuous and last_prompt_embeds is not None:
                best_prompt = self._embeddings_to_tokens(last_prompt_embeds)
            elif not use_continuous and last_prompt_tokens is not None:
                best_prompt = last_prompt_tokens
            else:
                best_prompt = []

        return best_prompt, float(best_overall_reward), self.trace

    def _optimize_continuous_prompt(self, completion_tokens: List[int], steps: int = 5) -> List[float]:
        """Run gradient steps on the continuous prompt embeddings and collect likelihoods."""
        likelihoods: List[float] = []
        for _ in range(max(1, steps)):
            self.prompt_optimizer.zero_grad()
            # include base embeddings (if any) when calculating likelihood
            if getattr(self, 'current_base_embeds', None) is not None:
                combined = torch.cat([self.current_base_embeds, self.prompt_embeddings], dim=0)
            else:
                combined = self.prompt_embeddings
            likelihood = self._get_likelihood_from_embeddings(combined, completion_tokens)
            likelihoods.append(float(likelihood))
            (-likelihood).backward()
            self.prompt_optimizer.step()
        return likelihoods

    def _run_gcg_updates(self, prompt_tokens: List[int], completion_tokens: List[int], *, steps: int,
                          top_k: int, batch_size: int) -> Tuple[List[int], List[float]]:
        """Apply multiple GCG steps in token space and return updated tokens and likelihood trace."""
        updated_tokens = prompt_tokens
        likelihoods: List[float] = []
        for _ in range(max(1, steps)):
            updated_tokens, likelihood = self._gcg_step(updated_tokens, completion_tokens, top_k=top_k, batch_size=batch_size)
            likelihoods.append(float(likelihood))
        return updated_tokens, likelihoods

    def _compute_importance_stats(self, *, use_continuous: bool, completion_tokens: List[int],
                                   prompt_embeddings: Optional[torch.Tensor] = None,
                                   prompt_tokens: Optional[List[int]] = None,
                                   tail_k: int = 3) -> Dict[str, float]:
        """Estimate token importance via gradient norms and return summary statistics."""
        if use_continuous:
            if prompt_embeddings is None or prompt_embeddings.shape[0] == 0:
                return {
                    'min_score': 0.0,
                    'mean_score': 0.0,
                    'max_score': 0.0,
                    'weakest_pos_norm': 0.0,
                    'tail_avg_score': 0.0
                }
            temp_embeds = prompt_embeddings.detach().clone().requires_grad_(True)
            # prepend base embeds if present so the gradient reflects base+suffix
            if getattr(self, 'current_base_embeds', None) is not None:
                combined = torch.cat([self.current_base_embeds, temp_embeds], dim=0)
            else:
                combined = temp_embeds
            self.agent.model.zero_grad(set_to_none=True)
            likelihood = self._get_likelihood_from_embeddings(combined, completion_tokens)
            (-likelihood).backward()
            grads = temp_embeds.grad.detach()
        else:
            if not prompt_tokens:
                return {
                    'min_score': 0.0,
                    'mean_score': 0.0,
                    'max_score': 0.0,
                    'weakest_pos_norm': 0.0,
                    'tail_avg_score': 0.0
                }
            embedding_layer = self.agent.model.get_input_embeddings()
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=self.agent.device)
            prompt_embeds = embedding_layer(prompt_tensor).detach().requires_grad_(True)
            completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=self.agent.device)
            completion_embeds = embedding_layer(completion_tensor)
            # include base embeds if present
            if getattr(self, 'current_base_embeds', None) is not None:
                base = self.current_base_embeds
                full_embeds = torch.cat([base, prompt_embeds, completion_embeds], dim=0).unsqueeze(0)
                prompt_len = base.shape[0] + prompt_embeds.shape[0]
            else:
                full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=0).unsqueeze(0)
                prompt_len = prompt_embeds.shape[0]
            self.agent.model.zero_grad(set_to_none=True)
            outputs = self.agent.model.gpt_neox(inputs_embeds=full_embeds)
            hidden_states = outputs.last_hidden_state
            logits = self.agent.model.embed_out(hidden_states)
            completion_logits = logits[0, prompt_len-1:-1]
            log_probs = F.log_softmax(completion_logits, dim=-1)
            target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
            (-target_log_probs.sum()).backward()
            grads = prompt_embeds.grad.detach()

        if grads is None or grads.numel() == 0:
            return {
                'min_score': 0.0,
                'mean_score': 0.0,
                'max_score': 0.0,
                'weakest_pos_norm': 0.0,
                'tail_avg_score': 0.0
            }

        scores = torch.norm(grads, dim=1)
        length = scores.shape[0]
        min_score = float(scores.min().item())
        mean_score = float(scores.mean().item())
        max_score = float(scores.max().item())
        weakest_idx = int(torch.argmin(scores).item())
        norm_pos = weakest_idx / max(length, 1)
        tail_count = min(tail_k, length)
        if tail_count > 0:
            tail_avg = float(scores.topk(tail_count, largest=False).values.mean().item())
        else:
            tail_avg = 0.0

        self.agent.model.zero_grad(set_to_none=True)

        return {
            'min_score': min_score,
            'mean_score': mean_score,
            'max_score': max_score,
            'weakest_pos_norm': float(norm_pos),
            'tail_avg_score': tail_avg
        }

    def _initialize_continuous_append(self, *, base_embeds: Optional[torch.Tensor],
                                       completion_tokens: List[int], ascent_steps: int = 6,
                                       step_size: float = 0.05) -> torch.Tensor:
        """Craft an appended embedding by following the gradient of the likelihood objective."""
        device = self.agent.device
        if base_embeds is not None and base_embeds.numel() > 0:
            base = base_embeds.detach()
            start = base.mean(dim=0)
        else:
            base = None
            start = torch.zeros(self.emb_dim, device=device)
        candidate = (start + 0.01 * torch.randn_like(start)).detach().requires_grad_(True)

        for _ in range(max(1, ascent_steps)):
            self.agent.model.zero_grad(set_to_none=True)
            if base is None:
                combined = candidate.unsqueeze(0)
            else:
                combined = torch.cat([base, candidate.unsqueeze(0)], dim=0)
            likelihood = self._get_likelihood_from_embeddings(combined, completion_tokens)
            grad = torch.autograd.grad(likelihood, candidate, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                candidate += step_size * grad
            candidate = candidate.detach().requires_grad_(True)

        self.agent.model.zero_grad(set_to_none=True)
        return candidate.detach()

    def _select_append_token(self, *, prompt_tokens: List[int], completion_tokens: List[int],
                              top_k: int) -> int:
        """Select a new token to append using the gradient signal of an inserted slot."""
        device = self.agent.device
        embedding_layer = self.agent.model.get_input_embeddings()

        if prompt_tokens:
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            prompt_embeds = embedding_layer(prompt_tensor).detach()
        else:
            prompt_tensor = None
            prompt_embeds = torch.empty(0, self.emb_dim, device=device)

        completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=device)
        completion_embeds = embedding_layer(completion_tensor).detach()

        candidate_embed = torch.zeros(self.emb_dim, device=device).detach().requires_grad_(True)

        # include base embeds at front if present
        if getattr(self, 'current_base_embeds', None) is not None:
            concat_parts = [self.current_base_embeds, prompt_embeds, candidate_embed.unsqueeze(0), completion_embeds]
        else:
            concat_parts = [prompt_embeds, candidate_embed.unsqueeze(0), completion_embeds]
        full_embeds = torch.cat(concat_parts, dim=0).unsqueeze(0)

        self.agent.model.zero_grad(set_to_none=True)
        outputs = self.agent.model.gpt_neox(inputs_embeds=full_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.agent.model.embed_out(hidden_states)

        base_len = (len(self.current_base_tokens) if getattr(self, 'current_base_tokens', None) else 0)
        prompt_len = base_len + (prompt_tensor.shape[0] if prompt_tensor is not None else 0)
        completion_logits = logits[0, prompt_len-1:-1]
        log_probs = F.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
        (-target_log_probs.sum()).backward()

        grad = candidate_embed.grad.detach()
        vocab_embeds = embedding_layer.weight.detach()
        scores = torch.matmul(vocab_embeds, -grad)

        self.agent.model.zero_grad(set_to_none=True)

        k = max(1, min(top_k, scores.shape[0]))
        top_indices = torch.topk(scores, k=k, largest=True).indices
        candidates = [int(idx.item()) for idx in top_indices if idx.item() not in self.agent.special_token_ids]
        if not candidates:
            candidates = [int(top_indices[0].item())]

        return random.choice(candidates)

    def _gcg_step(self, prompt_tokens: List[int], completion_tokens: List[int], *, top_k: int, batch_size: int) -> Tuple[List[int], float]:
        """Perform one Greedy Coordinate Gradient update in token space."""
        if not prompt_tokens:
            likelihood = self.agent.get_completion_likelihood(prompt_tokens, completion_tokens)
            return prompt_tokens, likelihood

        device = self.agent.device
        embedding_layer = self.agent.model.get_input_embeddings()

        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        prompt_embeds = embedding_layer(prompt_tensor).detach().requires_grad_(True)

        completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=device)
        completion_embeds = embedding_layer(completion_tensor)

        # include base embeddings if present
        if getattr(self, 'current_base_embeds', None) is not None:
            base = self.current_base_embeds
            full_embeds = torch.cat([base, prompt_embeds, completion_embeds], dim=0).unsqueeze(0)
            prompt_len = base.shape[0] + prompt_embeds.shape[0]
        else:
            full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=0).unsqueeze(0)
            prompt_len = prompt_embeds.shape[0]

        self.agent.model.zero_grad(set_to_none=True)
        outputs = self.agent.model.gpt_neox(inputs_embeds=full_embeds)
        hidden_states = outputs.last_hidden_state
        logits = self.agent.model.embed_out(hidden_states)
        # prompt_len already set above when base was considered
        completion_logits = logits[0, prompt_len-1:-1]
        log_probs = F.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
        loss = -target_log_probs.sum()
        loss.backward()

        grads = prompt_embeds.grad
        vocab_embeds = embedding_layer.weight.detach()

        candidate_sets: List[List[int]] = []
        # grads correspond only to the prompt_embeds (suffix) positions — not the base.
        suffix_len = prompt_embeds.shape[0]
        for i in range(suffix_len):
            grad_i = grads[i]
            scores = torch.matmul(vocab_embeds, -grad_i)
            k = min(top_k, scores.shape[0])
            topk_idx = torch.topk(scores, k=k, largest=True).indices
            filtered = [int(tok) for tok in topk_idx.tolist() if tok not in self.agent.special_token_ids]
            if not filtered:
                filtered = [int(tok) for tok in topk_idx.tolist()]
            candidate_sets.append(filtered)

        # base_likelihood should score base + suffix
        combined_tokens = (self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []) + prompt_tokens
        base_likelihood = self.agent.get_completion_likelihood(combined_tokens, completion_tokens)
        best_tokens = prompt_tokens.copy()
        best_likelihood = base_likelihood

        # Only consider indices within the suffix (prompt_tokens) when proposing swaps
        indices = list(range(suffix_len))
        for _ in range(max(1, batch_size)):
            if not indices:
                break
            position = random.choice(indices)
            candidates = candidate_sets[position]
            if not candidates:
                continue
            candidate_token = random.choice(candidates)
            if candidate_token == prompt_tokens[position]:
                continue
            candidate_prompt = prompt_tokens.copy()
            candidate_prompt[position] = candidate_token
            candidate_combined = (self.current_base_tokens if getattr(self, 'current_base_tokens', None) else []) + candidate_prompt
            likelihood = self.agent.get_completion_likelihood(candidate_combined, completion_tokens)
            if likelihood > best_likelihood:
                best_likelihood = likelihood
                best_tokens = candidate_prompt

        return best_tokens, best_likelihood

    def _get_likelihood_from_embeddings(self, prompt_embeds: torch.Tensor, completion_tokens: List[int]) -> torch.Tensor:
        """Calculate log P(completion | continuous prompt embeddings)"""
        # Convert completion tokens to embeddings
        completion_tensor = torch.tensor(completion_tokens, dtype=torch.long, device=self.agent.device)
        completion_embeds = self.agent.model.get_input_embeddings()(completion_tensor)
        
        # Concatenate prompt embeddings + completion embeddings
        full_embeds = torch.cat([prompt_embeds, completion_embeds], dim=0).unsqueeze(0)  # [1, seq_len, emb_dim]
        
        # Forward pass through transformer
        # We need to bypass the embedding layer and feed embeddings directly
        outputs = self.agent.model.gpt_neox(inputs_embeds=full_embeds)
        hidden_states = outputs.last_hidden_state
        
        # Get logits for completion positions only
        logits = self.agent.model.embed_out(hidden_states)  # [1, seq_len, vocab_size]
        prompt_len = prompt_embeds.shape[0]
        completion_logits = logits[0, prompt_len-1:-1]  # Shift for next token prediction
        
        # Calculate log probabilities for target completion tokens
        log_probs = F.log_softmax(completion_logits, dim=-1)
        target_log_probs = log_probs.gather(1, completion_tensor.unsqueeze(1)).squeeze()
        
        return target_log_probs.sum()
    
    def _embeddings_to_tokens(self, embeddings: torch.Tensor) -> List[int]:
        """Convert embeddings back to discrete tokens by finding nearest neighbors"""
        # Get all vocabulary embeddings
        vocab_embeds = self.agent.model.get_input_embeddings().weight  # [vocab_size, emb_dim]
        
        tokens = []
        for emb in embeddings:
            # Find closest token embedding
            distances = torch.norm(vocab_embeds - emb.unsqueeze(0), dim=1)
            closest_token = int(torch.argmin(distances).item())
            tokens.append(closest_token)
        
        return tokens

# Main function removed - use train.py and eval.py instead

if __name__ == "__main__":
    print("This module contains the core classes PromptRLAgent and LengthPolicyOptimizer.")
    print("Use train.py to train a model and eval.py to evaluate.")
    print("Example:")
    print("  python train.py --episodes 100")
    print("  python eval.py --test_prompt 'Your test text here'")
