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
            # Prefer a single GPU (device 0) for predictable scaling
            try:
                torch.cuda.set_device(0)
            except Exception:
                pass
            self.device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        # Enable cuDNN autotuner for fixed-size inputs to improve throughput
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
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

        # More expressive policy network: deeper MLP with LayerNorm and dropout
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        ).to(self.agent.device)

        # Value network for baseline (actor-critic style)
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.agent.device)

        # Optimizers
        self.prompt_embeddings = None
        self.prompt_optimizer = None
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        
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
                       base_prompt: Optional[str] = None,
                       # PPO params (if enabled)
                       use_ppo: bool = True,
                       ppo_epochs: int = 4,
                       ppo_clip: float = 0.2,
                       gamma: float = 0.99,
                       gae_lambda: float = 0.95,
                       value_coef: float = 0.5,
                       entropy_coef: float = 0.01) -> Tuple[List[int], float, List[float]]:
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
        # Update policy/value learning rates if caller overrides them
        if lr_policy is not None:
            for pg in self.policy_optimizer.param_groups:
                pg['lr'] = lr_policy
            for pg in self.value_optimizer.param_groups:
                pg['lr'] = lr_policy

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
            episode_states: List[torch.Tensor] = []
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
                # store state for later value estimation / advantage calculation
                episode_states.append(state_features)

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

            # Compute value estimates and advantages (either PPO with GAE or vanilla actor-critic)
            device = self.agent.device

            if len(episode_states) > 0:
                states_batch = torch.stack(episode_states)  # [T, state_dim]
                T = states_batch.shape[0]

                # values from critic (no grad needed for computing advantages)
                with torch.no_grad():
                    values = self.value_net(states_batch).squeeze()  # [T]

                # Convert rewards to tensor on same device
                rewards_tensor = torch.tensor(episode_rewards, dtype=torch.float32, device=device)

                if use_ppo:
                    # GAE advantage estimation
                    advantages = torch.zeros(T, dtype=torch.float32, device=device)
                    last_gae = 0.0
                    # bootstrap with 0 for episode end
                    for t in reversed(range(T)):
                        if t == T - 1:
                            next_value = 0.0
                        else:
                            next_value = float(values[t + 1].item())
                        delta = float(rewards_tensor[t].item()) + gamma * next_value - float(values[t].item())
                        last_gae = delta + gamma * gae_lambda * last_gae
                        advantages[t] = last_gae

                    returns_tensor = advantages + values.detach()
                    # normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # old log probs and actions
                    old_log_probs = torch.stack(episode_log_probs).detach().to(device)
                    actions_tensor = torch.tensor(episode_actions, dtype=torch.long, device=device)

                    # PPO epochs (simple full-batch updates)
                    for _ in range(max(1, ppo_epochs)):
                        policy_logits = self.policy_net(states_batch)
                        policy_probs = F.softmax(policy_logits, dim=-1)
                        policy_dist = torch.distributions.Categorical(policy_probs)
                        new_log_probs = policy_dist.log_prob(actions_tensor)
                        entropy = policy_dist.entropy().mean()

                        ratios = torch.exp(new_log_probs - old_log_probs)
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
                        policy_loss = -torch.mean(torch.min(surr1, surr2))

                        # value loss
                        value_preds = self.value_net(states_batch).squeeze()
                        value_loss = F.mse_loss(value_preds, returns_tensor)

                        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                        self.policy_optimizer.zero_grad()
                        self.value_optimizer.zero_grad()
                        total_loss.backward()
                        self.policy_optimizer.step()
                        self.value_optimizer.step()
                else:
                    # Vanilla actor-critic (REINFORCE-with-baseline)
                    returns = returns.to(values.device)
                    advantages = returns - values.detach()

                    policy_loss = torch.tensor(0.0, device=device)
                    for log_prob, adv in zip(episode_log_probs, advantages):
                        policy_loss = policy_loss - log_prob * adv

                    value_loss = F.mse_loss(values, returns)

                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss = policy_loss + 0.5 * value_loss
                    total_loss.backward()
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
            else:
                # fallback to REINFORCE if no stored states
                policy_loss = torch.tensor(0.0, device=device)
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

    def _optimize_continuous_prompt(
        self,
        completion_tokens: List[int],
        steps: int = 5,
        prompt_embeddings: Optional[torch.Tensor] = None,
        prompt_optimizer: Optional[optim.Optimizer] = None,
        base_embeds: Optional[torch.Tensor] = None
    ) -> List[float]:
        """Run gradient steps on continuous prompt embeddings and collect likelihoods."""
        target_embeddings = prompt_embeddings if prompt_embeddings is not None else self.prompt_embeddings
        target_optimizer = prompt_optimizer if prompt_optimizer is not None else self.prompt_optimizer
        base_prefix = base_embeds if base_embeds is not None else getattr(self, 'current_base_embeds', None)
        if target_embeddings is None or target_optimizer is None:
            return []
        likelihoods: List[float] = []
        for _ in range(max(1, steps)):
            target_optimizer.zero_grad()
            # include base embeddings (if any) when calculating likelihood
            if base_prefix is not None and base_prefix.numel() > 0:
                combined = torch.cat([base_prefix, target_embeddings], dim=0)
            else:
                combined = target_embeddings
            likelihood = self._get_likelihood_from_embeddings(combined, completion_tokens)
            likelihoods.append(float(likelihood))
            (-likelihood).backward()
            target_optimizer.step()
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
                                   tail_k: int = 3,
                                   base_embeds: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Estimate token importance via gradient norms and return summary statistics."""
        if base_embeds is None:
            base_embeds = getattr(self, 'current_base_embeds', None)
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
            if base_embeds is not None and base_embeds.numel() > 0:
                combined = torch.cat([base_embeds, temp_embeds], dim=0)
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
            if base_embeds is not None and base_embeds.numel() > 0:
                full_embeds = torch.cat([base_embeds, prompt_embeds, completion_embeds], dim=0).unsqueeze(0)
                prompt_len = base_embeds.shape[0] + prompt_embeds.shape[0]
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

    # ---------------- Batched continuous-mode optimizer ----------------
    def _get_likelihoods_from_embeddings_batch(self, prompt_embeds_batch: torch.Tensor,
                                               completion_tokens_list: List[List[int]],
                                               base_embeds_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Compute log P(completion | prompt) for a batch of continuous prompts.

        prompt_embeds_batch: [B, L, D]
        completion_tokens_list: list of length B, each a list of token ids
        base_embeds_list: optional list of length B of tensors [base_len, D]

        Returns: tensor shape [B] with summed log-probabilities for each example.
        """
        device = self.agent.device
        B, L, D = prompt_embeds_batch.shape
        embedding_layer = self.agent.model.get_input_embeddings()

        # Prepare completion embeddings and lengths
        comp_embeds_list = []
        comp_lens = []
        for comp_tokens in completion_tokens_list:
            if len(comp_tokens) == 0:
                comp_embeds = torch.empty(0, D, device=device)
            else:
                comp_tensor = torch.tensor(comp_tokens, dtype=torch.long, device=device)
                comp_embeds = embedding_layer(comp_tensor)
            comp_embeds_list.append(comp_embeds)
            comp_lens.append(comp_embeds.shape[0])

        base_lens = []
        if base_embeds_list is None:
            base_embeds_list = [None] * B
        for base in base_embeds_list:
            base_lens.append(base.shape[0] if (base is not None and base.numel() > 0) else 0)

        full_lens = [base_lens[i] + L + comp_lens[i] for i in range(B)]
        max_full = max(full_lens) if full_lens else L

        # pad embedding: use pad token embedding if available, otherwise zeros
        pad_id = getattr(self.agent.tokenizer, 'pad_token_id', None)
        if pad_id is not None:
            pad_embed = embedding_layer.weight[pad_id].detach()
        else:
            pad_embed = torch.zeros(D, device=device)

        # Build batched inputs_embeds [B, max_full, D]
        inputs_embeds = pad_embed.unsqueeze(0).unsqueeze(0).repeat(B, max_full, 1).to(device)
        for i in range(B):
            pos = 0
            if base_embeds_list[i] is not None and base_embeds_list[i].numel() > 0:
                be = base_embeds_list[i].to(device)
                bl = be.shape[0]
                inputs_embeds[i, pos:pos+bl, :] = be
                pos += bl

            # suffix embeddings
            inputs_embeds[i, pos:pos+L, :] = prompt_embeds_batch[i]
            pos += L

            # completion embeddings
            cel = comp_embeds_list[i]
            if cel.numel() > 0:
                inputs_embeds[i, pos:pos+cel.shape[0], :] = cel

        # Forward pass
        outputs = self.agent.model.gpt_neox(inputs_embeds=inputs_embeds)
        hidden_states = outputs.last_hidden_state  # [B, max_full, hidden]
        logits = self.agent.model.embed_out(hidden_states)  # [B, max_full, vocab]

        # For each example compute summed log-prob of completion tokens
        log_probs_per_example = []
        for i in range(B):
            prompt_len = base_lens[i] + L
            comp_len = comp_lens[i]
            if comp_len == 0:
                logp = torch.tensor(0.0, device=device)
            else:
                # indices: prompt_len-1 .. prompt_len-1+comp_len-1 => prompt_len-1 : prompt_len-1+comp_len
                comp_logits = logits[i, prompt_len-1: prompt_len-1+comp_len, :]
                comp_tokens = completion_tokens_list[i]
                target_tensor = torch.tensor(comp_tokens, dtype=torch.long, device=device)
                lps = F.log_softmax(comp_logits, dim=-1)
                token_log_probs = lps.gather(1, target_tensor.unsqueeze(1)).squeeze()
                logp = token_log_probs.sum()
            log_probs_per_example.append(logp)

        return torch.stack(log_probs_per_example)

    def optimize_prompts_batch(self, target_completions: List[str], episodes: int = 3,
                               steps_per_episode: int = 50, initial_prompt_length: int = 32,
                               lr_embeddings: float = 0.01, alpha: float = 1.0, beta: float = 0.1,
                               base_prompts: Optional[List[str]] = None,
                               inner_steps: int = 5) -> Tuple[List[List[int]], List[float], List[dict]]:
        """Batch optimize continuous prompt suffixes for multiple target completions.

        Returns (best_prompts_tokens_list, best_rewards_list, traces_list)
        This implementation currently supports continuous mode only.
        """
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []

        # Prepare completion token lists
        completion_tokens_list = [self.agent.tokenizer.encode(t, add_special_tokens=False) for t in target_completions]

        # Prepare base embeds list
        base_embeds_list: List[Optional[torch.Tensor]] = []
        if base_prompts is None:
            base_prompts = [None] * B
        for bp in base_prompts:
            if bp:
                btoks = self.agent.tokenizer.encode(bp, add_special_tokens=False)
                bt = torch.tensor(btoks, dtype=torch.long, device=device)
                be = self.agent.model.get_input_embeddings()(bt).detach()
                base_embeds_list.append(be)
            else:
                base_embeds_list.append(None)

        # Initialize batched suffix embeddings [B, L, D]
        D = self.emb_dim
        L = initial_prompt_length
        prompt_embeds = nn.Parameter(torch.randn(B, L, D, device=device) * 0.1)
        prompt_optimizer = optim.Adam([prompt_embeds], lr=lr_embeddings)

        best_prompts = [None] * B
        best_rewards = [float('-inf')] * B
        traces: List[dict] = []

        for ep in range(episodes):
            # inner continuous optimization on the batch
            for _ in range(max(1, inner_steps)):
                prompt_optimizer.zero_grad()
                # build combined embeddings per example: base + suffix
                # _get_likelihoods_from_embeddings_batch will handle concatenation with completions
                likelihoods = self._get_likelihoods_from_embeddings_batch(prompt_embeds, completion_tokens_list, base_embeds_list)
                # we want to maximize likelihood -> minimize negative
                loss = -likelihoods.sum()
                loss.backward()
                prompt_optimizer.step()

            # After inner steps, evaluate rewards and update bests
            with torch.no_grad():
                likelihoods = self._get_likelihoods_from_embeddings_batch(prompt_embeds, completion_tokens_list, base_embeds_list)
                for i in range(B):
                    length = L + (base_embeds_list[i].shape[0] if (base_embeds_list[i] is not None) else 0)
                    reward = alpha * float(likelihoods[i].item()) - beta * length
                    if reward > best_rewards[i]:
                        best_rewards[i] = float(reward)
                        # build full embeddings (base + suffix) and convert to tokens
                        if base_embeds_list[i] is not None:
                            combined = torch.cat([base_embeds_list[i], prompt_embeds[i].detach()], dim=0)
                        else:
                            combined = prompt_embeds[i].detach()
                        try:
                            best_prompts[i] = self._embeddings_to_tokens(combined)
                        except Exception:
                            best_prompts[i] = []

                # add a simple trace entry per episode (aggregated)
                traces.append({
                    'episode': ep,
                    'likelihoods': [float(l.item()) for l in likelihoods],
                    'best_rewards': best_rewards.copy()
                })

        return best_prompts, best_rewards, traces

    def optimize_prompts_batch_discrete(self, target_completions: List[str], episodes: int = 3,
                                        steps_per_episode: int = 50, initial_prompt_length: int = 32,
                                        gcg_top_k: int = 16, gcg_batch_size: int = 32, gcg_steps: int = 5,
                                        alpha: float = 1.0, beta: float = 0.1,
                                        base_prompts: Optional[List[str]] = None) -> Tuple[List[List[int]], List[float], List[dict]]:
        """Batched discrete (GCG) optimization for multiple prompts.

        This vectorizes the expensive model scoring of candidate prompts across the batch.
        Returns (best_prompts_tokens_list, best_rewards_list, traces_list).
        """
        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []

        embedding_layer = self.agent.model.get_input_embeddings()
        vocab_embeds = embedding_layer.weight.detach()
        vocab_size = vocab_embeds.shape[0]

        # Prepare completions and base embeddings
        completion_tokens_list = [self.agent.tokenizer.encode(t, add_special_tokens=False) for t in target_completions]
        if base_prompts is None:
            base_prompts = [None] * B
        base_embeds_list: List[Optional[torch.Tensor]] = []
        for bp in base_prompts:
            if bp:
                btoks = self.agent.tokenizer.encode(bp, add_special_tokens=False)
                bt = torch.tensor(btoks, dtype=torch.long, device=device)
                be = embedding_layer(bt).detach()
                base_embeds_list.append(be)
            else:
                base_embeds_list.append(None)

        # Initialize random prompts (token ids)
        L = initial_prompt_length
        prompt_tokens_list: List[List[int]] = []
        for _ in range(B):
            toks = [self.agent.get_random_token() for _ in range(L)]
            prompt_tokens_list.append(toks)

        best_prompts = [p.copy() for p in prompt_tokens_list]
        # initial scoring
        with torch.no_grad():
            # build prompt_embeds batch
            prompt_ids = torch.tensor(prompt_tokens_list, dtype=torch.long, device=device)
            prompt_embeds_batch = embedding_layer(prompt_ids)  # [B, L, D]
            likelihoods = self._get_likelihoods_from_embeddings_batch(prompt_embeds_batch, completion_tokens_list, base_embeds_list)
            best_likelihoods = [float(l.item()) for l in likelihoods]
            best_rewards = [alpha * bl - beta * (L + (base_embeds_list[i].shape[0] if base_embeds_list[i] is not None else 0)) for i, bl in enumerate(best_likelihoods)]

        traces: List[dict] = []

        # GCG iterations
        for ep in range(episodes):
            for step in range(max(1, gcg_steps)):
                # compute gradients w.r.t. prompt embeddings for candidate generation
                # Build prompt embeddings requiring grad
                prompt_ids = torch.tensor(prompt_tokens_list, dtype=torch.long, device=device)
                prompt_embeds = embedding_layer(prompt_ids).detach().requires_grad_(True)  # [B, L, D]

                # Build full batch inputs_embeds
                # reuse helper by passing prompt_embeds as-is and base/completion lists
                # compute negative log-prob sum and backward
                likelihoods = self._get_likelihoods_from_embeddings_batch(prompt_embeds, completion_tokens_list, base_embeds_list)
                loss = -likelihoods.sum()
                # backward to get grads on prompt_embeds
                self.agent.model.zero_grad(set_to_none=True)
                loss.backward()

                grads = prompt_embeds.grad.detach()  # [B, L, D]

                # Build candidate sets per example per position (top_k)
                candidate_sets: List[List[List[int]]] = []  # [B][pos] -> list of tokens
                for i in range(B):
                    sets_for_example: List[List[int]] = []
                    for pos in range(L):
                        grad_i = grads[i, pos]  # [D]
                        scores = torch.matmul(vocab_embeds, -grad_i)  # [V]
                        k = min(gcg_top_k, scores.shape[0])
                        topk_idx = torch.topk(scores, k=k, largest=True).indices
                        filtered = [int(tok.item()) for tok in topk_idx if int(tok.item()) not in self.agent.special_token_ids]
                        if not filtered:
                            filtered = [int(topk_idx[0].item())]
                        sets_for_example.append(filtered)
                    candidate_sets.append(sets_for_example)

                # Sample candidate swaps per example and score them in a single batched forward
                candidates_flat: List[List[int]] = []  # list of prompt token lists
                owner_idx: List[int] = []  # which example each candidate belongs to
                for i in range(B):
                    for _ in range(max(1, gcg_batch_size)):
                        # choose a random position to modify
                        pos = random.randrange(0, L)
                        cands = candidate_sets[i][pos]
                        if not cands:
                            continue
                        cand_token = random.choice(cands)
                        if cand_token == prompt_tokens_list[i][pos]:
                            continue
                        cand_prompt = prompt_tokens_list[i].copy()
                        cand_prompt[pos] = cand_token
                        candidates_flat.append(cand_prompt)
                        owner_idx.append(i)

                if not candidates_flat:
                    # nothing to try
                    continue

                # Score all candidates in smaller chunks to avoid OOM
                cand_completion_lists = [completion_tokens_list[owner_idx[j]] for j in range(len(owner_idx))]
                cand_base_embeds = [base_embeds_list[owner_idx[j]] for j in range(len(owner_idx))]

                cand_likelihoods_list: List[torch.Tensor] = []
                # start with a moderate chunk size and reduce on OOM
                chunk_size = 32
                if len(candidates_flat) < chunk_size:
                    chunk_size = len(candidates_flat)

                start_idx = 0
                while start_idx < len(candidates_flat):
                    end_idx = min(start_idx + chunk_size, len(candidates_flat))
                    try:
                        chunk_ids = torch.tensor(candidates_flat[start_idx:end_idx], dtype=torch.long, device=device)
                        chunk_embeds_batch = embedding_layer(chunk_ids)
                        chunk_comp_lists = cand_completion_lists[start_idx:end_idx]
                        chunk_base_embeds = cand_base_embeds[start_idx:end_idx]
                        with torch.no_grad():
                            chunk_ll = self._get_likelihoods_from_embeddings_batch(chunk_embeds_batch, chunk_comp_lists, chunk_base_embeds)
                        cand_likelihoods_list.append(chunk_ll)
                        # cleanup
                        del chunk_ids, chunk_embeds_batch, chunk_comp_lists, chunk_base_embeds, chunk_ll
                        start_idx = end_idx
                    except RuntimeError as e:
                        # likely OOM; try reducing chunk size
                        if 'out of memory' in str(e).lower() and chunk_size > 1:
                            chunk_size = max(1, chunk_size // 2)
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise

                if cand_likelihoods_list:
                    cand_likelihoods = torch.cat(cand_likelihoods_list)
                else:
                    cand_likelihoods = torch.empty(0, device=device)

                # For each example, find best improving candidate among its block
                # Map owner -> list of (candidate_idx_in_flat, likelihood)
                per_owner_best: Dict[int, Tuple[int, float]] = {}
                for idx_cand, owner in enumerate(owner_idx):
                    ll = float(cand_likelihoods[idx_cand].item())
                    if owner not in per_owner_best or ll > per_owner_best[owner][1]:
                        per_owner_best[owner] = (idx_cand, ll)

                # Apply improvements
                for owner, (cand_idx_flat, ll) in per_owner_best.items():
                    if ll > best_likelihoods[owner]:
                        # accept candidate
                        prompt_tokens_list[owner] = candidates_flat[cand_idx_flat]
                        best_likelihoods[owner] = ll
                        best_rewards[owner] = alpha * ll - beta * (L + (base_embeds_list[owner].shape[0] if base_embeds_list[owner] is not None else 0))

                traces.append({
                    'episode': ep,
                    'step': step,
                    'best_likelihoods': best_likelihoods.copy(),
                    'best_rewards': best_rewards.copy()
                })

        return best_prompts, best_rewards, traces

    def optimize_prompts_batch_ppo(
        self,
        target_completions: List[str],
        episodes: int = 3,
        steps_per_episode: int = 50,
        initial_prompt_length: int = 32,
        lr_embeddings: float = 0.01,
        lr_policy: float = 3e-4,
        alpha: float = 1.0,
        beta: float = 0.1,
        base_prompts: Optional[List[str]] = None,
        optimization_mode: str = "continuous",
        inner_steps: int = 5,
        use_ppo: bool = True,
        ppo_epochs: int = 4,
        ppo_clip: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ) -> Tuple[List[List[int]], List[float], List[dict]]:
        """Run PPO training over a batch of prompts in parallel.

        Currently only the continuous (embedding) inner optimizer is supported.
        """
        mode = optimization_mode.lower()
        if mode != "continuous":
            raise NotImplementedError("optimize_prompts_batch_ppo currently supports continuous mode only.")
        if lr_policy is not None:
            for pg in self.policy_optimizer.param_groups:
                pg['lr'] = lr_policy
            for pg in self.value_optimizer.param_groups:
                pg['lr'] = lr_policy

        device = self.agent.device
        B = len(target_completions)
        if B == 0:
            return [], [], []

        embedding_layer = self.agent.model.get_input_embeddings()

        completion_tokens_list = [
            self.agent.tokenizer.encode(t, add_special_tokens=False) for t in target_completions
        ]

        if base_prompts is None:
            base_prompts = [None] * B

        base_tokens_list: List[List[int]] = []
        base_embeds_list: List[Optional[torch.Tensor]] = []
        for bp in base_prompts:
            if bp:
                btoks = self.agent.tokenizer.encode(bp, add_special_tokens=False)
                base_tokens_list.append(btoks)
                bt = torch.tensor(btoks, dtype=torch.long, device=device)
                base_embeds_list.append(embedding_layer(bt).detach())
            else:
                base_tokens_list.append([])
                base_embeds_list.append(None)

        best_prompts: List[Optional[List[int]]] = [None] * B
        best_rewards: List[float] = [float('-inf')] * B
        traces: List[dict] = []
        global_step = 0

        for episode in range(episodes):
            # Initialize suffix parameters for each example
            prompt_params: List[nn.Parameter] = []
            prompt_opts: List[optim.Optimizer] = []
            lengths = []
            for _ in range(B):
                param = nn.Parameter(
                    torch.randn(initial_prompt_length, self.emb_dim, device=device) * 0.1
                )
                prompt_params.append(param)
                prompt_opts.append(optim.Adam([param], lr=lr_embeddings))
                lengths.append(initial_prompt_length)

            recent_likelihoods = [deque(maxlen=10) for _ in range(B)]
            episode_states: List[torch.Tensor] = []
            episode_actions: List[torch.Tensor] = []
            episode_log_probs: List[torch.Tensor] = []
            episode_rewards: List[torch.Tensor] = []

            for step in range(steps_per_episode):
                state_vectors: List[torch.Tensor] = []
                current_likelihoods: List[float] = []
                improvement_rates: List[float] = []
                steps_since_improv_list: List[int] = []
                importance_cache: List[Dict[str, float]] = []

                for i in range(B):
                    ll_trace = self._optimize_continuous_prompt(
                        completion_tokens_list[i],
                        steps=max(1, inner_steps),
                        prompt_embeddings=prompt_params[i],
                        prompt_optimizer=prompt_opts[i],
                        base_embeds=base_embeds_list[i]
                    )
                    if ll_trace:
                        current_ll = float(ll_trace[-1])
                    else:
                        combined = prompt_params[i].detach()
                        if base_embeds_list[i] is not None and base_embeds_list[i].numel() > 0:
                            combined = torch.cat([base_embeds_list[i], combined], dim=0)
                        current_ll = float(self._get_likelihood_from_embeddings(combined, completion_tokens_list[i]))

                    recent_likelihoods[i].append(current_ll)
                    current_likelihoods.append(current_ll)

                    if len(recent_likelihoods[i]) >= 3:
                        improvement_rates.append(float(recent_likelihoods[i][-1] - recent_likelihoods[i][0]))
                    else:
                        improvement_rates.append(0.0)

                    improvement_threshold = 0.1
                    steps_since = 0
                    for past_ll in reversed(list(recent_likelihoods[i])):
                        if current_ll - past_ll > improvement_threshold:
                            break
                        steps_since += 1
                    steps_since_improv_list.append(steps_since)

                    importance_stats = self._compute_importance_stats(
                        use_continuous=True,
                        completion_tokens=completion_tokens_list[i],
                        prompt_embeddings=prompt_params[i],
                        base_embeds=base_embeds_list[i]
                    )
                    importance_cache.append(importance_stats)

                    state_vector = torch.tensor(
                        [
                            lengths[i],
                            current_ll,
                            step / max(1, steps_per_episode),
                            improvement_rates[-1],
                            steps_since,
                            importance_stats['min_score'],
                            importance_stats['mean_score'],
                            importance_stats['max_score'],
                            importance_stats['weakest_pos_norm'],
                            importance_stats['tail_avg_score']
                        ],
                        dtype=torch.float32,
                        device=device
                    )
                    state_vectors.append(state_vector)

                states_batch = torch.stack(state_vectors)  # [B, state_dim]
                episode_states.append(states_batch)

                policy_logits = self.policy_net(states_batch)
                policy_probs = F.softmax(policy_logits, dim=-1)
                policy_dist = torch.distributions.Categorical(policy_probs)
                actions = policy_dist.sample()
                log_probs = policy_dist.log_prob(actions)

                episode_actions.append(actions)
                episode_log_probs.append(log_probs)

                batch_rewards = torch.zeros(B, dtype=torch.float32, device=device)

                final_ll_snapshot: List[float] = []
                for i in range(B):
                    action_val = int(actions[i].item())
                    new_length = lengths[i]

                    if action_val == 0 and lengths[i] > 1:
                        updated = prompt_params[i].detach()[:-1].clone()
                        prompt_params[i] = nn.Parameter(updated.requires_grad_(True))
                        prompt_opts[i] = optim.Adam([prompt_params[i]], lr=lr_embeddings)
                        new_length = lengths[i] - 1
                    elif action_val == 2:
                        current_embeds = prompt_params[i].detach()
                        if base_embeds_list[i] is not None and base_embeds_list[i].numel() > 0:
                            combined_base = torch.cat([base_embeds_list[i], current_embeds], dim=0)
                        else:
                            combined_base = current_embeds
                        new_token_embed = self._initialize_continuous_append(
                            base_embeds=combined_base,
                            completion_tokens=completion_tokens_list[i]
                        )
                        append_embed = new_token_embed.unsqueeze(0)
                        updated = torch.cat([current_embeds, append_embed], dim=0)
                        prompt_params[i] = nn.Parameter(updated.clone().detach().requires_grad_(True))
                        prompt_opts[i] = optim.Adam([prompt_params[i]], lr=lr_embeddings)
                        new_length = lengths[i] + 1

                    lengths[i] = new_length

                    if base_embeds_list[i] is not None and base_embeds_list[i].numel() > 0:
                        combined = torch.cat([base_embeds_list[i], prompt_params[i].detach()], dim=0)
                    else:
                        combined = prompt_params[i].detach()
                    final_ll = float(self._get_likelihood_from_embeddings(combined, completion_tokens_list[i]))
                    final_ll_snapshot.append(final_ll)

                    base_reward = alpha * final_ll - beta * lengths[i]
                    discovery_bonus = 0.0
                    if action_val == 0:
                        discovery_bonus += 0.1
                    if lengths[i] < initial_prompt_length and final_ll > -1.0:
                        discovery_bonus += (initial_prompt_length - lengths[i]) * 0.05

                    total_reward = base_reward + discovery_bonus
                    batch_rewards[i] = total_reward

                    if total_reward > best_rewards[i]:
                        best_rewards[i] = float(total_reward)
                        suffix_tokens = self._embeddings_to_tokens(prompt_params[i].detach())
                        base_ids = base_tokens_list[i]
                        best_prompts[i] = (base_ids + suffix_tokens) if base_ids else suffix_tokens

                episode_rewards.append(batch_rewards)
                traces.append({
                    'episode': episode,
                    'step': step,
                    'likelihoods': [float(v) for v in final_ll_snapshot],
                    'lengths': [int(L) for L in lengths],
                    'actions': [int(a.item()) for a in actions],
                    'rewards': [float(r) for r in batch_rewards.tolist()],
                    'best_rewards': best_rewards.copy(),
                    'global_step': global_step
                })
                global_step += 1

            # After collecting trajectories for the batch, run PPO (or vanilla) updates
            states_tensor = torch.stack(episode_states)  # [T, B, state_dim]
            actions_tensor = torch.stack(episode_actions)  # [T, B]
            log_probs_tensor = torch.stack(episode_log_probs)  # [T, B]
            rewards_tensor = torch.stack(episode_rewards)  # [T, B]
            T = states_tensor.shape[0]

            states_flat = states_tensor.view(T * B, self.state_dim)
            values_flat = self.value_net(states_flat).squeeze()
            values_tensor = values_flat.view(T, B)

            device = self.agent.device

            if use_ppo:
                advantages = torch.zeros_like(rewards_tensor, device=device)
                last_gae = torch.zeros(B, dtype=torch.float32, device=device)
                next_value = torch.zeros(B, dtype=torch.float32, device=device)
                for t in reversed(range(T)):
                    delta = rewards_tensor[t] + gamma * next_value - values_tensor[t]
                    last_gae = delta + gamma * gae_lambda * last_gae
                    advantages[t] = last_gae
                    next_value = values_tensor[t]

                returns_tensor = advantages + values_tensor
                advantages_flat = advantages.view(-1)
                returns_flat = returns_tensor.view(-1)
                advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

                old_log_probs_flat = log_probs_tensor.view(-1).detach()
                actions_flat = actions_tensor.view(-1)

                for _ in range(max(1, ppo_epochs)):
                    policy_logits = self.policy_net(states_flat)
                    policy_probs = F.softmax(policy_logits, dim=-1)
                    policy_dist = torch.distributions.Categorical(policy_probs)
                    new_log_probs = policy_dist.log_prob(actions_flat)
                    entropy = policy_dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - old_log_probs_flat)
                    surr1 = ratios * advantages_flat
                    surr2 = torch.clamp(ratios, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages_flat
                    policy_loss = -torch.mean(torch.min(surr1, surr2))

                    value_preds = self.value_net(states_flat).squeeze()
                    value_loss = F.mse_loss(value_preds, returns_flat)

                    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    self.policy_optimizer.zero_grad()
                    self.value_optimizer.zero_grad()
                    total_loss.backward()
                    self.policy_optimizer.step()
                    self.value_optimizer.step()
            else:
                returns_tensor = torch.zeros_like(rewards_tensor, device=device)
                next_return = torch.zeros(B, dtype=torch.float32, device=device)
                for t in reversed(range(T)):
                    next_return = rewards_tensor[t] + gamma * next_return
                    returns_tensor[t] = next_return
                returns_flat = returns_tensor.view(-1)
                values_detached = values_flat.detach()
                advantages = returns_flat - values_detached

                policy_loss = -(log_probs_tensor.view(-1) * advantages.detach()).mean()
                value_loss = F.mse_loss(values_flat, returns_flat)

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss = policy_loss + value_coef * value_loss
                total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

        self.trace = traces
        final_prompts: List[List[int]] = []
        for idx in range(B):
            if best_prompts[idx] is not None:
                final_prompts.append(best_prompts[idx])  # already includes base tokens
            else:
                if 'prompt_params' in locals() and idx < len(prompt_params):
                    suffix_tokens = self._embeddings_to_tokens(prompt_params[idx].detach())
                else:
                    suffix_tokens = []
                base_ids = base_tokens_list[idx]
                combined_tokens = (base_ids + suffix_tokens) if base_ids else suffix_tokens
                final_prompts.append(combined_tokens)

        return final_prompts, best_rewards, traces

# Main function removed - use train.py and eval.py instead

if __name__ == "__main__":
    print("This module contains the core classes PromptRLAgent and LengthPolicyOptimizer.")
    print("Use train.py to train a model and eval.py to evaluate.")
    print("Example:")
    print("  python train.py --episodes 100")
    print("  python eval.py --test_prompt 'Your test text here'")
