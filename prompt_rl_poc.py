"""
Proof of Concept: RL-based Prompt Length Optimization
Given a target completion, find the shortest prompt that maximizes P(completion | prompt)
"""

import torch
import torch.nn.functional as F
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import numpy as np
import random
from typing import List, Dict, Tuple
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
        # device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
        special_tokens = {self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 
                         self.tokenizer.bos_token_id}
        while True:
            token = random.randint(0, self.vocab_size - 1)
            if token not in special_tokens:
                return token

class LengthPolicyOptimizer:
    def __init__(self, agent: PromptRLAgent):
        self.agent = agent
        self.emb_dim = self.agent.model.get_input_embeddings().weight.shape[1]
        
        # Policy network that decides length changes based on current state
        self.policy_net = nn.Sequential(
            nn.Linear(self.emb_dim + 3, 128),  # +3 for current_length, likelihood, step_ratio
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 actions: REMOVE_TOKEN, KEEP_LENGTH
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
        
    def optimize_prompt(self, target_completion: str, episodes=100, steps_per_episode=50,
                       initial_prompt_length=32, lr_embeddings=0.01, lr_policy=3e-4,
                       alpha=1.0, beta=0.1, log_every=10) -> Tuple[List[int], float, List[float]]:
        """
        Train policy to learn optimal length adjustments while optimizing embeddings
        Policy learns when to remove/keep/add tokens based on current state
        """
        completion_tokens = self.agent.tokenizer.encode(target_completion, add_special_tokens=False)
        
        print(f"Target completion: '{target_completion}'")
        print(f"Training policy for {episodes} episodes, {steps_per_episode} steps each")
        print(f"Starting length: {initial_prompt_length}, α={alpha}, β={beta}")
        print("-" * 60)
        
        best_overall_reward = float('-inf')
        best_prompt = None
        
        for episode in range(episodes):
            # Reset for new episode
            current_length = initial_prompt_length
            self.prompt_embeddings = nn.Parameter(
                torch.randn(current_length, self.emb_dim, device=self.agent.device) * 0.1
            )
            self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
            
            episode_rewards = []
            episode_log_probs = []
            episode_actions = []
            
            for step in range(steps_per_episode):
                # Optimize embeddings for a few steps
                for _ in range(3):  # Quick embedding optimization
                    self.prompt_optimizer.zero_grad()
                    likelihood = self._get_likelihood_from_embeddings(self.prompt_embeddings, completion_tokens)
                    emb_loss = -likelihood
                    emb_loss.backward()
                    self.prompt_optimizer.step()
                
                # Get current state for policy
                with torch.no_grad():
                    likelihood = self._get_likelihood_from_embeddings(self.prompt_embeddings, completion_tokens)
                    prompt_summary = self.prompt_embeddings.mean(dim=0)  # Average embedding
                    
                    # State: [prompt_summary, current_length, likelihood, step_ratio]
                    state_features = torch.cat([
                        prompt_summary,
                        torch.tensor([current_length, likelihood, step/steps_per_episode], 
                                   device=self.agent.device, dtype=torch.float32)
                    ])
                
                # Policy decision
                policy_logits = self.policy_net(state_features)
                policy_probs = F.softmax(policy_logits, dim=-1)
                policy_dist = torch.distributions.Categorical(policy_probs)
                action = policy_dist.sample()
                log_prob = policy_dist.log_prob(action)
                
                # Execute action: 0=REMOVE, 1=KEEP
                action_val = int(action.item())
                new_length = current_length
                
                if action_val == 0 and current_length > 1:  # REMOVE
                    new_length = current_length - 1
                    # Remove last token embedding
                    self.prompt_embeddings = nn.Parameter(
                        self.prompt_embeddings[:-1].clone().detach().requires_grad_(True)
                    )
                    self.prompt_optimizer = optim.Adam([self.prompt_embeddings], lr=lr_embeddings)
                # action_val == 1: KEEP (no change)
                
                current_length = new_length
                
                # Calculate reward
                with torch.no_grad():
                    likelihood = self._get_likelihood_from_embeddings(self.prompt_embeddings, completion_tokens)
                    reward = alpha * likelihood - beta * current_length
                
                # Store for policy update
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                episode_actions.append(action_val)
                
                # Track best
                if reward > best_overall_reward:
                    best_overall_reward = reward
                    best_prompt = self._embeddings_to_tokens(self.prompt_embeddings.detach())
                
                # Logging
                self.likelihood_history.append(float(likelihood))
                self.length_history.append(current_length)
                self.action_history.append(action_val)
                
                if episode % log_every == 0 and step % 10 == 0:
                    print(f"Ep {episode:3d} Step {step:2d}: action={action_val} length={current_length} "
                          f"likelihood={likelihood:.3f} reward={reward:.3f}")
            
            # Policy update at end of episode (REINFORCE)
            episode_return = sum(episode_rewards)
            returns = []
            G = 0
            for r in reversed(episode_rewards):
                G = r + G  
                returns.insert(0, G)
            
            returns = torch.tensor(returns, device=self.agent.device)
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = 0
            for log_prob, ret in zip(episode_log_probs, returns):
                policy_loss = policy_loss - log_prob * ret
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            if episode % log_every == 0:
                avg_length = sum(self.length_history[-steps_per_episode:]) / steps_per_episode
                print(f"Episode {episode}: return={episode_return:.2f} avg_length={avg_length:.1f} "
                      f"best_reward={best_overall_reward:.3f}")
        
        return best_prompt, best_overall_reward, self.loss_history
    
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
