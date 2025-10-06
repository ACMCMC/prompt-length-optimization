#!/usr/bin/env python3
"""
Train the length policy optimizer and save the complete model.
"""
import torch
import argparse
import os
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer

def train_and_save(model_name="EleutherAI/pythia-410m", 
                   target="The quick brown fox jumps over the lazy dog.",
                   episodes=100, steps_per_episode=50, init_len=32,
                   save_path="models/trained_policy.pt"):
    
    print(f"Training policy optimizer...")
    print(f"Model: {model_name}")
    print(f"Episodes: {episodes}, Steps per episode: {steps_per_episode}")
    print(f"Initial length: {init_len}")
    
    # Initialize
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Train
    best_prompt, best_reward, history = optimizer.optimize_prompt(
        target,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        initial_prompt_length=init_len,
        log_every=10
    )
    
    # Save complete model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'policy_state_dict': optimizer.policy_net.state_dict(),
        'model_name': model_name,
        'best_prompt': best_prompt,
        'best_reward': best_reward,
        'training_args': {
            'episodes': episodes,
            'steps_per_episode': steps_per_episode,
            'init_len': init_len,
            'target': target
        }
    }, save_path)
    
    print(f"\nTraining complete!")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Model saved to: {save_path}")
    
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m")
    parser.add_argument("--target", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps_per_episode", type=int, default=50)
    parser.add_argument("--init_len", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="models/trained_policy.pt")
    args = parser.parse_args()
    
    train_and_save(
        model_name=args.model,
        target=args.target,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        init_len=args.init_len,
        save_path=args.save_path
    )