#!/usr/bin/env python3
"""
Train the length policy optimizer and save the complete model.
"""
import torch
import argparse
import os
import yaml
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
from plot_utils import plot_training_progress

def train_and_save(cfg):
    model_name = cfg['model']
    train_cfg = cfg['train']
    target = train_cfg['target']
    episodes = train_cfg['episodes']
    steps_per_episode = train_cfg['steps_per_episode']
    init_len = train_cfg['init_len']
    lr_embeddings = train_cfg.get('lr_embeddings', 0.01)
    lr_policy = train_cfg.get('lr_policy', 3e-4)
    alpha = train_cfg.get('alpha', 1.0)
    beta = train_cfg.get('beta', 0.1)
    log_every = train_cfg.get('log_every', 10)
    save_path = train_cfg.get('save_path', 'models/trained_policy.pt')
    no_plots = train_cfg.get('no_plots', False)
    plots_prefix = train_cfg.get('plots_prefix', 'training')
    seed = cfg.get('seed', 2262)
    
    print(f"Training policy optimizer (YAML config)...")
    print(f"Model: {model_name}")
    print(f"Episodes: {episodes}, Steps per episode: {steps_per_episode}")
    print(f"Initial length: {init_len}  alpha={alpha} beta={beta}")
    
    # Initialize
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Train
    best_prompt, best_reward, history = optimizer.optimize_prompt(
        target,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        initial_prompt_length=init_len,
        lr_embeddings=lr_embeddings,
        lr_policy=lr_policy,
        alpha=alpha,
        beta=beta,
        log_every=log_every
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

    # Plot training progress
    if not no_plots:
        pdf_path = plot_training_progress(
            optimizer.likelihood_history,
            optimizer.length_history,
            optimizer.action_history,
            out_dir=os.path.dirname(save_path) if os.path.dirname(save_path) else 'results',
            prefix=plots_prefix
        )
        print(f"Saved training plot to: {pdf_path}")
    
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    train_and_save(cfg)