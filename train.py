#!/usr/bin/env python3
"""
Train the prompt compression policy on the toxic-chat dataset.
"""
import torch
import argparse
import os
import yaml
import random
import time
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
from dataset_utils import ToxicChatDatasetManager
import numpy as np

def train_on_dataset_fast(cfg):
    """Fast training with optimized settings for speed."""
    model_name = cfg['model']
    train_cfg = cfg['train']
    
    # Reduced training hyperparameters for speed
    episodes_per_prompt = max(1, train_cfg.get('episodes_per_prompt', 3) // 2)  # Halve episodes
    steps_per_episode = max(20, train_cfg.get('steps_per_episode', 100) // 2)  # Halve steps
    init_len = train_cfg.get('init_len', 32)
    lr_embeddings = train_cfg.get('lr_embeddings', 0.01) * 2  # Double learning rate for faster convergence
    lr_policy = train_cfg.get('lr_policy', 3e-4) * 2
    alpha = train_cfg.get('alpha', 1.0)
    beta = train_cfg.get('beta', 0.2)
    save_path = train_cfg.get('save_path', 'models/trained_policy_fast.pt')
    
    # Batch processing for faster training
    batch_size = train_cfg.get('batch_size', 8)
    
    # Dataset parameters
    max_prompts = train_cfg.get('max_prompts', 50)
    min_prompt_length = train_cfg.get('min_prompt_length', 30)
    max_prompt_length = train_cfg.get('max_prompt_length', 150)
    
    seed = cfg.get('seed', 2262)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    print(f"Fast training mode with {max_prompts} prompts")
    print(f"Model: {model_name}")
    print(f"OPTIMIZED SETTINGS:")
    print(f"  Episodes per prompt: {episodes_per_prompt} (reduced)")
    print(f"  Steps per episode: {steps_per_episode} (reduced)")
    print(f"  Learning rates: {lr_embeddings:.3f} / {lr_policy:.6f} (increased)")
    print(f"  Batch processing: {batch_size} prompts")
    print(f"Alpha: {alpha}, Beta: {beta}")
    
    # Load dataset using the new dataset manager with config split ratios
    dataset_manager = ToxicChatDatasetManager(seed=seed)
    ds_cfg = cfg.get('dataset', {})
    prompts = dataset_manager.load_train_set(
        min_length=min_prompt_length,
        max_length=max_prompt_length,
        max_samples=max_prompts,
        train_ratio=ds_cfg.get('train_ratio', 0.7),
        val_ratio=ds_cfg.get('val_ratio', 0.15),
        test_ratio=ds_cfg.get('test_ratio', 0.15),
        use_cache=True
    )
    
    if not prompts:
        raise ValueError("No valid prompts found in dataset")
    
    # Initialize agent and optimizer
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Track training progress across all prompts
    all_rewards = []
    best_overall_reward = float('-inf')
    best_prompt = None
    best_prompt_text = None
    
    start_time = time.time()
    
    # Process prompts in batches for better progress tracking
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        
        print(f"\n[Batch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}] Processing prompts {batch_start+1}-{batch_end}")
        
        batch_start_time = time.time()
        
        for prompt_idx, prompt_text in enumerate(batch_prompts):
            global_idx = batch_start + prompt_idx
            
            try:
                # Train on this specific prompt with reduced parameters
                best_prompt_result, best_reward, history = optimizer.optimize_prompt(
                    prompt_text,
                    episodes=episodes_per_prompt,
                    steps_per_episode=steps_per_episode,
                    initial_prompt_length=init_len,
                    lr_embeddings=lr_embeddings,
                    lr_policy=lr_policy,
                    alpha=alpha,
                    beta=beta,
                    log_every=0  # Disable detailed logging for speed
                )
                
                all_rewards.append(float(best_reward))
                
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)  # Ensure it's a Python float
                    best_prompt = best_prompt_result
                    best_prompt_text = prompt_text
                
                # Quick progress update
                if (prompt_idx + 1) % max(1, len(batch_prompts) // 4) == 0:
                    print(f"  Progress: {prompt_idx + 1}/{len(batch_prompts)}, Latest reward: {best_reward:.3f}")
                
            except Exception as e:
                print(f"  Error on prompt {global_idx+1}: {e}")
                continue
        
        batch_time = time.time() - batch_start_time
        avg_time_per_prompt = batch_time / len(batch_prompts)
        
        print(f"  Batch completed in {batch_time:.1f}s ({avg_time_per_prompt:.2f}s/prompt)")
        print(f"  Best batch reward: {max(all_rewards[-len(batch_prompts):]) if all_rewards else 'N/A'}")
        print(f"  Overall best so far: {best_overall_reward:.3f}")
        
        # Progress estimate
        completed = len(all_rewards)
        remaining = len(prompts) - completed
        estimated_time_left = remaining * avg_time_per_prompt
        print(f"  ETA: {estimated_time_left/60:.1f} minutes ({completed}/{len(prompts)} prompts complete)")
    
    training_time = time.time() - start_time
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'model_name': model_name,
        'policy_state_dict': optimizer.policy_net.state_dict(),
        'config': cfg,
        'training_rewards': all_rewards,
        'best_reward': float(best_overall_reward),
        'best_prompt': best_prompt,
        'best_prompt_text': best_prompt_text,
        'training_time': training_time,
        'fast_mode_settings': {
            'episodes_per_prompt': episodes_per_prompt,
            'steps_per_episode': steps_per_episode,
            'lr_embeddings': lr_embeddings,
            'lr_policy': lr_policy
        }
    }
    
    torch.save(checkpoint, save_path)
    
    # Final statistics
    if all_rewards:
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        print(f"\n{'='*60}")
        print(f"FAST TRAINING COMPLETE!")
        print(f"Trained on {len(all_rewards)} prompts in {training_time:.1f}s")
        print(f"Speed: {len(all_rewards)/training_time:.2f} prompts/second")
        print(f"Average time per prompt: {training_time/len(all_rewards):.2f}s")
        print(f"Best reward: {best_overall_reward:.3f}")
        print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"Best prompt text: '{best_prompt_text[:80]}{'...' if len(best_prompt_text) > 80 else ''}'")
        print(f"Model saved to: {save_path}")
        print(f"OPTIMIZATION USED:")
        print(f"  - Reduced episodes: {episodes_per_prompt} (vs {train_cfg.get('episodes_per_prompt', 3)})")
        print(f"  - Reduced steps: {steps_per_episode} (vs {train_cfg.get('steps_per_episode', 100)})")
        print(f"  - Increased LR: {lr_embeddings:.3f} (vs {train_cfg.get('lr_embeddings', 0.01):.3f})")
        print(f"{'='*60}")
        
        # Generate plots if enabled (simple reward plot)
        if not train_cfg.get('no_plots', False):
            try:
                # Convert all rewards to plain Python floats to avoid CUDA tensor issues
                plot_rewards = [float(r.cpu() if hasattr(r, 'cpu') else r) for r in all_rewards]
                plot_best_reward = float(best_overall_reward.cpu() if hasattr(best_overall_reward, 'cpu') else best_overall_reward)
                
                # Create a simple reward plot since we don't have the full training history
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(plot_rewards) + 1), plot_rewards, 'b-', alpha=0.7, label='Rewards')
                plt.axhline(y=np.mean(plot_rewards), color='r', linestyle='--', alpha=0.7, label=f'Average: {np.mean(plot_rewards):.3f}')
                plt.axhline(y=plot_best_reward, color='g', linestyle='--', alpha=0.7, label=f'Best: {plot_best_reward:.3f}')
                plt.title(f"Fast Training Progress ({len(plot_rewards)} prompts)")
                plt.xlabel("Prompt Number")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                fmt = train_cfg.get('plots_format', 'png')
                plot_path = f"results/{train_cfg.get('plots_prefix', 'fast_training')}_rewards.{fmt}"
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Reward plot saved to: {plot_path}")
                
            except Exception as e:
                print(f"Could not generate plot: {e}")
    
    else:
        print("No successful training results!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--prompts", type=int, help="Number of prompts (overrides config)")
    parser.add_argument("--episodes", type=int, help="Episodes per prompt (overrides config)")
    parser.add_argument("--steps", type=int, help="Steps per episode (overrides config)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override config with command line args if provided
    if args.prompts:
        cfg['train']['max_prompts'] = args.prompts
    if args.episodes:
        cfg['train']['episodes_per_prompt'] = args.episodes
    if args.steps:
        cfg['train']['steps_per_episode'] = args.steps
    
    train_on_dataset_fast(cfg)