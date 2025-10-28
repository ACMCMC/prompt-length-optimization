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
from datasets import load_dataset
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
import numpy as np

def load_toxic_chat_dataset(split='train', max_samples=None, min_length=20, max_length=200):
    """Load and filter the toxic-chat dataset for training."""
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split)
    
    prompts = []
    for example in dataset:
        prompt = example.get('model_output', '')
        if prompt and min_length <= len(prompt) <= max_length:
            prompts.append(prompt.strip())
            if max_samples and len(prompts) >= max_samples:
                break
    print("prompts loaded:", prompts)
    return prompts

def decode_tokens(agent, token_ids):
    """Helper to decode token IDs into readable text."""
    if token_ids is None:
        return ""
    try:
        return agent.tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as exc:
        print(f"[warn] Failed to decode tokens: {exc}")
        return ""


def train_on_dataset(cfg, fast_mode=False, preview_first=True, log_every=None):
    """Train using the config values (optionally applying fast-mode overrides)."""
    model_name = cfg['model']
    train_cfg = cfg['train']

    # Base training hyperparameters from config
    episodes_per_prompt = train_cfg.get('episodes_per_prompt', 3)
    steps_per_episode = train_cfg.get('steps_per_episode', 100)
    init_len = train_cfg.get('init_len', 32)
    lr_embeddings = train_cfg.get('lr_embeddings', 0.01)
    lr_policy = train_cfg.get('lr_policy', 3e-4)
    alpha = train_cfg.get('alpha', 1.0)
    beta = train_cfg.get('beta', 0.2)
    save_path = train_cfg.get('save_path', 'models/trained_policy.pt')
    log_every = log_every if log_every is not None else train_cfg.get('log_every', 1)

    # Fast mode overrides halve iterations and boost learning rates
    if fast_mode:
        print("[info] Fast mode enabled – applying speed-focused overrides.")
        episodes_per_prompt = max(1, episodes_per_prompt // 2)
        steps_per_episode = max(20, steps_per_episode // 2)
        lr_embeddings *= 2
        lr_policy *= 2
        save_path = train_cfg.get('save_path_fast', 'models/trained_policy_fast.pt')

    # Dataset parameters
    max_prompts = train_cfg.get('max_prompts', 50)
    min_prompt_length = train_cfg.get('min_prompt_length', 30)
    max_prompt_length = train_cfg.get('max_prompt_length', 150)
    batch_size = max(1, train_cfg.get('batch_size', 8))

    seed = cfg.get('seed', 2262)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    mode_label = "FAST" if fast_mode else "FULL"
    print(f"\n=== {mode_label} TRAINING CONFIGURATION ===")
    print(f"Model: {model_name}")
    print(f"Prompts: up to {max_prompts} (length {min_prompt_length}-{max_prompt_length} chars)")
    print(f"Episodes per prompt: {episodes_per_prompt}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Learning rates: embeddings={lr_embeddings:.4f} policy={lr_policy:.6f}")
    print(f"Alpha={alpha}, Beta={beta}, Initial length={init_len} tokens")
    print(f"Batch size for progress reporting: {batch_size}")
    print(f"RL logging frequency: every {log_every} episode(s)\n")
    
    # Load dataset
    # prompts = load_toxic_chat_dataset(
    #     split='train', 
    #     max_samples=max_prompts,
    #     min_length=min_prompt_length,
    #     max_length=max_prompt_length
    # )

    prompts = ['You are a bad person. You do bad things. You are very bad.']
    
    if not prompts:
        raise ValueError("No valid prompts found in dataset")
    
    total_prompts = len(prompts)
    print(f"[info] Loaded {total_prompts} prompts from dataset.")

    # Initialize agent and optimizer
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Track training progress across all prompts
    all_rewards = []
    best_overall_reward = float('-inf')
    best_prompt = None
    best_prompt_text = None
    
    start_time = time.time()

    # Optional single-prompt preview (uses first prompt in list)
    processed_prompts = 0
    preview_plot_trace = train_cfg.get('preview_plot_trace', True)
    collect_prompt_traces = train_cfg.get('collect_prompt_traces', False)
    prompt_trace_dir = train_cfg.get('prompt_trace_dir', 'results/traces')

    if preview_first:
        preview_prompt_text = prompts[0]
        print("\n=== SINGLE PROMPT PREVIEW ===")
        print(f"Prompt 1/{total_prompts} (preview target completion text shown below):")
        print(f"----\n{preview_prompt_text}\n----")
        preview_start = time.time()
        preview_prompt, preview_reward, preview_trace = optimizer.optimize_prompt(
            preview_prompt_text,
            episodes=episodes_per_prompt,
            steps_per_episode=steps_per_episode,
            initial_prompt_length=init_len,
            lr_embeddings=lr_embeddings,
            lr_policy=lr_policy,
            alpha=alpha,
            beta=beta,
            log_every=max(1, log_every),
            collect_trace=True,
            plot_trace=preview_plot_trace,
            plot_path=os.path.join(prompt_trace_dir, "preview_trace.png") if preview_plot_trace else None
        )
        preview_time = time.time() - preview_start
        preview_decoded = decode_tokens(agent, preview_prompt)
        print("\n[preview] Complete.")
        print(f"[preview] Reward: {preview_reward:.4f}")
        print(f"[preview] Optimized prompt tokens: {preview_prompt}")
        print(f"[preview] Optimized prompt text: '{preview_decoded}'")
        print(f"[preview] Time elapsed: {preview_time:.1f}s\n")
        if preview_plot_trace:
            print(f"[preview] Trace plotting enabled. Latest plot stored in '{prompt_trace_dir}'.")

        processed_prompts += 1
        all_rewards.append(float(preview_reward))
        if preview_reward > best_overall_reward:
            best_overall_reward = float(preview_reward)
            best_prompt = preview_prompt
            best_prompt_text = preview_prompt_text
    else:
        print("[info] Preview disabled; proceeding directly to full training.\n")

    # Process remaining prompts
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        # Skip previewed prompt in current batch if applicable
        if preview_first and batch_start == 0:
            batch_prompts = batch_prompts[1:]
            if not batch_prompts:
                continue
        
        print(f"\n[Batch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}] Processing prompts {batch_start+1}-{batch_end}")
        batch_start_time = time.time()
        
        for prompt_idx, prompt_text in enumerate(batch_prompts):
            global_idx = batch_start + prompt_idx

            if preview_first and global_idx == 0:
                continue

            print(f"\n--- Prompt {global_idx + 1}/{total_prompts} ---")
            print(f"Target completion text (truncated to 200 chars): '{prompt_text[:200]}{'...' if len(prompt_text) > 200 else ''}'")
            prompt_start = time.time()
            try:
                # Train on this specific prompt (full config unless fast-mode overrides applied above)
                best_prompt_result, best_reward, history = optimizer.optimize_prompt(
                    prompt_text,
                    episodes=episodes_per_prompt,
                    steps_per_episode=steps_per_episode,
                    initial_prompt_length=init_len,
                    lr_embeddings=lr_embeddings,
                    lr_policy=lr_policy,
                    alpha=alpha,
                    beta=beta,
                    log_every=max(1, log_every),
                    collect_trace=collect_prompt_traces,
                    plot_trace=False
                )
                
                all_rewards.append(float(best_reward))
                processed_prompts += 1
                
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)  # Ensure it's a Python float
                    best_prompt = best_prompt_result
                    best_prompt_text = prompt_text

                elapsed = time.time() - prompt_start
                decoded_prompt = decode_tokens(agent, best_prompt_result)
                print(f"[prompt] Reward: {best_reward:.4f}")
                print(f"[prompt] Optimized prompt tokens: {best_prompt_result}")
                print(f"[prompt] Optimized prompt text: '{decoded_prompt}'")
                print(f"[prompt] Duration: {elapsed:.2f}s")
                
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
        completed = processed_prompts
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
        'training_settings': {
            'mode': mode_label.lower(),
            'episodes_per_prompt': episodes_per_prompt,
            'steps_per_episode': steps_per_episode,
            'lr_embeddings': lr_embeddings,
            'lr_policy': lr_policy,
            'alpha': alpha,
            'beta': beta,
            'initial_length': init_len
        }
    }
    
    torch.save(checkpoint, save_path)
    
    # Final statistics
    if all_rewards:
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        print(f"\n{'='*60}")
        print(f"{mode_label} TRAINING COMPLETE!")
        print(f"Trained on {processed_prompts} prompts in {training_time:.1f}s")
        print(f"Speed: {processed_prompts/training_time:.2f} prompts/second")
        print(f"Average time per prompt: {training_time/processed_prompts:.2f}s")
        print(f"Best reward: {best_overall_reward:.3f}")
        print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
        if best_prompt_text:
            preview_text = best_prompt_text[:80] + ('...' if len(best_prompt_text) > 80 else '')
            print(f"Best prompt target text: '{preview_text}'")
        # Print the actual optimized prompt (decoded from token IDs)
        if best_prompt is not None:
            # Load the agent to decode tokens
            agent = PromptRLAgent(model_name)
            optimized_prompt_text = agent.tokenizer.decode(best_prompt, skip_special_tokens=True)
            print(f"Optimized prompt (decoded): '{optimized_prompt_text}'")
        else:
            print("No optimized prompt found.")
        print(f"Model saved to: {save_path}")
        print("Training hyperparameters applied:")
        print(f"  - Episodes per prompt: {episodes_per_prompt}")
        print(f"  - Steps per episode: {steps_per_episode}")
        print(f"  - LR (embeddings / policy): {lr_embeddings:.4f} / {lr_policy:.6f}")
        print(f"  - Alpha / Beta: {alpha} / {beta}")
        print(f"  - Initial prompt length: {init_len}")
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
                plt.title(f"{mode_label.title()} Training Progress ({len(plot_rewards)} prompts)")
                plt.xlabel("Prompt Number")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                default_prefix = f"{mode_label.lower()}_training"
                plot_path = f"results/{train_cfg.get('plots_prefix', default_prefix)}_rewards.png"
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
    parser.add_argument("--fast", action="store_true", help="Enable fast training overrides")
    parser.add_argument("--no-preview", action="store_true", help="Skip single prompt preview phase")
    parser.add_argument("--log-every", type=int, help="Override RL optimizer logging frequency")
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

    train_on_dataset(
        cfg,
        fast_mode=args.fast,
        preview_first=not args.no_preview,
        log_every=args.log_every
    )
