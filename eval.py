#!/usr/bin/env python3
"""
Evaluate the trained policy on held-out examples from the toxic-chat dataset.
"""
import torch
import torch.nn.functional as F
import argparse
import os
import yaml
import random
import numpy as np
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
from dataset_utils import ToxicChatDatasetManager
from plot_utils import plot_eval_trace, save_trace_csv
import pandas as pd

def load_trained_model(model_path):
    """Load a trained policy model from disk."""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint['model_name']
    
    # Initialize agent and optimizer  
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Load the trained policy weights
    optimizer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    
    return agent, optimizer, checkpoint

def evaluate_prompt(cfg, agent, optimizer):
    """Evaluate a single prompt and return results."""
    eval_cfg = cfg['eval']
    test_prompt = eval_cfg['test_prompt']
    init_len = eval_cfg['init_len']
    max_policy_steps = eval_cfg['max_policy_steps']
    
    # Use optimize_prompt with minimal steps for evaluation
    best_prompt, best_likelihood, trace = optimizer.optimize_prompt(
        test_prompt,
        episodes=1,  # Single episode for evaluation
        steps_per_episode=max_policy_steps,
        initial_prompt_length=init_len,
        lr_embeddings=0.01,
        lr_policy=0.0003,
        alpha=cfg.get('train', {}).get('alpha', 1.0),
        beta=cfg.get('train', {}).get('beta', 0.1),
        log_every=0  # No logging during evaluation
    )
    
    # Calculate reward (negative of the combined loss)
    alpha = cfg.get('train', {}).get('alpha', 1.0)
    beta = cfg.get('train', {}).get('beta', 0.1)
    reward = alpha * best_likelihood - beta * len(best_prompt)
    
    # Decode the compressed prompt for display
    try:
        tokens = agent.tokenizer.encode(test_prompt)[:len(best_prompt)]
        compressed_prompt = agent.tokenizer.decode(tokens)
    except:
        compressed_prompt = str(best_prompt)  # Fallback if decoding fails
    
    return {
        'length': len(best_prompt),
        'likelihood': float(best_likelihood),
        'reward': float(reward),
        'compressed_prompt': compressed_prompt
    }

def load_test_prompts(seed=2262, max_samples=20, min_length=30, max_length=200, ds_cfg=None):
    """Load test prompts from the toxic-chat dataset using proper split."""
    print(f"Loading test prompts from toxic-chat dataset...")
    
    # Use the dataset manager to get the proper test split
    dataset_manager = ToxicChatDatasetManager(seed=seed)
    ds_cfg = ds_cfg or {}
    test_prompts = dataset_manager.load_test_set(
        min_length=min_length,
        max_length=max_length,
        max_samples=max_samples,
        train_ratio=ds_cfg.get('train_ratio', 0.7),
        val_ratio=ds_cfg.get('val_ratio', 0.15),
        test_ratio=ds_cfg.get('test_ratio', 0.15),
        use_cache=True
    )
    
    print(f"Loaded {len(test_prompts)} test prompts (length {min_length}-{max_length} chars)")
    return test_prompts

def evaluate_on_dataset(cfg, model_path):
    """Evaluate the trained policy on multiple test examples."""
    eval_cfg = cfg['eval']
    
    # Load test parameters
    max_test_prompts = eval_cfg.get('max_test_prompts', 20)
    init_len = eval_cfg.get('init_len', 32)
    max_policy_steps = eval_cfg.get('max_policy_steps', 50)
    min_prompt_length = eval_cfg.get('min_prompt_length', 20)
    max_prompt_length = eval_cfg.get('max_prompt_length', 200)
    
    results_file = eval_cfg.get('results_file', 'results/dataset_eval_results.csv')
    
    seed = cfg.get('seed', 2262)
    ds_cfg = cfg.get('dataset', {})
    
    print(f"Dataset evaluation with {max_test_prompts} test prompts")
    print(f"Model: {model_path}")
    
    # Load test prompts using proper test split
    test_prompts = load_test_prompts(
        seed=seed,
        max_samples=max_test_prompts,
        min_length=min_prompt_length,
        max_length=max_prompt_length,
        ds_cfg=ds_cfg
    )
    
    if not test_prompts:
        raise ValueError("No test prompts found")
    
    # Load trained model
    agent, optimizer, checkpoint = load_trained_model(model_path)
    
    print(f"Loaded model with {len(checkpoint.get('training_rewards', []))} training examples")
    best_reward_val = checkpoint.get('best_reward', None)
    if isinstance(best_reward_val, (int, float)):
        print(f"Original best training reward: {best_reward_val:.3f}")
    else:
        print("Original best training reward: unknown")
    
    # Get alpha and beta for reward calculation
    alpha = cfg.get('train', {}).get('alpha', 1.0)
    beta = cfg.get('train', {}).get('beta', 0.1)
    
    # Check if we should save plots
    save_plots = not eval_cfg.get('no_plots', True)
    plots_format = eval_cfg.get('plots_format', 'pdf')
    plots_prefix = eval_cfg.get('plots_prefix', 'eval')
    
    # Evaluate on each test prompt
    results = []
    
    for i, test_prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"Evaluating {i+1}/{len(test_prompts)}")
        print(f"Test prompt: '{test_prompt[:100]}{'...' if len(test_prompt) > 100 else ''}'")
        print(f"{'='*60}")
        
        try:
            # Evaluate using the optimizer directly
            best_prompt, best_reward, trace = optimizer.optimize_prompt(
                test_prompt,
                episodes=1,  # Single episode for evaluation
                steps_per_episode=max_policy_steps,
                initial_prompt_length=init_len,
                lr_embeddings=0.01,
                lr_policy=0.0003,
                alpha=alpha,
                beta=beta,
                log_every=0  # No logging during evaluation
            )
            
            # Get final likelihood from trace (best_reward is actually the final reward)
            final_likelihood = trace[-1]['likelihood'] if trace else 0.0
            final_reward = alpha * final_likelihood - beta * len(best_prompt)
            
            # Store results
            result_row = {
                'prompt_id': i,
                'prompt_text': test_prompt,
                'prompt_length_chars': len(test_prompt),
                'initial_tokens': init_len,
                'final_tokens': len(best_prompt),
                'compression_ratio': (init_len - len(best_prompt)) / init_len * 100,
                'final_likelihood': float(final_likelihood),
                'final_reward': float(final_reward),
                'compressed_prompt': str(best_prompt)
            }
            results.append(result_row)
            
            print(f"Result: {init_len}→{len(best_prompt)} tokens ({result_row['compression_ratio']:.1f}% compression)")
            print(f"Likelihood: {final_likelihood:.3f}, Reward: {final_reward:.3f}")
            
            # Generate per-prompt plot if enabled
            if save_plots and trace:
                try:
                    plot_path = plot_eval_trace(
                        trace,
                        out_dir="results/traces",
                        prefix=f"{plots_prefix}_prompt_{i:03d}",
                        alpha=alpha,
                        beta=beta
                    )
                    print(f"  Plot saved: {plot_path}")
                except Exception as plot_err:
                    print(f"  Warning: Could not generate plot: {plot_err}")
            
        except Exception as e:
            print(f"Error evaluating prompt {i+1}: {e}")
            # Add error result
            results.append({
                'prompt_id': i,
                'prompt_text': test_prompt,
                'prompt_length_chars': len(test_prompt),
                'initial_tokens': init_len,
                'final_tokens': init_len,
                'compression_ratio': 0.0,
                'final_likelihood': float('nan'),
                'final_reward': float('nan'),
                'compressed_prompt': 'ERROR',
                'error': str(e)
            })
            continue
    
    # Save results to CSV
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    df.to_csv(results_file, index=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("DATASET EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts evaluated: {len(results)}")
    print(f"Results saved to: {results_file}")
    
    # Filter out error results for stats
    valid_results = df[df['final_likelihood'].notna()]
    if len(valid_results) > 0:
        print(f"Valid evaluations: {len(valid_results)}")
        print(f"Average compression ratio: {valid_results['compression_ratio'].mean():.1f}% ± {valid_results['compression_ratio'].std():.1f}%")
        print(f"Average final likelihood: {valid_results['final_likelihood'].mean():.3f} ± {valid_results['final_likelihood'].std():.3f}")
        print(f"Average final reward: {valid_results['final_reward'].mean():.3f} ± {valid_results['final_reward'].std():.3f}")
        print(f"Compression distribution:")
        print(f"  No compression (0%): {(valid_results['compression_ratio'] == 0).sum()} prompts")
        print(f"  Light compression (1-25%): {((valid_results['compression_ratio'] > 0) & (valid_results['compression_ratio'] <= 25)).sum()} prompts")
        print(f"  Medium compression (26-50%): {((valid_results['compression_ratio'] > 25) & (valid_results['compression_ratio'] <= 50)).sum()} prompts")
        print(f"  Heavy compression (>50%): {(valid_results['compression_ratio'] > 50).sum()} prompts")
    
    if len(valid_results) != len(results):
        print(f"Failed evaluations: {len(results) - len(valid_results)}")
    
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--model_path", type=str, help="Path to trained model (overrides config)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Use provided model path or get from config
    model_path = args.model_path or cfg['train'].get('save_path', 'models/trained_policy.pt')
    
    evaluate_on_dataset(cfg, model_path)