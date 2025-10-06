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
from datasets import load_dataset
from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
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
        alpha=cfg.get('alpha', 1.0),
        beta=cfg.get('beta', 0.1),
        log_every=0  # No logging during evaluation
    )
    
    # Calculate reward (negative of the combined loss)
    alpha = cfg.get('alpha', 1.0)
    beta = cfg.get('beta', 0.1)
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

def load_test_prompts(max_samples=20, min_length=30, max_length=200):
    """Load test prompts from the toxic-chat dataset."""
    print(f"Loading test prompts from toxic-chat dataset...")
    
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split='train')
    print(f"Loaded {len(dataset)} samples")
    
    # Extract model outputs and filter by length
    prompts = []
    for example in dataset:
        # Use model_output column as the text to compress
        prompt = example.get('model_output', '')
        if prompt and min_length <= len(prompt) <= max_length:
            prompts.append(prompt.strip())
            if len(prompts) >= max_samples:
                break
    
    print(f"Selected {len(prompts)} test prompts (length {min_length}-{max_length} chars)")
    return prompts

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
    
    print(f"Dataset evaluation with {max_test_prompts} test prompts")
    print(f"Model: {model_path}")
    
    # Load test prompts
    test_prompts = load_test_prompts(
        max_samples=max_test_prompts,
        min_length=min_prompt_length,
        max_length=max_prompt_length
    )
    
    if not test_prompts:
        raise ValueError("No test prompts found")
    
    # Load trained model
    agent, optimizer, checkpoint = load_trained_model(model_path)
    
    print(f"Loaded model with {len(checkpoint.get('training_rewards', []))} training examples")
    print(f"Original best training reward: {checkpoint.get('best_reward', 'unknown'):.3f}")
    
    # Evaluate on each test prompt
    results = []
    
    for i, test_prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"Evaluating {i+1}/{len(test_prompts)}")
        print(f"Test prompt: '{test_prompt[:100]}{'...' if len(test_prompt) > 100 else ''}'")
        print(f"{'='*60}")
        
        try:
            # Evaluate using the optimizer directly
            best_prompt, best_likelihood, trace = optimizer.optimize_prompt(
                test_prompt,
                episodes=1,  # Single episode for evaluation
                steps_per_episode=max_policy_steps,
                initial_prompt_length=init_len,
                lr_embeddings=0.01,
                lr_policy=0.0003,
                alpha=cfg.get('alpha', 1.0),
                beta=cfg.get('beta', 0.1),
                log_every=0  # No logging during evaluation
            )
            
            # Calculate reward
            alpha = cfg.get('alpha', 1.0)
            beta = cfg.get('beta', 0.1)
            reward = alpha * best_likelihood - beta * len(best_prompt)
            
            # Store results
            result_row = {
                'prompt_id': i,
                'prompt_text': test_prompt,
                'prompt_length_chars': len(test_prompt),
                'initial_tokens': init_len,
                'final_tokens': len(best_prompt),
                'compression_ratio': (init_len - len(best_prompt)) / init_len * 100,
                'final_likelihood': float(best_likelihood),
                'final_reward': float(reward),
                'compressed_prompt': str(best_prompt)
            }
            results.append(result_row)
            
            print(f"Result: {init_len}→{len(best_prompt)} tokens ({result_row['compression_ratio']:.1f}% compression)")
            print(f"Likelihood: {best_likelihood:.3f}, Reward: {reward:.3f}")
            
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