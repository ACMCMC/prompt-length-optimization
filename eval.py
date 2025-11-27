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
import logging
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager
from prompt_optimization.plotting import plot_eval_trace, save_trace_csv
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_trained_model(model_path):
    """Load a trained policy model from disk."""
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint['model_name']
    
    # Initialize agent and optimizer  
    # For evaluation, we need to initialize with config values even though we're loading weights
    # Read from checkpoint config if available, otherwise use defaults from train config
    train_cfg = checkpoint.get('config', {}).get('train', {})
    policy_cfg = train_cfg.get('policy', {})
    grpo_cfg = train_cfg.get('grpo', train_cfg.get('ppo', {}))  # Support both 'grpo' and legacy 'ppo' keys
    
    epsilon = train_cfg.get('epsilon', 0.3)
    epsilon_decay = train_cfg.get('epsilon_decay', 0.998)
    epsilon_min = train_cfg.get('epsilon_min', 0.05)
    entropy_coef = train_cfg.get('entropy_coef', 0.05)
    temperature = train_cfg.get('temperature', 1.5)
    grpo_clip = grpo_cfg.get('clip', 0.2)
    grpo_epochs = grpo_cfg.get('epochs', 4)
    grpo_gamma = grpo_cfg.get('gamma', 0.99)
    grpo_gae_lambda = grpo_cfg.get('gae_lambda', 0.95)
    grpo_value_coef = grpo_cfg.get('value_coef', 0.5)
    policy_hidden_size = policy_cfg.get('hidden_size', 64)
    value_init_bias = policy_cfg.get('value_init_bias', -1000.0)
    value_init_gain = policy_cfg.get('value_init_gain', 0.1)
    max_grad_norm = grpo_cfg.get('max_grad_norm', 0.5)
    
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(
        agent,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        entropy_coef=entropy_coef,
        temperature=temperature,
        grpo_clip=grpo_clip,
        grpo_epochs=grpo_epochs,
        grpo_gamma=grpo_gamma,
        grpo_gae_lambda=grpo_gae_lambda,
        grpo_value_coef=grpo_value_coef,
        policy_hidden_size=policy_hidden_size,
        value_init_bias=value_init_bias,
        value_init_gain=value_init_gain,
        max_grad_norm=max_grad_norm
    )
    
    # Load the trained policy weights
    optimizer.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    
    return agent, optimizer, checkpoint

def evaluate_prompt(cfg, agent, optimizer):
    """Evaluate a single prompt and return results."""
    eval_cfg = cfg['eval']
    test_prompt = eval_cfg['test_prompt']
    init_len = eval_cfg['init_len']
    max_suffix_len = eval_cfg.get('max_suffix_len', 64)
    max_policy_steps = eval_cfg['max_policy_steps']
    optimization_mode = eval_cfg.get('optimization_mode', cfg.get('train', {}).get('optimization_mode', 'continuous'))
    # Get GCG config from eval or fall back to train config
    eval_gcg_cfg = eval_cfg.get('gcg', {})
    train_gcg_cfg = cfg.get('train', {}).get('gcg', {})
    gcg_cfg = {**train_gcg_cfg, **eval_gcg_cfg}  # eval overrides train
    gcg_top_k = gcg_cfg.get('top_k', 16)
    gcg_batch_size = gcg_cfg.get('batch_size', 32)
    gcg_steps = gcg_cfg.get('steps', 5)
    opt_mode = optimization_mode.lower()
    if 'ppo' in opt_mode:
        # PPO method removed during refactoring, fall back to standard optimization
        print(f"Warning: PPO mode requested but not available. Using standard {opt_mode} mode instead.")
        best_prompts_batch, best_rewards_batch, _, _ = optimizer.optimize_prompts_batch(
            target_completions=[test_prompt],
            episodes=1,
            steps_per_episode=max_policy_steps,
            initial_prompt_length=init_len,
            lr_embeddings=0.01,
            alpha=cfg.get('train', {}).get('alpha', 1.0),
            beta=cfg.get('train', {}).get('beta', 0.1),
            mode='continuous' if 'continuous' in opt_mode else 'discrete',
            batch_size=1,
            max_suffix_len=max_suffix_len,
            init_len=init_len
        )
        best_prompt = best_prompts_batch[0] if best_prompts_batch else []
        completion_tokens = agent.tokenizer.encode(test_prompt, add_special_tokens=False)
        best_likelihood = agent.get_completion_likelihood(best_prompt, completion_tokens) if best_prompt else 0.0
    else:
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
            log_every=0,  # No logging during evaluation
            optimization_mode=optimization_mode,
            gcg_top_k=gcg_top_k,
            gcg_batch_size=gcg_batch_size,
            gcg_steps=gcg_steps
        )
    
    # Calculate reward (negative of the combined loss)
    alpha = cfg.get('train', {}).get('alpha', 1.0)
    beta = cfg.get('train', {}).get('beta', 0.1)
    # Using logarithmic length penalty: log(1 + length) for more penalizing effect
    import math
    reward = alpha * best_likelihood - beta * math.log1p(len(best_prompt))
    
    # Decode the compressed prompt for display
    try:
        # Convert best_prompt tensor to list of token IDs
        if torch.is_tensor(best_prompt):
            best_prompt_tokens = best_prompt.cpu().tolist()
        elif isinstance(best_prompt, list):
            best_prompt_tokens = best_prompt
        else:
            best_prompt_tokens = []
        
        if best_prompt_tokens:
            compressed_prompt = agent.tokenizer.decode(best_prompt_tokens, skip_special_tokens=True)
        else:
            compressed_prompt = ''
    except Exception as e:
        logger.warning(f"Failed to decode compressed prompt: {e}")
        compressed_prompt = ''  # Fallback if decoding fails
    
    # Get length properly
    if torch.is_tensor(best_prompt):
        prompt_length = len(best_prompt)
    elif isinstance(best_prompt, list):
        prompt_length = len(best_prompt)
    else:
        prompt_length = 0
    
    return {
        'length': prompt_length,
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
    optimization_mode = eval_cfg.get('optimization_mode', cfg.get('train', {}).get('optimization_mode', 'continuous'))
    # Get GCG config from eval or fall back to train config
    eval_gcg_cfg = eval_cfg.get('gcg', {})
    train_gcg_cfg = cfg.get('train', {}).get('gcg', {})
    gcg_cfg = {**train_gcg_cfg, **eval_gcg_cfg}  # eval overrides train
    gcg_top_k = gcg_cfg.get('top_k', 16)
    gcg_batch_size = gcg_cfg.get('batch_size', 32)
    gcg_steps = gcg_cfg.get('steps', 5)
    
    results_file = eval_cfg.get('results_file', 'results/dataset_eval_results.csv')
    
    seed = cfg.get('seed', 2262)
    ds_cfg = cfg.get('dataset', {})
    
    print(f"Dataset evaluation with {max_test_prompts} test prompts")
    print(f"Model: {model_path}")
    
    # Determine dataset to evaluate (default: toxicchat)
    dataset_name = eval_cfg.get('dataset', cfg.get('dataset', {}).get('name', 'toxicchat'))

    # Load test prompts using proper test split or AdvBench if requested
    if dataset_name.lower() == 'advbench':
        from datasets import load_dataset
        print("Loading AdvBench test set...")
        raw = load_dataset("walledai/AdvBench", split='train')
        PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
        COMPLETION_KEYS = ["target", "completion", "output", "response", "answer"]
        test_prompts = []
        for ex in raw:
            base = None
            for k in PROMPT_KEYS:
                if k in ex and ex[k]:
                    base = ex[k]
                    break
            if base is None:
                continue
            target = None
            for k in COMPLETION_KEYS:
                if k in ex and ex[k] is not None:
                    target = ex[k]
                    break
            if target is None:
                target = ""
            test_prompts.append({'base': base.strip(), 'target': target.strip()})
        import random as _rand
        _rand.seed(seed)
        _rand.shuffle(test_prompts)
        test_prompts = test_prompts[:max_test_prompts]
        print(f"Loaded {len(test_prompts)} AdvBench test examples")
    else:
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
    
    # Get PPO parameters from config (for backward compatibility, though PPO is removed)
    ppo_cfg = cfg.get('train', {}).get('ppo', {})
    ppo_epochs = ppo_cfg.get('epochs', 1)
    ppo_clip = ppo_cfg.get('clip', 0.2)
    ppo_gamma = ppo_cfg.get('gamma', 0.99)
    ppo_lambda = ppo_cfg.get('gae_lambda', 0.95)
    ppo_value_coef = ppo_cfg.get('value_coef', 0.5)
    ppo_entropy_coef = ppo_cfg.get('entropy_coef', 0.01)
    
    # Check if we should save plots
    save_plots = not eval_cfg.get('no_plots', True)
    plots_format = eval_cfg.get('plots_format', 'pdf')
    plots_prefix = eval_cfg.get('plots_prefix', 'eval')
    
    # Evaluate in batches to allow vectorized/batched optimizers
    results = []
    batch_size = eval_cfg.get('batch_size', cfg.get('train', {}).get('batch_size', 8))

    def _get_from_batch_list(batch_list, idx_in_batch, default=0.0):
        """Extract value at idx_in_batch from batch-level list."""
        if batch_list and isinstance(batch_list, (list, tuple)) and idx_in_batch < len(batch_list):
            val = batch_list[idx_in_batch]
            return val if val is not None else default
        return default

    def _extract_final_likelihood(trace_obj, idx_in_batch=0):
        """Extract final likelihood from trace, preferring best_likelihoods."""
        if not trace_obj:
            return 0.0
        
        # Get the dict to extract from (either trace_obj itself or last episode)
        ep_dict = trace_obj if isinstance(trace_obj, dict) else (trace_obj[-1] if isinstance(trace_obj, (list, tuple)) and len(trace_obj) > 0 else None)
        
        if not isinstance(ep_dict, dict):
        return 0.0
        
        # Prefer best_likelihoods, fallback to likelihoods, then single 'likelihood' key
        if 'likelihood' in ep_dict:
            return float(ep_dict['likelihood'])
        best_ll = _get_from_batch_list(ep_dict.get('best_likelihoods'), idx_in_batch, None)
        if best_ll is not None:
            return float(best_ll)
        return float(_get_from_batch_list(ep_dict.get('likelihoods'), idx_in_batch, 0.0))

    for batch_start in range(0, len(test_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(test_prompts))
        batch = test_prompts[batch_start:batch_end]
        print(f"\n{'='*60}")
        print(f"Evaluating prompts {batch_start+1}-{batch_end} (batch size={len(batch)})")
        print(f"{'='*60}")

        # Prepare inputs
        targets = []
        bases = []
        raw_inputs = []
        for tp in batch:
            if isinstance(tp, dict):
                bases.append(tp.get('base', ''))
                targets.append(tp.get('target', ''))
                raw_inputs.append(tp)
            else:
                bases.append('')
                targets.append(tp)
                raw_inputs.append(tp)

        opt_mode = optimization_mode.lower()
        try:
            if 'ppo' in opt_mode:
                # PPO method removed during refactoring, fall back to standard optimization
                print(f"Warning: PPO mode requested but not available. Using standard {opt_mode} mode instead.")
                best_prompts_batch, best_rewards_batch, traces_batch, _ = optimizer.optimize_prompts_batch(
                    target_completions=targets,
                    episodes=1,
                    steps_per_episode=max_policy_steps,
                    initial_prompt_length=init_len,
                    lr_embeddings=0.01,
                    alpha=alpha,
                    beta=beta,
                    mode='continuous' if 'continuous' in opt_mode else 'discrete',
                    batch_size=len(targets),
                    max_suffix_len=max_suffix_len,
                    init_len=init_len
                )
            elif opt_mode == 'continuous':
                best_prompts_batch, best_rewards_batch, traces_batch, _ = optimizer.optimize_prompts_batch(
                    target_completions=targets,
                    episodes=1,
                    steps_per_episode=max_policy_steps,
                    initial_prompt_length=init_len,
                    lr_embeddings=0.01,
                    alpha=alpha,
                    beta=beta,
                    mode='continuous',
                    batch_size=len(targets),
                    max_suffix_len=max_suffix_len,
                    init_len=init_len
                )
            elif opt_mode == 'discrete':
                best_prompts_batch, best_rewards_batch, traces_batch, _ = optimizer.optimize_prompts_batch(
                    target_completions=targets,
                    episodes=1,
                    steps_per_episode=max_policy_steps,
                    initial_prompt_length=init_len,
                    lr_embeddings=0.01,
                    alpha=alpha,
                    beta=beta,
                    mode='discrete',
                    batch_size=len(targets),
                    max_suffix_len=max_suffix_len,
                    init_len=init_len
                )
            else:
                # fallback: run sequentially
                best_prompts_batch = []
                best_rewards_batch = []
                traces_batch = []
                for tp in raw_inputs:
                    if isinstance(tp, dict):
                        base_text = tp.get('base', '')
                        target_text = tp.get('target', '')
                        best_prompt, best_reward, trace = optimizer.optimize_prompt(
                            target_completion=target_text,
                            episodes=1,
                            steps_per_episode=max_policy_steps,
                            initial_prompt_length=init_len,
                            lr_embeddings=0.01,
                            lr_policy=0.0003,
                            alpha=alpha,
                            beta=beta,
                            log_every=0,
                            optimization_mode=optimization_mode,
                            gcg_top_k=gcg_top_k,
                            gcg_batch_size=gcg_batch_size,
                            gcg_steps=gcg_steps,
                            base_prompt=base_text
                        )
                    else:
                        best_prompt, best_reward, trace = optimizer.optimize_prompt(
                            tp,
                            episodes=1,
                            steps_per_episode=max_policy_steps,
                            initial_prompt_length=init_len,
                            lr_embeddings=0.01,
                            lr_policy=0.0003,
                            alpha=alpha,
                            beta=beta,
                            log_every=0,
                            optimization_mode=optimization_mode,
                            gcg_top_k=gcg_top_k,
                            gcg_batch_size=gcg_batch_size,
                            gcg_steps=gcg_steps
                        )
                    best_prompts_batch.append(best_prompt)
                    best_rewards_batch.append(best_reward)
                    traces_batch.append(trace)

            # Unpack batch results
            trace_source = traces_batch if traces_batch is not None else []
            for idx_in_batch, best_prompt in enumerate(best_prompts_batch):
                global_idx = batch_start + idx_in_batch
                best_reward = best_rewards_batch[idx_in_batch]

                # Determine texts
                input_prompt_text = bases[idx_in_batch] if bases[idx_in_batch] else (raw_inputs[idx_in_batch] if isinstance(raw_inputs[idx_in_batch], str) else '')
                target_completion_text = targets[idx_in_batch]

                # Get trace for this specific prompt (traces_batch is a list of traces, one per prompt)
                # Note: each trace is a list of episode dicts, where each dict has batch-level lists
                trace = trace_source[idx_in_batch] if idx_in_batch < len(trace_source) else []
                final_likelihood = _extract_final_likelihood(trace, idx_in_batch)
                completion_tokens = agent.tokenizer.encode(target_completion_text, add_special_tokens=False)
                avg_likelihood = float(final_likelihood) / max(len(completion_tokens), 1) if completion_tokens else float('nan')
                
                # Convert best_prompt tensor to list of token IDs for decoding
                if torch.is_tensor(best_prompt):
                    best_prompt_tokens = best_prompt.cpu().tolist()
                    best_prompt_length = len(best_prompt)
                elif isinstance(best_prompt, list):
                    best_prompt_tokens = best_prompt
                    best_prompt_length = len(best_prompt)
                else:
                    best_prompt_tokens = []
                    best_prompt_length = 0
                
                final_reward = alpha * final_likelihood - beta * best_prompt_length

                # Decode optimized prompt from token IDs
                try:
                    if best_prompt_tokens:
                        optimized_full_text = agent.tokenizer.decode(best_prompt_tokens, skip_special_tokens=True)
                    else:
                        optimized_full_text = ''
                except Exception as e:
                    logger.warning(f"Failed to decode optimized prompt: {e}")
                    optimized_full_text = ''

                # Extract optimized suffix (part after the base prompt)
                optimized_suffix_text = ''
                try:
                    if best_prompt_tokens and isinstance(input_prompt_text, str) and input_prompt_text:
                        base_ids = agent.tokenizer.encode(input_prompt_text, add_special_tokens=False)
                        if len(best_prompt_tokens) >= len(base_ids) and best_prompt_tokens[:len(base_ids)] == base_ids:
                            suffix_ids = best_prompt_tokens[len(base_ids):]
                            optimized_suffix_text = agent.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                        else:
                            # If base doesn't match, try to extract suffix by text replacement
                            optimized_suffix_text = optimized_full_text.replace(input_prompt_text, '', 1).strip()
                    else:
                        optimized_suffix_text = optimized_full_text
                except Exception as e:
                    logger.warning(f"Failed to extract optimized suffix: {e}")
                    optimized_suffix_text = optimized_full_text

                result_row = {
                    'prompt_id': global_idx,
                    'prompt_text': input_prompt_text,
                    'prompt_length_chars': len(input_prompt_text) if isinstance(input_prompt_text, str) else 0,
                    'initial_tokens': init_len,
                    'final_tokens': best_prompt_length,
                    'compression_ratio': (init_len - best_prompt_length) / init_len * 100 if init_len > 0 else 0.0,
                    'final_likelihood': float(final_likelihood),
                    'avg_likelihood': float(avg_likelihood) if not (isinstance(avg_likelihood, float) and np.isnan(avg_likelihood)) else float('nan'),
                    'final_reward': float(final_reward),
                    'target_completion': target_completion_text,
                    'optimized_full_prompt': optimized_full_text,
                    'optimized_suffix': optimized_suffix_text,
                    'compressed_prompt': optimized_full_text  # Use decoded text instead of tensor string
                }
                results.append(result_row)

                logger.debug(f"Result: {init_len}→{result_row['final_tokens']} tokens ({result_row['compression_ratio']:.1f}% compression)")
                logger.debug(f"Likelihood: {final_likelihood:.3f}, Avg token likelihood: {result_row['avg_likelihood'] if not np.isnan(result_row['avg_likelihood']) else 'N/A'}, Reward: {final_reward:.3f}")

                if save_plots and trace:
                    try:
                        # Convert batch-level trace to per-prompt trace for plotting
                        plot_trace = []
                        if isinstance(trace, (list, tuple)):
                            for ep_dict in trace:
                                if not isinstance(ep_dict, dict):
                                    continue
                                
                                # Extract values for this prompt from batch-level lists
                                likelihood = _get_from_batch_list(ep_dict.get('likelihoods'), idx_in_batch, 0.0)
                                best_likelihood = _get_from_batch_list(ep_dict.get('best_likelihoods'), idx_in_batch, likelihood)
                                
                                plot_trace.append({
                                    'step': ep_dict.get('step', ep_dict.get('episode', len(plot_trace))),
                                    'likelihood': float(likelihood),
                                    'best_likelihood': float(best_likelihood),
                                    'length': int(_get_from_batch_list(ep_dict.get('lengths'), idx_in_batch, 0))
                                })
                        
                        if plot_trace:
                            plot_path = plot_eval_trace(plot_trace, out_dir="results/traces", 
                            prefix=f"{plots_prefix}_prompt_{global_idx:03d}",
                                                       alpha=alpha, beta=beta)
                            logger.debug(f"  Plot saved: {plot_path}")
                    except Exception as plot_err:
                        import traceback
                        logger.warning(f"  Could not generate plot: {plot_err}")
                        traceback.print_exc()

        except Exception as e:
            import traceback
            print(f"Error evaluating prompts {batch_start+1}-{batch_end}: {repr(e)}")
            traceback.print_exc()
            # add error rows for each prompt in the batch
            for j in range(len(batch)):
                idx = batch_start + j
                tp = batch[j]
                results.append({
                    'prompt_id': idx,
                    'prompt_text': tp if isinstance(tp, str) else tp.get('base', ''),
                    'prompt_length_chars': len(tp) if isinstance(tp, str) else len(tp.get('base', '')),
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
