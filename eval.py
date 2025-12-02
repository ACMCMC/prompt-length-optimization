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
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager
from prompt_optimization.plotting import plot_eval_trace, save_trace_csv
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

def load_trained_model(model_path, cfg, reward_cfg=None):
    """Load a trained policy model from disk.
    
    Args:
        model_path: Path to the checkpoint file
        cfg: Full config dict (to reconstruct optimizer with matching architecture)
        reward_cfg: Optional reward config override
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint['model_name']
    
    # Initialize agent
    agent = PromptRLAgent(model_name=model_name)
    
    # Infer network architecture from checkpoint's state_dict
    policy_state = checkpoint['policy_state_dict']
    
    # Get dimensions from first layer: weight shape is [hidden1, state_dim]
    first_layer_shape = policy_state['0.weight'].shape
    hidden1 = first_layer_shape[0]
    state_dim = first_layer_shape[1]
    
    # Get second hidden layer size from layer 2
    hidden2 = policy_state['2.weight'].shape[0]
    
    # Get action dim from output layer
    action_dim = policy_state['4.weight'].shape[0]
    
    print(f"[Checkpoint] Detected architecture: state_dim={state_dim}, hidden=[{hidden1}, {hidden2}], actions={action_dim}")
    
    # Build matching network architecture
    policy_net = torch.nn.Sequential(
        torch.nn.Linear(state_dim, hidden1),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden1, hidden2),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden2, action_dim)
    ).to(agent.device)
    
    # Build matching value network if present
    if 'value_state_dict' in checkpoint:
        value_state = checkpoint['value_state_dict']
        v_hidden1 = value_state['0.weight'].shape[0]
        v_hidden2 = value_state['2.weight'].shape[0]
        value_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, v_hidden1),
            torch.nn.Tanh(),
            torch.nn.Linear(v_hidden1, v_hidden2),
            torch.nn.Tanh(),
            torch.nn.Linear(v_hidden2, 1)
        ).to(agent.device)
    else:
        # Default value network matching policy hidden sizes
        value_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden1),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden1, hidden2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden2, 1)
        ).to(agent.device)
    
    # Load weights
    policy_net.load_state_dict(policy_state)
    if 'value_state_dict' in checkpoint:
        value_net.load_state_dict(checkpoint['value_state_dict'])
    
    # Create optimizer with default settings (we'll override the networks)
    train_cfg = cfg.get('train', {})
    ppo_cfg = train_cfg.get('ppo', {})
    use_complex_network = ppo_cfg.get('use_complex_network', False)
    
    optimizer = LengthPolicyOptimizer(agent, reward_cfg=reward_cfg, use_complex_network=use_complex_network)
    
    # Replace networks with the ones we built from checkpoint
    optimizer.policy_net = policy_net
    optimizer.value_net = value_net
    optimizer.state_dim = state_dim
    
    # Recreate optimizers for the new networks
    optimizer.policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    optimizer.value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
    
    # Set curriculum exploration parameters from config
    curriculum_cfg = train_cfg.get('curriculum', {})
    optimizer.exploration_start = curriculum_cfg.get('exploration_start', 0.5)
    optimizer.exploration_end = curriculum_cfg.get('exploration_end', 0.05)
    optimizer.exploration_decay_episodes = curriculum_cfg.get('exploration_decay_episodes', 50)
    
    # Set reward shaping parameters from config
    reward_shaping_cfg = train_cfg.get('reward_shaping', {})
    optimizer.reward_mode = reward_shaping_cfg.get('mode', 'efficiency')
    optimizer.ll_threshold = reward_shaping_cfg.get('ll_threshold', -10.0)
    optimizer.hard_cap_ll = reward_shaping_cfg.get('hard_cap_ll', -8.0)
    optimizer.length_bonus_scale = reward_shaping_cfg.get('length_bonus_scale', 3.0)
    
    print(f"Loaded policy: hidden=[{hidden1}, {hidden2}], state_dim={state_dim}")
    print(f"Reward shaping: mode={optimizer.reward_mode}, ll_threshold={optimizer.ll_threshold}")
    
    return agent, optimizer, checkpoint

def evaluate_prompt(cfg, agent, optimizer):
    """Evaluate a single prompt and return results."""
    eval_cfg = cfg['eval']
    test_prompt = eval_cfg['test_prompt']
    init_len = eval_cfg['init_len']
    max_policy_steps = eval_cfg['max_policy_steps']
    optimization_mode = eval_cfg.get('optimization_mode', cfg.get('train', {}).get('optimization_mode', 'continuous'))
    gcg_top_k = eval_cfg.get('gcg_top_k', cfg.get('train', {}).get('gcg_top_k', 16))
    gcg_batch_size = eval_cfg.get('gcg_batch_size', cfg.get('train', {}).get('gcg_batch_size', 32))
    gcg_steps = eval_cfg.get('gcg_steps', cfg.get('train', {}).get('gcg_steps', 5))
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
            gcg_top_k=gcg_top_k,
            gcg_candidate_size=gcg_steps
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
    completion_tokens = agent.tokenizer.encode(test_prompt, add_special_tokens=False)
    if isinstance(best_prompt, torch.Tensor):
        best_prompt_tokens = best_prompt.detach().tolist()
    elif isinstance(best_prompt, list):
        best_prompt_tokens = best_prompt
    else:
        best_prompt_tokens = []
    
    # Use compute_shaped_reward (GCG branch API)
    reward = optimizer.compute_shaped_reward(
        likelihoods=torch.tensor([best_likelihood], device=agent.device),
        lengths=torch.tensor([len(best_prompt_tokens)], device=agent.device),
        initial_prompt_length=init_len,
        alpha=alpha,
        beta=beta
    )[0].item()
    
    # Decode the compressed prompt for display
    try:
        compressed_prompt = agent.tokenizer.decode(best_prompt_tokens, skip_special_tokens=True) if best_prompt_tokens else ''
    except Exception:
        compressed_prompt = str(best_prompt_tokens)  # Fallback if decoding fails
    
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

def init_wandb_eval(cfg, model_path):
    """Initialize wandb for evaluation run."""
    if not WANDB_AVAILABLE:
        print("wandb not installed, skipping logging")
        return None
    
    eval_cfg = cfg.get('eval', {})
    wandb_cfg = eval_cfg.get('wandb', {})
    
    if not wandb_cfg.get('enable', False):
        print("wandb disabled in config")
        return None
    
    project = wandb_cfg.get('project', cfg.get('train', {}).get('wandb', {}).get('project', 'prompt-length-optimization'))
    run_name = wandb_cfg.get('run_name', 'eval-' + os.path.basename(model_path).replace('.pt', ''))
    
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            'eval_config': eval_cfg,
            'model_path': model_path,
            'model_name': cfg.get('model', 'unknown'),
        },
        job_type='eval',
        tags=['eval'] + wandb_cfg.get('tags', []),
        reinit=True
    )
    print(f"wandb initialized: {run.url}")
    return run


def evaluate_on_dataset(cfg, model_path):
    """Evaluate the trained policy on multiple test examples."""
    eval_cfg = cfg['eval']
    
    # Initialize wandb if enabled
    wandb_run = init_wandb_eval(cfg, model_path)
    
    # Load test parameters
    max_test_prompts = eval_cfg.get('max_test_prompts', 20)
    init_len = eval_cfg.get('init_len', 32)
    max_suffix_len = eval_cfg.get('max_suffix_len', init_len * 2)
    max_policy_steps = eval_cfg.get('max_policy_steps', 50)
    min_prompt_length = eval_cfg.get('min_prompt_length', 20)
    max_prompt_length = eval_cfg.get('max_prompt_length', 200)
    optimization_mode = eval_cfg.get('optimization_mode', cfg.get('train', {}).get('optimization_mode', 'continuous'))
    gcg_top_k = eval_cfg.get('gcg_top_k', cfg.get('train', {}).get('gcg_top_k', 16))
    gcg_batch_size = eval_cfg.get('gcg_batch_size', cfg.get('train', {}).get('gcg_batch_size', 32))
    gcg_steps = eval_cfg.get('gcg_steps', cfg.get('train', {}).get('gcg_steps', 5))
    
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
    
    # Load trained model (pass cfg so optimizer architecture matches training)
    global_reward_cfg = cfg.get('reward', {})
    eval_reward_cfg = eval_cfg.get('reward', cfg.get('train', {}).get('reward', global_reward_cfg))
    agent, optimizer, checkpoint = load_trained_model(model_path, cfg, reward_cfg=eval_reward_cfg)
    
    # IMPORTANT: Disable exploration during evaluation (deterministic policy)
    optimizer.exploration_start = 0.0
    optimizer.exploration_end = 0.0
    optimizer.exploration_decay_episodes = 1
    print("Exploration disabled for eval (deterministic policy)")
    
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
    rl_algo = str(eval_cfg.get('rl_algo', cfg.get('train', {}).get('rl_algo', 'ppo'))).lower()
    use_ppo = rl_algo == 'ppo'
    
    # Check if we should save plots
    save_plots = not eval_cfg.get('no_plots', True)
    plots_format = eval_cfg.get('plots_format', 'pdf')
    plots_prefix = eval_cfg.get('plots_prefix', 'eval')
    
    # Evaluate in batches to allow vectorized/batched optimizers
    results = []
    batch_size = eval_cfg.get('batch_size', cfg.get('train', {}).get('batch_size', 8))

    def _extract_final_likelihood(trace_obj, idx_in_batch=0):
        """Robustly extract final likelihood for a single example from various trace shapes.

        Handles:
        - trace = [] -> 0.0
        - trace = list(dicts) where dict has key 'likelihood' (single-example traces)
        - trace = list(dicts) where dict has key 'likelihoods' (list per-batch)
        - trace = list(dicts) where dict has key 'best_likelihoods' (discrete batched)
        """
        if not trace_obj:
            return 0.0
        # If trace_obj is a dict (single aggregated trace), try common keys
        if isinstance(trace_obj, dict):
            if 'likelihood' in trace_obj:
                return float(trace_obj.get('likelihood', 0.0))
            if 'likelihoods' in trace_obj and isinstance(trace_obj['likelihoods'], (list, tuple)):
                vals = trace_obj['likelihoods']
                return float(vals[idx_in_batch]) if idx_in_batch < len(vals) else 0.0
            if 'best_likelihoods' in trace_obj and isinstance(trace_obj['best_likelihoods'], (list, tuple)):
                vals = trace_obj['best_likelihoods']
                return float(vals[idx_in_batch]) if idx_in_batch < len(vals) else 0.0

        # If trace_obj is a list of steps/episodes
        if isinstance(trace_obj, (list, tuple)) and len(trace_obj) > 0:
            last = trace_obj[-1]
            if isinstance(last, dict):
                if 'likelihood' in last:
                    return float(last.get('likelihood', 0.0))
                if 'likelihoods' in last and isinstance(last['likelihoods'], (list, tuple)):
                    vals = last['likelihoods']
                    return float(vals[idx_in_batch]) if idx_in_batch < len(vals) else 0.0
                if 'best_likelihoods' in last and isinstance(last['best_likelihoods'], (list, tuple)):
                    vals = last['best_likelihoods']
                    return float(vals[idx_in_batch]) if idx_in_batch < len(vals) else 0.0

        # fallback
        return 0.0

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
                    max_suffix_len=max_suffix_len,
                    init_len=init_len,
                    gcg_top_k=gcg_top_k,
                    gcg_candidate_size=gcg_batch_size,
                    gcg_steps_per_action=gcg_steps,
                    base_prompts=bases
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
                    use_ppo=use_ppo,
                    ppo_epochs=ppo_epochs,
                    ppo_clip=ppo_clip,
                    gamma=ppo_gamma,
                    gae_lambda=ppo_lambda,
                    value_coef=ppo_value_coef,
                    entropy_coef=ppo_entropy_coef,
                    max_suffix_len=max_suffix_len,
                    init_len=init_len,
                    gcg_top_k=gcg_top_k,
                    gcg_candidate_size=gcg_batch_size,
                    gcg_steps_per_action=gcg_steps,
                    base_prompts=bases
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
                    use_ppo=use_ppo,
                    ppo_epochs=ppo_epochs,
                    ppo_clip=ppo_clip,
                    gamma=ppo_gamma,
                    gae_lambda=ppo_lambda,
                    value_coef=ppo_value_coef,
                    entropy_coef=ppo_entropy_coef,
                    max_suffix_len=max_suffix_len,
                    init_len=init_len,
                    gcg_top_k=gcg_top_k,
                    gcg_candidate_size=gcg_batch_size,
                    gcg_steps_per_action=gcg_steps,
                    base_prompts=bases
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

                trace = trace_source
                final_likelihood = _extract_final_likelihood(trace, idx_in_batch)
                completion_tokens = agent.tokenizer.encode(target_completion_text, add_special_tokens=False)
                avg_likelihood = float(final_likelihood) / max(len(completion_tokens), 1) if completion_tokens else float('nan')
                if isinstance(best_prompt, torch.Tensor):
                    best_prompt_tokens = best_prompt.detach().tolist()
                elif isinstance(best_prompt, list):
                    best_prompt_tokens = best_prompt
                else:
                    best_prompt_tokens = []
                
                # Use compute_shaped_reward (GCG branch API)
                reward_value = optimizer.compute_shaped_reward(
                    likelihoods=torch.tensor([final_likelihood], device=agent.device),
                    lengths=torch.tensor([len(best_prompt_tokens)], device=agent.device),
                    initial_prompt_length=init_len,
                    alpha=alpha,
                    beta=beta
                )[0].item()
                final_reward = float(reward_value)

                try:
                    optimized_full_text = agent.tokenizer.decode(best_prompt_tokens, skip_special_tokens=True) if best_prompt_tokens else ''
                except Exception:
                    optimized_full_text = str(best_prompt_tokens)

                optimized_suffix_text = ''
                try:
                    if best_prompt_tokens and isinstance(input_prompt_text, str) and input_prompt_text:
                        base_ids = agent.tokenizer.encode(input_prompt_text, add_special_tokens=False)
                        if len(best_prompt_tokens) >= len(base_ids) and best_prompt_tokens[:len(base_ids)] == base_ids:
                            suffix_ids = best_prompt_tokens[len(base_ids):]
                            optimized_suffix_text = agent.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                        else:
                            optimized_suffix_text = optimized_full_text.replace(input_prompt_text, '', 1).strip()
                    else:
                        optimized_suffix_text = optimized_full_text
                except Exception:
                    optimized_suffix_text = optimized_full_text

                result_row = {
                    'prompt_id': global_idx,
                    'prompt_text': input_prompt_text,
                    'prompt_length_chars': len(input_prompt_text) if isinstance(input_prompt_text, str) else 0,
                    'initial_tokens': init_len,
                    'final_tokens': len(best_prompt_tokens),
                    'compression_ratio': (init_len - len(best_prompt_tokens)) / init_len * 100,
                    'final_likelihood': float(final_likelihood),
                    'avg_likelihood': float(avg_likelihood) if not (isinstance(avg_likelihood, float) and np.isnan(avg_likelihood)) else float('nan'),
                    'final_reward': float(final_reward),
                    'target_completion': target_completion_text,
                    'optimized_full_prompt': optimized_full_text,
                    'optimized_suffix': optimized_suffix_text,
                    'compressed_prompt': str(best_prompt)
                }
                results.append(result_row)

                print(f"Result: {init_len}→{result_row['final_tokens']} tokens ({result_row['compression_ratio']:.1f}% compression)")
                print(f"Likelihood: {final_likelihood:.3f}, Avg token likelihood: {result_row['avg_likelihood'] if not np.isnan(result_row['avg_likelihood']) else 'N/A'}, Reward: {final_reward:.3f}")

                # Log per-prompt metrics to wandb
                if wandb_run:
                    wandb.log({
                        'prompt_id': global_idx,
                        'final_tokens': result_row['final_tokens'],
                        'compression_ratio': result_row['compression_ratio'],
                        'final_likelihood': float(final_likelihood) if not np.isnan(final_likelihood) else None,
                        'avg_likelihood': float(result_row['avg_likelihood']) if not np.isnan(result_row['avg_likelihood']) else None,
                        'final_reward': float(final_reward) if not np.isnan(final_reward) else None,
                    })

                if save_plots and trace:
                    try:
                        plot_path = plot_eval_trace(
                            trace,
                            out_dir="results/traces",
                            prefix=f"{plots_prefix}_prompt_{global_idx:03d}",
                            alpha=alpha,
                            beta=beta
                        )
                        print(f"  Plot saved: {plot_path}")
                    except Exception as plot_err:
                        print(f"  Warning: Could not generate plot: {plot_err}")

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
    
    # Log summary metrics and results table to wandb
    if wandb_run:
        summary_metrics = {
            'total_prompts': len(results),
            'valid_prompts': len(valid_results),
            'failed_prompts': len(results) - len(valid_results),
        }
        if len(valid_results) > 0:
            summary_metrics.update({
                'mean_compression_ratio': valid_results['compression_ratio'].mean(),
                'std_compression_ratio': valid_results['compression_ratio'].std(),
                'mean_final_likelihood': valid_results['final_likelihood'].mean(),
                'std_final_likelihood': valid_results['final_likelihood'].std(),
                'mean_final_reward': valid_results['final_reward'].mean(),
                'std_final_reward': valid_results['final_reward'].std(),
                'no_compression_count': int((valid_results['compression_ratio'] == 0).sum()),
                'light_compression_count': int(((valid_results['compression_ratio'] > 0) & (valid_results['compression_ratio'] <= 25)).sum()),
                'medium_compression_count': int(((valid_results['compression_ratio'] > 25) & (valid_results['compression_ratio'] <= 50)).sum()),
                'heavy_compression_count': int((valid_results['compression_ratio'] > 50).sum()),
            })
        
        wandb.log(summary_metrics)
        
        # Log results as a table
        results_table = wandb.Table(dataframe=df)
        wandb.log({'eval_results': results_table})
        
        # Save results file as artifact
        artifact = wandb.Artifact('eval_results', type='results')
        artifact.add_file(results_file)
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print(f"wandb run finished")
    
    return results_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--model_path", type=str, help="Path to trained model (overrides config)")
    parser.add_argument("--max_prompts", type=int, help="Max test prompts (overrides config)")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Override max_test_prompts if provided
    if args.max_prompts:
        cfg['eval']['max_test_prompts'] = args.max_prompts
    
    # Use provided model path or get from config
    model_path = args.model_path or cfg['train'].get('save_path', 'models/trained_policy.pt')
    
    evaluate_on_dataset(cfg, model_path)
