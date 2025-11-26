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
import csv
from datetime import datetime
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager
import numpy as np

def train_on_dataset(cfg, fast_mode=False, dataset_name: str = "advbench"):
    """Train the prompt compression policy. Set fast_mode=True for a speed-focused run.

    dataset_name: 'advbench' or 'toxicchat' (default 'advbench')
    """
    model_name = cfg['model']
    train_cfg = cfg['train']

    base_episodes = train_cfg.get('episodes_per_prompt', 3)
    base_steps = train_cfg.get('steps_per_episode', 100)
    init_len = train_cfg.get('init_len', 32)
    base_lr_embeddings = train_cfg.get('lr_embeddings', 0.01)
    base_lr_policy = train_cfg.get('lr_policy', 3e-4)
    alpha = train_cfg.get('alpha', 1.0)
    beta = train_cfg.get('beta', 0.2)
    optimization_mode = train_cfg.get('optimization_mode', 'continuous')
    gcg_top_k = train_cfg.get('gcg_top_k', 16)
    gcg_batch_size = train_cfg.get('gcg_batch_size', 32)
    gcg_steps = train_cfg.get('gcg_steps', 5)
    save_path = train_cfg.get('save_path', 'models/trained_policy.pt')
    ppo_cfg = train_cfg.get('ppo', {})
    ppo_epochs = ppo_cfg.get('epochs', 4)
    ppo_clip = ppo_cfg.get('clip', 0.2)
    ppo_gamma = ppo_cfg.get('gamma', 0.99)
    ppo_lambda = ppo_cfg.get('gae_lambda', 0.95)
    ppo_value_coef = ppo_cfg.get('value_coef', 0.5)
    ppo_entropy_coef = ppo_cfg.get('entropy_coef', 0.01)
    rl_algo = str(train_cfg.get('rl_algo', 'ppo')).lower()
    use_ppo = rl_algo == 'ppo'
    one_batch = bool(train_cfg.get('one_batch', False))

    # Optional W&B logging
    wandb_cfg = cfg.get('wandb', train_cfg.get('wandb', {})) if isinstance(train_cfg, dict) else {}
    use_wandb = bool(wandb_cfg.get('enable', False))
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb.init(
                project=wandb_cfg.get('project', 'prompt-length-optimization'),
                name=wandb_cfg.get('run_name', f"{mode_name.lower()}_{dataset_name}_{rl_algo}"),
                config=cfg
            )
        except Exception as e:
            print(f"Warning: failed to initialize wandb ({e}), disabling wandb logging.")
            use_wandb = False
    global_reward_cfg = cfg.get('reward', {})
    reward_cfg = train_cfg.get('reward', global_reward_cfg)

    # Default to a larger prompt-batch to better utilize a single GPU
    batch_size = train_cfg.get('batch_size', 16)

    max_prompts = train_cfg.get('max_prompts', 50)
    min_prompt_length = train_cfg.get('min_prompt_length', 30)
    max_prompt_length = train_cfg.get('max_prompt_length', 150)

    episodes_per_prompt = base_episodes
    steps_per_episode = base_steps
    lr_embeddings = base_lr_embeddings
    lr_policy = base_lr_policy

    if fast_mode:
        episodes_per_prompt = max(1, base_episodes // 2)
        steps_per_episode = max(20, base_steps // 2)
        lr_embeddings = base_lr_embeddings * 2
        lr_policy = base_lr_policy * 2

    seed = cfg.get('seed', 2262)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Force single GPU (cuda:0) if available and enable cuDNN autotuner for throughput
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    mode_name = "Fast" if fast_mode else "Standard"
    print(f"{mode_name} training with {max_prompts} prompts")
    print(f"Model: {model_name}")
    print(f"Episodes per prompt: {episodes_per_prompt}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Learning rates: {lr_embeddings:.3f} / {lr_policy:.6f}")
    print(f"Batch size: {batch_size}")
    print(f"Alpha: {alpha}, Beta: {beta}")
    if fast_mode:
        print("Fast mode applies half episodes/steps and doubles learning rates relative to config values.")
    
    # Load dataset according to selected dataset_name
    prompts = []
    ds_cfg = cfg.get('dataset', {})
    if dataset_name.lower() == 'advbench':
        # Load AdvBench and extract (base prompt, target completion) pairs
        from datasets import load_dataset
        print("Loading AdvBench dataset...")
        raw = load_dataset("walledai/AdvBench", split='train')
        PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
        COMPLETION_KEYS = ["target", "completion", "output", "response", "answer"]
        for ex in raw:
            # find prompt-like field
            base = None
            for k in PROMPT_KEYS:
                if k in ex and ex[k]:
                    base = ex[k]
                    break
            if base is None:
                continue
            # find completion-like field
            target = None
            for k in COMPLETION_KEYS:
                if k in ex and ex[k] is not None:
                    target = ex[k]
                    break
            if target is None:
                target = ""
            prompts.append({'base': base.strip(), 'target': target.strip()})
        # deterministic sampling/shuffle
        import random as _rand
        _rand.seed(seed)
        _rand.shuffle(prompts)
        prompts = prompts[:max_prompts]
        print(f"Loaded {len(prompts)} AdvBench examples")
    else:
        # Fallback: toxic-chat using existing manager (only base prompt available)
        dataset_manager = ToxicChatDatasetManager(seed=seed)
        toxic_prompts = dataset_manager.load_train_set(
            min_length=min_prompt_length,
            max_length=max_prompt_length,
            max_samples=max_prompts,
            train_ratio=ds_cfg.get('train_ratio', 0.7),
            val_ratio=ds_cfg.get('val_ratio', 0.15),
            test_ratio=ds_cfg.get('test_ratio', 0.15),
            use_cache=True
        )
        prompts = [{'base': p, 'target': ''} for p in toxic_prompts]

    if not prompts:
        raise ValueError("No valid prompts found in dataset")
    
    # Initialize agent and optimizer
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent, reward_cfg=reward_cfg)
    # Prepare metrics output
    metrics_dir = "results"
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "training_metrics.csv")
    # write header if file doesn't exist
    if not os.path.exists(metrics_path):
        with open(metrics_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([
                'timestamp', 'batch_idx', 'global_prompt_idx', 'local_prompt_idx',
                'episode_count', 'final_likelihood', 'best_likelihood', 'best_episode', 'best_reward', 'base_text'
            ])

    # Shared helper to extract per-prompt final and best likelihood from batched traces
    def _extract_metrics_from_traces(traces_list, idx):
        final_ll = None
        best_ll = float('-inf')
        best_ep = None
        best_reward_val = None
        # iterate through trace entries in order
        for t in traces_list:
            ll_val = None
            if isinstance(t, dict):
                if 'likelihoods' in t and isinstance(t['likelihoods'], (list, tuple)):
                    try:
                        ll_val = float(t['likelihoods'][idx])
                    except Exception:
                        ll_val = None
                elif 'best_likelihoods' in t and isinstance(t['best_likelihoods'], (list, tuple)):
                    try:
                        ll_val = float(t['best_likelihoods'][idx])
                    except Exception:
                        ll_val = None
                elif 'likelihood' in t and (not isinstance(t['likelihood'], (list, tuple))):
                    try:
                        ll_val = float(t['likelihood'])
                    except Exception:
                        ll_val = None
            if ll_val is not None:
                # update final and best
                final_ll = ll_val
                if ll_val > best_ll:
                    best_ll = ll_val
                    best_ep = t.get('episode', t.get('step', None)) if isinstance(t, dict) else None
                    best_reward_val = None
                    if isinstance(t, dict) and 'best_rewards' in t and isinstance(t['best_rewards'], (list, tuple)):
                        try:
                            best_reward_val = float(t['best_rewards'][idx])
                        except Exception:
                            best_reward_val = None
        if final_ll is None:
            final_ll = 0.0
        if best_ll == float('-inf'):
            best_ll = final_ll
        return float(final_ll), float(best_ll), best_ep, best_reward_val
    
    # Track training progress across all prompts
    all_rewards = []
    best_overall_reward = float('-inf')
    best_prompt = None
    best_prompt_text = None
    
    start_time = time.time()
    
    optimization_mode_lower = optimization_mode.lower()

    # Process prompts in batches for better progress tracking
    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_summaries = []
        
        print(f"\n[Batch {batch_start//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}] Processing prompts {batch_start+1}-{batch_end}")
        
        batch_start_time = time.time()
        
        for prompt_idx, prompt_record in enumerate(batch_prompts):
            # We'll handle the entire batch at once if continuous mode is selected
            pass

        # Select optimization pipeline based on config
        opt_mode = optimization_mode_lower
        if 'ppo' in opt_mode:
            # PPO removed, use standard continuous mode
            targets = [p.get('target', '') for p in batch_prompts]
            best_results, best_rewards_batch, traces = optimizer.optimize_prompts_batch(
                target_completions=targets,
                episodes=episodes_per_prompt,
                steps_per_episode=steps_per_episode,
                initial_prompt_length=init_len,
                lr_embeddings=lr_embeddings,
                alpha=alpha,
                beta=beta,
                mode='continuous',
                use_ppo=use_ppo,
                ppo_epochs=ppo_epochs,
                ppo_clip=ppo_clip,
                gamma=ppo_gamma,
                gae_lambda=ppo_lambda,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef
            )

            for prompt_idx, (best_prompt_result, best_reward) in enumerate(zip(best_results, best_rewards_batch)):
                global_idx = batch_start + prompt_idx
                all_rewards.append(float(best_reward))
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)
                    best_prompt = best_prompt_result
                    best_prompt_text = batch_prompts[prompt_idx].get('base', '')

                try:
                    optimized_text = agent.tokenizer.decode(best_prompt_result, skip_special_tokens=True) if best_prompt_result else ''
                except Exception:
                    optimized_text = str(best_prompt_result)

                optimized_suffix_text = optimized_text
                try:
                    final_ll, best_ll, best_ep, best_reward_val = _extract_metrics_from_traces(traces, prompt_idx)
                except Exception:
                    final_ll, best_ll, best_ep, best_reward_val = 0.0, 0.0, None, None

                print(f"PPO metrics (prompt local idx={prompt_idx}, global idx={global_idx}): final_ll={final_ll:.3f}, best_ll={best_ll:.3f}, best_ep={best_ep}, best_reward={best_reward_val}")

                try:
                    with open(metrics_path, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow([
                            datetime.utcnow().isoformat(),
                            batch_start // batch_size,
                            global_idx,
                            prompt_idx,
                            len(traces),
                            final_ll,
                            best_ll,
                            best_ep,
                            best_reward_val if best_reward_val is not None else best_reward,
                            batch_prompts[prompt_idx].get('base', '')[:200]
                        ])
                except Exception as _:
                    print("Warning: failed to write training metrics to CSV")

                print(f"Input (base prompt): {batch_prompts[prompt_idx].get('base', '')[:200]}{'...' if len(batch_prompts[prompt_idx].get('base','')) > 200 else ''}")
                print(f"Optimized full prompt: {optimized_text[:300]}{'...' if len(optimized_text) > 300 else ''}")
                print(f"Optimized suffix: {optimized_suffix_text[:200]}{'...' if len(optimized_suffix_text) > 200 else ''}")
                print(f"Target completion: {batch_prompts[prompt_idx].get('target','')[:200]}{'...' if len(batch_prompts[prompt_idx].get('target','')) > 200 else ''}")
                batch_summaries.append({
                    'input': batch_prompts[prompt_idx].get('base', '')[:200],
                    'optimized_full': optimized_text,
                    'optimized_suffix': optimized_suffix_text,
                    'target': batch_prompts[prompt_idx].get('target', '')
                })
                batch_summaries.append({
                    'input': batch_prompts[prompt_idx].get('base', '')[:200],
                    'optimized_full': optimized_text,
                    'optimized_suffix': optimized_suffix_text,
                    'target': batch_prompts[prompt_idx].get('target', '')
                })
                batch_summaries.append({
                    'input': batch_prompts[prompt_idx].get('base', '')[:200],
                    'optimized_full': optimized_text,
                    'optimized_suffix': optimized_suffix_text,
                    'target': batch_prompts[prompt_idx].get('target', '')
                })

                if (prompt_idx + 1) % max(1, len(batch_prompts) // 4) == 0:
                    print(f"  Progress: {prompt_idx + 1}/{len(batch_prompts)}, Latest PPO reward: {best_reward:.3f}")

        elif opt_mode == 'continuous':
            targets = [p.get('target', '') for p in batch_prompts]
            best_results, best_rewards_batch, traces = optimizer.optimize_prompts_batch(
                target_completions=targets,
                episodes=episodes_per_prompt,
                steps_per_episode=steps_per_episode,
                initial_prompt_length=init_len,
                lr_embeddings=lr_embeddings,
                alpha=alpha,
                beta=beta,
                mode='continuous',
                use_ppo=use_ppo,
                ppo_epochs=ppo_epochs,
                ppo_clip=ppo_clip,
                gamma=ppo_gamma,
                gae_lambda=ppo_lambda,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef
            )

            # unpack and report per-prompt
            # --- batch-level trace logging ---
            try:
                if traces:
                    print(f"Batch traces (total entries={len(traces)}) - showing per-step likelihoods/rewards:")
                    # If many trace entries, show head/tail to avoid huge logs
                    show_all = len(traces) <= 50
                    entries_to_show = traces if show_all else (traces[:10] + traces[-10:])
                    for t in entries_to_show:
                        if 'likelihoods' in t:
                            ll = t['likelihoods']
                            print(f"  Ep {t.get('episode', '?')} likelihoods: {[f'{v:.3f}' for v in ll]}")
                        elif 'best_likelihoods' in t:
                            bl = t['best_likelihoods']
                            print(f"  Ep {t.get('episode', '?')} step {t.get('step', '?')} best_likelihoods: {[f'{v:.3f}' for v in bl]}")
                        else:
                            # generic trace dump
                            print(f"  trace entry: {t}")
                    if not show_all:
                        print(f"  ... omitted {len(traces)-20} intermediate trace entries ...")
            except Exception as _:
                print("  (could not pretty-print traces)")
            

            for prompt_idx, (best_prompt_result, best_reward) in enumerate(zip(best_results, best_rewards_batch)):
                global_idx = batch_start + prompt_idx
                all_rewards.append(float(best_reward))
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)
                    best_prompt = best_prompt_result
                    best_prompt_text = batch_prompts[prompt_idx].get('base', '')

                try:
                    optimized_text = agent.tokenizer.decode(best_prompt_result, skip_special_tokens=True) if best_prompt_result else ''
                except Exception:
                    optimized_text = str(best_prompt_result)

                optimized_suffix_text = optimized_text

                # Extract final and best likelihoods from traces for this prompt
                try:
                    final_ll, best_ll, best_ep, best_reward_val = _extract_metrics_from_traces(traces, prompt_idx)
                except Exception:
                    final_ll, best_ll, best_ep, best_reward_val = 0.0, 0.0, None, None

                # Print per-prompt episode metrics
                print(f"Episode metrics (prompt local idx={prompt_idx}, global idx={global_idx}): final_ll={final_ll:.3f}, best_ll={best_ll:.3f}, best_ep={best_ep}, best_reward={best_reward_val}")

                # Append to CSV for later reference
                try:
                    with open(metrics_path, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow([
                            datetime.utcnow().isoformat(),
                            batch_start // batch_size,
                            global_idx,
                            prompt_idx,
                            len(traces),
                            final_ll,
                            best_ll,
                            best_ep,
                            best_reward_val if best_reward_val is not None else best_reward,
                            batch_prompts[prompt_idx].get('base', '')[:200]
                        ])
                except Exception as _:
                    print("Warning: failed to write training metrics to CSV")
                print(f"Input (base prompt): {batch_prompts[prompt_idx].get('base', '')[:200]}{'...' if len(batch_prompts[prompt_idx].get('base','')) > 200 else ''}")
                print(f"Optimized full prompt: {optimized_text[:300]}{'...' if len(optimized_text) > 300 else ''}")
                print(f"Optimized suffix: {optimized_suffix_text[:200]}{'...' if len(optimized_suffix_text) > 200 else ''}")
                print(f"Target completion: {batch_prompts[prompt_idx].get('target','')[:200]}{'...' if len(batch_prompts[prompt_idx].get('target','')) > 200 else ''}")

                if (prompt_idx + 1) % max(1, len(batch_prompts) // 4) == 0:
                    print(f"  Progress: {prompt_idx + 1}/{len(batch_prompts)}, Latest reward: {best_reward:.3f}")
        elif optimization_mode.lower() == 'discrete':
            # Use the batched discrete (GCG) optimizer for this whole batch
            print(f"Running batched discrete optimizer on batch size={len(batch_prompts)}")
            targets = [p.get('target', '') for p in batch_prompts]
            best_results, best_rewards_batch, traces = optimizer.optimize_prompts_batch(
                target_completions=targets,
                episodes=episodes_per_prompt,
                steps_per_episode=steps_per_episode,
                initial_prompt_length=init_len,
                lr_embeddings=lr_embeddings,
                alpha=alpha,
                beta=beta,
                mode='discrete',
                use_ppo=use_ppo,
                ppo_epochs=ppo_epochs,
                ppo_clip=ppo_clip,
                gamma=ppo_gamma,
                gae_lambda=ppo_lambda,
                value_coef=ppo_value_coef,
                entropy_coef=ppo_entropy_coef
            )

            for prompt_idx, (best_prompt_result, best_reward) in enumerate(zip(best_results, best_rewards_batch)):
                global_idx = batch_start + prompt_idx
                all_rewards.append(float(best_reward))
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)
                    best_prompt = best_prompt_result
                    best_prompt_text = batch_prompts[prompt_idx].get('base', '')

                try:
                    optimized_text = agent.tokenizer.decode(best_prompt_result, skip_special_tokens=True) if best_prompt_result else ''
                except Exception:
                    optimized_text = str(best_prompt_result)

                optimized_suffix_text = optimized_text
                # Extract final and best likelihoods from traces for this prompt (discrete)
                try:
                    final_ll, best_ll, best_ep, best_reward_val = _extract_metrics_from_traces(traces, prompt_idx)
                except Exception:
                    final_ll, best_ll, best_ep, best_reward_val = 0.0, 0.0, None, None

                print(f"Episode metrics (prompt local idx={prompt_idx}, global idx={global_idx}): final_ll={final_ll:.3f}, best_ll={best_ll:.3f}, best_ep={best_ep}, best_reward={best_reward_val}")

                # Append to CSV for later reference
                try:
                    with open(metrics_path, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow([
                            datetime.utcnow().isoformat(),
                            batch_start // batch_size,
                            global_idx,
                            prompt_idx,
                            len(traces),
                            final_ll,
                            best_ll,
                            best_ep,
                            best_reward_val if best_reward_val is not None else best_reward,
                            batch_prompts[prompt_idx].get('base', '')[:200]
                        ])
                except Exception:
                    print("Warning: failed to write training metrics to CSV")

                print(f"Input (base prompt): {batch_prompts[prompt_idx].get('base', '')[:200]}{'...' if len(batch_prompts[prompt_idx].get('base','')) > 200 else ''}")
                print(f"Optimized full prompt: {optimized_text[:300]}{'...' if len(optimized_text) > 300 else ''}")
                print(f"Optimized suffix: {optimized_suffix_text[:200]}{'...' if len(optimized_suffix_text) > 200 else ''}")
                print(f"Target completion: {batch_prompts[prompt_idx].get('target','')[:200]}{'...' if len(batch_prompts[prompt_idx].get('target','')) > 200 else ''}")

                if (prompt_idx + 1) % max(1, len(batch_prompts) // 4) == 0:
                    print(f"  Progress: {prompt_idx + 1}/{len(batch_prompts)}, Latest reward: {best_reward:.3f}")
            # --- batch-level trace logging for discrete optimizer ---
            try:
                if traces:
                    print(f"Batch traces (total entries={len(traces)}) - showing per-step likelihoods/rewards:")
                    show_all = len(traces) <= 50
                    entries_to_show = traces if show_all else (traces[:10] + traces[-10:])
                    for t in entries_to_show:
                        if 'best_likelihoods' in t:
                            bl = t['best_likelihoods']
                            print(f"  Ep {t.get('episode','?')} step {t.get('step','?')} best_likelihoods: {[f'{v:.3f}' for v in bl]}")
                        elif 'likelihoods' in t:
                            ll = t['likelihoods']
                            print(f"  Ep {t.get('episode','?')} likelihoods: {[f'{v:.3f}' for v in ll]}")
                        else:
                            print(f"  trace entry: {t}")
                    if not show_all:
                        print(f"  ... omitted {len(traces)-20} intermediate trace entries ...")
            except Exception:
                print("  (could not pretty-print discrete traces)")

        else:
            # Fallback to per-prompt sequential processing for other/unknown modes
            for prompt_idx, prompt_record in enumerate(batch_prompts):
                global_idx = batch_start + prompt_idx

                try:
                    base_text = prompt_record.get('base', '')
                    target_text = prompt_record.get('target', '')

                    # Train on this specific prompt with reduced parameters; pass target and base explicitly
                    best_prompt_result, best_reward, history = optimizer.optimize_prompt(
                        target_completion=target_text,
                        episodes=episodes_per_prompt,
                        steps_per_episode=steps_per_episode,
                        initial_prompt_length=init_len,
                        lr_embeddings=lr_embeddings,
                        lr_policy=lr_policy,
                        alpha=alpha,
                        beta=beta,
                        log_every=0,  # Disable detailed logging for speed
                        optimization_mode=optimization_mode,
                        gcg_top_k=gcg_top_k,
                        gcg_batch_size=gcg_batch_size,
                        gcg_steps=gcg_steps,
                        base_prompt=base_text
                    )

                    all_rewards.append(float(best_reward))

                    if best_reward > best_overall_reward:
                        best_overall_reward = float(best_reward)  # Ensure it's a Python float
                        best_prompt = best_prompt_result
                        best_prompt_text = base_text

                    # Print input, optimized suffix/full prompt, and target completion for transparency
                    try:
                        optimized_text = agent.tokenizer.decode(best_prompt_result, skip_special_tokens=True) if best_prompt_result else ''
                    except Exception:
                        optimized_text = str(best_prompt_result)

                    # Determine optimized suffix by removing base token ids if possible
                    optimized_suffix_text = ''
                    try:
                        if isinstance(best_prompt_result, list) and base_text:
                            base_ids = agent.tokenizer.encode(base_text, add_special_tokens=False)
                            if len(best_prompt_result) >= len(base_ids) and best_prompt_result[:len(base_ids)] == base_ids:
                                suffix_ids = best_prompt_result[len(base_ids):]
                                optimized_suffix_text = agent.tokenizer.decode(suffix_ids, skip_special_tokens=True)
                            else:
                                optimized_suffix_text = optimized_text.replace(base_text, '', 1).strip()
                        else:
                            optimized_suffix_text = optimized_text
                    except Exception:
                        optimized_suffix_text = optimized_text

                    print(f"Input (base prompt): {base_text[:200]}{'...' if len(base_text) > 200 else ''}")
                    print(f"Optimized full prompt: {optimized_text[:300]}{'...' if len(optimized_text) > 300 else ''}")
                    print(f"Optimized suffix: {optimized_suffix_text[:200]}{'...' if len(optimized_suffix_text) > 200 else ''}")
                    print(f"Target completion: {target_text[:200]}{'...' if len(target_text) > 200 else ''}")

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
        # Print concise per-prompt summary at end of batch
        for idx, summary in enumerate(batch_summaries):
            print(f"[Batch summary] Prompt {batch_start + idx + 1}:")
            print(f"  Input: {summary['input'][:180]}{'...' if len(summary['input']) > 180 else ''}")
            print(f"  Optimized suffix: {summary['optimized_suffix'][:180]}{'...' if len(summary['optimized_suffix']) > 180 else ''}")
            print(f"  Target completion: {summary['target'][:180]}{'...' if len(summary['target']) > 180 else ''}")
        if use_wandb:
            try:
                import wandb  # type: ignore
                wandb.log({
                    "batch/index": batch_start // batch_size,
                    "batch/mean_best_reward": float(np.mean(all_rewards[-len(batch_prompts):])) if batch_prompts else 0.0,
                    "batch/best_reward": float(max(all_rewards[-len(batch_prompts):])) if batch_prompts else 0.0,
                    "batch/time_sec": batch_time
                })
            except Exception as _:
                print("Warning: failed to log to wandb for this batch.")
        
        # Progress estimate
        completed = len(all_rewards)
        remaining = len(prompts) - completed
        estimated_time_left = remaining * avg_time_per_prompt
        print(f"  ETA: {estimated_time_left/60:.1f} minutes ({completed}/{len(prompts)} prompts complete)")

        if one_batch:
            print("One-batch mode enabled; stopping after first batch.")
            break
    
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
        'fast_mode': fast_mode,
        'episodes_per_prompt': episodes_per_prompt,
        'steps_per_episode': steps_per_episode,
        'lr_embeddings': lr_embeddings,
        'lr_policy': lr_policy
    }
    if fast_mode:
        checkpoint['fast_mode_settings'] = {
            'base_episodes_per_prompt': base_episodes,
            'base_steps_per_episode': base_steps,
            'base_lr_embeddings': base_lr_embeddings,
            'base_lr_policy': base_lr_policy
        }
    
    torch.save(checkpoint, save_path)
    
    # Final statistics
    if all_rewards:
        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        
        print(f"\n{'='*60}")
        print(f"{mode_name.upper()} TRAINING COMPLETE!")
        print(f"Trained on {len(all_rewards)} prompts in {training_time:.1f}s")
        print(f"Speed: {len(all_rewards)/training_time:.2f} prompts/second")
        print(f"Average time per prompt: {training_time/len(all_rewards):.2f}s")
        print(f"Best reward: {best_overall_reward:.3f}")
        print(f"Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
        print(f"Best prompt text: '{best_prompt_text[:80]}{'...' if len(best_prompt_text) > 80 else ''}'")
        print(f"Model saved to: {save_path}")
        print(f"Training hyperparameters used:")
        print(f"  Episodes per prompt: {episodes_per_prompt}")
        print(f"  Steps per episode : {steps_per_episode}")
        print(f"  LR (embeddings)   : {lr_embeddings:.4f}")
        print(f"  LR (policy)       : {lr_policy:.6f}")
        if fast_mode:
            print(f"  Base config episodes: {base_episodes}")
            print(f"  Base config steps   : {base_steps}")
            print(f"  Base LR embeddings  : {base_lr_embeddings:.4f}")
            print(f"  Base LR policy      : {base_lr_policy:.6f}")
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
                plt.title(f"{mode_name} Training Progress ({len(plot_rewards)} prompts)")
                plt.xlabel("Prompt Number")
                plt.ylabel("Reward")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                fmt = train_cfg.get('plots_format', 'png')
                default_prefix = 'fast_training' if fast_mode else 'training'
                plot_prefix = train_cfg.get('plots_prefix', default_prefix)
                plot_path = f"results/{plot_prefix}_rewards.{fmt}"
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
    parser.add_argument("--fast", action="store_true", help="Use speed-optimized hyperparameters")
    parser.add_argument("--dataset", type=str, default="advbench", choices=["advbench", "toxicchat"], help="Dataset to use (default: advbench)")
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
    
    train_on_dataset(cfg, fast_mode=args.fast, dataset_name=args.dataset)
