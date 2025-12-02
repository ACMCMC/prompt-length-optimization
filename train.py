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
import logging
from datetime import datetime
from tqdm import tqdm
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_on_dataset(cfg, fast_mode=False, dataset_name: str = "advbench", use_wandb: bool = None, wandb_project: str = None):
    """Train the prompt compression policy. Set fast_mode=True for a speed-focused run.

    dataset_name: 'advbench' or 'toxicchat' (default 'advbench')
    use_wandb: Whether to log to wandb (None = use config, True/False = override)
    wandb_project: Wandb project name (None = use config, str = override)
    """
    model_name = cfg['model']
    train_cfg = cfg['train']

    # Get wandb settings from config
    config_use_wandb = train_cfg['use_wandb']
    config_wandb_project = train_cfg['wandb_project']
    
    # Override with function args if provided
    if use_wandb is None:
        use_wandb = config_use_wandb
    if wandb_project is None:
        wandb_project = config_wandb_project

    base_episodes = train_cfg['episodes_per_prompt']
    base_steps = train_cfg['steps_per_episode']
    init_len = train_cfg['init_len']
    max_suffix_len = train_cfg['max_suffix_len']
    base_lr_embeddings = train_cfg['lr_embeddings']
    base_lr_policy = train_cfg['lr_policy']
    alpha = train_cfg['alpha']
    beta = train_cfg['beta']
    optimization_mode = train_cfg['optimization_mode']
    gcg_cfg = train_cfg['gcg']
    gcg_top_k = gcg_cfg['top_k']
    gcg_batch_size = gcg_cfg['batch_size']
    gcg_max_batch_size = gcg_cfg['max_batch_size']
    gcg_steps = gcg_cfg['steps']
    save_path = train_cfg['save_path']
    grpo_cfg = train_cfg.get('grpo') or train_cfg.get('ppo')  # Support both 'grpo' and legacy 'ppo' keys
    if grpo_cfg is None:
        raise ValueError("Either 'grpo' or 'ppo' config must be provided in YAML")
    grpo_epochs = grpo_cfg['epochs']
    grpo_clip = grpo_cfg['clip']
    grpo_gamma = grpo_cfg['gamma']
    grpo_lambda = grpo_cfg['gae_lambda']
    grpo_value_coef = grpo_cfg['value_coef']
    grpo_entropy_coef = grpo_cfg.get('entropy_coef')  # Optional, falls back to train.entropy_coef

    # Default to a larger prompt-batch to better utilize a single GPU
    batch_size = train_cfg['batch_size']

    max_prompts = train_cfg['max_prompts']
    min_prompt_length = train_cfg['min_prompt_length']
    max_prompt_length = train_cfg['max_prompt_length']

    episodes_per_prompt = base_episodes
    steps_per_episode = base_steps
    lr_embeddings = base_lr_embeddings
    lr_policy = base_lr_policy

    if fast_mode:
        episodes_per_prompt = max(1, base_episodes // 2)
        steps_per_episode = max(20, base_steps // 2)
        lr_embeddings = base_lr_embeddings * 2
        lr_policy = base_lr_policy * 2

    # Get RL/GRPO parameters (needed for wandb config)
    epsilon = train_cfg['epsilon']
    epsilon_decay = train_cfg['epsilon_decay']
    epsilon_min = train_cfg['epsilon_min']
    entropy_coef = train_cfg.get('entropy_coef')  # Optional, can be overridden by grpo.entropy_coef
    if entropy_coef is None:
        entropy_coef = grpo_entropy_coef  # Fall back to GRPO entropy if not set
    temperature = train_cfg['temperature']
    
    # Initialize wandb if available and requested (after variables are set)
    wandb_initialized = False
    if use_wandb and WANDB_AVAILABLE:
        try:
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    name=f"policy_training_{optimization_mode}",
                    config={
                        'model': model_name,
                        'optimization_mode': optimization_mode,
                        'episodes_per_prompt': episodes_per_prompt,
                        'steps_per_episode': steps_per_episode,
                        'init_len': init_len,
                        'max_prompts': max_prompts,
                        'batch_size': batch_size,
                        'lr_embeddings': lr_embeddings,
                        'lr_policy': lr_policy,
                        'alpha': alpha,
                        'beta': beta,
                        'epsilon': epsilon,
                        'epsilon_decay': epsilon_decay,
                        'epsilon_min': epsilon_min,
                        'entropy_coef': entropy_coef,
                        'temperature': temperature,
                        'grpo_epochs': grpo_epochs,
                        'grpo_clip': grpo_clip,
                        'grpo_gamma': grpo_gamma,
                        'grpo_gae_lambda': grpo_lambda,
                        'grpo_value_coef': grpo_value_coef,
                        'grpo_entropy_coef': grpo_entropy_coef,
                        'dataset': dataset_name,
                        'seed': cfg['seed']
                    }
                )
                wandb_initialized = True
                logger.info(f"Wandb initialized: project={wandb_project}, run={wandb.run.name}")
                # Log initial metrics to verify wandb is working (use step=0, will be first step)
                try:
                    wandb.log({'train/started': 1, 'train/max_prompts': max_prompts, 'train/batch_size': batch_size}, step=0)
                    logger.info(f"Wandb test log sent. View at: {wandb.run.url}")
                except Exception as e:
                    logger.warning(f"Failed to send test log to wandb: {e}")
            else:
                wandb_initialized = True  # Already initialized (e.g., by sweep)
                logger.info("Wandb already initialized (likely by sweep)")
                # Log initial metrics to verify wandb is working (use step=0, will be first step)
                try:
                    wandb.log({'train/started': 1, 'train/max_prompts': max_prompts, 'train/batch_size': batch_size}, step=0)
                    logger.info(f"Wandb test log sent. View at: {wandb.run.url}")
                except Exception as e:
                    logger.warning(f"Failed to send test log to wandb: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            wandb_initialized = False
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("Wandb requested but not available (wandb not installed)")
        wandb_initialized = False
    else:
        wandb_initialized = False

    seed = cfg['seed']
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
            train_ratio=ds_cfg['train_ratio'],
            val_ratio=ds_cfg['val_ratio'],
            test_ratio=ds_cfg['test_ratio'],
            use_cache=True
        )
        prompts = [{'base': p, 'target': ''} for p in toxic_prompts]

    if not prompts:
        raise ValueError("No valid prompts found in dataset")
    
    # Initialize agent and optimizer
    agent = PromptRLAgent(model_name=model_name)
    # Note: epsilon, epsilon_decay, epsilon_min, entropy_coef, temperature
    # are already defined above (before wandb init) for wandb config
    grpo_cfg = train_cfg.get('grpo') or train_cfg.get('ppo')  # Support both 'grpo' and legacy 'ppo' keys
    if grpo_cfg is None:
        raise ValueError("Either 'grpo' or 'ppo' config must be provided in YAML")
    grpo_clip = grpo_cfg['clip']
    grpo_epochs = grpo_cfg['epochs']
    grpo_gamma = grpo_cfg['gamma']
    grpo_gae_lambda = grpo_cfg['gae_lambda']
    grpo_value_coef = grpo_cfg['value_coef']
    grpo_entropy_coef = grpo_cfg.get('entropy_coef')  # Optional, falls back to train.entropy_coef
    if grpo_entropy_coef is None:
        grpo_entropy_coef = entropy_coef
    
    # Policy network architecture parameters
    policy_cfg = train_cfg['policy']
    policy_hidden_size = policy_cfg['hidden_size']
    max_grad_norm = grpo_cfg['max_grad_norm']
    
    optimizer = LengthPolicyOptimizer(
        agent, 
        epsilon=epsilon, 
        epsilon_decay=epsilon_decay, 
        epsilon_min=epsilon_min,
        entropy_coef=grpo_entropy_coef,
        temperature=temperature,
        grpo_clip=grpo_clip,
        grpo_epochs=grpo_epochs,
        grpo_gamma=grpo_gamma,
        grpo_gae_lambda=grpo_gae_lambda,
        grpo_value_coef=grpo_value_coef,
        policy_hidden_size=policy_hidden_size,
        max_grad_norm=max_grad_norm
    )
    
    # Set projection parameters for continuous_proj mode
    projection_weight = train_cfg['projection_weight']
    distance_metric = train_cfg['distance_metric']
    optimizer.projection_weight = projection_weight
    optimizer.distance_metric = distance_metric
    
    # Set GCG parameters for discrete mode
    optimizer.gcg_steps = gcg_steps
    optimizer.gcg_top_k = gcg_top_k
    optimizer.gcg_batch_size = gcg_batch_size
    optimizer.gcg_max_batch_size = gcg_max_batch_size
    
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
    
    # Prepare policy metrics output
    policy_metrics_path = os.path.join(metrics_dir, "policy_training_metrics.csv")
    if not os.path.exists(policy_metrics_path):
        with open(policy_metrics_path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([
                'timestamp', 'batch_idx', 'episode', 'avg_reward', 'avg_return', 
                'avg_advantage', 'std_advantage', 'policy_loss', 'value_loss', 'entropy', 'epsilon',
                'avg_value_pred', 'value_pred_error'
            ])

    # Shared helper to extract per-prompt final and best likelihood from batched traces
    def _extract_metrics_from_traces(traces_list, idx):
        """Extract metrics from traces for a specific prompt index."""
        final_ll = None
        best_ll = float('-inf')
        best_ep = None
        best_reward_val = None
        
        if not traces_list:
            return 0.0, 0.0, None, None
        
        # iterate through trace entries in order
        for t in traces_list:
            if not isinstance(t, dict):
                continue
                
            # Prefer best_likelihoods over likelihoods (best_likelihoods tracks the best so far)
            ll_val = None
            
            # Try best_likelihoods first (this is the best so far)
            if 'best_likelihoods' in t:
                best_ll_list = t['best_likelihoods']
                if isinstance(best_ll_list, (list, tuple)) and len(best_ll_list) > idx:
                    try:
                        ll_val = float(best_ll_list[idx])
                        if not (isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf'))):
                            # Valid value
                            pass
                        else:
                            ll_val = None
                    except (IndexError, ValueError, TypeError):
                        ll_val = None
            
            # Fallback to current likelihoods
            if ll_val is None and 'likelihoods' in t:
                ll_list = t['likelihoods']
                if isinstance(ll_list, (list, tuple)) and len(ll_list) > idx:
                    try:
                        ll_val = float(ll_list[idx])
                        if not (isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf'))):
                            # Valid value
                            pass
                        else:
                            ll_val = None
                    except (IndexError, ValueError, TypeError):
                        ll_val = None
            
            # Last fallback: single likelihood value
            if ll_val is None and 'likelihood' in t:
                try:
                    ll_val = float(t['likelihood'])
                    if not (isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf'))):
                        # Valid value
                        pass
                    else:
                        ll_val = None
                except (ValueError, TypeError):
                    ll_val = None
            
            if ll_val is not None:
                # update final (last non-None value)
                final_ll = ll_val
                # update best if this is better (likelihoods are log probs, so higher is better)
                if ll_val > best_ll:
                    best_ll = ll_val
                    best_ep = t.get('episode', t.get('step', None))
                    # Try to get best_reward from this trace entry
                    if 'best_rewards' in t and isinstance(t['best_rewards'], (list, tuple)) and len(t['best_rewards']) > idx:
                        try:
                            best_reward_val = float(t['best_rewards'][idx])
                        except (IndexError, ValueError, TypeError):
                            pass
                    elif 'rewards' in t and isinstance(t['rewards'], (list, tuple)) and len(t['rewards']) > idx:
                        try:
                            best_reward_val = float(t['rewards'][idx])
                        except (IndexError, ValueError, TypeError):
                            pass
        
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

    def process_batch_results(best_results, best_rewards_batch, traces, batch_prompts, 
                              batch_start, batch_size, metrics_path, episodes_per_prompt):
        """Process results from a batch optimization run. Returns updated best tracking."""
        nonlocal all_rewards, best_overall_reward, best_prompt, best_prompt_text
        
        # Debug: check lengths match
        if len(best_results) != len(batch_prompts):
            logger.warning(f"Length mismatch: best_results has {len(best_results)} entries, batch_prompts has {len(batch_prompts)} entries")
            # Truncate or pad to match
            min_len = min(len(best_results), len(batch_prompts))
            best_results = best_results[:min_len]
            best_rewards_batch = best_rewards_batch[:min_len]

            for prompt_idx, (best_prompt_result, best_reward) in enumerate(zip(best_results, best_rewards_batch)):
                global_idx = batch_start + prompt_idx
                all_rewards.append(float(best_reward))
            
                if best_reward > best_overall_reward:
                    best_overall_reward = float(best_reward)
                    best_prompt = best_prompt_result
                    best_prompt_text = batch_prompts[prompt_idx].get('base', '')

            # Decode optimized prompt
                try:
                    optimized_text = agent.tokenizer.decode(best_prompt_result, skip_special_tokens=True) if best_prompt_result else ''
                except Exception:
                    optimized_text = str(best_prompt_result)

                optimized_suffix_text = optimized_text

            # Extract metrics from traces
            # traces is a list of step-level trace dicts (one per step), each containing batch-level data
            # We extract data for this specific prompt from all step traces using prompt_idx
            prompt_traces = traces[prompt_idx]  # Get the list of step-level traces for this prompt
            final_ll, best_ll, best_ep, best_reward_val = _extract_metrics_from_traces(
                prompt_traces, prompt_idx
            )

            logger.debug(
                f"Metrics (prompt local idx={prompt_idx}, global idx={global_idx}): "
                f"final_ll={final_ll:.3f}, best_ll={best_ll:.3f}, best_ep={best_ep}, "
                f"best_reward={best_reward_val if best_reward_val is not None else best_reward}"
            )
            
            # Write to CSV
            try:
                with open(metrics_path, 'a', newline='') as fh:
                    writer = csv.writer(fh)
                    writer.writerow([
                        datetime.utcnow().isoformat(),
                        batch_start // batch_size,
                        global_idx,
                        prompt_idx,
                        episodes_per_prompt,
                        final_ll,
                        best_ll,
                        best_ep,
                        best_reward_val if best_reward_val is not None else best_reward,
                        batch_prompts[prompt_idx].get('base', '')[:200]
                    ])
            except Exception:
                logger.warning("Failed to write training metrics to CSV")

            # Log details
            logger.debug(
                f"Input (base prompt): {batch_prompts[prompt_idx].get('base', '')[:200]}"
                f"{'...' if len(batch_prompts[prompt_idx].get('base','')) > 200 else ''}"
            )
            logger.debug(
                f"Optimized full prompt: {optimized_text[:300]}"
                f"{'...' if len(optimized_text) > 300 else ''}"
            )
            logger.debug(
                f"Optimized suffix: {optimized_suffix_text[:200]}"
                f"{'...' if len(optimized_suffix_text) > 200 else ''}"
            )
            logger.debug(
                f"Target completion: {batch_prompts[prompt_idx].get('target','')[:200]}"
                f"{'...' if len(batch_prompts[prompt_idx].get('target','')) > 200 else ''}"
            )

            if (prompt_idx + 1) % max(1, len(batch_prompts) // 4) == 0:
                logger.debug(
                    f"  Progress: {prompt_idx + 1}/{len(batch_prompts)}, "
                    f"Latest reward: {best_reward:.3f}"
                )
    
    def log_traces(traces, mode_name):
        """Log batch-level trace information."""
        try:
            if traces:
                logger.debug(
                    f"Batch traces ({mode_name}, total entries={len(traces)}) - "
                    f"showing per-step likelihoods/rewards:"
                )
                show_all = len(traces) <= 50
                entries_to_show = traces if show_all else (traces[:10] + traces[-10:])
                for t in entries_to_show:
                    if 'best_likelihoods' in t:
                        bl = t['best_likelihoods']
                        logger.debug(
                            f"  Ep {t.get('episode','?')} step {t.get('step','?')} "
                            f"best_likelihoods: {[f'{v:.3f}' for v in bl]}"
                        )
                    elif 'likelihoods' in t:
                        ll = t['likelihoods']
                        logger.debug(
                            f"  Ep {t.get('episode', '?')} likelihoods: "
                            f"{[f'{v:.3f}' for v in ll]}"
                        )
                    else:
                        logger.debug(f"  trace entry: {t}")
                if not show_all:
                    logger.debug(f"  ... omitted {len(traces)-20} intermediate trace entries ...")
        except Exception:
            logger.debug(f"  (could not pretty-print {mode_name} traces)")
    
    def run_batch_optimization(batch_prompts, mode, batch_idx, episode_idx, num_batches):
        """Run batch optimization for a given mode and episode. Returns (results, rewards, traces, policy_metrics)."""
        prefixes = [p.get('base', '') for p in batch_prompts]
        targets = [p.get('target', '') for p in batch_prompts]
        logger.debug(f"Running batched {mode} optimizer on batch size={len(batch_prompts)}, episode {episode_idx+1}")
        
        # Create wandb logging function if wandb is initialized
        wandb_log_fn = None
        if wandb_initialized:
            def log_to_wandb(log_dict, step=None):
                try:
                    wandb.log(log_dict, step=step, commit=True)  # commit=True to flush immediately
                except Exception as e:
                    logger.warning(f"Failed to log to wandb: {e}")
            wandb_log_fn = log_to_wandb
            logger.info(f"Wandb logging enabled for batch optimization")

        # Compute global step offset for this batch and episode to ensure monotonic step numbers
        # Structure: Episode 0 (all batches), then Episode 1 (all batches), etc.
        # Formula: episode * num_batches * steps_per_episode + batch_idx * steps_per_episode
        global_step_offset = episode_idx * num_batches * steps_per_episode + batch_idx * steps_per_episode
        
        # Get rollouts_per_prompt from config
        rollouts_per_prompt = train_cfg['rollouts_per_prompt']
        
        # Run only ONE episode for this batch (we cycle through episodes in the outer loop)
        best_results, best_rewards_batch, traces, policy_metrics = optimizer.optimize_prompts_batch(
            prefixes=prefixes,
            target_completions=targets,
            episodes=1,  # Only one episode per call
            steps_per_episode=steps_per_episode,
            initial_prompt_length=init_len,
            lr_embeddings=lr_embeddings,
            alpha=alpha,
            beta=beta,
            mode=mode,
            batch_size=batch_size,
            max_suffix_len=max_suffix_len,
            init_len=init_len,
            wandb_log_fn=wandb_log_fn,
            global_step_offset=global_step_offset,
            rollouts_per_prompt=rollouts_per_prompt
        )
        
        return best_results, best_rewards_batch, traces, policy_metrics
    
    # Map optimization mode to the mode string for optimize_prompts_batch
    mode_map = {
        'continuous': 'continuous',
        'continuous_proj': 'continuous_proj',
        'discrete': 'discrete',
    }
    
    # Handle legacy 'ppo' mode (map to continuous)
    if 'ppo' in optimization_mode_lower:
        mode_map['ppo'] = 'continuous'
    
    # Calculate number of batches
    num_batches = (len(prompts) - 1) // batch_size + 1
    
    # NEW STRUCTURE: Cycle through episodes, processing all batches per episode
    # This provides more diverse experience per policy update
    # Episode 0: Batch 0, Batch 1, Batch 2, ... (update policy after each batch)
    # Episode 1: Batch 0, Batch 1, Batch 2, ... (update policy after each batch)
    # etc.
    for episode_idx in tqdm(range(episodes_per_prompt), desc="Episodes"):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_idx + 1}/{episodes_per_prompt}")
        print(f"{'='*60}")
        
        # Process all batches for this episode
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"Batches (ep {episode_idx+1})", leave=False):
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_idx = batch_start // batch_size
            
            print(f"\n[Episode {episode_idx+1}/{episodes_per_prompt}, Batch {batch_idx+1}/{num_batches}] Processing prompts {batch_start+1}-{batch_end}")
            
            batch_start_time = time.time()

            # Determine which mode to use
            opt_mode = optimization_mode_lower
            if opt_mode in mode_map:
                # Run batch optimization with the mapped mode (only 1 episode)
                mode = mode_map[opt_mode]
                best_results, best_rewards_batch, traces, policy_metrics = run_batch_optimization(
                    batch_prompts, mode, batch_idx, episode_idx, num_batches
                )
                
                # Calculate the last step of this episode for logging policy/batch metrics
                # This ensures we log at the same step as the last step-level metric
                # We add a small offset to ensure it's logged after all step-level metrics
                last_episode_step = episode_idx * num_batches * steps_per_episode + batch_idx * steps_per_episode + steps_per_episode - 1
                # Use the next step to ensure it's after all step-level metrics for this episode
                policy_batch_step = last_episode_step + 1
                
                # Log traces
                log_traces(traces, mode)
                
                # Save policy training metrics
                try:
                    with open(policy_metrics_path, 'a', newline='') as fh:
                        writer = csv.writer(fh)
                        for pm in policy_metrics:
                            writer.writerow([
                                datetime.utcnow().isoformat(),
                                pm.get('batch_idx', batch_idx),
                                pm.get('episode', episode_idx),
                                pm.get('avg_reward', 0.0),
                                pm.get('avg_return', 0.0),
                                pm.get('avg_advantage', 0.0),
                                pm.get('std_advantage', 1.0),
                                pm.get('policy_loss', 0.0),
                                pm.get('value_loss', 0.0),
                                pm.get('entropy', 0.0),
                                pm.get('epsilon', 0.0),
                                pm.get('avg_value_pred', 0.0),
                                pm.get('value_pred_error', 0.0)
                            ])
                    
                    # Log policy metrics to wandb at the last step of the episode
                    # This should be done immediately after the episode completes, before the next episode starts
                    if wandb_initialized and policy_metrics:
                        for pm in policy_metrics:
                            log_dict = {
                                'policy/avg_reward': pm.get('avg_reward', 0.0),
                                'policy/avg_return': pm.get('avg_return', 0.0),
                                'policy/policy_loss': pm.get('policy_loss', 0.0),
                                'policy/entropy': pm.get('entropy', 0.0),
                                'policy/epsilon': pm.get('epsilon', 0.0),
                                'batch': pm.get('batch_idx', batch_idx),
                                'episode': episode_idx  # Use episode_idx from outer loop
                            }
                            if pm.get('avg_advantage', 0.0) != 0.0:  # GRPO
                                log_dict['policy/avg_advantage'] = pm.get('avg_advantage', 0.0)
                                log_dict['policy/std_advantage'] = pm.get('std_advantage', 1.0)
                                log_dict['policy/value_loss'] = pm.get('value_loss', 0.0)
                                log_dict['policy/avg_value_pred'] = pm.get('avg_value_pred', 0.0)
                                log_dict['policy/avg_return'] = pm.get('avg_return', 0.0)
                                log_dict['policy/value_pred_error'] = pm.get('value_pred_error', 0.0)
                            # Log at the step after the last step-level metric to ensure monotonic ordering
                            wandb.log(log_dict, step=policy_batch_step, commit=True)
                    
                    # Log policy metrics summary to console
                    if policy_metrics:
                        latest = policy_metrics[-1]
                        logger.info(
                            f"Policy metrics (Episode {episode_idx+1}, Batch {batch_idx+1}): "
                            f"avg_reward={latest.get('avg_reward', 0.0):.3f}, "
                            f"policy_loss={latest.get('policy_loss', 0.0):.4f}, "
                            f"entropy={latest.get('entropy', 0.0):.4f}, "
                            f"epsilon={latest.get('epsilon', 0.0):.3f}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to save policy metrics: {e}")
                
                # Log batch-level summary to wandb at the last step of the episode
                # This should be done immediately after the episode completes, before the next episode starts
                if wandb_initialized and best_rewards_batch:
                    batch_avg_reward = np.mean(best_rewards_batch)
                    batch_max_reward = np.max(best_rewards_batch)
                    batch_min_reward = np.min(best_rewards_batch)
                    try:
                        # Log at the step after the last step-level metric to ensure monotonic ordering
                        wandb.log({
                            'batch/avg_reward': batch_avg_reward,
                            'batch/max_reward': batch_max_reward,
                            'batch/min_reward': batch_min_reward,
                            'batch/prompts_processed': len(best_rewards_batch),
                            'batch/batch_idx': batch_idx,
                            'batch/episode': episode_idx
                        }, step=policy_batch_step, commit=True)
                    except Exception as e:
                        logger.warning(f"Failed to log batch metrics to wandb: {e}")
                
                # Process results
                process_batch_results(
                    best_results, best_rewards_batch, traces, batch_prompts,
                    batch_start, batch_size, metrics_path, episodes_per_prompt
                )
            else:
                # Unknown mode - log error and skip batch
                logger.error(
                    f"Unknown optimization mode: '{optimization_mode}'. "
                    f"Supported modes: {list(mode_map.keys())}. Skipping batch."
                )
                # Add placeholder rewards to maintain indexing
                for _ in batch_prompts:
                    all_rewards.append(float('-inf'))
        
        batch_time = time.time() - batch_start_time
        avg_time_per_prompt = batch_time / len(batch_prompts)
        
        print(f"  Batch completed in {batch_time:.1f}s ({avg_time_per_prompt:.2f}s/prompt)")
        if best_rewards_batch:
            print(f"  Best batch reward: {max(best_rewards_batch):.3f}")
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
                
                fmt = train_cfg['plots_format']
                plot_prefix = train_cfg['plots_prefix']
                plot_path = f"results/{plot_prefix}_rewards.{fmt}"
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Reward plot saved to: {plot_path}")
                
            except Exception as e:
                print(f"Could not generate plot: {e}")
    
    else:
        print("No successful training results!")
    
    # Finish wandb run if it was initialized
    if wandb_initialized:
        try:
            if wandb.run is not None:
                wandb.finish()
                logger.info("Wandb run finished")
        except Exception as e:
            logger.warning(f"Error finishing wandb run: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--prompts", type=int, help="Number of prompts (overrides config)")
    parser.add_argument("--episodes", type=int, help="Episodes per prompt (overrides config)")
    parser.add_argument("--steps", type=int, help="Steps per episode (overrides config)")
    parser.add_argument("--fast", action="store_true", help="Use speed-optimized hyperparameters")
    parser.add_argument("--dataset", type=str, default="advbench", choices=["advbench", "toxicchat"], help="Dataset to use (default: advbench)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="prompt-optimization", help="Wandb project name")
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
        dataset_name=args.dataset,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project
    )
    
    # Note: wandb.finish() is called at the end of train_on_dataset if wandb was initialized
