"""
Run experiments comparing different optimization modes.
Supports wandb integration for tracking and hyperparameter sweeps.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
import random
import numpy as np
import wandb
from typing import Dict, List, Optional
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_experiment(
    mode: str,
    config: Dict,
    fast_mode: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "prompt-optimization",
    wandb_run_name: Optional[str] = None
) -> Dict:
    """
    Run a single experiment with a given optimization mode.
    
    Args:
        mode: Optimization mode ('discrete', 'continuous', 'continuous_proj')
        config: Configuration dictionary (from config.yaml, optionally overridden by wandb.config)
        fast_mode: If True, use reduced parameters for quick testing
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        wandb_run_name: Custom run name (defaults to mode)
    
    Returns:
        Dictionary with experiment results
    """
    train_cfg = config['train']
    seed = config.get('seed', 2262)
    set_seed(seed)
    
    # Check if wandb.config exists (from sweep), use it to override config values
    wandb_config = None
    try:
        if wandb.run is not None:
            wandb_config = wandb.config
    except:
        pass
    
    # Get all parameters from config.yaml, override with wandb.config if available
    # No hardcoded defaults - everything comes from config.yaml
    if wandb_config:
        episodes_per_prompt = wandb_config.get('episodes_per_prompt', train_cfg['episodes_per_prompt'])
        steps_per_episode = wandb_config.get('steps_per_episode', train_cfg['steps_per_episode'])
        max_prompts = wandb_config.get('max_prompts', train_cfg['max_prompts'])
        init_len = wandb_config.get('init_len', train_cfg['init_len'])
        lr_embeddings = wandb_config.get('lr_embeddings', train_cfg['lr_embeddings'])
        alpha = wandb_config.get('alpha', train_cfg['alpha'])
        beta = wandb_config.get('beta', train_cfg['beta'])
        projection_weight = wandb_config.get('projection_weight', train_cfg['projection_weight'])
        distance_metric = wandb_config.get('distance_metric', train_cfg['distance_metric'])
    else:
        episodes_per_prompt = train_cfg['episodes_per_prompt']
        steps_per_episode = train_cfg['steps_per_episode']
        max_prompts = train_cfg['max_prompts']
        init_len = train_cfg['init_len']
        lr_embeddings = train_cfg['lr_embeddings']
        alpha = train_cfg['alpha']
        beta = train_cfg['beta']
        projection_weight = train_cfg['projection_weight']
        distance_metric = train_cfg['distance_metric']
    
    # Apply fast_mode scaling if requested (reduces parameters for quick testing)
    if fast_mode:
        episodes_per_prompt = max(1, episodes_per_prompt // 15)  # Reduce episodes
        steps_per_episode = max(1, steps_per_episode // 30)  # Reduce steps
        max_prompts = min(3, max_prompts)  # Limit to 3 prompts
        init_len = max(4, init_len // 4)  # Reduce initial length
    
    # Initialize wandb (only if not already initialized by sweep)
    if use_wandb:
        try:
            if wandb.run is None:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"{mode}_{'fast' if fast_mode else 'full'}",
                    config={
                        'mode': mode,
                        'episodes_per_prompt': episodes_per_prompt,
                        'steps_per_episode': steps_per_episode,
                        'init_len': init_len,
                        'max_prompts': max_prompts,
                        'lr_embeddings': lr_embeddings,
                        'alpha': alpha,
                        'beta': beta,
                        'projection_weight': projection_weight if mode == 'continuous_proj' else None,
                        'distance_metric': distance_metric if mode == 'continuous_proj' else None,
                        'seed': seed,
                        'fast_mode': fast_mode
                    }
                )
        except:
            pass
    
    # Load dataset (using AdvBench for proper prompt-completion pairs)
    from datasets import load_dataset
    print("Loading AdvBench dataset...")
    raw = load_dataset("walledai/AdvBench", split='train')
    PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
    COMPLETION_KEYS = ["target", "completion", "output", "response", "answer"]
    
    prompt_completion_pairs = []
    for ex in raw:
        # Find prompt-like field
        base = None
        for k in PROMPT_KEYS:
            if k in ex and ex[k]:
                base = ex[k]
                break
        if base is None:
            continue
        # Find completion-like field
        target = None
        for k in COMPLETION_KEYS:
            if k in ex and ex[k] is not None:
                target = ex[k]
                break
        if target is None:
            target = ""
        prompt_completion_pairs.append({'base': base.strip(), 'target': target.strip()})
    
    # Deterministic sampling/shuffle
    import random as _rand
    _rand.seed(seed)
    _rand.shuffle(prompt_completion_pairs)
    prompt_completion_pairs = prompt_completion_pairs[:max_prompts]
    print(f"Loaded {len(prompt_completion_pairs)} AdvBench examples")
    
    if not prompt_completion_pairs:
        raise ValueError("No valid prompts found in dataset")
    
    # Initialize agent and optimizer
    model_name = config.get('model', 'EleutherAI/pythia-70m')
    agent = PromptRLAgent(model_name=model_name)
    optimizer = LengthPolicyOptimizer(agent)
    
    # Set projection parameters for continuous_proj mode
    if mode == 'continuous_proj':
        optimizer.projection_weight = projection_weight
        optimizer.distance_metric = distance_metric
    
    # Track metrics
    all_rewards = []
    all_lengths = []
    all_likelihoods = []
    all_projection_losses = []
    
    # Store examples for saving
    examples = []
    num_examples_to_save = 5
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {mode}")
    print(f"{'='*60}")
    
    # Train on prompts in batches of 64
    batch_size = 64
    for batch_start in range(0, len(prompt_completion_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(prompt_completion_pairs))
        batch_pairs = prompt_completion_pairs[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1} (prompts {batch_start+1}-{batch_end}/{len(prompt_completion_pairs)})")
        
        # Collect completions for this batch
        batch_completions = []
        batch_base_prompts = []
        batch_indices = []
        
        for i, pair in enumerate(batch_pairs):
            completion = pair.get('target', '')
            base_prompt = pair.get('base', '')
            if completion:
                batch_completions.append(completion)
                batch_base_prompts.append(base_prompt)
                batch_indices.append(batch_start + i)
        
        if not batch_completions:
            continue
        
        # Optimize all prompts in batch in parallel
        optimized_prompts, rewards, traces, _ = optimizer.optimize_prompts_batch(
            target_completions=batch_completions,
            episodes=episodes_per_prompt,
            steps_per_episode=steps_per_episode,
            initial_prompt_length=init_len,
            lr_embeddings=lr_embeddings,
            alpha=alpha,
            beta=beta,
            mode=mode,
            batch_size=batch_size
        )
        
        # Process results for each prompt in the batch
        for batch_idx, (global_idx, completion, base_prompt) in enumerate(zip(batch_indices, batch_completions, batch_base_prompts)):
            if batch_idx < len(optimized_prompts) and optimized_prompts[batch_idx].numel() > 0:
                final_reward = rewards[batch_idx] if batch_idx < len(rewards) else 0.0
                final_length = len(optimized_prompts[batch_idx]) if optimized_prompts[batch_idx].numel() > 0 else 0
                
                # Decode optimized prompt
                optimized_prompt_text = agent.tokenizer.decode(optimized_prompts[batch_idx], skip_special_tokens=True)
                
                # Compute likelihood for final prompt
                if final_length > 0:
                    completion_tokens = agent.tokenizer.encode(completion, return_tensors='pt').to(agent.device)[0]
                    prompt_tokens = optimized_prompts[batch_idx]
                
                with torch.no_grad():
                    # Use get_likelihoods_batch which expects embeddings or tokens
                    # For tokens, we need to convert to embeddings first
                    if mode in ['continuous', 'continuous_proj']:
                        # Already embeddings, but we have tokens now - need to get embeddings
                        embedding_layer = agent.model.get_input_embeddings()
                        prompt_embeds = embedding_layer(prompt_tokens.unsqueeze(0))
                        comp_embeds = embedding_layer(completion_tokens.unsqueeze(0))
                        likelihoods = agent.get_likelihoods_batch(
                            prompt_embeds, 
                            completion_tokens.unsqueeze(0),
                            torch.tensor([len(completion_tokens)], device=agent.device),
                            requires_grad=False
                        )
                    else:
                        # Discrete mode - tokens directly
                        embedding_layer = agent.model.get_input_embeddings()
                        prompt_embeds = embedding_layer(prompt_tokens.unsqueeze(0))
                        likelihoods = agent.get_likelihoods_batch(
                            prompt_embeds,
                            completion_tokens.unsqueeze(0),
                            torch.tensor([len(completion_tokens)], device=agent.device),
                            requires_grad=False
                        )
                    likelihood = likelihoods[0].item()
                
                all_likelihoods.append(likelihood)
            else:
                all_likelihoods.append(float('-inf'))
            
            all_rewards.append(final_reward)
            all_lengths.append(final_length)
            
            # Extract projection loss from traces (if available)
            # Traces is a list of episode traces, get projection loss from last episode if available
            projection_loss = None
            if traces and len(traces) > 0:
                projection_loss = traces[-1].get('projection_loss', None)
            if projection_loss is not None:
                all_projection_losses.append(projection_loss)
            
            # Save example (first few, evenly spaced, and best/worst)
            if len(examples) < num_examples_to_save or global_idx % (len(prompt_completion_pairs) // num_examples_to_save) == 0:
                example = {
                    'base_prompt': base_prompt,
                    'optimized_prompt': optimized_prompt_text,
                    'completion': completion,
                    'reward': final_reward,
                    'length': final_length,
                    'likelihood': all_likelihoods[-1] if all_likelihoods else float('-inf'),
                    'projection_loss': projection_loss
                }
                examples.append(example)
                # Keep only the most recent examples if we exceed the limit
                if len(examples) > num_examples_to_save:
                    examples = examples[-num_examples_to_save:]
            
            # Log to wandb
            if use_wandb:
                log_dict = {
                    'reward': final_reward,
                    'prompt_length': final_length,
                    'likelihood': all_likelihoods[-1] if all_likelihoods else 0.0,
                    'prompt_idx': global_idx
                }
                if projection_loss is not None:
                    log_dict['projection_loss'] = projection_loss
                wandb.log(log_dict)
    
    # Compute summary statistics
    if all_rewards:
        results = {
            'mode': mode,
            'num_prompts': len(all_rewards),
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_length': np.mean(all_lengths),
            'std_length': np.std(all_lengths),
            'mean_likelihood': np.mean([ll for ll in all_likelihoods if ll != float('-inf')]),
            'std_likelihood': np.std([ll for ll in all_likelihoods if ll != float('-inf')]),
            'min_length': np.min(all_lengths),
            'max_length': np.max(all_lengths),
            'examples': examples
        }
        if all_projection_losses:
            results['mean_projection_loss'] = np.mean(all_projection_losses)
            results['std_projection_loss'] = np.std(all_projection_losses)
    else:
        results = {
            'mode': mode,
            'num_prompts': 0,
            'error': 'No valid results',
            'examples': []
        }
    
    # Save examples to file
    if examples:
        output_dir = Path("experiments/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        examples_file = output_dir / f"examples_{mode}_{wandb_run_name or 'default'}.txt"
        with open(examples_file, 'w') as f:
            f.write(f"Optimization Mode: {mode}\n")
            f.write(f"Number of Examples: {len(examples)}\n")
            f.write("=" * 80 + "\n\n")
            for idx, ex in enumerate(examples, 1):
                f.write(f"Example {idx}:\n")
                f.write(f"  Base Prompt: {ex['base_prompt']}\n")
                f.write(f"  Optimized Prompt: {ex['optimized_prompt']}\n")
                f.write(f"  Completion: {ex['completion']}\n")
                f.write(f"  Reward: {ex['reward']:.3f}\n")
                f.write(f"  Length: {ex['length']}\n")
                f.write(f"  Likelihood: {ex['likelihood']:.3f}\n")
                if ex['projection_loss'] is not None:
                    f.write(f"  Projection Loss: {ex['projection_loss']:.4f}\n")
                f.write("\n" + "-" * 80 + "\n\n")
        print(f"Saved {len(examples)} examples to {examples_file}")
        
        # Log examples to wandb as a table
        if use_wandb:
            try:
                examples_table = wandb.Table(columns=["base_prompt", "optimized_prompt", "completion", "reward", "length", "likelihood", "projection_loss"])
                for ex in examples:
                    examples_table.add_data(
                        ex['base_prompt'],
                        ex['optimized_prompt'],
                        ex['completion'],
                        ex['reward'],
                        ex['length'],
                        ex['likelihood'],
                        ex['projection_loss'] if ex['projection_loss'] is not None else "N/A"
                    )
                wandb.log({"examples": examples_table})
            except:
                pass
    
    # Log summary to wandb
    if use_wandb:
        if 'error' not in results:
            log_dict = {
                'final_mean_reward': results['mean_reward'],
                'final_mean_length': results['mean_length'],
                'final_mean_likelihood': results['mean_likelihood']
            }
            if 'mean_projection_loss' in results:
                log_dict['final_mean_projection_loss'] = results['mean_projection_loss']
            wandb.log(log_dict)
        wandb.finish()
    
    return results


def compare_modes(
    config_path: str = "config.yaml",
    fast_mode: bool = True,
    use_wandb: bool = True,
    wandb_project: str = "prompt-optimization"
) -> Dict:
    """
    Run experiments for all three optimization modes and compare results.
    
    Args:
        config_path: Path to config file
        fast_mode: If True, use reduced parameters for quick testing
        use_wandb: Whether to log to wandb
        wandb_project: Wandb project name
        projection_weight: Weight for projection regularization
        distance_metric: Distance metric for projection
    
    Returns:
        Dictionary with comparison results
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    modes = ['discrete', 'continuous', 'continuous_proj']
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Comparing optimization modes")
    print(f"{'='*60}\n")
    
    for mode in modes:
        try:
            result = run_single_experiment(
                mode=mode,
                config=config,
                fast_mode=fast_mode,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                wandb_run_name=f"{mode}_comparison"
            )
            results[mode] = result
            print(f"\n{mode.upper()} Results:")
            if 'error' not in result:
                print(f"  Mean Reward: {result['mean_reward']:.3f} ± {result['std_reward']:.3f}")
                print(f"  Mean Length: {result['mean_length']:.1f} ± {result['std_length']:.1f}")
                print(f"  Mean Likelihood: {result['mean_likelihood']:.3f} ± {result['std_likelihood']:.3f}")
                if 'mean_projection_loss' in result:
                    print(f"  Mean Projection Loss: {result['mean_projection_loss']:.4f} ± {result['std_projection_loss']:.4f}")
            else:
                print(f"  Error: {result['error']}")
        except Exception as e:
            print(f"\nError running {mode}: {e}")
            results[mode] = {'error': str(e)}
    
    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if all('error' not in results[m] for m in modes):
        has_projection_loss = any('mean_projection_loss' in results[m] for m in modes)
        if has_projection_loss:
            print(f"\n{'Mode':<20} {'Reward':<15} {'Length':<15} {'Likelihood':<15} {'Proj Loss':<15}")
            print("-" * 80)
            for mode in modes:
                r = results[mode]
                proj_loss_str = f"{r.get('mean_projection_loss', 0.0):>6.4f} ± {r.get('std_projection_loss', 0.0):>4.4f}" if 'mean_projection_loss' in r else "N/A"
                print(f"{mode:<20} {r['mean_reward']:>8.3f} ± {r['std_reward']:>4.3f}  "
                      f"{r['mean_length']:>6.1f} ± {r['std_length']:>4.1f}  "
                      f"{r['mean_likelihood']:>8.3f} ± {r['std_likelihood']:>4.3f}  {proj_loss_str}")
        else:
            print(f"\n{'Mode':<20} {'Reward':<15} {'Length':<15} {'Likelihood':<15}")
            print("-" * 65)
            for mode in modes:
                r = results[mode]
                print(f"{mode:<20} {r['mean_reward']:>8.3f} ± {r['std_reward']:>4.3f}  "
                      f"{r['mean_length']:>6.1f} ± {r['std_length']:>4.1f}  "
                      f"{r['mean_likelihood']:>8.3f} ± {r['std_likelihood']:>4.3f}")
    
    return results


def run_experiment(
    mode: str,
    config_path: str = "config.yaml",
    fast_mode: bool = False,
    use_wandb: bool = True,
    wandb_project: str = "prompt-optimization"
) -> Dict:
    """
    Run a single experiment (convenience wrapper).
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return run_single_experiment(
        mode=mode,
        config=config,
        fast_mode=fast_mode,
        use_wandb=use_wandb,
        wandb_project=wandb_project
    )

