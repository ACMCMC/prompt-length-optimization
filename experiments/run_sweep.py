#!/usr/bin/env python3
"""
Script for wandb hyperparameter sweeps.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import wandb
from experiments.run_experiment import run_single_experiment
import yaml

def main():
    wandb.init()
    config = wandb.config
    
    # Load base config from config.yaml
    with open("config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
    
    # wandb.config overrides will be automatically detected by run_single_experiment
    # No need to manually update base_config - run_single_experiment handles wandb.config directly
    
    # Run experiment (wandb.config will be automatically detected and used to override config.yaml values)
    result = run_single_experiment(
        mode=config.mode,
        config=base_config,
        fast_mode=False,
        use_wandb=True,
        wandb_project=wandb.run.project,
        wandb_run_name=None  # Use wandb's auto naming
    )
    
    if 'error' not in result:
        log_dict = {
            'final_mean_reward': result['mean_reward'],
            'final_mean_length': result['mean_length'],
            'final_mean_likelihood': result['mean_likelihood']
        }
        if 'mean_projection_loss' in result:
            log_dict['final_mean_projection_loss'] = result['mean_projection_loss']
        wandb.log(log_dict)

if __name__ == "__main__":
    main()

