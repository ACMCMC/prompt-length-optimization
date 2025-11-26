#!/usr/bin/env python3
"""
Script to run comparison experiments for all three optimization modes.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiment import compare_modes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare optimization modes")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (reduced parameters)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="prompt-optimization", help="Wandb project name")
    args = parser.parse_args()
    
    results = compare_modes(
        config_path=args.config,
        fast_mode=args.fast,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project
    )
    
    print("\n✓ Experiments completed!")

