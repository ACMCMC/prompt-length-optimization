# Experiments Package

This package provides tools for running experiments comparing different optimization modes and performing hyperparameter sweeps with wandb.

## Quick Start

### Compare All Modes

```bash
python3 experiments/run_comparison.py --fast --no-wandb
```

### Run Single Experiment

```python
from experiments.run_experiment import run_experiment
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

result = run_experiment(
    mode='continuous_proj',
    config_path='config.yaml',
    fast_mode=False,
    use_wandb=True,
    projection_weight=0.1,
    distance_metric='l2'
)
```

### Run Wandb Sweep

1. Initialize sweep:
```bash
wandb sweep experiments/wandb_sweep.yaml
```

2. Run agent:
```bash
wandb agent <sweep_id>
```

## Available Modes

- **discrete**: GCG-based discrete token optimization
- **continuous**: Continuous embedding optimization
- **continuous_proj**: Continuous optimization with projection regularization

## Parameters

- `--config`: Path to config file (default: `config.yaml`)
- `--fast`: Run in fast mode (reduced parameters for quick testing)
- `--no-wandb`: Disable wandb logging
- `--wandb-project`: Wandb project name (default: `prompt-optimization`)
- `--projection-weight`: Projection regularization weight (default: 0.1)
- `--distance-metric`: Distance metric for projection (`l2` or `dot`, default: `l2`)

## Output

Experiments output:
- Mean reward (higher is better)
- Mean prompt length (lower is better for compression)
- Mean likelihood (higher is better)

All metrics are logged to wandb if enabled.

