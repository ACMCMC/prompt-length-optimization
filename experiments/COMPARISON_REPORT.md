# Optimization Modes Comparison Report

## Summary

Successfully ran experiments comparing three optimization modes:
1. **Discrete** (GCG-based token optimization)
2. **Continuous** (Embedding optimization)
3. **Continuous with Projection** (Embedding optimization + vocabulary proximity regularization)

## Results (Fast Mode - 3 prompts, 2 episodes, 5 steps)

| Mode | Mean Reward | Mean Length | Mean Likelihood | Projection Loss |
|------|------------|-------------|-----------------|-----------------|
| Discrete | -80.553 ± 24.698 | 12.3 ± 2.9 | -79.770 ± 24.756 | 0.0000 (no projection) |
| Continuous | -79.088 ± 22.960 | 11.0 ± 2.2 | -84.154 ± 21.963 | 2.2798 ± 0.0161 |
| Continuous Proj | -79.088 ± 22.960 | 11.0 ± 2.2 | -84.154 ± 21.963 | 2.2798 ± 0.0161 |

**Note**: Projection loss measures the average L2 distance from optimized embeddings to their nearest vocabulary token at the final projection step. This is where information loss occurs when converting embeddings to discrete tokens.

## Observations

1. **Discrete mode**: 
   - Slightly worse reward but better likelihood
   - Longer prompts on average
   - More variance in results

2. **Continuous modes**:
   - Better reward scores
   - Shorter prompts (better compression)
   - Slightly worse likelihood scores
   - Less variance

3. **Continuous vs Continuous Proj**:
   - Currently producing identical results (needs investigation)
   - **Projection loss is now tracked**: Both show ~2.28 average distance to nearest vocab token
   - Projection regularization should reduce this loss, but may need:
     - Higher projection weight (currently 0.1)
     - More training steps
     - Different distance metric (trying 'dot' instead of 'l2')
   - **Key insight**: We ARE projecting at the end (embeddings → tokens), and that's where information loss happens. The regularization should minimize this loss.

## Next Steps

1. Run full-scale experiments with more prompts and episodes
2. Tune projection regularization weight
3. Investigate why continuous_proj produces identical results to continuous
4. Perform hyperparameter sweeps with wandb

## Usage

```bash
# Quick comparison
python3 experiments/run_comparison.py --fast --no-wandb

# Full comparison with wandb
python3 experiments/run_comparison.py --wandb-project prompt-optimization

# Run sweep
wandb sweep experiments/wandb_sweep.yaml
```
