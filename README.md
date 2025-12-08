# Prompt Length Optimization with Reinforcement Learning

We train a reinforcement learning policy to balance two goals when optimizing prompts: maximize the likelihood of a target completion while keeping the prompt short. Standard methods like GCG only optimize for likelihood and ignore length.

**Authors**: Aldan Creo, Atharv Nair (UC San Diego)

## The Problem

Prompt optimization methods find tokens that make a language model produce a specific response. But they treat prompt length as fixed. Longer prompts cost more to run, are slower, and waste context. We want to compress prompts while keeping high likelihood.

## Our Approach

We train a small RL policy (2-layer MLP) that decides when to:
- **Shrink**: Remove a token from the suffix
- **Grow**: Add a token to the suffix
- **Optimize**: Run the inner optimizer (e.g., GCG) on the suffix

The policy treats both the LM and the optimizer as black boxes, so it works with any plug-in optimizer.

**Result**: We compress adversarial suffixes by up to 37% while maintaining comparable likelihood values.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the policy (AdvBench dataset, Pythia-70M)
python train.py --config config.yaml

# Quick test
python train.py --config config_smoke_test.yaml

# Evaluate
python eval.py --config config.yaml
```

## How It Works

**MDP Setup:**
- **State**: Suffix length, log-likelihood, likelihood ratio, episode progress (5 dimensions)
- **Actions**: `shrink`, `grow`, `optimize`
- **Reward**: `α * log_likelihood - β * normalized_length`
- **Policy**: Trained with GRPO (Group Relative Policy Optimization)

**Optimization Modes:**
- `discrete`: GCG for token-level optimization (what we use)
- `continuous`: Optimize embeddings directly
- `continuous_proj`: Continuous + projection penalty

We focus on discrete mode because continuous modes had large projection losses when mapping embeddings back to tokens.

## Configuration

Edit `config.yaml`:
- **Training**: `episodes_per_prompt`, `steps_per_episode`, `batch_size`
- **Reward**: `alpha` (likelihood weight), `beta` (length penalty)
- **Optimizer**: GCG settings, GRPO hyperparameters
- **Dataset**: AdvBench or ToxicChat

Use `config_smoke_test.yaml` for quick testing with minimal settings.

## Results

Training outputs:
- `models/trained_policy.pt` - Policy weights
- `results/eval_results.csv` - Evaluation metrics
- Training plots showing likelihood, length, and reward curves

The policy learns to compress suffixes while maintaining or improving likelihood. See the paper for detailed results.

## Code Structure

```
prompt_optimization/
├── agent.py          # RL agent and GRPO implementation
├── optimizer.py      # GCG and continuous optimizers
├── interface.py      # Model interface
└── datasets.py       # AdvBench and ToxicChat loaders

train.py              # Main training script
eval.py               # Evaluation script
config.yaml           # Full configuration
config_smoke_test.yaml # Quick test configuration
```
