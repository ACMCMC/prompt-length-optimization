# Prompt Length Optimization via Reinforcement Learning

This repository explores how to shorten prompts without sacrificing the likelihood of a fixed target completion. A frozen causal language model (Pythia) is paired with a reinforcement learning (RL) policy that decides when to add, remove, or keep tokens while an inner gradient loop refines continuous prompt embeddings.

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train with the fast configuration (recommended for experimentation):

```bash
python train.py --config config.yaml
```

Evaluate a saved policy on held-out prompts:

```bash
python eval.py --config config.yaml
```

All generated artefacts (checkpoints, reward traces, plots) are written to `models/` and `results/`.

---

## Method Overview

Given a desired completion \( y = (y_1, \ldots, y_m) \), we seek a discrete prompt \( p \) that maximises the trade-off

\[
R(p) = \alpha \log P_\theta(y \mid p) - \beta \, |p|,
\]

where \( P_\theta \) is the frozen language model, \( \alpha \) weights fidelity to the target completion, and \( \beta \) penalises prompt length.

The optimisation loop alternates between:

1. **Continuous embedding updates**  
   For the current prompt embeddings \( \tilde{p} \), run several gradient steps on
   \[
   \mathcal{L}_{\text{emb}} = -\log P_\theta(y \mid \tilde{p}),
   \]
   using Adam to improve completion likelihood while keeping the prompt length fixed.

2. **RL-driven structure edits**  
   A policy network \( \pi_\phi(a \mid s) \) observes scalar features (likelihood, improvement trends, gradient norms, entropy) plus compact embedding summaries of the prompt and top next-token candidate. It samples actions from:
   - `REMOVE_LAST`
   - `KEEP`
   - `RETRACT` (restore previously removed token)
   - `ADD_FOCUSED` (append a top-k candidate or the next completion token)
   - `ADD_RANDOM`

   After executing the action, the prompt embeddings are adjusted and the reward \( R(p) \) (with shaping bonuses) is recorded.

3. **Policy update**  
   At the end of each episode, REINFORCE with normalised returns updates the policy parameters:
   \[
   \mathcal{L}_{\text{policy}} = - \sum_{t} \log \pi_\phi(a_t \mid s_t) \, \hat{G}_t.
   \]

4. **Discrete refinement**  
   The best prompt found so far is greedily pruned token-by-token if the reward does not drop.

---

## Key Components

- **`prompt_rl_poc.py`**
  - `PromptRLAgent`: wraps tokenizer/model loading, device selection (CUDA/MPS/CPU), and likelihood utilities.
  - `LengthPolicyOptimizer`: implements the two-level optimisation loop, state construction, action execution, and policy training.

- **`train.py`**
  - Loads toxic-chat prompts via `dataset_utils.py`.
  - Iterates through prompts in batches, invoking `LengthPolicyOptimizer.optimize_prompt`.
  - Saves checkpoints containing policy weights, reward history, and metadata.

- **`eval.py`**
  - Reloads a checkpoint and re-optimises each evaluation prompt for a single episode.
  - Writes compression statistics to CSV and optionally generates trace plots (`plot_utils.py`).

- **`config.yaml`**
  - Controls model name, dataset filters, reward weights (\(\alpha, \beta\)), inner-loop steps, maximum policy length, and plotting options.

---

## Configuration Highlights

| Key | Purpose |
| --- | --- |
| `train.init_len` | Initial prompt length in tokens before RL editing. |
| `train.lr_embeddings` / `train.embedding_opt_steps` | Inner-loop Adam configuration for continuous prompt optimisation. |
| `train.lr_policy` | Learning rate for the policy network. |
| `train.top_k_tokens` | Number of high-probability candidates considered by `ADD_FOCUSED`. |
| `train.policy_max_prompt_len` | Upper bound on prompt length during exploration. |
| `train.alpha`, `train.beta` | Reward weights for likelihood and length. |
| `eval.*` | Mirrors the training knobs for evaluation runs. |

Modify `config.yaml` to tailor the experiment (model size, number of prompts, reward balance, plotting).

---

## Repository Structure

```
prompt-length-optimization/
├── prompt_rl_poc.py         # RL agent and optimisation loop
├── train.py                 # Dataset-driven training pipeline
├── eval.py                  # Evaluation on held-out prompts
├── dataset_utils.py         # Toxic-chat dataset loading/splitting
├── plot_utils.py            # Plotting utilities for rewards/traces
├── config.yaml              # Experiment configuration
├── requirements.txt         # Python dependencies
└── results/, models/        # Generated outputs (ignored by git)
```

---

## Notes

- The code automatically prefers Apple’s Metal backend (`torch.backends.mps`) on macOS when available.
- Reward shaping and action design are intentionally modular; try adjusting `beta`, `policy_max_prompt_len`, or the action bonuses to study different compression behaviours.
- For quick iteration, lower `train.max_prompts`, `train.episodes_per_prompt`, and `train.steps_per_episode`; then scale up once the policy behaves as expected.

Happy experimenting!

