# Prompt Length Optimization with Reinforcement Learning# Prompt Length Optimization with Reinforcement Learning# Prompt & Generation Length Optimization via RL



A reinforcement learning approach to optimize prompt compression while maintaining generation quality. Uses the toxic-chat dataset to train policies that balance prompt length reduction with likelihood preservation.



## Quick StartA reinforcement learning approach to optimize prompt compression while maintaining generation quality. Uses the toxic-chat dataset to train policies that balance prompt length reduction with likelihood preservation.**Research Project** | *Aldan Creo* | MSDS @ UC San Diego



1. **Install dependencies:**

```bash

pip install -r requirements.txt## Quick Start---

```



2. **Train the policy:**

```bash1. **Install dependencies:**## 🎯 Research Concept

# Fast training (recommended - 50 prompts in ~2-3 minutes)

python train_fast.py --config config.yaml```bash



# Standard training (50 prompts in ~15-20 minutes)  pip install -r requirements.txtThis project explores the application of **reinforcement learning** to optimize prompt discovery for **fixed target completions**. Given a desired generation/completion, we want to find the minimal prompt that maximizes the posterior probability of that specific completion.

python train.py --config config.yaml

``````



3. **Evaluate on test set:**### Core Question

```bash

python eval.py --config config.yaml2. **Train the policy:**

```

```bash> *Given a target completion, can we train an RL agent to find the shortest prompt that maximizes P(completion | prompt)?*

## How It Works

python train.py --config config.yaml

- **Dataset**: Uses `lmsys/toxic-chat` model outputs as training data

- **Policy**: Learns when to compress vs. continue optimizing based on improvement rates```**Key Insight**: Traditional prompt optimization uses gradient descent on input embeddings, but this doesn't optimize for prompt length. We introduce an RL agent that can dynamically add/remove tokens to balance:

- **Reward**: Balances likelihood preservation (α) with length reduction (β)

- **Actions**: REMOVE (compress), KEEP (optimize), RETRACT (undo compression)1. **Likelihood Maximization**: High P(target_completion | prompt)  



## Performance Optimizations ⚡3. **Evaluate on test set:**2. **Length Minimization**: Shortest possible prompt



We've implemented several speed optimizations that provide **6-9x speedup**:```bash



- **Fast Training Mode** (`train_fast.py`): Optimized hyperparameters for faster convergencepython eval.py --config config.yaml---

- **Parallel Processing** (`train_parallel.py`): Multi-worker training (experimental)

- **Batch Processing**: Efficient progress tracking and vectorized operations```



See `OPTIMIZATION_RESULTS.md` for detailed performance analysis.## 💡 Key Ideas



## Configuration## How It Works



Edit `config.yaml` to adjust:### 1. RL-Based Prompt Length Optimization

- **Training**: Episodes per prompt, steps per episode, learning rates

- **Dataset**: Number of prompts, length filters  - **Dataset**: Uses `lmsys/toxic-chat` model outputs as training data

- **Reward**: α (likelihood weight) and β (compression penalty)

- **Performance**: Batch size, parallel workers- **Policy**: Learns when to compress vs. continue optimizing based on improvement rates**Concept**: Frame prompt discovery as an RL task where the agent learns to build optimal prompts token by token.

- **Evaluation**: Test set size, output paths

- **Reward**: Balances likelihood preservation (α) with length reduction (β)

- **Policy modes**: Set `train.optimization_mode: ppo_parallel` to enable the batched PPO policy updates described in this repo. Tune the `train.ppo.*` hyperparameters (clip, epochs, γ, λ, etc.) to control policy stability when running parallel batches.

## Quick Examples

- **Actions**: REMOVE (compress), KEEP (optimize), RETRACT (undo compression)**RL Formulation**:

```bash

# Quick experiment (10 prompts, ~30 seconds)- **State**: Current prompt sequence + target completion + current likelihood score

python train_fast.py --config config.yaml --prompts 10 --episodes 1 --steps 20

## Configuration- **Action**: Add token at start, remove token from start, or stop

# Development training (50 prompts, ~2 minutes)

python train_fast.py --config config.yaml --prompts 50 --episodes 1 --steps 30- **Reward**: α * log P(completion | prompt) - β * prompt_length



# Full training (100 prompts, ~5 minutes)Edit `config.yaml` to adjust:- **Policy**: Learns when to add/remove tokens to optimize the dual objective

python train_fast.py --config config.yaml --prompts 100 --episodes 2 --steps 50

```- **Training**: Episodes per prompt, steps per episode, learning rates



## Results- **Dataset**: Number of prompts, length filters  **Simplification**: Only modify tokens at sequence start to avoid complex positional encoding issues.



Training produces:- **Reward**: α (likelihood weight) and β (compression penalty)

- `models/trained_policy.pt` - Trained policy weights

- `results/eval_results.csv` - Evaluation metrics per test prompt- **Evaluation**: Test set size, output paths**Approach**:

- `OPTIMIZATION_RESULTS.md` - Performance analysis and speedup details

- Start with preset number of tokens (random or heuristic initialization)

## Research Background

## Results- Agent decides whether to add/remove tokens at the beginning

This work explores using reinforcement learning to find minimal prompts that maximize the posterior probability of target completions. The policy learns to balance:

- Evaluate P(target_completion | current_prompt) after each action

1. **Compression**: Removing tokens to reduce prompt length

2. **Optimization**: Continuing embedding optimization for better likelihoodTraining produces:- Optimize for high likelihood with minimal prompt length

3. **Quality**: Maintaining generation quality through likelihood preservation

- `models/trained_policy.pt` - Trained policy weights

The approach uses real-world model outputs from the toxic-chat dataset to ensure robust learning across diverse text types.
- `results/eval_results.csv` - Evaluation metrics per test prompt### 2. Comparison to Gradient-Based Methods

- Plots showing training progress and compression performance

**Traditional Approach**:

## Research Background- Gradient descent on input embeddings: `∇_embeddings log P(completion | prompt)`

- Final projection onto discrete token IDs

This work explores using reinforcement learning to find minimal prompts that maximize the posterior probability of target completions. The policy learns to balance:- **Limitation**: Fixed prompt length, no length optimization



1. **Compression**: Removing tokens to reduce prompt length**Our RL Approach**:

2. **Optimization**: Continuing embedding optimization for better likelihood- Dynamic prompt length via add/remove actions

3. **Quality**: Maintaining generation quality through likelihood preservation- Direct optimization of length-likelihood tradeoff

- More flexible than gradient-based methods for length constraints

The approach uses real-world model outputs from the toxic-chat dataset to ensure robust learning across diverse text types.
**Advantage**: Can discover that shorter prompts might actually yield higher likelihood for some completions.

---

## 🎓 Related Work & Context

### Yangkun Wang's ACL Paper
- **Key Innovation**: DPO with continuous representations
- Combines gradient-based optimization with discrete token mapping
- Currently fixed-length prompts
- Challenge: Making prompt length differentiable

### Existing Approaches
- **Gumbel-Softmax**: Rough approximation for gradient-based prompt optimization
- **Soft Prompts**: Continuous prompt embeddings (but not length-variable)
- **Prompt Tuning**: Optimizes prompt content, not length

### Open Challenges
- Prompt length isn't differentiable
- Hard to pass gradients through autoregressive generation
- RL requires reasonable initialization (can't start from random)

---

## 🔬 Research Approach

### RL Agent Design

**State Representation**:
```python
state = {
    'current_prompt': [token_ids],
    'target_completion': [token_ids], 
    'current_likelihood': float,
    'prompt_length': int
}
```

**Action Space**:
- `ADD_TOKEN_START`: Prepend a token to the prompt
- `REMOVE_TOKEN_START`: Remove first token from prompt  
- `STOP`: Finish prompt construction

**Reward Function**:
```python
R = α * log P(completion | prompt) - β * len(prompt) + γ * stop_bonus
```

**Policy Training**:
1. Start with random/heuristic prompt initialization
2. Agent takes actions to modify prompt
3. Evaluate likelihood after each modification
4. Update policy using PPO/DPO based on reward signal

### Experimental Setup

**Baseline Comparisons**:
- Gradient descent on embeddings (fixed length)
- Random prompt search
- Greedy token removal/addition
- Human-written prompts

**Datasets**:
- Common sense reasoning completions
- Code generation targets
- Creative writing samples
- Factual question-answer pairs

**Metrics**:
- Final likelihood P(completion | discovered_prompt)
- Prompt length efficiency (likelihood per token)
- Convergence speed and stability
- Generalization across completion types

---

## 📊 Why This Matters

### Practical Benefits
1. **Memory Efficiency**: Shorter prompts → more space for longer generations
2. **Cost Reduction**: Fewer tokens → lower inference costs
3. **Interpretability**: Minimal prompts reveal what the model actually needs
4. **Elegance**: Information-theoretic appeal of minimal sufficient representations

### Research Value
- Bridges discrete optimization (token selection) and continuous optimization (gradients)
- Novel application of RL to LLM efficiency
- Insights into attention mechanisms and token importance

---

## 🚧 Open Questions

1. **Technical Feasibility**
   - Can we make length optimization differentiable enough for practical training?
   - How do we handle positional encoding when removing tokens?
   - What's the right balance between sparsity and stability?

2. **Empirical Validation**
   - Does prompt compression actually improve downstream performance?
   - What's the quality-length tradeoff curve?
   - Do learned compressions generalize across tasks?

3. **Scalability**
   - Is the training cost worth the inference savings?
   - Can this work with very large models (70B+)?
   - How does this interact with existing optimizations (KV caching, etc.)?

---

## 🛠️ Getting Started

*Coming soon: experiment setup, baseline implementations, and initial results*

### Prerequisites
- PyTorch / JAX
- Transformers library
- RL framework (TBD: stable-baselines3, TRL, etc.)

### Roadmap
- [ ] Literature review
- [ ] Baseline implementation
- [ ] Attention sparsity experiments
- [ ] RL policy training
- [ ] Benchmarking & evaluation

---

## 📝 Notes & Ideas

### From Email Discussion (Oct 2025)

**Yangkun's Insights**:
- Inference cost dominated by generation, not prompt length
- Generation compression might be more impactful
- Gumbel-Softmax doesn't work well for LMs
- Training autoregressive model with gradients is challenging

**Secondary Benefits of Prompt Compression**:
- Memory efficiency in context window
- Interpretability improvements
- Human preference for conciseness
- Information-theoretic elegance

**Concerns**:
- Uncertain about scalability benefits
- May be focusing on wrong bottleneck (prompt vs generation)
- Technical challenges in making length differentiable

---

## 📚 References

*To be populated with relevant papers*

- Yangkun Wang et al. - ACL 2025 (DPO with continuous representations)
- Prompt compression literature
- Attention sparsity papers
- RL for LLM optimization

---

## 📧 Contact

**Aldan Creo**  
MSDS @ Halıcıoğlu Data Science Institute  
University of California San Diego  
🌐 [acmc.fyi](https://acmc.fyi)

---

*Last Updated: October 3, 2025*
