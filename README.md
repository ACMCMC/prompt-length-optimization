# Prompt & Generation Length Optimization via RL

**Research Project** | *Aldan Creo* | MSDS @ UC San Diego

---

## 🎯 Research Concept

This project explores the application of **reinforcement learning** to optimize prompt discovery for **fixed target completions**. Given a desired generation/completion, we want to find the minimal prompt that maximizes the posterior probability of that specific completion.

### Core Question

> *Given a target completion, can we train an RL agent to find the shortest prompt that maximizes P(completion | prompt)?*

**Key Insight**: Traditional prompt optimization uses gradient descent on input embeddings, but this doesn't optimize for prompt length. We introduce an RL agent that can dynamically add/remove tokens to balance:
1. **Likelihood Maximization**: High P(target_completion | prompt)  
2. **Length Minimization**: Shortest possible prompt

---

## 💡 Key Ideas

### 1. RL-Based Prompt Length Optimization

**Concept**: Frame prompt discovery as an RL task where the agent learns to build optimal prompts token by token.

**RL Formulation**:
- **State**: Current prompt sequence + target completion + current likelihood score
- **Action**: Add token at start, remove token from start, or stop
- **Reward**: α * log P(completion | prompt) - β * prompt_length
- **Policy**: Learns when to add/remove tokens to optimize the dual objective

**Simplification**: Only modify tokens at sequence start to avoid complex positional encoding issues.

**Approach**:
- Start with preset number of tokens (random or heuristic initialization)
- Agent decides whether to add/remove tokens at the beginning
- Evaluate P(target_completion | current_prompt) after each action
- Optimize for high likelihood with minimal prompt length

### 2. Comparison to Gradient-Based Methods

**Traditional Approach**:
- Gradient descent on input embeddings: `∇_embeddings log P(completion | prompt)`
- Final projection onto discrete token IDs
- **Limitation**: Fixed prompt length, no length optimization

**Our RL Approach**:
- Dynamic prompt length via add/remove actions
- Direct optimization of length-likelihood tradeoff
- More flexible than gradient-based methods for length constraints

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
