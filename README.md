# Prompt & Generation Length Optimization via RL

**Research Project** | *Aldan Creo* | MSDS @ UC San Diego

---

## 🎯 Research Concept

This project explores the application of **reinforcement learning** to optimize language model generation for both **quality** (generation likelihood) and **efficiency** (minimal token length). The core idea is to develop policies that can produce high-quality outputs while minimizing computational cost.

### Core Question

> *Can we train a policy to maximize generation likelihood while incurring minimal token length cost?*

This spans two complementary directions:
1. **Prompt Compression**: Minimize input token count while maintaining output quality
2. **Generation Compression**: Minimize output token count while maintaining semantic fidelity

---

## 💡 Key Ideas

### 1. Attention-Based Prompt Compression

**Concept**: Modify the optimization objective to encourage attention sparsity in the prompt.

**Approach**:
- Use L1 regularization on attention weights to push certain prompt token attentions → 0
- When a token's attention contribution becomes negligible, it can be "masked out"
- Progressively reduce prompt length while maintaining generation quality

**Challenges**:
- Must selectively mask tokens (not all)
- Positional encoding complications when tokens are removed
- Risk of optimization instability (too many competing objectives)

**Potential Loss Function**:
```
L = L_generation + λ * L_sparsity
```
Where `L_sparsity` encourages specific attention weights to approach zero.

### 2. RL Policy for Length-Quality Tradeoff

**Concept**: Frame the problem as an RL task where the agent learns to balance generation quality against token budget.

**Formulation**:
- **State**: Current generation context (prompt + tokens generated so far)
- **Action**: Next token to generate (or decision to stop)
- **Reward**: Combination of likelihood score and length penalty

**Reward Function**:
```
R = α * log P(generation | prompt) - β * token_count
```

### 3. Generation Compression (Future Direction)

**Concept**: Compress generated tokens rather than prompt tokens, as generation dominates inference cost.

**Motivation**:
- Inference cost is dominated by generation length, not prompt length
- More impactful for scalability
- Different technical challenges than prompt compression

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

## 🔬 Potential Research Directions

### Direction A: Attention Sparsity for Prompt Compression
**Goal**: Learn to identify and remove unnecessary prompt tokens

**Approach**:
1. Fine-tune model with sparsity-inducing loss on attention weights
2. Threshold attention contributions to identify removable tokens
3. Iteratively compress prompts while monitoring generation quality

**Metrics**:
- Compression ratio (original length / compressed length)
- Generation quality (perplexity, task performance)
- Attention distribution entropy

### Direction B: RL-Based Token Budget Optimization
**Goal**: Train a policy to generate high-quality outputs within token budgets

**Approach**:
1. Define reward as likelihood - length_penalty
2. Use PPO/DPO to train policy
3. Experiment with different penalty schedules

**Metrics**:
- Pareto frontier of quality vs. length
- Inference cost reduction
- Quality degradation at various compression levels

### Direction C: Generation Compression
**Goal**: Compress generated outputs while maintaining semantic content

**Approach**:
1. Train model to generate more information-dense tokens
2. Post-generation compression via paraphrasing
3. Multi-stage generation with refinement

**Metrics**:
- Semantic similarity (embedding distance)
- Information retention (task-specific metrics)
- Compression ratio

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
