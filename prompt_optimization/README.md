# Prompt Optimization Package

Modular package for RL-based prompt length optimization with pluggable continuous/discrete optimizers.

## Package Structure

```
prompt_optimization/
├── __init__.py          # Package exports
├── agent.py             # PromptRLAgent: model interactions
├── optimizer.py         # LengthPolicyOptimizer: RL policy
├── interface.py         # BasePromptOptimizer: abstract interface
├── continuous.py        # ContinuousPromptOptimizer: embedding optimization
└── discrete.py         # DiscretePromptOptimizer: GCG token optimization
```

## Usage

```python
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer

# Initialize agent
agent = PromptRLAgent(model_name="EleutherAI/pythia-70m")

# Create optimizer
optimizer = LengthPolicyOptimizer(agent)

# Optimize prompts
target_completions = ["hello world", "test completion"]
prompts, rewards, traces = optimizer.optimize_prompts_batch(
    target_completions=target_completions,
    episodes=3,
    steps_per_episode=50,
    mode="continuous"  # or "discrete"
)
```

## Running Tests

```bash
pytest tests/ -v
```

All 18 tests pass, covering:
- Agent initialization and likelihood computation
- Continuous optimizer interface and methods
- Discrete optimizer interface
- Policy optimizer integration
- Interface contract compliance

