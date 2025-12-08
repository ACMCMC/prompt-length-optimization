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
import yaml
from prompt_optimization import PromptRLAgent, LengthPolicyOptimizer

config = yaml.safe_load(open("config.yaml"))
train_cfg = config["train"]
grpo_cfg = train_cfg["grpo"]
policy_cfg = train_cfg["policy"]

agent = PromptRLAgent(model_name=config["model"])
optimizer = LengthPolicyOptimizer(
    agent,
    epsilon=train_cfg["epsilon"],
    epsilon_decay=train_cfg["epsilon_decay"],
    epsilon_min=train_cfg["epsilon_min"],
    entropy_coef=grpo_cfg["entropy_coef"],
    temperature=train_cfg["temperature"],
    grpo_clip=grpo_cfg["clip"],
    grpo_epochs=grpo_cfg["epochs"],
    grpo_gamma=grpo_cfg["gamma"],
    policy_hidden_size=policy_cfg["hidden_size"],
    max_grad_norm=grpo_cfg["max_grad_norm"],
)

prompts, rewards, traces = optimizer.optimize_prompts_batch(
    target_completions=["hello world", "test completion"],
    episodes=train_cfg["episodes_per_prompt"],
    steps_per_episode=train_cfg["steps_per_episode"],
    initial_prompt_length=train_cfg["init_len"],
    lr_embeddings=train_cfg["lr_embeddings"],
    alpha=train_cfg["alpha"],
    beta=train_cfg["beta"],
    mode=train_cfg["optimization_mode"],
    batch_size=train_cfg["batch_size"],
    max_suffix_len=train_cfg["max_suffix_len"],
    init_len=train_cfg["init_len"],
    rollouts_per_prompt=train_cfg["rollouts_per_prompt"],
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

