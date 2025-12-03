"""Unit tests for LengthPolicyOptimizer"""

import torch
import pytest
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.optimizer import LengthPolicyOptimizer

@pytest.fixture
def agent():
    return PromptRLAgent(model_name="EleutherAI/pythia-70m")

@pytest.fixture
def optimizer(agent):
    return LengthPolicyOptimizer(
        agent,
        epsilon=0.3,
        epsilon_decay=0.97,
        epsilon_min=0.05,
        entropy_coef=0.01,
        temperature=1.5,
        grpo_clip=0.2,
        grpo_epochs=2,
        grpo_gamma=0.99,
        policy_hidden_size=64,
        max_grad_norm=0.5,
    )

def test_optimizer_initialization(optimizer):
    """Test optimizer initializes correctly"""
    assert optimizer.agent is not None
    assert optimizer.policy_net is not None
    assert optimizer.policy_optimizer is not None
    assert optimizer.state_dim == 5

def test_prepare_completions(optimizer):
    """Test completion preparation"""
    completions = ["hello world", "test"]
    tokens, lengths = optimizer._prepare_completions(completions)
    
    assert tokens.shape[0] == 2
    assert lengths.shape == (2,)
    assert lengths[0].item() >= lengths[1].item()  # First should be longer or equal

def test_optimize_prompts_batch_empty(optimizer):
    """Test optimization with empty batch"""
    prompts, rewards, traces = optimizer.optimize_prompts_batch(
        target_completions=[],
        episodes=1,
        steps_per_episode=2
    )
    
    assert prompts == []
    assert rewards == []
    assert traces == []

def test_optimize_prompts_batch_continuous(optimizer):
    """Test continuous mode optimization (short run)"""
    completions = ["test completion"]
    prompts, rewards, traces = optimizer.optimize_prompts_batch(
        target_completions=completions,
        episodes=1,
        steps_per_episode=2,
        initial_prompt_length=3,
        mode="continuous"
    )
    
    assert len(prompts) == 1
    assert len(rewards) == 1
    assert len(traces) == 1
    assert isinstance(prompts[0], torch.Tensor)

def test_optimize_prompts_batch_discrete(optimizer):
    """Test discrete mode optimization (short run)"""
    completions = ["test completion"]
    prompts, rewards, traces = optimizer.optimize_prompts_batch(
        target_completions=completions,
        episodes=1,
        steps_per_episode=2,
        initial_prompt_length=3,
        mode="discrete"
    )
    
    assert len(prompts) == 1
    assert len(rewards) == 1
    assert len(traces) == 1
    assert isinstance(prompts[0], torch.Tensor)


def test_group_advantages_zero_mean_per_prompt(optimizer):
    """Per-prompt baselines should zero-center advantages."""
    rewards = torch.tensor(
        [
            [1.0, 3.0, 4.0, 8.0],
            [2.0, 6.0, 5.0, 7.0],
        ]
    )
    prompt_indices = torch.tensor([0, 0, 1, 1])
    advantages = optimizer._compute_group_advantages(rewards, prompt_indices)

    # First two columns correspond to prompt 0, last two to prompt 1
    assert torch.allclose(advantages[:, :2].mean(dim=1), torch.zeros(advantages.shape[0]))
    assert torch.allclose(advantages[:, 2:].mean(dim=1), torch.zeros(advantages.shape[0]))


def test_group_advantages_global_baseline(optimizer):
    """Without indices, advantages fall back to global baseline."""
    rewards = torch.tensor([[2.0, 6.0, 10.0]])
    advantages = optimizer._compute_group_advantages(rewards, None)
    assert torch.allclose(advantages.mean(dim=1), torch.zeros(advantages.shape[0]))


def test_clipped_policy_loss_matches_formula(optimizer):
    """Clipped surrogate loss should follow GRPO paper."""
    ratio = torch.tensor([[1.2, 0.8]])
    advantages = torch.tensor([[1.0, -1.0]])
    loss = optimizer._clipped_policy_loss(ratio, advantages, clip_epsilon=0.1)

    expected_first = min(1.2 * 1.0, 1.1 * 1.0)  # clipped ratio = 1.1
    expected_second = min(0.8 * -1.0, 0.9 * -1.0)  # clipped ratio = 0.9
    expected = -((expected_first + expected_second) / 2.0)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_optimizer_has_no_value_network(optimizer):
    """Ensure no critic/value network is kept around."""
    assert not hasattr(optimizer, "value_net")

