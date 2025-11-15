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
    return LengthPolicyOptimizer(agent)

def test_optimizer_initialization(optimizer):
    """Test optimizer initializes correctly"""
    assert optimizer.agent is not None
    assert optimizer.policy_net is not None
    assert optimizer.policy_optimizer is not None
    assert optimizer.state_dim == 4

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

