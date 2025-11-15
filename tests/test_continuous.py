"""Unit tests for ContinuousPromptOptimizer"""

import torch
import pytest
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.continuous import ContinuousPromptOptimizer

@pytest.fixture
def agent():
    return PromptRLAgent(model_name="EleutherAI/pythia-70m")

@pytest.fixture
def optimizer(agent):
    return ContinuousPromptOptimizer(
        agent=agent,
        initial_prompt_length=5,
        max_prompt_len=10,
        batch_size=2,
        lr_embeddings=0.01
    )

def test_continuous_initialization(optimizer):
    """Test continuous optimizer initializes correctly"""
    assert optimizer.batch_size == 2
    assert optimizer.initial_prompt_length == 5
    assert optimizer.max_prompt_len == 10
    assert optimizer.prompt_embeds.shape == (2, 10, optimizer.D)

def test_initialize_prompts(optimizer):
    """Test prompt initialization"""
    prompt_data, lengths = optimizer.initialize_prompts()
    
    assert prompt_data.shape[0] == 2
    assert lengths.shape == (2,)
    assert torch.all(lengths == 5)

def test_get_likelihoods(optimizer, agent):
    """Test likelihood computation"""
    prompt_data, lengths = optimizer.initialize_prompts()
    
    completion_texts = ["test", "completion"]
    completion_tokens_list = [agent.tokenizer.encode(t, add_special_tokens=False) for t in completion_texts]
    max_comp = max(len(ct) for ct in completion_tokens_list)
    pad_id = getattr(agent.tokenizer, 'pad_token_id', 0)
    completion_tokens = torch.tensor([
        ct + [pad_id] * (max_comp - len(ct)) for ct in completion_tokens_list
    ], dtype=torch.long, device=agent.device)
    completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=agent.device)
    
    likelihoods = optimizer.get_likelihoods(prompt_data, lengths, completion_tokens, completion_lengths)
    
    assert likelihoods.shape == (2,)
    assert torch.all(torch.isfinite(likelihoods))

def test_apply_length_action(optimizer):
    """Test length action application"""
    prompt_data, lengths = optimizer.initialize_prompts()
    actions = torch.tensor([0, 2], device=optimizer.device)  # remove, add
    
    updated_data, updated_lengths = optimizer.apply_length_action(prompt_data, lengths, actions)
    
    assert updated_lengths[0].item() == 4  # removed one
    assert updated_lengths[1].item() == 6  # added one

def test_to_tokens(optimizer):
    """Test embedding to token conversion"""
    prompt_data, lengths = optimizer.initialize_prompts()
    tokens = optimizer.to_tokens(prompt_data, lengths)
    
    assert tokens.shape[0] == 2
    assert tokens.dtype == torch.long

def test_clone_prompt(optimizer):
    """Test prompt cloning"""
    prompt_data, lengths = optimizer.initialize_prompts()
    cloned = optimizer.clone_prompt(prompt_data, idx=0, length=5)
    
    assert cloned.shape == (5, optimizer.D)
    assert not cloned.requires_grad  # Should be detached

