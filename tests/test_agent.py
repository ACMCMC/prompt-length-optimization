"""Unit tests for PromptRLAgent"""

import torch
import pytest
from prompt_optimization.agent import PromptRLAgent

@pytest.fixture
def agent():
    """Create a test agent with small model"""
    return PromptRLAgent(model_name="EleutherAI/pythia-70m")

def test_agent_initialization(agent):
    """Test agent initializes correctly"""
    assert agent.device is not None
    assert agent.vocab_size > 0
    assert agent.model is not None
    assert agent.tokenizer is not None

def test_get_random_token(agent):
    """Test random token generation"""
    token = agent.get_random_token()
    assert isinstance(token, int)
    assert 0 <= token < agent.vocab_size
    assert token not in agent.special_token_ids

def test_get_likelihoods_batch(agent):
    """Test batched likelihood computation"""
    D = agent.model.get_input_embeddings().weight.shape[1]
    B, L = 2, 5
    prompt_embeds = torch.randn(B, L, D, device=agent.device)
    
    # Create simple completion tokens
    completion_texts = ["hello", "world"]
    completion_tokens_list = [agent.tokenizer.encode(t, add_special_tokens=False) for t in completion_texts]
    max_comp = max(len(ct) for ct in completion_tokens_list)
    pad_id = getattr(agent.tokenizer, 'pad_token_id', 0)
    completion_tokens = torch.tensor([
        ct + [pad_id] * (max_comp - len(ct)) for ct in completion_tokens_list
    ], dtype=torch.long, device=agent.device)
    completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=agent.device)
    
    likelihoods = agent.get_likelihoods_batch(prompt_embeds, completion_tokens, completion_lengths, requires_grad=False)
    
    assert likelihoods.shape == (B,)
    assert torch.all(torch.isfinite(likelihoods))

def test_get_likelihoods_batch_empty_completion(agent):
    """Test likelihood computation with empty completion"""
    D = agent.model.get_input_embeddings().weight.shape[1]
    B, L = 1, 3
    prompt_embeds = torch.randn(B, L, D, device=agent.device)
    completion_tokens = torch.zeros(B, 1, dtype=torch.long, device=agent.device)
    completion_lengths = torch.zeros(B, dtype=torch.long, device=agent.device)
    
    likelihoods = agent.get_likelihoods_batch(prompt_embeds, completion_tokens, completion_lengths, requires_grad=False)
    
    assert likelihoods.shape == (B,)
    assert likelihoods[0].item() == 0.0

