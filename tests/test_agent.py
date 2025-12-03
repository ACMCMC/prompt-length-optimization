"""Unit tests for PromptRLAgent"""

import torch
import pytest
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.model_inputs import ModelBatchedInput

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
    prefixes = ["Question: 1+1?\nAnswer:", "Prompt: say hello\nOutput:"]
    completions = [" 2", " hello"]
    model_input = ModelBatchedInput(
        prefix_texts=prefixes,
        completion_texts=completions,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=0,
        init_len=0,
        mode="discrete",
    )

    likelihoods = agent.get_likelihoods_batch(model_input, requires_grad=False)

    assert likelihoods.shape == (len(prefixes),)
    assert torch.all(torch.isfinite(likelihoods))

def test_get_likelihoods_batch_empty_completion(agent):
    """Test likelihood computation with empty completion"""
    prefixes = ["Q: value?\nA:", "Another question:"]
    completions = ["", ""]
    model_input = ModelBatchedInput(
        prefix_texts=prefixes,
        completion_texts=completions,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=0,
        init_len=0,
        mode="discrete",
    )

    likelihoods = agent.get_likelihoods_batch(model_input, requires_grad=False)

    assert likelihoods.shape == (len(prefixes),)
    assert torch.allclose(likelihoods, torch.zeros_like(likelihoods))


