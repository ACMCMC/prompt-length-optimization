"""Unit tests for BasePromptOptimizer interface"""

import torch
import pytest
from prompt_optimization.interface import BasePromptOptimizer
from prompt_optimization.agent import PromptRLAgent

class MockOptimizer(BasePromptOptimizer):
    """Mock implementation for testing interface"""
    
    def initialize_prompts(self):
        return torch.randn(self.batch_size, self.max_prompt_len, self.emb_dim, device=self.device), \
               torch.full((self.batch_size,), self.initial_prompt_length, dtype=torch.long, device=self.device)
    
    def get_likelihoods(self, prompt_data, lengths, completion_tokens, completion_lengths, requires_grad=False):
        return torch.zeros(self.batch_size, device=self.device)
    
    def apply_length_action(self, prompt_data, lengths, actions):
        return prompt_data, lengths
    
    def inner_optimization_step(self, prompt_data, lengths, completion_tokens, completion_lengths, step):
        return prompt_data, torch.zeros(self.batch_size, device=self.device)
    
    def to_tokens(self, prompt_data, lengths):
        return torch.zeros(self.batch_size, lengths.max().item(), dtype=torch.long, device=self.device)
    
    def clone_prompt(self, prompt_data, idx, length):
        return prompt_data[idx, :length].clone()

@pytest.fixture
def agent():
    return PromptRLAgent(model_name="EleutherAI/pythia-70m")

@pytest.fixture
def mock_optimizer(agent):
    return MockOptimizer(agent, initial_prompt_length=5, max_prompt_len=10, batch_size=2, lr_embeddings=0.01)

def test_interface_initialization(mock_optimizer):
    """Test interface initializes correctly"""
    assert mock_optimizer.batch_size == 2
    assert mock_optimizer.initial_prompt_length == 5
    assert mock_optimizer.max_prompt_len == 10
    assert mock_optimizer.device == mock_optimizer.agent.device

def test_interface_methods_exist(mock_optimizer):
    """Test all required methods exist"""
    assert hasattr(mock_optimizer, 'initialize_prompts')
    assert hasattr(mock_optimizer, 'get_likelihoods')
    assert hasattr(mock_optimizer, 'apply_length_action')
    assert hasattr(mock_optimizer, 'inner_optimization_step')
    assert hasattr(mock_optimizer, 'to_tokens')
    assert hasattr(mock_optimizer, 'clone_prompt')

def test_initialize_prompts_returns_correct_shape(mock_optimizer):
    """Test initialize_prompts returns correct shapes"""
    prompt_data, lengths = mock_optimizer.initialize_prompts()
    assert prompt_data.shape[0] == 2
    assert lengths.shape == (2,)

