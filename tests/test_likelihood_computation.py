"""Test likelihood computation and trace storage to debug the 0.0 issue."""

import pytest
import torch
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.optimizers.continuous import ContinuousPromptOptimizer


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


def test_likelihood_computation_produces_values(optimizer, agent):
    """Test that likelihood computation actually produces non-zero values."""
    prompt_data, lengths = optimizer.initialize_prompts()
    
    # Create completion tokens
    completion_texts = ["test completion", "another test"]
    completion_tokens_list = [agent.tokenizer.encode(t, add_special_tokens=False) for t in completion_texts]
    max_comp = max(len(ct) for ct in completion_tokens_list)
    pad_id = getattr(agent.tokenizer, 'pad_token_id', 0)
    completion_tokens = torch.tensor([
        ct + [pad_id] * (max_comp - len(ct)) for ct in completion_tokens_list
    ], dtype=torch.long, device=agent.device)
    completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=agent.device)
    
    # Compute likelihoods
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, completion_tokens, completion_lengths, requires_grad=False
    )
    
    # Likelihoods should be negative (log probabilities) but not zero
    assert likelihoods.shape == (2,)
    assert torch.all(torch.isfinite(likelihoods))
    # They should be significantly negative (not close to 0)
    # For a small model with random prompts, they'll be quite negative
    assert torch.all(likelihoods < -1.0)  # Should be negative log probs
    print(f"Likelihoods: {likelihoods.tolist()}")


def test_trace_storage_format(optimizer, agent):
    """Test that traces are stored in the correct format."""
    prompt_data, lengths = optimizer.initialize_prompts()
    
    # Create completion tokens (must match batch_size=2)
    completion_texts = ["test", "another"]
    completion_tokens_list = [agent.tokenizer.encode(t, add_special_tokens=False) for t in completion_texts]
    max_comp = max(len(ct) for ct in completion_tokens_list)
    pad_id = getattr(agent.tokenizer, 'pad_token_id', 0)
    completion_tokens = torch.tensor([
        ct + [pad_id] * (max_comp - len(ct)) for ct in completion_tokens_list
    ], dtype=torch.long, device=agent.device)
    completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=agent.device)
    
    # Compute likelihoods
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, completion_tokens, completion_lengths, requires_grad=False
    )
    
    # Simulate trace storage (as done in optimizer.py)
    traces = []
    for step in range(3):
        # Simulate improving likelihoods
        step_likelihoods = likelihoods - step * 2.0  # Improving over time
        best_likelihoods = step_likelihoods  # For simplicity, same as current
        
        trace_entry = {
            'episode': 0,
            'step': step,
            'rewards': [float(r) for r in (step_likelihoods - 5.0)],  # Simple reward
            'likelihoods': [float(l) for l in step_likelihoods],
            'best_likelihoods': [float(l) for l in best_likelihoods],
            'lengths': [int(l) for l in lengths]
        }
        traces.append(trace_entry)
    
    # Verify trace structure
    assert len(traces) == 3
    for trace in traces:
        assert 'likelihoods' in trace
        assert 'best_likelihoods' in trace
        assert isinstance(trace['likelihoods'], list)
        assert isinstance(trace['best_likelihoods'], list)
        assert len(trace['likelihoods']) == 2  # batch_size=2
        assert len(trace['best_likelihoods']) == 2
        # Likelihoods should not be 0.0
        assert all(ll != 0.0 for ll in trace['likelihoods'])
        assert all(ll != 0.0 for ll in trace['best_likelihoods'])
        # Should be negative (log probabilities)
        assert all(ll < 0.0 for ll in trace['likelihoods'])
    
    # Test extraction for index 0 (use the function from test_metrics_extraction)
    from tests.test_metrics_extraction import _extract_metrics_from_traces
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    
    assert final_ll != 0.0
    assert best_ll != 0.0
    assert final_ll < 0.0  # Should be negative
    assert best_ll < 0.0
    print(f"Extracted final_ll: {final_ll}, best_ll: {best_ll}")
    
    # Test extraction for index 1
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 1)
    assert final_ll != 0.0
    assert best_ll != 0.0


def test_likelihoods_not_zero_with_bos_initialization(optimizer, agent):
    """Test that BOS initialization doesn't produce zero likelihoods."""
    prompt_data, lengths = optimizer.initialize_prompts()
    
    # Verify initialization uses BOS
    # (This is tested in other tests, but we want to ensure likelihoods work)
    
    # Must match batch_size=2
    completion_texts = ["hello world", "test completion"]
    completion_tokens_list = [agent.tokenizer.encode(t, add_special_tokens=False) for t in completion_texts]
    max_comp = max(len(ct) for ct in completion_tokens_list)
    pad_id = getattr(agent.tokenizer, 'pad_token_id', 0)
    completion_tokens = torch.tensor([
        ct + [pad_id] * (max_comp - len(ct)) for ct in completion_tokens_list
    ], dtype=torch.long, device=agent.device)
    completion_lengths = torch.tensor([len(ct) for ct in completion_tokens_list], dtype=torch.long, device=agent.device)
    
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, completion_tokens, completion_lengths, requires_grad=False
    )
    
    # Even with BOS tokens, we should get some likelihood (negative log prob)
    assert likelihoods.shape == (2,)
    assert torch.all(torch.isfinite(likelihoods))
    # Should be negative but not zero
    assert torch.all(likelihoods < 0.0)
    assert torch.all(likelihoods != 0.0)
    print(f"BOS initialization likelihoods: {likelihoods.tolist()}")

