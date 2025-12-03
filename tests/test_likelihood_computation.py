"""Test likelihood computation and trace storage to debug the 0.0 issue."""

import pytest
import torch
import torch.nn.functional as F
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.optimizers.continuous import ContinuousPromptOptimizer
from prompt_optimization.model_inputs import ModelBatchedInput


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
        lr_embeddings=0.01,
        max_suffix_len=8,
        init_len=5,
    )


def _build_model_input(agent, optimizer, prefixes, completions, mode="continuous"):
    return ModelBatchedInput(
        prefix_texts=prefixes,
        completion_texts=completions,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=optimizer.max_suffix_len,
        init_len=optimizer.initial_prompt_length,
        mode=mode,
    )


def test_likelihood_computation_produces_values(optimizer, agent):
    """Test that likelihood computation actually produces non-zero values."""
    prefixes = ["Explain gravity:", "Translate to French:"]
    completions = [" Gravity pulls.", " Bonjour."]
    model_input = _build_model_input(agent, optimizer, prefixes, completions)
    prompt_data, lengths = optimizer.initialize_prompts(model_input)

    # Compute likelihoods
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, model_input, requires_grad=False
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
    prefixes = ["Prompt A:", "Prompt B:"]
    completions = [" sample completion", " another completion"]
    model_input = _build_model_input(agent, optimizer, prefixes, completions)
    prompt_data, lengths = optimizer.initialize_prompts(model_input)
    
    # Compute likelihoods
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, model_input, requires_grad=False
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
    prefixes = ["Describe BOS usage:", "Another prompt:"]
    completions = [" hello world", " test completion"]
    model_input = _build_model_input(agent, optimizer, prefixes, completions)
    prompt_data, lengths = optimizer.initialize_prompts(model_input)
    
    likelihoods = optimizer.get_likelihoods(
        prompt_data, lengths, model_input, requires_grad=False
    )
    
    # Even with BOS tokens, we should get some likelihood (negative log prob)
    assert likelihoods.shape == (2,)
    assert torch.all(torch.isfinite(likelihoods))
    # Should be negative but not zero
    assert torch.all(likelihoods < 0.0)
    assert torch.all(likelihoods != 0.0)
    print(f"BOS initialization likelihoods: {likelihoods.tolist()}")


@pytest.mark.xfail(
    reason="Second-half completion likelihoods not yet significantly higher; tracked bug"
)
def test_completion_second_half_more_probable(agent):
    """Split completion log-likelihoods and ensure latter half is at least half as negative."""
    prefixes = ["Question: 5+7=?\nAnswer:", "Explain the concept of GCG:"]
    rare_chunk = " qzxvbnm"
    completions = [
        rare_chunk * 20 + " the" * 10,
        rare_chunk * 20 + " easy" * 10,
    ]

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

    input_ids, attention_mask = model_input.get_model_input_ids_and_attention_mask()
    outputs = agent.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    token_log_probs, comp_mask = agent.compute_completion_log_probs(outputs.logits, model_input)
    completion_start = model_input.get_completion_start_pos()

    for idx in range(token_log_probs.shape[0]):
        length = comp_mask[idx].sum().item()
        if length < 4:
            continue
        start = int(completion_start[idx].item())
        slice_logits = outputs.logits[idx, start - 1 : start - 1 + length, :]
        log_probs = F.log_softmax(slice_logits, dim=-1)
        tokens = model_input.completion_input_ids[idx][comp_mask[idx]]

        print(f"\nPrompt {idx}: {prefixes[idx]}")
        for pos in range(length):
            token_id = tokens[pos].item()
            token_text = agent.tokenizer.decode([token_id]).strip().replace("\n", "\\n")
            assigned_log_prob = log_probs[pos, token_id].item()
            assigned_prob = torch.exp(log_probs[pos, token_id]).item()
            topk = log_probs[pos].topk(5)
            top_tokens = [
                (
                    agent.tokenizer.decode([topk.indices[i].item()]).strip().replace("\n", "\\n"),
                    topk.values[i].item(),
                )
                for i in range(5)
            ]
            print(
                f"  token[{pos}]='{token_text}' logp={assigned_log_prob:.4f} "
                f"p={assigned_prob:.4f} top-5={top_tokens}"
            )

        half = length // 2
        first_half = token_log_probs[idx, :half]
        second_half = token_log_probs[idx, half:length]
        mean_first = first_half.mean()
        mean_second = second_half.mean()
        assert mean_first < 0.0
        # Later tokens should be closer to 0 (less negative magnitude)
        assert mean_second >= mean_first  # no worse than first half
        assert mean_second.abs() <= 0.3 * mean_first.abs()

