"""Unit tests for PromptRLAgent"""

import torch
import torch.nn.functional as F
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


def _contiguous_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Construct per-example contiguous position ids based on active tokens."""
    cumsum = attention_mask.long().cumsum(dim=1) - 1
    cumsum = torch.clamp(cumsum, min=0)
    return torch.where(attention_mask.bool(), cumsum, torch.zeros_like(cumsum))


def _manual_likelihood_with_compact_positions(agent: PromptRLAgent, model_input: ModelBatchedInput) -> torch.Tensor:
    input_ids, attention_mask = model_input.get_model_input_ids_and_attention_mask()
    position_ids = _contiguous_position_ids(attention_mask)
    outputs = agent.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    logits = outputs.logits
    completion_start = model_input.get_completion_start_pos()
    comp_tokens = model_input.completion_input_ids
    comp_mask = model_input.completion_attention_mask.bool()

    token_log_probs = logits.new_zeros(comp_tokens.shape)
    for idx in range(logits.size(0)):
        length = comp_mask[idx].sum().item()
        if length == 0:
            continue
        start = completion_start[idx].item() - 1
        end = start + length
        comp_logits = logits[idx, start:end, :]
        log_probs = F.log_softmax(comp_logits, dim=-1)
        tokens = comp_tokens[idx][comp_mask[idx]]
        gathered = log_probs.gather(1, tokens.unsqueeze(-1)).squeeze(-1)
        token_log_probs[idx, :length] = gathered
    masked_log_probs = torch.where(comp_mask, token_log_probs, torch.zeros_like(token_log_probs))
    return masked_log_probs.sum(dim=-1)


def test_likelihoods_use_compact_positions(agent):
    """Ensure likelihood computation matches contiguous position encoding."""
    prefixes = ["Question: 1+1?\nAnswer:", "Compute 3+4."]
    completions = [" 2", " 7"]
    model_input = ModelBatchedInput(
        prefix_texts=prefixes,
        completion_texts=completions,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=8,
        init_len=2,
        mode="discrete",
    )

    actual = agent.get_likelihoods_batch(model_input, requires_grad=False)
    manual = _manual_likelihood_with_compact_positions(agent, model_input)

    assert actual.shape == manual.shape
    assert torch.allclose(actual, manual, atol=1e-6)

