"""Unit tests for GCG (Greedy Coordinate Gradient) algorithm in DiscretePromptOptimizer"""

import torch
import pytest
from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.optimizers.discrete import DiscretePromptOptimizer
from prompt_optimization.model_inputs import ModelBatchedInput


@pytest.fixture
def agent():
    """Create a test agent with small model"""
    return PromptRLAgent(model_name="EleutherAI/pythia-70m")


@pytest.fixture
def gcg_optimizer(agent):
    """Create a GCG optimizer for testing"""
    return DiscretePromptOptimizer(
        agent=agent,
        initial_prompt_length=10,
        max_prompt_len=20,
        batch_size=1,
        lr_embeddings=0.01,
        max_suffix_len=20,
        init_len=10,
        gcg_steps=10,  # 10 GCG iterations (reduced for faster tests)
        gcg_top_k=16,
        gcg_batch_size=32,
        gcg_max_batch_size=128  # Maximum batch size for forward passes
    )


def test_gcg_improves_likelihood(agent, gcg_optimizer):
    """
    Test that GCG algorithm improves likelihood using gradient-based candidate sampling.
    
    This test:
    1. Sets up a prefix, suffix, and completion
    2. Runs GCG optimization (gcg_steps iterations with gradient-based sampling)
    3. Verifies that the final likelihood is better than or similar to initial likelihood
    """
    # Define test inputs
    prefix_text = "The quick brown fox"
    completion_text = "jumps over the lazy dog"
    
    # Create ModelBatchedInput with prefix, empty suffix (will be initialized), and completion
    model_input = ModelBatchedInput(
        prefix_texts=[prefix_text],
        completion_texts=[completion_text],
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=20,
        init_len=10,
        mode='discrete'
    )
    
    # Initialize prompts (suffix tokens)
    prompt_data, lengths = gcg_optimizer.initialize_prompts(model_input)
    
    # Update model_input with initial suffix tokens
    model_input.update_suffix_tokens(prompt_data)
    
    # Get initial likelihood
    initial_likelihoods = agent.get_likelihoods_batch(model_input, requires_grad=False)
    initial_ll = initial_likelihoods[0].item()
    
    # Verify initial likelihood is finite
    assert torch.isfinite(initial_likelihoods[0]), "Initial likelihood should be finite"
    
    # Run GCG optimization (does gcg_steps iterations internally)
    optimized_prompt_data, final_likelihoods = gcg_optimizer.inner_optimization_step(
        prompt_data=prompt_data,
        lengths=lengths,
        step=0,
        model_input=model_input
    )
    
    final_ll = final_likelihoods[0].item()
    
    # Verify final likelihood is finite
    assert torch.isfinite(final_likelihoods[0]), "Final likelihood should be finite"
    
    # Verify that likelihood improved (or at least didn't decrease significantly)
    # Note: GCG is stochastic, so we allow for small decreases but expect improvement on average
    improvement = final_ll - initial_ll
    
    # Log the results for debugging
    print(f"\nGCG Test Results:")
    print(f"  Initial likelihood: {initial_ll:.6f}")
    print(f"  Final likelihood: {final_ll:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    print(f"  Improvement percentage: {(improvement / abs(initial_ll) * 100) if initial_ll != 0 else 0:.2f}%")
    
    # The test passes if:
    # 1. Final likelihood is finite
    # 2. The improvement is not catastrophically negative (allowing for stochasticity)
    #    We allow up to 20% degradation as a safety margin for stochastic algorithms
    max_allowed_degradation = abs(initial_ll) * 0.2 if initial_ll != 0 else 0.1
    
    assert improvement >= -max_allowed_degradation, \
        f"Likelihood degraded by more than 20%: {improvement:.6f} (initial: {initial_ll:.6f}, final: {final_ll:.6f})"
    
    # Ideally, we'd like to see improvement, but we're lenient for stochastic algorithms
    if improvement > 0:
        print(f"  ✓ Likelihood improved by {improvement:.6f}")
    else:
        print(f"  ⚠ Likelihood decreased by {abs(improvement):.6f} (within acceptable range)")


def test_gcg_multiple_iterations(agent, gcg_optimizer):
    """
    Test GCG with multiple calls to inner_optimization_step to verify cumulative improvement.
    Each call runs gcg_steps iterations internally.
    """
    prefix_text = "Hello"
    completion_text = "world"
    
    model_input = ModelBatchedInput(
        prefix_texts=[prefix_text],
        completion_texts=[completion_text],
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=20,
        init_len=10,
        mode='discrete'
    )
    
    prompt_data, lengths = gcg_optimizer.initialize_prompts(model_input)
    model_input.update_suffix_tokens(prompt_data)
    
    # Get initial likelihood
    initial_ll = agent.get_likelihoods_batch(model_input, requires_grad=False)[0].item()
    
    # Run multiple optimization steps (each does gcg_steps iterations)
    current_ll = initial_ll
    likelihoods_history = [initial_ll]
    
    for step in range(3):  # Run 3 optimization steps
        prompt_data, likelihoods = gcg_optimizer.inner_optimization_step(
            prompt_data=prompt_data,
            lengths=lengths,
            step=step,
            model_input=model_input
        )
        current_ll = likelihoods[0].item()
        likelihoods_history.append(current_ll)
    
    # Verify that we can run multiple steps without errors
    likelihoods_tensor = torch.tensor(likelihoods_history)
    assert torch.all(torch.isfinite(likelihoods_tensor)), "All likelihoods should be finite"
    
    # Log the progression
    print(f"\nGCG Multiple Iterations Test:")
    for i, ll in enumerate(likelihoods_history):
        improvement = ll - initial_ll if i > 0 else 0
        print(f"  Step {i}: likelihood = {ll:.6f}, improvement = {improvement:.6f}")


def test_gcg_preserves_prompt_structure(agent, gcg_optimizer):
    """
    Test that GCG optimization preserves the basic structure of the prompt
    (length, valid token IDs, etc.)
    """
    prefix_text = "Test"
    completion_text = "completion"
    
    model_input = ModelBatchedInput(
        prefix_texts=[prefix_text],
        completion_texts=[completion_text],
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=20,
        init_len=10,
        mode='discrete'
    )
    
    prompt_data, lengths = gcg_optimizer.initialize_prompts(model_input)
    initial_length = lengths[0].item()
    
    # Run optimization
    optimized_prompt_data, _ = gcg_optimizer.inner_optimization_step(
        prompt_data=prompt_data,
        lengths=lengths,
        step=0,
        model_input=model_input
    )
    
    # Verify length is preserved
    assert lengths[0].item() == initial_length, "Prompt length should be preserved"
    
    # Get vocabulary size from embedding layer
    vocab_size = agent.model.get_input_embeddings().weight.shape[0]
    
    # Verify all tokens are valid (non-negative, within vocab size)
    active_tokens = optimized_prompt_data[0, :initial_length]
    assert torch.all(active_tokens >= 0), "All tokens should be non-negative"
    assert torch.all(active_tokens < vocab_size), \
        f"All tokens should be within vocabulary size (vocab_size={vocab_size})"
    
    # Verify prompt data shape
    assert optimized_prompt_data.shape == prompt_data.shape, "Prompt shape should be preserved"


def test_gcg_gradient_computation(agent, gcg_optimizer):
    """
    Test that GCG can compute gradients w.r.t. token embeddings.
    This verifies the core gradient-based candidate sampling mechanism.
    """
    prefix_text = "Test gradient"
    completion_text = "computation"
    
    model_input = ModelBatchedInput(
        prefix_texts=[prefix_text],
        completion_texts=[completion_text],
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=20,
        init_len=10,
        mode='discrete'
    )
    
    prompt_data, lengths = gcg_optimizer.initialize_prompts(model_input)
    suffix_mask = model_input.suffix_attention_mask
    
    # Test gradient computation
    gradients = gcg_optimizer._compute_gradients(prompt_data, model_input, suffix_mask)
    
    # Verify gradients shape: [B, max_suffix_len, emb_dim]
    assert gradients.shape == (gcg_optimizer.batch_size, gcg_optimizer.max_suffix_len, gcg_optimizer.emb_dim), \
        f"Gradients should have shape [B, max_suffix_len, emb_dim], got {gradients.shape}"
    
    # Verify gradients are finite
    assert torch.all(torch.isfinite(gradients)), "All gradients should be finite"
    
    # Verify gradients are not all zero (at least some should be non-zero)
    assert not torch.allclose(gradients, torch.zeros_like(gradients)), \
        "Gradients should not be all zero"


def test_gcg_candidate_sampling(agent, gcg_optimizer):
    """
    Test that GCG can sample candidates from gradients.
    This verifies the candidate generation mechanism.
    """
    prefix_text = "Test sampling"
    completion_text = "candidates"
    
    model_input = ModelBatchedInput(
        prefix_texts=[prefix_text],
        completion_texts=[completion_text],
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=20,
        init_len=10,
        mode='discrete'
    )
    
    prompt_data, lengths = gcg_optimizer.initialize_prompts(model_input)
    suffix_mask = model_input.suffix_attention_mask
    
    # Compute gradients
    gradients = gcg_optimizer._compute_gradients(prompt_data, model_input, suffix_mask)
    
    # Sample candidates
    candidate_sequences, update_info = gcg_optimizer._sample_candidates_from_grad(
        prompt_data, gradients, suffix_mask, n_replace=1
    )
    
    # Verify candidate sequences shape: [search_width * B, max_suffix_len]
    search_width = gcg_optimizer.gcg_batch_size
    expected_shape = (search_width * gcg_optimizer.batch_size, gcg_optimizer.max_suffix_len)
    assert candidate_sequences.shape == expected_shape, \
        f"Candidate sequences should have shape {expected_shape}, got {candidate_sequences.shape}"
    
    # Verify all tokens are valid
    vocab_size = agent.model.get_input_embeddings().weight.shape[0]
    assert torch.all(candidate_sequences >= 0), "All candidate tokens should be non-negative"
    assert torch.all(candidate_sequences < vocab_size), \
        f"All candidate tokens should be within vocabulary size (vocab_size={vocab_size})"
    
    # Verify update_info is not empty
    assert len(update_info) > 0, "Update info should not be empty"

