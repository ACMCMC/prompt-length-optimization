"""Unit tests for reward function calculation."""

import pytest
import torch


def compute_reward(likelihoods, lengths, alpha=1.0, beta=1.0):
    """
    Compute reward: reward = alpha * likelihood - beta * length
    
    Args:
        likelihoods: Log probabilities (negative values, less negative = better)
        lengths: Prompt lengths (positive values, shorter = better)
        alpha: Weight for likelihood term
        beta: Weight for length penalty term
    
    Returns:
        rewards: Computed rewards (negative values, less negative = better)
    """
    if isinstance(likelihoods, (int, float)):
        likelihoods = torch.tensor([likelihoods])
    if isinstance(lengths, (int, float)):
        lengths = torch.tensor([lengths])
    
    return alpha * likelihoods - beta * lengths.float()


class TestRewardFunction:
    """Test suite for reward function."""
    
    def test_basic_formula(self):
        """Test basic reward formula calculation."""
        likelihood = torch.tensor([-50.0])
        length = torch.tensor([20])
        alpha = 1.0
        beta = 1.0
        
        reward = compute_reward(likelihood, length, alpha, beta)
        expected = 1.0 * (-50.0) - 1.0 * 20.0
        assert reward.item() == pytest.approx(expected)
    
    def test_shorter_length_gives_higher_reward_same_likelihood(self):
        """Test that shorter lengths give higher rewards for the same likelihood."""
        likelihood = torch.tensor([-50.0])
        short_length = torch.tensor([20])
        long_length = torch.tensor([60])
        alpha = 1.0
        beta = 5.0
        
        reward_short = compute_reward(likelihood, short_length, alpha, beta)
        reward_long = compute_reward(likelihood, long_length, alpha, beta)
        
        # Shorter length should give higher (less negative) reward
        assert reward_short.item() > reward_long.item(), \
            f"Short length reward ({reward_short.item()}) should be > long length reward ({reward_long.item()})"
    
    def test_higher_likelihood_gives_higher_reward_same_length(self):
        """Test that higher (less negative) likelihoods give higher rewards for the same length."""
        good_likelihood = torch.tensor([-50.0])
        bad_likelihood = torch.tensor([-80.0])
        length = torch.tensor([30])
        alpha = 1.0
        beta = 5.0
        
        reward_good = compute_reward(good_likelihood, length, alpha, beta)
        reward_bad = compute_reward(bad_likelihood, length, alpha, beta)
        
        # Higher likelihood should give higher (less negative) reward
        assert reward_good.item() > reward_bad.item(), \
            f"Good likelihood reward ({reward_good.item()}) should be > bad likelihood reward ({reward_bad.item()})"
    
    def test_best_case_vs_worst_case(self):
        """Test that best case (good likelihood + short length) beats worst case."""
        good_likelihood = torch.tensor([-50.0])
        bad_likelihood = torch.tensor([-80.0])
        short_length = torch.tensor([20])
        long_length = torch.tensor([60])
        alpha = 1.0
        beta = 5.0
        
        reward_best = compute_reward(good_likelihood, short_length, alpha, beta)
        reward_worst = compute_reward(bad_likelihood, long_length, alpha, beta)
        
        assert reward_best.item() > reward_worst.item(), \
            f"Best case reward ({reward_best.item()}) should be > worst case reward ({reward_worst.item()})"
    
    def test_alpha_beta_scaling(self):
        """Test that alpha and beta correctly scale their respective terms."""
        likelihood = torch.tensor([-50.0])
        length = torch.tensor([20])
        
        # Test with different alpha values
        reward_alpha1 = compute_reward(likelihood, length, alpha=1.0, beta=1.0)
        reward_alpha2 = compute_reward(likelihood, length, alpha=2.0, beta=1.0)
        
        # Higher alpha makes likelihood term more important
        # Since likelihood is negative, higher alpha makes reward more negative (lower)
        assert reward_alpha2.item() < reward_alpha1.item(), \
            "Higher alpha should decrease reward (likelihood is negative, so scaling makes it more negative)"
        
        # Test with different beta values
        reward_beta1 = compute_reward(likelihood, length, alpha=1.0, beta=1.0)
        reward_beta5 = compute_reward(likelihood, length, alpha=1.0, beta=5.0)
        
        # Higher beta should make length penalty more important (lower reward)
        assert reward_beta1.item() > reward_beta5.item(), \
            "Higher beta should decrease reward (length penalty becomes more negative)"
    
    def test_edge_case_zero_length(self):
        """Test edge case with zero length."""
        likelihood = torch.tensor([-50.0])
        zero_length = torch.tensor([0])
        normal_length = torch.tensor([20])
        alpha = 1.0
        beta = 5.0
        
        reward_zero = compute_reward(likelihood, zero_length, alpha, beta)
        reward_normal = compute_reward(likelihood, normal_length, alpha, beta)
        
        # Zero length should give highest reward
        assert reward_zero.item() > reward_normal.item(), \
            "Zero length should give highest reward"
    
    def test_edge_case_very_negative_likelihood(self):
        """Test edge case with very negative likelihood."""
        very_bad_likelihood = torch.tensor([-200.0])
        normal_likelihood = torch.tensor([-50.0])
        length = torch.tensor([30])
        alpha = 1.0
        beta = 5.0
        
        reward_very_bad = compute_reward(very_bad_likelihood, length, alpha, beta)
        reward_normal = compute_reward(normal_likelihood, length, alpha, beta)
        
        # Normal likelihood should give higher reward
        assert reward_normal.item() > reward_very_bad.item(), \
            "Normal likelihood should give higher reward than very negative likelihood"
    
    def test_batch_processing(self):
        """Test reward computation on batches of values."""
        likelihoods = torch.tensor([-50.0, -80.0, -60.0])
        lengths = torch.tensor([20, 60, 40])
        alpha = 1.0
        beta = 5.0
        
        rewards = compute_reward(likelihoods, lengths, alpha, beta)
        
        assert rewards.shape == (3,), "Rewards should have shape (3,)"
        
        # First should be best (good likelihood + short length)
        # Second should be worst (bad likelihood + long length)
        assert rewards[0].item() > rewards[1].item(), \
            "Best case should have higher reward than worst case"
        assert rewards[0].item() > rewards[2].item(), \
            "Best case should have higher reward than middle case"
        assert rewards[2].item() > rewards[1].item(), \
            "Middle case should have higher reward than worst case"
    
    def test_reward_always_negative(self):
        """Test that rewards are always negative (since both terms are negative)."""
        likelihoods = torch.tensor([-50.0, -80.0, -100.0])
        lengths = torch.tensor([10, 30, 50])
        alpha = 1.0
        beta = 1.0
        
        rewards = compute_reward(likelihoods, lengths, alpha, beta)
        
        # All rewards should be negative
        assert torch.all(rewards < 0), "All rewards should be negative"
    
    def test_length_dominance_with_high_beta(self):
        """Test that high beta makes length the dominant factor."""
        likelihood = torch.tensor([-50.0])  # Good likelihood
        short_length = torch.tensor([20])
        long_length = torch.tensor([60])
        
        # With very high beta, length should dominate
        beta_high = 100.0
        reward_short_high_beta = compute_reward(likelihood, short_length, alpha=1.0, beta=beta_high)
        reward_long_high_beta = compute_reward(likelihood, long_length, alpha=1.0, beta=beta_high)
        
        # Even with good likelihood, short length should win
        assert reward_short_high_beta.item() > reward_long_high_beta.item(), \
            "With high beta, short length should give higher reward even with same likelihood"
    
    def test_likelihood_dominance_with_high_alpha(self):
        """Test that high alpha makes likelihood the dominant factor."""
        good_likelihood = torch.tensor([-50.0])
        bad_likelihood = torch.tensor([-80.0])
        length = torch.tensor([40])  # Same length
        
        # With very high alpha, likelihood should dominate
        alpha_high = 100.0
        reward_good_high_alpha = compute_reward(good_likelihood, length, alpha=alpha_high, beta=1.0)
        reward_bad_high_alpha = compute_reward(bad_likelihood, length, alpha=alpha_high, beta=1.0)
        
        # Even with same length, good likelihood should win
        assert reward_good_high_alpha.item() > reward_bad_high_alpha.item(), \
            "With high alpha, good likelihood should give higher reward even with same length"

