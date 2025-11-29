"""Unit tests for metrics extraction from traces."""

import pytest
import torch


def _extract_metrics_from_traces(traces_list, idx):
    """Extract metrics from traces for a specific prompt index."""
    final_ll = None
    best_ll = float('-inf')
    best_ep = None
    best_reward_val = None
    
    if not traces_list:
        return 0.0, 0.0, None, None
    
    # iterate through trace entries in order
    for t in traces_list:
        if not isinstance(t, dict):
            continue
            
        # Prefer best_likelihoods over likelihoods (best_likelihoods tracks the best so far)
        ll_val = None
        
        # Try best_likelihoods first (this is the best so far)
        if 'best_likelihoods' in t:
            best_ll_list = t['best_likelihoods']
            if isinstance(best_ll_list, (list, tuple)) and len(best_ll_list) > idx:
                try:
                    ll_val = float(best_ll_list[idx])
                    # Check for NaN/Inf
                    if isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf')):
                        ll_val = None
                except (IndexError, ValueError, TypeError):
                    ll_val = None
        
        # Fallback to current likelihoods
        if ll_val is None and 'likelihoods' in t:
            ll_list = t['likelihoods']
            if isinstance(ll_list, (list, tuple)) and len(ll_list) > idx:
                try:
                    ll_val = float(ll_list[idx])
                    # Check for NaN/Inf
                    if isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf')):
                        ll_val = None
                except (IndexError, ValueError, TypeError):
                    ll_val = None
        
        # Last fallback: single likelihood value
        if ll_val is None and 'likelihood' in t:
            try:
                ll_val = float(t['likelihood'])
                # Check for NaN/Inf
                if isinstance(ll_val, float) and (ll_val != ll_val or ll_val == float('inf') or ll_val == float('-inf')):
                    ll_val = None
            except (ValueError, TypeError):
                ll_val = None
        
        if ll_val is not None:
            # update final (last non-None value)
            final_ll = ll_val
            # update best if this is better (likelihoods are log probs, so higher is better)
            if ll_val > best_ll:
                best_ll = ll_val
                best_ep = t.get('episode', t.get('step', None))
                # Try to get best_reward from this trace entry
                if 'best_rewards' in t and isinstance(t['best_rewards'], (list, tuple)) and len(t['best_rewards']) > idx:
                    try:
                        best_reward_val = float(t['best_rewards'][idx])
                    except (IndexError, ValueError, TypeError):
                        pass
                elif 'rewards' in t and isinstance(t['rewards'], (list, tuple)) and len(t['rewards']) > idx:
                    try:
                        best_reward_val = float(t['rewards'][idx])
                    except (IndexError, ValueError, TypeError):
                        pass
    
    if final_ll is None:
        final_ll = 0.0
    if best_ll == float('-inf'):
        best_ll = final_ll
    
    return float(final_ll), float(best_ll), best_ep, best_reward_val


def test_extract_metrics_empty_traces():
    """Test extraction with empty traces."""
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces([], 0)
    assert final_ll == 0.0
    assert best_ll == 0.0
    assert best_ep is None
    assert best_reward is None


def test_extract_metrics_simple_trace():
    """Test extraction with simple trace structure."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-10.5, -20.3, -15.7],
            'best_likelihoods': [-10.5, -20.3, -15.7],
            'rewards': [1.0, 2.0, 3.0]
        },
        {
            'episode': 0,
            'step': 1,
            'likelihoods': [-9.2, -19.1, -14.5],
            'best_likelihoods': [-9.2, -19.1, -14.5],
            'rewards': [1.5, 2.5, 3.5]
        }
    ]
    
    # Test for index 0
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -9.2  # Last value
    assert best_ll == -9.2   # Best value (highest, least negative)
    assert best_ep == 0
    assert best_reward == 1.5
    
    # Test for index 1
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 1)
    assert final_ll == -19.1
    assert best_ll == -19.1
    assert best_reward == 2.5


def test_extract_metrics_best_likelihoods_priority():
    """Test that best_likelihoods takes priority over likelihoods."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-10.0, -20.0],
            'best_likelihoods': [-8.0, -18.0],  # Better than likelihoods
            'rewards': [1.0, 2.0]
        }
    ]
    
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -8.0  # Should use best_likelihoods
    assert best_ll == -8.0


def test_extract_metrics_improving_likelihoods():
    """Test extraction when likelihoods improve over time."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-50.0, -60.0],
            'best_likelihoods': [-50.0, -60.0],
            'rewards': [-10.0, -20.0]
        },
        {
            'episode': 0,
            'step': 1,
            'likelihoods': [-45.0, -55.0],
            'best_likelihoods': [-45.0, -55.0],  # Improved
            'rewards': [-5.0, -15.0]
        },
        {
            'episode': 1,
            'step': 2,
            'likelihoods': [-40.0, -50.0],
            'best_likelihoods': [-40.0, -50.0],  # Best so far
            'rewards': [0.0, -10.0]
        }
    ]
    
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -40.0  # Last value
    assert best_ll == -40.0   # Best value
    assert best_ep == 1       # Episode where best was found
    assert best_reward == 0.0


def test_extract_metrics_with_zero_likelihoods():
    """Test extraction when likelihoods are zero (should still work)."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [0.0, 0.0],
            'best_likelihoods': [0.0, 0.0],
            'rewards': [-5.0, -10.0]
        }
    ]
    
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == 0.0
    assert best_ll == 0.0
    assert best_ep == 0


def test_extract_metrics_negative_likelihoods():
    """Test extraction with negative likelihoods (log probabilities)."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-100.5, -200.3],
            'best_likelihoods': [-100.5, -200.3],
            'rewards': [-50.0, -150.0]
        }
    ]
    
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -100.5
    assert best_ll == -100.5
    assert best_reward == -50.0


def test_extract_metrics_index_out_of_bounds():
    """Test extraction when index is out of bounds."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-10.0],  # Only 1 element
            'best_likelihoods': [-10.0],
            'rewards': [1.0]
        }
    ]
    
    # Index 0 should work
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -10.0
    
    # Index 1 should return 0.0 (default)
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 1)
    assert final_ll == 0.0
    assert best_ll == 0.0


def test_extract_metrics_missing_keys():
    """Test extraction when some keys are missing."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-10.0, -20.0],
            # Missing best_likelihoods
            'rewards': [1.0, 2.0]
        }
    ]
    
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -10.0
    assert best_ll == -10.0


def test_extract_metrics_nan_inf_handling():
    """Test that NaN and Inf values are handled correctly."""
    import math
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-10.0, float('nan'), float('inf')],
            'best_likelihoods': [-10.0, float('nan'), float('inf')],
            'rewards': [1.0, 2.0, 3.0]
        }
    ]
    
    # Index 0: valid value
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -10.0
    
    # Index 1: NaN should be skipped, return 0.0
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 1)
    assert final_ll == 0.0
    
    # Index 2: Inf should be skipped, return 0.0
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 2)
    assert final_ll == 0.0


def test_extract_metrics_real_world_scenario():
    """Test with a realistic scenario matching the CSV data structure."""
    traces = [
        {
            'episode': 0,
            'step': 0,
            'likelihoods': [-95.5, -89.6, -124.5],
            'best_likelihoods': [-95.5, -89.6, -124.5],
            'rewards': [-64.85, -89.58, -124.50],
            'lengths': [32, 32, 32]
        },
        {
            'episode': 0,
            'step': 1,
            'likelihoods': [-90.2, -85.1, -120.3],
            'best_likelihoods': [-90.2, -85.1, -120.3],  # Improved
            'rewards': [-59.55, -84.11, -119.30],
            'lengths': [31, 31, 31]
        },
        {
            'episode': 1,
            'step': 100,
            'likelihoods': [-88.5, -82.3, -118.0],
            'best_likelihoods': [-88.5, -82.3, -118.0],  # Best so far
            'rewards': [-57.85, -81.23, -117.00],
            'lengths': [30, 30, 30]
        }
    ]
    
    # Test for prompt 0
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 0)
    assert final_ll == -88.5
    assert best_ll == -88.5
    assert best_ep == 1
    assert best_reward == -57.85
    
    # Test for prompt 1
    final_ll, best_ll, best_ep, best_reward = _extract_metrics_from_traces(traces, 1)
    assert final_ll == -82.3
    assert best_ll == -82.3
    assert best_ep == 1


