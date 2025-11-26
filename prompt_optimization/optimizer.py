"""
Thin wrapper around the GCG LengthPolicyOptimizer implementation.
Provided for API parity with the master branch.
"""

from prompt_rl_poc import LengthPolicyOptimizer as _LengthPolicyOptimizer


class LengthPolicyOptimizer(_LengthPolicyOptimizer):
    """Alias for backward compatibility."""

    pass


__all__ = ["LengthPolicyOptimizer"]
