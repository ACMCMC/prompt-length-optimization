"""
Continuous prompt optimizer stub for compatibility.

The GCG branch handles prompt optimization inside ``LengthPolicyOptimizer`` via
its continuous mode, so this class simply surfaces a clear error message to
use that path instead of the old modular optimizer.
"""


class ContinuousPromptOptimizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ContinuousPromptOptimizer is not used in the GCG branch. "
            "Use LengthPolicyOptimizer with optimization_mode='continuous' instead."
        )


__all__ = ["ContinuousPromptOptimizer"]
