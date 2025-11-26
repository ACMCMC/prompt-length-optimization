"""
Discrete prompt optimizer stub for compatibility.

The GCG branch implements discrete/token-space optimization directly inside
``LengthPolicyOptimizer`` via its GCG routines. This stub exists to keep import
paths aligned with master while guiding callers to the supported API.
"""


class DiscretePromptOptimizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "DiscretePromptOptimizer is handled by LengthPolicyOptimizer "
            "with optimization_mode='discrete' in this branch."
        )


__all__ = ["DiscretePromptOptimizer"]
