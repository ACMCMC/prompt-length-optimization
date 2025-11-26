"""
Continuous optimizer with projection stub for compatibility.

The projection-regularized path is implemented inside ``LengthPolicyOptimizer``
in this branch; this placeholder preserves API shape without duplicating logic.
"""


class ContinuousPromptOptimizerWithProjection:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ContinuousPromptOptimizerWithProjection is not used in the GCG branch. "
            "Use LengthPolicyOptimizer with the appropriate reward/config settings."
        )


__all__ = ["ContinuousPromptOptimizerWithProjection"]
