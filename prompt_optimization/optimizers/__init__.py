"""
Optimizer stubs for compatibility with the master branch.

The GCG branch centralizes optimization inside ``LengthPolicyOptimizer``. These
classes intentionally point callers to that implementation.
"""

from prompt_optimization.optimizers.continuous import ContinuousPromptOptimizer
from prompt_optimization.optimizers.continuous_proj import ContinuousPromptOptimizerWithProjection
from prompt_optimization.optimizers.discrete import DiscretePromptOptimizer

__all__ = [
    "ContinuousPromptOptimizer",
    "ContinuousPromptOptimizerWithProjection",
    "DiscretePromptOptimizer",
]
