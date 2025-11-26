"""
Compatibility stub matching the master branch's interface module.

The GCG branch folds optimization logic into ``LengthPolicyOptimizer`` and the
underlying prompt optimizers; this stub preserves import compatibility without
changing behavior.
"""

from abc import ABC, abstractmethod


class BasePromptOptimizer(ABC):
    """Abstract base retained for compatibility; not used in the GCG branch."""

    @abstractmethod
    def initialize_prompts(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")

    @abstractmethod
    def get_likelihoods(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")

    @abstractmethod
    def apply_length_action(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")

    @abstractmethod
    def inner_optimization_step(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")

    @abstractmethod
    def to_tokens(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")

    @abstractmethod
    def clone_prompt(self, *args, **kwargs):
        raise NotImplementedError("Use LengthPolicyOptimizer in prompt_rl_poc.py")


__all__ = ["BasePromptOptimizer"]
