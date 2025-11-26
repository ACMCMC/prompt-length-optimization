"""
Thin wrapper around the GCG PromptRLAgent implementation.
Provided for API parity with the master branch.
"""

from prompt_rl_poc import PromptRLAgent as _PromptRLAgent


class PromptRLAgent(_PromptRLAgent):
    """Alias for backward compatibility."""

    pass


__all__ = ["PromptRLAgent"]
