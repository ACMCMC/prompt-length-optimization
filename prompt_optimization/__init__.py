"""
Compatibility shims to make the GCG branch look like the master package layout.

We reuse the core implementations living in ``prompt_rl_poc.py`` so downstream
code can continue importing ``prompt_optimization.PromptRLAgent`` and
``prompt_optimization.LengthPolicyOptimizer`` without changing behavior.
"""

from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer
from prompt_optimization.datasets import ToxicChatDatasetManager  # re-export for parity

__all__ = [
    "PromptRLAgent",
    "LengthPolicyOptimizer",
    "ToxicChatDatasetManager",
]
