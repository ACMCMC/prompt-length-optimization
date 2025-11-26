"""
Dataset utilities wrapper for compatibility with the master branch layout.

The actual implementation lives in ``dataset_utils.py``; this module simply
re-exports the same class and helpers so imports under ``prompt_optimization``
continue to work.
"""

from dataset_utils import ToxicChatDatasetManager, print_dataset_info

__all__ = ["ToxicChatDatasetManager", "print_dataset_info"]
