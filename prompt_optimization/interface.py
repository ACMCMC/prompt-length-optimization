"""
Abstract interface for prompt optimization methods.
This allows plug-and-play implementations for continuous vs discrete optimization.
"""

from abc import ABC, abstractmethod
import torch
from typing import Tuple
from .model_inputs import ModelBatchedInput


class BasePromptOptimizer(ABC):
    """Abstract base class for prompt optimization methods."""

    def __init__(
        self,
        agent,
        initial_prompt_length: int,
        max_prompt_len: int,
        batch_size: int,
        lr_embeddings: float,
        max_suffix_len: int,
        init_len: int,
    ):
        """
        Args:
            agent: PromptRLAgent instance
            initial_prompt_length: Starting length for prompts
            max_prompt_len: Maximum allowed prompt length (for suffix, should equal max_suffix_len)
            batch_size: Number of prompts to optimize in parallel
            lr_embeddings: Learning rate for embedding optimization (if applicable)
            max_suffix_len: Maximum suffix length from config (fixed size for batched suffix embeddings)
            init_len: Initial number of suffix positions with attention mask = 1 from config
        """
        self.agent = agent
        self.initial_prompt_length = initial_prompt_length
        self.max_prompt_len = max_prompt_len
        self.max_suffix_len = max_suffix_len
        self.init_len = init_len
        self.batch_size = batch_size
        self.lr_embeddings = lr_embeddings
        self.device = agent.device
        self.emb_dim = agent.model.get_input_embeddings().weight.shape[1]

    @abstractmethod
    def initialize_prompts(
        self, model_input: "ModelBatchedInput"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize prompts for batch optimization using ModelBatchedInput.

        Args:
            model_input: ModelBatchedInput instance for BOS initialization

        Returns:
            prompt_data: The prompt representation (embeddings or tokens)
            lengths: Current lengths tensor [B]
        """
        pass

    @abstractmethod
    def get_likelihoods(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        model_input: ModelBatchedInput,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Compute likelihoods for current prompts.

        Args:
            prompt_data: Current prompt representation
            lengths: Current lengths [B]
            model_input: ModelBatchedInput instance with all inputs
            requires_grad: Whether gradients are needed

        Returns:
            likelihoods: [B] tensor of log likelihoods
        """
        pass

    @abstractmethod
    def apply_length_action(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply length modification actions (add/remove tokens).

        Args:
            prompt_data: Current prompt representation
            lengths: Current lengths [B]
            actions: Action tensor [B] where 0=remove, 1=keep, 2=add

        Returns:
            updated_prompt_data: Updated prompt representation
            updated_lengths: Updated lengths [B]
        """
        pass

    @abstractmethod
    def inner_optimization_step(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        step: int,
        model_input: ModelBatchedInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one step of inner optimization (e.g., gradient updates, GCG replacements).
        This is called before policy actions are applied.

        Args:
            prompt_data: Current prompt representation
            lengths: Current lengths [B]
            step: Current step number in episode
            model_input: ModelBatchedInput instance with all inputs

        Returns:
            updated_prompt_data: Updated prompt representation
            updated_likelihoods: Current likelihoods after optimization [B]
        """
        pass

    @abstractmethod
    def to_tokens(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert prompt representation to token IDs for final output.

        Args:
            prompt_data: Prompt representation
            lengths: Current lengths [B]

        Returns:
            tokens: Token IDs [B, max_len] (padded to max_len)
        """
        pass

    @abstractmethod
    def clone_prompt(
        self, prompt_data: torch.Tensor, idx: int, length: int
    ) -> torch.Tensor:
        """
        Clone a single prompt from the batch (for saving best prompts).

        Args:
            prompt_data: Batch prompt representation
            idx: Index in batch
            length: Length of this prompt

        Returns:
            cloned_prompt: Cloned prompt representation
        """
        pass
