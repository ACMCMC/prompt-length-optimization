"""
Continuous prompt optimization: optimizes embeddings directly, then projects to tokens.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from ..interface import BasePromptOptimizer
from ..model_inputs import ModelBatchedInput


class ContinuousPromptOptimizer(BasePromptOptimizer):
    """Optimizes prompts in continuous embedding space, then projects to tokens."""

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
        super().__init__(
            agent,
            initial_prompt_length,
            max_prompt_len,
            batch_size,
            lr_embeddings,
            max_suffix_len,
            init_len,
        )
        self.embedding_layer = agent.model.get_input_embeddings()
        self.D = self.emb_dim

        # Initialize prompt embeddings as learnable parameters (will be initialized with BOS via ModelBatchedInput)
        # Suffix size is fixed to max_suffix_len from config
        self.prompt_embeds = nn.Parameter(
            torch.zeros(
                batch_size, self.max_suffix_len, self.emb_dim, device=self.device
            )
        )
        self.prompt_optimizer = optim.Adam([self.prompt_embeds], lr=lr_embeddings)

    def initialize_prompts(
        self, model_input: ModelBatchedInput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize with BOS embeddings from ModelBatchedInput."""
        lengths = torch.full(
            (self.batch_size,),
            self.initial_prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        # Reset embeddings to BOS token embedding from ModelBatchedInput
        with torch.no_grad():
            self.prompt_embeds.data = model_input.initialize_suffix_embeddings()
        return self.prompt_embeds, lengths

    def get_likelihoods(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        model_input: ModelBatchedInput,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Compute likelihoods from embeddings using ModelBatchedInput."""
        # Update model_input with current suffix embeddings
        model_input.update_suffix_embeddings(prompt_data)
        return self.agent.get_likelihoods_batch(
            model_input, requires_grad=requires_grad
        )

    def apply_length_action(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add/remove tokens by modifying attention mask and initializing new positions."""
        B = prompt_data.shape[0]

        # For remove (action=0): handled by ModelBatchedInput.remove_suffix_token
        # For add (action=2): initialize new position with BOS embedding
        add_mask = actions == 2

        if add_mask.any():
            add_indices = torch.nonzero(add_mask, as_tuple=False).squeeze(-1)
            # Find first inactive position for each item and initialize with BOS
            # We need to get BOS embedding from model_input, but we don't have it here
            # So we'll initialize it when the suffix is updated in the optimizer
            # For now, just mark that we need initialization
            for idx in add_indices:
                current_active = lengths[idx].item()
                if current_active < self.max_suffix_len:
                    # Initialize with zero, will be set to BOS when updated via model_input
                    prompt_data.data[idx, current_active].zero_()

        # Lengths are updated via ModelBatchedInput attention mask, return as-is
        return prompt_data, lengths

    def inner_optimization_step(
        self,
        prompt_data: torch.Tensor,
        lengths: torch.Tensor,
        step: int,
        model_input: ModelBatchedInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient descent optimization step to maximize likelihood.
        """
        # Ensure embeddings require gradients
        if not prompt_data.requires_grad:
            prompt_data.requires_grad_(True)

        # Update model_input with current suffix embeddings
        model_input.update_suffix_embeddings(prompt_data)

        # Compute likelihoods with gradients
        likelihoods = self.agent.get_likelihoods_batch(model_input, requires_grad=True)

        # Loss: negative likelihood (we want to maximize likelihood = minimize negative likelihood)
        loss = -likelihoods.mean()

        # Gradient descent step
        self.prompt_optimizer.zero_grad()
        loss.backward()
        self.prompt_optimizer.step()

        # Get final likelihoods (detached for return)
        with torch.no_grad():
            model_input.update_suffix_embeddings(prompt_data)
            final_likelihoods = self.agent.get_likelihoods_batch(
                model_input, requires_grad=False
            )

        return prompt_data, final_likelihoods

    def to_tokens(
        self, prompt_data: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Project embeddings to nearest token IDs (vectorized)."""
        vocab_embeds = self.embedding_layer.weight.detach()  # [vocab_size, D]
        B = prompt_data.shape[0]
        max_len = lengths.max().item()

        if max_len == 0:
            return torch.zeros(B, 0, dtype=torch.long, device=self.device)

        # Extract active embeddings only (masked by lengths)
        active_embeds = prompt_data[:, :max_len].detach()  # [B, max_len, D]

        # Batched distance computation: [B, max_len, vocab_size]
        distances = torch.cdist(active_embeds, vocab_embeds)

        # Batched argmin: [B, max_len]
        tokens = distances.argmin(dim=-1)

        # Mask invalid positions (beyond actual length)
        length_mask = torch.arange(max_len, device=self.device).unsqueeze(
            0
        ) < lengths.unsqueeze(-1)
        tokens = torch.where(length_mask, tokens, torch.zeros_like(tokens))

        return tokens

    def clone_prompt(
        self, prompt_data: torch.Tensor, idx: int, length: int
    ) -> torch.Tensor:
        """Clone a single prompt's embeddings."""
        return prompt_data[idx, :length].clone().detach()
