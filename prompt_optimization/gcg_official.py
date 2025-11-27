"""
Minimal GCG utilities adapted from the official llm-attacks implementation.
Source: https://github.com/llm-attacks/llm-attacks (minimal_gcg/opt_utils.py)
Kept intentionally close to the original to mirror candidate sampling and gradients.
"""

import torch
import torch.nn as nn


def _get_embedding_layer(model):
    """Return the model input embedding layer."""
    return model.get_input_embeddings()


def _get_embedding_matrix(model):
    """Return the embedding weight matrix."""
    return _get_embedding_layer(model).weight


def _get_embeddings(model, input_ids):
    """Lookup embeddings for given token ids."""
    return _get_embedding_layer(model)(input_ids)


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Compute gradients of the loss w.r.t. the coordinates (control tokens).

    Args:
        model: Causal LM.
        input_ids: Tensor of token ids for control + target.
        input_slice: slice over the control tokens.
        target_slice: slice over the target tokens.
        loss_slice: slice over logits positions to score targets.
    """
    model_device = next(model.parameters()).device
    embed_weights = _get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model_device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model_device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # stitch with the rest of the embeddings
    embeds = _get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1
    )

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    loss.backward()

    grad = one_hot.grad.clone()
    grad = grad / grad.norm(dim=-1, keepdim=True)
    return grad


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):
    """
    Sample candidate control sequences following the official GCG sampling rule.
    Picks evenly spaced positions and replaces with one of the top-k gradient tokens.
    """
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float("inf")

    topk = min(topk, grad.shape[1])
    top_indices = (-grad).topk(topk, dim=1).indices  # [L, topk]
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    # Sample positions uniformly; handles batch_size > len(control_toks)
    pos_choices = torch.randint(0, len(control_toks), (batch_size,), device=grad.device)
    tok_choices = torch.randint(0, topk, (batch_size,), device=grad.device)
    new_token_val = top_indices[pos_choices, tok_choices].unsqueeze(-1)  # [B, 1]
    new_token_pos = pos_choices.unsqueeze(-1)  # [B, 1]
    new_control_toks = original_control_toks.scatter_(1, new_token_pos, new_token_val)
    return new_control_toks
