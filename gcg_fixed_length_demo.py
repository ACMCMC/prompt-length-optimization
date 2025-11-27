#!/usr/bin/env python3
"""
Fixed-length GCG demo using the official gradient + sampling update.
Optimizes a suffix of fixed length against a target completion, optionally with a fixed prefix.
"""
import argparse
from typing import List, Tuple

import torch

from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.gcg_official import token_gradients, sample_control


DEFAULT_COMPLETION = (
    "Therefore the company expects to deliver revenue growth that outpaces market forecasts and "
    "ultimately report earnings that beat the consensus estimate."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-length GCG demo (prefix + suffix).")
    parser.add_argument("completion", nargs="?", default=DEFAULT_COMPLETION, help="Target completion text")
    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="HF model name")
    parser.add_argument("--steps", type=int, default=20, help="Number of GCG passes")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k tokens per coordinate")
    parser.add_argument("--candidates", type=int, default=32, help="Candidate samples per pass")
    parser.add_argument("--suffix-len", type=int, default=32, help="Fixed suffix length to optimize")
    parser.add_argument("--prefix", type=str, default="", help="Optional fixed prefix text")
    parser.add_argument("--seed-suffix", type=str, default=None, help="Optional seed suffix text (overrides random init)")
    parser.add_argument("--print-trace", action="store_true", help="Print likelihood trace per pass")
    return parser.parse_args()


def decode_tokens(agent: PromptRLAgent, token_ids: List[int]) -> str:
    try:
        return agent.tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        return str(token_ids)


def score_suffix(agent: PromptRLAgent, prefix_ids: List[int], suffix_ids: torch.Tensor,
                 completion_ids: torch.Tensor) -> torch.Tensor:
    """Compute log P(completion | prefix + suffix) for a batch of suffix candidates."""
    device = agent.device
    embedding_layer = agent.model.get_input_embeddings()
    comp_len = completion_ids.shape[0]
    comp_batch = completion_ids.unsqueeze(0).expand(suffix_ids.shape[0], -1).to(device)
    comp_lengths = torch.full((suffix_ids.shape[0],), comp_len, device=device, dtype=torch.long)
    prefix_tensor = None
    prefix_lengths = None
    if prefix_ids:
        pref = torch.tensor(prefix_ids, device=device, dtype=torch.long)
        prefix_tensor = pref.unsqueeze(0).expand(suffix_ids.shape[0], -1)
        prefix_lengths = torch.full((suffix_ids.shape[0],), len(prefix_ids), device=device, dtype=torch.long)
    suffix_embeds = embedding_layer(suffix_ids.to(device))
    ll_batch = agent.get_likelihoods_batch(
        suffix_embeds,
        comp_batch,
        comp_lengths,
        requires_grad=False,
        prefix_tokens=prefix_tensor,
        prefix_lengths=prefix_lengths,
    )
    return ll_batch


def gcg_fixed(agent: PromptRLAgent, prefix_ids: List[int], suffix_ids: List[int],
              completion_ids: List[int], steps: int, top_k: int, candidates: int) -> Tuple[List[int], List[float]]:
    """Run fixed-length GCG on the suffix."""
    device = agent.device
    suffix = torch.tensor(suffix_ids, device=device, dtype=torch.long)
    comp = torch.tensor(completion_ids, device=device, dtype=torch.long)
    trace = []
    not_allowed = torch.tensor(list(agent.special_token_ids), device=device) if agent.special_token_ids else None

    for _ in range(steps):
        # Build full input ids for gradient
        parts = []
        if prefix_ids:
            parts.append(torch.tensor(prefix_ids, device=device, dtype=torch.long))
        parts.append(suffix)
        parts.append(comp)
        input_ids = torch.cat(parts, dim=0)
        pref_len = len(prefix_ids)
        L = suffix.shape[0]
        control_slice = slice(pref_len, pref_len + L)
        target_slice = slice(pref_len + L, pref_len + L + len(comp))
        loss_slice = slice(pref_len + L - 1, pref_len + L - 1 + len(comp))

        agent.model.zero_grad(set_to_none=True)
        grad = token_gradients(agent.model, input_ids, control_slice, target_slice, loss_slice)

        # Sample candidates for this pass
        cand_suffixes = sample_control(
            suffix,
            grad,
            batch_size=candidates,
            topk=top_k,
            temp=1,
            not_allowed_tokens=not_allowed,
        )

        # Score and pick best
        ll_batch = score_suffix(agent, prefix_ids, cand_suffixes, comp)
        best_idx = torch.argmax(ll_batch)
        suffix = cand_suffixes[best_idx]
        trace.append(ll_batch[best_idx].item())

    return suffix.cpu().tolist(), trace


def main() -> None:
    args = parse_args()
    agent = PromptRLAgent(model_name=args.model)

    prefix_ids = agent.tokenizer.encode(args.prefix, add_special_tokens=False) if args.prefix else []
    if args.seed_suffix is not None:
        suffix_ids = agent.tokenizer.encode(args.seed_suffix, add_special_tokens=False)
        if len(suffix_ids) == 0:
            raise ValueError("Seed suffix must produce at least one token")
        # pad/trim to desired length
        if len(suffix_ids) < args.suffix_len:
            pad_id = agent.tokenizer.bos_token_id or 0
            suffix_ids = suffix_ids + [pad_id] * (args.suffix_len - len(suffix_ids))
        else:
            suffix_ids = suffix_ids[:args.suffix_len]
    else:
        suffix_ids = [agent.get_random_token() for _ in range(args.suffix_len)]

    completion_ids = agent.tokenizer.encode(args.completion, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Completion must produce at least one token")

    # Baseline LL
    base_ll = score_suffix(
        agent,
        prefix_ids,
        torch.tensor(suffix_ids, device=agent.device).unsqueeze(0),
        torch.tensor(completion_ids, device=agent.device),
    )[0].item()

    final_suffix, ll_trace = gcg_fixed(
        agent,
        prefix_ids=prefix_ids,
        suffix_ids=suffix_ids,
        completion_ids=completion_ids,
        steps=args.steps,
        top_k=args.top_k,
        candidates=args.candidates,
    )

    final_ll = score_suffix(
        agent,
        prefix_ids,
        torch.tensor(final_suffix, device=agent.device).unsqueeze(0),
        torch.tensor(completion_ids, device=agent.device),
    )[0].item()

    comp_len = len(completion_ids)
    print("Initial stats")
    print(f"  Prefix length : {len(prefix_ids)}")
    print(f"  Suffix length : {len(suffix_ids)}")
    print(f"  Log-likelihood (sum): {base_ll:.4f} (avg/token: {base_ll/comp_len:.4f})")

    print("\nAfter GCG")
    print(f"  Log-likelihood (sum): {final_ll:.4f} (avg/token: {final_ll/comp_len:.4f})")
    print(f"  Improvement (sum)   : {final_ll - base_ll:.4f} (avg/token: {(final_ll - base_ll)/comp_len:.4f})")
    full_prompt_ids = prefix_ids + final_suffix
    print(f"  Full prompt tokens  : {full_prompt_ids}")
    decoded = decode_tokens(agent, full_prompt_ids)
    print(f"  Full prompt text    : {decoded if decoded else '<decoded to empty string>'}")

    if args.print_trace:
        print("\nLikelihood trace (per pass):")
        for i, v in enumerate(ll_trace, 1):
            print(f"  Pass {i:02d}: sum={v:.4f} avg/token={v/comp_len:.4f}")


if __name__ == "__main__":
    main()
