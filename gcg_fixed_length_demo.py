#!/usr/bin/env python3
"""
Fixed-length GCG demo using the official gradient + sampling update.
Optimizes a suffix of fixed length against a target completion, optionally with a fixed prefix.
"""
import argparse
from typing import List, Tuple
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import os

import torch

from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.gcg_official import token_gradients, sample_control

PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
COMPLETION_KEYS = ["target", "completion", "output", "response", "answer"]

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
    parser.add_argument("--sweep", action="store_true", help="Sweep suffix lengths from 1..suffix-len and report best")
    parser.add_argument("--prefix", type=str, default="", help="Optional fixed prefix text")
    parser.add_argument("--seed-suffix", type=str, default=None, help="Optional seed suffix text (overrides random init)")
    parser.add_argument("--print-trace", action="store_true", help="Print likelihood trace per pass")
    parser.add_argument("--advbench", action="store_true", help="Load a single AdvBench example as prefix+completion")
    parser.add_argument("--advbench-index", type=int, default=0, help="Index of AdvBench sample to use (default 0)")
    parser.add_argument("--output", type=str, default="results/gcg_sweep_results.json", help="Path to save sweep results")
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
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    agent = PromptRLAgent(model_name=args.model)
    try:
        import matplotlib.pyplot as plt
        plotting_available = True
    except Exception:
        plotting_available = False

    # Optionally load an AdvBench example
    adv_prefix_text = None
    adv_completion_text = None
    if args.advbench:
        from datasets import load_dataset
        raw = load_dataset("walledai/AdvBench", split="train")
        idx = max(0, min(args.advbench_index, len(raw) - 1))
        ex = raw[idx]
        base = ""
        for k in PROMPT_KEYS:
            if k in ex and ex[k]:
                base = ex[k]
                break
        target = ""
        for k in COMPLETION_KEYS:
            if k in ex and ex[k]:
                target = ex[k]
                break
        adv_prefix_text = base.strip()
        adv_completion_text = target.strip()
        print(f"Loaded AdvBench example idx={idx}")
        print(f"  Base prompt (prefix): {adv_prefix_text[:120]}{'...' if len(adv_prefix_text) > 120 else ''}")
        print(f"  Target completion   : {adv_completion_text[:120]}{'...' if len(adv_completion_text) > 120 else ''}")

    # Resolve prefix/completion text
    prefix_text = adv_prefix_text if adv_prefix_text is not None else args.prefix
    completion_text = adv_completion_text if adv_completion_text is not None else args.completion

    prefix_ids = agent.tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
    completion_ids = agent.tokenizer.encode(completion_text, add_special_tokens=False)
    if not completion_ids:
        raise ValueError("Completion must produce at least one token")

    def make_seed_suffix(length: int) -> List[int]:
        if args.seed_suffix is not None:
            toks = agent.tokenizer.encode(args.seed_suffix, add_special_tokens=False)
            if len(toks) == 0:
                raise ValueError("Seed suffix must produce at least one token")
            if len(toks) < length:
                pad_id = agent.tokenizer.bos_token_id or 0
                toks = toks + [pad_id] * (length - len(toks))
            else:
                toks = toks[:length]
            return toks
        return [agent.get_random_token() for _ in range(length)]

    def run_for_length(length: int):
        suffix_ids = make_seed_suffix(length)
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
        return {
            "length": length,
            "base_ll": base_ll,
            "final_ll": final_ll,
            "improve": final_ll - base_ll,
            "suffix": final_suffix,
            "trace": ll_trace,
        }

    results = []
    if args.sweep:
        with ThreadPool(mp.cpu_count()) as pool:
            results = pool.map(run_for_length, list(range(1, args.suffix_len + 1)))
        best = max(results, key=lambda r: r["final_ll"])
        comp_len = len(completion_ids)
        print(f"Sweep results (1..{args.suffix_len})")
        for r in results:
            decoded_suffix = decode_tokens(agent, r["suffix"])
            print(f"  L={r['length']:2d} final_ll={r['final_ll']:.4f} improve={r['improve']:.4f} suffix='{decoded_suffix}'")
        print("\nBest result")
        print(f"  Length              : {best['length']}")
        print(f"  Log-likelihood (sum): {best['final_ll']:.4f} (avg/token: {best['final_ll']/comp_len:.4f})")
        print(f"  Improvement (sum)   : {best['improve']:.4f} (avg/token: {best['improve']/comp_len:.4f})")
        full_prompt_ids = prefix_ids + best["suffix"]
        print(f"  Full prompt tokens  : {full_prompt_ids}")
        decoded = decode_tokens(agent, full_prompt_ids)
        print(f"  Full prompt text    : {decoded if decoded else '<decoded to empty string>'}")
        if args.print_trace:
            print("\nTrace for best length:")
            for i, v in enumerate(best["trace"], 1):
                print(f"  Pass {i:02d}: sum={v:.4f} avg/token={v/comp_len:.4f}")
        if plotting_available:
            lengths = [r["length"] for r in results]
            finals = [r["final_ll"] for r in results]
            bases = [r["base_ll"] for r in results]
            plt.figure(figsize=(7, 4))
            plt.plot(lengths, finals, label="Final LL", marker="o")
            plt.plot(lengths, bases, label="Base LL", linestyle="--", marker="x", alpha=0.7)
            plt.xlabel("Suffix length")
            plt.ylabel("Log-likelihood (sum)")
            plt.title("GCG sweep over suffix length")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("gcg_sweep.png", dpi=120)
            print("Saved sweep plot to gcg_sweep.png")
        # Save results to JSON
        try:
            import json, os
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            serializable = []
            for r in results:
                serializable.append({
                    "length": r["length"],
                    "base_ll": r["base_ll"],
                    "final_ll": r["final_ll"],
                    "improve": r["improve"],
                    "suffix_tokens": r["suffix"],
                    "suffix_text": decode_tokens(agent, r["suffix"]),
                })
            with open(args.output, "w") as f:
                json.dump({
                    "prefix": prefix_text,
                    "completion": completion_text,
                    "results": serializable,
                    "best": {
                        "length": best["length"],
                        "final_ll": best["final_ll"],
                        "improve": best["improve"],
                        "suffix_tokens": best["suffix"],
                        "suffix_text": decode_tokens(agent, best["suffix"]),
                    }
                }, f, indent=2)
            print(f"Saved sweep data to {args.output}")
        except Exception as e:
            print(f"Could not save results to {args.output}: {e}")
    else:
        res = run_for_length(args.suffix_len)
        comp_len = len(completion_ids)
        print("Initial stats")
        print(f"  Prefix length : {len(prefix_ids)}")
        print(f"  Suffix length : {res['length']}")
        print(f"  Log-likelihood (sum): {res['base_ll']:.4f} (avg/token: {res['base_ll']/comp_len:.4f})")

        print("\nAfter GCG")
        print(f"  Log-likelihood (sum): {res['final_ll']:.4f} (avg/token: {res['final_ll']/comp_len:.4f})")
        print(f"  Improvement (sum)   : {res['improve']:.4f} (avg/token: {res['improve']/comp_len:.4f})")
        full_prompt_ids = prefix_ids + res["suffix"]
        print(f"  Full prompt tokens  : {full_prompt_ids}")
        decoded = decode_tokens(agent, full_prompt_ids)
        print(f"  Full prompt text    : {decoded if decoded else '<decoded to empty string>'}")

        if args.print_trace:
            print("\nLikelihood trace (per pass):")
            for i, v in enumerate(res["trace"], 1):
                print(f"  Pass {i:02d}: sum={v:.4f} avg/token={v/comp_len:.4f}")


if __name__ == "__main__":
    main()
