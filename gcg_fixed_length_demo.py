#!/usr/bin/env python3
"""Run standalone Greedy Coordinate Gradient (GCG) token optimization at fixed length."""
import argparse
from typing import List

from prompt_rl_poc import PromptRLAgent, LengthPolicyOptimizer


DEFAULT_COMPLETION = (
    " therefore the company expects to deliver revenue growth that outpaces market forecasts, "
    "expand its gross margins through new efficiency programs, and ultimately report earnings "
    "that beat the consensus estimate by a comfortable margin."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone GCG prompt refinement")
    parser.add_argument(
        "completion",
        type=str,
        nargs="?",
        default=DEFAULT_COMPLETION,
        help="Target completion text (defaults to a complex earnings-guidance statement)",
    )
    parser.add_argument(
        "--model", type=str, default="EleutherAI/pythia-70m", help="HF model name"
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of GCG refinement passes")
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=16,
        help="Top-k token candidates considered per coordinate",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=32,
        help="Number of coordinate samples per pass",
    )
    parser.add_argument(
        "--init-len",
        dest="init_len",
        type=int,
        default=32,
        help="Initial prompt length (uses BOS tokens by default)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Optional seed prompt; overrides BOS-based initialization",
    )
    parser.add_argument("--print-trace", action="store_true", help="Print per-pass likelihood trace")
    return parser.parse_args()


def decode_tokens(agent: PromptRLAgent, token_ids: List[int]) -> str:
    try:
        return agent.tokenizer.decode(token_ids)
    except Exception:  # pragma: no cover - fallback for decoding issues
        return str(token_ids)


def main() -> None:
    args = parse_args()

    agent = PromptRLAgent(model_name=args.model)
    optimizer = LengthPolicyOptimizer(agent)

    if args.prompt:
        prompt_tokens = agent.tokenizer.encode(args.prompt, add_special_tokens=False)
        if not prompt_tokens:
            raise ValueError("Seed prompt must produce at least one token")
    else:
        bos_id = agent.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("Tokenizer does not define a BOS token; provide --prompt instead")
        prompt_tokens = [bos_id] * args.init_len

    completion_tokens = agent.tokenizer.encode(args.completion, add_special_tokens=False)
    if not completion_tokens:
        raise ValueError("Completion must produce at least one token")

    base_ll_total = agent.get_completion_likelihood(prompt_tokens, completion_tokens)
    final_tokens, ll_trace = optimizer._run_gcg_updates(  # pylint: disable=protected-access
        prompt_tokens,
        completion_tokens,
        steps=args.steps,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    final_ll_total = agent.get_completion_likelihood(final_tokens, completion_tokens)
    completion_len = len(completion_tokens)
    base_ll_avg = base_ll_total / completion_len
    final_ll_avg = final_ll_total / completion_len

    print("Initial stats")
    print(f"  Length       : {len(prompt_tokens)} tokens")
    print(f"  Log-likelihood (sum): {base_ll_total:.4f}")
    print(f"  Log-likelihood (avg): {base_ll_avg:.4f}")
    if not args.prompt:
        print("  Seed prompt  : <BOS> repeated")

    print("\nAfter GCG")
    print(f"  Log-likelihood (sum): {final_ll_total:.4f}")
    print(f"  Log-likelihood (avg): {final_ll_avg:.4f}")
    print(f"  Improvement (sum)   : {final_ll_total - base_ll_total:.4f}")
    print(f"  Improvement (avg)   : {final_ll_avg - base_ll_avg:.4f}")
    print(f"  Prompt tokens : {final_tokens}")
    decoded = decode_tokens(agent, final_tokens)
    print(f"  Prompt text   : {decoded if decoded else '<decoded to empty string>'}")

    if args.print_trace:
        print("\nLikelihood trace (per pass):")
        for idx, value in enumerate(ll_trace, start=1):
            avg_val = value / completion_len if completion_len else float('nan')
            print(f"  Pass {idx:02d}: sum={value:.4f} avg={avg_val:.4f}")


if __name__ == "__main__":
    main()
