#!/usr/bin/env python3
"""
Download AdvBench and save a fixed subset of prefix/target pairs with length stats.

Outputs a JSONL file with 64 examples by default:
    {
      "id": <int>,                 # index in the original split
      "prefix": <str>,             # selected prompt/instruction/etc.
      "target": <str>,             # selected target/completion/etc.
      "source_prompt_key": <str>,  # which key was used for prefix
      "source_target_key": <str>,  # which key was used for target
      "prefix_len_chars": <int>,
      "target_len_chars": <int>,
      "prefix_len_tokens": <int>,
      "target_len_tokens": <int>
    }

Usage:
    python scripts/create_advbench_subset.py \
        --output data/advbench_subset_64.jsonl \
        --num-examples 64 \
        --tokenizer EleutherAI/pythia-70m
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


PROMPT_KEYS = ["prompt", "instruction", "input", "question"]
TARGET_KEYS = ["target", "completion", "output", "response", "answer"]


def pick_prompt_and_target(example: dict) -> Optional[Tuple[str, str, str, str]]:
    """Pick the first available prompt and target fields from known key sets."""
    prefix_key = next((k for k in PROMPT_KEYS if k in example and example[k]), None)
    target_key = next((k for k in TARGET_KEYS if k in example and example[k]), None)
    if prefix_key is None or target_key is None:
        return None
    return (
        example[prefix_key],
        example[target_key],
        prefix_key,
        target_key,
    )


def main():
    parser = argparse.ArgumentParser(description="Create a fixed AdvBench subset with stats.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/advbench_subset_64.jsonl"),
        help="Path to the JSONL file to write.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=64,
        help="Number of prefix/target pairs to save.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/pythia-70m",
        help="Tokenizer name for token-length stats.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading AdvBench dataset...")
    ds = load_dataset("walledai/AdvBench", split="train")
    print(f"Loaded {len(ds)} examples.")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selected = []
    for idx, ex in enumerate(ds):
        picked = pick_prompt_and_target(ex)
        if picked is None:
            continue
        prefix, target, p_key, t_key = picked

        prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]

        record = {
            "id": idx,
            "prefix": prefix,
            "target": target,
            "source_prompt_key": p_key,
            "source_target_key": t_key,
            "prefix_len_chars": len(prefix),
            "target_len_chars": len(target),
            "prefix_len_tokens": len(prefix_tokens),
            "target_len_tokens": len(target_tokens),
        }
        selected.append(record)
        if len(selected) >= args.num_examples:
            break

    if len(selected) < args.num_examples:
        raise ValueError(
            f"Only found {len(selected)} usable examples, fewer than requested {args.num_examples}."
        )

    print(f"Writing {len(selected)} examples to {args.output} ...")
    with args.output.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Done.")


if __name__ == "__main__":
    main()
