#!/usr/bin/env python3
"""
Run a detached GCG sweep over suffix lengths and visualize per-token log-likelihood gains.

The script samples prompts/completions, runs GCG for each suffix length, records the
starting/ending per-token log-likelihood, and saves a plot plus optional CSV.
"""

import argparse
import logging
import os
import random
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import torch
import yaml

from prompt_optimization.agent import PromptRLAgent
from prompt_optimization.model_inputs import ModelBatchedInput
from prompt_optimization.optimizers.discrete import DiscretePromptOptimizer


PLOT_PALETTE = {
    "color_1": "#007ACC",
    "color_2": "#C200D6",
    "color_3": "#D50000",
}
PLOT_COLORS = list(PLOT_PALETTE.values())

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


PROMPT_FIELDS = ["prompt", "instruction", "input", "question"]
COMPLETION_FIELDS = ["target", "completion", "output", "response", "answer"]


def setup_plot_style():
    """Set up consistent plotting style across all plots."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "IBMPlexSans-Regular.ttf"
    try:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [prop.get_name()]
        sns.set_style(
            "whitegrid",
            {"font.family": ["sans-serif"], "font.sans-serif": [prop.get_name()]},
        )
    except Exception as e:
        logger.warning(f"Could not load IBM Plex Sans font from {font_path}: {e}")
        logger.info("Falling back to default font")

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep suffix lengths with GCG and plot per-token log-likelihood."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to training config for defaults.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="advbench",
        choices=["advbench"],
        help="Dataset to sample from (AdvBench only for now).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of prompt/completion pairs to sample.",
    )
    parser.add_argument(
        "--min-suffix-len",
        type=int,
        default=1,
        help="Minimum suffix length to evaluate.",
    )
    parser.add_argument(
        "--max-suffix-len",
        type=int,
        default=32,
        help="Maximum suffix length to evaluate.",
    )
    parser.add_argument(
        "--gcg-steps",
        type=int,
        default=128,
        help="Number of GCG steps to run per suffix length.",
    )
    parser.add_argument(
        "--gcg-top-k",
        type=int,
        default=None,
        help="Override for GCG top-k (defaults to config value).",
    )
    parser.add_argument(
        "--gcg-batch-size",
        type=int,
        default=None,
        help="Override for GCG batch size (defaults to config value).",
    )
    parser.add_argument(
        "--gcg-max-batch-size",
        type=int,
        default=None,
        help="Override for GCG max batch size (defaults to config value).",
    )
    parser.add_argument(
        "--length-batch-size",
        type=int,
        default=24,
        help="How many suffix lengths to evaluate in parallel.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override (defaults to config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (defaults to config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/gcg_length_sweep.pdf",
        help="Path to save the plot (format inferred from extension).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/gcg_length_sweep.csv",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--show-gcg-progress",
        action="store_true",
        help="If set, keep tqdm progress bars from the underlying optimizer.",
    )
    return parser.parse_args()


def load_advbench_pairs(
    num_samples: int, seed: int, min_len: int, max_len: int
) -> List[Dict[str, str]]:
    """Load AdvBench and return prompt/completion dicts."""
    from datasets import load_dataset

    raw = load_dataset("walledai/AdvBench", split="train")
    pairs: List[Dict[str, str]] = []

    for example in raw:
        base = None
        for key in PROMPT_FIELDS:
            val = example.get(key)
            if val:
                base = val.strip()
                break
        if base is None:
            continue
        if len(base) < min_len or len(base) > max_len:
            continue

        completion = None
        for key in COMPLETION_FIELDS:
            val = example.get(key)
            if val:
                completion = val.strip()
                break
        if not completion:
            continue

        pairs.append({"base": base, "target": completion})

    if len(pairs) < num_samples:
        raise ValueError(
            f"Only {len(pairs)} valid AdvBench pairs available, need {num_samples}."
        )

    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:num_samples]


def ensure_dir(path: str) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def run_gcg_for_length_batch(
    agent: PromptRLAgent,
    prefixes: List[str],
    completions: List[str],
    lengths_to_eval: List[int],
    gcg_steps: int,
    gcg_top_k: int,
    gcg_batch_size: int,
    gcg_max_batch_size: int,
    lr_embeddings: float,
    init_seed: int,
) -> Dict[int, Tuple[float, float]]:
    """Run GCG on a batch of suffix lengths and return per-length stats."""
    repeats = len(prefixes)
    batch_prefixes: List[str] = []
    batch_completions: List[str] = []
    length_labels: List[int] = []
    for suffix_len in lengths_to_eval:
        batch_prefixes.extend(prefixes)
        batch_completions.extend(completions)
        length_labels.extend([suffix_len] * repeats)

    max_suffix_len = max(lengths_to_eval)
    optimizer = DiscretePromptOptimizer(
        agent=agent,
        initial_prompt_length=max_suffix_len,
        max_prompt_len=max_suffix_len,
        batch_size=len(batch_prefixes),
        lr_embeddings=lr_embeddings,
        max_suffix_len=max_suffix_len,
        init_len=max_suffix_len,
        gcg_steps=gcg_steps,
        gcg_top_k=gcg_top_k,
        gcg_batch_size=gcg_batch_size,
        gcg_max_batch_size=gcg_max_batch_size,
    )

    model_input = ModelBatchedInput(
        prefix_texts=batch_prefixes,
        completion_texts=batch_completions,
        tokenizer=agent.tokenizer,
        device=agent.device,
        embedding_layer=agent.model.get_input_embeddings(),
        max_suffix_len=max_suffix_len,
        init_len=max_suffix_len,
        mode="discrete",
    )

    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    prompt_data, lengths = optimizer.initialize_prompts(model_input)
    target_lengths = torch.tensor(
        length_labels,
        dtype=torch.long,
        device=optimizer.device,
    )
    lengths.copy_(target_lengths)

    arange = torch.arange(max_suffix_len, device=optimizer.device).unsqueeze(0)
    suffix_mask = (arange < target_lengths.unsqueeze(1)).long()
    model_input.suffix_attention_mask = suffix_mask.clone()

    bos_token_id = agent.tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = agent.tokenizer.pad_token_id
    if bos_token_id is None:
        bos_token_id = 0
    inactive_mask = suffix_mask == 0
    prompt_data = prompt_data.clone()
    prompt_data[inactive_mask] = bos_token_id
    model_input.update_suffix_tokens(prompt_data)

    completion_token_counts = (
        model_input.completion_attention_mask.sum(dim=1).float().clamp(min=1.0)
    )

    with torch.no_grad():
        initial_lls = optimizer.get_likelihoods(
            prompt_data, lengths, model_input, requires_grad=False
        )

    _, final_lls = optimizer.inner_optimization_step(
        prompt_data, lengths, step=0, model_input=model_input
    )
    completion_token_counts_final = (
        model_input.completion_attention_mask.sum(dim=1).float().clamp(min=1.0)
    )

    results: Dict[int, Tuple[float, float]] = {}
    for suffix_len in lengths_to_eval:
        mask = target_lengths == suffix_len
        init_mean = (
            (initial_lls[mask] / completion_token_counts[mask]).mean().item()
        )
        final_mean = (
            (final_lls[mask] / completion_token_counts_final[mask]).mean().item()
        )
        results[suffix_len] = (init_mean, final_mean)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main() -> None:
    args = parse_args()

    if not args.output.lower().endswith(".pdf"):
        raise ValueError("Output plot must be a PDF file (use .pdf extension).")

    if args.num_samples != 1:
        logger.info("Overriding num-samples to 1 for efficiency.")
        args.num_samples = 1

    if args.length_batch_size != 24:
        logger.info("Overriding length-batch-size to 24 for efficiency.")
        args.length_batch_size = 24

    with open(args.config, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    seed = args.seed or cfg.get("seed", 2262)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_cfg = cfg.get("train", {})
    dataset_cfg = cfg.get("dataset", {})
    min_prompt_length = train_cfg.get("min_prompt_length", 30)
    max_prompt_length = train_cfg.get("max_prompt_length", 150)
    lr_embeddings = train_cfg.get("lr_embeddings", 0.01)

    gcg_cfg = train_cfg.get("gcg", {})
    gcg_top_k = args.gcg_top_k or gcg_cfg.get("top_k", 64)
    gcg_batch_size = args.gcg_batch_size or gcg_cfg.get("batch_size", 64)
    gcg_max_batch_size = args.gcg_max_batch_size or gcg_cfg.get(
        "max_batch_size", 256
    )

    if not args.show_gcg_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")

    if args.dataset.lower() != "advbench":
        raise ValueError("Only AdvBench is supported in this script.")

    if args.min_suffix_len == 1 and args.max_suffix_len == 32:
        if args.gcg_steps != 128:
            logger.info("Overriding GCG steps to 128 for sweep across lengths 1-32.")
        args.gcg_steps = 128

    prompt_pairs = load_advbench_pairs(
        num_samples=args.num_samples,
        seed=seed,
        min_len=min_prompt_length,
        max_len=max_prompt_length,
    )
    prefixes = [p["base"] for p in prompt_pairs]
    completions = [p["target"] for p in prompt_pairs]

    model_name = args.model or cfg.get("model", "EleutherAI/pythia-70m")
    agent = PromptRLAgent(model_name=model_name)

    print(
        f"Running GCG sweep on {len(prefixes)} samples "
        f"(steps={args.gcg_steps}, top_k={gcg_top_k}, batch={gcg_batch_size})"
    )

    lengths = list(range(args.min_suffix_len, args.max_suffix_len + 1))
    length_results: Dict[int, Tuple[float, float]] = {}
    batch_size = max(1, args.length_batch_size)
    length_batches = [
        lengths[idx : idx + batch_size] for idx in range(0, len(lengths), batch_size)
    ]

    for batch_lengths in length_batches:
        print(f"Evaluating suffix lengths {batch_lengths}...")
        batch_stats = run_gcg_for_length_batch(
            agent=agent,
            prefixes=prefixes,
            completions=completions,
            lengths_to_eval=batch_lengths,
            gcg_steps=args.gcg_steps,
            gcg_top_k=gcg_top_k,
            gcg_batch_size=gcg_batch_size,
            gcg_max_batch_size=gcg_max_batch_size,
            lr_embeddings=lr_embeddings,
            init_seed=seed,
        )
        for suffix_len in batch_lengths:
            init_ll, final_ll = batch_stats[suffix_len]
            length_results[suffix_len] = (init_ll, final_ll)
            print(
                f"  len={suffix_len:2d}: start={init_ll:.3f}, "
                f"final={final_ll:.3f}, delta={final_ll - init_ll:.3f}"
            )

    initial_curve = [length_results[length][0] for length in lengths]
    final_curve = [length_results[length][1] for length in lengths]

    ensure_dir(args.output)
    setup_plot_style()
    plt.figure(figsize=(8, 5))
    plt.plot(
        lengths,
        initial_curve,
        label="Initial per-token LL",
        marker="o",
        color=PLOT_PALETTE["color_1"],
    )
    plt.plot(
        lengths,
        final_curve,
        label="Final per-token LL",
        marker="s",
        color=PLOT_PALETTE["color_3"],
    )
    plt.xlabel("Suffix length (tokens)")
    plt.ylabel("Average per-token log-likelihood")
    plt.title("GCG sweep over suffix lengths")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")

    if args.csv:
        ensure_dir(args.csv)
        with open(args.csv, "w") as csv_file:
            csv_file.write("suffix_len,initial_per_token_ll,final_per_token_ll\n")
            for length, init_ll, final_ll in zip(lengths, initial_curve, final_curve):
                csv_file.write(f"{length},{init_ll:.6f},{final_ll:.6f}\n")
        print(f"Wrote metrics to {args.csv}")


if __name__ == "__main__":
    main()

