#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for top-k/top-p sampling kernel.

Usage:
    # Run with default parameters
    python bench_topk_topp_sampling.py

    # Run with custom shape
    python bench_topk_topp_sampling.py --shape 32 128256 --k 10 --p 0.9

    # Save results to CSV
    python bench_topk_topp_sampling.py -o results.csv

    # Run sweep across multiple configurations
    python bench_topk_topp_sampling.py --sweep
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import triton
from triton.testing import runtime

from aiter.ops import sampling  # noqa: F401

DEVICE = "cuda"

# Default sweep parameters
BATCH_SIZES = [1, 19, 99, 989]
VOCAB_SIZES = [111, 500, 32000, 128256]
K_VALUES = [1, 50]
P_VALUES = [0.1, 0.9]


def bench_topk_topp_sampling_latency(
    batch_size: int,
    vocab_size: int,
    k: int,
    p: float,
    warmup: int = 25,
    rep: int = 100,
) -> tuple[float, float, float]:
    """
    Benchmark the top-k/top-p sampling kernel.

    Returns:
        Tuple of (median, p20, p80) latency in milliseconds.
    """
    # Generate random normalized probabilities
    pre_norm_prob = torch.rand(
        batch_size, vocab_size, device=DEVICE, dtype=torch.float32
    )
    probs = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)

    # Clear GPU cache for consistent measurements
    runtime.driver.active.get_empty_cache_for_benchmark().zero_()

    def fn():
        return torch.ops.aiter.top_k_top_p_sampling_from_probs(
            probs,
            None,  # indices
            None,  # top_k_arr
            k,  # top_k_val
            None,  # top_p_arr
            p,  # top_p_val
            True,  # deterministic
        )

    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    return ms, p20, p80


def run_single_benchmark(args):
    """Run benchmark for a single configuration."""
    batch_size, vocab_size = args.shape
    k, p = args.k, args.p

    print("\nBenchmarking top-k/top-p sampling:")
    print(f"  batch_size={batch_size}, vocab_size={vocab_size}, k={k}, p={p}")
    print()

    ms, p20, p80 = bench_topk_topp_sampling_latency(batch_size, vocab_size, k, p)

    print("Results:")
    print(f"  Median latency: {ms:.4f} ms")
    print(f"  P20 latency:    {p20:.4f} ms")
    print(f"  P80 latency:    {p80:.4f} ms")
    print(f"  Throughput:     {batch_size / ms * 1000:.2f} samples/sec")

    if args.o:
        _save_results_csv(
            args.o,
            [(batch_size, vocab_size, k, p, ms, p20, p80)],
        )


def run_sweep_benchmark(args):
    """Run benchmark across multiple configurations."""
    batch_sizes = BATCH_SIZES if args.batch_sizes is None else args.batch_sizes
    vocab_sizes = VOCAB_SIZES if args.vocab_sizes is None else args.vocab_sizes
    k_values = K_VALUES if args.k_values is None else args.k_values
    p_values = P_VALUES if args.p_values is None else args.p_values

    configs = list(itertools.product(batch_sizes, vocab_sizes, k_values, p_values))
    total = len(configs)

    print(f"\nRunning sweep benchmark across {total} configurations...")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  vocab_sizes: {vocab_sizes}")
    print(f"  k_values:    {k_values}")
    print(f"  p_values:    {p_values}")
    print()

    results = []
    header = f"{'batch':>8} {'vocab':>8} {'k':>6} {'p':>6} {'median_ms':>12} {'p20_ms':>10} {'p80_ms':>10} {'samples/s':>12}"
    print(header)
    print("-" * len(header))

    for i, (batch_size, vocab_size, k, p) in enumerate(configs, 1):
        ms, p20, p80 = bench_topk_topp_sampling_latency(batch_size, vocab_size, k, p)
        throughput = batch_size / ms * 1000

        results.append((batch_size, vocab_size, k, p, ms, p20, p80))

        print(
            f"{batch_size:>8} {vocab_size:>8} {k:>6} {p:>6.2f} "
            f"{ms:>12.4f} {p20:>10.4f} {p80:>10.4f} {throughput:>12.2f}"
        )

    print()
    print(f"Completed {total} configurations.")

    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath: str, results: list):
    """Save benchmark results to CSV file."""
    path = Path(filepath)
    with open(path, "w") as f:
        f.write(
            "batch_size,vocab_size,k,p,median_ms,p20_ms,p80_ms,throughput_samples_per_sec\n"
        )
        for batch_size, vocab_size, k, p, ms, p20, p80 in results:
            throughput = batch_size / ms * 1000
            f.write(
                f"{batch_size},{vocab_size},{k},{p},{ms:.6f},{p20:.6f},{p80:.6f},{throughput:.2f}\n"
            )
    print(f"Results saved to {path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Top-K/Top-P Sampling",
        description="Benchmark the top-k/top-p sampling kernel performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("BATCH", "VOCAB"),
        default=[32, 128256],
        help="Batch size and vocabulary size for single benchmark.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k value for single benchmark.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.9,
        help="Top-p value for single benchmark.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run sweep across multiple configurations instead of single benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        dest="batch_sizes",
        help=f"Batch sizes for sweep (default: {BATCH_SIZES}).",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=int,
        nargs="+",
        dest="vocab_sizes",
        help=f"Vocabulary sizes for sweep (default: {VOCAB_SIZES}).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        dest="k_values",
        help=f"Top-k values for sweep (default: {K_VALUES}).",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        dest="p_values",
        help=f"Top-p values for sweep (default: {P_VALUES}).",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file path for results.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Number of benchmark repetitions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sweep:
        run_sweep_benchmark(args)
    else:
        run_single_benchmark(args)


if __name__ == "__main__":
    main()
