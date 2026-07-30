#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused MoE route-sort + MXFP4 activation-quant kernel.

Measures the pre-GEMM plumbing of an MXFP4 MoE decode step, comparing the
conventional two-launch sequence against the fused kernel:

    baseline    moe_sorting + fused_dynamic_mxfp4_quant_moe_sort
    fused       mxfp4_moe_sort_quant_fwd (compact per-token scales)
    fused+ss    the above + mxfp4_moe_sort_fwd (sorted-row scale layout)

Each variant is CUDA-graph captured and replayed, so the numbers are launch
cost inclusive and free of Python overhead.

Usage:
    # Run with default parameters (Kimi decode shape)
    python bench_mxfp4_moe_sort_quant.py

    # Run a single route variant at selected token counts
    python bench_mxfp4_moe_sort_quant.py --routes 384,8 -t 1 8 32

    # Save results to CSV
    python bench_mxfp4_moe_sort_quant.py -o results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    _USE_CK_MOE_SORTING,
    _USE_FLYDSL_MOE_SORTING,
    moe_sorting,
)
from aiter.ops.quant import fused_dynamic_mxfp4_quant_moe_sort, mxfp4_moe_sort_fwd

DEVICE = "cuda"

# Default sweep parameters (Kimi K2 TP8 shard)
MODEL_DIM = 7168
BLOCK_M = 32
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]
ROUTES = [(384, 8), (385, 9)]


def time_graph(fn, iters: int, warmup: int, reps: int) -> float:
    """CUDA-graph capture `fn`, then return the best per-replay time in us."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1000.0)
    return best


def make_inputs(tokens: int, experts: int, topk: int, model_dim: int):
    """Build hidden states and a route assignment for `experts`/`topk`.

    When `experts` exceeds the 384 routed experts the last slot is pinned to the
    fused shared expert, matching how the Kimi E=385/topk=9 config routes.
    """
    torch.manual_seed(1000 + tokens)
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device=DEVICE)
    routed = 384
    if experts == routed:
        topk_ids = torch.stack(
            [
                torch.randperm(experts, dtype=dtypes.i32, device=DEVICE)[:topk]
                for _ in range(tokens)
            ]
        )
        topk_weights = torch.rand((tokens, topk), dtype=dtypes.fp32, device=DEVICE)
    else:
        routed_ids = torch.stack(
            [
                torch.randperm(routed, dtype=dtypes.i32, device=DEVICE)[: topk - 1]
                for _ in range(tokens)
            ]
        )
        shared_id = torch.full((tokens, 1), routed, dtype=dtypes.i32, device=DEVICE)
        topk_ids = torch.cat((routed_ids, shared_id), dim=1)
        topk_weights = torch.cat(
            (
                torch.rand((tokens, topk - 1), dtype=dtypes.fp32, device=DEVICE),
                torch.ones((tokens, 1), dtype=dtypes.fp32, device=DEVICE),
            ),
            dim=1,
        )
    return hidden, topk_ids, topk_weights


def bench_sort_quant_latency(
    tokens: int,
    experts: int,
    topk: int,
    model_dim: int = MODEL_DIM,
    block_m: int = BLOCK_M,
    iters: int = 300,
    warmup: int = 30,
    reps: int = 3,
) -> tuple[float, float, float, float]:
    """
    Benchmark the pre-GEMM plumbing variants.

    Returns:
        Tuple of (sort only, baseline sort+quant, fused, fused+sorted scale)
        latencies in microseconds.
    """
    hidden, topk_ids, topk_weights = make_inputs(tokens, experts, topk, model_dim)

    def baseline_sort():
        return moe_sorting(
            topk_ids, topk_weights, experts, model_dim, dtypes.bf16, block_m
        )

    def baseline_sort_quant():
        _ids, weights, _exp, num_valid, _buf = baseline_sort()
        fused_dynamic_mxfp4_quant_moe_sort(
            hidden,
            sorted_ids=_ids,
            num_valid_ids=num_valid,
            token_num=tokens,
            topk=topk,
            block_size=block_m,
            sorted_weights=weights,
        )

    # Output buffers for the fused kernel, laid out as aiter.fused_moe allocates them.
    route_count = tokens * topk
    active_experts = min(experts, route_count)
    max_sorted = (
        (route_count + active_experts * (block_m - 1) + block_m - 1)
        // block_m
        * block_m
    )
    sorted_ids = torch.empty(max_sorted, dtype=dtypes.i32, device=DEVICE)
    sorted_weights = torch.empty(max_sorted, dtype=dtypes.fp32, device=DEVICE)
    sorted_expert_ids = torch.empty(
        max_sorted // block_m, dtype=dtypes.i32, device=DEVICE
    )
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=DEVICE)
    moe_buf = torch.empty((tokens, model_dim), dtype=dtypes.bf16, device=DEVICE)
    act_quant = torch.empty((tokens, model_dim // 2), dtype=dtypes.fp4x2, device=DEVICE)
    act_scale = torch.empty(
        (tokens, model_dim // 32), dtype=dtypes.fp8_e8m0, device=DEVICE
    )

    def fused():
        aiter.mxfp4_moe_sort_quant_fwd(
            hidden,
            topk_ids,
            topk_weights,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            act_quant,
            act_scale,
            experts,
        )

    fused()
    torch.cuda.synchronize()

    def fused_sorted_scale():
        fused()
        mxfp4_moe_sort_fwd(
            act_scale,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=tokens,
            cols=model_dim,
        )

    return (
        time_graph(baseline_sort, iters, warmup, reps),
        time_graph(baseline_sort_quant, iters, warmup, reps),
        time_graph(fused, iters, warmup, reps),
        time_graph(fused_sorted_scale, iters, warmup, reps),
    )


def run_benchmark(args):
    """Run the plumbing benchmark across every route variant and token count."""
    print(f"arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"flydsl_sorting={_USE_FLYDSL_MOE_SORTING} ck_sorting={_USE_CK_MOE_SORTING}")

    results = []
    for experts, topk in args.routes:
        print(
            f"\n=== E={experts} topk={topk} model_dim={args.model_dim} "
            f"block_m={args.block_m} (us) ==="
        )
        header = (
            f"{'M':>8} {'sort':>10} {'sort+quant':>12} {'fused':>10} "
            f"{'fused+ss':>10} {'fused':>8} {'fused+ss':>10}"
        )
        print(header)
        print("-" * len(header))

        for tokens in args.tokens:
            sort_us, base_us, fused_us, ss_us = bench_sort_quant_latency(
                tokens,
                experts,
                topk,
                model_dim=args.model_dim,
                block_m=args.block_m,
                iters=args.iters,
                warmup=args.warmup,
                reps=args.reps,
            )
            results.append((experts, topk, tokens, sort_us, base_us, fused_us, ss_us))
            print(
                f"{tokens:>8} {sort_us:>10.2f} {base_us:>12.2f} {fused_us:>10.2f} "
                f"{ss_us:>10.2f} {base_us / fused_us:>7.2f}x "
                f"{base_us / ss_us:>9.2f}x"
            )

    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath: str, results: list):
    """Save benchmark results to CSV file."""
    path = Path(filepath)
    with open(path, "w") as f:
        f.write(
            "experts,topk,tokens,sort_us,baseline_sort_quant_us,fused_us,"
            "fused_sorted_scale_us,speedup_fused,speedup_fused_sorted_scale\n"
        )
        f.writelines(
            f"{experts},{topk},{tokens},{sort_us:.4f},{base_us:.4f},"
            f"{fused_us:.4f},{ss_us:.4f},"
            f"{base_us / fused_us:.4f},{base_us / ss_us:.4f}\n"
            for experts, topk, tokens, sort_us, base_us, fused_us, ss_us in results
        )
    print(f"\nResults saved to {path.resolve()}")


def _route(value: str) -> tuple[int, int]:
    experts, topk = value.split(",")
    return int(experts), int(topk)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 MoE Fused Sort+Quant",
        description=(
            "Benchmark the fused MoE route-sort + MXFP4 activation-quant kernel "
            "against the conventional two-launch sequence."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=TOKENS,
        help="Token counts to sweep.",
    )
    parser.add_argument(
        "--routes",
        type=_route,
        nargs="+",
        metavar="E,TOPK",
        default=ROUTES,
        help="Route variants as comma-separated expert/topk pairs.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        dest="model_dim",
        default=MODEL_DIM,
        help="Model hidden dimension.",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        dest="block_m",
        default=BLOCK_M,
        help="Sort block size M.",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file path for results.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of graph replays per timed rep.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warmup replays before each timed rep.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of timed reps; the best is reported.",
    )
    return parser.parse_args()


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
