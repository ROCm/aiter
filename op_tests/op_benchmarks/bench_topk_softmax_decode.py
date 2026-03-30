# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Microbenchmark: fused topk_softmax_decode (1 kernel) vs separate fused_topk + moe_sorting (2 kernels)
for M=1 decode shapes.
"""

import torch
import aiter
from aiter import dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.test_common import run_perftest
import argparse


BLOCK_SIZE_M = 32


def bench_separate(gating_output, topk, num_experts, block_size, renormalize, model_dim, dtype):
    """Benchmark the separate fused_topk + moe_sorting path."""
    device = gating_output.device
    M = gating_output.shape[0]
    hidden_states = torch.randn(M, 1, dtype=gating_output.dtype, device=device)

    def fn():
        topk_weights, topk_ids = fused_topk(
            hidden_states, gating_output, topk, renormalize
        )
        moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            dtype,
            block_size,
        )

    return fn


def bench_fused(gating_output, topk, num_experts, block_size, renormalize, model_dim, dtype):
    """Benchmark the fused topk_softmax_decode path."""
    device = gating_output.device
    M = gating_output.shape[0]
    max_num_tokens_padded = topk + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)

    def fn():
        aiter.topk_softmax_decode(
            gating_output,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts,
            topk,
            block_size,
            renormalize,
        )

    return fn


def main():
    parser = argparse.ArgumentParser(description="Benchmark topk_softmax_decode")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    device = "cuda"
    model_dim = 7168
    M = 1

    configs = [
        (128, 8),
        (128, 6),
        (128, 4),
        (64, 8),
        (64, 4),
        (32, 4),
    ]

    print(f"{'E':>5}  {'topk':>5}  {'separate_us':>12}  {'fused_us':>12}  {'speedup':>8}")
    print("-" * 55)

    for E, topk in configs:
        gating_output = torch.randn(M, E, dtype=dtype, device=device)

        fn_sep = bench_separate(
            gating_output, topk, E, BLOCK_SIZE_M, True, model_dim, dtype
        )
        fn_fused = bench_fused(
            gating_output, topk, E, BLOCK_SIZE_M, True, model_dim, dtype
        )

        # Warmup
        for _ in range(args.warmup):
            fn_sep()
            fn_fused()
        torch.cuda.synchronize()

        # Benchmark separate
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn_sep()
        end.record()
        torch.cuda.synchronize()
        sep_us = start.elapsed_time(end) * 1000 / args.iters

        # Benchmark fused
        start.record()
        for _ in range(args.iters):
            fn_fused()
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) * 1000 / args.iters

        speedup = sep_us / fused_us if fused_us > 0 else float("inf")
        print(f"{E:>5}  {topk:>5}  {sep_us:>12.2f}  {fused_us:>12.2f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
