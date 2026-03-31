# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Microbenchmark: fused grouped_topk_decode (1 kernel) vs
separate grouped_topk + moe_sorting (2 kernels) for M=1 decode shapes.
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.topk import biased_grouped_topk_hip, grouped_topk
from aiter.fused_moe import moe_sorting
import argparse

BLOCK_SIZE_M = 32


def bench_separate(gating_output, topk, num_experts, num_expert_group,
                   topk_group, block_size, renormalize, model_dim, dtype,
                   correction_bias=None):
    device = gating_output.device
    M = gating_output.shape[0]
    topk_weights = torch.empty(M, topk, dtype=dtypes.fp32, device=device)
    topk_ids = torch.empty(M, topk, dtype=dtypes.i32, device=device)

    if correction_bias is not None:
        def fn():
            biased_grouped_topk_hip(
                gating_output, correction_bias, topk_weights, topk_ids,
                num_expert_group, topk_group, renormalize,
            )
            moe_sorting(topk_ids, topk_weights, num_experts, model_dim, dtype, block_size)
    else:
        def fn():
            grouped_topk(
                gating_output, topk_weights, topk_ids,
                num_expert_group, topk_group, renormalize,
            )
            moe_sorting(topk_ids, topk_weights, num_experts, model_dim, dtype, block_size)
    return fn


def bench_fused_hip(gating_output, topk, num_experts, num_expert_group,
                    topk_group, block_size, renormalize, model_dim, dtype,
                    correction_bias=None):
    device = gating_output.device
    M = gating_output.shape[0]
    max_num_tokens_padded = topk + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=dtype, device=device)

    bias_tensor = correction_bias if correction_bias is not None else torch.empty(0, dtype=dtype, device=device)

    def fn():
        aiter.grouped_topk_moe_sorting(
            gating_output,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_expert_group,
            topk_group,
            topk,
            block_size,
            renormalize,
            False,
            bias_tensor,
            1.0,
        )
    return fn


def main():
    parser = argparse.ArgumentParser(description="Benchmark grouped_topk_decode")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    device = "cuda"
    model_dim = 7168
    M = 1

    configs = [
        # (E, topk, num_expert_group, topk_group, use_bias)
        (256, 8, 8, 4, True),   # DeepSeek-V3
        (256, 8, 8, 4, False),
        (128, 8, 8, 4, True),
        (128, 8, 8, 4, False),
    ]

    print(f"{'E':>5}  {'topk':>4}  {'g':>2}  {'tg':>2}  {'bias':>5}"
          f"  {'separate_us':>12}  {'fused_us':>12}  {'speedup':>8}")
    print("-" * 75)

    for E, topk, g, tg, use_bias in configs:
        gating_output = torch.randn(M, E, dtype=dtype, device=device)
        correction_bias = torch.randn(E, dtype=dtype, device=device) if use_bias else None

        fn_sep = bench_separate(
            gating_output, topk, E, g, tg, BLOCK_SIZE_M, True, model_dim, dtype,
            correction_bias,
        )
        fn_fused = bench_fused_hip(
            gating_output, topk, E, g, tg, BLOCK_SIZE_M, True, model_dim, dtype,
            correction_bias,
        )

        for _ in range(args.warmup):
            fn_sep()
            fn_fused()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            fn_sep()
        end.record()
        torch.cuda.synchronize()
        sep_us = start.elapsed_time(end) * 1000 / args.iters

        start.record()
        for _ in range(args.iters):
            fn_fused()
        end.record()
        torch.cuda.synchronize()
        fused_us = start.elapsed_time(end) * 1000 / args.iters

        speedup = sep_us / fused_us if fused_us > 0 else float("inf")
        bias_str = "True" if use_bias else "False"
        print(f"{E:>5}  {topk:>4}  {g:>2}  {tg:>2}  {bias_str:>5}"
              f"  {sep_us:>12.2f}  {fused_us:>12.2f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
