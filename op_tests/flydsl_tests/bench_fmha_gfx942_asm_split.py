#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Matched wall-clock benchmark for production ASM versus unified split-K."""

import argparse
import math
import os
import statistics
import time

import triton  # noqa: F401  # Must precede torch on this ROCm environment.
import torch

from aiter.ops.mha import fmha_v3_varlen_fwd


def production_asm(q, k, v, cu_q, cu_k, scale):
    out, _, _, _ = fmha_v3_varlen_fwd(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q.shape[0],
        k.shape[0],
        0,
        0.0,
        scale,
        0.0,
        False,
        False,
        -1,
        -1,
        False,
        False,
        1,
    )
    return out


def measure(fn, warmup: int, samples: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1e3)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sq", type=int, default=4096)
    parser.add_argument("--sk", type=int, default=42700)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    generator = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(
        args.sq, args.heads, 192, dtype=torch.bfloat16, generator=generator
    ).cuda()
    k = torch.randn(
        args.sk, args.heads, 192, dtype=torch.bfloat16, generator=generator
    ).cuda()
    v = torch.randn(
        args.sk, args.heads, 128, dtype=torch.bfloat16, generator=generator
    ).cuda()
    cu_q = torch.tensor([0, args.sq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, args.sk], dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(192)

    if args.splits != 3:
        raise ValueError("the integrated ticket path is fixed to three KV splits")
    baseline = lambda: production_asm(q, k, v, cu_q, cu_k, scale)
    treatment = baseline
    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = "0"
    reference = baseline()
    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = "1"
    actual = treatment()
    cosine_difference = 1.0 - 2.0 * (reference.double() * actual.double()).sum().item() / (
        reference.double().square().sum().item()
        + actual.double().square().sum().item()
    )
    if cosine_difference >= 1e-4:
        raise RuntimeError(f"correctness gate failed: cosine difference={cosine_difference}")

    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = "0"
    baseline_ms = measure(baseline, args.warmup, args.samples)
    os.environ["AITER_FMHA_HD192_SPLIT_KV"] = "1"
    split_ms = measure(treatment, args.warmup, args.samples)
    flop = args.heads * 2 * args.sq * args.sk * (192 + 128)
    baseline_median = statistics.median(baseline_ms)
    split_median = statistics.median(split_ms)
    print(f"gfx942 Sq={args.sq} Sk={args.sk} H={args.heads} S={args.splits}")
    print(
        "production_asm "
        f"median={baseline_median:.3f}ms min={min(baseline_ms):.3f}ms "
        f"max={max(baseline_ms):.3f}ms tflops={flop / baseline_median / 1e9:.1f}"
    )
    print(
        "unified_split "
        f"median={split_median:.3f}ms min={min(split_ms):.3f}ms "
        f"max={max(split_ms):.3f}ms tflops={flop / split_median / 1e9:.1f}"
    )
    print(
        f"speedup={(baseline_median / split_median):.3f}x "
        f"latency_reduction={(1.0 - split_median / baseline_median) * 100:.1f}% "
        f"cosine_difference={cosine_difference:.3e}"
    )


if __name__ == "__main__":
    main()
