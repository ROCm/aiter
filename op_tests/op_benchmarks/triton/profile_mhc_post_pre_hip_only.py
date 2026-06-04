# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse

import torch
import triton

import aiter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HIP-only mHC post+pre profiling harness"
    )
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-n", type=int, default=4)
    parser.add_argument("-C", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert args.n == 4, "HIP mhc_post_pre currently supports n == 4"
    assert torch.cuda.is_available(), "CUDA/ROCm device is required"

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Input setup is intentionally outside the measured callable. The tensor
    # values are bounded to keep sigmoid/Sinkhorn numerically stable while
    # avoiding Triton/reference/checkAllclose work in the profiling window.
    layer_input = torch.randn(args.M, args.C, dtype=dtype, device=device) * 0.1
    residual = torch.randn(args.M, args.n, args.C, dtype=dtype, device=device) * 0.1
    post_mix = torch.sigmoid(
        torch.randn(args.M, args.n, 1, dtype=torch.float32, device=device)
    )
    comb_logits = torch.randn(
        args.M, args.n, args.n, dtype=torch.float32, device=device
    )
    comb_mix = torch.softmax(comb_logits, dim=-1)
    fn = torch.randn(
        args.n * 2 + args.n * args.n,
        args.n * args.C,
        dtype=torch.float32,
        device=device,
    ) * 1e-4
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(args.n * 2 + args.n * args.n, dtype=torch.float32, device=device)

    def run_once():
        return aiter.mhc_post_pre(
            layer_input,
            residual,
            post_mix,
            comb_mix,
            fn,
            hc_scale,
            hc_base,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=20,
        )

    # Trigger JIT and cache before benchmark/profiler repetitions.
    out = run_once()
    torch.cuda.synchronize()
    for _ in range(args.warmup):
        out = run_once()
    torch.cuda.synchronize()

    def run_profiled():
        result = None
        for _ in range(args.iters):
            result = run_once()
        return result

    ms = triton.testing.do_bench(run_profiled, warmup=0, rep=args.rep)
    # Touch outputs so the compiler/runtime cannot elide the call. This is not
    # in the timed region because do_bench synchronizes around the callable.
    torch.cuda.synchronize()
    checksum = sum(float(t.float().sum().item()) for t in out)
    print(
        f"M={args.M} n={args.n} C={args.C} iters={args.iters}: "
        f"hip={ms / args.iters:.4f} ms/iter checksum={checksum:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
