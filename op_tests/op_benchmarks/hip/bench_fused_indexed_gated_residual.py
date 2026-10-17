# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the HIP fused residual + gate[indices] * x against the eager
expression, which materialises the gathered gate and the product at [M, N]."""

import argparse

import torch

from aiter import fused_indexed_gated_residual

# [M, N] arrays moved: fused reads residual and x and writes out; eager also
# writes the gathered gate, and writes and re-reads the product.
_ARRAYS = {"fused": 3, "eager": 7}
# MiniMax-H3: hidden 5376; 114660 tokens for a 15 s 768p video, 57330 per rank
# with two-way sequence parallelism.
_TOKENS = (1024, 4096, 16384, 57330, 114660)


def _event_time_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def make(M, N, G, scattered, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(M, N, generator=gen, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, N, generator=gen, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(G, N, generator=gen, device="cuda", dtype=torch.bfloat16)
    if scattered:
        indices = torch.randint(0, G, (M,), generator=gen, device="cuda")
    else:
        indices = torch.arange(M, device="cuda") * G // M
    return residual, x, gate, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, nargs="*", default=list(_TOKENS))
    parser.add_argument("-N", type=int, default=5376, help="Hidden size")
    parser.add_argument("-G", type=int, default=9, help="Gate table rows")
    parser.add_argument(
        "--scattered",
        action="store_true",
        help="Random index per token instead of runs of one index",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    print(
        f"{'M':>8} {'N':>6} {'fused_us':>10} {'eager_us':>10} {'speedup':>8} {'fused_GB/s':>11}"
    )
    for M in args.M:
        residual, x, gate, indices = make(M, args.N, args.G, args.scattered)
        out = torch.empty_like(residual)
        fused = _event_time_us(
            lambda: fused_indexed_gated_residual(residual, x, gate, indices, out=out),
            args.warmup,
            args.iters,
        )
        eager = _event_time_us(
            lambda: residual + gate[indices] * x, args.warmup, args.iters
        )
        gbps = _ARRAYS["fused"] * M * args.N * residual.element_size() / fused * 1e-3
        print(
            f"{M:>8} {args.N:>6} {fused:>10.1f} {eager:>10.1f} {eager / fused:>7.2f}x {gbps:>11.1f}"
        )


if __name__ == "__main__":
    main()
