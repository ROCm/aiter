# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The shapes every Triton GEMM test runs.

All GEMM tests call get_shapes(). Kernel limits come in as arguments, so each
kernel runs the shapes it supports. Shapes that pin one kernel's bug or config
stay in that kernel's test file.
"""

# (M, N, K)
SHAPES = (
    # Tails in every dimension.
    [(1, 1, 1), (3, 5, 2), (17, 200, 384), (31, 7168, 4608), (159, 17389, 597)]
    + [(1024, 1024, 1024), (4096, 4096, 4096)]
    # Decode to small prefill over model (N, K): DSR1 router, low N / high K
    # (split-K), high N / low K, GPT-OSS QKV.
    + [
        (m, n, k)
        for n, k in [(256, 7168), (1280, 8192), (8192, 1024), (5120, 2880)]
        for m in (1, 32, 128, 1024)
    ]
    # DSR1 MLA absorbed BMMs: W_UK (N=512, K=128) and W_UV (N=128, K=512).
    + [(m, n, k) for n, k in [(512, 128), (128, 512)] for m in (1, 32, 128)]
    # Prefill and a very large N (LL3 405B FC1).
    + [(8192, 5120, 2880), (256, 106496, 16384)]
)


def get_shapes(n_mult=1, k_mult=1, min_k=1, max_size=None, batch=None, n2=None):
    """SHAPES that fit a kernel.

    n_mult, k_mult, min_k: keep shapes with N % n_mult == 0, K % k_mult == 0, K >= min_k.
    max_size: keep shapes with M * N * K <= max_size.
    batch: return (B, M, N, K), B taken round-robin from this list.
    n2: return (M, N, N2, K) for two-output GEMMs, N2 taken round-robin from this list.
    """
    shapes = [
        (m, n, k)
        for m, n, k in SHAPES
        if n % n_mult == 0
        and k % k_mult == 0
        and k >= min_k
        and (max_size is None or m * n * k <= max_size)
    ]
    if batch is not None:
        return [(batch[i % len(batch)], m, n, k) for i, (m, n, k) in enumerate(shapes)]
    if n2 is not None:
        return [(m, n, n2[i % len(n2)], k) for i, (m, n, k) in enumerate(shapes)]
    return shapes
