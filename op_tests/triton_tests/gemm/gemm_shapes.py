# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The shapes every Triton GEMM test runs.

All GEMM tests call get_shapes(). Kernel limits come in as arguments, so each
kernel runs the shapes it supports. Shapes that pin one kernel's bug or config
stay in that kernel's test file.
"""

# (M, N, K)
EDGE_CASES = [
    (1, 1, 1),
    (3, 5, 2),
    (17, 200, 384),
    (31, 7168, 4608),
    (159, 17389, 597),
]

# (M, N, K), M <= 256
DECODE = [
    # DSR1 router
    (1, 256, 7168),
    (32, 256, 7168),
    (128, 256, 7168),
    # Low N, high K (split-K)
    (1, 1280, 8192),
    (32, 1280, 8192),
    (128, 1280, 8192),
    # High N, low K
    (1, 8192, 1024),
    (32, 8192, 1024),
    (128, 8192, 1024),
    # GPT-OSS QKV
    (1, 5120, 2880),
    (32, 5120, 2880),
    (128, 5120, 2880),
    # DSR1 MLA absorbed BMMs: W_UK (N=512, K=128) and W_UV (N=128, K=512)
    (1, 512, 128),
    (32, 512, 128),
    (128, 512, 128),
    (1, 128, 512),
    (32, 128, 512),
    (128, 128, 512),
    # LL3 405B FC1 (very large N)
    (256, 106496, 16384),
]

# (M, N, K), M >= 1024
PREFILL = [
    (1024, 256, 7168),
    (1024, 1280, 8192),
    (1024, 8192, 1024),
    (1024, 5120, 2880),
    (1024, 1024, 1024),
    (4096, 4096, 4096),
    (8192, 5120, 2880),
]

SHAPES = EDGE_CASES + DECODE + PREFILL


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
