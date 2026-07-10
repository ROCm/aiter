# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for the FlyDSL gemm_decode BF16 kernel.

C[M, N] = A[M, K] @ B[N, K]^T  (warp-per-scalar, small-M decode GEMM)

Shapes cover DeepSeek-V3 projection layers at typical decode batch sizes.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flydsl")
import flydsl.expr as fx  # noqa: E402
from aiter.ops.flydsl.kernels.gemm_decode import gemm_decode_bf16  # noqa: E402

torch.set_default_device("cuda")

# (M, N, K) -- decode-typical shapes: small M, large N and K
SHAPES = [
    (1, 16384, 7168),  # DeepSeek-V3 Q/K/V/O-proj, bs=1
    (2, 16384, 7168),  # bs=2
    (3, 16384, 7168),  # bs=3
    (4, 16384, 7168),  # bs=4
]

WARMUP = 50
REPEAT = 200


def _ref(A, B):
    """Float32 reference: C = A @ B^T cast back to BF16."""
    return (A.float() @ B.float().T).bfloat16()


def _bench(fn, A, B, C, M, N, K):
    """Returns median kernel time in microseconds."""
    stream = fx.Stream(None)
    for _ in range(WARMUP):
        fn(A, B, C, M, N, K, stream=stream)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(REPEAT):
        fn(A, B, C, M, N, K, stream=stream)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / REPEAT  # us


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_gemm_decode_correctness(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)
    C = torch.zeros(M, N, dtype=torch.bfloat16)

    gemm_decode_bf16(A, B, C, M, N, K)
    torch.cuda.synchronize()

    ref = _ref(A, B)
    assert torch.allclose(
        C, ref, atol=0.5, rtol=0.1
    ), f"gemm_decode M={M} N={N} K={K}: max diff {(C - ref).abs().max():.4f}"


@pytest.mark.parametrize("M,N,K", SHAPES)
def test_gemm_decode_performance(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)
    C = torch.zeros(M, N, dtype=torch.bfloat16)

    us = _bench(gemm_decode_bf16, A, B, C, M, N, K)

    bytes_transferred = (M * K + N * K + M * N) * 2  # BF16 = 2 bytes
    bw_gbs = bytes_transferred / (us * 1e-6) / 1e9
    print(f"\n[perf] M={M} N={N} K={K}: {us:.1f} us  {bw_gbs:.0f} GB/s")

    # Must complete in reasonable time (< 1 ms for these shapes)
    assert us < 1000, f"gemm_decode too slow: {us:.1f} us"
