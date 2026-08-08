# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance test for the FlyDSL gemm_decode BF16 kernel.

C[M, N] = A[M, K] @ B[N, K]^T  (warp-per-scalar, small-M decode GEMM)

Shapes cover DeepSeek-V3 projection layers at typical decode batch sizes.
"""

import argparse

import torch

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.gemm_decode import gemm_decode_bf16
from aiter.test_common import checkAllclose, perftest

torch.set_default_device("cuda")


@perftest()
def run_ref(A, B):
    return (A.float() @ B.float().T).bfloat16()


@perftest()
def run_gemm_decode(A, B, C, M, N, K):
    stream = fx.Stream(None)
    gemm_decode_bf16(A, B, C, M, N, K, stream=stream)
    return C


def test_gemm_decode(M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(N, K, dtype=torch.bfloat16)
    C = torch.zeros(M, N, dtype=torch.bfloat16)

    ref, avg_ref = run_ref(A, B)
    out, avg_kernel = run_gemm_decode(A, B, C, M, N, K)

    bytes_transferred = (M * K + N * K + M * N) * 2  # BF16 = 2 bytes
    bw_gbs = bytes_transferred / (avg_kernel * 1e-6) / 1e9

    msg = (
        f"[perf] M={M} N={N} K={K}: "
        f"ref avg: {avg_ref:<8.2f} us, "
        f"kernel avg: {avg_kernel:<8.2f} us, "
        f"BW: {bw_gbs:.0f} GB/s"
    )
    checkAllclose(out, ref, atol=0.5, rtol=0.1, msg=msg)


parser = argparse.ArgumentParser(
    description="Test FlyDSL gemm_decode BF16 kernel performance and correctness",
)
parser.add_argument(
    "-M",
    type=int,
    nargs="*",
    default=[1, 2, 3, 4],
    help="Decode batch sizes (number of rows). e.g.: -M 1 2 3 4",
)
parser.add_argument(
    "-N",
    type=int,
    nargs="?",
    default=16384,
    help="Output dimension (number of columns). e.g.: -N 16384",
)
parser.add_argument(
    "-K",
    type=int,
    nargs="?",
    default=7168,
    help="Input dimension. e.g.: -K 7168",
)
args = parser.parse_args()

for M in args.M:
    test_gemm_decode(M, args.N, args.K)
