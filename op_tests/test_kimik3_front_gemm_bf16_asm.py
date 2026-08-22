# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for Kimi-K3 front GEMM BF16 ASM kernels.

Tests the two hand-tuned AMDGCN ASM BF16 TN GEMM kernels against torch.mm
at the Kimi-K3 TP=8 serving shapes where they are tuned:

  M=64,  N=7168, K=7168  (CONC=64 decode batch)   -> -31 % vs hipBLASLt
  M=128, N=7168, K=7168  (CONC=128 decode batch)  -> -7 % vs hipBLASLt

Both kernels are gfx950 only.  The dispatch uses direct hipModuleLaunchKernel
to match the Triton kernarg ABI (dense layout, no CK p2/p3 padding).

Run:
    python op_tests/test_kimik3_front_gemm_bf16_asm.py
"""

import math
import unittest

import torch

from aiter.test_common import perftest

NUM_ITERS = 200

# Kimi-K3 TP=8 front GEMM shapes: A[M,K] @ B[N,K].T -> C[M,N], BF16 TN layout
_SHAPES = [
    (64, 7168, 7168),
    (128, 7168, 7168),
]


@perftest(num_iters=NUM_ITERS)
def _bench_torch(A, B):
    return torch.mm(A, B.t())


@perftest(num_iters=NUM_ITERS)
def _bench_asm(A, B, C):
    from aiter.ops.flydsl.kernels.kimi_k3_front_gemm_bf16 import front_gemm_bf16_asm

    front_gemm_bf16_asm(A, B, C)
    return C


def _snr_db(ref, cand):
    diff = (ref.float() - cand.float()).norm()
    sig = ref.float().norm()
    if sig == 0:
        return float("inf")
    return 20 * math.log10((sig / (diff + 1e-12)).item())


def _gfx():
    if not torch.cuda.is_available():
        return ""
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", "").split(":", 1)[0]


def _is_gfx950():
    return _gfx() == "gfx950"


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
@unittest.skipIf(not _is_gfx950(), "gfx950 (MI355X) required")
class TestKimik3FrontGemmBf16Asm(unittest.TestCase):
    def _make_inputs(self, M, N, K, seed=42):
        torch.manual_seed(seed)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
        C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        return A, B, C

    def _test_shape(self, M, N, K):
        from aiter.ops.flydsl.kernels.kimi_k3_front_gemm_bf16 import (
            front_gemm_bf16_asm,
            is_kimi_k3_front_gemm_bf16_supported,
        )

        if not is_kimi_k3_front_gemm_bf16_supported(M):
            self.skipTest(f"M={M} ASM kernel not available")

        A, B, C = self._make_inputs(M, N, K)
        ref = torch.mm(A, B.t())
        front_gemm_bf16_asm(A, B, C)
        torch.cuda.synchronize()

        snr = _snr_db(ref, C)
        self.assertGreater(snr, 30.0, f"M={M}: SNR {snr:.1f} dB below threshold")

    def test_m64_correctness(self):
        self._test_shape(64, 7168, 7168)

    def test_m128_correctness(self):
        self._test_shape(128, 7168, 7168)


def _run_perf():
    from aiter.ops.flydsl.kernels.kimi_k3_front_gemm_bf16 import (
        is_kimi_k3_front_gemm_bf16_supported,
    )

    if not _is_gfx950():
        print("SKIP: gfx950 not detected")
        return

    print("\nKimi-K3 Front GEMM BF16 ASM  [gfx950, BF16 TN layout]")
    print("-" * 65)
    print(f"{'Shape':>20} {'torch (us)':>12} {'ASM (us)':>10} {'Speedup':>10}")
    print("-" * 65)

    for M, N, K in _SHAPES:
        if not is_kimi_k3_front_gemm_bf16_supported(M):
            print(f"{'M='+str(M):>20}  SKIP (kernel not available)")
            continue
        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
        B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
        C = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        _, torch_us = _bench_torch(A, B)
        _, asm_us = _bench_asm(A, B, C)

        speedup = (torch_us - asm_us) / torch_us * 100
        shape_str = f"M={M} N={N} K={K}"
        print(f"{shape_str:>20} {torch_us:>12.2f} {asm_us:>10.2f} {speedup:>+9.1f}%")

    print("-" * 65)


if __name__ == "__main__":
    _run_perf()
    unittest.main(exit=False)
