# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""BF16 TN GEMM Triton kernels for Kimi-K3 serving shapes on gfx950.

Tile parameters were determined by Evolve-Kernel island search (2026-08-05)
from Triton-compiled baselines.  The winning tile configs beat hipBLASLt at
the Kimi-K3 decode serving shapes:

  M=64,  N=7168, K=7168: -31 % vs hipBLASLt  (BM=64, BN=32, BK=128, 4w, 3st)
  M=128, N=7168, K=7168: -7 % vs hipBLASLt   (BM=128, BN=64, BK=128, 4w, 3st)

The corresponding hand-tuned .amdgcn/.co sources are in hsa/gfx950/bf16gemm/
for auditability; dispatch here uses Triton so that the kernarg preload ABI
(gfx950 SGPR s2..s15) is handled correctly by the Triton runtime.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

_HSA_DIR = Path(__file__).resolve().parents[4] / "hsa" / "gfx950" / "bf16gemm"

# Tuned configs found by Evolve-Kernel (2026-08-05)
_CONFIGS: dict[int, dict] = {
    64: dict(BLOCK_M=64, BLOCK_N=32, BLOCK_K=128, num_warps=4, num_stages=3),
    128: dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
}


@functools.lru_cache(maxsize=None)
def _rocm_arch(device_idx: int) -> str:
    props = torch.cuda.get_device_properties(device_idx)
    arch = getattr(props, "gcnArchName", "")
    return arch.split(":", 1)[0]


def is_kimi_k3_front_gemm_bf16_supported(
    M: int, device: Optional[torch.device] = None
) -> bool:
    """Return True when the tuned kernel for M is available on this GPU."""
    if not torch.cuda.is_available():
        return False
    if M not in _CONFIGS:
        return False
    try:
        idx = torch.cuda.current_device() if device is None else device.index or 0
        return _rocm_arch(idx) == "gfx950"
    except Exception:
        return False


@triton.jit
def _bf16_tn_gemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    A_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(B_ptrs, mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(a, b.T)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    od = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        C + oc[:, None] * stride_cm + od[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=(oc[:, None] < M) & (od[None, :] < N),
    )


def front_gemm_bf16_asm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
    """BF16 TN GEMM: C = A @ B.T  (A[M,K], B[N,K] -> C[M,N]).

    Uses the Evolve-Kernel tuned tile config for M in {64, 128} on gfx950.
    Caller must guard with is_kimi_k3_front_gemm_bf16_supported(M).

    Args:
        A: [M, K] bf16, row-major.
        B: [N, K] bf16, row-major (kernel computes A @ B.T internally).
        C: [M, N] bf16, output written in-place.
    """
    M, K = A.shape
    N = B.shape[0]
    cfg = _CONFIGS[M]
    BM, BN = cfg["BLOCK_M"], cfg["BLOCK_N"]
    grid = (math.ceil(N / BN), math.ceil(M / BM))
    _bf16_tn_gemm_kernel[grid](
        A,
        B,
        C,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        **cfg,
    )
