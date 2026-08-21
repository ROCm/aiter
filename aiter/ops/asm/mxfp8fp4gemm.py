# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 F8GEMM -- ASM, kernarg preload mode. benchmark-only, private.
#   a8w4: A mxfp8 (e4m3) x B mxfp4 (e2m1), OCP MX e8m0 block scales (block=32).
#   a8w8: A mxfp8 (e4m3) x B mxfp8 (e4m3), OCP MX e8m0 block scales (block=32).
# Kernel variant is auto-selected by the .cu heuristic unless kernelName given.
# See csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu.

import torch
from torch import Tensor

from aiter.jit.core import compile_ops
from aiter.utility import dtypes


@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8_mxfp4_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp8_mxfp4_gemm_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
    B: Tensor,  # B:[N, K/2] mxfp4 e2m1 (always preshuffled)
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0 (shuffled)
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0 (shuffled)
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str | None = None,
    a_preshuffle: int = 1,
) -> None: ...


def _gemm_a8w4_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3
    B: Tensor,  # B:[N, K/2] mxfp4 e2m1
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Tensor:
    """gfx1250 MXFP8 (activation) x MXFP4 (weight) GEMM (a8w4). D[M,N] bf16 =
    A @ B^T with e8m0 block scales. Kernel auto-selected from M/N/K unless
    ``kernelName`` is given.

    K is taken from A (mxfp8, ``A.shape[1] == K``); B is packed mxfp4 with
    ``B.shape == [N, K/2]``."""
    M = A.shape[0]
    N = B.shape[0]
    out = torch.empty((M, N), dtype=dtype, device=A.device)
    _mxfp8_mxfp4_gemm_asm(
        A,
        B,
        ScaleA,
        ScaleB,
        out,
        kernelName if kernelName else None,
        int(bool(a_preshuffle)),
    )
    return out


@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8_mxfp8_gemm_asm",
    ffi_type="ctypes",
)
def _mxfp8_mxfp8_gemm_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3 (preshuffled if a_preshuffle=1)
    B: Tensor,  # B:[N, K]   mxfp8 e4m3 (always preshuffled)
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0 (shuffled)
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0 (shuffled)
    out: Tensor,  # Out:[M, N] bf16
    kernelName: str | None = None,
    a_preshuffle: int = 1,
) -> None: ...


def _gemm_a8w8_asm(
    A: Tensor,  # A:[M, K]   mxfp8 e4m3
    B: Tensor,  # B:[N, K]   mxfp8 e4m3
    ScaleA: Tensor,  # ScaleA:[M, K/32] e8m0
    ScaleB: Tensor,  # ScaleB:[N, K/32] e8m0
    dtype: torch.dtype = dtypes.bf16,
    a_preshuffle: bool = True,
    kernelName: str = "",
) -> Tensor:
    """gfx1250 MXFP8 x MXFP8 GEMM (a8w8). D[M,N] bf16 = A @ B^T with e8m0 block
    scales. Kernel auto-selected from M/N/K unless ``kernelName`` is given."""
    M = A.shape[0]
    N = B.shape[0]
    out = torch.empty((M, N), dtype=dtype, device=A.device)
    _mxfp8_mxfp8_gemm_asm(
        A,
        B,
        ScaleA,
        ScaleB,
        out,
        kernelName if kernelName else None,
        int(bool(a_preshuffle)),
    )
    return out
