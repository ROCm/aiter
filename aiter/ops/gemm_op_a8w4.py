# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 (mi400) MXFP8 x MXFP4 GEMM (a8w4) -- ASM, kernarg preload mode.
# A (activation) is mxfp8 (e4m3, 1 byte/elem); B (weight) is mxfp4 (e2m1,
# 2 elems/byte). Both operands carry OCP MX e8m0 block scales (block=32). The
# kernel variant is auto-selected by the .cu heuristic unless an explicit
# kernelName is given. See csrc/py_itfs_cu/asm_mxfp8fp4gemm_mi400.cu.

from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes


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
    kernelName: Optional[str] = None,
    a_preshuffle: int = 1,
) -> None: ...


def gemm_a8w4_mxfp8(
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
