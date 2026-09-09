# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Shared split-K plumbing for the gfx1250 MXFP8 x {MXFP8, MXFP4} ASM GEMMs.
#
# The 256x256 kernel can split K: split s accumulates only its own K range and writes
# D as (splitk, M, N) WITHOUT reducing it. aiter has no split-K reduce kernel (the
# a8w8 blockscale asm accumulates with atomics instead), so the planes are summed here
# with a plain torch reduce (fp32 accumulation, like the poc host golden) -- a single
# kernel, since at these sizes the reduce is launch-bound.
#
# The count comes from the .cu (choose_splitk), which owns tile selection and the
# constraint checks; this side only asks for it, sizes the buffer and reduces.

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_mxfp8fp4gemm_asm",
    fc_name="mxfp8fp4_gemm_splitk",
    ffi_type="ctypes",
)
def mxfp8fp4_gemm_splitk(
    M: int,
    N: int,
    K: int,
    b_is_fp4: int,
    a_preshuffle: int,
    kernelName: str | None = None,
) -> int: ...


def gemm_with_splitk(
    gemm,  # _mxfp8_mxfp8_gemm_asm / _mxfp8_mxfp4_gemm_asm
    A: Tensor,
    B: Tensor,
    ScaleA: Tensor,
    ScaleB: Tensor,
    b_is_fp4: int,
    a_preshuffle: int,
    kernelName: str,
    dtype: torch.dtype,
    splitk: int = 0,
) -> Tensor:
    """Run ``gemm`` under ``splitk`` (0 = the count the dispatch picks), returning
    D[M,N]. An explicit count is only checked against the kernel's hard constraints,
    so it may go deeper than the dispatch would."""
    M, N, K = A.shape[0], B.shape[0], A.shape[1]
    knl = kernelName if kernelName else None
    if splitk <= 0:
        splitk = mxfp8fp4_gemm_splitk(M, N, K, b_is_fp4, a_preshuffle, knl)
    out = torch.empty(
        (splitk, M, N) if splitk > 1 else (M, N), dtype=dtype, device=A.device
    )
    gemm(A, B, ScaleA, ScaleB, out, knl, a_preshuffle, splitk)
    if splitk == 1:
        return out
    return torch.sum(out, dim=0, dtype=dtype)
