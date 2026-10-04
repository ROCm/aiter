# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A4W4 on assembly ports of FlyDSL's MXFP4 GEMM (``hsa/gfx950/f4flygemm``), 256-wide N tile only: one code
object per (M, N, K). Operands are plain MXFP4 rows ``[rows, K/2]`` uint8 and E8M0 scales in FlyDSL's prepacked
layout (``rows*K/32`` bytes; B packed for the 256-wide tile with the manifest's ``b_ilv``).
"""

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_gemm_a4w4_fly_asm", fc_name="gemm_a4w4_fly_asm", ffi_type="ctypes")
def _gemm_a4w4_fly_asm(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, out: Tensor, K: int
) -> None: ...


def gemm_a4w4_fly_asm(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, out: Tensor, K: int
) -> Tensor:
    """``out[M, N] = A @ B^T``; raises if no code object exists for exactly (M, N, K)."""
    if out.ndim != 2:
        raise ValueError(f"gemm_a4w4_fly_asm expects a 2D output, got {out.ndim}D")
    _gemm_a4w4_fly_asm(A, B, A_scale, B_scale, out, K)
    return out
