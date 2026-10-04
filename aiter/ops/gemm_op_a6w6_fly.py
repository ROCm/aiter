# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A6W6 on assembly ports of FlyDSL's MXFP6 GEMM (``hsa/gfx950/f6flygemm``): one code object per
(M, N, K). Operands are MXFP6 in K128-blocked planes, one uint8 buffer per operand (C0 ``[rows/16, K/128, 16, 64]``
then C1 ``[rows/32, K/128, 32, 32]``, ``rows*K*3/4`` bytes), and E8M0 scales in FlyDSL's prepacked MXFP4 layout at
the 256 tile with no B interleave (``rows*K/32`` bytes). An optional bf16 ``bias[N]`` is added in the store epilogue
as fp32(acc) + fp32(bias) with one rounding (A6W6's bias epilogue); it has its own code object per shape.
"""

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_gemm_a6w6_fly_asm", fc_name="gemm_a6w6_fly_asm", ffi_type="ctypes")
def _gemm_a6w6_fly_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    bias: Tensor | None = None,
) -> None: ...


def gemm_a6w6_fly_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    bias: Tensor | None = None,
) -> Tensor:
    """``out[M, N] = A @ B^T (+ bias)``; raises if no code object exists for exactly (M, N, K) and the bias choice."""
    if out.ndim != 2:
        raise ValueError(f"gemm_a6w6_fly_asm expects a 2D output, got {out.ndim}D")
    _gemm_a6w6_fly_asm(A, B, A_scale, B_scale, out, K, bias)
    return out
