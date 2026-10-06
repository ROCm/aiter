# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated: ``gemm_a6w6_fly_asm`` is ``gemm_a6w6_tilescale`` (``aiter.ops.gemm_op_tilescale``). Operands are
tilescale FP6 buffers and scale slabs; the kernels moved into the tilescale family unchanged."""

from torch import Tensor

from .gemm_op_tilescale import gemm_a6w6_tilescale


def gemm_a6w6_fly_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    bias: Tensor | None = None,
) -> Tensor:
    return gemm_a6w6_tilescale(A, B, A_scale, B_scale, out, K, bias)
