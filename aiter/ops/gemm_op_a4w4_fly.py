# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated: ``gemm_a4w4_fly_asm`` is ``gemm_a4w4_tilescale`` (``aiter.ops.gemm_op_tilescale``) with role B's
scale interleave taken from the kernel manifest. The kernels moved into the tilescale family unchanged."""

from torch import Tensor

from .gemm_op_tilescale import a4w4_b_ilv, gemm_a4w4_tilescale


def gemm_a4w4_fly_asm(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor, out: Tensor, K: int) -> Tensor:
    M, N = out.shape
    return gemm_a4w4_tilescale(A, B, A_scale, B_scale, out, K, a4w4_b_ilv(M, N, K))
