# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from ..jit.core import (
    compile_ops,
    AITER_CORE_DIR,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num


def gemm_a4w4(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    bias: Tensor,  # bias:[1, N] f32
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
) -> torch.Tensor:
    """
    A4W4 GEMM kernel for AMD GPUs.
    This function is a wrapper for the A4W4 GEMM kernel.
    It is used to perform matrix multiplication with 4-bit quantization.
    """

    # Get the number of compute units
    cu_num = get_cu_num()

    # Load the A4W4 GEMM kernel
    func = gemm_a4w4_asm

    return func(A, B, A_scale, B_scale, out, bias, alpha, beta)


@compile_ops("module_gemm_a4w4_asm")
def gemm_a4w4_asm(
    A: Tensor,  # A:[M, K/2] f4x2
    B: Tensor,  # B:[N, K/2] f4x2
    A_scale: Tensor,  # A_scale:[M, K/32] e8m0 paded
    B_scale: Tensor,  # B_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
    bias: Tensor,  # bias:[1, N] f32
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
) -> torch.Tensor: ...


@compile_ops("module_gemm_a4w4_blockscale")
def gemm_a4w4_blockscale(
    XQ: Tensor,  # XQ:[M, K/2] f4x2
    WQ: Tensor,  # WQ:[N, K/2] f4x2
    x_scale: Tensor,  # x_scale:[M, K/32] e8m0 paded
    w_scale: Tensor,  # w_scale:[N, K/32] e8m0 paded
    out: Tensor,  # Out:[M, N] bf16
): ...
