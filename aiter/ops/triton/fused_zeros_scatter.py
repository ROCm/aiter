# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_zeros_scatter import (
    _fused_zeros_scatter_kernel
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

def fused_zeros_scatter(a, indices, scores):
    (row,col) = a.shape
    n_a = row * col
    (row_i, col_i) = indices.shape
    n_indices = row_i * col_i

    _fused_zeros_scatter_kernel[(1,)](
        a, indices, scores, n_indices, n_a, col
    )