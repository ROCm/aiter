# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.gluon._gluon_kernels.softmax import _softmax_kernel_online

_LOGGER = AiterTritonLogger()
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def softmax(x):
    """
    Computes the row-wise softmax of a 2D input tensor.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as 'x', where softmax has been
        applied along the last dimension (row-wise).

    Note:
        - The input tensor 'x' must reside on the GPU.
    """
    _LOGGER.info(f"SOFTMAX: x={tuple(x.shape)}")
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    waves_per_eu = 2
    num_warps = 8

    buffer_size = 16
    threads_per_warp = 64

    # each thread should only be loading as many elements
    # as can fit in the load buffer (determined by element size)
    cols_per_thread = triton.cdiv(BLOCK_SIZE, num_warps * threads_per_warp)
    size_per_thread = min(cols_per_thread, buffer_size // x.element_size())

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        size_per_thread,
        threads_per_warp,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )

    return y
