# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as acp
from aiter.ops.triton.gluon._gluon_kernels.softmax import _softmax_kernel_online


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
    num_stages = 2

    buffer_size_128 = 16
    buffer_size_32 = 4
    threads_per_warp = 64

    # Each thread should only be loading as many elements as can fit in the load buffer (determined by element size)
    # (i.e. size_per_thread * (x.element_size() * 8) == 128 or 32, for async_copy load functions)
    cols_per_thread = triton.cdiv(BLOCK_SIZE, num_warps * threads_per_warp)
    size_per_thread = min(cols_per_thread, buffer_size_128 // x.element_size())
    if (
        size_per_thread * x.element_size() != buffer_size_128
        and size_per_thread * x.element_size() != buffer_size_32
    ):
        size_per_thread = buffer_size_32 // x.element_size()

    num_programs = n_rows

    # We do not want a direct-to-LDS copy for sizes < 4 bytes
    async_copy_threshold = 4 // x.element_size()
    use_async_copy = n_cols > async_copy_threshold and BLOCK_SIZE > async_copy_threshold

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
        num_stages,
        use_async_copy,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )

    return y
