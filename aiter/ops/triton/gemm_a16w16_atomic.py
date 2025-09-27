# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm_a16w16_atomic import (
    _gemm_a16_w16_atomic_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w16_atomic(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 16 bit matmul Y = X x W
    NOTE: If dtype is set to bf16, aggregation in bf16 with atomic_add will lead to slight precision loss.
    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_A16W16_ATOMIC: x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)} "
    )

    w = w.T

    M, K = x.shape
    K, N = w.shape

    if config is None:
        config = _get_config(M, N, K)

    if y is None:
        # atomic add requires 0 tensor
        if config["NUM_KSPLIT"] == 1:
            y = torch.empty((M, N), dtype=dtype, device=x.device)
        else:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        * META["NUM_KSPLIT"],
    )

    _gemm_a16_w16_atomic_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        **config,
    )

    return y
