# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.gemm_a16w16_gated import (
    _gemm_a16_w16_gated_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w16_gated(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Computes the 16 bit matmul Y = X x W
    Uses the first half of the output (along the N dim) as a gate for the second half (e.g for SwiGLU)

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N//2).
    If this is none, then it's created by this API and returned as output.
    - activation: Optional activation function to apply to the output. One of ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu")

    Returns:
    - Y: The output matrix with shape (M, N//2).
    """
    _LOGGER.info(f"GEMM_A16W16_GATED: x={tuple(x.shape)} w={tuple(w.shape)}")

    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    M, K = x.shape
    N, K = w.shape

    assert N % 2 == 0, "Weight shape incompatible with gating (N not divisible by 2)"

    w = w.T

    if y is None:
        y = torch.empty((M, N // 2), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _gemm_a16_w16_gated_kernel[grid](
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
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        **config,
    )

    return y
