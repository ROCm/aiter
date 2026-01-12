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
    Computes 16 bit gated matrix multiplication Y = X @ W^T with gating mechanism (e.g., SwiGLU).
    Uses first half of W output as gate for second half, producing (M, N//2) output.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed. N must be even.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N//2).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).
        activation (Optional[str]): Activation function applied to gate ("gelu", "gelu_tanh",
            "silu", "silu_exp2", "relu").

    Returns:
        torch.Tensor: Gated output with shape (M, N//2).
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
        config, _ = _get_config(M, N, K)

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
