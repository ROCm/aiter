# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.batched_gemm_a8w8 import (
    _batched_gemm_a8w8_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def batched_gemm_a8w8(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with per-batch scaling.
    Each batch element is independently scaled back to higher precision.

    Args:
        XQ (torch.Tensor): INT8 input batch with shape (B, M, K).
        WQ (torch.Tensor): INT8 weight batch with shape (B, N, K), internally transposed.
        x_scale (torch.Tensor): Scale for XQ with shape (B, M, 1).
        w_scale (torch.Tensor): Scale for WQ with shape (B, 1, N).
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output batch with shape (B, M, N).
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8W8: x={tuple(XQ.shape)} w={tuple(WQ.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Make sure XQ and WQ are contiguous in memory
    XQ = XQ.contiguous()
    WQ = WQ.contiguous()

    # Check constraints.
    assert XQ.shape[0] == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert XQ.shape[2] == WQ.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    # Transpose N and K dimensions of WQ: (B, N, K) -> (B, K, N)
    WQ = WQ.transpose(1, 2)

    B = XQ.shape[0]
    M = XQ.shape[1]
    K = XQ.shape[2]
    N = WQ.shape[2]

    has_bias = bias is not None
    if YQ is None:
        YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)

    if config is None:
        config, _ = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _batched_gemm_a8w8_kernel[grid](
        XQ,
        WQ,
        YQ,
        x_scale,
        w_scale,
        bias,
        M,
        N,
        K,
        XQ.stride(0),
        XQ.stride(1),
        XQ.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0),
        YQ.stride(1),
        YQ.stride(2),
        x_scale.stride(0),
        w_scale.stride(0),
        bias.stride(0) if has_bias else 0,
        has_bias,
        **config,
    )

    return YQ
