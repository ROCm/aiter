# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_copy_x import (
    _fused_gemm_a16w16_copy_x_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _gemm_a16w16_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter import dtypes

_LOGGER = AiterTritonLogger()


def fused_gemm_a16w16_copy_x(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    copy_dtype: Optional[torch.dtype] = None,
    y: Optional[torch.Tensor] = None,
    x_copy: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes 16-bit matmul Y = X @ W^T and also emits a downcasted copy of X.

    This fuses the GEMM with the activation-quantization downcast that
    immediately follows the router-gate GEMM in MoE flows (e.g. DSv4),
    avoiding a separate kernel/DRAM pass to cast X from BF16 to FP8.

    The fused kernel uses a single 1D grid split into two regions: GEMM tiles
    occupy `NUM_KSPLIT * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)` programs and an
    additional `cdiv(M, BLOCK_M) * cdiv(K, BLOCK_K)` programs handle the
    downcast copy.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output Y datatype (BF16 or FP16).
        copy_dtype (Optional[torch.dtype]): Output X-copy datatype
            (defaults to aiter.dtypes.fp8).
        y (Optional[torch.Tensor]): Pre-allocated output with shape (M, N).
        x_copy (Optional[torch.Tensor]): Pre-allocated downcasted X copy with
            shape (M, K).
        config (Optional[dict]): Kernel tuning parameters.
        activation (Optional[str]): Activation fused into Y.
        skip_reduce (Optional[bool]): Skip split-K reduction for Y.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Y, x_copy). When skip_reduce=True
        and NUM_KSPLIT > 1, Y has shape (NUM_KSPLIT, M, N).
    """

    _LOGGER.info(f"FUSED_GEMM_A16W16_COPY_X: x={tuple(x.shape)} w={tuple(w.shape)}")
    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    if copy_dtype is None:
        copy_dtype = dtypes.fp8

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if config is None:
        config, _ = _get_config(M, N, K)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if x_copy is None:
        x_copy = torch.empty((M, K), dtype=copy_dtype, device=x.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        + triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )
    _fused_gemm_a16w16_copy_x_kernel[grid](
        x,
        w,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_copy,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_copy.stride(0),
        x_copy.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp, x_copy

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a16w16_reduce_kernel[grid_reduce](
            bias,
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            ADD_BIAS=(bias is not None),
        )

    return y, x_copy
