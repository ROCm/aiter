# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.fused.fused_gemm_a16w16_quant_x import (
    _fused_gemm_a16w16_quant_x_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_QUANT_BLOCK_SIZE = 32


def fused_gemm_a16w16_quant_x(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    y: Optional[torch.Tensor] = None,
    x_quant: Optional[torch.Tensor] = None,
    x_scales: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes 16-bit matmul Y = X @ W^T and also emits an MXFP8-quantized X.

    This fuses the GEMM with the activation per-1x32 MXFP8 quantization that
    immediately follows the router-gate GEMM in MoE flows (e.g. DSv4),
    avoiding a separate kernel/DRAM pass to cast X from BF16 to FP8 + e8m0
    scales.

    The fused kernel uses a single 1D grid split into two regions: GEMM tiles
    occupy `NUM_KSPLIT * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N)` programs and an
    additional `cdiv(M, BLOCK_M) * cdiv(K, BLOCK_K)` programs handle the
    MXFP8 quant of X.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K). K must be a multiple
            of 32.
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output Y datatype (BF16 or FP16).
        quant_dtype (Optional[torch.dtype]): FP8 dtype for the MXFP8 quantized
            X values (defaults to torch.float8_e4m3fn).
        y (Optional[torch.Tensor]): Pre-allocated output with shape (M, N).
        x_quant (Optional[torch.Tensor]): Pre-allocated MXFP8 quantized X with
            shape (M, K).
        x_scales (Optional[torch.Tensor]): Pre-allocated uint8 e8m0 scales with
            shape (M, K // 32).
        config (Optional[dict]): Kernel tuning parameters.
        activation (Optional[str]): Activation fused into Y.
        skip_reduce (Optional[bool]): Skip split-K reduction for Y.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (Y, x_quant, x_scales).
        When skip_reduce=True and NUM_KSPLIT > 1, Y has shape (NUM_KSPLIT, M, N).
    """

    _LOGGER.info(f"FUSED_GEMM_A16W16_QUANT_X: x={tuple(x.shape)} w={tuple(w.shape)}")
    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."
    assert (
        x.shape[1] % _QUANT_BLOCK_SIZE == 0
    ), f"K={x.shape[1]} must be a multiple of {_QUANT_BLOCK_SIZE} for MXFP8 quant"

    if quant_dtype is None:
        quant_dtype = torch.float8_e4m3fn

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if config is None:
        config, _ = _get_config(M, N, K)

    assert (
        config["BLOCK_SIZE_K"] % _QUANT_BLOCK_SIZE == 0
    ), f"BLOCK_SIZE_K={config['BLOCK_SIZE_K']} must be a multiple of {_QUANT_BLOCK_SIZE}"

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if x_quant is None:
        x_quant = torch.empty((M, K), dtype=quant_dtype, device=x.device)

    if x_scales is None:
        x_scales = torch.empty(
            (M, K // _QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
        )

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
    _fused_gemm_a16w16_quant_x_kernel[grid](
        x,
        w,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_quant,
        x_scales,
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
        x_quant.stride(0),
        x_quant.stride(1),
        x_scales.stride(0),
        x_scales.stride(1),
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp, x_quant, x_scales

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            bias,
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
            ADD_BIAS=(bias is not None),
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            KERNEL_NAME="_fused_gemm_a16w16_quant_x_reduce_kernel",
        )

    return y, x_quant, x_scales
