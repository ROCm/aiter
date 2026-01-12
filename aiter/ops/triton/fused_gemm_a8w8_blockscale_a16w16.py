# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_gemm_a8w8_blockscale_a16w16 import (
    _fused_gemm_a8w8_blockscale_a16w16_kernel,
    _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_gemm_a8w8_blockscale_a16w16(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_fp8_scale: torch.Tensor,
    w_fp8_scale: torch.Tensor,
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    bias_fp8: Optional[torch.Tensor] = None,
    bias_bf16: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y_fp8: Optional[torch.Tensor] = None,
    y_bf16: Optional[torch.Tensor] = None,
    skip_reduce: Optional[bool] = False,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT + B using the block-scale quantization approach for x_fp8 and w_fp8.
    Computes the 16 bit matmul Y = X x WT + B for x_bf16 and w_bf16

    This fusion is primarily aiming for fusing the gate up-projections and MOE gating:
        gate up-projections: (M, K) x (2N, K) = (M, 2N)
        MOE gating: (M, K) x (N, K) + (N, ) = (M, N)

    Key parameters:
    - x_fp8: Matrix X with shape (M, K).
    - w_fp8: Matrix W with shape (N_fp8, K).
    - x_fp8_scale: Scale tensor for X with shape (M, *scale_k).
    - w_fp8_scale: Scale tensor for W with shape (**scale_n, *scale_k).
    - x_bf16: Matrix X with shape (M, K).
    - w_bf16: Matrix W with shape (N_bf16, K).

    Note: M, N, K must be identical for x_fp8 and x_bf16, but the N-dim fow w_fp8 and w_bf16 can be different

    Returns:
    - y_fp8: The output matrix with shape (M, N_fp8).
    - y_bf16: The output matrix with shape (M, N_bf16).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k
    **scale_n = (N_fp8 + scale_block_size_n - 1) // scale_block_size_n
    """
    _LOGGER.info(
        f"FUSED_GEMM_A8W8_BLOCKSCALE_A16W16: x_fp8={tuple(x_fp8.shape)} w_fp8={tuple(w_fp8.shape)} x_fp8_scale={tuple(x_fp8_scale.shape)} w_scale={tuple(w_fp8_scale.shape)} x_bf16={tuple(x_bf16.shape)} w_bf16={tuple(w_bf16.shape)}"
    )

    M, K = x_fp8.shape
    N_fp8, K = w_fp8.shape
    M, K = x_bf16.shape
    N_bf16, K = w_bf16.shape

    # Check constraints.
    assert (
        x_fp8.shape[0] == x_bf16.shape[0]
    ), "M-dim should be identical for x_fp8 and x_bf16"
    assert (
        x_fp8.shape[1] == x_bf16.shape[1]
    ), "K-dim should be identical for x_fp8 and x_bf16"
    assert x_fp8.shape[1] == w_fp8.shape[1], "Incompatible dimensions!!!"
    assert x_bf16.shape[1] == w_bf16.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w_fp8 = w_fp8.T
    w_bf16 = w_bf16.T
    w_fp8_scale = w_fp8_scale.T

    if config is None:
        config, _ = _get_config(M, N_fp8, N_bf16, K)

    if y_fp8 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_fp8 = torch.empty((M, N_fp8), dtype=dtype, device=x_fp8.device)

    if y_bf16 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_bf16 = torch.empty((M, N_bf16), dtype=dtype, device=x_bf16.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        y_fp8_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_fp8),
            dtype=torch.float32,
            device=y_fp8.device if y_fp8 is not None else x_fp8.device,
        )
        y_bf16_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_bf16),
            dtype=torch.float32,
            device=y_bf16.device if y_bf16 is not None else x_bf16.device,
        )
    else:
        y_fp8_pp = None
        y_bf16_pp = None

    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_fp8_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N_fp8, w_fp8_scale.shape[1]))

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * (
                triton.cdiv(N_fp8, META["BLOCK_SIZE_N"])
                + triton.cdiv(N_bf16, META["BLOCK_SIZE_N"])
            )
        ),
    )
    _fused_gemm_a8w8_blockscale_a16w16_kernel[grid](
        x_fp8,
        w_fp8,
        bias_fp8,
        x_fp8_scale,
        w_fp8_scale,
        y_fp8 if config["NUM_KSPLIT"] == 1 else y_fp8_pp,
        x_bf16,
        w_bf16,
        bias_bf16,
        y_bf16 if config["NUM_KSPLIT"] == 1 else y_bf16_pp,
        M,
        N_fp8,
        N_bf16,
        K,
        x_fp8.stride(0),
        x_fp8.stride(1),
        w_fp8.stride(0),
        w_fp8.stride(1),
        x_fp8_scale.stride(0),
        x_fp8_scale.stride(1),
        w_fp8_scale.stride(0),
        w_fp8_scale.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(0),
        y_fp8.stride(0) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(1),
        y_fp8.stride(1) if config["NUM_KSPLIT"] == 1 else y_fp8_pp.stride(2),
        x_bf16.stride(0),
        x_bf16.stride(1),
        w_bf16.stride(0),
        w_bf16.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(0),
        y_bf16.stride(0) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(1),
        y_bf16.stride(1) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(2),
        ADD_BIAS_FP8=(bias_fp8 is not None),
        ADD_BIAS_BF16=(bias_bf16 is not None),
        SKIP_REDUCE=skip_reduce,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_fp8_pp, y_bf16_pp
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N_fp8, REDUCE_BLOCK_SIZE_N)
            + triton.cdiv(N_bf16, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel[grid_reduce](
            bias_fp8,
            y_fp8_pp,
            y_fp8,
            bias_bf16,
            y_bf16_pp,
            y_bf16,
            M,
            N_fp8,
            N_bf16,
            y_fp8_pp.stride(0),
            y_fp8_pp.stride(1),
            y_fp8_pp.stride(2),
            y_fp8.stride(0),
            y_fp8.stride(1),
            y_bf16_pp.stride(0),
            y_bf16_pp.stride(1),
            y_bf16_pp.stride(2),
            y_bf16.stride(0),
            y_bf16.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS_FP8=(bias_fp8 is not None),
            ADD_BIAS_BF16=(bias_bf16 is not None),
        )

    return y_fp8, y_bf16
