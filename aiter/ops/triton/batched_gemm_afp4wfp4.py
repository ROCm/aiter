# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton._triton_kernels.batched_gemm_afp4wfp4 import (
    _batched_gemm_afp4_wfp4_kernel,
    _batched_gemm_afp4_wfp4_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def batched_gemm_afp4wfp4(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes batched FP4 matrix multiplication Y[i] = X[i] @ W[i]^T with FP4 activations and weights.

    Args:
        x (torch.Tensor): FP4 E2M1 input batch with shape (B, M, K).
        w (torch.Tensor): FP4 E2M1 weight batch with shape (B, N, K), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (B, M, K//32).
            One scale per 32 elements in K dimension.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (B, N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        torch.Tensor: Output batch with shape (B, M, N).
    """
    _LOGGER.info(
        f"BATCHED_GEMM_AFP4WFP4: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x.shape)} w_scale={tuple(w.shape)}"
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    w = w.transpose(1, 2)
    Bx, M, K = x.shape
    Bw, K, N = w.shape
    By, _, _ = y.shape
    assert Bx == Bw == By
    Batch = Bx

    if config is None:
        config, _ = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (Batch, config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (Batch, config["NUM_KSPLIT"], M, N),
                dtype=torch.float32,
                device=y.device,
            )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        Batch,
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _batched_gemm_afp4_wfp4_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        y.stride(2) if config["NUM_KSPLIT"] == 1 else y_pp.stride(3),
        x_scales.stride(0),
        x_scales.stride(1),
        x_scales.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            Batch,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _batched_gemm_afp4_wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            config["NUM_KSPLIT"],
        )
