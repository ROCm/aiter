# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton._triton_kernels.gemm_a8wfp4 import (
    _gemm_a8wfp4_kernel,
    _gemm_afp4_wfp4_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def gemm_a8wfp4(
    x,
    w,
    y,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    config: Optional[dict] = None,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP8 activations and FP4 weights.

    Args:
        x (torch.Tensor): FP8 E4M3 input matrix with shape (M, K).
        w (torch.Tensor): Packed FP4 weight matrix with shape (N, K//2), internally transposed.
            Each uint8 contains 2 FP4 values.
        y (torch.Tensor): Pre-allocated output tensor with shape (M, N).
        x_scales (torch.Tensor): FP32 per-row scale for x with shape (M, 1).
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

     Note:
    - The logical shape of W after unpacking would be (N, K)
    - Every 32 consecutive elements in the K dimension of W share
    one E8M0 scale

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    _LOGGER.info(
        f"GEMM_A8FP4: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scales.shape)} w_scale={tuple(w_scales.shape)}  "
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K_packed = w.shape
    w = w.T

    assert (
        K_packed == K // 2
    ), f"Inconsistent shapes: x has K={K} but w has K_packed={K_packed}, expected {K//2}"
    assert x_scales.shape[0] == M and w_scales.shape == (
        N,
        K // 32,
    ), f"Scale shapes incorrect: x_scales should be ({M}, 1), got {x_scales.shape}; w_scales should be ({N}, {K//32}), got {w_scales.shape}"

    if config is None:
        config, _ = _get_config(M, N, K)

    if M <= 128:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
            )
    else:
        SPLITK_BLOCK_SIZE = 2 * K
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            config["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    y_final = y if config["NUM_KSPLIT"] == 1 else y_pp
    stride_am, stride_ak = x.stride()
    stride_bk, stride_bn = w.stride()
    stride_ck, stride_cm, stride_cn = (
        (0, y.stride(0), y.stride(1)) if config["NUM_KSPLIT"] == 1 else y_pp.stride()
    )
    stride_asm, stride_ask = x_scales.stride()
    stride_bsn, stride_bsk = w_scales.stride()

    _gemm_a8wfp4_kernel[grid](
        x,
        w,
        y_final,
        x_scales,
        w_scales,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_ck,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bsn,
        stride_bsk,
        RAW_MASKED_LOADS=True,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"]))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_afp4_wfp4_reduce_kernel[grid_reduce](
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
            config["NUM_KSPLIT"],
        )
