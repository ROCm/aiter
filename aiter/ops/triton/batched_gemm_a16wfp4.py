# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton._triton_kernels.batched_gemm_a16wfp4 import (
    _batched_gemm_a16wfp4_reduce_kernel,
    _batched_gemm_a16wfp4_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.gemm_a16wfp4 import (
    get_splitk,
)

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def batched_gemm_a16wfp4(
    x,
    w,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    transpose_bm: Optional[bool] = False,
    prequant: Optional[bool] = True,
    y_scale: Optional[torch.Tensor] = None,
):
    """
    Computes batched FP4 matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to MXFP4 during computation, W is pre-quantized FP4.

    Args:
        x (torch.Tensor): Higher precision input batch with shape (B, M, K) (BF16 or FP16).
            Quantized to MXFP4 on-the-fly during GEMM.
        w (torch.Tensor): FP4 E2M1 weight batch with shape (B, N, K), internally transposed.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (B, N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.


    Returns:
        torch.Tensor: Output batch with shape (B, M, N).
    """
    _LOGGER.info(
        f"BATCHED_GEMM_AFP4WFP_PREQUANT: x={tuple(x.shape)} w={tuple(w.shape)} w_scale={tuple(w.shape)}"
    )

    assert prequant is True, "prequant = False is not yet supported"

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    Bx, M, K = x.shape
    Bw, N, K = w.shape
    assert Bx == Bw
    B = Bx

    if config is None:
        config = _get_config(M, N, K)

    if y is None:
        if transpose_bm:
            y = torch.empty((M, B, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device=x.device)
    else:
        if transpose_bm:
            assert (
                y.shape[0] == M and y.shape[1] == B and y.shape[2] == N
            ), f"Output dimension error {y.shape} {B} {M} {N}"
        else:
            assert (
                y.shape[0] == B and y.shape[1] == M and y.shape[2] == N
            ), f"Output dimension error {y.shape} {B} {M} {N}"

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (B, config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (B, config["NUM_KSPLIT"], M, N),
                dtype=torch.float32,
                device=y.device,
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    if config["NUM_KSPLIT"] == 1:
        stride_ck = 0
        stride_cn = y.stride(2)
        if transpose_bm:
            stride_cb = y.stride(1)
            stride_cm = y.stride(0)
        else:
            stride_cb = y.stride(0)
            stride_cm = y.stride(1)
    else:
        stride_cb = y_pp.stride(0)
        stride_ck = y_pp.stride(1)
        stride_cm = y_pp.stride(2)
        stride_cn = y_pp.stride(3)

    grid = lambda META: (  # noqa: E731
        B,
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _batched_gemm_a16wfp4_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        w_scales,
        y_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        stride_cb,
        stride_ck,
        stride_cm,
        stride_cn,
        w_scales.stride(0),
        w_scales.stride(1),
        w_scales.stride(2),
        PRE_QUANT=prequant,
        HAVE_Y_SCALE=(y_scale is not None),
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
            B,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _batched_gemm_a16wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            y.stride(0) if transpose_bm else y.stride(1),
            y.stride(1) if transpose_bm else y.stride(0),
            y.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            config["NUM_KSPLIT"],
        )
    return y
