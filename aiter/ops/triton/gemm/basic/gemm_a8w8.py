# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _gemm_a8w8_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    num_ksplit: int = 1,
    skip_reduce: bool = False,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).
        num_ksplit (int): Number of K-dimension splits for split-K GEMM. Default 1 (no split).
        skip_reduce (bool): If True and num_ksplit > 1, return partial results of shape
            (NUM_KSPLIT, M, N) without reducing.

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    M, K = x.shape
    N, K = w.shape

    w = w.T

    if config is None:
        config, _ = _get_config(M, N, K)

    if num_ksplit > 1:
        config["NUM_KSPLIT"] = num_ksplit
        compute_splitk_params(config, K)

    NUM_KSPLIT = config["NUM_KSPLIT"]

    if y is None and (NUM_KSPLIT == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if NUM_KSPLIT > 1:
        y_pp = torch.empty(
            (NUM_KSPLIT, M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    config.pop("cache_modifier", None)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y if NUM_KSPLIT == 1 else y_pp,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if NUM_KSPLIT == 1 else y_pp.stride(0),
        y.stride(0) if NUM_KSPLIT == 1 else y_pp.stride(1),
        y.stride(1) if NUM_KSPLIT == 1 else y_pp.stride(2),
        (bias is not None) and (NUM_KSPLIT == 1),
        NUM_XCDS=get_num_xcds(),
        **config,
    )

    if NUM_KSPLIT > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a8w8_reduce_kernel[grid_reduce](
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
            bias is not None,
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(NUM_KSPLIT),
        )

    return y
