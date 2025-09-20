# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton._triton_kernels.gemm_a16w16 import (
    _gemm_a16_w16_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def gemm_a16w16(
    x,
    w,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
):
    """
    Computes the 16 bit matmul Y = X x W

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
    - Y: Output Matrix Y with shape (M, N).
    If this is none, then it's created by this API and returned as output.
    - activation: Optional activation function to apply to the output.
    One of ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu"). Default is None.

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(f"GEMM_A16W16: x={tuple(x.shape)} w={tuple(w.shape)}")
    # Shape checks
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
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
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a16w16_reduce_kernel[grid_reduce](
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
        )

    return y
