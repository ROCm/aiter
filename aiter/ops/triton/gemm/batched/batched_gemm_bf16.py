# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_bf16 import (
    _batched_gemm_bf16_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _batched_gemm_splitk_reduce_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def batched_gemm_bf16(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes batched 16 bit matrix multiplication Y[i] = X[i] @ W[i]^T with optional bias.

    Args:
        XQ (torch.Tensor): Input batch with shape (B, M, K) (BF16 or FP16).
        WQ (torch.Tensor): Weight batch with shape (B, N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output batch with shape (B, M, N).
    """
    _LOGGER.info(f"BATCHED_GEMM_BF16: x={tuple(XQ.shape)} w={tuple(WQ.shape)}")

    # Make sure XQ and WQ are contiguous in memory
    XQ = XQ.contiguous()
    WQ = WQ.contiguous()

    # Check constraints.
    assert XQ.shape[0] == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert XQ.shape[2] == WQ.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_bf16"
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

    num_ksplit = config["NUM_KSPLIT"]

    # When splitting along K, each split writes its own partial-sum plane into a
    # (NUM_KSPLIT, B, M, N) fp32 buffer that is then reduced into YQ.
    if num_ksplit > 1:
        y_pp = torch.empty(
            (num_ksplit, B, M, N),
            dtype=torch.float32,
            device=XQ.device,
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        B,
        META["NUM_KSPLIT"]
        * triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _batched_gemm_bf16_kernel[grid](
        XQ,
        WQ,
        YQ if num_ksplit == 1 else y_pp,
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
        YQ.stride(0) if num_ksplit == 1 else y_pp.stride(1),
        YQ.stride(1) if num_ksplit == 1 else y_pp.stride(2),
        YQ.stride(2) if num_ksplit == 1 else y_pp.stride(3),
        0 if num_ksplit == 1 else y_pp.stride(0),
        bias.stride(0) if has_bias else 0,
        has_bias,
        **config,
    )

    if num_ksplit > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            B,
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _batched_gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            YQ,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y_pp.stride(3),
            YQ.stride(0),
            YQ.stride(1),
            YQ.stride(2),
            bias.stride(0) if has_bias else 0,
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(num_ksplit),
            ADD_BIAS=has_bias,
            activation="",
            use_activation=False,
            KERNEL_NAME="_batched_gemm_bf16_reduce_kernel",
        )

    return YQ
