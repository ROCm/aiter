# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.quant import _mxfp4_quant_op
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.gemm_afp4wfp4_pre_quant_atomic import (
    _gemm_afp4_wfp4_pre_quant_kernel,
    _get_config,
)

_LOGGER = AiterTritonLogger()


def gemm_afp4wfp4_pre_quant(
    x,
    w,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes matrix multiplication Y = X @ W^T with on-the-fly FP4 quantization of activations.
    X is quantized to MXFP4 during computation, W is pre-quantized FP4. Uses atomic operations for split-K reduction.

    Args:
        x (torch.Tensor): Higher precision input matrix with shape (M, K) (BF16 or FP16).
            Quantized to FP4 E2M1 on-the-fly during GEMM.
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
            Must be zero-initialized for atomic operations.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_AFP4WFP4_PRE_QUANT_ATOMIC: x={tuple(x.shape)} w={tuple(w.shape)} w_scale={tuple(w_scales.shape)} "
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # inner kernel expects (K, N)
    w = w.T

    if y is None:
        y = torch.zeros((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_afp4_wfp4_pre_quant_kernel[grid](
        x,
        w,
        y,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0,
        y.stride(0),
        y.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    return y
