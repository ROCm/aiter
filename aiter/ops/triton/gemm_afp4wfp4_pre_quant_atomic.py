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
    Computes the matmul Y = X x W
    W is an e2m1 fp4 tensor and w_scales is an e8m0 tensor.
    Every 32 elements in the K dimension share one e8m0 scale.
    X gets quantized to the microscale fp4 (mxfp4) format before the GEMM.


    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - W_scales: Matrix with shape (N, K // 32)

    Returns:
    - Y: The output matrix with shape (M, N).
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
