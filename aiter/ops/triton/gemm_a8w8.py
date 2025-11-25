# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton._triton_kernels.gemm_a8w8 import _gemm_a8w8_kernel, _get_config
from aiter.ops.triton.utils.device_info import get_num_xcds

from aiter.ops.triton.gluon.gemm_a8w8 import _gemm_a8w8_kernel as _gluon_gemm_a8w8_kernel

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
):
    """
    Computes the 8 bit matmul Y = X x WT, applies a conversion scale and optionally adds a bias
    to the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: First scale tensor with shape (M, 1).
    - W_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    - Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    M, K = x.shape
    N, K = w.shape

    # Transpose w (kernel expects (K, N))
    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        **config,
    )

    return y


def gluon_gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT, applies a conversion scale and optionally adds a bias
    to the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix W with shape (N, K).
    - X_scale: First scale tensor with shape (M, 1).
    - W_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    - Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output

    Returns:
    - Y: The output matrix with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    M, K = x.shape
    N, K = w.shape

    # Transpose w (kernel expects (K, N))
    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)
    config = config.copy()
    if "NUM_WARPS" not in config:
        config["NUM_WARPS"] = 4
    if "cache_modifier" not in config:
        config["cache_modifier"] = ".cg"
    if "cdna_version" not in config:
        config["cdna_version"] = 4 if "gfx950" in arch_info.get_arch() else 3

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gluon_gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        **config,
    )

    return y
