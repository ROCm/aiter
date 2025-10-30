# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton._triton_kernels.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _get_config,
)


def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    transpose_bm: Optional[bool] = False,
    transpose_bm_in: Optional[bool] = False,
    config: Optional[dict] = None,
):
    """
    Computes the matmul YQ[i] = XQ[i] x WQ[i]T and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - XQ: Batch tensor XQ with shape (B, M, K) if transpose_bm_in == False else (M, B, K).
    - WQ: Batch tensor WQ with shape (B, N, K).
    - W_scale: Second scale batch tensor with shape (1, ).
    - Bias: Bias batch tensor with shape (B, 1, N).
    - YQ: Output Matrix Y with shape (B, M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - YQ: The output batch tensor with shape (B, M, N) if transpose_bm == False else (M, B, N).
    """

    # Check constraints.
    if not transpose_bm_in:
        B = X.shape[0]
        M = X.shape[1]
    else:
        M = X.shape[0]
        B = X.shape[1]
    K = X.shape[2]
    N = WQ.shape[1]

    assert B == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert K == WQ.shape[2], "Incompatible K dimensions!!!"
    assert (
        triton.next_power_of_2(group_size) == group_size
    ), "group_size mush be power of 2"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    WQ = WQ.transpose(1, 2)

    has_bias = bias is not None
    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    else:
        if transpose_bm:
            assert (
                YQ.shape[0] == M and YQ.shape[1] == B and YQ.shape[2] == N
            ), "Output dimension error"
        else:
            assert (
                YQ.shape[0] == B and YQ.shape[1] == M and YQ.shape[2] == N
            ), "Output dimension error"

    if config is None:
        config = _get_config(M, N, K)
    config["BLOCK_SIZE_K"] = group_size

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    DTYPE_MAX = (
        torch.finfo(WQ.dtype).max
        if torch.is_floating_point(WQ)
        else torch.iinfo(WQ.dtype).max
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[
        grid
    ](
        X,
        WQ,
        YQ,
        w_scale,
        bias,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        bias.stride(0) if has_bias else 0,
        has_bias,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    return YQ
