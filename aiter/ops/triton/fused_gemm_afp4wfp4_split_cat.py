# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.fused_gemm_afp4wfp4_split_cat import (
    _fused_gemm_afp4wfp4_split_cat,
    _fused_gemm_afp4wfp4_split_cat_reduce,
    _get_config,
)
from aiter.ops.triton.gemm_afp4wfp4 import get_splitk
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_gemm_afp4wfp4_split_cat(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    S1: int,
    S2: int,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    config: Optional[dict] = None,
):
    """
    Computes the 4 bit matmul C = X @ W^T using the block-scale quantization approach.
    Then split the product C into C1 and C2 with sizes S1 and S2 at the last dimension respectively.
    Finally concatenate Y to C1 at the last dimension.

    Equivalent to the following sequence:
    c = (x @ w).view(-1, y.shape(1), S1 + S2)
    c1, c2 = c.split([S1, S2], dim=-1)
    c1 = c1.cat(y, dim=-1)
    return c1, c2

    Key parameters:
    - x: Matrix X with shape (M, K // 2).
    - w: Matrix W with shape (N, K // 2).
    - y: Tensor Y with shape (M, D, S3).
    - x_scale: Scale tensor for X with shape (M, *scale_k).
    - w_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    Returns:
    - c1: The output matrix with shape (M, D, S1 + S3).
    - c2: The output matrix with shape (M, D, S2).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k -> ceil_div(K, scale_block_size_k)
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n -> ceil_div(N, scale_block_size_n)

    NOTE: N must be D * (S1 + S2)
    """
    _LOGGER.info(
        f"FUSED_gemm_afp4wfp4_SPLIT_CAT: x={tuple(x.shape)} w={tuple(w.shape)} y={tuple(y.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape
    M, D, S3 = y.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"
    assert y.shape[0] == x.shape[0], "Incompatible dimensions!!!"
    assert N == D * (S1 + S2), "N is not D * (S1 + S2)"

    # Transpose w and w_scale
    w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    if config is None:
        config = _get_config(M, N, K)

    c1 = torch.empty((M, D, S1 + S3), dtype=dtype, device=x.device)
    c2 = torch.empty((M, D, S2), dtype=dtype, device=x.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        config["NUM_KSPLIT"] = 1
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

    if config["NUM_KSPLIT"] > 1:
        c_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        c_pp = None

    # S3 block sizes
    config["BLOCK_SIZE_S3"] = triton.next_power_of_2(
        triton.cdiv(D * S3, triton.cdiv(N, config["BLOCK_SIZE_N"]))
    )

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )
    _fused_gemm_afp4wfp4_split_cat[grid](
        x,
        w,
        y,
        c1 if config["NUM_KSPLIT"] == 1 else c_pp,
        c2,
        x_scale,
        w_scale,
        M,
        N,
        K,
        D,
        S1,
        S2,
        S3,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        c1.stride(1) if config["NUM_KSPLIT"] == 1 else c_pp.stride(0),
        c1.stride(0) if config["NUM_KSPLIT"] == 1 else c_pp.stride(1),
        c1.stride(2) if config["NUM_KSPLIT"] == 1 else c_pp.stride(2),
        c2.stride(0),
        c2.stride(1),
        c2.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        REDUCE_BLOCK_SIZE_S3 = triton.next_power_of_2(
            triton.cdiv(D * S3, triton.cdiv(N, REDUCE_BLOCK_SIZE_N))
        )
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_afp4wfp4_split_cat_reduce[grid_reduce](
            c_pp,
            c1,
            c2,
            y,
            M,
            N,
            D,
            S1,
            S2,
            S3,
            c_pp.stride(0),
            c_pp.stride(1),
            c_pp.stride(2),
            c1.stride(0),
            c1.stride(1),
            c1.stride(2),
            c2.stride(0),
            c2.stride(1),
            c2.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            REDUCE_BLOCK_SIZE_S3,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return c1, c2
