# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.fused_kv_proj_cat import (
    _fused_kv_proj_cat_kernel,
    _fused_kv_proj_cat_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_kv_proj_cat(
    kv_c: torch.Tensor,
    w: torch.Tensor,
    k_pe: torch.Tensor,
    kv_c_scale: torch.Tensor,
    w_scale: torch.Tensor,
    nope_head_dim: int,
    v_head_dim: int,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    config: Optional[dict] = None,
):
    """
    Computes the 8 bit matmul Y = X x WT using the block-scale quantization approach.
    Then split the product Y into k_nope and v, and concatenate k_pe to k_nope at the last dimension.

    Key parameters:
    - kv_c: Matrix X with shape (M, K).
    - w: Matrix W with shape (N, K).
    - k_pe: Matrix with shape (M, H, R).
    - kv_c_scale: Scale tensor for X with shape (M, *scale_k).
    - w_scale: Scale tensor for W with shape (**scale_n, *scale_k).

    M is the KV sequence length, K is the latent dimension for K/V, N is H * (P + V)
    H is the number of attention heads, P is nope dimension, V is the V head dimension
    R is the rope dimension

    Returns:
    - k: The output matrix with shape (M, H, P+R).
    - v: The output matrix with shape (M, H, V).

    *scale_k = (K + scale_block_size_k - 1) // scale_block_size_k -> ceil_div(K, scale_block_size_k)
    **scale_n = (N + scale_block_size_n - 1) // scale_block_size_n -> ceil_div(N, scale_block_size_n)
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE_CAT: k_c_normed={tuple(kv_c.shape)} w={tuple(w.shape)} k_c_normed_scale={tuple(kv_c_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = kv_c.shape
    N, K = w.shape
    M, H, R = k_pe.shape
    P, V = nope_head_dim, v_head_dim

    # Check constraints.
    assert kv_c.shape[1] == w.shape[1], "Incompatible dimensions!!!"
    assert k_pe.shape[0] == kv_c.shape[0], "Incompatible dimensions!!!"
    assert N == H * (P + V), "N is not H*(P+V)"

    # Transpose w and w_scale
    w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    if config is None:
        config = _get_config(M, N, K)

    k = torch.empty((M, H, P + R), dtype=dtype, device=kv_c.device)
    v = torch.empty((M, H, V), dtype=dtype, device=kv_c.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        k_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=kv_c.device,
        )
    else:
        k_pp = None

    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(
        config["BLOCK_SIZE_K"], 16
    )  # minimum block size is 16 for perf

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scale.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scale.shape[1])
    )  # scale_block_size_n

    # Block size R
    config["BLOCK_SIZE_R"] = triton.next_power_of_2(
        triton.cdiv(H * R, triton.cdiv(N, config["BLOCK_SIZE_N"]))
    )

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )
    _fused_kv_proj_cat_kernel[grid](
        kv_c,
        w,
        k_pe,
        k if config["NUM_KSPLIT"] == 1 else k_pp,
        v,
        kv_c_scale,
        w_scale,
        M,
        N,
        K,
        H,
        R,
        P,
        V,
        kv_c.stride(0),
        kv_c.stride(1),
        w.stride(0),
        w.stride(1),
        k_pe.stride(0),
        k_pe.stride(1),
        k_pe.stride(2),
        k.stride(1) if config["NUM_KSPLIT"] == 1 else k_pp.stride(0),
        k.stride(0) if config["NUM_KSPLIT"] == 1 else k_pp.stride(1),
        k.stride(2) if config["NUM_KSPLIT"] == 1 else k_pp.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        kv_c_scale.stride(0),
        kv_c_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        REDUCE_BLOCK_SIZE_R = triton.next_power_of_2(triton.cdiv(H * R, triton.cdiv(N, REDUCE_BLOCK_SIZE_N)))
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_kv_proj_cat_reduce_kernel[grid_reduce](
            k_pp,
            k,
            v,
            k_pe,
            M,
            N,
            H,
            R,
            P,
            V,
            k_pp.stride(0),
            k_pp.stride(1),
            k_pp.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            k_pe.stride(0),
            k_pe.stride(1),
            k_pe.stride(2),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            REDUCE_BLOCK_SIZE_R,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return k, v
