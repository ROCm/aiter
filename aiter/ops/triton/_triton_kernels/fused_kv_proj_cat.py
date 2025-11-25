# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_kv_proj_cat_kernel(
    # Pointers to matrices
    kv_c_ptr,
    w_ptr,
    k_pe_ptr,
    k_ptr, # [K, M, N] when splitK, otherwise [M, H, P+R]
    v_ptr, # [M, H, V]
    kv_c_scale_ptr,
    w_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    H,
    R,
    P,
    V,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_kv_c_m,
    stride_kv_c_k,
    stride_w_k,
    stride_w_n,
    stride_k_pe_m,
    stride_k_pe_h,
    stride_k_pe_r,
    stride_k_out_k, # dimension K when splitK, otherwise dimension H
    stride_k_out_m,
    stride_k_out_n, # dimension N when splitK, otherwise dimension P+R
    stride_v_out_m,
    stride_v_out_h,
    stride_v_out_v,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_kv_c_m > 0)
    tl.assume(stride_kv_c_k > 0)
    tl.assume(stride_w_k > 0)
    tl.assume(stride_w_n > 0)
    tl.assume(stride_k_pe_m > 0)
    tl.assume(stride_k_pe_h >= 0) # due to expand
    tl.assume(stride_k_pe_r > 0)
    tl.assume(stride_k_out_k > 0)
    tl.assume(stride_k_out_m > 0)
    tl.assume(stride_k_out_n > 0)
    tl.assume(stride_v_out_h > 0)
    tl.assume(stride_v_out_m > 0)
    tl.assume(stride_v_out_v > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_kv_c_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_w_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        kv_c_ptrs = kv_c_ptr + (
            offs_kv_c_m[:, None] * stride_kv_c_m + offs_k_split[None, :] * stride_kv_c_k
        )
        w_ptrs = w_ptr + (
            offs_k_split[:, None] * stride_w_k + offs_w_n[None, :] * stride_w_n
        )

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        kv_c_scale_ptrs = (
            kv_c_scale_ptr + offs_kv_c_m * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_w_scale_n = offs_w_n // GROUP_N
        w_scale_ptrs = (
            w_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_w_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if k_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(kv_c_ptrs)
                b = tl.load(w_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    kv_c_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            kv_c_scale = tl.load(kv_c_scale_ptrs)
            w_scale = tl.load(w_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += (
                tl.dot(a, b, input_precision="ieee")
                * kv_c_scale[:, None]
                * w_scale[None, :]
            )

            # Advance the ptrs to the next K block.
            kv_c_ptrs += BLOCK_SIZE_K * stride_kv_c_k
            w_ptrs += BLOCK_SIZE_K * stride_w_k

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            kv_c_scale_ptrs += offs_ks_step * stride_ascale_k
            w_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(k_ptr.type.element_ty) # [BLOCK_SIZE_M, BLOCK_SIZE_N]

        if NUM_KSPLIT == 1:
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_h = offs_n // (P + V)
            offs_pv = offs_n % (P + V)

            # Write back the block of the output matrix k_nope with masks.
            k_nope_ptrs = (
                k_ptr
                + stride_k_out_m * offs_m[:, None]
                + stride_k_out_k * offs_h[None, :]
                + stride_k_out_n * offs_pv[None, :]
            )
            k_nope_mask = (
                (offs_m[:, None] < M)
                & (offs_h[None, :] < H)
                & (offs_pv[None, :] < P)
            )
            tl.store(k_nope_ptrs, c, mask=k_nope_mask)

            # Write back the block of the output matrix v with masks.
            v_ptrs = (
                v_ptr
                + stride_v_out_m * offs_m[:, None] 
                + stride_v_out_h * offs_h[None, :]
                + stride_v_out_v * (offs_pv[None, :] - P)
            )
            v_mask = (
                (offs_m[:, None] < M)
                & (offs_h[None, :] < H)
                & (offs_pv[None, :] >= P)
                & (offs_pv[None, :] < P + V)
            )
            tl.store(v_ptrs, c, mask=v_mask)

            # Handle k_pe
            offs_n = pid_n * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R).to(tl.int64)
            offs_h = offs_n // R
            offs_r = offs_n % R

            # Load k_pe
            k_pe_ptrs = (
                k_pe_ptr
                + stride_k_pe_m * offs_m[:, None]
                + stride_k_pe_h * offs_h[None, :]
                + stride_k_pe_r * offs_r[None, :]
            )
            k_pe_mask = (
                (offs_m[:, None] < M)
                & (offs_h[None, :] < H)
                & (offs_r[None, :] < R)
            )
            k_pe = tl.load(k_pe_ptrs, mask=k_pe_mask)

            # Concat k_pe to the output matrix k.
            k_pe_ptrs = (
                k_ptr
                + stride_k_out_m * offs_m[:, None]
                + stride_k_out_k * offs_h[None, :]
                + stride_k_out_n * (offs_r[None, :] + P)
            )
            tl.store(k_pe_ptrs, k_pe, mask=k_pe_mask)
        else:
            # SPLIT K
            # Write back the block of the output matrix C with masks.
            offs_k_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_k_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            k_out_ptrs = (
                k_ptr
                + stride_k_out_m * offs_k_out_m[:, None]
                + stride_k_out_n * offs_k_out_n[None, :]
                + stride_k_out_k * pid_k
            )
            k_mask = (offs_k_out_m[:, None] < M) & (offs_k_out_n[None, :] < N)
            tl.store(k_out_ptrs, c, mask=k_mask)



@triton.jit
def _fused_kv_proj_cat_reduce_kernel(
    c_ptr,
    k_ptr,
    v_ptr,
    k_pe_ptr,
    M,
    N,
    H,
    R,
    P,
    V,
    stride_c_k,
    stride_c_m,
    stride_c_n,
    stride_k_m,
    stride_k_h,
    stride_k_pr,
    stride_v_m,
    stride_v_h,
    stride_v_v,
    stride_k_pe_m,
    stride_k_pe_h,
    stride_k_pe_r,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    tl.assume(stride_c_k > 0)
    tl.assume(stride_c_m > 0)
    tl.assume(stride_c_n > 0)
    tl.assume(stride_k_m > 0)
    tl.assume(stride_k_pr > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_ptrs = (
        c_ptr
        + (offs_k[:, None, None] * stride_c_k)
        + (offs_m[None, :, None] * stride_c_m)
        + (offs_n[None, None, :] * stride_c_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_ptrs)
    else:
        c = tl.load(c_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0)

    c = c.to(k_ptr.type.element_ty)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_h = offs_n // (P + V)
    offs_pv = offs_n % (P + V)

    # store k_nope to output matrix k
    k_nope_ptrs = (
        k_ptr
        + stride_k_m * offs_m[:, None]
        + stride_k_h * offs_h[None, :]
        + stride_k_pr * offs_pv[None, :]
    )
    k_nope_mask = (
        (offs_m[:, None] < M)
        & (offs_h[None, :] < H)
        & (offs_pv[None, :] < P)
    )
    tl.store(k_nope_ptrs, c, mask=k_nope_mask)

    # store v to output matrix v
    v_ptrs = (
        v_ptr
        + stride_v_m * offs_m[:, None] 
        + stride_v_h * offs_h[None, :]
        + stride_v_v * (offs_pv[None, :] - P)
    )
    v_mask = (
        (offs_m[:, None] < M)
        & (offs_h[None, :] < H)
        & (offs_pv[None, :] >= P)
        & (offs_pv[None, :] < P + V)
    )
    tl.store(v_ptrs, c, mask=v_mask)

    # handle k_pe
    offs_n = pid_n * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_h = offs_n // R
    offs_r = offs_n % R

    # load k_pe
    k_pe_ptrs = (
        k_pe_ptr
        + stride_k_pe_m * offs_m[:, None]
        + stride_k_pe_h * offs_h[None, :]
        + stride_k_pe_r * offs_r[None, :]
    )
    k_pe_mask = (
        (offs_m[:, None] < M)
        & (offs_h[None, :] < H)
        & (offs_r[None, :] < R)
    )
    k_pe = tl.load(k_pe_ptrs, mask=k_pe_mask)

    # concat k_pe to k_nope
    k_pe_ptrs = (
        k_ptr
        + stride_k_m * offs_m[:, None]
        + stride_k_h * offs_h[None, :]
        + stride_k_pr * (offs_r[None, :] + P)
    )
    tl.store(k_pe_ptrs, k_pe, mask=k_pe_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
) -> dict:
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if M < 32 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif M <= 128:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32 and "medium_M32" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64 and "medium_M64" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128 and "medium_M128" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["medium_M128"]
    elif M <= 256 and "large" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["large"]
    else:
        BLK_M = triton.next_power_of_2(M)
        if f"xlarge_M{BLK_M}" in _get_config._config_dict[key]:
            return _get_config._config_dict[key][f"xlarge_M{BLK_M}"]
        elif "xlarge" in _get_config._config_dict[key]:
            return _get_config._config_dict[key]["xlarge"]

    return _get_config._config_dict[key]["any"]
