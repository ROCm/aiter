# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import os
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


_fused_gemm_a8w8_blockscale_mul_add_repr = make_kernel_repr(
    "_fused_gemm_a8w8_blockscale_mul_add_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "cache_modifier",
        "NUM_KSPLIT",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
    }
)
@triton.jit(repr=_fused_gemm_a8w8_blockscale_mul_add_repr)
def _fused_gemm_a8w8_blockscale_mul_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_a_ptr,
    c_b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    stride_cam,
    stride_can,
    stride_cbm,
    stride_cbn,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    IS_A_TENSOR: tl.constexpr,
    IS_B_TENSOR: tl.constexpr,
    FUSE_TYPE: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are in the microscale fp4 (mxfp4) format.
    A_scales and B_scales are in e8m0 format.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)
    tl.assume(stride_cam > 0)
    tl.assume(stride_can > 0)
    tl.assume(stride_cbm > 0)
    tl.assume(stride_cbn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        # TODO: fix remap
        # remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Create pointers for first block of A and B input matrices
        # The BLOCK sizes are of the elements and in fp4 we pack 2 per uint8 container.
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0,
                    cache_modifier=cache_modifier,
                )

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += (
                tl.dot(a, b, input_precision="ieee")
                * a_scale[:, None]
                * b_scale[None, :]
            )

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        if NUM_KSPLIT == 1:
            if IS_A_SCALAR and IS_A_TENSOR:
                c_a = tl.load(c_a_ptr)
            elif IS_A_SCALAR:
                c_a = c_a_ptr
            else:
                c_a = tl.load(
                    c_a_ptr
                    + stride_cam * offs_cm[:, None]
                    + stride_can * offs_cn[None, :],
                    mask=c_mask,
                )
            c_a = c_a.to(tl.float32)

            if IS_B_SCALAR and IS_B_TENSOR:
                c_b = tl.load(c_b_ptr)
            elif IS_B_SCALAR:
                c_b = c_b_ptr
            else:
                c_b = tl.load(
                    c_b_ptr
                    + stride_cbm * offs_cm[:, None]
                    + stride_cbn * offs_cn[None, :],
                    mask=c_mask,
                )
            c_b = c_b.to(tl.float32)

            if FUSE_TYPE == 0:
                accumulator = c_a * accumulator + c_b
            else:
                accumulator = c_b * c_a + accumulator

        c = accumulator.to(c_ptr.type.element_ty)

        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        tl.store(c_ptrs, c, mask=c_mask)


_fused_gemm_a8w8_blockscale_mul_add_reduce_repr = make_kernel_repr(
    "_fused_gemm_a8w8_blockscale_mul_add_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
    ],
)


@triton.heuristics({})  # dummy heuristics to invoke kernel re-naming
@triton.jit(repr=_fused_gemm_a8w8_blockscale_mul_add_reduce_repr)
def _fused_gemm_a8w8_blockscale_mul_add_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    c_a_ptr,
    c_b_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    stride_cam,
    stride_can,
    stride_cbm,
    stride_cbn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    IS_A_SCALAR: tl.constexpr,
    IS_B_SCALAR: tl.constexpr,
    IS_A_TENSOR: tl.constexpr,
    IS_B_TENSOR: tl.constexpr,
    FUSE_TYPE: tl.constexpr,
):

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
    c = tl.sum(c, axis=0)

    if IS_A_SCALAR and IS_A_TENSOR:
        c_a = tl.load(c_a_ptr)
    elif IS_A_SCALAR:
        c_a = c_a_ptr
    else:
        c_a = tl.load(
            c_a_ptr + stride_cam * offs_m[:, None] + stride_can * offs_n[None, :]
        )
    c_a = c_a.to(tl.float32)

    if IS_B_SCALAR and IS_B_TENSOR:
        c_b = tl.load(c_b_ptr)
    elif IS_B_SCALAR:
        c_b = c_b_ptr
    else:
        c_b = tl.load(
            c_b_ptr + stride_cbm * offs_m[:, None] + stride_cbn * offs_n[None, :]
        )
    c_b = c_b.to(tl.float32)

    if FUSE_TYPE == 0:
        c = c_a * c + c_b
    else:
        c = c_b * c_a + c
    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_arch()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_arch()
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
