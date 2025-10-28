# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import os
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_fp8_quant import _fp8_quant_op
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a16w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_bscale_k,
    stride_bscale_n,
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
    GRID_MN: tl.constexpr,
    PREQUANT: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
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

        # Create pointers for first block of A and B input matrices
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
        offs_ks = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        offs_bsn = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr + offs_ks * stride_bscale_k + offs_bsn * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            b_scale = tl.load(b_scale_ptrs)

            if PREQUANT:
                a, a_scale = _fp8_quant_op(
                    a, BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_K, DTYPE_MAX, DTYPE_MIN
                )
                a = a.to(b_ptr.type.element_ty).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K)
                a_scale = a_scale.reshape(BLOCK_SIZE_M)
                accumulator += (
                    tl.dot(a, b, input_precision="ieee")
                    * a_scale[:, None]
                    * b_scale[None, :]
                )
            else:
                b = b.to(a_ptr.type.element_ty)
                accumulator += tl.dot(a, b, input_precision="ieee") * b_scale[None, :]

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W8_BLOCKSCALE.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A16W8_BLOCKSCALE-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if M < 16 and "small" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small"]
    elif M < 32 and "small_M16" in _get_config._config_dict[key]:
        return _get_config._config_dict[key]["small_M16"]
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
