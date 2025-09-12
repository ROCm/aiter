# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import os
import triton
import triton.language as tl
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
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_cm,
    stride_in_cn,
    stride_in_biasb,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    DTYPE_MIN: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    cache_modifier: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_a8w8 function
    below

    Computes the matmul C[i] = A[i] x B[i] and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N).
    - A_scale: First scale batch tensor with shape (B, M, 1).
    - B_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_biasb = tl.cast(stride_in_biasb, tl.int64)

    tl.assume(stride_ab > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cb > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_biasb > 0)

    # -----------------------------------------------------------
    # Get batch program id
    batch_id = tl.program_id(axis=0)
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    batch_id = tl.cast(batch_id, tl.int64)
    pid_m = tl.cast(pid_m, tl.int64)
    pid_n = tl.cast(pid_n, tl.int64)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(batch_id >= 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (
        batch_id * stride_ab
        + offs_am[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = b_ptr + (
        batch_id * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    one_over_DTYPE_MAX = 1.0 / DTYPE_MAX
    b_scale = tl.load(b_scale_ptr)

    acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        m = tl.maximum(tl.max(tl.abs(a), axis=-1), 1e-10)[:, None]
        a_scale = m.to(tl.float32) * one_over_DTYPE_MAX
        a_scale_recip = 1.0 / a_scale
        a = tl.clamp(a * a_scale_recip, DTYPE_MIN, DTYPE_MAX).to(b_ptr.dtype.element_ty)

        accumulator += tl.dot(a, b, input_precision="ieee") * a_scale

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator *= b_scale

    if HAS_BIAS:
        offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias)
        accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (
        c_ptr
        + stride_cb * batch_id
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
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
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    key = f"{N}_{K}"
    if key not in _get_config._config_dict.keys():
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N={N}-K={K}.json"
        if os.path.exists(fpath):
            with open(fpath, "r") as file:
                config = json.load(file)
                _get_config._config_dict[key] = config
        else:
            key = "default"  # fall back to default config

    if M <= 16:
        return _get_config._config_dict[key]["small"]
    elif M <= 128:
        BLK_M = triton.next_power_of_2(M)
        if BLK_M == 32:
            return _get_config._config_dict[key]["medium_M32"]
        elif BLK_M == 64:
            return _get_config._config_dict[key]["medium_M64"]
        elif BLK_M == 128:
            return _get_config._config_dict[key]["medium_M128"]
    elif M <= 256:
        return _get_config._config_dict[key]["large"]
    else:
        return _get_config._config_dict[key]["xlarge"]
