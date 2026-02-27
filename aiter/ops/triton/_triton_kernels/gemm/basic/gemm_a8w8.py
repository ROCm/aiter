# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config, compute_splitk_params

import triton

_gemm_a8w8_repr = make_kernel_repr(
    "_gemm_a8w8_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
        "HAS_BIAS",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a8w8_repr)
def _gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
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
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N) if NUM_KSPLIT==1, (NUM_KSPLIT, M, N) otherwise.
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    # remap so that XCDs get continous chunks of pids (of CHUNK_SIZE).
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
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
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Create pointers for the scale tensors and load them
        offs_a_scale = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) % M
        offs_b_scale = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
        a_scale = tl.load(a_scale_ptr + offs_a_scale)
        b_scale = tl.load(b_scale_ptr + offs_b_scale)

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
                b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

            accumulator += tl.dot(a, b, input_precision="ieee")

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Apply scale
        accumulator *= a_scale[:, None] * b_scale[None, :]

        # Add bias (only when not splitting K; bias is added after reduce)
        if HAS_BIAS:
            offs_bias = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) % N
            bias = tl.load(bias_ptr + offs_bias)
            accumulator = accumulator.to(bias_ptr.type.element_ty) + bias[None, :]

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_gemm_a8w8_reduce_repr = make_kernel_repr(
    "_gemm_a8w8_reduce_kernel",
    [
        "HAS_BIAS",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
    ],
)


@triton.heuristics({})
@triton.jit(repr=_gemm_a8w8_reduce_repr)
def _gemm_a8w8_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    bias_ptr,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

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

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n)
        c = c + bias[None, :]

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


def _get_config(
    M: int,
    N: int,
    K: int,
):
    config, is_tunned = get_gemm_config("GEMM-A8W8", M, N, K)
    return compute_splitk_params(config, K), is_tunned
