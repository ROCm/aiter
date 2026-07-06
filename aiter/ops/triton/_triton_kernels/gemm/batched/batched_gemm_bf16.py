# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)

import triton

_batched_gemm_bf16_repr = make_kernel_repr(
    "_batched_gemm_bf16_kernel",
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
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_batched_gemm_bf16_repr)
def _batched_gemm_bf16_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    # stride along the split-K (partial-sum) dimension of the output; 0 when
    # NUM_KSPLIT == 1 (the output is the final (B, M, N) tensor).
    stride_ck,
    stride_biasb,
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
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call batched_gemm_bf16 function
    below

    Computes the matmul C[i] = A[i] x B[i] for every i in a given batch and optionally adds a bias to each result.

    When NUM_KSPLIT > 1 the K dimension is split across NUM_KSPLIT program
    groups: each group reduces a SPLITK_BLOCK_SIZE-wide slice of K into its own
    partial-sum plane and the planes are summed by the split-K reduce kernel.

    Key parameters:
    - A: Batch tensor A with shape (B, M, K).
    - B: Batch tensor B with shape (B, K, N).
    - C: Batch tensor C with shape (B, M, N), or partial sums with shape
         (NUM_KSPLIT, B, M, N) when NUM_KSPLIT > 1.
    - Bias: Bias batch tensor with shape (B, 1, N).
    """

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
    pid_unified = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT

    if NUM_KSPLIT == 1:
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
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    # Cast batch id and batch dimension strides to int64 to avoid int32 overflow during offset calculation
    # Note: If you're attempting to cast strides to int64 to prevent integer overflow, use `tl.cast` instead of `.to()`.
    # See https://github.com/ROCm/aiter/pull/597 for rationale
    batch_id = tl.cast(batch_id, tl.int64)
    stride_ab = tl.cast(stride_ab, tl.int64)
    stride_bb = tl.cast(stride_bb, tl.int64)
    stride_cb = tl.cast(stride_cb, tl.int64)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Create pointers for first block of A and B input matrices
        # Cast M/N offsets to int64 to avoid int32 overflow in offs*stride
        # products (offs_am*stride_am, offs_cm*stride_cm). Large per-batch
        # matrices (M*K or M*N > 2^31) overflow otherwise.
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = tl.cast(
            (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, tl.int64
        )
        offs_bn = tl.cast(
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, tl.int64
        )
        a_ptrs = a_ptr + (
            batch_id * stride_ab
            + offs_am[:, None] * stride_am
            + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            batch_id * stride_bb
            + offs_k_split[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        for k in range(num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                    other=0.0,
                    cache_modifier=cache_modifier,
                )

            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Add bias only on the non-split path; for NUM_KSPLIT > 1 the bias is
        # added once during the split-K reduction so it is not counted per-split.
        if HAS_BIAS and NUM_KSPLIT == 1:
            offs_bias = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            bias = tl.load(bias_ptr + batch_id * stride_biasb + offs_bias).to(
                dtype=acc_dtype
            )
            accumulator += bias[None, :]

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        # int64 to avoid int32 overflow in offs_cm*stride_cm for large M*N.
        offs_cm = tl.cast(pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), tl.int64)
        offs_cn = tl.cast(pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N), tl.int64)
        c_ptrs = (
            c_ptr
            + pid_k * stride_ck
            + stride_cb * batch_id
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
):

    # BF16 uses the shared 16-bit activation / 16-bit weight batched GEMM config.
    config, is_tunned = get_gemm_config("BATCHED_GEMM-A16W16", M, N, K)
    return compute_splitk_params(config, K), is_tunned
