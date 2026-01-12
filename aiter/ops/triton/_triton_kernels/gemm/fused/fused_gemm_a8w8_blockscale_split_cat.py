# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
import functools
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_split_cat(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    y_ptr,
    c1_ptr,
    c2_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    D,
    S1,
    S2,
    S3,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_a_m` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_a_m,
    stride_a_k,
    stride_b_k,
    stride_b_n,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    stride_c1_k,
    stride_c1_m,
    stride_c1_n,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
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
    BLOCK_SIZE_S3: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_a_m > 0)
    tl.assume(stride_a_k > 0)
    tl.assume(stride_b_k > 0)
    tl.assume(stride_b_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)
    tl.assume(stride_c1_k > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_n > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_s > 0)
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
        offs_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_b_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_a_m[:, None] * stride_a_m + offs_k_split[None, :] * stride_a_k
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_b_k + offs_b_n[None, :] * stride_b_n
        )

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_a_m * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_b_n // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c1_ptr.type.element_ty != tl.int8 else tl.int32
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

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += (
                tl.dot(a, b, input_precision="ieee")
                * a_scale[:, None]
                * b_scale[None, :]
            )

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * stride_b_k

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c1_ptr.type.element_ty)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

        if NUM_KSPLIT == 1:
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_d = offs_n // (S1 + S2)
            offs_s = offs_n % (S1 + S2)

            # Write back the block of the output matrix C1 with masks.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * offs_s[None, :]
            )
            c1_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
            )
            tl.store(c1_ptrs, c, mask=c1_mask)

            # Write back the block of the output matrix C2 with masks.
            c2_ptrs = (
                c2_ptr
                + stride_c2_m * offs_m[:, None]
                + stride_c2_d * offs_d[None, :]
                + stride_c2_s * (offs_s[None, :] - S1)
            )
            c2_mask = (
                (offs_m[:, None] < M)
                & (offs_d[None, :] < D)
                & (offs_s[None, :] >= S1)
                & (offs_s[None, :] < S1 + S2)
            )
            tl.store(c2_ptrs, c, mask=c2_mask)

            # Handle y
            offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3).to(tl.int64)
            offs_d = offs_n // S3
            offs_s = offs_n % S3

            # Load y
            y_ptrs = (
                y_ptr
                + stride_y_m * offs_m[:, None]
                + stride_y_d * offs_d[None, :]
                + stride_y_s * offs_s[None, :]
            )
            y_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
            )
            y = tl.load(y_ptrs, mask=y_mask)

            # Concat y to the output matrix C1.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * (offs_s[None, :] + S1)
            )
            tl.store(c1_ptrs, y, mask=y_mask)
        else:
            # SPLIT K
            # Write back the block of the output matrix C with masks.
            offs_c1_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_c1_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_c1_m[:, None]
                + stride_c1_n * offs_c1_n[None, :]
                + stride_c1_k * pid_k
            )
            c1_mask = (offs_c1_m[:, None] < M) & (offs_c1_n[None, :] < N)
            tl.store(c1_ptrs, c, mask=c1_mask)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_preshuffle_split_cat(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    y_ptr,
    c1_ptr,
    c2_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    D,
    S1,
    S2,
    S3,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_a_m` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_a_m,
    stride_a_k,
    stride_b_n,
    stride_b_k,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    stride_c1_k,
    stride_c1_m,
    stride_c1_n,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
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
    BLOCK_SIZE_S3: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    tl.assume(stride_a_m > 0)
    tl.assume(stride_a_k > 0)
    tl.assume(stride_b_k > 0)
    tl.assume(stride_b_n > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)
    tl.assume(stride_c1_k > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_n > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_s > 0)
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
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_k_shuffle = pid_k * SPLITK_BLOCK_SIZE * 16 + offs_k_shuffle_arr

        offs_a_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_b_n = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % N
        a_ptrs = a_ptr + (
            offs_a_m[:, None] * stride_a_m + offs_k_split[None, :] * stride_a_k
        )
        b_ptrs = b_ptr + (
            offs_b_n[:, None] * stride_b_n + offs_k_shuffle[None, :] * stride_b_k
        )

        # Create pointers for the scales
        offs_bs_n = (pid_n * (BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_a_m * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bs_n // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c1_ptr.type.element_ty != tl.int8 else tl.int32
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

            b = (
                b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 32,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
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
            a_ptrs += BLOCK_SIZE_K * stride_a_k
            b_ptrs += BLOCK_SIZE_K * 16 * stride_b_k

            # k_cur = k * BLOCK_SIZE_K // GROUP_K
            # k_nxt = (k + 1) * BLOCK_SIZE_K // GROUP_K
            # offs_ks = k_nxt - k_cur
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c1_ptr.type.element_ty)  # [BLOCK_SIZE_M, BLOCK_SIZE_N]

        if NUM_KSPLIT == 1:
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            offs_d = offs_n // (S1 + S2)
            offs_s = offs_n % (S1 + S2)

            # Write back the block of the output matrix C1 with masks.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * offs_s[None, :]
            )
            c1_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
            )
            tl.store(c1_ptrs, c, mask=c1_mask)

            # Write back the block of the output matrix C2 with masks.
            c2_ptrs = (
                c2_ptr
                + stride_c2_m * offs_m[:, None]
                + stride_c2_d * offs_d[None, :]
                + stride_c2_s * (offs_s[None, :] - S1)
            )
            c2_mask = (
                (offs_m[:, None] < M)
                & (offs_d[None, :] < D)
                & (offs_s[None, :] >= S1)
                & (offs_s[None, :] < S1 + S2)
            )
            tl.store(c2_ptrs, c, mask=c2_mask)

            # Handle y
            offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3).to(tl.int64)
            offs_d = offs_n // S3
            offs_s = offs_n % S3

            # Load y
            y_ptrs = (
                y_ptr
                + stride_y_m * offs_m[:, None]
                + stride_y_d * offs_d[None, :]
                + stride_y_s * offs_s[None, :]
            )
            y_mask = (
                (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
            )
            y = tl.load(y_ptrs, mask=y_mask)

            # Concat y to the output matrix C1.
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_m[:, None]
                + stride_c1_k * offs_d[None, :]
                + stride_c1_n * (offs_s[None, :] + S1)
            )
            tl.store(c1_ptrs, y, mask=y_mask)
        else:
            # SPLIT K
            # Write back the block of the output matrix C with masks.
            offs_c1_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
            offs_c1_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            c1_ptrs = (
                c1_ptr
                + stride_c1_m * offs_c1_m[:, None]
                + stride_c1_n * offs_c1_n[None, :]
                + stride_c1_k * pid_k
            )
            c1_mask = (offs_c1_m[:, None] < M) & (offs_c1_n[None, :] < N)
            tl.store(c1_ptrs, c, mask=c1_mask)


@triton.jit
def _fused_gemm_a8w8_blockscale_split_cat_reduce(
    c_ptr,
    c1_ptr,
    c2_ptr,
    y_ptr,
    M,
    N,
    D,
    S1,
    S2,
    S3,
    stride_c_k,
    stride_c_m,
    stride_c_n,
    stride_c1_m,
    stride_c1_d,
    stride_c1_s,
    stride_c2_m,
    stride_c2_d,
    stride_c2_s,
    stride_y_m,
    stride_y_d,
    stride_y_s,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_S3: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    tl.assume(stride_c_k > 0)
    tl.assume(stride_c_m > 0)
    tl.assume(stride_c_n > 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_d > 0)
    tl.assume(stride_c1_s > 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_d > 0)
    tl.assume(stride_c2_s > 0)
    tl.assume(stride_y_m > 0)
    tl.assume(stride_y_s > 0)

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

    c = c.to(c1_ptr.type.element_ty)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_d = offs_n // (S1 + S2)
    offs_s = offs_n % (S1 + S2)

    # store c to output matrix c1
    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_s * offs_s[None, :]
    )
    c1_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S1)
    tl.store(c1_ptrs, c, mask=c1_mask)

    # store c to output matrix c2
    c2_ptrs = (
        c2_ptr
        + stride_c2_m * offs_m[:, None]
        + stride_c2_d * offs_d[None, :]
        + stride_c2_s * (offs_s[None, :] - S1)
    )
    c2_mask = (
        (offs_m[:, None] < M)
        & (offs_d[None, :] < D)
        & (offs_s[None, :] >= S1)
        & (offs_s[None, :] < S1 + S2)
    )
    tl.store(c2_ptrs, c, mask=c2_mask)

    # handle y
    offs_n = pid_n * BLOCK_SIZE_S3 + tl.arange(0, BLOCK_SIZE_S3)
    offs_d = offs_n // S3
    offs_s = offs_n % S3

    # load y
    y_ptrs = (
        y_ptr
        + stride_y_m * offs_m[:, None]
        + stride_y_d * offs_d[None, :]
        + stride_y_s * offs_s[None, :]
    )
    y_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D) & (offs_s[None, :] < S3)
    y = tl.load(y_ptrs, mask=y_mask)

    # concat y to c1
    c1_ptrs = (
        c1_ptr
        + stride_c1_m * offs_m[:, None]
        + stride_c1_d * offs_d[None, :]
        + stride_c1_s * (offs_s[None, :] + S1)
    )
    tl.store(c1_ptrs, y, mask=y_mask)


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
) -> dict:
    shuffle_suffix = "_PRESHUFFLED" if shuffle else ""
    config_name = f"GEMM-A8W8_BLOCKSCALE{shuffle_suffix}"

    return get_gemm_config(config_name, M, N, K)
