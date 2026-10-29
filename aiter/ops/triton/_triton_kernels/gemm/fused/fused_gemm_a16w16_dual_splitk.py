# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import (
    pid_grid,
    remap_xcd,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_fused_gemm_a16w16_dual_splitk_repr = make_kernel_repr(
    "_fused_gemm_a16w16_dual_splitk_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N1",
        "BLOCK_SIZE_K1",
        "NUM_KSPLIT1",
        "BLOCK_SIZE_N2",
        "BLOCK_SIZE_K2",
        "NUM_KSPLIT2",
        "GROUP_SIZE_M",
        "EVEN_K1",
        "EVEN_K2",
        "cache_modifier",
    ],
)


@triton.jit
def _dual_splitk_tile(
    a_ptr,
    b_ptr,
    c_ptr,
    pid_unified,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """One [BLOCK_SIZE_M, BLOCK_SIZE_N] tile (or one split-K partial of it) of
    C = A x B, laid out as in _gemm_a16_w16_kernel."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    if NUM_KSPLIT == 1:
        pid = remap_xcd(pid, num_pid_m * num_pid_n)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    # A split past K stores a zero partial, so every NUM_KSPLIT row is defined.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if split_k_start < K:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        if EVEN_MN:
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        else:
            offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        k_span = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K) - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)
        for k in range(num_k_iter):
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
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.type.element_ty)
    offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64
    )
    offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(
        tl.int64
    )
    c_ptrs = (
        c_ptr
        + stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
        + pid_k * stride_ck
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.heuristics(
    {
        "EVEN_K1": lambda args: args["K"] % args["BLOCK_SIZE_K1"] == 0
        and args["SPLITK_BLOCK_SIZE1"] % args["BLOCK_SIZE_K1"] == 0,
        "EVEN_K2": lambda args: args["K"] % args["BLOCK_SIZE_K2"] == 0
        and args["SPLITK_BLOCK_SIZE2"] % args["BLOCK_SIZE_K2"] == 0,
        "EVEN_MN1": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N1"] % args["BLOCK_SIZE_N1"] == 0,
        "EVEN_MN2": lambda args: args["M"] % args["BLOCK_SIZE_M"] == 0
        and args["N2"] % args["BLOCK_SIZE_N2"] == 0,
    }
)
@triton.jit(repr=_fused_gemm_a16w16_dual_splitk_repr)
def _fused_gemm_a16w16_dual_splitk_kernel(
    a_ptr,
    b_ptr,
    c1_ptr,
    c2_ptr,
    M,
    N1,
    N2,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_c1_k,
    stride_c1_m,
    stride_c1_n,
    stride_c2_k,
    stride_c2_m,
    stride_c2_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    NUM_KSPLIT1: tl.constexpr,
    SPLITK_BLOCK_SIZE1: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
    NUM_KSPLIT2: tl.constexpr,
    SPLITK_BLOCK_SIZE2: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K1: tl.constexpr,
    EVEN_K2: tl.constexpr,
    EVEN_MN1: tl.constexpr,
    EVEN_MN2: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """C1 = A x B[:, :N1] and C2 = A x B[:, N1:] in one launch, each part with its
    own tile and split-K. The grid is max(part-1 grid, part-2 grid): a program runs its
    part-1 tile, then its part-2 tile, skipping either past that part's grid."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_c1_k >= 0)
    tl.assume(stride_c1_m > 0)
    tl.assume(stride_c1_n > 0)
    tl.assume(stride_c2_k >= 0)
    tl.assume(stride_c2_m > 0)
    tl.assume(stride_c2_n > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid1 = num_pid_m * tl.cdiv(N1, BLOCK_SIZE_N1) * NUM_KSPLIT1
    grid2 = num_pid_m * tl.cdiv(N2, BLOCK_SIZE_N2) * NUM_KSPLIT2
    if pid < grid1:
        _dual_splitk_tile(
            a_ptr,
            b_ptr,
            c1_ptr,
            pid,
            M,
            N1,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_c1_k,
            stride_c1_m,
            stride_c1_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N1,
            BLOCK_SIZE_K1,
            GROUP_SIZE_M,
            NUM_KSPLIT1,
            SPLITK_BLOCK_SIZE1,
            EVEN_K1,
            EVEN_MN1,
            cache_modifier,
        )
    if pid < grid2:
        _dual_splitk_tile(
            a_ptr,
            b_ptr + N1 * stride_bn,
            c2_ptr,
            pid,
            M,
            N2,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_c2_k,
            stride_c2_m,
            stride_c2_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N2,
            BLOCK_SIZE_K2,
            1,
            NUM_KSPLIT2,
            SPLITK_BLOCK_SIZE2,
            EVEN_K2,
            EVEN_MN2,
            cache_modifier,
        )


_fused_gemm_a16w16_dual_splitk_reduce_repr = make_kernel_repr(
    "_fused_gemm_a16w16_dual_splitk_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT1",
        "MAX_KSPLIT1",
        "ACTUAL_KSPLIT2",
        "MAX_KSPLIT2",
    ],
)


@triton.jit
def _dual_splitk_reduce_tile(
    c_in_ptr,
    c_out_ptr,
    pid_m,
    pid_n,
    M,
    N,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
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
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0)
    c = tl.sum(c, axis=0).to(c_out_ptr.type.element_ty)
    c_out_ptrs = (
        c_out_ptr
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_out_ptrs, c, mask=c_mask)


@triton.jit(repr=_fused_gemm_a16w16_dual_splitk_reduce_repr)
def _fused_gemm_a16w16_dual_splitk_reduce_kernel(
    c1_in_ptr,
    c1_out_ptr,
    c2_in_ptr,
    c2_out_ptr,
    M,
    N1,
    N2,
    stride_c1_in_k,
    stride_c1_in_m,
    stride_c1_in_n,
    stride_c1_out_m,
    stride_c1_out_n,
    stride_c2_in_k,
    stride_c2_in_m,
    stride_c2_in_n,
    stride_c2_out_m,
    stride_c2_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT1: tl.constexpr,
    MAX_KSPLIT1: tl.constexpr,
    ACTUAL_KSPLIT2: tl.constexpr,
    MAX_KSPLIT2: tl.constexpr,
):
    """Sums the split-K partials of each part that was split (ACTUAL_KSPLIT > 1)."""
    tl.assume(stride_c1_out_m > 0)
    tl.assume(stride_c1_out_n > 0)
    tl.assume(stride_c2_in_k > 0)
    tl.assume(stride_c2_in_m > 0)
    tl.assume(stride_c2_in_n > 0)
    tl.assume(stride_c2_out_m > 0)
    tl.assume(stride_c2_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    if ACTUAL_KSPLIT1 > 1:
        num_pid_n1 = tl.cdiv(N1, BLOCK_SIZE_N)
        if pid_n < num_pid_n1:
            _dual_splitk_reduce_tile(
                c1_in_ptr,
                c1_out_ptr,
                pid_m,
                pid_n,
                M,
                N1,
                stride_c1_in_k,
                stride_c1_in_m,
                stride_c1_in_n,
                stride_c1_out_m,
                stride_c1_out_n,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                ACTUAL_KSPLIT1,
                MAX_KSPLIT1,
            )
        else:
            _dual_splitk_reduce_tile(
                c2_in_ptr,
                c2_out_ptr,
                pid_m,
                pid_n - num_pid_n1,
                M,
                N2,
                stride_c2_in_k,
                stride_c2_in_m,
                stride_c2_in_n,
                stride_c2_out_m,
                stride_c2_out_n,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                ACTUAL_KSPLIT2,
                MAX_KSPLIT2,
            )
    else:
        _dual_splitk_reduce_tile(
            c2_in_ptr,
            c2_out_ptr,
            pid_m,
            pid_n,
            M,
            N2,
            stride_c2_in_k,
            stride_c2_in_m,
            stride_c2_in_n,
            stride_c2_out_m,
            stride_c2_out_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            ACTUAL_KSPLIT2,
            MAX_KSPLIT2,
        )


def _get_config(
    M: int,
    N1: int,
    N2: int,
    K: int,
):
    # Custom file naming: N1={N1}-N2={N2}-K={K}
    return get_gemm_config(
        "FUSED-GEMM-A16W16-DUAL_SPLITK",
        M,
        specialized_filename=f"N1={N1}-N2={N2}-K={K}",
    )
