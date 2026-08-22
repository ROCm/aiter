# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)

import triton

_fused_gemm_a16w16_copy_x_repr = make_kernel_repr(
    "_fused_gemm_a16w16_copy_x_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "EVEN_MN",
        "cache_modifier",
        "activation",
        "use_activation",
        "ADD_BIAS",
        "SKIP_REDUCE",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % (args["SPLITK_BLOCK_SIZE"]) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
        "EVEN_MN": lambda args: (args["M"] % args["BLOCK_SIZE_M"] == 0)
        and (args["N"] % args["BLOCK_SIZE_N"] == 0),
    }
)
@triton.jit(
    repr=_fused_gemm_a16w16_copy_x_repr,
    do_not_specialize=["M", "N"],
)
def _fused_gemm_a16w16_copy_x_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    a_copy_ptr,
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
    stride_a_copy_m,
    stride_a_copy_k,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
    activation: tl.constexpr,
    use_activation: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
):
    """Kernel that computes C = A x B and also emits a downcasted copy of A.

    The grid is laid out as a single 1D program-id space split into two
    contiguous regions:

      * `[0, NUM_KSPLIT * num_pid_m * num_pid_n)` runs the GEMM (identical to
        the unfused a16w16 kernel).
      * `[NUM_KSPLIT * num_pid_m * num_pid_n, GEMM_GRID + num_pid_m * num_pid_k_copy)`
        runs the pure downcast-copy of A: each program handles one
        `BLOCK_SIZE_M x BLOCK_SIZE_K` tile of A and writes it to `a_copy_ptr`.

    Decoupling the copy from the GEMM body keeps the GEMM hot loop unchanged
    and lets the copy programs schedule independently.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_a_copy_m > 0)
    tl.assume(stride_a_copy_k > 0)

    pid_unified = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k_copy = tl.cdiv(K, BLOCK_SIZE_K)
    GEMM_GRID = num_pid_m * num_pid_n * NUM_KSPLIT

    if pid_unified < GEMM_GRID:
        # ---- GEMM branch ----------------------------------------------------
        pid_unified = remap_xcd(pid_unified, GEMM_GRID, NUM_XCDS=8)
        pid_k = pid_unified % NUM_KSPLIT
        pid = pid_unified // NUM_KSPLIT

        if NUM_KSPLIT == 1:
            pid_m, pid_n = pid_grid(
                pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M
            )
        else:
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)
        tl.assume(pid_k >= 0)

        split_k_start = pid_k * SPLITK_BLOCK_SIZE
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

            acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
            if ADD_BIAS:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator = tl.load(bias_ptr + offs_bn).to(dtype=acc_dtype)
                    accumulator = tl.broadcast_to(
                        accumulator[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

            split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
            k_span = split_k_end - split_k_start
            num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

            for k in range(num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k[None, :] < k_span - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < k_span - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if use_activation and NUM_KSPLIT == 1:
                accumulator = activation(accumulator)

            c = accumulator.to(c_ptr.type.element_ty)
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr
                + stride_cm * offs_cm[:, None]
                + stride_cn * offs_cn[None, :]
                + pid_k * stride_ck
            )
            if EVEN_MN:
                tl.store(c_ptrs, c)
            else:
                c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
                tl.store(c_ptrs, c, mask=c_mask)
    else:
        # ---- Downcast-copy branch ------------------------------------------
        pid_copy = pid_unified - GEMM_GRID
        pid_m = pid_copy // num_pid_k_copy
        pid_k = pid_copy % num_pid_k_copy

        tl.assume(pid_m >= 0)
        tl.assume(pid_k >= 0)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_copy_ptrs = a_copy_ptr + (
            offs_m[:, None] * stride_a_copy_m + offs_k[None, :] * stride_a_copy_k
        )

        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        tl.store(a_copy_ptrs, a.to(a_copy_ptr.type.element_ty), mask=mask)


def _get_config(
    M: int,
    N: int,
    K: int,
):
    # Use the same tuning portal as the unfused gemm_a16w16 — the extra
    # downcast copy is assumed not to shift the optimal config.
    config, is_tunned = get_gemm_config("GEMM-A16W16", M, N, K)
    return compute_splitk_params(config, K), is_tunned
