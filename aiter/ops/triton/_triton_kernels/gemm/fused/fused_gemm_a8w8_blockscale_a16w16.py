# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils.gemm_config_utils import get_gemm_config


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN_FP8": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp8"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_kernel(
    # Pointers to matrices
    a_fp8_ptr,
    b_fp8_ptr,
    bias_fp8_ptr,
    a_fp8_scale_ptr,
    b_fp8_scale_ptr,
    c_fp8_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp8,
    N_bf16,
    K,
    stride_a_fp8_m,
    stride_a_fp8_k,
    stride_b_fp8_k,
    stride_b_fp8_n,
    stride_a_fp8_scale_m,
    stride_a_fp8_scale_k,
    stride_b_fp8_scale_k,
    stride_b_fp8_scale_n,
    stride_c_fp8_k,
    stride_c_fp8_m,
    stride_c_fp8_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN_FP8: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp8_m > 0)
    tl.assume(stride_a_fp8_k > 0)
    tl.assume(stride_b_fp8_k > 0)
    tl.assume(stride_b_fp8_n > 0)
    tl.assume(stride_c_fp8_k > 0)
    tl.assume(stride_c_fp8_m > 0)
    tl.assume(stride_c_fp8_n > 0)
    tl.assume(stride_a_fp8_scale_m > 0)
    tl.assume(stride_a_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_k > 0)
    tl.assume(stride_b_fp8_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp8 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        GRID_MN: tl.constexpr = GRID_MN_FP8 + GRID_MN_BF16
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        acc_dtype = tl.float32 if c_fp8_ptr.type.element_ty != tl.int8 else tl.int32

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        if pid_n < num_pid_n_fp8:
            offs_b_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
            a_fp8_ptrs = a_fp8_ptr + (
                offs_am[:, None] * stride_a_fp8_m
                + offs_k_split[None, :] * stride_a_fp8_k
            )
            b_fp8_ptrs = b_fp8_ptr + (
                offs_k_split[:, None] * stride_b_fp8_k
                + offs_b_fp8_n[None, :] * stride_b_fp8_n
            )

            offs_ks = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
            a_scale_ptrs = (
                a_fp8_scale_ptr
                + offs_am * stride_a_fp8_scale_m
                + offs_ks * stride_a_fp8_scale_k
            )
            offs_bsn = offs_b_fp8_n // GROUP_N
            b_scale_ptrs = (
                b_fp8_scale_ptr
                + offs_ks * stride_b_fp8_scale_k
                + offs_bsn * stride_b_fp8_scale_n
            )

            if ADD_BIAS_FP8:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_fp8 = tl.load(bias_fp8_ptr + offs_b_fp8_n).to(
                        dtype=acc_dtype
                    )
                    accumulator_fp8 = tl.broadcast_to(
                        accumulator_fp8[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_fp8 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator_fp8 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_fp8_ptrs)
                    b = tl.load(b_fp8_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_fp8_ptrs,
                        mask=offs_k[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_fp8_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                a_scale = tl.load(a_scale_ptrs)
                b_scale = tl.load(b_scale_ptrs)

                accumulator_fp8 += (
                    tl.dot(a, b, input_precision="ieee")
                    * a_scale[:, None]
                    * b_scale[None, :]
                )

                a_fp8_ptrs += BLOCK_SIZE_K * stride_a_fp8_k
                b_fp8_ptrs += BLOCK_SIZE_K * stride_b_fp8_k
                a_scale_ptrs += offs_ks_step * stride_a_fp8_scale_k
                b_scale_ptrs += offs_ks_step * stride_b_fp8_scale_k

            c_fp8 = accumulator_fp8.to(c_fp8_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp8_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp8_ptrs = (
                c_fp8_ptr
                + stride_c_fp8_m * offs_cm[:, None]
                + stride_c_fp8_n * offs_c_fp8_n[None, :]
                + pid_k * stride_c_fp8_k
            )
            c_fp8_mask = (offs_cm[:, None] < M) & (offs_c_fp8_n[None, :] < N_fp8)
            tl.store(c_fp8_ptrs, c_fp8, mask=c_fp8_mask)
        else:
            pid_n -= num_pid_n_fp8

            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            if ADD_BIAS_BF16:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_bf16 = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                        dtype=acc_dtype
                    )
                    accumulator_bf16 = tl.broadcast_to(
                        accumulator_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_bf16 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                    )
            else:
                accumulator_bf16 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_bf16 += tl.dot(a, b, input_precision="ieee")

                a_ptrs += BLOCK_SIZE_K * stride_a_bf16_k
                b_ptrs += BLOCK_SIZE_K * stride_b_bf16_k

            c_bf16 = accumulator_bf16.to(c_bf16_ptr.type.element_ty)

            # Write back the block of the output matrix C with masks.
            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_bf16_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_bf16_ptrs = (
                c_bf16_ptr
                + stride_c_bf16_m * offs_cm[:, None]
                + stride_c_bf16_n * offs_c_bf16_n[None, :]
                + pid_k * stride_c_bf16_k
            )
            c_bf16_mask = (offs_cm[:, None] < M) & (offs_c_bf16_n[None, :] < N_bf16)
            tl.store(c_bf16_ptrs, c_bf16, mask=c_bf16_mask)


@triton.jit
def _fused_gemm_a8w8_blockscale_a16w16_reduce_kernel(
    bias_fp8_ptr,
    c_fp8_in_ptr,
    c_fp8_out_ptr,
    bias_bf16_ptr,
    c_bf16_in_ptr,
    c_bf16_out_ptr,
    M,
    N_fp8,
    N_bf16,
    stride_c_fp8_in_k,
    stride_c_fp8_in_m,
    stride_c_fp8_in_n,
    stride_c_fp8_out_m,
    stride_c_fp8_out_n,
    stride_c_bf16_in_k,
    stride_c_bf16_in_m,
    stride_c_bf16_in_n,
    stride_c_bf16_out_m,
    stride_c_bf16_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS_FP8: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
):

    tl.assume(stride_c_fp8_in_k > 0)
    tl.assume(stride_c_fp8_in_m > 0)
    tl.assume(stride_c_fp8_in_n > 0)
    tl.assume(stride_c_fp8_out_m > 0)
    tl.assume(stride_c_fp8_out_n > 0)

    tl.assume(stride_c_bf16_in_k > 0)
    tl.assume(stride_c_bf16_in_m > 0)
    tl.assume(stride_c_bf16_in_n > 0)
    tl.assume(stride_c_bf16_out_m > 0)
    tl.assume(stride_c_bf16_out_n > 0)

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

    num_pid_n_fp8 = tl.cdiv(N_fp8, BLOCK_SIZE_N)
    offs_k = tl.arange(0, MAX_KSPLIT)
    acc_dtype = tl.float32 if c_fp8_in_ptr.type.element_ty != tl.int8 else tl.int32

    if pid_n < num_pid_n_fp8:
        offs_fp8_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp8
        c_fp8_in_ptrs = (
            c_fp8_in_ptr
            + (offs_k[:, None, None] * stride_c_fp8_in_k)
            + (offs_m[None, :, None] * stride_c_fp8_in_m)
            + (offs_fp8_n[None, None, :] * stride_c_fp8_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_fp8 = tl.load(c_fp8_in_ptrs)
        else:
            c_fp8 = tl.load(
                c_fp8_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_fp8 = tl.sum(c_fp8, axis=0)
        if ADD_BIAS_FP8:
            bias_fp8 = tl.load(bias_fp8_ptr + offs_fp8_n).to(dtype=acc_dtype)
            bias_fp8 = tl.broadcast_to(bias_fp8[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
            c_fp8 += bias_fp8

        c_fp8 = c_fp8.to(c_fp8_out_ptr.type.element_ty)

        c_fp8_out_ptrs = (
            c_fp8_out_ptr
            + (offs_m[:, None] * stride_c_fp8_out_m)
            + (offs_fp8_n[None, :] * stride_c_fp8_out_n)
        )

        tl.store(c_fp8_out_ptrs, c_fp8)
    else:
        pid_n -= num_pid_n_fp8

        offs_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
        c_bf16_in_ptrs = (
            c_bf16_in_ptr
            + (offs_k[:, None, None] * stride_c_bf16_in_k)
            + (offs_m[None, :, None] * stride_c_bf16_in_m)
            + (offs_bf16_n[None, None, :] * stride_c_bf16_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_bf16 = tl.load(c_bf16_in_ptrs)
        else:
            c_bf16 = tl.load(
                c_bf16_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_bf16 = tl.sum(c_bf16, axis=0)
        if ADD_BIAS_BF16:
            bias_bf16 = tl.load(bias_bf16_ptr + offs_bf16_n).to(dtype=acc_dtype)
            bias_bf16 = tl.broadcast_to(
                bias_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
            )
            c_bf16 += bias_bf16

        c_bf16 = c_bf16.to(c_bf16_out_ptr.type.element_ty)

        c_bf16_out_ptrs = (
            c_bf16_out_ptr
            + (offs_m[:, None] * stride_c_bf16_out_m)
            + (offs_bf16_n[None, :] * stride_c_bf16_out_n)
        )
        c_bf16_mask = (offs_m[:, None] < M) & (offs_bf16_n[None, :] < N_bf16)
        tl.store(c_bf16_out_ptrs, c_bf16, mask=c_bf16_mask)


def _get_config(
    M: int,
    N_fp8: int,
    N_bf16: int,
    K: int,
):

    # Custom file naming: N8={N_fp8}-N16={N_bf16}-K={K}
    specialized_filename = f"N8={N_fp8}-N16={N_bf16}-K={K}"
    return get_gemm_config(
        "FUSED-GEMM-A8W8_BLOCKSCALE-A16W16",
        M,
        specialized_filename=specialized_filename,
    )
