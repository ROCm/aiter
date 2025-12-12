# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton.kernel_repr import make_kernel_repr
from ..utils.gemm_config_utils import get_gemm_config


_fused_gemm_afp4wfp4_a16w16_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_a16w16_kernel",
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
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN_FP4": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp4"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_gemm_afp4wfp4_a16w16_repr)
def _fused_gemm_afp4wfp4_a16w16_kernel(
    # Pointers to matrices
    a_fp4_ptr,
    b_fp4_ptr,
    bias_fp4_ptr,
    a_fp4_scale_ptr,
    b_fp4_scale_ptr,
    c_fp4_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp4,
    N_bf16,
    K,
    stride_a_fp4_m,
    stride_a_fp4_k,
    stride_b_fp4_k,
    stride_b_fp4_n,
    stride_a_fp4_scale_m,
    stride_a_fp4_scale_k,
    stride_b_fp4_scale_n,
    stride_b_fp4_scale_k,
    stride_c_fp4_k,
    stride_c_fp4_m,
    stride_c_fp4_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP4: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    GRID_MN_FP4: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp4_m > 0)
    tl.assume(stride_a_fp4_k > 0)
    tl.assume(stride_b_fp4_k > 0)
    tl.assume(stride_b_fp4_n > 0)
    tl.assume(stride_c_fp4_k > 0)
    tl.assume(stride_c_fp4_m > 0)
    tl.assume(stride_c_fp4_n > 0)
    tl.assume(stride_a_fp4_scale_m > 0)
    tl.assume(stride_a_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32
    GRID_MN: tl.constexpr = GRID_MN_FP4 + GRID_MN_BF16

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp4 = tl.cdiv(N_fp4, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp4 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:

        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

        if pid_n < num_pid_n_fp4:
            offs_k_fp4 = tl.arange(0, BLOCK_SIZE_K // 2)
            offs_k_fp4_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k_fp4
            offs_b_fp4_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp4
            a_fp4_ptrs = a_fp4_ptr + (
                offs_am[:, None] * stride_a_fp4_m
                + offs_k_fp4_split[None, :] * stride_a_fp4_k
            )
            b_fp4_ptrs = b_fp4_ptr + (
                offs_k_fp4_split[:, None] * stride_b_fp4_k
                + offs_b_fp4_n[None, :] * stride_b_fp4_n
            )

            offs_k_fp4_scale = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
            ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            a_fp4_scale_ptrs = (
                a_fp4_scale_ptr
                + offs_am[:, None] * stride_a_fp4_scale_m
                + offs_k_fp4_scale[None, :] * stride_a_fp4_scale_k
            )
            # B scales are N x K even though B operand is K x N.
            b_fp4_scale_ptrs = (
                b_fp4_scale_ptr
                + offs_b_fp4_n[:, None] * stride_b_fp4_scale_n
                + offs_k_fp4_scale[None, :] * stride_b_fp4_scale_k
            )

            if ADD_BIAS_FP4:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_fp4 = tl.load(bias_fp4_ptr + offs_b_fp4_n).to(
                        dtype=tl.float32
                    )
                    accumulator_fp4 = tl.broadcast_to(
                        accumulator_fp4[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_fp4 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_fp4 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                a_scale = tl.load(a_fp4_scale_ptrs)
                b_scale = tl.load(b_fp4_scale_ptrs, cache_modifier=cache_modifier)

                if EVEN_K:
                    a = tl.load(a_fp4_ptrs)
                    b = tl.load(b_fp4_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_fp4_ptrs,
                        mask=offs_k_fp4[None, :] < K - k * (BLOCK_SIZE_K // 2),
                        other=0.0,
                    )
                    b = tl.load(
                        b_fp4_ptrs,
                        mask=offs_k_fp4[:, None] < K - k * (BLOCK_SIZE_K // 2),
                        other=0.0,
                        cache_modifier=cache_modifier,
                    )

                accumulator_fp4 += tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")

                a_fp4_ptrs += (BLOCK_SIZE_K // 2) * stride_a_fp4_k
                b_fp4_ptrs += (BLOCK_SIZE_K // 2) * stride_b_fp4_k
                a_fp4_scale_ptrs += (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                ) * stride_a_fp4_scale_k
                b_fp4_scale_ptrs += (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                ) * stride_b_fp4_scale_k

            c_fp4 = accumulator_fp4.to(c_fp4_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp4_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp4_ptrs = (
                c_fp4_ptr
                + stride_c_fp4_m * offs_cm[:, None]
                + stride_c_fp4_n * offs_c_fp4_n[None, :]
                + pid_k * stride_c_fp4_k
            )
            c_fp4_mask = (offs_cm[:, None] < M) & (offs_c_fp4_n[None, :] < N_fp4)
            tl.store(c_fp4_ptrs, c_fp4, mask=c_fp4_mask)
        else:
            pid_n -= num_pid_n_fp4
            offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
            offs_k_bf16_split = pid_k * SPLITK_BLOCK_SIZE + offs_k_bf16
            K = 2 * K

            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_bf16_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_bf16_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            if ADD_BIAS_BF16:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_bf16 = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                        dtype=tl.float32
                    )
                    accumulator_bf16 = tl.broadcast_to(
                        accumulator_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_bf16 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_bf16 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k_bf16[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k_bf16[:, None] < K - k * BLOCK_SIZE_K,
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


_fused_gemm_afp4wfp4_preshuffle_a16w16_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_preshuffle_a16w16_kernel",
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
        "EVEN_K": lambda args: (args["K"] % (args["BLOCK_SIZE_K"] // 2) == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % (args["SPLITK_BLOCK_SIZE"] // 2) == 0),
        "GRID_MN_FP4": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_fp4"], args["BLOCK_SIZE_N"]),
        "GRID_MN_BF16": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N_bf16"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_fused_gemm_afp4wfp4_preshuffle_a16w16_repr)
def _fused_gemm_afp4wfp4_preshuffle_a16w16_kernel(
    # Pointers to matrices
    a_fp4_ptr,
    b_fp4_ptr,
    bias_fp4_ptr,
    a_fp4_scale_ptr,
    b_fp4_scale_ptr,
    c_fp4_ptr,
    a_bf16_ptr,
    b_bf16_ptr,
    bias_bf16_ptr,
    c_bf16_ptr,
    # Matrix dimensions
    M,
    N_fp4,
    N_bf16,
    K,
    stride_a_fp4_m,
    stride_a_fp4_k,
    stride_b_fp4_n,
    stride_b_fp4_k,
    stride_a_fp4_scale_m,
    stride_a_fp4_scale_k,
    stride_b_fp4_scale_n,
    stride_b_fp4_scale_k,
    stride_c_fp4_k,
    stride_c_fp4_m,
    stride_c_fp4_n,
    stride_a_bf16_m,
    stride_a_bf16_k,
    stride_b_bf16_k,
    stride_b_bf16_n,
    stride_c_bf16_k,
    stride_c_bf16_m,
    stride_c_bf16_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    ADD_BIAS_FP4: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    GRID_MN_FP4: tl.constexpr,
    GRID_MN_BF16: tl.constexpr,
    SKIP_REDUCE: tl.constexpr,
    cache_modifier: tl.constexpr,
):

    tl.assume(stride_a_fp4_m > 0)
    tl.assume(stride_a_fp4_k > 0)
    tl.assume(stride_b_fp4_k > 0)
    tl.assume(stride_b_fp4_n > 0)
    tl.assume(stride_c_fp4_k > 0)
    tl.assume(stride_c_fp4_m > 0)
    tl.assume(stride_c_fp4_n > 0)
    tl.assume(stride_a_fp4_scale_m > 0)
    tl.assume(stride_a_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_k > 0)
    tl.assume(stride_b_fp4_scale_n > 0)

    tl.assume(stride_a_bf16_m > 0)
    tl.assume(stride_a_bf16_k > 0)
    tl.assume(stride_b_bf16_k > 0)
    tl.assume(stride_b_bf16_n > 0)
    tl.assume(stride_c_bf16_m > 0)
    tl.assume(stride_c_bf16_n > 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32
    GRID_MN: tl.constexpr = GRID_MN_FP4 + GRID_MN_BF16

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN * NUM_KSPLIT, NUM_XCDS=8)

    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n_fp4 = tl.cdiv(N_fp4, BLOCK_SIZE_N)
    num_pid_n_bf16 = tl.cdiv(N_bf16, BLOCK_SIZE_N)
    num_pid_n = num_pid_n_fp4 + num_pid_n_bf16

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE // 2) < K:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M

        if pid_n < num_pid_n_fp4:
            num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE // 2, BLOCK_SIZE_K // 2)

            offs_k_fp4 = tl.arange(0, BLOCK_SIZE_K // 2)
            offs_k_fp4_shuffle_arr = tl.arange(0, (BLOCK_SIZE_K // 2) * 16)
            offs_k_fp4_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k_fp4
            offs_k_fp4_shuffle = (
                pid_k * (SPLITK_BLOCK_SIZE // 2) * 16 + offs_k_fp4_shuffle_arr
            )

            offs_b_fp4_n = (
                pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)
            ) % N_fp4
            a_fp4_ptrs = a_fp4_ptr + (
                offs_am[:, None] * stride_a_fp4_m
                + offs_k_fp4_split[None, :] * stride_a_fp4_k
            )
            b_fp4_ptrs = b_fp4_ptr + (
                offs_b_fp4_n[:, None] * stride_b_fp4_n
                + offs_k_fp4_shuffle[None, :] * stride_b_fp4_k
            )

            offs_b_fp4_scale_n = (
                pid_n * (BLOCK_SIZE_N // 32) + tl.arange(0, (BLOCK_SIZE_N // 32))
            ) % N_fp4
            offs_k_fp4_scale = (
                pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE) * 32
            ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE * 32)
            b_fp4_scale_ptrs = (
                b_fp4_scale_ptr
                + offs_b_fp4_scale_n[:, None] * stride_b_fp4_scale_n
                + offs_k_fp4_scale[None, :] * stride_b_fp4_scale_k
            )

            if BLOCK_SIZE_M < 32:
                offs_ks_non_shufl = (
                    pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)
                ) + tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                a_fp4_scale_ptrs = (
                    a_fp4_scale_ptr
                    + offs_am[:, None] * stride_a_fp4_scale_m
                    + offs_ks_non_shufl[None, :] * stride_a_fp4_scale_k
                )
            else:
                offs_a_fp4_scale_m = (
                    pid_m * (BLOCK_SIZE_M // 32) + tl.arange(0, (BLOCK_SIZE_M // 32))
                ) % M
                a_fp4_scale_ptrs = (
                    a_fp4_scale_ptr
                    + offs_a_fp4_scale_m[:, None] * stride_a_fp4_scale_m
                    + offs_k_fp4_scale[None, :] * stride_a_fp4_scale_k
                )

            if ADD_BIAS_FP4:
                offs_b_fp4_n_bias = (
                    pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                ) % N_fp4
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_fp4 = tl.load(bias_fp4_ptr + offs_b_fp4_n_bias).to(
                        dtype=tl.float32
                    )
                    accumulator_fp4 = tl.broadcast_to(
                        accumulator_fp4[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_fp4 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_fp4 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if BLOCK_SIZE_M < 32:
                    a_scale = tl.load(a_fp4_scale_ptrs)
                else:
                    a_scale = (
                        tl.load(a_fp4_scale_ptrs)
                        .reshape(
                            BLOCK_SIZE_M // 32,
                            BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                            4,
                            16,
                            2,
                            2,
                            1,
                        )
                        .permute(0, 5, 3, 1, 4, 2, 6)
                        .reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                    )

                b_scale = (
                    tl.load(b_fp4_scale_ptrs, cache_modifier=cache_modifier)
                    .reshape(
                        BLOCK_SIZE_N // 32,
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE // 8,
                        4,
                        16,
                        2,
                        2,
                        1,
                    )
                    .permute(0, 5, 3, 1, 4, 2, 6)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
                )

                if EVEN_K:
                    a = tl.load(a_fp4_ptrs)
                    b = tl.load(b_fp4_ptrs, cache_modifier=cache_modifier)
                # else:
                #     a = tl.load(
                #         a_fp4_ptrs,
                #         mask=offs_k[None, :] < K - k * (BLOCK_SIZE_K // 2),
                #         other=0.0
                #     )
                #     b = tl.load(
                #         b_fp4_ptrs,
                #         mask=offs_k[:, None] < K - k * (BLOCK_SIZE_K // 2),
                #         other=0.0,
                #         cache_modifier=cache_modifier,
                #     )

                b = (
                    b.reshape(
                        1,
                        BLOCK_SIZE_N // 16,
                        BLOCK_SIZE_K // 64,
                        2,
                        16,
                        16,
                    )
                    .permute(0, 1, 4, 2, 3, 5)
                    .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K // 2)
                    .trans(1, 0)
                )

                accumulator_fp4 += tl.dot_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1")

                a_fp4_ptrs += (BLOCK_SIZE_K // 2) * stride_a_fp4_k
                b_fp4_ptrs += (BLOCK_SIZE_K // 2) * 16 * stride_b_fp4_k
                if BLOCK_SIZE_M < 32:
                    a_fp4_scale_ptrs += (
                        BLOCK_SIZE_K // SCALE_GROUP_SIZE
                    ) * stride_a_fp4_scale_k
                else:
                    a_fp4_scale_ptrs += BLOCK_SIZE_K * stride_a_fp4_scale_k
                b_fp4_scale_ptrs += BLOCK_SIZE_K * stride_b_fp4_scale_k

            c_fp4 = accumulator_fp4.to(c_fp4_ptr.type.element_ty)

            offs_cm = pid_m.to(tl.int64) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
                tl.int64
            )
            offs_c_fp4_n = pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(
                0, BLOCK_SIZE_N
            ).to(tl.int64)
            c_fp4_ptrs = (
                c_fp4_ptr
                + stride_c_fp4_m * offs_cm[:, None]
                + stride_c_fp4_n * offs_c_fp4_n[None, :]
                + pid_k * stride_c_fp4_k
            )
            c_fp4_mask = (offs_cm[:, None] < M) & (offs_c_fp4_n[None, :] < N_fp4)
            tl.store(c_fp4_ptrs, c_fp4, mask=c_fp4_mask, cache_modifier=".wt")
        else:
            pid_n -= num_pid_n_fp4
            K = 2 * K

            num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

            offs_k_bf16 = tl.arange(0, BLOCK_SIZE_K)
            offs_k_bf16_split = pid_k * (SPLITK_BLOCK_SIZE) + offs_k_bf16
            offs_b_bf16_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_bf16
            a_ptrs = a_bf16_ptr + (
                offs_am[:, None] * stride_a_bf16_m
                + offs_k_bf16_split[None, :] * stride_a_bf16_k
            )
            b_ptrs = b_bf16_ptr + (
                offs_k_bf16_split[:, None] * stride_b_bf16_k
                + offs_b_bf16_n[None, :] * stride_b_bf16_n
            )

            if ADD_BIAS_BF16:
                if NUM_KSPLIT == 1 or (SKIP_REDUCE and pid_k == 0):
                    accumulator_bf16 = tl.load(bias_bf16_ptr + offs_b_bf16_n).to(
                        dtype=tl.float32
                    )
                    accumulator_bf16 = tl.broadcast_to(
                        accumulator_bf16[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N)
                    )
                else:
                    accumulator_bf16 = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
            else:
                accumulator_bf16 = tl.zeros(
                    (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                )

            for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
                if EVEN_K:
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=offs_k_bf16[None, :] < K - k * BLOCK_SIZE_K,
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=offs_k_bf16[:, None] < K - k * BLOCK_SIZE_K,
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


_gemm_afp4wfp4_a16w16_reduce_repr = make_kernel_repr(
    "_fused_gemm_afp4wfp4_a16w16_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
        "ADD_BIAS_FP4",
        "ADD_BIAS_BF16",
    ],
)


@triton.heuristics({})  # dummy heuristics to invoke kernel re-naming
@triton.jit(repr=_gemm_afp4wfp4_a16w16_reduce_repr)
def _fused_gemm_afp4wfp4_a16w16_reduce_kernel(
    bias_fp4_ptr,
    c_fp4_in_ptr,
    c_fp4_out_ptr,
    bias_bf16_ptr,
    c_bf16_in_ptr,
    c_bf16_out_ptr,
    M,
    N_fp4,
    N_bf16,
    stride_c_fp4_in_k,
    stride_c_fp4_in_m,
    stride_c_fp4_in_n,
    stride_c_fp4_out_m,
    stride_c_fp4_out_n,
    stride_c_bf16_in_k,
    stride_c_bf16_in_m,
    stride_c_bf16_in_n,
    stride_c_bf16_out_m,
    stride_c_bf16_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
    ADD_BIAS_FP4: tl.constexpr,
    ADD_BIAS_BF16: tl.constexpr,
):

    tl.assume(stride_c_fp4_in_k > 0)
    tl.assume(stride_c_fp4_in_m > 0)
    tl.assume(stride_c_fp4_in_n > 0)
    tl.assume(stride_c_fp4_out_m > 0)
    tl.assume(stride_c_fp4_out_n > 0)

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

    num_pid_n_fp4 = tl.cdiv(N_fp4, BLOCK_SIZE_N)
    offs_k = tl.arange(0, MAX_KSPLIT)
    acc_dtype = tl.float32 if c_fp4_in_ptr.type.element_ty != tl.int8 else tl.int32

    if pid_n < num_pid_n_fp4:
        offs_fp4_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N_fp4
        c_fp4_in_ptrs = (
            c_fp4_in_ptr
            + (offs_k[:, None, None] * stride_c_fp4_in_k)
            + (offs_m[None, :, None] * stride_c_fp4_in_m)
            + (offs_fp4_n[None, None, :] * stride_c_fp4_in_n)
        )

        if ACTUAL_KSPLIT == MAX_KSPLIT:
            c_fp4 = tl.load(c_fp4_in_ptrs)
        else:
            c_fp4 = tl.load(
                c_fp4_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT, other=0.0
            )
        c_fp4 = tl.sum(c_fp4, axis=0)
        if ADD_BIAS_FP4:
            bias_fp4 = tl.load(bias_fp4_ptr + offs_fp4_n).to(dtype=acc_dtype)
            bias_fp4 = tl.broadcast_to(bias_fp4[None, :], (BLOCK_SIZE_M, BLOCK_SIZE_N))
            c_fp4 += bias_fp4

        c_fp4 = c_fp4.to(c_fp4_out_ptr.type.element_ty)

        c_fp4_out_ptrs = (
            c_fp4_out_ptr
            + (offs_m[:, None] * stride_c_fp4_out_m)
            + (offs_fp4_n[None, :] * stride_c_fp4_out_n)
        )

        tl.store(c_fp4_out_ptrs, c_fp4)
    else:
        pid_n -= num_pid_n_fp4

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
    N_fp4: int,
    N_bf16: int,
    K: int,
    shuffle: bool = False,
):
    config_name = (
        "FUSED-GEMM-AFP4WFP4-A16W16"
        if not shuffle
        else "FUSED-GEMM-AFP4WFP4_PRESHUFFLED-A16W16"
    )
    # Custom file naming: N4={N_fp4}-N16={N_bf16}-K={2*K}
    # Note: N and K are not passed to get_gemm_config here, as they are encoded in the specialized_filename.
    # This differs from most other usages, where N and K are required as explicit arguments.
    specialized_filename = f"N4={N_fp4}-N16={N_bf16}-K={2*K}"
    return get_gemm_config(config_name, M, specialized_filename=specialized_filename)
