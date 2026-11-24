# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_gemm_a16w4_repr = make_kernel_repr(
    "_gemm_a16w4_kernel",
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
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GROUP_K",
        "SCALE_PER_TILE",
        "SCALE_AFTER_DOT",
        "HAS_ZEROS",
        "HAS_PERM",
        "ADD_BIAS",
    ],
)


@triton.jit
def _unpack_uint4(w, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """Unpack int32 words [BLOCK_SIZE_N, BLOCK_SIZE_K // 8] to uint4 values [BLOCK_SIZE_N, BLOCK_SIZE_K].

    Nibble i of a word is element i of its 8 K values. The join tree only relabels
    registers, so each thread keeps its K values contiguous.
    """
    x0 = w & 0xF
    x1 = (w >> 4) & 0xF
    x2 = (w >> 8) & 0xF
    x3 = (w >> 12) & 0xF
    x4 = (w >> 16) & 0xF
    x5 = (w >> 20) & 0xF
    x6 = (w >> 24) & 0xF
    x7 = (w >> 28) & 0xF
    x = tl.join(
        tl.join(tl.join(x0, x4), tl.join(x2, x6)),
        tl.join(tl.join(x1, x5), tl.join(x3, x7)),
    )
    return tl.reshape(x, (BLOCK_SIZE_N, BLOCK_SIZE_K))


@triton.heuristics(
    {
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_gemm_a16w4_repr)
def _gemm_a16w4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scales_ptr,
    b_zeros_ptr,
    a_perm_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_bsn,
    stride_bsk,
    stride_bzn,
    stride_bzk,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    SCALE_PER_TILE: tl.constexpr,
    SCALE_AFTER_DOT: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    HAS_PERM: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Computes C = A x dequant(B)^T with group-wise uint4 weights.

    A has shape (M, K). B has shape (N, K // 8) in int32 words, 8 uint4 values of
    consecutive K in each word. The dequantized weight is (q - zero) * scale, with
    one scale and zero for each N and each group of GROUP_K values of K. The zero
    is 8 when HAS_ZEROS is False. With HAS_PERM, column k of B multiplies column
    a_perm[k] of A. With NUM_KSPLIT > 1, C has shape (NUM_KSPLIT, M, N) in fp32.
    """
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_ck >= 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_bsn > 0)
    tl.assume(stride_bsk >= 0)
    tl.assume(stride_bzn >= 0)
    tl.assume(stride_bzk >= 0)

    pid_unified = tl.program_id(axis=0)
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
        k_begin = pid_k * num_k_iter
        k_end = tl.minimum(k_begin + num_k_iter, tl.cdiv(K, BLOCK_SIZE_K))

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_kw = tl.arange(0, BLOCK_SIZE_K // 8)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

        # The weight tile is the first dot operand: acc[n, m] = sum_k W[n, k] * A[m, k].
        a_cols = a_ptr + offs_am[None, :] * stride_am
        b_ptrs = (
            b_ptr
            + offs_bn[:, None] * stride_bn
            + (k_begin * (BLOCK_SIZE_K // 8) + offs_kw)[None, :] * stride_bk
        )
        bs_rows = b_scales_ptr + offs_bn * stride_bsn
        bz_rows = b_zeros_ptr + offs_bn * stride_bzn

        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(k_begin, k_end):
            offs_ks = k * BLOCK_SIZE_K + offs_k
            if HAS_PERM:
                if EVEN_K:
                    offs_ka = tl.load(a_perm_ptr + offs_ks)
                else:
                    offs_ka = tl.load(a_perm_ptr + offs_ks, mask=offs_ks < K, other=0)
            else:
                offs_ka = offs_ks
            if EVEN_K:
                a = tl.load(a_cols + offs_ka[:, None] * stride_ak)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_cols + offs_ka[:, None] * stride_ak,
                    mask=(offs_ks < K)[:, None],
                    other=0.0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=(k * (BLOCK_SIZE_K // 8) + offs_kw < K // 8)[None, :],
                    other=0,
                    cache_modifier=cache_modifier,
                )
            q = _unpack_uint4(b, BLOCK_SIZE_N, BLOCK_SIZE_K)

            if SCALE_PER_TILE:
                # The K tile is in one group: one scale and zero per row.
                g = (k * BLOCK_SIZE_K) // GROUP_K
                scale = tl.load(bs_rows + g * stride_bsk).to(tl.float32)
                if HAS_ZEROS:
                    zero = tl.load(bz_rows + g * stride_bzk).to(tl.int32)[:, None]
                else:
                    zero = 8
                if SCALE_AFTER_DOT:
                    # q - zero is a small integer: the conversion and the dot are exact.
                    w = (
                        (q - zero)
                        .to(tl.float32)
                        .to(a_ptr.dtype.element_ty, fp_downcast_rounding="rtz")
                    )
                    acc += tl.dot(w, a) * scale[:, None]
                else:
                    w = ((q - zero).to(tl.float32) * scale[:, None]).to(
                        a_ptr.dtype.element_ty
                    )
                    acc = tl.dot(w, a, acc)
            else:
                # The K tile has BLOCK_SIZE_K // GROUP_K groups: load their scales
                # and zeros as one block and broadcast them over each group.
                NUM_G: tl.constexpr = BLOCK_SIZE_K // GROUP_K
                offs_g = (k * BLOCK_SIZE_K) // GROUP_K + tl.arange(0, NUM_G)
                g_mask = (offs_g < tl.cdiv(K, GROUP_K))[None, :]
                scale = tl.load(
                    bs_rows[:, None] + offs_g[None, :] * stride_bsk,
                    mask=g_mask,
                    other=0.0,
                ).to(tl.float32)
                if HAS_ZEROS:
                    zero = tl.load(
                        bz_rows[:, None] + offs_g[None, :] * stride_bzk,
                        mask=g_mask,
                        other=0,
                    ).to(tl.int32)[:, :, None]
                else:
                    zero = 8
                q = tl.reshape(q, (BLOCK_SIZE_N, NUM_G, GROUP_K))
                w = ((q - zero).to(tl.float32) * scale[:, :, None]).to(
                    a_ptr.dtype.element_ty
                )
                acc = tl.dot(tl.reshape(w, (BLOCK_SIZE_N, BLOCK_SIZE_K)), a, acc)

            b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        if ADD_BIAS:
            acc += tl.load(bias_ptr + offs_bn).to(tl.float32)[:, None]
        c = acc.to(c_ptr.type.element_ty)
        c_ptrs = (
            c_ptr
            + pid_k * stride_ck
            + offs_cm[None, :] * stride_cm
            + offs_cn[:, None] * stride_cn
        )
        c_mask = (offs_cm[None, :] < M) & (offs_cn[:, None] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(M: int, N: int, K: int):
    return get_gemm_config("GEMM-A16W4", M, N, K)
