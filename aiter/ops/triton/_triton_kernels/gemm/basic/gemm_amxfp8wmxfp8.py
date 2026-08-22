# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)

_gemm_amxfp8wmxfp8_repr = make_kernel_repr(
    "_gemm_amxfp8wmxfp8_kernel",
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
    ],
)


@triton.heuristics(
    {
        # Fast (unmasked) path only when the K-loop tiles K exactly AND every
        # split is a whole number of BLOCK_SIZE_K tiles that never overshoots K.
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
    }
)
@triton.jit(repr=_gemm_amxfp8wmxfp8_repr)
def _gemm_amxfp8wmxfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
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
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Kernel for computing the matmul C = A x B.
    A and B inputs are FP8 e4m3 (1 byte per element).
    A_scales are e8m0 (uint8) with shape (M, K // 32) — one scale per 1x32 block.
    B_scales are e8m0 (uint8) with shape (N, K // 32) — one scale per 1x32 block.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    Output dtype is determined by c_ptr (bf16 or fp16).

    Both operands are microscaled the same way (1x32 e8m0), so B-scales are
    loaded exactly like A-scales but indexed along N instead of M — no
    broadcast is needed (unlike the 128x128 weight-scale gemm_afp8wfp8 kernel).

    When NUM_KSPLIT > 1, K is split into NUM_KSPLIT partitions of
    SPLITK_BLOCK_SIZE elements; partition pid_k writes its partial result to the
    slab c_ptr + pid_k * stride_ck, and a downstream reduce kernel sums them.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsn > 0)
    tl.assume(stride_bsk > 0)

    # A/B: one e8m0 scale per 32 elements along K (1x32 microscaling).
    SCALE_GROUP_SIZE: tl.constexpr = 32

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    # Remap the unified (K-split x tile) pid so each XCD gets contiguous chunks.
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

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Clamp this split's K range to K so the last split never runs past the
        # end (SPLITK_BLOCK_SIZE and K are both multiples of 32, so k_span is too).
        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        # Data pointers, offset to this split's K start. offs_am/offs_bn wrap OOB
        # rows/cols to valid ones (the store below masks them out).
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # Scale pointers, offset to this split's K start in scale groups.
        # A-scales are (M, K // 32); B-scales are (N, K // 32).
        offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
        offs_ks_split = (split_k_start // SCALE_GROUP_SIZE) + offs_ks
        a_scale_ptrs = (
            a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks_split[None, :] * stride_ask
        )
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks_split[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
                a_scales = tl.load(a_scale_ptrs)
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                k_remaining = k_span - k * BLOCK_SIZE_K
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0)
                b = tl.load(
                    b_ptrs,
                    mask=offs_k[:, None] < k_remaining,
                    other=0,
                    cache_modifier=cache_modifier,
                )
                # k_span is a multiple of 32, so partial tiles end on a 32-group
                # boundary and whole scale groups are masked (127 == 2^0 == 1.0).
                scale_k_remaining = (k_span // SCALE_GROUP_SIZE) - k * (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                )
                scale_mask = offs_ks[None, :] < scale_k_remaining
                a_scales = tl.load(a_scale_ptrs, mask=scale_mask, other=127)
                b_scales = tl.load(
                    b_scale_ptrs, mask=scale_mask, other=127, cache_modifier=cache_modifier
                )

            accumulator = tl.dot_scaled(
                a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
            )

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back this block of C. When NUM_KSPLIT > 1 each pid_k writes to a
        # separate slab (offset by pid_k * stride_ck) for the reduce stage.
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


_gemm_amxfp8wmxfp8_preshuffle_repr = make_kernel_repr(
    "_gemm_amxfp8wmxfp8_preshuffle_kernel",
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
    ],
)


@triton.heuristics(
    {
        # Fast (unmasked) path only when the K-loop tiles K exactly AND every
        # split is a whole number of BLOCK_SIZE_K tiles that never overshoots K.
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
    }
)
@triton.jit(repr=_gemm_amxfp8wmxfp8_preshuffle_repr)
def _gemm_amxfp8wmxfp8_preshuffle_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
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
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Preshuffle variant of _gemm_amxfp8wmxfp8_kernel — same MXFP8 x MXFP8 matmul
    (both operands FP8 e4m3 with 1x32 e8m0 block scales), but the weight tensor
    has been permuted by shuffle_weight(..., layout=(16, 16)): 16-row N tiles are
    interleaved with their 32-col K chunks in storage. The kernel reads the
    shuffled tile in storage order (BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16), then
    reshape+permute+trans in-kernel to restore the logical (K, N) tile before
    tl.dot_scaled.

    Only the weight *data* is shuffled — the 1x32 scales are NOT. B-scales are
    loaded exactly like the non-preshuffle kernel (indexed by logical N and
    K-group), since the in-kernel unshuffle reconstructs the logical weight tile
    and the scaled matmul then sees identical operands.

    When NUM_KSPLIT > 1, K is split into NUM_KSPLIT partitions of
    SPLITK_BLOCK_SIZE elements; partition pid_k writes its partial result to the
    slab c_ptr + pid_k * stride_ck, and a downstream reduce kernel sums them.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsn > 0)
    tl.assume(stride_bsk > 0)

    # A/B: one e8m0 scale per 32 elements along K (1x32 microscaling).
    SCALE_GROUP_SIZE: tl.constexpr = 32

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    # Remap the unified (K-split x tile) pid so each XCD gets contiguous chunks.
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

    split_k_start = pid_k * SPLITK_BLOCK_SIZE
    if split_k_start < K:
        # Clamp this split's K range to K so the last split never runs past the
        # end (SPLITK_BLOCK_SIZE and K are both multiples of 32, so k_span is too).
        split_k_end = tl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
        k_span = split_k_end - split_k_start
        num_k_iter = tl.cdiv(k_span, BLOCK_SIZE_K)

        # A pointers, offset to this split's K start. offs_am wraps OOB rows to
        # valid ones (the store below masks them out).
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = split_k_start + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak)

        # Preshuffled B pointers. Storage is viewed as (N // 16, K * 16): pid_n
        # indexes BLOCK_SIZE_N // 16 N-tiles, and the K axis is expanded 16x in
        # bytes. Offset the K-byte axis by this split's start (split_k_start * 16)
        # and wrap OOB N-tiles to valid ones (masked out in the store).
        offs_bn_shuffle = (
            pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)
        ) % (N // 16)
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_shuffle = split_k_start * 16 + offs_k_shuffle_arr
        b_ptrs = b_ptr + (
            offs_bn_shuffle[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )

        # Scale pointers, offset to this split's K start in scale groups. Both
        # A-scales (M, K // 32) and B-scales (N, K // 32) are unshuffled; B-scales
        # are indexed by the logical N row (matching the in-kernel unshuffle).
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_ks = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
        offs_ks_split = (split_k_start // SCALE_GROUP_SIZE) + offs_ks
        a_scale_ptrs = (
            a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks_split[None, :] * stride_ask
        )
        b_scale_ptrs = (
            b_scales_ptr + offs_bn[:, None] * stride_bsn + offs_ks_split[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_k_iter):
            if EVEN_K:
                a = tl.load(a_ptrs)
                b_shuf = tl.load(b_ptrs, cache_modifier=cache_modifier)
                a_scales = tl.load(a_scale_ptrs)
                b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
            else:
                k_remaining = k_span - k * BLOCK_SIZE_K
                a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0)
                # The shuffled K-byte axis maps 32 logical-K elements to 512
                # contiguous bytes, so masking bytes < k_remaining * 16 zeroes
                # exactly the b entries with logical k >= k_remaining (k_remaining
                # is a multiple of 32 because k_span and BLOCK_SIZE_K are).
                b_shuf = tl.load(
                    b_ptrs,
                    mask=offs_k_shuffle_arr[None, :] < k_remaining * 16,
                    other=0,
                    cache_modifier=cache_modifier,
                )
                # k_span is a multiple of 32, so partial tiles end on a 32-group
                # boundary and whole scale groups are masked (127 == 2^0 == 1.0).
                scale_k_remaining = (k_span // SCALE_GROUP_SIZE) - k * (
                    BLOCK_SIZE_K // SCALE_GROUP_SIZE
                )
                scale_mask = offs_ks[None, :] < scale_k_remaining
                a_scales = tl.load(a_scale_ptrs, mask=scale_mask, other=127)
                b_scales = tl.load(
                    b_scale_ptrs, mask=scale_mask, other=127, cache_modifier=cache_modifier
                )

            # Unshuffle B in-kernel to logical (K, N). Inverse of shuffle_weight:
            # shuffle:   (N//16, 16, K//32, 2, 16) --[perm 0,1,3,4,2,5]-> (N//16, K//32, 2, 16, 16)
            # unshuffle: (N//16, K//32, 2, 16, 16) --[perm 0,1,4,2,3,5]-> (N//16, 16, K//32, 2, 16)
            # then flatten to (N, K) and trans to (K, N).
            b = (
                b_shuf.reshape(
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

            accumulator = tl.dot_scaled(
                a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
            )

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * 16 * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back this block of C. When NUM_KSPLIT > 1 each pid_k writes to a
        # separate slab (offset by pid_k * stride_ck) for the reduce stage.
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


def _get_config(M: int, N: int, K: int, shuffle: bool = False):
    name = "GEMM-AMXFP8WMXFP8_PRESHUFFLED" if shuffle else "GEMM-AMXFP8WMXFP8"
    config, is_tuned = get_gemm_config(name, M, N, K)
    return compute_splitk_params(config, K), is_tuned
