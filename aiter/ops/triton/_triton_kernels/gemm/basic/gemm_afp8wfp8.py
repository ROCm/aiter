# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

_gemm_afp8wfp8_repr = make_kernel_repr(
    "_gemm_afp8wfp8_kernel",
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
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_gemm_afp8wfp8_repr)
def _gemm_afp8wfp8_kernel(
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
    A_scales are e8m0 (uint8) with shape (M, K // 32).
    B_scales are stored compact e8m0 (uint8) with shape (N // 128, K // 128),
    representing 128x128 weight blocks. Broadcast inside kernel to (N, K // 32).
    A has shape (M, K), B has shape (K, N) and C has shape (M, N).
    Output dtype is determined by c_ptr (bf16 or fp16).
    NUM_KSPLIT == 1 only (no split-K in this first version).
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN, NUM_XCDS=8)

    pid = pid_unified
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    # Scale group sizes
    SCALE_GROUP_SIZE: tl.constexpr = 32  # A: per 32 elements along K
    B_SCALE_K_GROUP: tl.constexpr = 128  # B: per 128 along K
    B_SCALE_N_GROUP: tl.constexpr = 128  # B: per 128 along N

    # Number of K iterations
    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)

    # Create pointers for first block of A and B input matrices
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # A-scale pointers: per-row (M) and per scale group (K // 32)
    offs_ks_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
    a_scale_ptrs = (
        a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks_a[None, :] * stride_ask
    )

    # B-scale pointers: compact (N // 128, K // 128) — broadcast inside the kernel
    # Each scale covers a 128(N) x 128(K) block.
    # We want a tile of shape (BLOCK_SIZE_N, BLOCK_SIZE_K // 32) per K-iter.
    # Broadcast by floor-division: row index n -> n // 128; k-scale index ks -> ks // 4
    # (since ks counts 32-element groups, and 128 / 32 = 4).
    offs_bsn = offs_bn // B_SCALE_N_GROUP  # (BLOCK_SIZE_N,)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_scale_k_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

    for k in range(0, num_k_iter):
        # K base for this iteration (in elements)
        k_base = k * BLOCK_SIZE_K

        # ---- Load A scales (M, BLOCK_SIZE_K // 32) ----
        if EVEN_K:
            a_scales = tl.load(a_scale_ptrs)
        else:
            a_scale_mask = offs_scale_k_a[None, :] < (
                K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
            a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

        # ---- Load and broadcast B scales (BLOCK_SIZE_N, BLOCK_SIZE_K // 32) ----
        # B-scale K index in compact layout (K // 128 stride)
        k_scale_idx_b = (k_base + tl.arange(0, BLOCK_SIZE_K) // SCALE_GROUP_SIZE) // (
            B_SCALE_K_GROUP // SCALE_GROUP_SIZE
        )
        # k_scale_idx_b: shape (BLOCK_SIZE_K,), but we want one per 32-group.
        # Recompute correctly:
        offs_bsk = (
            k_base + offs_scale_k_a * SCALE_GROUP_SIZE
        ) // B_SCALE_K_GROUP  # (BLOCK_SIZE_K // 32,)
        b_scale_ptrs = (
            b_scales_ptr
            + offs_bsn[:, None] * stride_bsn
            + offs_bsk[None, :] * stride_bsk
        )
        if EVEN_K:
            b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
        else:
            # OOB along K: load with the same mask as a-scales
            b_scale_mask = offs_scale_k_a[None, :] < (
                K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
            b_scales = tl.load(
                b_scale_ptrs,
                mask=b_scale_mask,
                other=127,
                cache_modifier=cache_modifier,
            )

        # ---- Load A, B data ----
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                other=0,
                cache_modifier=cache_modifier,
            )

        accumulator = tl.dot_scaled(
            a, a_scales, "e4m3", b, b_scales, "e4m3", accumulator
        )

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask

    c = accumulator.to(c_ptr.type.element_ty)

    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_gemm_afp8wfp8_preshuffle_repr = make_kernel_repr(
    "_gemm_afp8wfp8_preshuffle_kernel",
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
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0),
    }
)
@triton.jit(repr=_gemm_afp8wfp8_preshuffle_repr)
def _gemm_afp8wfp8_preshuffle_kernel(
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
    EVEN_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
    waves_per_eu: tl.constexpr,
    matrix_instr_nonkdim: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Preshuffle variant of _gemm_afp8wfp8_kernel. Weight tensor has been shuffled
    via aiter.ops.shuffle.shuffle_weight(layout=(16, 16)) so that 16-row N tiles
    are interleaved with their 32-col K chunks in storage. The kernel loads the
    shuffled tile in storage order (BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16) then
    reshape+permute+trans inside the kernel to restore logical (K, N) layout
    before tl.dot_scaled. Scales remain in the unshuffled compact 128x128 layout.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_asm > 0)
    tl.assume(stride_ask > 0)
    tl.assume(stride_bsk > 0)
    tl.assume(stride_bsn > 0)

    GRID_MN = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)

    pid_unified = tl.program_id(axis=0)
    pid_unified = remap_xcd(pid_unified, GRID_MN, NUM_XCDS=8)

    pid = pid_unified
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    SCALE_GROUP_SIZE: tl.constexpr = 32  # A: per-32 along K
    B_SCALE_K_GROUP: tl.constexpr = 128  # B compact: per-128 along K
    B_SCALE_N_GROUP: tl.constexpr = 128  # B compact: per-128 along N

    num_k_iter = tl.cdiv(K, BLOCK_SIZE_K)

    # A pointers (unchanged from non-preshuffle kernel).
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # B pointers for preshuffled layout. The shuffled storage is viewed as
    # (N // 16, K * 16) elements. pid_n indexes BLOCK_SIZE_N // 16 N-tiles per
    # step; the K dimension is expanded by 16x in byte addresses.
    offs_bn_shuffle = pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)
    offs_k_shuffle = tl.arange(0, BLOCK_SIZE_K * 16)
    b_ptrs = b_ptr + (
        offs_bn_shuffle[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
    )

    # A-scale pointers: per-row M, per 32-K group.
    offs_ks_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)
    a_scale_ptrs = (
        a_scales_ptr + offs_am[:, None] * stride_asm + offs_ks_a[None, :] * stride_ask
    )

    # B-scale pointers: compact (N // 128, K // 128). The N index needs the
    # ORIGINAL (logical) row, not the shuffled row index, so use offs_bn_logical.
    offs_bn_logical = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bsn = offs_bn_logical // B_SCALE_N_GROUP

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    offs_scale_k_a = tl.arange(0, BLOCK_SIZE_K // SCALE_GROUP_SIZE)

    for k in range(0, num_k_iter):
        k_base = k * BLOCK_SIZE_K

        # Load A scales.
        if EVEN_K:
            a_scales = tl.load(a_scale_ptrs)
        else:
            a_scale_mask = offs_scale_k_a[None, :] < (
                K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
            a_scales = tl.load(a_scale_ptrs, mask=a_scale_mask, other=127)

        # Load and broadcast B scales (same as non-preshuffle path).
        offs_bsk = (k_base + offs_scale_k_a * SCALE_GROUP_SIZE) // B_SCALE_K_GROUP
        b_scale_ptrs = (
            b_scales_ptr
            + offs_bsn[:, None] * stride_bsn
            + offs_bsk[None, :] * stride_bsk
        )
        if EVEN_K:
            b_scales = tl.load(b_scale_ptrs, cache_modifier=cache_modifier)
        else:
            b_scale_mask = offs_scale_k_a[None, :] < (
                K // SCALE_GROUP_SIZE - k * (BLOCK_SIZE_K // SCALE_GROUP_SIZE)
            )
            b_scales = tl.load(
                b_scale_ptrs,
                mask=b_scale_mask,
                other=127,
                cache_modifier=cache_modifier,
            )

        # Load A and B (preshuffled).
        if EVEN_K:
            a = tl.load(a_ptrs)
            b_shuf = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b_shuf = tl.load(
                b_ptrs,
                mask=offs_k_shuffle[None, :] < (K - k * BLOCK_SIZE_K) * 16,
                other=0,
                cache_modifier=cache_modifier,
            )

        # Unshuffle B in-kernel. Inverse of the shuffle_weight permute:
        # shuffle:   (N // 16, 16, K // 32, 2, 16) --[perm 0,1,3,4,2,5]-> (N // 16, K // 32, 2, 16, 16)
        # unshuffle: (N // 16, K // 32, 2, 16, 16) --[perm 0,1,4,2,3,5]-> (N // 16, 16, K // 32, 2, 16)
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

        # Advance pointers.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * 16 * stride_bk
        a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask

    c = accumulator.to(c_ptr.type.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
):
    """Load the best tuned config for (M, N, K) via the standard aiter JSON
    config mechanism. Falls back to the generic fallback file or to
    _DEFAULT_CONFIG if no JSON is available."""
    if shuffle:
        return get_gemm_config("GEMM-AFP8WFP8-PRESHUFFLE", M, N, K)
    else:
        return get_gemm_config("GEMM-AFP8WFP8", M, N, K)
