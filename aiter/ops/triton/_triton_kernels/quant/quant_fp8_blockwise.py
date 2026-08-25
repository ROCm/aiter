# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    x_tile_max = tl.maximum(x_tile_max, 1e-4)
    x_scales_tile = FP8_MAX / x_tile_max
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


# Blockwise quantize
@triton.jit
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(
        x_tile, x_tile_abs, AXIS, FP8_MAX
    )

    # Store
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    # Store scale
    if AXIS == 1:
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:
        scale_offs = pid_m * N + offs_n
        scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


@triton.jit
def compute_m_range(
    pid, batch_size, seg_indptr, scales_seg_indptr_ptr, BLOCK_SIZE: tl.constexpr
):
    bid = 0
    for bs in range(batch_size):
        tiles = tl.load(scales_seg_indptr_ptr + bs)
        if pid >= tiles:
            bid = bs
    idx_start = tl.load(scales_seg_indptr_ptr + bid)

    m_range_start = tl.load(seg_indptr + bid) + (pid - idx_start) * BLOCK_SIZE
    m_range_end = min(tl.load(seg_indptr + bid + 1), m_range_start + BLOCK_SIZE)
    return m_range_start, m_range_end, bid


# Blockwise for Segment M
@triton.jit
def quant_fp8_blockwise_segment_m_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    N,
    batch_size,
    seg_indptr,
    scales_seg_indptr_ptr,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    total_m_block = tl.load(scales_seg_indptr_ptr + batch_size)
    if pid_m >= total_m_block:
        return

    m_range_start, m_range_end, _bid = compute_m_range(
        pid_m, batch_size, seg_indptr, scales_seg_indptr_ptr, BLOCK_SIZE
    )
    if m_range_end - m_range_start == 0:
        return

    offs_m = m_range_start + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < m_range_end) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)

    # Store
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    scale_offs = pid_m * N + offs_n
    scale_mask = offs_n < N
    x_scales_tile_inv = tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE)
    tl.store(
        x_scales_ptr + scale_offs,
        x_scales_tile_inv,
        mask=scale_mask,
    )


# w_ptr         [B, M, N]
# w_fp8_ptr     [B, M, N] FP8
# w_scales_ptr  [B, M // BLOCK_SIZE, N // BLOCK_SIZE] FP32
@triton.jit
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    w_tile_max = tl.max(w_tile_abs)  # [1]
    w_tile_max = tl.maximum(w_tile_max, 1e-4)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = w_tile * w_scales
    w_fp8_tile = tl.clamp(w_fp8_tile, min=-FP8_MAX, max=FP8_MAX)

    # Store
    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    # Store scale
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


# x_ptr             [M, N]
# x_fp8_row_ptr     [M, N] FP8
# x_fp8_col_ptr     [M, N] FP8
# x_scales_row_ptr  [M, N // BLOCK_SIZE] FP32
# x_scales_col_ptr  [M // BLOCK_SIZE, N] FP32
@triton.jit
def quant_fp8_blockwise_for_act_grad_kernel(
    x_ptr,
    x_fp8_row_ptr,
    x_scales_row_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    # Row-wise quantization
    x_fp8_tile_row, x_scales_tile_row = compute_scale_and_quant(
        x_tile, x_tile_abs, 1, FP8_MAX
    )

    # Col-wise quantization
    x_fp8_tile_col, x_scales_tile_col = compute_scale_and_quant(
        x_tile, x_tile_abs, 0, FP8_MAX
    )

    # Store
    x_fp8_row_ptrs = x_fp8_row_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_fp8_col_ptrs = x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(
        x_fp8_row_ptrs, x_fp8_tile_row.to(x_fp8_row_ptr.dtype.element_ty), mask=mask
    )
    tl.store(
        x_fp8_col_ptrs, x_fp8_tile_col.to(x_fp8_col_ptr.dtype.element_ty), mask=mask
    )

    # Store row-wise scales inverse: [M, N // BLOCK_SIZE]
    row_scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    x_scales_tile_row_inv = tl.reshape(1.0 / x_scales_tile_row, BLOCK_SIZE)
    tl.store(
        x_scales_row_ptr + row_scale_offs,
        x_scales_tile_row_inv,
        mask=offs_m < M,
    )

    # Store col-wise scales inverse: [M // BLOCK_SIZE, N]
    col_scale_offs = pid_m * N + offs_n
    x_scales_tile_col_inv = tl.reshape(1.0 / x_scales_tile_col, BLOCK_SIZE)
    tl.store(
        x_scales_col_ptr + col_scale_offs,
        x_scales_tile_col_inv,
        mask=offs_n < N,
    )


# Re-quantize FP8 (row-wise 1×BLOCK) → FP8 (col-wise BLOCK×1) in one pass.
# Avoids a BF16 roundtrip: dequant with saved row scales, then re-quant along the
# column axis.  Used in the blockwise2d WGrad backward (Jet-RL §4.2) where the
# forward activation was stored as FP8 (1×128) and WGrad needs it col-wise (128×1).
#
# x_fp8_ptr    [M, K]              FP8 input, row-wise 1×BLOCK quantized
# x_scales_ptr [M, K//BLOCK_SIZE]  float32 dequant row scales
# y_fp8_ptr    [M, K]              FP8 output, col-wise BLOCK×1 quantized
# y_scales_ptr [M//BLOCK_SIZE, K]  float32 dequant col scales
@triton.jit
def requant_fp8_row_to_col_kernel(
    x_fp8_ptr,
    x_scales_ptr,
    y_fp8_ptr,
    y_scales_ptr,
    M,
    K,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_k = tl.cast(pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Load FP8 tile and dequant to float32 using saved row scales.
    # Row scale index: each row has K//BLOCK_SIZE scales; tile (pid_m, pid_k) reads
    # one scale per row — the scale for the pid_k-th block along K.
    x_fp8_tile = tl.load(
        x_fp8_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask, other=0.0
    )
    x_f32 = x_fp8_tile.to(tl.float32)

    row_scale_offs = offs_m * tl.cdiv(K, BLOCK_SIZE) + pid_k
    row_scales = tl.load(x_scales_ptr + row_scale_offs, mask=offs_m < M, other=1.0)
    x_f32 = x_f32 * row_scales[:, None]  # broadcast: (BLOCK,1) * (BLOCK, BLOCK)

    # Col-wise (axis=0) requant: one scale per column in this tile.
    x_abs = tl.abs(x_f32)
    col_amax = tl.max(x_abs, axis=0, keep_dims=True)  # (1, BLOCK)
    col_amax = tl.maximum(col_amax, 1e-4)
    col_scale = FP8_MAX / col_amax
    y_f32 = tl.clamp(x_f32 * col_scale, min=-FP8_MAX, max=FP8_MAX)

    tl.store(
        y_fp8_ptr + offs_m[:, None] * K + offs_k[None, :],
        y_f32.to(y_fp8_ptr.dtype.element_ty),
        mask=mask,
    )

    # Col dequant scales: shape (M//BLOCK_SIZE, K), one float per column element.
    col_scale_inv = tl.reshape(1.0 / col_scale, BLOCK_SIZE)  # (BLOCK,) dequant per col
    tl.store(y_scales_ptr + pid_m * K + offs_k, col_scale_inv, mask=offs_k < K)
