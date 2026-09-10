# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.common.segment_tile import (
    _find_segment_tile_range,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# repr keys are the int constexprs that select the compiled variant; FP8_MAX
# (a float) is omitted so a fractional value can't put a "." in the trace name.
_quant_fp8_blockwise_kernel_repr = make_kernel_repr(
    "quant_fp8_blockwise_kernel", ["BLOCK_SIZE", "AXIS", "DUAL"]
)
_quant_fp8_blockwise_segment_m_kernel_repr = make_kernel_repr(
    "quant_fp8_blockwise_segment_m_kernel", ["BLOCK_SIZE"]
)
_quant_fp8_blockwise_for_weight_kernel_repr = make_kernel_repr(
    "quant_fp8_blockwise_for_weight_kernel", ["BLOCK_SIZE"]
)
_requant_fp8_row_to_col_kernel_repr = make_kernel_repr(
    "requant_fp8_row_to_col_kernel", ["BLOCK_SIZE"]
)


@triton.jit
def _compute_scale_and_quant(x_tile, x_tile_abs, axis, FP8_MAX):
    x_tile_max = tl.max(x_tile_abs, axis=axis, keep_dims=True)
    # Clamp Inf before dividing: FP8_MAX / Inf = 0 and x * 0 = NaN for any Inf
    # element in the tile. Mapping Inf -> FP8_MAX saturates those elements via
    # the clamp below instead of producing NaN.
    x_tile_max = tl.minimum(x_tile_max, FP8_MAX)
    # Tiny floor to prevent division by zero on all-zero blocks. 1e-30 (not
    # 1e-4) so small-but-nonzero blocks still use the full FP8 dynamic range.
    x_tile_max = tl.maximum(x_tile_max, 1e-30)
    x_scales_tile = FP8_MAX / x_tile_max
    x_fp8_tile = x_tile * x_scales_tile
    x_fp8_tile = tl.clamp(x_fp8_tile, min=-FP8_MAX, max=FP8_MAX)
    return x_fp8_tile, x_scales_tile


# Blockwise quantize. AXIS selects the scale axis (1 = row-wise 1xBLOCK,
# 0 = col-wise BLOCKx1). When DUAL, also emit the col-wise (axis=0) copy from
# the same loaded tile — the activation-gradient path needs both directions.
@triton.jit(repr=_quant_fp8_blockwise_kernel_repr)
def quant_fp8_blockwise_kernel(
    x_ptr,
    x_fp8_ptr,
    x_scales_ptr,
    x_fp8_col_ptr,
    x_scales_col_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    AXIS: tl.constexpr,
    DUAL: tl.constexpr,
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

    # Primary (AXIS) quantization.
    x_fp8_tile, x_scales_tile = _compute_scale_and_quant(
        x_tile, x_tile_abs, AXIS, FP8_MAX
    )
    tl.store(
        x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :],
        x_fp8_tile.to(x_fp8_ptr.dtype.element_ty),
        mask=mask,
    )
    if AXIS == 1:  # row-wise scale: [M, N // BLOCK_SIZE]
        scale_offs = offs_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
        scale_mask = offs_m < M
    else:  # col-wise scale: [M // BLOCK_SIZE, N]
        scale_offs = tl.cast(pid_m, tl.int64) * N + offs_n
        scale_mask = offs_n < N
    tl.store(
        x_scales_ptr + scale_offs,
        tl.reshape(1.0 / x_scales_tile, BLOCK_SIZE),
        mask=scale_mask,
    )

    # Secondary col-wise (axis=0) copy for the dual activation-gradient path.
    if DUAL:
        x_fp8_col, x_scales_col = _compute_scale_and_quant(
            x_tile, x_tile_abs, 0, FP8_MAX
        )
        tl.store(
            x_fp8_col_ptr + offs_m[:, None] * N + offs_n[None, :],
            x_fp8_col.to(x_fp8_col_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            x_scales_col_ptr + tl.cast(pid_m, tl.int64) * N + offs_n,
            tl.reshape(1.0 / x_scales_col, BLOCK_SIZE),
            mask=offs_n < N,
        )


# Blockwise for Segment M
@triton.jit(repr=_quant_fp8_blockwise_segment_m_kernel_repr)
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

    m_range_start, m_range_end, _bid = _find_segment_tile_range(
        pid_m, batch_size, seg_indptr, scales_seg_indptr_ptr, BLOCK_SIZE
    )
    if m_range_end - m_range_start == 0:
        return

    # int64 offsets: m_range_start is a global token offset that spans all
    # segments, so m_range_start * N overflows int32 at long context.
    offs_m = tl.cast(m_range_start + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < m_range_end) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    x_ptrs = x_ptr + offs_m[:, None] * N + offs_n[None, :]
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_tile_abs = tl.abs(x_tile)

    x_fp8_tile, x_scales_tile = _compute_scale_and_quant(x_tile, x_tile_abs, 0, FP8_MAX)

    # Store
    x_fp8_ptrs = x_fp8_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(x_fp8_ptrs, x_fp8_tile.to(x_fp8_ptr.dtype.element_ty), mask=mask)

    scale_offs = tl.cast(pid_m, tl.int64) * N + offs_n
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
@triton.jit(repr=_quant_fp8_blockwise_for_weight_kernel_repr)
def quant_fp8_blockwise_for_weight_kernel(
    w_ptr,
    w_fp8_ptr,
    w_scales_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # int64 for the data offset: bid * M * N overflows int32 for stacked MoE
    # weights (e.g. B=256, M*N≈15M wraps past 2^31 around bid=146).
    bid = tl.program_id(axis=0).to(tl.int64)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    batch_offset_w = bid * M * N
    batch_offset_scales = bid * tl.cdiv(M, BLOCK_SIZE) * tl.cdiv(N, BLOCK_SIZE)

    offs_m = tl.cast(pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load [BLOCK_SIZE, BLOCK_SIZE]
    w_ptrs = w_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

    w_tile_abs = tl.abs(w_tile)
    # Global (2-D) amax: _compute_scale_and_quant requires a 1-D axis so we
    # inline the same semantics here for the scalar-scale weight case.
    w_tile_max = tl.max(w_tile_abs)
    w_tile_max = tl.minimum(w_tile_max, FP8_MAX)  # Inf guard (see helper)
    w_tile_max = tl.maximum(w_tile_max, 1e-30)  # zero-block guard (see helper)
    w_scales = FP8_MAX / w_tile_max
    w_fp8_tile = tl.clamp(w_tile * w_scales, min=-FP8_MAX, max=FP8_MAX)

    # Store
    w_fp8_ptrs = w_fp8_ptr + batch_offset_w + offs_m[:, None] * N + offs_n[None, :]
    tl.store(w_fp8_ptrs, w_fp8_tile.to(w_fp8_ptr.dtype.element_ty), mask=mask)
    # Store scale
    scale_offs = batch_offset_scales + pid_m * tl.cdiv(N, BLOCK_SIZE) + pid_n
    w_scales_inv = 1.0 / w_scales
    tl.store(w_scales_ptr + scale_offs, w_scales_inv)


# Re-quantize FP8 (row-wise 1×BLOCK) → FP8 (col-wise BLOCK×1) in one pass.
# Avoids a BF16 roundtrip: dequant with saved row scales, then re-quant along the
# column axis.  Used in the blockwise2d WGrad backward (Jet-RL §4.2) where the
# forward activation was stored as FP8 (1×128) and WGrad needs it col-wise (128×1).
#
# x_fp8_ptr    [M, K]              FP8 input, row-wise 1×BLOCK quantized
# x_scales_ptr [M, K//BLOCK_SIZE]  float32 dequant row scales
# y_fp8_ptr    [M, K]              FP8 output, col-wise BLOCK×1 quantized
# y_scales_ptr [M//BLOCK_SIZE, K]  float32 dequant col scales
@triton.jit(repr=_requant_fp8_row_to_col_kernel_repr)
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

    # Col-wise (axis=0) requant via shared helper: Inf guard + zero-block floor.
    x_abs = tl.abs(x_f32)
    y_f32, col_scale = _compute_scale_and_quant(x_f32, x_abs, 0, FP8_MAX)

    tl.store(
        y_fp8_ptr + offs_m[:, None] * K + offs_k[None, :],
        y_f32.to(y_fp8_ptr.dtype.element_ty),
        mask=mask,
    )

    # Col dequant scales: shape (M//BLOCK_SIZE, K), one float per column element.
    col_scale_inv = tl.reshape(1.0 / col_scale, BLOCK_SIZE)  # (BLOCK,) dequant per col
    tl.store(y_scales_ptr + pid_m * K + offs_k, col_scale_inv, mask=offs_k < K)
