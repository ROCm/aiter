# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import (
    _mxfp4_pack_op,
    _mxfp4_scale_from_amax,
    _mxfp4_sr_pack,
    _mxfp4_sr_random_words,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.mxfp4 import (
    mxfp4_apply_quant_scale,
    mxfp4_shuffled_data_offsets,
    mxfp4_swizzled_scale_offsets,
    normalized_hadamard16,
)

_dual_layout_quant_mxfp4_repr = make_kernel_repr(
    "_dual_layout_quant_mxfp4_kernel",
    [
        "BLOCK_SIZE",
        "HADAMARD_SIZE",
        "USE_SR_ROW",
        "USE_SR_TRANSPOSED",
        "SWIZZLE_SCALE",
        "SHUFFLE_COL",
        "num_warps",
        "num_stages",
    ],
)


@triton.jit(repr=_dual_layout_quant_mxfp4_repr)
def _dual_layout_quant_mxfp4_kernel(
    x_ptr,
    row_ptr,
    row_scale_ptr,
    transposed_ptr,
    transposed_scale_ptr,
    sign_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_row_m_in,
    stride_row_n_in,
    stride_row_scale_m_in,
    stride_row_scale_n_in,
    stride_transposed_m_in,
    stride_transposed_n_in,
    stride_transposed_scale_m_in,
    stride_transposed_scale_n_in,
    M,
    N,
    philox_seed,
    philox_offset_row,
    philox_offset_transposed,
    BLOCK_SIZE: tl.constexpr,
    HADAMARD_SIZE: tl.constexpr,
    USE_SR_ROW: tl.constexpr,
    USE_SR_TRANSPOSED: tl.constexpr,
    SWIZZLE_SCALE: tl.constexpr,
    SHUFFLE_COL: tl.constexpr,
    SCALE_STRIPE: tl.constexpr,
    SCALE_KCHUNK: tl.constexpr,
    SHUFFLE_TILE_ROWS: tl.constexpr,
    SHUFFLE_UNIT_BYTES: tl.constexpr,
):
    """Read one tile and emit MXFP4(x) plus MXFP4(H16(x.T))."""
    tl.static_assert(BLOCK_SIZE == 32)
    tl.static_assert(HADAMARD_SIZE == 16)

    pid_m = tl.cast(tl.program_id(0), tl.int64)
    pid_n = tl.cast(tl.program_id(1), tl.int64)

    stride_x_m = tl.cast(stride_x_m_in, tl.int64)
    stride_x_n = tl.cast(stride_x_n_in, tl.int64)
    stride_row_m = tl.cast(stride_row_m_in, tl.int64)
    stride_row_n = tl.cast(stride_row_n_in, tl.int64)
    stride_row_scale_m = tl.cast(stride_row_scale_m_in, tl.int64)
    stride_row_scale_n = tl.cast(stride_row_scale_n_in, tl.int64)
    stride_transposed_m = tl.cast(stride_transposed_m_in, tl.int64)
    stride_transposed_n = tl.cast(stride_transposed_n_in, tl.int64)
    stride_transposed_scale_m = tl.cast(stride_transposed_scale_m_in, tl.int64)
    stride_transposed_scale_n = tl.cast(stride_transposed_scale_n_in, tl.int64)

    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    x = tl.load(
        x_ptr + rows[:, None] * stride_x_m + cols[None, :] * stride_x_n,
        cache_modifier=".cg",
    )
    x_fp32 = x.to(tl.float32)

    row_grouped = x_fp32.reshape(BLOCK_SIZE, 1, BLOCK_SIZE)
    row_amax = tl.max(tl.abs(row_grouped), axis=-1, keep_dims=True)
    row_scales, row_quant_scale = _mxfp4_scale_from_amax(row_amax)
    row_scales = row_scales.reshape(BLOCK_SIZE, 1)
    if USE_SR_ROW:
        row_random_words = _mxfp4_sr_random_words(
            rows,
            pid_n,
            N,
            philox_seed,
            philox_offset_row,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
        row_packed = _mxfp4_sr_pack(
            x,
            row_scales,
            row_random_words,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        row_packed = _mxfp4_pack_op(
            mxfp4_apply_quant_scale(
                row_grouped, row_scales.reshape(BLOCK_SIZE, 1, 1), row_quant_scale
            ).reshape(BLOCK_SIZE, BLOCK_SIZE),
            BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            USE_ASM=True,
        )

    packed_cols = pid_n * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(tl.int64)
    tl.store(
        row_ptr + rows[:, None] * stride_row_m + packed_cols[None, :] * stride_row_n,
        row_packed,
    )
    row_scale_cols = N // BLOCK_SIZE
    if SWIZZLE_SCALE:
        row_scale_offsets = mxfp4_swizzled_scale_offsets(
            rows,
            pid_n,
            row_scale_cols,
            stripe=SCALE_STRIPE,
            kchunk=SCALE_KCHUNK,
        )
    else:
        row_scale_offsets = rows * stride_row_scale_m + pid_n * stride_row_scale_n
    tl.store(row_scale_ptr + row_scale_offsets, row_scales.reshape(BLOCK_SIZE))

    sign = tl.load(sign_ptr + tl.arange(0, HADAMARD_SIZE).to(tl.int64)).to(tl.float32)
    HADAMARD_ROWS: tl.constexpr = BLOCK_SIZE * (BLOCK_SIZE // HADAMARD_SIZE)
    transposed = tl.trans(x_fp32).reshape(HADAMARD_ROWS, HADAMARD_SIZE)
    transposed = transposed * sign[None, :]
    transposed = normalized_hadamard16(transposed, rows=HADAMARD_ROWS).reshape(
        BLOCK_SIZE, BLOCK_SIZE
    )

    transposed_grouped = transposed.reshape(BLOCK_SIZE, 1, BLOCK_SIZE)
    transposed_amax = tl.max(tl.abs(transposed_grouped), axis=-1, keep_dims=True)
    transposed_scales, transposed_quant_scale = _mxfp4_scale_from_amax(transposed_amax)
    transposed_scales = transposed_scales.reshape(BLOCK_SIZE, 1)
    if USE_SR_TRANSPOSED:
        transposed_random_words = _mxfp4_sr_random_words(
            cols,
            pid_m,
            M,
            philox_seed,
            philox_offset_transposed,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
        )
        transposed_packed = _mxfp4_sr_pack(
            transposed,
            transposed_scales,
            transposed_random_words,
            BLOCK_SIZE_M=BLOCK_SIZE,
            BLOCK_SIZE_N=BLOCK_SIZE,
            MXFP4_QUANT_BLOCK_SIZE=BLOCK_SIZE,
        )
    else:
        transposed_packed = _mxfp4_pack_op(
            mxfp4_apply_quant_scale(
                transposed_grouped,
                transposed_scales.reshape(BLOCK_SIZE, 1, 1),
                transposed_quant_scale,
            ).reshape(BLOCK_SIZE, BLOCK_SIZE),
            BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE,
            USE_ASM=True,
        )

    packed_rows = pid_m * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2).to(tl.int64)
    if SHUFFLE_COL:
        transposed_offsets = mxfp4_shuffled_data_offsets(
            cols[:, None],
            packed_rows[None, :],
            M // 2,
            tile_rows=SHUFFLE_TILE_ROWS,
            unit_bytes=SHUFFLE_UNIT_BYTES,
        )
    else:
        transposed_offsets = (
            cols[:, None] * stride_transposed_m
            + packed_rows[None, :] * stride_transposed_n
        )
    tl.store(transposed_ptr + transposed_offsets, transposed_packed)

    transposed_scale_cols = M // BLOCK_SIZE
    if SWIZZLE_SCALE:
        transposed_scale_offsets = mxfp4_swizzled_scale_offsets(
            cols,
            pid_m,
            transposed_scale_cols,
            stripe=SCALE_STRIPE,
            kchunk=SCALE_KCHUNK,
        )
    else:
        transposed_scale_offsets = (
            cols * stride_transposed_scale_m + pid_m * stride_transposed_scale_n
        )
    tl.store(
        transposed_scale_ptr + transposed_scale_offsets,
        transposed_scales.reshape(BLOCK_SIZE),
    )
