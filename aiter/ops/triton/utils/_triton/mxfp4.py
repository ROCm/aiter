# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

MXFP4_SCALE_STRIPE = 32
MXFP4_SCALE_KCHUNK = 8
MXFP4_SHUFFLE_TILE_ROWS = 16
MXFP4_SHUFFLE_UNIT_BYTES = 8
MXFP4_SHUFFLE_GROUP_BYTES = 32


@triton.jit
def mxfp4_apply_quant_scale(x, scale_e8m0, quant_scale):
    """Scale values without flushing the raw E8M0 254 endpoint."""
    safe_scale = tl.where(scale_e8m0 == 254, tl.exp2(-126.0), quant_scale)
    scaled = x * safe_scale
    return tl.where(scale_e8m0 == 254, scaled * 0.5, scaled)


@triton.jit
def normalized_hadamard16(x, rows: tl.constexpr):
    """Apply normalized Hadamard-16 to the final dimension."""
    x = x * 0.25
    x_reshaped = tl.reshape(x, (rows, 8, 2, 1))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 4, 2, 2))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 2, 2, 4))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    x = tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )

    x_reshaped = tl.reshape(x, (rows, 1, 2, 8))
    x_permuted = tl.permute(x_reshaped, (0, 1, 3, 2))
    upper, lower = tl.split(x_permuted)
    return tl.reshape(
        tl.permute(tl.join(upper + lower, upper - lower), (0, 1, 3, 2)),
        (rows, 16),
    )


@triton.jit
def mxfp4_shuffled_data_offsets(
    rows,
    byte_cols,
    num_byte_cols,
    tile_rows: tl.constexpr,
    unit_bytes: tl.constexpr,
):
    """Return flat offsets for AITER's gfx950 shuffled B layout."""
    units_per_group: tl.constexpr = 4
    tile = rows // tile_rows
    row = rows % tile_rows
    unit = byte_cols // unit_bytes
    within_unit = byte_cols % unit_bytes
    group = unit // units_per_group
    rest = unit % units_per_group
    pair = rest // 2
    half = rest % 2
    groups = num_byte_cols // (unit_bytes * units_per_group)
    dst_unit = ((tile * groups + group) * 2 + pair) * (tile_rows * 2) + row * 2 + half
    return dst_unit * unit_bytes + within_unit


@triton.jit
def mxfp4_swizzled_scale_offsets(
    rows,
    cols,
    num_cols,
    stripe: tl.constexpr,
    kchunk: tl.constexpr,
):
    """Return flat offsets for AITER's gfx950 MX scale layout."""
    stripe_group = rows // stripe
    row_half = (rows % stripe) // (stripe // 2)
    row = rows % (stripe // 2)
    chunk = cols // kchunk
    col_half = (cols % kchunk) // (kchunk // 2)
    col = cols % (kchunk // 2)
    within = (
        (chunk * (kchunk // 2) + col) * (stripe * 2) + row * 4 + col_half * 2 + row_half
    )
    return stripe_group * (num_cols * stripe) + within
