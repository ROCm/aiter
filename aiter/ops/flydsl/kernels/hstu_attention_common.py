# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared indexing helpers for the FlyDSL HSTU attention kernels.

The forward and backward kernels use the same address-computation idioms with
operand roles relabelled. These are factored here so there is a single source of
truth, and so the thread-coordinate decomposition is expressed with FlyDSL
layout algebra (idx2crd over a make_layout) rather than hand-rolled integer
division/modulo.

All helpers build FlyDSL expressions and must be called from inside a
@flyc.kernel body.
"""

from __future__ import annotations

import flydsl.expr as fx


def decode_lane(tid, num_waves: int, warp_size: int, mfma_n: int):
    """Decompose a flat thread id into (wave_id, lane, lane_div_n, lane_mod_n).

    Uses layout algebra: tid indexes a (num_waves, warp_size) layout to split
    wave/lane, and the lane indexes a (warp_size/mfma_n, mfma_n) layout to split
    the MFMA lane coordinate. Equivalent to tid//warp_size, tid%warp_size,
    lane//mfma_n, lane%mfma_n, expressed as coordinate maps.
    """
    # idx2crd/get yield index-typed coordinates; cast back to Int32 to match the
    # kernels' i32 address arithmetic.
    wave_lane = fx.idx2crd(tid, fx.make_layout((num_waves, warp_size), (warp_size, 1)))
    wave_id = fx.Int32(fx.get(wave_lane, 0))
    lane = fx.Int32(fx.get(wave_lane, 1))

    lane_split = fx.idx2crd(
        lane, fx.make_layout((warp_size // mfma_n, mfma_n), (mfma_n, 1))
    )
    lane_div_n = fx.Int32(fx.get(lane_split, 0))
    lane_mod_n = fx.Int32(fx.get(lane_split, 1))
    return wave_id, lane, lane_div_n, lane_mod_n


def grouped_loader(t, dim: int, g: int):
    """Return a loader that reads a contiguous g-wide vector from a jagged 3D
    tensor t[row, head, :], grouping the trailing dim into (dim/g, g) via a
    layout so the group index selects the vector.

    The row*row_stride term is carried by the tensor arg's own i64-strided
    layout (the leading index is i64), so the base address cannot overflow on
    large packed tensors; the g-wide in-row vector load is a small, coalesced
    access over the < int32 in-row span.
    """
    in_row = fx.make_layout((dim // g, g), (g, 1))

    def load(row_i64, head_val, colgrp):
        sub = t[row_i64, head_val, None]
        return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

    return load


def swz_col(tile_row, col, swz_rows: int, swz_shift: int):
    """XOR swizzle of an LDS column by the tile row (period swz_rows, shift
    swz_shift). Shared by the forward K tile and the backward streamed Q/K LDS
    tiles so the DMA global-fetch column and the LDS read column stay in sync.
    """
    return col ^ ((tile_row & fx.Int32(swz_rows - 1)) << fx.Int32(swz_shift))
