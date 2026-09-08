# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared helpers for the FlyDSL HSTU attention forward and backward kernels.

Two kinds of helpers live here so there is a single source of truth across the
forward, the KV-owned (dV/dK) backward, and the Q-owned (dQ) backward kernels:

* Host-side geometry constants and small host helpers (dtype mapping, per-arch
  DMA/swizzle parameters, LDS capacity) shared by all HSTU kernel builders.
* FlyDSL-expression indexing idioms (thread-coordinate decomposition, jagged
  loaders, LDS column swizzle). These build FlyDSL expressions and must be
  called from inside a @flyc.kernel body.
"""

from __future__ import annotations

import functools
import math as host_math

import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

_LOG2E = host_math.log2(host_math.e)


# ---- Kernel geometry constants (shared by forward and backward) ----

WARP_SIZE = 64
# grid decoded group-major for locality
NUM_GRID_GROUPS = 8
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16
MFMA_LANE_K = 4
MFMA_LANE_K_LOG2 = 2
assert (1 << MFMA_LANE_K_LOG2) == MFMA_LANE_K
MFMA_ELEMS_PER_LANE = (MFMA_M * MFMA_N) // WARP_SIZE


def _dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")


def _arch_dma_params(arch: str | None = None):
    """K-staging params (DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT).

    K columns are XOR-swizzled off LDS banks: swizzled_col = col ^ ((row & (ROWS-1)) << SHIFT).
    gfx942: 32 banks -> dword DMA -> (16, 2); gfx950: 64 banks -> dwordx4 DMA -> (8, 3).
    Both tile a 64-element block and the mask maxes < 64, so the XOR stays in-row (HEAD_DIM_K % 64 == 0).
    """
    if arch is None:
        arch = get_rocm_arch()
    if (arch or "").startswith("gfx942"):
        dma_bytes, k_swz_rows, k_swz_shift = 4, 16, 2
    else:
        dma_bytes, k_swz_rows, k_swz_shift = 16, 8, 3
    return dma_bytes, dma_bytes // 2, k_swz_rows, k_swz_shift


@functools.lru_cache(maxsize=16384)
def lds_cap_bytes(arch: str | None = None) -> int:
    if arch is None:
        arch = get_rocm_arch()
    return get_lds_capacity_bytes(arch)


def decode_lane(tid, num_waves: int, warp_size: int, mfma_n: int):
    """Decompose a flat thread id into (wave_id, lane, lane_div_n, lane_mod_n).

    Uses layout algebra: tid indexes a (num_waves, warp_size) layout to split
    wave/lane, and the lane indexes a (warp_size/mfma_n, mfma_n) layout to split
    the MFMA lane coordinate. Equivalent to tid//warp_size, tid%warp_size,
    lane//mfma_n, lane%mfma_n, expressed as coordinate maps.
    """
    # get_/get_scalar yield a single coordinate mode; cast back to Int32 to match
    # the kernels' i32 address arithmetic.
    wave_lane = fx.idx2crd(tid, fx.make_layout((num_waves, warp_size), (warp_size, 1)))
    wave_id = fx.Int32(fx.get_scalar(fx.get_(wave_lane, 0)))
    lane = fx.Int32(fx.get_scalar(fx.get_(wave_lane, 1)))

    lane_split = fx.idx2crd(
        lane, fx.make_layout((warp_size // mfma_n, mfma_n), (mfma_n, 1))
    )
    lane_div_n = fx.Int32(fx.get_scalar(fx.get_(lane_split, 0)))
    lane_mod_n = fx.Int32(fx.get_scalar(fx.get_(lane_split, 1)))
    return wave_id, lane, lane_div_n, lane_mod_n


def grouped_loader(t, dim: int, g: int):
    """Return a loader that reads a contiguous g-wide vector from a jagged 3D
    tensor t[row, head, :], grouping the trailing dim into (dim/g, g) via a
    layout so the group index selects the vector.
    """
    in_row = fx.make_layout((dim // g, g), (g, 1))

    def load(row_i64, head_val, colgrp):
        sub = t[row_i64, head_val, None]
        return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

    return load


def swz_col(tile_row, col, swz_rows: int, swz_shift: int):
    """XOR swizzle of an LDS column by the tile row (period swz_rows, shift
    swz_shift), matching the forward kernel's K-swizzle. Shared by the streamed
    Q (dV/dK kernel) and streamed K (dQ kernel) LDS tiles.
    """
    return col ^ ((tile_row & fx.Int32(swz_rows - 1)) << fx.Int32(swz_shift))
