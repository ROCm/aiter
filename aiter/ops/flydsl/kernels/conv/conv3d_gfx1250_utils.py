# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx1250 pieces for FlyDSL implicit-GEMM conv3d (TDM + WMMA).

Geometry helpers shared with gfx950 are imported from ``conv3d_gfx950_utils``.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .conv3d_gfx950_utils import (  # noqa: F401 — re-export for callers
    BF16_BYTES,
    CONV_COMPILE_HINTS,
    OOB_SENTINEL_ELEM,
    _as_stream,
    flat_buffer_view,
    out_extent,
)

# gfx1250 WMMA bf16 → f32
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32
WAVE_SIZE = 32
LDS_PAD = 8
TILE_K = WMMA_K
# Channels packed per K-tile (same as TILE_K for bf16 WMMA).
CGP_ALIGN = TILE_K

DEFAULT_TILE = (128, 128, 2, 4)  # tile_m, tile_n, m_warp, n_warp
DEFAULT_BLOCK_ROWS = 8  # R such that R * C = tile_m with C % 16 == 0


def validate_launch_config_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows):
    """Why this tile cannot compile on gfx1250, or None."""
    block_threads = wave_m * wave_n * WAVE_SIZE
    if block_threads > 1024:
        return f"BLOCK_THREADS={block_threads} exceeds 1024"
    if block_rows < 1 or tile_m % block_rows:
        return f"tile_m={tile_m} not divisible by block_rows={block_rows}"
    cols = tile_m // block_rows
    if cols % WMMA_M:
        return f"block cols={cols} not divisible by WMMA_M={WMMA_M}"
    if tile_m % (wave_m * WMMA_M):
        return f"TILE_M={tile_m} not divisible by WAVE_M*{WMMA_M}"
    if tile_n % (wave_n * WMMA_N):
        return f"TILE_N={tile_n} not divisible by WAVE_N*{WMMA_N}"
    return None


class TileConfig1250(NamedTuple):
    tile_m: int
    tile_n: int
    tile_k: int
    wave_m: int
    wave_n: int
    block_threads: int
    block_rows: int
    block_cols: int
    mi_m: int
    mi_n: int
    num_warps: int
    warp_tile_m: int
    warp_tile_n: int
    lds_k_stride: int
    lds_a_elems_per_buf: int
    lds_b_elems_per_buf: int


def make_tile_config_gfx1250(tile, block_rows):
    tile_m, tile_n, wave_m, wave_n = tile
    invalid = validate_launch_config_gfx1250(tile_m, tile_n, wave_m, wave_n, block_rows)
    assert invalid is None, invalid
    block_threads = wave_m * wave_n * WAVE_SIZE
    cols = tile_m // block_rows
    lds_k = TILE_K + LDS_PAD
    return TileConfig1250(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=TILE_K,
        wave_m=wave_m,
        wave_n=wave_n,
        block_threads=block_threads,
        block_rows=block_rows,
        block_cols=cols,
        mi_m=tile_m // wave_m // WMMA_M,
        mi_n=tile_n // wave_n // WMMA_N,
        num_warps=wave_m * wave_n,
        warp_tile_m=tile_m // wave_m,
        warp_tile_n=tile_n // wave_n,
        lds_k_stride=lds_k,
        lds_a_elems_per_buf=tile_m * lds_k + LDS_PAD,
        lds_b_elems_per_buf=tile_n * lds_k + LDS_PAD,
    )


class ConvGeometry1250(NamedTuple):
    do: int
    ho: int
    wo: int
    dhw: int
    hw_o: int
    npq: int
    cgp: int
    crs: int


def make_conv_geometry_gfx1250(n, cgp, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, dt, dh, dw):
    """Output extents after host pre-pad (kernel sees pt=ph=pw=0)."""
    do = out_extent(d, pt, dt, kt, st)
    ho = out_extent(h, ph, dh, kh, sh)
    wo = out_extent(w, pw, dw, kw, sw)
    dhw = do * ho * wo
    return ConvGeometry1250(
        do=do,
        ho=ho,
        wo=wo,
        dhw=dhw,
        hw_o=ho * wo,
        npq=n * dhw,
        cgp=cgp,
        crs=cgp * kt * kh * kw,
    )


class LaunchGrid1250(NamedTuple):
    grid_m: int
    grid_y: int
    tile_m: int
    tile_n: int
    block_rows: int
    block_cols: int
    hb: int
    wb: int
    do: int
    ho: int
    wo: int
    n: int
    groups: int
    kg: int
    cgp: int
    tiles_per_group: int
    n_tail: bool
    row_chk: bool
    k_tiles: int


def make_launch_grid_gfx1250(n, geom, cfg, groups, k):
    rows, cols = cfg.block_rows, cfg.block_cols
    hb = -(-geom.ho // rows)
    wb = -(-geom.wo // cols)
    grid_m = n * geom.do * hb * wb
    kg = k // groups
    tiles_per_group = (kg + cfg.tile_n - 1) // cfg.tile_n
    n_tail = kg % cfg.tile_n != 0
    row_chk = geom.ho % rows != 0 or geom.wo % cols != 0
    k_tiles = geom.crs // cfg.tile_k
    assert geom.crs % cfg.tile_k == 0, f"crs={geom.crs} not divisible by tile_k={cfg.tile_k}"
    return LaunchGrid1250(
        grid_m=grid_m,
        grid_y=groups * tiles_per_group,
        tile_m=cfg.tile_m,
        tile_n=cfg.tile_n,
        block_rows=rows,
        block_cols=cols,
        hb=hb,
        wb=wb,
        do=geom.do,
        ho=geom.ho,
        wo=geom.wo,
        n=n,
        groups=groups,
        kg=kg,
        cgp=geom.cgp,
        tiles_per_group=tiles_per_group,
        n_tail=n_tail,
        row_chk=row_chk,
        k_tiles=k_tiles,
    )


def decode_block_2d(grid):
    """Device: decode (bid_x, bid_y) → (n, ot, oh0, ow0, n_offset, n_local, gi)."""
    bid_x = fx.Int64(gpu.block_id("x"))
    bid_y = fx.Int32(gpu.block_id("y"))
    wb = fx.Int64(grid.wb)
    hb = fx.Int64(grid.hb)
    owb = bid_x % wb
    band = bid_x // wb
    oh0 = (band % hb) * grid.block_rows
    ot_n = band // hb
    do = fx.Int64(grid.do)
    n_idx = ot_n // do
    ot = ot_n % do
    ow0 = owb * grid.block_cols

    if const_expr(grid.groups > 1):
        gi = bid_y // grid.tiles_per_group
        n_local = (bid_y % grid.tiles_per_group) * grid.tile_n
        n_offset = gi * grid.kg + n_local
    else:
        gi = fx.Int32(0)
        n_offset = bid_y * grid.tile_n
        n_local = n_offset
    return n_idx, ot, oh0, ow0, n_offset, n_local, gi


def make_shared_storage_gfx1250(elem_ty, cfg, num_buffers, patch_elems=0):
    """LDS: num_buffers A tiles + num_buffers B tiles; or patch double-buffer for A."""
    if patch_elems:
        a_elems = 2 * patch_elems
        b_elems = num_buffers * cfg.lds_b_elems_per_buf
    else:
        a_elems = num_buffers * cfg.lds_a_elems_per_buf
        b_elems = num_buffers * cfg.lds_b_elems_per_buf

    @fx.struct
    class SharedStorage:
        a: fx.Array[elem_ty, a_elems, 16]
        b: fx.Array[elem_ty, b_elems, 16]

    return SharedStorage


def lane_bases_rowmajor(warp_base, reps, lds_stride, lane16, lane_kgrp):
    """Per-WMMA-atom LDS element offsets for a non-transposed (M,K)/(N,K) tile."""
    row_off = (warp_base + lane16) * lds_stride
    k_off = lane_kgrp * 8
    return [
        row_off + rep * WMMA_M * lds_stride + k_off for rep in range_constexpr(reps)
    ]


def load_wmma_frag(mem, base_idx, slot_off, ks, rep, bases, lds_stride, elem_ty):
    """Load one 16-wide bf16 WMMA fragment (two vec8 halves) from padded LDS."""
    vec8_ty = T.vec(8, elem_ty.ir_type)
    halves = []
    for k_half in range_constexpr(2):
        off = bases[rep] + slot_off + (ks * WMMA_K + k_half * 16)
        halves.append(fx.ptr_load(fx.add_offset(mem, off), result_type=vec8_ty))
    return halves[0].shuffle(halves[1], list(range(16)))


def rmem_vec(v, n, dtype):
    t = fx.make_rmem_tensor(n, dtype)
    t.store(v if isinstance(v, fx.Vector) else fx.Vector(v))
    return t
