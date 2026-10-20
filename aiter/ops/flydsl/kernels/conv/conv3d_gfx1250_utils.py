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
    MAX_DYN_DIVIDEND,
    MAX_GRID_YZ,
    OOB_SENTINEL_ELEM,
    SPLITK_AUTO_MAX_STAGING_BYTES,
    SPLITK_MAX_STAGING_BYTES,
    _as_stream,
    buffer_atomic_add,
    dyn_divisor,
    flat_buffer_view,
    out_extent,
    pack_reciprocal,
    static_divisor,
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


def make_conv_geometry_gfx1250(
    n, cgp, d, h, w, k, kt, kh, kw, st, sh, sw, pt, ph, pw, dt, dh, dw
):
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
    splitk: int
    tiles_per_split: int
    use_splitk: bool
    big_out: bool


def make_launch_grid_gfx1250(n, geom, cfg, groups, k, splitk=1):
    rows, cols = cfg.block_rows, cfg.block_cols
    hb = -(-geom.ho // rows)
    wb = -(-geom.wo // cols)
    grid_m = n * geom.do * hb * wb
    kg = k // groups
    tiles_per_group = (kg + cfg.tile_n - 1) // cfg.tile_n
    n_tail = kg % cfg.tile_n != 0
    row_chk = geom.ho % rows != 0 or geom.wo % cols != 0
    k_tiles = geom.crs // cfg.tile_k
    assert (
        geom.crs % cfg.tile_k == 0
    ), f"crs={geom.crs} not divisible by tile_k={cfg.tile_k}"
    splitk = max(1, min(int(splitk), k_tiles))
    assert (
        k_tiles % splitk == 0
    ), f"splitk={splitk} does not divide k_tiles={k_tiles}; pick via _resolve_splitk"
    tiles_per_split = k_tiles // splitk
    use_splitk = splitk > 1
    big_out = (n * k * geom.do * geom.ho * geom.wo * BF16_BYTES) > 0x7FFFFFFF
    need_chk = row_chk or n_tail
    assert not need_chk or not (use_splitk or big_out), (
        f"{'split-K' if use_splitk else 'BIG_OUT'} cannot mask a tile tail "
        f"(row_chk={row_chk}, n_tail={n_tail})"
    )
    assert (
        not use_splitk or geom.npq * k * 4 <= SPLITK_MAX_STAGING_BYTES
    ), f"split-K staging {geom.npq * k * 4}B exceeds {SPLITK_MAX_STAGING_BYTES}B"
    assert splitk <= MAX_GRID_YZ, f"splitk={splitk} exceeds grid.z limit {MAX_GRID_YZ}"
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
        splitk=splitk,
        tiles_per_split=tiles_per_split,
        use_splitk=use_splitk,
        big_out=big_out,
    )


def decode_block_2d(grid, dyn=None):
    """Device: decode (bid_x, bid_y[, bid_z]) → coords + k_off.

    ``dyn`` is a ``DynShapeArgs1250`` on the variable-resolution path; otherwise
    extents come from the compile-time ``grid``.
    """
    bid_x = fx.Int64(gpu.block_id("x"))
    bid_y = fx.Int32(gpu.block_id("y"))
    if const_expr(dyn is not None):
        wb = fx.Int64(dyn.wb)
        hb = fx.Int64(dyn.hb)
        do = fx.Int64(dyn.do)
        div_wb = dyn.div_wb
        div_hb = dyn.div_hb
        div_do = dyn.div_do
        band, owb = div_wb.divmod(bid_x)
        ot_n, ohb = div_hb.divmod(band)
        n_idx, ot = div_do.divmod(ot_n)
        oh0 = ohb * grid.block_rows
        ow0 = owb * grid.block_cols
    else:
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

    if const_expr(grid.use_splitk):
        split_idx = fx.Int64(gpu.block_id("z"))
        k_off = split_idx * grid.tiles_per_split * TILE_K
    else:
        k_off = fx.Int64(0)
    return n_idx, ot, oh0, ow0, n_offset, n_local, gi, k_off


class DynShapeArgs1250:
    """Runtime scalars for gfx1250 dyn_hw (2D block grid + extents)."""

    FIELDS = (
        "d",
        "h",
        "w",
        "do",
        "ho",
        "wo",
        "hw_o",
        "dhw",
        "npq",
        "hb",
        "wb",
        "rcp_wb",
        "rcp_hb",
        "rcp_do",
        "grid_m",
    )

    __slots__ = FIELDS + ("div_wb", "div_hb", "div_do")

    def __init__(self, *values, unit=(False, False, False)):
        if len(values) != len(self.FIELDS):
            raise ValueError(
                f"DynShapeArgs1250 takes {len(self.FIELDS)} values, got {len(values)}"
            )
        for name, v in zip(self.FIELDS, values):
            setattr(self, name, v)
        unit_wb, unit_hb, unit_do = unit

        def _div(is_unit, value, rcp):
            return static_divisor(1) if is_unit else dyn_divisor(value, rcp)

        self.div_wb = _div(unit_wb, self.wb, self.rcp_wb)
        self.div_hb = _div(unit_hb, self.hb, self.rcp_hb)
        self.div_do = _div(unit_do, self.do, self.rcp_do)


def dyn_shape_values_gfx1250(param, geom, grid):
    """Host-side DynShapeArgs1250 values in FIELDS order."""
    if geom.npq >= MAX_DYN_DIVIDEND:
        raise ValueError(f"npq={geom.npq} reaches the {MAX_DYN_DIVIDEND} dyn_hw bound")
    return (
        param.d,
        param.h,
        param.w,
        geom.do,
        geom.ho,
        geom.wo,
        geom.hw_o,
        geom.dhw,
        geom.npq,
        grid.hb,
        grid.wb,
        pack_reciprocal(grid.wb),
        pack_reciprocal(grid.hb),
        pack_reciprocal(geom.do),
        grid.grid_m,
    )


def unit_divisors_gfx1250(grid, geom):
    """Which of (wb, hb, do) are 1 (folded static divisor)."""
    return (grid.wb == 1, grid.hb == 1, geom.do == 1)


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
