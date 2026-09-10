# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith as _arith
from flydsl.expr.typing import T


def crd2idx(crd, layout):
    """crd2idx returning an index-typed ir.Value (unwraps fly.int_tuple)."""
    scalar = fx.get_scalar(fx.crd2idx(crd, layout)).ir_value()
    if isinstance(scalar.type, ir.IndexType):
        return scalar
    return _arith.IndexCastOp(T.index, scalar).result


def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col XOR ((row & (k_blocks16 - 1)) * 16)

    k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
    bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
    """
    from flydsl.expr import arith as _swz_arith

    mask = k_blocks16 - _swz_arith.index(1)
    rem = _swz_arith.andi(row, mask)
    return col ^ (rem * 16)


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor)."""
    return index // minor_extent, index % minor_extent


def _buffer_load_vec(
    buffer_ops,
    rsrc,
    idx,
    *,
    elem_type,
    vec_elems,
    elem_bytes,
    offset_in_bytes,
    cache_modifier=0,
):
    """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
    from flydsl.expr import arith as _ld_arith

    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4

    if offset_in_bytes:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(2))
    elif elem_bytes == 2:
        idx_i32 = _ld_arith.shrui(idx, _ld_arith.index(1))
    else:
        idx_i32 = idx

    i32_val = buffer_ops.buffer_load(
        rsrc,
        idx_i32,
        vec_width=vec_width,
        dtype=T.i32,
        cache_modifier=cache_modifier,
    )
    if vec_width == 1:
        i32_vec = fx.Vector.from_elements([i32_val], fx.Numeric.from_ir_type(T.i32))
    else:
        i32_vec = i32_val
    return fx.Vector(i32_vec).bitcast(fx.Numeric.from_ir_type(elem_type))


@dataclass(frozen=True)
class PreshuffleScaleLayout:
    """Container returned by `make_preshuffle_scale_layout`.

    The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
    Callers compute flat index directly with plain arith::

        idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
    """

    layout_scale: object
    stride_n0: object
    stride_k0: object
    stride_klane: object


def make_preshuffle_scale_layout(
    arith,
    *,
    c_mn: ir.Value,
    c_k: ir.Value,
    mn_pack: int = 2,
    k_pack: int = 2,
    elem_bytes: int = 4,
    scale_block_size: int = 32,
) -> PreshuffleScaleLayout:
    """Build scale layout matching aiter/CK preshuffle for FP4/FP8 microscale.

    Layout shape: ``(c_mn1, c_k1, 4, 16)`` where
    ``c_mn1 = c_mn / 16 / mn_pack`` and ``c_k1 = (c_k / scale_block_size) / 4 / k_pack``.
    """
    c16 = fx.Index(16)
    c4 = fx.Index(4)
    c_k_scale = c_k // fx.Index(scale_block_size)

    c_mn1 = (c_mn // c16) // fx.Index(mn_pack)
    c_k1 = (c_k_scale // c4) // fx.Index(k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    c_mn1_i32 = arith.index_cast(T.i32, c_mn1)
    c_k1_i32 = arith.index_cast(T.i32, c_k1)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)

    layout_scale = fx.make_layout(
        (c_mn1_i32, c_k1_i32, 4, 16),
        stride=(stride_n0_i32, stride_k0_i32, stride_klane_i32, 1),
    )

    return PreshuffleScaleLayout(
        layout_scale=layout_scale,
        stride_n0=stride_n0,
        stride_k0=stride_k0,
        stride_klane=stride_klane,
    )


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    k_major: bool = False,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels.

    When *k_major* is True the block-level order is K-major (``k_blk`` outermost),
    matching the ``(0,3,1,4,2,5)`` shuffle permutation.  The default N-major
    order (``k_major=False``) matches the legacy ``(0,1,3,4,2,5)`` permutation.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = fx.Index(16)
    c_kpack = fx.Index(kpack_bytes)

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    n0 = c_n // c16

    c_kpack_elems = (
        c_kpack
        if elem_bytes == 1
        else (c_kpack // arith.constant(int(elem_bytes), index=True))
    )

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = fx.Index(32)
        c2 = fx.Index(2)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = fx.Index(64)
        c4 = fx.Index(4)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
        stride_klane = c16 * stride_nlane
        stride_k0 = c4 * stride_klane
        stride_n0 = c_k0 * stride_k0

    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = arith.index_cast(T.i32, n0)
    c_k0_i32 = arith.index_cast(T.i32, c_k0)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
    stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def tile_chunk_coord_i32(
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = fx.idx2crd(fx.Int32(tile_idx_i32), layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    *,
    elem_type,
    idx_i32: ir.Value,
    rsrc,
    vec_elems: int = 16,
    elem_bytes: int = 1,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    return _buffer_load_vec(
        buffer_ops,
        rsrc,
        idx_i32,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def _lds_store_xor16(
    *,
    lds_ptr,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part: ir.Value,
    elem_bytes: int,
):
    """Store one chunk into LDS (fly u8 ptr) with CK-style XOR16 K swizzle."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_swz_bytes = swizzle_xor16(row_local, col_local_i32 * tx_c4, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    idx0 = crd2idx((fx.Int32(row_local), fx.Int32(col_swz)), layout_lds) + lds_base
    byte_off = fx.Int64(idx0) if elem_bytes == 1 else fx.Int64(idx0) * elem_bytes
    u8_ptr = fx.recast_iter(fx.Uint8, lds_ptr)
    fx.ptr_store(fx.Vector(vec_part).bitcast(fx.Uint8), u8_ptr + byte_off)


def lds_store_16b_xor16(
    *,
    lds_ptr,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS (fly u8 ptr) with CK-style XOR16 K swizzle."""
    _lds_store_xor16(
        lds_ptr=lds_ptr,
        layout_lds=layout_lds,
        row_local=row_local,
        col_local_i32=col_local_i32,
        tx_c4=tx_c4,
        k_blocks16=k_blocks16,
        lds_base=lds_base,
        vec_part=vec_part_i32x4,
        elem_bytes=elem_bytes,
    )


def lds_store_8b_xor16(
    *,
    lds_ptr,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS (fly u8 ptr) with CK-style XOR16 K swizzle."""
    _lds_store_xor16(
        lds_ptr=lds_ptr,
        layout_lds=layout_lds,
        row_local=row_local,
        col_local_i32=col_local_i32,
        tx_c4=tx_c4,
        k_blocks16=k_blocks16,
        lds_base=lds_base,
        vec_part=vec_part_i32x2,
        elem_bytes=elem_bytes,
    )


def lds_store_4b_xor16(
    *,
    lds_ptr,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS (fly u8 ptr) with CK-style XOR16 K swizzle."""
    _lds_store_xor16(
        lds_ptr=lds_ptr,
        layout_lds=layout_lds,
        row_local=row_local,
        col_local_i32=col_local_i32,
        tx_c4=tx_c4,
        k_blocks16=k_blocks16,
        lds_base=lds_base,
        vec_part=vec_part_i32x1,
        elem_bytes=elem_bytes,
    )


def xcd_remap_bx_by(
    bx,
    by,
    c_m,
    *,
    tile_m: int,
    tile_n: int,
    N: int,
    xcd_swizzle: int,
    num_xcds: int = 8,
):
    """Remap (bx, by) for L2-cache reuse via XCD swizzle.

    No-op when ``xcd_swizzle <= 0``. Otherwise:
      1. Linearize the original (bx, by) grid round-robin across ``num_xcds``
         XCDs so that contiguous workgroup ids stay on the same XCD.
      2. Re-tile that 1-D order with an M-major group of size ``xcd_swizzle``,
         folding the tail group when ``gy`` does not divide evenly.

    Designed to be called inside a ``@flyc.kernel`` immediately after::

        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx, by = xcd_remap_bx_by(bx, by, c_m, tile_m=..., tile_n=..., N=...,
                                 xcd_swizzle=xcd_swizzle)

    ``c_m`` is the dynamic ``fx.Index`` for runtime ``M``; ``tile_m``,
    ``tile_n``, ``N`` and ``xcd_swizzle`` are compile-time Python ints.
    """
    if xcd_swizzle <= 0:
        return bx, by

    _c1 = fx.arith.constant(1, index=True)
    _c_tm = fx.arith.constant(tile_m, index=True)
    _gx = fx.arith.constant(N // tile_n, index=True)
    _gy = (c_m + _c_tm - _c1) // _c_tm

    _linear_id = bx * _gx + by
    _num_wgs = _gx * _gy

    _c_xcds = fx.arith.constant(num_xcds, index=True)
    _q = _num_wgs // _c_xcds
    _r = _num_wgs % _c_xcds
    _xcd = _linear_id % _c_xcds
    _in_xcd = _linear_id // _c_xcds
    _xcd_lt_r = fx.arith.cmpi(CmpIPredicate.ult, _xcd, _r)
    _clip = fx.arith.select(_xcd_lt_r, _xcd, _r)
    _wgid = _xcd * _q + _clip + _in_xcd

    _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
    _num_wgid_in_group = _c_wgm * _gx
    _group_id = _wgid // _num_wgid_in_group
    _first_pid_m = _group_id * _c_wgm
    _remaining_m = _gy - _first_pid_m
    _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
    _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

    _wgid_in_group = _wgid % _num_wgid_in_group
    new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
    new_by = _wgid_in_group // _group_size_m
    return new_bx, new_by


__all__ = [
    "PreshuffleBLayout",
    "PreshuffleScaleLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "split_row_major_2d",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
    "xcd_remap_bx_by",
]
