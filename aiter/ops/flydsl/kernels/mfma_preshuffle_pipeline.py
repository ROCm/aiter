"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl.expr.typing import T
import flydsl.expr as fx

from .layout_utils import crd2idx, idx2crd, get as layout_get


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


def _buffer_load_vec(
    buffer_ops, vector, rsrc, idx, *, elem_type, vec_elems, elem_bytes, offset_in_bytes
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

    i32_val = buffer_ops.buffer_load(rsrc, idx_i32, vec_width=vec_width, dtype=T.i32)
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


@dataclass(frozen=True)
class PreshuffleScaleLayout:
    """Container returned by `make_preshuffle_scale_layout`.

    The scale layout is ``(c_mn1, c_k1, 4, 16) : (stride_n0, stride_k0, stride_klane, 1)``.
    Callers compute flat index directly with plain arith::

        idx = mni * stride_n0 + ku * stride_k0 + k_lane * stride_klane + n_lane
    """

    layout_scale: object  # fly layout value (same as PreshuffleBLayout.layout_b)
    stride_n0: object  # index-typed MLIR value (dynamic)
    stride_k0: object  # index-typed MLIR value (= 64)
    stride_klane: object  # index-typed MLIR value (= 16)


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
    from .layout_utils import _div_pow2

    c16 = arith.constant(16, index=True)
    c4 = arith.constant(4, index=True)
    c_mn_pack = arith.constant(mn_pack, index=True)
    c_k_pack = arith.constant(k_pack, index=True)
    c_k_scale = _div_pow2(c_k, scale_block_size)

    c_mn1 = _div_pow2(_div_pow2(c_mn, 16), mn_pack)
    c_k1 = _div_pow2(_div_pow2(c_k_scale, 4), k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    stride_nlane = arith.constant(1, index=True)
    stride_klane = c16
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k1 * stride_k0

    # Build fly layout (i32 strides for fx.make_layout).
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


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels."""
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = arith.constant(16, index=True)
    c64 = arith.constant(64, index=True)
    c4 = arith.constant(4, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    from .layout_utils import _div_pow2

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = _div_pow2(c_k_bytes, 64)
    n0 = _div_pow2(c_n, 16)

    c_kpack_elems = c_kpack if elem_bytes == 1 else _div_pow2(c_kpack, int(elem_bytes))

    stride_nlane = c_kpack_elems
    stride_klane = c16 * stride_nlane
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k0 * stride_k0

    # fly.make_shape requires i32/i64 for dynamic operands (not index).
    # Convert dynamic index values to i32; use Python ints for static constants.
    kpack_elems_static = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
    n0_i32 = arith.index_cast(T.i32, n0)
    c_k0_i32 = arith.index_cast(T.i32, c_k0)
    stride_n0_i32 = arith.index_cast(T.i32, stride_n0)
    stride_k0_i32 = arith.index_cast(T.i32, stride_k0)
    stride_klane_i32 = arith.index_cast(T.i32, stride_klane)
    stride_nlane_i32 = arith.index_cast(T.i32, stride_nlane)

    stride_b = (stride_n0_i32, stride_k0_i32, stride_klane_i32, stride_nlane_i32, 1)
    layout_b = fx.make_layout((n0_i32, c_k0_i32, 4, 16, kpack_elems_static), stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
    """Convert one i32 (4 signed int8 bytes) to 4 bf16 packed as i64.

    Uses shift-based f32->bf16 truncation (lshr 16) instead of arith.truncf
    which on gfx942 expands to ~5 VALU per element. The shift is exact for
    unscaled int8 values and introduces <0.5 ULP error for scaled values.
    """
    vec1_i32_t = T.vec(1, T.i32)
    vec2_i32 = T.i32x2
    vec4_i8 = T.i8x4
    vec1_i64 = T.vec(1, T.i64)

    v1 = vector.from_elements(vec1_i32_t, [val_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    f32_vals = []
    for i in range(4):
        val_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        v = arith.sitofp(T.f32, val_i8)
        if scale_val is not None:
            v = v * scale_val
        f32_vals.append(v)

    c16 = arith.constant(16, type=T.i32)
    c_ffff0000 = arith.constant(0xFFFF0000, type=T.i32)
    bits0 = arith.bitcast(T.i32, f32_vals[0])
    bits1 = arith.bitcast(T.i32, f32_vals[1])
    bits2 = arith.bitcast(T.i32, f32_vals[2])
    bits3 = arith.bitcast(T.i32, f32_vals[3])
    i32_lo = arith.shrui(bits0, c16) | (bits1 & c_ffff0000)
    i32_hi = arith.shrui(bits2, c16) | (bits3 & c_ffff0000)

    v2 = vector.from_elements(vec2_i32, [i32_lo, i32_hi])
    v64 = vector.bitcast(vec1_i64, v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def load_b_raw_w4a16(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ku: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 8,
):
    """Phase 1 of W4A16 B load: issue buffer_load_dword, return raw packed i32.

    Same address calculation as the int4 unpack path in load_b_pack_k32
    but using ku-based indexing for 2-phase latency hiding.
    """
    if kpack_bytes != 8:
        raise ValueError(f"W4A16 requires kpack_bytes=8, got {kpack_bytes!r}")

    c64 = arith.constant(64, index=True)
    half_bytes = kpack_bytes // 2
    c2_idx = arith.constant(2, index=True)
    c4_idx = arith.constant(4, index=True)

    k0_base = base_k // c64
    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = arith.constant(k1_layout_offset, index=True) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * arith.constant(half_bytes, index=True)

    coord_pack = (n_blk, k0, k1_local, n_intra, arith.constant(0, index=True))
    idx_pack = crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + k2_base

    b4 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_bytes,
        elem_type=elem_type,
        vec_elems=4,
        elem_bytes=1,
        offset_in_bytes=True,
    )
    packed32 = vector.extract(
        vector.bitcast(T.vec(1, T.i32), b4),
        static_position=[0],
        dynamic_position=[],
    )
    return packed32


def unpack_b_w4a16(packed32, arith, vector, scale_val=None):
    """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

    Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA.
    """
    c_08080808 = arith.constant(0x08080808, type=T.i32)
    c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=T.i32)
    c_1e = arith.constant(0x1E, type=T.i32)
    c_4_i32 = arith.constant(4, type=T.i32)

    s0 = (packed32 & c_08080808) * c_1e
    even = (packed32 & c_0f0f0f0f) | s0

    t = packed32 >> c_4_i32
    s1 = (t & c_08080808) * c_1e
    odd = (t & c_0f0f0f0f) | s1

    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def load_b_pack_k32(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = arith.constant(64, index=True)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
    idx_pack = crd2idx(coord_pack, layout_b)

    if unpack_int4:
        idx_bytes = idx_pack + k2_base
        b4 = _buffer_load_vec(
            buffer_ops,
            vector,
            b_rsrc,
            idx_bytes,
            elem_type=elem_type,
            vec_elems=4,
            elem_bytes=1,
            offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )

        c_08080808 = arith.constant(0x08080808, type=T.i32)
        c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=T.i32)
        c_1e = arith.constant(0x1E, type=T.i32)
        c_4_i32 = arith.constant(4, type=T.i32)

        s0 = (packed32 & c_08080808) * c_1e
        even = (packed32 & c_0f0f0f0f) | s0

        t = packed32 >> c_4_i32
        s1 = (t & c_08080808) * c_1e
        odd = (t & c_0f0f0f0f) | s1

        v2 = vector.from_elements(T.vec(2, T.i32), [even, odd])
        v64 = vector.bitcast(T.vec(1, T.i64), v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    vec_elems = kpack_bytes // int(elem_bytes)
    b16 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=(elem_bytes == 1),
    )

    b_i32x4 = vector.bitcast(T.i32x4, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


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
    coord_local = idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = layout_get(coord_local, 0)
    col_local_i32 = layout_get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    vector,
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
        vector,
        rsrc,
        idx_i32,
        elem_type=elem_type,
        vec_elems=vec_elems,
        elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    vector.store(v16, lds_memref, [idx0])


def lds_store_8b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    vector.store(v8, lds_memref, [idx0])


def lds_store_4b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
    coord_store = (row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    vector.store(v4, lds_memref, [idx0])


def lds_load_pack_k32(
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
    col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = (curr_row_a_lds, col_base_swz)
        idx_a16 = crd2idx(coord_a16, layout_lds) + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = (curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(coord_a, layout_lds) + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


__all__ = [
    "PreshuffleBLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "load_b_pack_k32",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
]
