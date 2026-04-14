"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl.expr import arith as _arith
import flydsl.expr as fx


def crd2idx(crd, layout):
    """crd2idx returning an index-type scalar (unwraps fly.int_tuple)."""
    result = fx.crd2idx(crd, layout)
    scalar = fx.get_scalar(result)
    if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
        scalar = _arith.IndexCastOp(T.index, scalar).result
    return scalar


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
    buffer_ops,
    vector,
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
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


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
    arith.constant(mn_pack, index=True)
    arith.constant(k_pack, index=True)
    c_k_scale = _div_pow2(c_k, scale_block_size)

    c_mn1 = _div_pow2(_div_pow2(c_mn, 16), mn_pack)
    c_k1 = _div_pow2(_div_pow2(c_k_scale, 4), k_pack)
    if elem_bytes != mn_pack * k_pack:
        raise ValueError(
            f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}"
        )

    arith.constant(1, index=True)
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

    c16 = arith.constant(16, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    from .layout_utils import _div_pow2

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = _div_pow2(c_k_bytes, 64)
    n0 = _div_pow2(c_n, 16)

    c_kpack_elems = c_kpack if elem_bytes == 1 else _div_pow2(c_kpack, int(elem_bytes))

    stride_nlane = c_kpack_elems

    if k_major:
        c32 = arith.constant(32, index=True)
        c2 = arith.constant(2, index=True)
        c_k0 = c_k_bytes // c32
        klane_dim = 2
        stride_klane = c16 * stride_nlane
        stride_n0 = c2 * stride_klane
        stride_k0 = n0 * stride_n0
    else:
        c64 = arith.constant(64, index=True)
        c4 = arith.constant(4, index=True)
        c_k0 = c_k_bytes // c64
        klane_dim = 4
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
    layout_b = fx.make_layout(
        (n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b
    )
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

    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    bits0 = arith.bitcast(T.i32, f32_vals[0])
    bits1 = arith.bitcast(T.i32, f32_vals[1])
    bits2 = arith.bitcast(T.i32, f32_vals[2])
    bits3 = arith.bitcast(T.i32, f32_vals[3])
    i32_lo = (bits0 >> c16) | (bits1 & c_ffff0000)
    i32_hi = (bits2 >> c16) | (bits3 & c_ffff0000)

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

    c64 = fx.Index(64)
    half_bytes = kpack_bytes // 2
    c2_idx = fx.Index(2)
    c4_idx = fx.Index(4)

    k0_base = base_k // c64
    k1_layout_offset = ku * 2
    lane_div_32 = lane_div_16 // c2_idx
    total_k1 = fx.Index(k1_layout_offset) + lane_div_32
    k0 = k0_base + (total_k1 // c4_idx)
    k1_local = total_k1 % c4_idx
    lane_odd = lane_div_16 % c2_idx
    k2_base = lane_odd * fx.Index(half_bytes)

    coord_pack = (n_blk, k0, k1_local, n_intra, fx.Index(0))
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
    c_08080808 = fx.Int32(0x08080808)
    c_0f0f0f0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    c_4_i32 = fx.Int32(4)

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

    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
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

        c_08080808 = fx.Int32(0x08080808)
        c_0f0f0f0f = fx.Int32(0x0F0F0F0F)
        c_1e = fx.Int32(0x1E)
        c_4_i32 = fx.Int32(4)

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
    coord_local = fx.idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = fx.get(coord_local, 0)
    col_local_i32 = fx.get(coord_local, 1)
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


def _cvt_scalef32_pk_bf16_fp4(packed_i32, scale_f32, byte_idx, arith, vector):
    """GFX950 hardware: v_cvt_scalef32_pk_bf16_fp4.

    Converts 2 FP4 E2M1 nibbles (from *byte_idx* of *packed_i32*) to
    2 bf16 values (already scaled by *scale_f32*), returned as i32
    (2 packed bf16).

    One instruction replaces ~36 VALU of the software path.
    """
    from flydsl._mlir.dialects import llvm

    byte_idx_i32 = arith.constant(byte_idx, type=T.i32)
    result_v2bf16 = llvm.call_intrinsic(
        T.vec(2, T.bf16),
        "llvm.amdgcn.cvt.scalef32.pk.bf16.fp4",
        [packed_i32, scale_f32, byte_idx_i32],
        [],
        [],
    )
    vec1_i32_t = T.vec(1, T.i32)
    return vector.extract(
        vector.bitcast(vec1_i32_t, result_v2bf16),
        static_position=[0],
        dynamic_position=[],
    )


def _fp4x4_in_i32_to_bf16x4_i64(packed4, arith, vector, scale_f32=None):
    """Convert 4 FP4 E2M1 nibbles (in 4 bytes of i32) to 4 bf16 packed as i64.

    Each byte of *packed4* holds one nibble in bits [3:0]:
      bit[3] = sign, bits[2:1] = exponent (bias=1), bit[0] = mantissa.

    Unsigned value table (3-bit index):
      000->0.0, 001->0.5, 010->1.0, 011->1.5,
      100->2.0, 101->3.0, 110->4.0, 111->6.0

    *scale_f32*, when provided, is an f32 E8M0 block-scale multiplied
    into every element before truncation to bf16.
    """
    vec1_i32_t = T.vec(1, T.i32)
    vec2_i32 = T.i32x2
    vec4_i8 = T.i8x4
    vec1_i64 = T.vec(1, T.i64)

    v1 = vector.from_elements(vec1_i32_t, [packed4])
    i8x4 = vector.bitcast(vec4_i8, v1)

    c1 = arith.constant(1, type=T.i32)
    c3_shift = arith.constant(3, type=T.i32)
    c7 = arith.constant(7, type=T.i32)
    c22 = arith.constant(22, type=T.i32)
    c23 = arith.constant(23, type=T.i32)
    c31 = arith.constant(31, type=T.i32)
    c126 = arith.constant(126, type=T.i32)
    c_zero = arith.constant(0, type=T.i32)
    c_half_bits = arith.constant(0x3F000000, type=T.i32)  # 0.5f

    f32_vals = []
    for i in range(4):
        nibble_i8 = vector.extract(i8x4, static_position=[i], dynamic_position=[])
        n = arith.extui(T.i32, nibble_i8)

        sign_bit = arith.andi(arith.shrui(n, c3_shift), c1)
        unsigned_val = arith.andi(n, c7)
        exp_field = arith.shrui(unsigned_val, c1)
        mant_field = arith.andi(unsigned_val, c1)

        f32_norm = arith.ori(
            arith.shli(arith.addi(exp_field, c126), c23),
            arith.shli(mant_field, c22),
        )

        is_zero = arith.cmpi(arith.CmpIPredicate.eq, unsigned_val, c_zero)
        is_subnorm = arith.cmpi(arith.CmpIPredicate.eq, unsigned_val, c1)

        f32_bits = arith.select(
            is_zero,
            c_zero,
            arith.select(is_subnorm, c_half_bits, f32_norm),
        )
        f32_bits = arith.ori(f32_bits, arith.shli(sign_bit, c31))

        v = arith.bitcast(T.f32, f32_bits)
        if scale_f32 is not None:
            v = v * scale_f32
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


def load_b_raw_mxfp4(
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
    kpack_bytes: int = 16,
):
    """Load 4 bytes of packed FP4 from a kpack=16 preshuffle layout.

    Addressing for kpack=16 (``shuffle_weight_a16w4`` format):
      - Layout shape: ``(n0, k0, klane=4, nlane=16, kpack=16)``
      - The A-side LDS has klane stride = 8 bf16 elements, advancing
        by 32 bf16 per ku step.  B must match: each klane loads 4 bytes
        (8 FP4 = 8 K elements) at K_start = base_k + ku*32 + lane*8.
      - In the preshuffle layout this maps to:
          k0 = base_k//128 + ku//4
          klane_hw = ku % 4          (compile-time)
          kpack_byte = lane_div_16*4  (runtime)

    Returns a single i32 containing 4 packed bytes (8 FP4 nibbles).
    """
    if kpack_bytes != 16:
        raise ValueError(f"MXFP4 requires kpack_bytes=16, got {kpack_bytes!r}")

    c128 = arith.constant(128, index=True)
    c4 = arith.constant(4, index=True)

    k0_base = base_k // c128
    k0 = k0_base + arith.constant(ku // 4, index=True)
    klane_hw = arith.constant(ku % 4, index=True)
    byte_offset = lane_div_16 * c4

    coord_pack = (n_blk, k0, klane_hw, n_intra, arith.constant(0, index=True))
    idx_pack = crd2idx(coord_pack, layout_b)
    idx_bytes = idx_pack + byte_offset

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


def load_b_raw_mxfp4_dwordx4(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: "ir.Value",
    n_blk: "ir.Value",
    n_intra: "ir.Value",
    lane_div_16: "ir.Value",
    elem_type: "ir.Type",
    kpack_bytes: int = 16,
    cache_modifier: int = 0,
):
    """Load 16 bytes (vec4_i32) of packed FP4 via buffer_load_dwordx4.

    CK-style addressing: klane = lane_div_16, loading the full kpack
    for the thread's sub-lane. Returns vec4_i32 where i32[j] contains
    8 FP4 elements for kIter j.

    Layout: ``(n0, k0, klane=4, nlane=16, kpack=16)``
    """
    if kpack_bytes != 16:
        raise ValueError(f"MXFP4 requires kpack_bytes=16, got {kpack_bytes!r}")

    c128 = arith.constant(128, index=True)
    k0 = base_k // c128

    coord_pack = (n_blk, k0, lane_div_16, n_intra, arith.constant(0, index=True))
    idx_pack = crd2idx(coord_pack, layout_b)

    b16 = _buffer_load_vec(
        buffer_ops,
        vector,
        b_rsrc,
        idx_pack,
        elem_type=elem_type,
        vec_elems=16,
        elem_bytes=1,
        offset_in_bytes=True,
        cache_modifier=cache_modifier,
    )
    return vector.bitcast(T.vec(4, T.i32), b16)


def unpack_b_mxfp4_bf16(packed32, arith, vector, scale_f32=None, use_hw_cvt=True):
    """Unpack 8 FP4 E2M1 nibbles (packed in i32) to 2 x i64 (8 bf16).

    Each byte of *packed32* holds two FP4 nibbles: low nibble = K_even,
    high nibble = K_even+1.  For ``mfma_f32_16x16x16bf16_1k`` the B
    operand needs 4 **consecutive** K values per i64.  So we unpack the
    lower 2 bytes (4 consecutive nibbles) into b0 and the upper 2 bytes
    into b1.

    *scale_f32* is the decoded E8M0 block-scale (as f32).

    When *use_hw_cvt* is True (default), uses the GFX950 hardware
    instruction ``v_cvt_scalef32_pk_bf16_fp4`` which converts 2 FP4
    nibbles → 2 bf16 (with scale) in a single VALU cycle.  This
    replaces ~144 VALU of the software fallback with 4 instructions.

    Returns ``(b0, b1)`` -- two i64 values, each containing 4 bf16 for
    one ``mfma_f32_16x16x16bf16_1k`` call.
    """
    if use_hw_cvt and scale_f32 is not None:
        return _unpack_b_mxfp4_bf16_hw(packed32, arith, vector, scale_f32)

    return _unpack_b_mxfp4_bf16_sw(packed32, arith, vector, scale_f32)


def _unpack_b_mxfp4_bf16_hw(packed32, arith, vector, scale_f32):
    """Hardware fast-path: 4 × v_cvt_scalef32_pk_bf16_fp4."""
    vec2_i32 = T.i32x2
    vec1_i64 = T.vec(1, T.i64)

    # Byte 0,1 → b0 (4 bf16 for first MFMA)
    lo0 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 0, arith, vector)
    lo1 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 1, arith, vector)
    v2_lo = vector.from_elements(vec2_i32, [lo0, lo1])
    v64_lo = vector.bitcast(vec1_i64, v2_lo)
    b0 = vector.extract(v64_lo, static_position=[0], dynamic_position=[])

    # Byte 2,3 → b1 (4 bf16 for second MFMA)
    hi0 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 2, arith, vector)
    hi1 = _cvt_scalef32_pk_bf16_fp4(packed32, scale_f32, 3, arith, vector)
    v2_hi = vector.from_elements(vec2_i32, [hi0, hi1])
    v64_hi = vector.bitcast(vec1_i64, v2_hi)
    b1 = vector.extract(v64_hi, static_position=[0], dynamic_position=[])

    return (b0, b1)


def _unpack_b_mxfp4_bf16_sw(packed32, arith, vector, scale_f32):
    """Software fallback for non-GFX950 targets."""
    c_0f = arith.constant(0x0F, type=T.i32)
    c4 = arith.constant(4, type=T.i32)
    c8 = arith.constant(8, type=T.i32)
    c12 = arith.constant(12, type=T.i32)
    c16 = arith.constant(16, type=T.i32)
    c20 = arith.constant(20, type=T.i32)
    c24 = arith.constant(24, type=T.i32)
    c28 = arith.constant(28, type=T.i32)

    n0 = packed32 & c_0f
    n1 = arith.shrui(packed32, c4) & c_0f
    n2 = arith.shrui(packed32, c8) & c_0f
    n3 = arith.shrui(packed32, c12) & c_0f
    first = n0 | arith.shli(n1, c8) | arith.shli(n2, c16) | arith.shli(n3, c24)

    n4 = arith.shrui(packed32, c16) & c_0f
    n5 = arith.shrui(packed32, c20) & c_0f
    n6 = arith.shrui(packed32, c24) & c_0f
    n7 = arith.shrui(packed32, c28) & c_0f
    second = n4 | arith.shli(n5, c8) | arith.shli(n6, c16) | arith.shli(n7, c24)

    b0 = _fp4x4_in_i32_to_bf16x4_i64(first, arith, vector, scale_f32=scale_f32)
    b1 = _fp4x4_in_i32_to_bf16x4_i64(second, arith, vector, scale_f32=scale_f32)
    return (b0, b1)


def load_e8m0_scale_f32(
    buffer_ops,
    arith,
    vector,
    *,
    rsrc,
    scale_layout,
    ku,
    mni,
    lane_div_16,
    lane_mod_16,
    mn_pack=2,
    k_pack=2,
):
    """Load E8M0 scale bytes and decode to f32 = 2^(e - 127).

    The scale buffer is pre-shuffled by ``shuffle_scale_a16w4`` into the
    layout ``(c_mn1, c_k1, KLane=4, NLane=16)`` where each element is
    a packed i32 holding ``mn_pack * k_pack`` E8M0 bytes.

    *ku* and *mni* are dynamic MLIR index-typed SSA values representing
    the scale's K-block and N-block indices.  This function computes:

        idx = mni * stride_n0 + ku * stride_k0
              + lane_div_16 * stride_klane + lane_mod_16

    loads one i32 (4 packed E8M0 bytes for mn_pack=2, k_pack=2),
    and extracts the byte at sub-position ``(k_sub, mn_sub)`` given by:

        byte_pos = k_sub * mn_pack + mn_sub

    ``k_sub`` and ``mn_sub`` are used to select a specific scale from the
    4-byte pack.  They default to 0, which selects the first packed scale.

    Returns a list of ``k_pack`` f32 SSA values (one per K sub-position).
    """
    idx = (
        mni * scale_layout.stride_n0
        + ku * scale_layout.stride_k0
        + lane_div_16 * scale_layout.stride_klane
        + lane_mod_16
    )
    raw_i32 = buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=T.i32)

    vec4_i8 = T.i8x4
    vec1_i32_t = T.vec(1, T.i32)
    v1 = vector.from_elements(vec1_i32_t, [raw_i32])
    i8x4 = vector.bitcast(vec4_i8, v1)

    c23 = arith.constant(23, type=T.i32)
    results = []
    for ks in range(k_pack):
        byte_pos = ks * mn_pack
        byte_val = vector.extract(i8x4, static_position=[byte_pos], dynamic_position=[])
        byte_u32 = arith.extui(T.i32, byte_val)
        scale_bits = arith.shli(byte_u32, c23)
        results.append(arith.bitcast(T.f32, scale_bits))
    return results


__all__ = [
    "PreshuffleBLayout",
    "PreshuffleScaleLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "load_b_pack_k32",
    "load_b_raw_mxfp4",
    "load_b_raw_w4a16",
    "load_e8m0_scale_f32",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
    "unpack_b_mxfp4_bf16",
    "unpack_b_w4a16",
]
