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
from flydsl.expr.arith import CmpFPredicate
from flydsl.expr.typing import T


def crd2idx(crd, layout):
    """crd2idx returning an index-typed ir.Value (unwraps fly.int_tuple).

    Version-agnostic: flydsl>=0.2.2 ``get_scalar`` returns an ``ArithValue``
    wrapper (unwrap via ``.ir_value()``), while 0.2.1 returns the raw
    ``ir.Value`` directly.
    """
    scalar = fx.get_scalar(fx.crd2idx(crd, layout))
    if hasattr(scalar, "ir_value"):
        scalar = scalar.ir_value()
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


def lds_row_major_idx(row, col, row_stride, base=None):
    """Linearize a 2D LDS coordinate with explicit index arithmetic."""
    idx = row * row_stride + col
    return idx if base is None else idx + base


def split_row_major_2d(index, minor_extent):
    """Split a linear row-major index into (major, minor)."""
    return index // minor_extent, index % minor_extent


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
        raise ValueError(f"elem_bytes of scale must be {mn_pack} * {k_pack}, got {elem_bytes!r}")

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

    c_kpack_elems = c_kpack if elem_bytes == 1 else (c_kpack // arith.constant(int(elem_bytes), index=True))

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
    layout_b = fx.make_layout((n0_i32, c_k0_i32, klane_dim, 16, kpack_elems_static), stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def _unpack_int4_to_int8_pair(packed32):
    """Split packed int4 dword into two int8 dwords (even/odd nibbles).

    7-op bit manipulation shared by all int4 unpack paths (W4A8, W4A16, W4A_FP8).
    """
    c_08 = fx.Int32(0x08080808)
    c_0f = fx.Int32(0x0F0F0F0F)
    c_1e = fx.Int32(0x1E)
    c_4 = fx.Int32(4)
    s0 = (packed32 & c_08) * c_1e
    even = (packed32 & c_0f) | s0
    t = packed32 >> c_4
    s1 = (t & c_08) * c_1e
    odd = (t & c_0f) | s1
    return even, odd


def _pack_i32_pair_to_i64(lo, hi, vector):
    """Pack two i32 values into one i64 via vector bitcast."""
    v2 = vector.from_elements(T.vec(2, T.i32), [lo, hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


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
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)
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


def _int4_to_bf16x4_i64_gfx950(packed32, nibble_offsets, arith, vector, scale_val=None, defer_scale16=False):
    """Convert 4 int4 nibbles to 4 bf16 packed as i64 using gfx950 instructions.

    Uses v_cvt_off_f32_i4_sdwa with byte_sel to avoid per-nibble shifts.
    Even nibbles (0,2,4,6) → SDWA BYTE_0/1/2/3 on original src.
    Odd nibbles (1,3,5,7)  → SDWA BYTE_0/1/2/3 on (src >> 4).
    Only 1 shift total instead of 7.

    When defer_scale16=True, the ×16 correction factor for v_cvt_off_f32_i4 is
    omitted and must be applied later (e.g. in the epilogue).  This saves VALU
    in the hot loop and uses v_cvt_pk_bf16_f32 for proper f32→bf16 conversion.
    """
    from flydsl._mlir.dialects._arith_ops_gen import MulFOp as _MulFOp
    from flydsl.expr import rocdl

    _uw = _arith._to_raw
    _av = _arith.ArithValue

    src_even = packed32
    src_odd = packed32 >> fx.Int32(4)

    f32_vals = []
    for nib in nibble_offsets:
        byte_idx = nib // 2
        src = src_odd if (nib % 2) else src_even
        v = rocdl.cvt_off_f32_i4(src, byte_sel=byte_idx)
        f32_vals.append(v)

    if defer_scale16:
        # Skip ×16; multiply by scale_val only if groupwise.
        if scale_val is not None:
            raw_scale = _uw(scale_val)
            f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Use v_cvt_pk_bf16_f32 for proper f32→bf16 (no bit-shift trick needed).
        i32_lo = rocdl.cvt_pk_bf16_f32(f32_vals[0], f32_vals[1])
        i32_hi = rocdl.cvt_pk_bf16_f32(f32_vals[2], f32_vals[3])
    else:
        c16 = fx.Float32(16.0)
        if scale_val is not None:
            effective_scale = scale_val * c16
        else:
            effective_scale = c16
        raw_scale = _uw(effective_scale)
        f32_vals = [_MulFOp(v, raw_scale).result for v in f32_vals]
        # Truncate f32→bf16 via bit-shift (exact for scaled int values).
        c16_shift = fx.Int32(16)
        c_ffff0000 = fx.Int32(0xFFFF0000)
        bf16_vals = [arith.bitcast(T.i32, _av(v)) for v in f32_vals]
        i32_lo = (bf16_vals[0] >> c16_shift) | (bf16_vals[1] & c_ffff0000)
        i32_hi = (bf16_vals[2] >> c16_shift) | (bf16_vals[3] & c_ffff0000)

    v2 = vector.from_elements(T.vec(2, T.i32), [i32_lo, i32_hi])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def unpack_b_w4a16(packed32, arith, vector, scale_val=None, use_gfx950_cvt=False, defer_scale16=False):
    """Phase 2 of W4A16 B load: unpack int4->int8 + convert int8->bf16.

    Takes raw packed32 from load_b_raw_w4a16 and produces (b0, b1) --
    two i64 values each containing 4 bf16 for one MFMA.

    When use_gfx950_cvt=True, uses v_cvt_off_f32_i4 + v_cvt_pk_bf16_f32
    for ~2x fewer VALU instructions.

    When defer_scale16=True (requires use_gfx950_cvt=True), the ×16
    correction for v_cvt_off_f32_i4 is omitted; caller must apply it
    in the epilogue.
    """
    if use_gfx950_cvt:
        b0 = _int4_to_bf16x4_i64_gfx950(packed32, [0, 2, 4, 6], arith, vector, scale_val, defer_scale16=defer_scale16)
        b1 = _int4_to_bf16x4_i64_gfx950(packed32, [1, 3, 5, 7], arith, vector, scale_val, defer_scale16=defer_scale16)
        return (b0, b1)
    even, odd = _unpack_int4_to_int8_pair(packed32)
    b0 = _i8x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _i8x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def _fp4_code_to_f32_bits(code, arith):
    """Convert one MX-FP4 (E2M1) 4-bit code (in low nibble of ``code``) to the
    IEEE-754 f32 bit pattern (as an i32), pure-ALU (no gfx950 scaled-convert).

    E2M1 layout: [sign:1][exp:2][mantissa:1]. The 16 representable values are
    {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives (matches
    ``fp4_utils.mxfp4_to_f32`` LUT order).

    Magnitude index ``mag = code & 7`` maps to f32 as:
      mag>=2 (normal): exp_field = mag>>1, mant = mag&1
                       f32_exp = exp_field + 126, f32_mantissa_msb = mant
      mag==1 -> 0.5 (0x3F000000), mag==0 -> 0.0 (subnormal fixups).
    The sign bit (code bit 3) is OR'd into f32 bit 31.
    """
    sign = (code & fx.Int32(0x8)) << fx.Int32(28)  # code bit3 -> f32 bit31
    mag = code & fx.Int32(0x7)
    exp_field = mag >> fx.Int32(1)  # logical shift (0..3)
    mant = mag & fx.Int32(0x1)
    norm_bits = ((exp_field + fx.Int32(126)) << fx.Int32(23)) | (mant << fx.Int32(22))
    is_zero = arith.cmpi(CmpIPredicate.eq, mag, fx.Int32(0))
    is_half = arith.cmpi(CmpIPredicate.eq, mag, fx.Int32(1))
    mag_bits = arith.select(is_half, fx.Int32(0x3F000000), norm_bits)  # 0.5
    mag_bits = arith.select(is_zero, fx.Int32(0x00000000), mag_bits)  # 0.0
    return mag_bits | sign


def _fp4x4_in_i32_to_bf16x4_i64(nib_i32, arith, vector, scale_val=None):
    """Convert one i32 holding 4 E2M1 codes (low nibble of each byte) to 4 bf16
    packed as i64. Mirrors ``_i8x4_in_i32_to_bf16x4_i64`` packing.

    f32->bf16 uses the lshr-16 truncation (exact here: every E2M1 magnitude has
    <=3 significant mantissa bits, so the low 16 f32 bits are zero).

    When ``scale_val`` is given (an f32 per-32-element block scale), each value is
    multiplied by it before bf16 truncation. For MX block scales (E8M0 = pure
    powers of two) this exponent-only multiply is exact and leaves the low 16
    f32 bits zero, so the lshr-16 truncation stays exact.
    """
    f32_bits = [_fp4_code_to_f32_bits((nib_i32 >> fx.Int32(8 * i)) & fx.Int32(0xF), arith) for i in range(4)]
    if scale_val is not None:
        scaled = []
        for b in f32_bits:
            fv = arith.bitcast(T.f32, _arith._to_raw(b))
            fv = fv * scale_val
            scaled.append(arith.bitcast(T.i32, _arith._to_raw(fv)))
        f32_bits = scaled
    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    i32_lo = (f32_bits[0] >> c16) | (f32_bits[1] & c_ffff0000)
    i32_hi = (f32_bits[2] >> c16) | (f32_bits[3] & c_ffff0000)
    v2 = vector.from_elements(T.i32x2, [_arith._to_raw(i32_lo), _arith._to_raw(i32_hi)])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def _fp4x4_nibbles_to_fp8x4_i32(nib_i32, arith):
    """Map 4 E2M1 codes (low nibble of each byte of ``nib_i32``) to 4 E4M3FNUZ
    fp8 bytes packed in one i32, with branchless SWAR integer ops.

    E2M1 magnitude (mag = code & 7) -> E4M3FNUZ byte via a single ``v_perm_b32``
    byte-LUT (mag is already a 0..7 byte index):
      mag: 0    1    2    3    4    5    6    7
      fp8: 0x00 0x38 0x40 0x44 0x48 0x4C 0x50 0x54
    The sign bit (code bit3) goes to fp8 bit7, forced off when mag==0 so E2M1
    -0.0 maps to fp8 +0.0 (0x80 is NaN in FNUZ), matching ``mxfp4_to_f32``.

    The fp8 magnitudes equal the E2M1 magnitudes exactly (E2M1's 8 values are a
    subset of E4M3), so this remap is lossless -- no scale is folded here; the
    per-32 E8M0 block scale is applied later (deferred post-MFMA).
    """
    from flydsl.expr import rocdl

    c7 = fx.Int32(7)
    mag = nib_i32 & fx.Int32(0x07070707)
    # v_perm_b32 byte pool {SRC_HI:SRC_LO}: sel 0..3 -> SRC_LO bytes, 4..7 -> SRC_HI.
    src_lo = fx.Int32(0x44403800)  # bytes[0..3] = fp8[mag 0..3]
    src_hi = fx.Int32(0x54504C48)  # bytes[4..7] = fp8[mag 4..7]
    magbyte = _arith.ArithValue(
        rocdl.perm_b32(_arith._to_raw(src_hi), _arith._to_raw(src_lo), _arith._to_raw(mag))
    )
    nz = ((mag + fx.Int32(0x7F7F7F7F)) >> c7) & fx.Int32(0x01010101)  # 1 per byte if mag>=1
    signbit = (nib_i32 & fx.Int32(0x08080808)) << fx.Int32(4)  # code bit3 -> fp8 bit7 (0x80)
    signmask = nz << c7  # 0x80 where mag>=1 (drop sign on zero -> avoid FNUZ NaN)
    return magbyte | (signbit & signmask)


def dequant_fp4_to_fp8(packed32, arith, vector):
    """Decode 8 packed MX-FP4 (E2M1) nibbles -> one i64 of 8 E4M3FNUZ fp8 bytes,
    the operand for one native fp8 K32 MFMA (``mfma_f32_16x16x32_fp8_fp8``) on
    gfx942 (CDNA3). Weight decode for the native-fp8 MFMA path.

    ``packed32`` holds 8 nibbles in the shared packed-4-bit layout: byte i = low
    nibble ``v[i]`` | high nibble ``v[i+4]`` (i in 0..3). The even nibbles
    ``v0..v3`` and odd nibbles ``v4..v7`` are each remapped E2M1->fp8
    (``_fp4x4_nibbles_to_fp8x4_i32``) and concatenated so the returned i64 is fp8
    bytes ``[v0,v1,v2,v3,v4,v5,v6,v7]`` -- the same 8 contiguous-K values the bf16
    decode path splits across two K16 MFMAs,
    here fed as a single K32 fp8 operand.

    No scale is folded: the per-32 E8M0 block scale (both activation and weight)
    is applied as a deferred post-MFMA FMA on the f32 partial, exact because E8M0
    is a power of two and constant across the block's K reduction.
    """
    c_0f0f = fx.Int32(0x0F0F0F0F)
    even = packed32 & c_0f0f  # v0,v1,v2,v3 in bytes 0..3
    odd = (packed32 >> fx.Int32(4)) & c_0f0f  # v4,v5,v6,v7 in bytes 0..3
    fe = _arith._to_raw(_fp4x4_nibbles_to_fp8x4_i32(even, arith))  # fp8[v0..v3]
    fo = _arith._to_raw(_fp4x4_nibbles_to_fp8x4_i32(odd, arith))  # fp8[v4..v7]
    v2 = vector.from_elements(T.i32x2, [fe, fo])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def scale_fp8_i32(fp8_i32, scale_val, arith, vector):
    """Fold an f32 (power-of-two E8M0) block scale into 4 E4M3FNUZ fp8 bytes.

    Decodes the 4 fp8 bytes to f32 (hardware ``cvt_pk_f32_fp8``), multiplies by
    ``scale_val``, and re-encodes to fp8 (``cvt_pk_fp8_f32``) with the FNUZ
    tiny-negative -> +0 guard (avoids the 0x80 NaN encoding). Returns the scaled
    4 fp8 bytes packed in one i32.

    Native-fp8 activation path: lets the per-32 E8M0 activation scale be applied
    at the gmem->LDS load while keeping the activation in fp8 (so the native fp8
    K32 MFMA can consume it). Because E8M0 is a power of two, ``fp8 * 2^k`` is
    exactly representable in fp8 whenever it stays in range, so this is lossless
    for in-range blocks (only saturating at the fp8 dynamic-range edges).
    """
    from flydsl.expr import rocdl

    lo = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=False)
    hi = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=True)
    f = [
        vector.extract(lo, static_position=[0], dynamic_position=[]),
        vector.extract(lo, static_position=[1], dynamic_position=[]),
        vector.extract(hi, static_position=[0], dynamic_position=[]),
        vector.extract(hi, static_position=[1], dynamic_position=[]),
    ]
    c0 = arith.constant(0.0, type=T.f32)
    c_neg_uf = arith.constant(-(2.0**-8), type=T.f32)
    scaled = []
    for fv in f:
        sv = arith.ArithValue(_arith._to_raw(fv)) * scale_val
        # FNUZ guard: values in (-2^-8, 0) round to 0x80 (NaN); clamp to +0.
        is_tn = arith.andi(
            arith.cmpf(CmpFPredicate.OLT, _arith._to_raw(sv), c0),
            arith.cmpf(CmpFPredicate.OGT, _arith._to_raw(sv), c_neg_uf),
        )
        scaled.append(arith.select(is_tn, c0, sv))
    p = arith.constant(0, type=T.i32)
    p = rocdl.cvt_pk_fp8_f32(T.i32, scaled[0], scaled[1], p, 0)
    p = rocdl.cvt_pk_fp8_f32(T.i32, scaled[2], scaled[3], p, 1)
    return p


def scale_fp8x8_i64(fp8_i64, scale_val, arith, vector):
    """Fold an f32 (power-of-two E8M0) block scale into 8 E4M3FNUZ fp8 bytes (i64).

    Splits the i64 into its two i32 halves and applies :func:`scale_fp8_i32` to
    each, returning the scaled 8 fp8 bytes packed back into one i64. Because the
    scale is a power of two, ``fp8 * 2^k`` is exact in fp8 for in-range values,
    so this is lossless (only saturating at the fp8 dynamic-range edges).

    Native-fp8 weight path: lets the per-32 E8M0 *weight* block scale be folded into
    the fp8 weight operand *before* the K32 MFMA. Necessary because a single fp8
    16x16x32 MFMA reduces K across two MX-FP4 blocks (the K layout pairs each lane
    group ``klane`` with the contiguous stripe ``K[klane*16 .. +15]`` -> block
    ``klane//2``), so a deferred post-MFMA scalar cannot represent the per-block
    weight scale. Each lane's 8 weight bytes lie in exactly one block, so folding
    one ``scale_val`` per (lane, K64 step) is exact.
    """
    v1 = vector.from_elements(T.vec(1, T.i64), [_arith._to_raw(fp8_i64)])
    v2 = vector.bitcast(T.i32x2, v1)
    lo = vector.extract(v2, static_position=[0], dynamic_position=[])
    hi = vector.extract(v2, static_position=[1], dynamic_position=[])
    slo = scale_fp8_i32(lo, scale_val, arith, vector)
    shi = scale_fp8_i32(hi, scale_val, arith, vector)
    out2 = vector.from_elements(T.i32x2, [_arith._to_raw(slo), _arith._to_raw(shi)])
    out64 = vector.bitcast(T.vec(1, T.i64), out2)
    return vector.extract(out64, static_position=[0], dynamic_position=[])


def _fp4x4_in_i32_to_bf16x4_i64_via_fp8(nib_i32, arith, vector):
    """Fast E2M1->bf16 for 4 nibbles (low nibble of each byte of ``nib_i32``).

    Builds the E4M3FNUZ fp8 byte for all 4 nibbles with branchless SWAR integer
    ops (once per i32, not per-nibble), then uses the hardware ``cvt_pk_f32_fp8``
    to expand fp8->f32 -- avoiding the per-nibble IEEE-bit construction and the
    cmp/select special-casing of the pure-ALU path.

    E2M1 magnitude (mag = code & 7) -> E4M3FNUZ byte, gathered for all 4 nibbles
    with a single ``v_perm_b32`` byte-LUT (mag is already a 0..7 byte index):
      mag: 0    1    2    3    4    5    6    7
      fp8: 0x00 0x38 0x40 0x44 0x48 0x4C 0x50 0x54
    The sign bit (code bit3) goes to fp8 bit7, but is forced off when mag==0 so
    E2M1 -0.0 maps to fp8 +0.0 (0x80 is NaN in FNUZ), matching the reference LUT.

    f32->bf16 uses the lshr-16 truncation (exact: every value has <=3 mantissa
    bits, so the low 16 f32 bits are zero).
    """
    from flydsl.expr import rocdl

    fp8_i32 = _fp4x4_nibbles_to_fp8x4_i32(nib_i32, arith)

    lo = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=False)
    hi = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=True)
    f = [
        vector.extract(lo, static_position=[0], dynamic_position=[]),
        vector.extract(lo, static_position=[1], dynamic_position=[]),
        vector.extract(hi, static_position=[0], dynamic_position=[]),
        vector.extract(hi, static_position=[1], dynamic_position=[]),
    ]
    bits = [arith.bitcast(T.i32, _arith._to_raw(fv)) for fv in f]
    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    i32_lo = (bits[0] >> c16) | (bits[1] & c_ffff0000)
    i32_hi = (bits[2] >> c16) | (bits[3] & c_ffff0000)
    v2 = vector.from_elements(T.i32x2, [_arith._to_raw(i32_lo), _arith._to_raw(i32_hi)])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def dequant_fp4_to_bf16(packed32, arith, vector, scale_val=None):
    """Dequantize 8 packed MX-FP4 (E2M1) nibbles -> (b0, b1), two i64 each
    holding 4 bf16, for one bf16 MFMA K64 micro-step on gfx942 (CDNA3).

    Drop-in replacement for ``unpack_b_w4a16`` on the FP4 weight path: same
    even/odd nibble split (b0 = nibbles 0,2,4,6; b1 = 1,3,5,7) so the existing
    int4_bf16 preshuffle layout and MFMA consumption order are reused unchanged.

    ``scale_val`` is the f32 per-32-element E8M0 block scale (one block == one
    K64 micro-step). Because E8M0 scales are powers of two, applying it here as
    an exponent-only multiply on the dequantized weight is exact and equivalent
    to a post-MFMA scale of the dot product (the scale is constant across the
    block's K reduction). Pass ``None`` for the unscaled magnitude.
    """
    c_0f0f = fx.Int32(0x0F0F0F0F)
    even = packed32 & c_0f0f
    odd = (packed32 >> fx.Int32(4)) & c_0f0f
    if scale_val is None:
        # Fast path (gfx942): SWAR fp8-byte build + hardware cvt_pk_f32_fp8.
        b0 = _fp4x4_in_i32_to_bf16x4_i64_via_fp8(even, arith, vector)
        b1 = _fp4x4_in_i32_to_bf16x4_i64_via_fp8(odd, arith, vector)
        return (b0, b1)
    b0 = _fp4x4_in_i32_to_bf16x4_i64(even, arith, vector, scale_val=scale_val)
    b1 = _fp4x4_in_i32_to_bf16x4_i64(odd, arith, vector, scale_val=scale_val)
    return (b0, b1)


def e8m0_to_f32_inkernel(e8m0_i32, arith):
    """Decode an E8M0 block-scale byte (biased exponent) to an f32 scale.

    Equivalent to ``v_ldexp_f32(1.0, byte - 127)`` for the normal range
    (byte in 1..254): the result is ``2^(byte-127)`` built directly as
    ``(byte << 23)`` reinterpreted as f32. The byte==0 (tiny) / byte==0xFF
    (NaN) specials of ``fp4_utils.e8m0_to_f32`` are not reproduced because MX
    weight scales are always in the normal range.
    """
    bits = (e8m0_i32 & fx.Int32(0xFF)) << fx.Int32(23)
    return arith.bitcast(T.f32, _arith._to_raw(bits))


def _fp8x4_in_i32_to_bf16x4_i64(fp8_i32, arith, vector, scale_val=None):
    """Convert one i32 holding 4 E4M3FNUZ fp8 bytes to 4 bf16 packed in one i64.

    gfx942 activation decode for the a8w4 path: expand fp8->f32 with the hardware
    ``cvt_pk_f32_fp8`` (the same intrinsic the FP4 weight decode uses), then
    truncate f32->bf16 via the lshr-16 path. The truncation is exact: E4M3 has
    only 3 mantissa bits, so the low 16 bits of the f32 representation are zero.

    ``scale_val`` is an optional f32 per-32-element E8M0 block scale (one scale
    for the 4 contiguous-K bytes in this i32, which lie in the same MX block).
    Because E8M0 scales are powers of two, folding it in here is exact and
    equivalent to a post-MFMA scale of the dot product. Pass ``None`` to leave
    the magnitude unscaled (deferred-scale path).
    """
    from flydsl.expr import rocdl

    lo = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=False)
    hi = rocdl.cvt_pk_f32_fp8(T.f32x2, _arith._to_raw(fp8_i32), word_sel=True)
    f = [
        vector.extract(lo, static_position=[0], dynamic_position=[]),
        vector.extract(lo, static_position=[1], dynamic_position=[]),
        vector.extract(hi, static_position=[0], dynamic_position=[]),
        vector.extract(hi, static_position=[1], dynamic_position=[]),
    ]
    if scale_val is not None:
        f = [arith.ArithValue(_arith._to_raw(fv)) * scale_val for fv in f]
    bits = [arith.bitcast(T.i32, _arith._to_raw(fv)) for fv in f]
    c16 = fx.Int32(16)
    c_ffff0000 = fx.Int32(0xFFFF0000)
    i32_lo = (bits[0] >> c16) | (bits[1] & c_ffff0000)
    i32_hi = (bits[2] >> c16) | (bits[3] & c_ffff0000)
    v2 = vector.from_elements(T.i32x2, [_arith._to_raw(i32_lo), _arith._to_raw(i32_hi)])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def dequant_fp8_to_bf16(fp8_i64, arith, vector, scale_val=None):
    """Dequantize 8 packed E4M3FNUZ fp8 activation bytes (one i64) -> (a0, a1),
    two i64 each holding 4 bf16, for one bf16 MFMA K-step on gfx942 (CDNA3).

    Mirror of ``dequant_fp4_to_bf16`` for the activation side of the a8w4 path:
    the fp8 byte stream is split into the low/high i32 (4 bytes each) and each is
    expanded fp8->f32->bf16. ``scale_val`` (optional, f32 power-of-two E8M0 block
    scale) is folded in exactly when provided.
    """
    v2 = vector.bitcast(T.i32x2, vector.from_elements(T.vec(1, T.i64), [_arith._to_raw(fp8_i64)]))
    lo_i32 = vector.extract(v2, static_position=[0], dynamic_position=[])
    hi_i32 = vector.extract(v2, static_position=[1], dynamic_position=[])
    a0 = _fp8x4_in_i32_to_bf16x4_i64(lo_i32, arith, vector, scale_val=scale_val)
    a1 = _fp8x4_in_i32_to_bf16x4_i64(hi_i32, arith, vector, scale_val=scale_val)
    return (a0, a1)


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
    return_raw_packed32: bool = False,
    klane_override: ir.Value = None,
    half_override: ir.Value = None,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.

    When ``return_raw_packed32`` is set (requires the packed-4bit layout,
    ``kpack_bytes=8``), returns the raw ``packed32`` (8 packed 4-bit codes)
    using this loader's K32 addressing -- the caller decodes it (e.g. MX-FP4
    E2M1->fp8 for the native-fp8 path) so the weight K-layout matches
    the fp8 activation LDS layout (which this loader is co-designed with),
    unlike the bf16/16x16x16 ``load_b_raw_w4a16`` addressing.

    ``klane_override`` / ``half_override`` (runtime ``ir.Value``s, default
    ``None``) re-route the K-lane coordinate (``k1``) and the within-pack 8-code
    half (``k2``) WITHOUT changing the B memory layout. They exist for the
    native-fp8 stage2 "block-major" read: each fp8 K32 MFMA must consume exactly one MX
    (per-32) scale block, which requires distributing one block's 32 K across all
    4 lane groups (8 each) instead of the default 16-contiguous-per-lane stripe.
    Only the per-lane addressing of the same preshuffled bytes changes.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
    if return_raw_packed32 and kpack_bytes != 8:
        raise ValueError("return_raw_packed32 requires kpack_bytes=8 (packed 4-bit layout)")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = fx.Index(64)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes // c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16 if klane_override is None else klane_override
    half_bytes = kpack_bytes // 2
    if half_override is None:
        k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)
    else:
        k2_base = half_override * arith.constant(half_bytes, index=True)

    coord_pack = (n_blk, k0, k1, n_intra, fx.Index(0))
    idx_pack = crd2idx(tuple(fx.Int32(c) for c in coord_pack), layout_b)

    if unpack_int4 or return_raw_packed32:
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
        if return_raw_packed32:
            return packed32
        even, odd = _unpack_int4_to_int8_pair(packed32)
        return _pack_i32_pair_to_i64(even, odd, vector)

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
    coord_local = fx.idx2crd(fx.Int32(tile_idx_i32), layout_tile_div4)
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
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
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
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
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
    idx0 = crd2idx(tuple(fx.Int32(c) for c in coord_store), layout_lds) + lds_base
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
        idx_a16 = crd2idx(tuple(fx.Int32(c) for c in coord_a16), layout_lds) + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = (curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(tuple(fx.Int32(c) for c in coord_a), layout_lds) + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


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
    _gy = (c_m + _c_tm - _c1) / _c_tm

    _linear_id = bx * _gx + by
    _num_wgs = _gx * _gy

    _c_xcds = fx.arith.constant(num_xcds, index=True)
    _wgs_per_xcd = _num_wgs / _c_xcds
    _wgid = (_linear_id % _c_xcds) * _wgs_per_xcd + (_linear_id / _c_xcds)

    _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
    _num_wgid_in_group = _c_wgm * _gx
    _group_id = _wgid / _num_wgid_in_group
    _first_pid_m = _group_id * _c_wgm
    _remaining_m = _gy - _first_pid_m
    _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
    _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

    _wgid_in_group = _wgid % _num_wgid_in_group
    new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
    new_by = _wgid_in_group / _group_size_m
    return new_bx, new_by


__all__ = [
    "PreshuffleBLayout",
    "PreshuffleScaleLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_row_major_idx",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "make_preshuffle_scale_layout",
    "load_b_pack_k32",
    "split_row_major_2d",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
    "unpack_b_w4a16",
    "xcd_remap_bx_by",
]


# ---------------------------------------------------------------------------
# Groupwise scale load helper (shared by W4A16 and W4A8 groupwise paths)
# ---------------------------------------------------------------------------


def _load_groupwise_scale(
    buffer_ops,
    arith,
    *,
    scale_rsrc,
    expert_offset,
    n_blk,
    n_intra,
    k_pos,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    scale_dtype=None,
):
    """Load one per-group scale value from the scale buffer.

    Computes the linear index into the scale tensor from expert offset,
    N position, and group index derived from ``k_pos``.

    For bf16 scales the tensor uses ``(E, G//2, N, 2)`` layout — two
    adjacent groups for the same N position are packed into one dword.
    We load the raw i32 dword (no extraction) so it can be carried as
    loop state without register copies.  Use :func:`extract_bf16_scale`
    in the compute phase to obtain the f32 value.
    """
    c16 = fx.Index(16)
    n_global = n_blk * c16 + n_intra
    c_group_size = fx.Index(group_size)
    c_npe = fx.Index(n_per_expert)
    group_idx = k_pos // c_group_size
    if scale_dtype is None:
        scale_dtype = T.f32

    if scale_dtype == T.bf16:
        # (E, G//2, N, 2) layout: dword at [e, pair, n] holds bf16 scales
        # for groups 2*pair and 2*pair+1.
        pair_idx = group_idx >> fx.Index(1)  # group_idx // 2
        # Dword index: same flat formula but with G//2 groups
        num_pairs = num_groups // 2
        c_npm1 = fx.Index(num_pairs - 1)
        dword_base = expert_offset * c_npm1 + n_global
        dword_elem = dword_base + pair_idx * c_npe
        dword_idx = arith.index_cast(T.i32, dword_elem)
        # Return raw i32 dword — extraction deferred to compute phase.
        scale_val = buffer_ops.buffer_load(scale_rsrc, dword_idx, vec_width=1, dtype=T.i32)
    else:
        # (E, G, N) layout with f32 dtype
        c_gm1 = fx.Index(num_groups - 1)
        base_scale = expert_offset * c_gm1 + n_global
        elem_idx = base_scale + group_idx * c_npe
        scale_idx_i32 = arith.index_cast(T.i32, elem_idx)
        scale_val = buffer_ops.buffer_load(scale_rsrc, scale_idx_i32, vec_width=1, dtype=T.f32)
    return scale_val


def extract_bf16_scale(arith, scale_raw_i32, ku: int):
    """Extract f32 scale from raw i32 dword loaded by bf16 groupwise path.

    In the ``(E, G//2, N, 2)`` layout two adjacent groups share one dword.
    ``ku`` determines which half: even ku → low bf16, odd ku → high bf16.
    """
    if ku % 2 == 0:
        # Low bf16: shift left by 16 to place in upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 << fx.Int32(16))
    else:
        # High bf16: mask upper 16 bits → f32
        return arith.bitcast(T.f32, scale_raw_i32 & fx.Int32(0xFFFF0000))


# ---------------------------------------------------------------------------
# W4A16 groupwise load / unpack helpers
# ---------------------------------------------------------------------------


def load_b_raw_w4a16_groupwise(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k,
    ku: int,
    n_blk,
    n_intra,
    lane_div_16,
    elem_type,
    scale_rsrc,
    expert_offset,
    num_groups: int,
    group_size: int,
    n_per_expert: int,
    kpack_bytes: int = 8,
    scale_dtype=None,
):
    """Phase 1 of W4A16 groupwise B load: buffer_loads for weight + scale.

    Reuses :func:`load_b_raw_w4a16` for the weight load, then issues an
    additional ``buffer_load_dword`` for the per-group scale.

    Returns ``(packed32, scale_val)``.
    """
    packed32 = load_b_raw_w4a16(
        buffer_ops,
        arith,
        vector,
        arg_b=arg_b,
        b_rsrc=b_rsrc,
        layout_b=layout_b,
        base_k=base_k,
        ku=ku,
        n_blk=n_blk,
        n_intra=n_intra,
        lane_div_16=lane_div_16,
        elem_type=elem_type,
        kpack_bytes=kpack_bytes,
    )
    k_pos = base_k + fx.Index(ku * 32)
    scale_val = _load_groupwise_scale(
        buffer_ops,
        arith,
        scale_rsrc=scale_rsrc,
        expert_offset=expert_offset,
        n_blk=n_blk,
        n_intra=n_intra,
        k_pos=k_pos,
        num_groups=num_groups,
        group_size=group_size,
        n_per_expert=n_per_expert,
        scale_dtype=scale_dtype,
    )
    return (packed32, scale_val)


def unpack_b_w4a16_groupwise(packed32, scale_val, arith, vector, use_gfx950_cvt=False):
    """Phase 2 of W4A16 groupwise: unpack + scale + convert to bf16."""
    return unpack_b_w4a16(packed32, arith, vector, scale_val=scale_val, use_gfx950_cvt=use_gfx950_cvt)
