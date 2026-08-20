# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Lockstep quad-fanout INT4 two-shot all-reduce (gfx942, TP8).

Same **256-thread Q4 FMA + last-sector map** as the 856.76 µs keeper:
4 B INT4 / thread (``ds_write_b32``), 28 ``v_pk_fma_f16`` on consume,
8-peer ``global_store_dwordx4 nt``, 4 B flags. Compile-time
``codec='c16q4'`` replaces only the scale:

- Two consecutive Q4 threads (16 fp16) share one signed E4M3 of the
  CodecQ4 extremum. Packed nibble path stays INT4.
- Four such bytes pack into the same i32 scale slot eight Q4 threads
  already owned (``SCALE_I32_OFF + tid//8``). Still 128 B of scales,
  1024 B INT4, rank-tile **1152 B**. Not 64-writer 16 B ownership,
  not E8M0, not 1536 B B-scales.

``quant_dtype``: ``'fp16'`` (packed ``v_pk_*_f16`` after in-kernel
bf16 to fp16; default) or ``'bf16'`` (fp32 codec math). Payload HBM
is GEMM bf16.
Pack into slim LDS (8 × 1152 B = 9216 B). Sequential 1-deep inbox.
Grid ``min(tiles, 1216)``, 256 threads. Last-sector 64 B seq pad
(IPC stride 1216 B at ``super_tile=1``). Compile-time ``super_tile``
``ST∈{1,2,4,8}`` concatenates ST rank-tiles under **one** last-sector
(RS and AG): stride ``ST*1152+64`` B, fewer RTTs, same two-shot.
Host keeps ST=1 when ``num_tiles ≤ GRID`` (decode / tile-limited).
Parameterized by ``nbytes``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, rocdl, scf
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops
from .qr_int4_mem import (
    _CM_NT,
    _dsl_if_only,
    _extract_i64,
    _invalidate_l1,
    _load_device_ptr,
    _load_i32_uncached,
    _load_v4i32,
    _make_rsrc,
    _pack_i64_vec,
    _raw,
    _store_v4i32,
)

WORLD = 8
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
TILE_FP16 = TILE_BYTES // 2
GRID = 304 * 4
PHASES = 2
PHASE_RS = 0
PHASE_AG = 1
RANK_TILE_BYTES = 1152
RANK_TILE_I32 = RANK_TILE_BYTES // 4
# Last-sector release: one extra 64 B line after CodecQ4 so seq does
# not clobber scales. Stride of the IPC rank-tile is 1216 B.
SUPER_TILES = (1, 2, 4, 8)
RELEASE_I32_OFF = RANK_TILE_I32
WIRE_TILE_I32 = RANK_TILE_I32 + 16
WIRE_TILE_BYTES = WIRE_TILE_I32 * 4
SCALE_I32_OFF = 256
# Q4 map: 8 threads still share one i32 scale *slot*. c16q4 puts four
# E4M3 bytes in that slot (one per pair of threads / 16 fp16).
GROUP = 8
PAIR = 2
WAVE = 64
WAVES = 4
QUAD_LANES = 4
QUADS_PER_WAVE = WAVE // QUAD_LANES
N_SECTORS = RANK_TILE_BYTES // 64
PACK_I32 = WORLD * RANK_TILE_I32
LDS_BYTES = PACK_I32 * 4

# CodecQ4<half> constants (two packed fp16 lanes per i32).
_K_SCALE_FACTOR = 0xB000B000  # -1/8
_K_SCALE_EPS = 0x00010001
_K_RANGE_MIN = 0xC800C800  # -8
_K_RANGE_MAX = 0x47004700  # +7
_K_RANGE_BIAS = 0x00080008  # +8 as i16x2
_K_MASK_000F = 0x000F000F
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2

# CodecQ4 bf16 path: scale math in fp32. HIP's bf16 CodecQ4 uses the same
# -1/8 / [-8, +7] map; gfx942 has no v_pk_*_bf16, so f32 is the native ALU.


def _i32_ty():
    return ir.IntegerType.get_signless(32)


def _f32_ty():
    return ir.F32Type.get()


def _pk2(op, a, b):
    return fx.Int32(
        llvm.inline_asm(
            _i32_ty(),
            [_raw(a), _raw(b)],
            f"{op} $0, $1, $2",
            "=v,v,v",
            has_side_effects=False,
        )
    )


def _pk_max(a, b):
    return _pk2("v_pk_max_f16", a, b)


def _pk_min(a, b):
    return _pk2("v_pk_min_f16", a, b)


def _pk_mul(a, b):
    return _pk2("v_pk_mul_f16", a, b)


def _pk_add_f16(a, b):
    return _pk2("v_pk_add_f16", a, b)


def _pk_fma_f16(a, b, c):
    """Packed fp16 FMA: ``a * b + c`` as ``v_pk_fma_f16`` (both halves).

    ``_pk_mul`` / ``_pk_add_f16`` are inline asm, so LLVM cannot contract
    them. gfx942 has no packed bf16 FMA; this is the fp16 CodecQ4 path.
    """
    return fx.Int32(
        llvm.inline_asm(
            _i32_ty(),
            [_raw(a), _raw(b), _raw(c)],
            "v_pk_fma_f16 $0, $1, $2, $3",
            "=v,v,v,v",
            has_side_effects=False,
        )
    )


def _pk_add_i16(a, b):
    return _pk2("v_pk_add_i16", a, b)


def _shrui(v, n):
    return v.shrui(fx.Int32(n))


def _pk_rcp(a):
    lo = fx.Int32(
        llvm.inline_asm(
            _i32_ty(),
            [_raw(a)],
            "v_rcp_f16 $0, $1",
            "=v,v",
            has_side_effects=False,
        )
    )
    hi_src = _shrui(a, 16)
    hi = fx.Int32(
        llvm.inline_asm(
            _i32_ty(),
            [_raw(hi_src)],
            "v_rcp_f16 $0, $1",
            "=v,v",
            has_side_effects=False,
        )
    )
    return (lo & fx.Int32(0xFFFF)) | (hi << fx.Int32(16))


def _cvt_f16_f32(f):
    return fx.Int32(
        llvm.inline_asm(
            _i32_ty(),
            [_raw(f)],
            "v_cvt_f16_f32 $0, $1",
            "=v,v",
            has_side_effects=False,
        )
    )


def _cvt_pk_f16_f32(a, b):
    lo = _cvt_f16_f32(a)
    hi = _cvt_f16_f32(b)
    return (lo & fx.Int32(0xFFFF)) | (hi << fx.Int32(16))


def _cvt_f32_f16(bits):
    return fx.Float32(
        llvm.inline_asm(
            _f32_ty(),
            [_raw(bits)],
            "v_cvt_f32_f16 $0, $1",
            "=v,v",
            has_side_effects=False,
        )
    )


def _enable_fp16_ovfl():
    llvm.InlineAsmOp(None, [], "s_setreg_imm32_b32 0xdc1, 1", "", has_side_effects=True)


def _wait_vmem():
    rocdl.s_waitcnt(0)


def _wait_lds():
    """lgkmcnt(0) only so the next sub-tile may reuse 9216 B LDS while NT stores fly."""
    fly_rocdl.s_waitcnt(lgkmcnt=0)


def _fabs(x):
    zero = fx.Float32(0.0)
    return (x < zero).select(-x, x)


def _unpack_f16x2(packed):
    return _cvt_f32_f16(packed), _cvt_f32_f16(_shrui(packed, 16))


def _packed_abs_max(wmax, wmin):
    a0, a1 = _unpack_f16x2(wmax)
    b0, b1 = _unpack_f16x2(wmin)
    r0 = (_fabs(a0) > _fabs(b0)).select(a0, b0)
    r1 = (_fabs(a1) > _fabs(b1)).select(a1, b1)
    return _cvt_pk_f16_f32(r0, r1)


def _pair_signed_ext_f16(atom):
    """Signed CodecQ4 extremum of 16 fp16: this thread's 8 + neighbor (xor 1)."""
    wmax = _pk_max(_pk_max(atom[0], atom[1]), _pk_max(atom[2], atom[3]))
    wmin = _pk_min(_pk_min(atom[0], atom[1]), _pk_min(atom[2], atom[3]))
    wmax = _pk_max(wmax, gpu.shuffle_xor(wmax, 1, WAVE))
    wmin = _pk_min(wmin, gpu.shuffle_xor(wmin, 1, WAVE))
    pk = _packed_abs_max(wmax, wmin)
    a0, a1 = _unpack_f16x2(pk)
    return (_fabs(a0) > _fabs(a1)).select(a0, a1)


def _atom_bf16_to_f16(atom):
    out = []
    for i in range_constexpr(4):
        pair = atom[i]
        lo = (pair & fx.Int32(0xFFFF)) << fx.Int32(16)
        hi = _shrui(pair, 16) << fx.Int32(16)
        out.append(_cvt_pk_f16_f32(lo.bitcast(fx.Float32), hi.bitcast(fx.Float32)))
    return fx.Vector.from_elements(out, fx.Int32)


def _f32_to_bf16_bits(f):
    u = f.bitcast(fx.Int32)
    u = u + fx.Int32(0x7FFF) + (_shrui(u, 16) & fx.Int32(1))
    return _shrui(u, 16) & fx.Int32(0xFFFF)


def _atom_f16_to_bf16(atom):
    out = []
    for i in range_constexpr(4):
        f0, f1 = _unpack_f16x2(atom[i])
        lo = _f32_to_bf16_bits(f0)
        hi = _f32_to_bf16_bits(f1)
        out.append(lo | (hi << fx.Int32(16)))
    return fx.Vector.from_elements(out, fx.Int32)


def _unpack_bf16x2(packed):
    """bf16x2 → two f32. Lossless: bf16 is the top 16 bits of f32."""
    lo = (packed & fx.Int32(0xFFFF)) << fx.Int32(16)
    hi = _shrui(packed, 16) << fx.Int32(16)
    return lo.bitcast(fx.Float32), hi.bitcast(fx.Float32)


def _pack_bf16x2(a, b):
    """Pack two f32 into bf16x2. gfx942 has no v_cvt_pk_bf16_f32 (CDNA4)."""
    lo = _f32_to_bf16_bits(a)
    hi = _f32_to_bf16_bits(b)
    return (lo & fx.Int32(0xFFFF)) | (hi << fx.Int32(16))


def _fma_f32(a, b, c):
    """``a * b + c`` as ``v_fma_f32``. gfx942 has no packed bf16 FMA."""
    return fx.Float32(
        llvm.inline_asm(
            _f32_ty(),
            [_raw(a), _raw(b), _raw(c)],
            "v_fma_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            has_side_effects=False,
        )
    )


def _max_f32(a, b):
    return (a > b).select(a, b)


def _min_f32(a, b):
    return (a < b).select(a, b)


def _shuf_f32(x, delta):
    return fx.Int32(gpu.shuffle_down(x.bitcast(fx.Int32), delta, WAVE)).bitcast(
        fx.Float32
    )


def _bcast_f32(x, lane):
    return fx.Int32(gpu.shuffle_idx(x.bitcast(fx.Int32), lane, WAVE)).bitcast(
        fx.Float32
    )


def _xor1_f32(x):
    return fx.Int32(gpu.shuffle_xor(x.bitcast(fx.Int32), 1, WAVE)).bitcast(fx.Float32)


def _pair_signed_ext_bf16(atom):
    """Signed CodecQ4 extremum of 16 bf16: this thread's 8 + neighbor (xor 1)."""
    lo_max, hi_max = _unpack_bf16x2(atom[0])
    lo_min, hi_min = lo_max, hi_max
    for i in range_constexpr(4):
        a, b = _unpack_bf16x2(atom[i])
        lo_max = _max_f32(lo_max, a)
        hi_max = _max_f32(hi_max, b)
        lo_min = _min_f32(lo_min, a)
        hi_min = _min_f32(hi_min, b)
    lo_max = _max_f32(lo_max, _xor1_f32(lo_max))
    hi_max = _max_f32(hi_max, _xor1_f32(hi_max))
    lo_min = _min_f32(lo_min, _xor1_f32(lo_min))
    hi_min = _min_f32(hi_min, _xor1_f32(hi_min))
    lo = (_fabs(lo_max) > _fabs(lo_min)).select(lo_max, lo_min)
    hi = (_fabs(hi_max) > _fabs(hi_min)).select(hi_max, hi_min)
    return (_fabs(lo) > _fabs(hi)).select(lo, hi)


def _clamp_q4(x):
    return _min_f32(_max_f32(x, fx.Float32(-8.0)), fx.Float32(7.0))


def _clamp_i32(x, lo, hi):
    return (x < lo).select(lo, (x > hi).select(hi, x))


def _f32_to_e4m3(x):
    """Signed E4M3 of a f32: 1 sign + 4 exp (bias 7, e=0 still implicit 1) + 3 mant.

    Group-16 wire scale. Not IEEE OCP E4M3 denorms: e=0 still encodes
    ``(1+m/8)*2^-7`` so typical CodecQ4 extrema (~0.1) stay in range after
    the later ×−1/8. Byte 0 is +0.
    """
    zero = fx.Float32(0.0)
    is_z = x == zero
    sign = (x < zero).select(fx.Int32(0x80), fx.Int32(0))
    bits = _fabs(x).bitcast(fx.Int32)
    e = (_shrui(bits, 23) & fx.Int32(255)) - fx.Int32(127)
    mant = bits & fx.Int32(0x7FFFFF)
    m3 = _shrui(mant + fx.Int32(1 << 19), 20)
    carry = m3 == fx.Int32(8)
    e = e + carry.select(fx.Int32(1), fx.Int32(0))
    m3 = carry.select(fx.Int32(0), m3)
    e4 = _clamp_i32(e + fx.Int32(7), fx.Int32(0), fx.Int32(15))
    byte = sign | (e4 << fx.Int32(3)) | (m3 & fx.Int32(7))
    return is_z.select(fx.Int32(0), byte)


def _e4m3_to_f32(b):
    is_z = b == fx.Int32(0)
    sign = (b & fx.Int32(0x80)) != fx.Int32(0)
    e4 = _shrui(b, 3) & fx.Int32(15)
    m3 = b & fx.Int32(7)
    mag_bits = ((e4 + fx.Int32(120)) << fx.Int32(23)) | (m3 << fx.Int32(20))
    mag = mag_bits.bitcast(fx.Float32)
    signed = sign.select(-mag, mag)
    return is_z.select(fx.Float32(0.0), signed)


def _pack_e4m3_word(e, lane):
    """Four pair-E4M3 bytes into the Q4 i32 scale slot (lanes 0,2,4,6 of GROUP)."""
    base = (lane // GROUP) * GROUP
    e0 = fx.Int32(gpu.shuffle_idx(e, base, WAVE))
    e1 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(2), WAVE))
    e2 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(4), WAVE))
    e3 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(6), WAVE))
    b = fx.Int32(0xFF)
    return (
        (e0 & b)
        | ((e1 & b) << fx.Int32(8))
        | ((e2 & b) << fx.Int32(16))
        | ((e3 & b) << fx.Int32(24))
    )


def _e4m3_decoding_scale_fp16(e):
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    return _cvt_pk_f16_f32(d, d)


def _e4m3_decoding_scale_bf16(e):
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    return _pack_bf16x2(d, d)


def _e4m3_from_scale_word(word, tid):
    """Byte ``(tid % 8)//2`` of the packed E4M3 i32."""
    pair_in_slot = (tid % GROUP) // PAIR
    return word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)


def _quant_atom_bf16(atom, elo, ehi):
    q = []
    for i in range_constexpr(4):
        a, b = _unpack_bf16x2(atom[i])
        i0 = fx.Int32(fx.roundeven(_clamp_q4(a * elo))) + fx.Int32(8)
        i1 = fx.Int32(fx.roundeven(_clamp_q4(b * ehi))) + fx.Int32(8)
        q.append((i0 & fx.Int32(0xFFFF)) | (i1 << fx.Int32(16)))
    return q[0] | (q[1] << fx.Int32(4)) | (q[2] << fx.Int32(8)) | (q[3] << fx.Int32(12))


def _codec_quant_bf16(atom, lane, tid):
    ext = _pair_signed_ext_bf16(atom)
    e = _f32_to_e4m3(ext)
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    enc = fx.Float32(1.0) / (d + fx.Float32(1e-7))
    packed = _quant_atom_bf16(atom, enc, enc)
    is_leader = (tid % GROUP) == 0
    return packed, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant_bf16(packed, scale):
    slo, shi = _unpack_bf16x2(scale)
    out = []
    mask = fx.Int32(_K_MASK_000F)
    eight = fx.Float32(8.0)
    for i in range_constexpr(4):
        q4 = _shrui(packed, i * 4) & mask
        flo = (fx.Float32(q4 & fx.Int32(0xFFFF)) - eight) * slo
        fhi = (fx.Float32(_shrui(q4, 16)) - eight) * shi
        out.append(_pack_bf16x2(flo, fhi))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_dequant_acc_bf16(packed, scale, acc):
    """``(q-8)*scale + acc`` as ``v_fma_f32`` per bf16 lane."""
    slo, shi = _unpack_bf16x2(scale)
    out = []
    mask = fx.Int32(_K_MASK_000F)
    eight = fx.Float32(8.0)
    for i in range_constexpr(4):
        q4 = _shrui(packed, i * 4) & mask
        recon_lo = fx.Float32(q4 & fx.Int32(0xFFFF)) - eight
        recon_hi = fx.Float32(_shrui(q4, 16)) - eight
        a0, a1 = _unpack_bf16x2(acc[i])
        out.append(
            _pack_bf16x2(_fma_f32(recon_lo, slo, a0), _fma_f32(recon_hi, shi, a1))
        )
    return fx.Vector.from_elements(out, fx.Int32)


def _rint_i16_pair(packed_f16):
    f0, f1 = _unpack_f16x2(packed_f16)
    i0 = fx.Int32(fx.roundeven(f0))
    i1 = fx.Int32(fx.roundeven(f1))
    return (i0 & fx.Int32(0xFFFF)) | (i1 << fx.Int32(16))


def _quant_atom_fp16(atom, enc_pk):
    q = []
    for i in range_constexpr(4):
        w = _pk_mul(atom[i], enc_pk)
        w = _pk_max(w, fx.Int32(_K_RANGE_MIN))
        w = _pk_min(w, fx.Int32(_K_RANGE_MAX))
        q.append(_pk_add_i16(_rint_i16_pair(w), fx.Int32(_K_RANGE_BIAS)))
    return q[0] | (q[1] << fx.Int32(4)) | (q[2] << fx.Int32(8)) | (q[3] << fx.Int32(12))


def _codec_quant(atom, lane, tid):
    ext = _pair_signed_ext_f16(atom)
    e = _f32_to_e4m3(ext)
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    enc = _cvt_pk_f16_f32(
        fx.Float32(1.0) / (d + fx.Float32(1e-7)),
        fx.Float32(1.0) / (d + fx.Float32(1e-7)),
    )
    packed = _quant_atom_fp16(atom, enc)
    is_leader = (tid % GROUP) == 0
    return packed, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant(packed, scale):
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = fx.Int32(_K_HALF2_1032)
    for i in range_constexpr(4):
        q4 = (_shrui(packed, i * 4) & mask) | bias_hi
        w = _pk_add_f16(q4, bias_lo)
        out.append(_pk_mul(w, scale))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_dequant_acc(packed, scale, acc):
    """Nibble reconstruct ``(q-8)``, then ``v_pk_fma_f16`` into *acc*.

    ``v_dot2_f32_f16`` is not used: the two fp16 lanes are independent
    channels (lo/hi of each packed pair), not two products into one f32.
    """
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = fx.Int32(_K_HALF2_1032)
    for i in range_constexpr(4):
        q4 = (_shrui(packed, i * 4) & mask) | bias_hi
        w = _pk_add_f16(q4, bias_lo)
        out.append(_pk_fma_f16(w, scale, acc[i]))
    return fx.Vector.from_elements(out, fx.Int32)


def _store_v4i32_nt_nowait(rsrc, elem_off, data):
    buffer_ops.buffer_store(data, rsrc, elem_off, cache_modifier=_CM_NT)


def _uniform_i64(addr):
    """Wave-uniform i64 → SGPR pair so ``make_buffer_rsrc`` does not waterfall.

    ``peer_vec[rank]`` is the same pointer in every lane, but it lives in VGPRs
    after the dynamic extract. LLVM then waterfalls every ``buffer_load`` that
    uses that rsrc (``v_readfirstlane`` + exec mask per load). One
    ``readfirstlane`` here moves the descriptor to SGPRs for the whole kernel.
    """
    raw = addr.ir_value() if hasattr(addr, "ir_value") else _raw(addr)
    return fx.Int64(rocdl.readfirstlane(ir.IntegerType.get_signless(64), raw))


def _store_v4i32_nt_global(addr_i64, data):
    """Per-lane NT ``global_store_dwordx4``. Buffer-rsrc stores waterfall when
    lanes target different peers; VGPR addresses keep the quad-fanout lockstep.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, _raw(addr_i64)).result
    llvm.InlineAsmOp(
        None,
        [ptr, _raw(data)],
        "global_store_dwordx4 $0, $1, off nt",
        "v,v",
        has_side_effects=True,
    )


def _load_i32_nt(rsrc, elem_off):
    return fx.Int32(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=_CM_NT
        )
    )


@_dsl_if_only
def _wait_flag(flag_rsrc, color):
    i32 = T.i32
    initial = _load_i32_uncached(flag_rsrc)
    wait_loop = scf.WhileOp([i32], [initial])
    wait_cond_block = ir.Block.create_at_start(wait_loop.before, [i32])
    wait_body_block = ir.Block.create_at_start(wait_loop.after, [i32])
    with ir.InsertionPoint(wait_cond_block):
        current = wait_cond_block.arguments[0]
        should_wait = fx.Int32(current) != color
        scf.ConditionOp(_raw(should_wait), [current])
    with ir.InsertionPoint(wait_body_block):
        nxt = _load_i32_uncached(flag_rsrc)
        _invalidate_l1()
        scf.YieldOp([nxt])


@fx.struct
class PackStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]


def make_qr_int4_quad_fanout_kernel(
    *,
    world_size: int = WORLD,
    quant_dtype: str = "fp16",
    codec: str = "c16q4",
    super_tile: int = 1,
):
    """Compile a TP8 INT4 two-shot with lockstep 64 B sector stores.

    ``codec`` is compile-time ``'c16q4'``: 256-thread Q4 FMA map with
    group-16 E4M3 scales. ``quant_dtype`` is ``'fp16'`` (packed FMA) or
    ``'bf16'`` (fp32 ALU); payload HBM is still GEMM bf16. ``super_tile``
    ST in {1,2,4,8} concatenates ST rank-tiles under one last-sector per
    phase. Release join is workgroup barrier (tid<8 poll); payload fanout
    is lockstep 8-peer across the WG. Last-sector store is wave 0
    (``linear < 8``).
    """
    if world_size != WORLD:
        raise ValueError(f"only world_size={WORLD} is implemented, got {world_size}")
    if quant_dtype not in ("bf16", "fp16"):
        raise ValueError(f"quant_dtype must be 'bf16' or 'fp16', got {quant_dtype!r}")
    if codec != "c16q4":
        raise ValueError(f"this worktree only implements codec='c16q4', got {codec!r}")
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    use_bf16 = quant_dtype == "bf16"
    release_i32_off = super_tile * RANK_TILE_I32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4
    quant_fn = _codec_quant_bf16 if use_bf16 else _codec_quant
    dequant_fn = _codec_dequant_bf16 if use_bf16 else _codec_dequant
    dequant_acc_fn = _codec_dequant_acc_bf16 if use_bf16 else _codec_dequant_acc
    e4m3_scale_fn = _e4m3_decoding_scale_bf16 if use_bf16 else _e4m3_decoding_scale_fp16

    flags_i32 = PHASES * GRID * WORLD

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def qr_int4_quad_fanout(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
    ):
        if not use_bf16:
            _enable_fp16_ovfl()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
        wave, lane = fx.idx2crd(tid, thread_layout).unpack()
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, liq = fx.idx2crd(lane, quad_layout).unpack()
        linear = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout((WORLD, RANK_TILE_I32), (RANK_TILE_I32, 1))
        # 64 B NT sectors of one 1152 B rank-tile: (sector, lane-in-quad)
        # -> i32 start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((N_SECTORS, QUAD_LANES), (16, 4))
        atom_i32_layout = fx.make_layout((ATOMS, BLOCK), (BLOCK * 4, 4))
        # tid -> (scale_slot, pair_in_slot, lane_in_pair). Four E4M3 bytes
        # live in the same i32 eight Q4 threads already stored.
        scale_own_layout = fx.make_layout(
            (BLOCK // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
        )
        scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(
            tid, scale_own_layout
        ).unpack()
        fanout_s8 = fx.make_layout((WORLD, 8), (1, WORLD))
        fanout_s2 = fx.make_layout((WORLD, 2), (1, WORLD))
        color_layout = fx.make_layout((GRID,), (1,))
        # Inbox: (phase, bid, src, sub) -> i32 base of that rank-tile.
        # Last-sector pad is ST * RANK_TILE_I32 after the ST tiles, not a
        # fake tensor mode.
        wire_slot_layout = fx.make_layout(
            (PHASES, GRID, WORLD, super_tile),
            (
                GRID * WORLD * wire_tile_i32,
                WORLD * wire_tile_i32,
                wire_tile_i32,
                RANK_TILE_I32,
            ),
        )

        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peers = [_load_device_ptr(peer_ptrs, i) for i in range(WORLD)]
        peer_vec = _pack_i64_vec(peers)
        self_rsrc = _make_rsrc(_uniform_i64(_extract_i64(peer_vec, rank)))
        in_rsrc = buffer_ops.create_buffer_resource_from_addr(
            inp_ptr, num_records_bytes=nbytes
        )
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            out_ptr, num_records_bytes=nbytes
        )
        color_rsrc = _make_rsrc(colors_ptr)

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _rank_tile_i32(phase, src):
            return _sub_tile_i32(phase, src, fx.Int32(0))

        def _hbm_i32(tile, atom):
            slot = fx.get_scalar(fx.crd2idx((atom, tid), atom_i32_layout))
            return tile * fx.Int32(TILE_BYTES // 4) + slot

        def _load_color():
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            return fx.Int32(
                buffer_ops.buffer_load(color_rsrc, off, vec_width=1, dtype=T.i32)
            )

        def _store_color(color):
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            buffer_ops.buffer_store(color, color_rsrc, off)
            _wait_vmem()

        def _load_tile_atoms(tile):
            atoms = []
            for atom in range_constexpr(ATOMS):
                raw = _load_v4i32(in_rsrc, _hbm_i32(tile, atom))
                atoms.append(raw if use_bf16 else _atom_bf16_to_f16(raw))
            return atoms

        def _store_tile_atoms(tile, atoms):
            for atom in range_constexpr(ATOMS):
                packed = atoms[atom] if use_bf16 else _atom_f16_to_bf16(atoms[atom])
                _store_v4i32(out_rsrc, _hbm_i32(tile, atom), packed)

        def _lds_write_packet(peer, packed, scale, is_leader):
            fx.memref_store(packed, pack, (peer, tid))
            if is_leader:
                fx.memref_store(
                    scale, pack, (peer, fx.Int32(SCALE_I32_OFF) + scale_slot)
                )

        def _pack_rs(atoms):
            for dest in range_constexpr(WORLD):
                packed, scale, is_leader = quant_fn(atoms[dest], lane, tid)
                _lds_write_packet(fx.Int32(dest), packed, scale, is_leader)

        def _pack_ag(acc):
            packed, scale, is_leader = quant_fn(acc, lane, tid)
            for dest in range_constexpr(WORLD):
                _lds_write_packet(fx.Int32(dest), packed, scale, is_leader)

        def _fanout_nt(phase, inbox_src, sub):
            for stripe in range_constexpr(3):
                n_sec = 2 if stripe == 2 else 8
                fanout = fanout_s2 if stripe == 2 else fanout_s8
                limit = fx.Int32(WORLD * n_sec)

                def _emit(lin, limit=limit, fanout=fanout, stripe=stripe):
                    safe = (lin < limit).select(lin, fx.Int32(0))
                    peer, sec_in = fx.idx2crd(safe, fanout).unpack()
                    sector = fx.Int32(stripe * 8) + sec_in
                    if lin < limit:
                        vec_idx = fx.get_scalar(
                            fx.crd2idx((sector, liq), nt_own_layout)
                        )
                        # 4xi32 NT vector cannot go through the i32 pack view.
                        v4 = fx.ptr_load(
                            smem_ptr + _pack_off(peer, vec_idx),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        dest = _extract_i64(peer_vec, peer)
                        byte_off = fx.Int64(
                            _sub_tile_i32(phase, inbox_src, sub) + vec_idx
                        ) * fx.Int64(4)
                        _store_v4i32_nt_global(dest + byte_off, v4)

                _emit(linear)

        def _fanout_release(phase, inbox_src, color):
            """Last 64 B after ST rank-tiles: seq=color on all 16 i32."""
            limit = fx.Int32(WORLD)
            safe = (linear < limit).select(linear, fx.Int32(0))
            if linear < limit:
                vec_idx = fx.Int32(release_i32_off) + liq * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                dest = _extract_i64(peer_vec, safe)
                byte_off = fx.Int64(
                    _rank_tile_i32(phase, inbox_src) + vec_idx
                ) * fx.Int64(4)
                _store_v4i32_nt_global(dest + byte_off, v4)

        def _publish(phase, inbox_src, color):
            _wait_vmem()
            gpu.barrier()
            _fanout_release(phase, inbox_src, color)

        def _wait_release(phase, color):
            if tid < WORLD:
                elem = _rank_tile_i32(phase, tid) + fx.Int32(release_i32_off)
                _wait_flag(
                    _make_rsrc(
                        _extract_i64(peer_vec, rank) + fx.Int64(elem) * fx.Int64(4)
                    ),
                    color,
                )
            gpu.barrier()

        def _recv_q4(phase, src, sub):
            base = _sub_tile_i32(phase, src, sub)
            packed = _load_i32_nt(self_rsrc, base + tid)
            word = _load_i32_nt(self_rsrc, base + fx.Int32(SCALE_I32_OFF) + scale_slot)
            e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
            scale = e4m3_scale_fn(e)
            return packed, scale

        def _recv_atom(phase, src, sub):
            packed, scale = _recv_q4(phase, src, sub)
            return dequant_fn(packed, scale)

        def _reduce_rs(sub):
            acc = None
            for src in range_constexpr(WORLD):
                packed, scale = _recv_q4(PHASE_RS, fx.Int32(src), sub)
                if acc is None:
                    acc = dequant_fn(packed, scale)
                else:
                    acc = dequant_acc_fn(packed, scale, acc)
            return acc

        def _recv_ag(sub):
            gathered = []
            for src in range_constexpr(WORLD):
                gathered.append(_recv_atom(PHASE_AG, fx.Int32(src), sub))
            return gathered

        n_block_tiles = (num_tiles - bid + fx.Int32(GRID - 1)) // fx.Int32(GRID)
        zero = fx.Int32(0)
        one = fx.Int32(1)
        color = _load_color()
        if super_tile == 1:
            for i in range(zero, n_block_tiles, one):
                tile = bid + i * fx.Int32(GRID)
                atoms = _load_tile_atoms(tile)
                _pack_rs(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_RS, rank, zero)
                _publish(PHASE_RS, rank, color)

                _wait_release(PHASE_RS, color)
                acc = _reduce_rs(zero)

                _pack_ag(acc)
                gpu.barrier()
                _fanout_nt(PHASE_AG, rank, zero)
                _publish(PHASE_AG, rank, color)

                _wait_release(PHASE_AG, color)
                gathered = _recv_ag(zero)
                _store_tile_atoms(tile, gathered)

                color = color + one
        else:
            st_i = fx.Int32(super_tile)
            for i in range(zero, n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(zero, n_this, one):
                    tile = bid + (i + s) * fx.Int32(GRID)
                    atoms = _load_tile_atoms(tile)
                    _pack_rs(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_RS, rank, s)
                    if (s + one) < n_this:
                        _wait_lds()

                _publish(PHASE_RS, rank, color)
                _wait_release(PHASE_RS, color)

                for s in range(zero, n_this, one):
                    acc = _reduce_rs(s)
                    _pack_ag(acc)
                    gpu.barrier()
                    _fanout_nt(PHASE_AG, rank, s)
                    if (s + one) < n_this:
                        _wait_lds()

                _publish(PHASE_AG, rank, color)
                _wait_release(PHASE_AG, color)

                for s in range(zero, n_this, one):
                    gathered = _recv_ag(s)
                    tile = bid + (i + s) * fx.Int32(GRID)
                    _store_tile_atoms(tile, gathered)

                color = color + one
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{BLOCK},{BLOCK}"

    @flyc.jit
    def launch_qr_int4_quad_fanout(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        qr_int4_quad_fanout(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    launch_qr_int4_quad_fanout.func.__name__ = f"launch_qr_int4_quad_fanout_ws{world_size}_{codec}_{quant_dtype}_st{super_tile}"
    try:
        qr_int4_quad_fanout.func.__name__ = (
            f"qr_int4_quad_fanout_{codec}_{quant_dtype}_st{super_tile}"
        )
    except AttributeError:
        pass
    return {
        "launch": launch_qr_int4_quad_fanout,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * GRID * WORLD * wire_tile_bytes,
        "lds_bytes": LDS_BYTES,
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
        "rank_tile_bytes": RANK_TILE_BYTES,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "grid": GRID,
        "block": BLOCK,
        "quant_dtype": quant_dtype,
        "codec": codec,
    }
