# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""INT4, INT6 and FP16 wire codecs for QRInt4.

A rank-tile is 256 threads x one 16 B atom, quantized in packed fp16 with a
group-16 signed E4M3 scale. INT4 is one nibble plane; INT6 adds a dense 2-bit
plane, which keeps the nibble plane byte-identical to INT4's and every region
on the 64 B fabric sector grid. FP16 is a passthrough wire format -- the
thread's eight fp16 values verbatim, no quantization -- used to test the
reduce-scatter/all-gather transport in isolation from the codec.

Imported by the mesh and ring kernels, which must agree on it byte for byte.
Depends on ``qr_int_shared`` for ``BLOCK``, ``WAVE`` and ``WAVES``, and on
``I32_BYTES``.
"""

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr

from .qr_int_shared import BLOCK, I32_BYTES, WAVE, WAVES

RANK_TILE_BYTES = 1152
RANK_TILE_I32 = RANK_TILE_BYTES // 4
SUPER_TILES = (1, 8)
# 1024 B INT4 (256 i32) then 128 B group-16 E4M3 (32 i32). Rank-tile 1152 B.
SCALE_I32_OFF = 256
# Two threads (PAIR) share one E4M3; GROUP threads share the i32 slot.
GROUP = 8
PAIR = 2
N_SECTORS = RANK_TILE_BYTES // 64

# Dequant bit-trick: code | 0x6400 then + (-(1024+bias)) as f16x2 reconstructs
# (q - bias). fp16 with exponent field 1024.0 holds the integer in its low
# mantissa bits, so this works for any field that fits below bit 10 -- 4 bits
# with bias 8, 6 bits with bias 32.
_K_MASK_000F = 0x000F000F
_K_MASK_0003 = 0x00030003
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2 = -(1024 + 8)
_K_HALF2_1056 = 0xE420E420  # -1056.0 fp16x2 = -(1024 + 32)

# Largest finite fp16. The encode scale is materialised as fp16, so anything
# above this becomes Inf there; see _codec_quant.
_FP16_MAX = 65504.0


@dataclass(frozen=True)
class Codec:
    """One wire format for a rank-tile: how it packs, and the geometry that implies.

    ``bias`` is both the zero point of the unsigned code and the magnitude of
    the most negative one, because the codec maps a group's signed extremum
    onto ``-bias`` -- that is what uses the asymmetric range fully. The
    decoding factor is therefore ``-1/bias``.

    INT6 is INT4's nibble plane plus a dense 2-bit plane, rather than a 48-bit
    field per thread. A 48-bit field straddles i32 boundaries and leaves the
    regions off the 64 B fabric sector grid; two planes keep every region
    sector-aligned (16 + 8 + 2 = 26) and leave the nibble plane byte-identical
    to INT4's, so ``_fanout_to_next`` needs nothing but a different sector
    count.
    """

    name: str
    bits: int
    bias: int | None
    #: fp16x2 constant added after the ``| 0x6400`` trick: -(1024 + bias).
    dequant_bias: int | None
    #: i32 offset of the dense 2-bit plane in the rank-tile; None when the
    #: codec has only a nibble plane.
    hi2_i32_off: int | None
    scale_i32_off: int | None
    rank_tile_i32: int
    #: Payload i32 a thread contributes per atom.
    n_words_per_thread: int

    @property
    def has_scale(self) -> bool:
        return self.scale_i32_off is not None

    @property
    def dec_step(self) -> float:
        return -1.0 / self.bias

    @property
    def qmin(self) -> float:
        return -float(self.bias)

    @property
    def qmax(self) -> float:
        return float(self.bias - 1)

    @property
    def rank_tile_bytes(self) -> int:
        return self.rank_tile_i32 * I32_BYTES

    @property
    def n_sectors(self) -> int:
        return self.rank_tile_bytes // 64

    def plane_slots(self, tid):
        """``[(i32 offset within the rank-tile, store predicate)]``, one per
        payload word, in the order :func:`_codec_quant` returns them.

        The offset is where this thread's word lives; the predicate says
        whether this thread is the one that stores it -- a Python ``True``
        when every thread owns its word (INT4's nibble plane, fp16's four
        dense planes), or ``hi2_leader`` when a lane pair shares one slot
        (INT6's 2-bit plane). A load ignores the predicate: every thread reads
        a plane's word regardless of who wrote it.
        """
        if self.n_words_per_thread == 1:
            return [(tid, True)]
        if self.hi2_i32_off is not None:
            hi2_leader, hi2_slot = hi2_slot_of(tid)
            return [(tid, True), (fx.Int32(self.hi2_i32_off) + hi2_slot, hi2_leader)]
        # Dense multi-word codec (fp16): every thread owns every word, at a
        # fixed stride of one rank-tile row (BLOCK i32) per word.
        return [
            (fx.Int32(w * BLOCK) + tid, True)
            for w in range_constexpr(self.n_words_per_thread)
        ]


# 1024 B nibbles then 128 B scale; 1152 B rank-tile, 18 sectors.
INT4 = Codec(
    name="int4",
    bits=4,
    bias=8,
    dequant_bias=_K_HALF2_1032,
    hi2_i32_off=None,
    scale_i32_off=SCALE_I32_OFF,
    rank_tile_i32=RANK_TILE_I32,
    n_words_per_thread=1,
)
# 1024 B nibbles, 512 B 2-bit plane, 128 B scale; 1664 B rank-tile, 26 sectors.
INT6 = Codec(
    name="int6",
    bits=6,
    bias=32,
    dequant_bias=_K_HALF2_1056,
    hi2_i32_off=256,
    scale_i32_off=384,
    rank_tile_i32=416,
    n_words_per_thread=2,
)
# Four dense fp16x2 planes, no scale region: 4096 B rank-tile, 64 sectors.
# Passthrough wire format.
FP16 = Codec(
    name="fp16",
    bits=16,
    bias=None,
    dequant_bias=None,
    hi2_i32_off=None,
    scale_i32_off=None,
    rank_tile_i32=BLOCK * 4,
    n_words_per_thread=4,
)
CODECS = {c.name: c for c in (INT4, INT6, FP16)}


def thread_lane(tid):
    """(wave, lane) for *tid* in a ``WAVES x WAVE`` block.

    ``lane`` is the codec's own required argument to :func:`_codec_quant` /
    :func:`_codec_dequant` (the pairing and shuffle width both key off it);
    ``wave`` is only a byproduct callers use for their own fanout layouts.
    Identical across the mesh, ring and codec-test kernels, so extracted here
    rather than repeated in each.
    """
    thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
    return fx.idx2crd(tid, thread_layout).unpack()


def scale_slot_of(tid):
    """(scale_slot, pair_in_slot) -- this thread's group-16 E4M3 scale slot,
    and which half of the ``PAIR`` sharing that slot this thread is.

    ``scale_slot`` addresses the i32 the byte lives in; ``pair_in_slot``
    selects which of the two packed bytes within it, via
    :func:`_scale_from_word`.
    """
    scale_own_layout = fx.make_layout(
        (BLOCK // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
    )
    scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(tid, scale_own_layout).unpack()
    return scale_slot, pair_in_slot


def hi2_slot_of(tid):
    """(hi2_leader, hi2_slot) for INT6's dense 2-bit plane.

    Threads ``2t`` and ``2t+1`` share one i32 of that plane (see
    :func:`_compact_hi2`); ``hi2_leader`` is whether this thread is the even
    one that stores it, ``hi2_slot`` is which i32. Meaningless for INT4
    (``codec.n_words_per_thread == 1``), so callers that only ever run INT4 need not
    call this at all.
    """
    hi2_leader = (tid & fx.Int32(1)) == fx.Int32(0)
    hi2_slot = tid.shrui(fx.Int32(1))
    return hi2_leader, hi2_slot


def _scale_from_word(codec, word, pair_in_slot):
    """This thread's decoding scale, out of a packed group-16 E4M3 word.

    ``None`` for a codec with no scale plane (fp16 passthrough).
    """
    if not codec.has_scale:
        return None
    e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
    return _e4m3_decoding_scale(codec, e)


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _minnumf(a, b):
    return fx.Vector(fx.arith.minnumf(a, b), a.shape, a.dtype)


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    There is no FlyDSL wrapper; ``s_setreg_imm32_b32 0xdc1, 1`` writes
    ``hwreg(HW_REG_MODE, offset=23, size=2)``.
    """
    llvm.InlineAsmOp(None, [], "s_setreg_imm32_b32 0xdc1, 1", "", has_side_effects=True)


def _shuffle_f16x2(vec, xor_off):
    return _f16x2(fx.Int32(gpu.shuffle_xor(_i32(vec), xor_off, WAVE)))


def _pair_signed_ext_f16(atom):
    """Signed extremum of 16 fp16 (this thread's 8 + xor-1 neighbor)."""
    p0, p1, p2, p3 = (
        _f16x2(atom[0]),
        _f16x2(atom[1]),
        _f16x2(atom[2]),
        _f16x2(atom[3]),
    )
    wmax = fx.maxnumf(fx.maxnumf(p0, p1), fx.maxnumf(p2, p3))
    wmin = _minnumf(_minnumf(p0, p1), _minnumf(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = _minnumf(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _f32_to_e4m3(x):
    """Signed E4M3 of a f32: 1 sign + 4 exp (bias 7, e=0 still implicit 1) + 3 mant.

    Group-16 wire scale. Not IEEE OCP E4M3 denorms: e=0 still encodes
    ``(1+m/8)*2^-7`` so typical INT4 extrema (~0.1) stay in range after
    ×−1/8. Byte 0 is +0.
    """
    is_z = x == fx.Float32(0.0)
    sign = (x < fx.Float32(0.0)).select(fx.Int32(0x80), fx.Int32(0))
    bits = abs(x).bitcast(fx.Int32)
    e = (bits.shrui(fx.Int32(23)) & fx.Int32(255)) - fx.Int32(127)
    mant = bits & fx.Int32(0x7FFFFF)
    m3 = (mant + fx.Int32(1 << 19)).shrui(fx.Int32(20))
    carry = m3 == fx.Int32(8)
    e = e + carry.select(fx.Int32(1), fx.Int32(0))
    m3 = carry.select(fx.Int32(0), m3)
    e4 = e + fx.Int32(7)
    e4 = (e4 < fx.Int32(0)).select(
        fx.Int32(0), (e4 > fx.Int32(15)).select(fx.Int32(15), e4)
    )
    byte = sign | (e4 << fx.Int32(3)) | (m3 & fx.Int32(7))
    return is_z.select(fx.Int32(0), byte)


def _e4m3_to_f32(b):
    is_z = b == fx.Int32(0)
    sign = (b & fx.Int32(0x80)) != fx.Int32(0)
    e4 = b.shrui(fx.Int32(3)) & fx.Int32(15)
    m3 = b & fx.Int32(7)
    mag_bits = ((e4 + fx.Int32(120)) << fx.Int32(23)) | (m3 << fx.Int32(20))
    mag = mag_bits.bitcast(fx.Float32)
    signed = sign.select(-mag, mag)
    return is_z.select(fx.Float32(0.0), signed)


def _pack_e4m3_word(e, lane):
    """Four pair-E4M3 bytes into the i32 scale slot (lanes 0,2,4,6 of GROUP)."""
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


def _e4m3_decoding_scale(codec, e):
    return _splat_f16x2(_e4m3_to_f32(e) * fx.Float32(codec.dec_step))


def _clamp_f32(x, lo, hi):
    x = (x < fx.Float32(lo)).select(fx.Float32(lo), x)
    return (x > fx.Float32(hi)).select(fx.Float32(hi), x)


def _quant_atom_fp16(codec, atom, enc_pk):
    """Quantize 8 fp16 into this codec's planes, as packed i32 words."""
    q = []
    lo = _splat_f16x2(fx.Float16(codec.qmin))
    hi = _splat_f16x2(fx.Float16(codec.qmax))
    bias = fx.Vector.filled(2, fx.Int16(codec.bias), fx.Int16)
    for i in range_constexpr(4):
        w = _minnumf(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(fx.roundeven(w).to(fx.Int16) + bias))
    if codec.n_words_per_thread == 1:
        return (
            q[0]
            | (q[1] << fx.Int32(4))
            | (q[2] << fx.Int32(8))
            | (q[3] << fx.Int32(12)),
        )
    # Every code is masked here. In INT4 each field already fills its whole
    # nibble, so the shift-or cannot collide; a 6-bit code would overrun its
    # neighbour's slot if left whole.
    m4 = fx.Int32(_K_MASK_000F)
    m2 = fx.Int32(_K_MASK_0003)
    lo4 = [qi & m4 for qi in q]
    hi2 = [qi.shrui(fx.Int32(4)) & m2 for qi in q]
    packed_lo = (
        lo4[0]
        | (lo4[1] << fx.Int32(4))
        | (lo4[2] << fx.Int32(8))
        | (lo4[3] << fx.Int32(12))
    )
    packed_hi = (
        hi2[0]
        | (hi2[1] << fx.Int32(2))
        | (hi2[2] << fx.Int32(4))
        | (hi2[3] << fx.Int32(6))
    )
    return (packed_lo, packed_hi)


def _compact_hi2(packed_hi, lane):
    """Two threads' 2-bit planes into the single i32 they share on the wire.

    ``packed_hi`` carries its 8 live bits at [0..7] and [16..23] -- the f16x2
    pairing puts a thread's even elements in the low half of every i32 and its
    odd ones in the high half. Squeeze those to 16 dense bits, then merge with
    the xor-1 neighbour. Both lanes of the pair compute the same word; only the
    even one stores it, at ``hi2_i32_off + (tid >> 1)``.

    ``lane & 1 == tid & 1``, so this is the same pairing
    :func:`_pair_signed_ext_f16` already uses for the group-16 extremum -- no
    second convention is introduced.
    """
    c = (packed_hi & fx.Int32(0xFF)) | (packed_hi.shrui(fx.Int32(8)) & fx.Int32(0xFF00))
    other = fx.Int32(gpu.shuffle_xor(c, 1, WAVE))
    is_even = (lane & fx.Int32(1)) == fx.Int32(0)
    return is_even.select(c | (other << fx.Int32(16)), other | (c << fx.Int32(16)))


def _expand_hi2(word, tid):
    """Inverse of :func:`_compact_hi2`, for the calling thread's half."""
    c = word.shrui((tid & fx.Int32(1)) * fx.Int32(16)) & fx.Int32(0xFFFF)
    return (c & fx.Int32(0xFF)) | ((c & fx.Int32(0xFF00)) << fx.Int32(8))


def _codec_quant(codec, atom, lane, tid):
    """Quantize one atom. Returns ``(words, e4m3_word, is_leader)``.

    ``words`` is this codec's payload i32s in wire order, and is opaque to the
    caller: the ring restages received words verbatim on its all-gather lap and
    must not have to know how many there are.

    A codec with no scale plane (fp16 passthrough) skips quantization
    entirely.
    """
    if not codec.has_scale:
        return tuple(atom[i] for i in range_constexpr(4)), None, False
    ext = _pair_signed_ext_f16(atom)
    e = _f32_to_e4m3(ext)
    d = _e4m3_to_f32(e) * fx.Float32(codec.dec_step)
    # Clamp before the fp16 splat. An all-zero group gives d == 0, so the
    # reciprocal is 1e7 -- Inf once narrowed to fp16, and 0 * Inf is NaN. Only
    # MODE.FP16_OVFL saturation has been keeping that from poisoning a tile,
    # and INT6 cuts the headroom fourfold: |d| is four times smaller for a
    # given extremum, so 1/|d| peaks near 4096 rather than 1024.
    enc = _clamp_f32(
        fx.Float32(1.0) / (d + fx.Float32(1e-7)), -_FP16_MAX, _FP16_MAX
    )
    words = _quant_atom_fp16(codec, atom, _splat_f16x2(enc))
    if codec.n_words_per_thread == 2:
        words = (words[0], _compact_hi2(words[1], lane))
    is_leader = (tid % GROUP) == 0
    return words, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant(codec, words, scale, tid, acc=None):
    """Unpack four codes to f16x2, scale, optionally FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.

    *words* are as they sit on the wire, so the 2-bit plane is still compacted
    and is expanded here -- once, outside the loop.

    A codec with no scale plane (fp16 passthrough) has nothing to unpack:
    *words* are already the atom's four dwords, so this reduces to a packed
    fp16 add into *acc* (or a passthrough when *acc* is ``None``).
    """
    if not codec.has_scale:
        if acc is None:
            return fx.Vector.from_elements(list(words[:4]), fx.Int32)
        out = [_i32(_f16x2(words[i]) + _f16x2(acc[i])) for i in range_constexpr(4)]
        return fx.Vector.from_elements(out, fx.Int32)
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = _f16x2(fx.Int32(codec.dequant_bias))
    packed = words[0]
    if codec.n_words_per_thread == 2:
        hi = _expand_hi2(words[1], tid)
        m2 = fx.Int32(_K_MASK_0003)
    for i in range_constexpr(4):
        code = packed.shrui(fx.Int32(i * 4)) & mask
        if codec.n_words_per_thread == 2:
            code = code | ((hi.shrui(fx.Int32(i * 2)) & m2) << fx.Int32(4))
        dq = _f16x2(code | bias_hi) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_load(codec, get, tid, scale_slot):
    """Read one packet through ``get(i32_off_in_tile) -> i32``.

    Returns ``(words, e4m3_word)`` exactly as they sit on the wire -- the 2-bit
    plane stays compacted -- so a forwarding path can restage them byte for
    byte without decoding. ``e4m3_word`` is ``None`` for a codec with no scale
    plane (fp16 passthrough).
    """
    words = tuple(get(off) for off, _pred in codec.plane_slots(tid))
    scale_word = (
        get(fx.Int32(codec.scale_i32_off) + scale_slot) if codec.has_scale else None
    )
    return words, scale_word
