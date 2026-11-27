# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""INT4, INT5, INT6 and FP16 wire codecs for quick-allreduce.

A rank-tile is ``block`` threads x one 16 B atom, quantized in packed fp16 with
a signed group-16 scale (a pair-shared-exponent float, see :func:`_pse_scale`).
INT4 is one nibble plane;
INT5/INT6 add a dense 1-bit or 2-bit plane on the same 64 B sector grid, which
keeps the nibble plane byte-identical to INT4's. FP16 is a passthrough wire
format -- the
thread's eight fp16 values verbatim, no quantization -- used to test the
reduce-scatter/all-gather transport in isolation from the codec.

Limitations
--------------------

The kernels convert the bf16 payload to fp16 on load and back on store, for
every codec, and all in-kernel arithmetic is packed fp16. Two limits follow.

* fp16 holds at most 65504. A larger magnitude saturates to 65504 (see
  :func:`_clamp_fp16_overflow`), and a magnitude below fp16's resolution loses
  mantissa bits that bf16 has. The FP16 wire format is therefore not lossless
  for bf16 data outside the fp16 range.
* The group scale spans ``[2**-12, 1.875 * 2**19]``, so every fp16 group
  extremum fits and nothing clips below the fp16 limit above. INT4 rounds its
  3-bit mantissa to nearest, within 1/16 of the extremum; INT5 and INT6 round
  it up. Adjacent groups
  share an exponent, within ``[-4, +3]`` of each other; past that the smaller
  group's scale is raised and loses mantissa bits, though never clips. INT5
  floors the scale at ``2**-11`` and INT6 at ``2**-10`` (see
  :attr:`Codec.scale_min_e5`). NaN and Inf are not propagated: they become
  finite values.

``block`` is a build parameter: it sets the tile width, and through it how
many blocks a payload gets.

Imported by the mesh and ring kernels, which must agree on it byte for byte.
Depends on ``transport`` for ``BLOCK``, ``WAVE`` and ``I32_BYTES``.
"""

import functools
import math
from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr

from ...allreduce_policy import (
    BLOCK_ALIGN,
    MAX_BLOCK,
    SUPPORTED_BLOCKS,
)
from ..kernels_common import ceildiv
from .transport import (
    BLOCK,
    I32_BYTES,
    WAVE,
)

# Two threads (PAIR) share one group-16 scale; GROUP threads share the i32 slot.
GROUP = 8
PAIR = 2

# Every workgroup width in ``SUPPORTED_BLOCKS`` can build a codec.
#
# Every region of a rank-tile has to be a whole number of 64 B fabric sectors:
# the fanout moves one sector per quad, so a region ending mid-sector would
# ship the next region's bytes with it and overwrite them in the peer's inbox.
# The nibble, 1-bit and 2-bit planes are 4*block, block and 2*block bytes,
# whole sectors at every width here. The scale region is block/2 bytes, a whole sector only from
# block 128 up; at 64 it is padded, see :func:`codecs_for_block`.
# i32 per 64 B fabric sector.
SECTOR_I32 = 16


def validate_block(block: int) -> int:
    block = int(block)
    # Besides ``SUPPORTED_BLOCKS``, the fused schedules size a workgroup to one
    # token row (block = hidden / 8), on any BLOCK_ALIGN multiple up to
    # MAX_BLOCK. Both keep every rank-tile region on the 64 B sector grid.
    if block in SUPPORTED_BLOCKS or (
        BLOCK_ALIGN <= block <= MAX_BLOCK and block % BLOCK_ALIGN == 0
    ):
        return block
    raise ValueError(
        f"codec block must be one of {SUPPORTED_BLOCKS} or a multiple of "
        f"{BLOCK_ALIGN} in [{BLOCK_ALIGN}, {MAX_BLOCK}], got {block!r}"
    )


def _round_up_to_sector(n_i32: int) -> int:
    return ceildiv(n_i32, SECTOR_I32) * SECTOR_I32


# Dequant bit-trick: code | 0x6400 then + (-(1024+bias)) as f16x2 reconstructs
# (q - bias). fp16 with exponent field 1024.0 holds the integer in its low
# mantissa bits, so this works for any field that fits below bit 10 -- 4 bits
# with bias 8, 5 bits with bias 16, 6 bits with bias 32.
_K_MASK_000F = 0x000F000F
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2 = -(1024 + 8)
_K_HALF2_1040 = 0xE410E410  # -1040.0 fp16x2 = -(1024 + 16)
_K_HALF2_1056 = 0xE420E420  # -1056.0 fp16x2 = -(1024 + 32)

# Largest finite fp16. The encode scale is materialised as fp16, so anything
# above this becomes Inf there; see Codec.scale_min_e5.
_FP16_MAX = 65504.0
# Exponent bias of the group scale.
_SCALE_BIAS = 12
# Names the group-scale codec in every mesh and ring launch name. FlyDSL 0.5.0
# keys its cache on the traced IR, so a codec change rebuilds the kernels
# without a new tag; the tag only labels the binaries.
SCALE_CODEC_TAG = "pse3b12"


@dataclass(frozen=True)
class Codec:
    """One wire format for a rank-tile: how it packs, and the geometry that implies.

    ``bias`` is both the zero point of the unsigned code and the magnitude of
    the most negative one, because the codec maps a group's signed extremum
    onto ``-bias`` -- that is what uses the asymmetric range fully. The
    decoding factor is therefore ``-1/bias``.

    INT5/INT6 are INT4's nibble plane plus a dense extra-bit plane, rather
    than a 5- or 6-bit field per thread. A packed odd-width field straddles
    i32 boundaries and leaves the regions off the 64 B fabric sector grid;
    two planes keep every region sector-aligned (INT5: 16 + 4 + 2 = 22;
    INT6: 16 + 8 + 2 = 26) and leave the nibble plane byte-identical to
    INT4's, so fanout needs nothing but a different sector count.
    """

    name: str
    bits: int
    bias: int | None
    #: fp16x2 constant added after the ``| 0x6400`` trick: -(1024 + bias).
    dequant_bias: int | None
    #: i32 offset of the dense extra-bit plane (INT5 1-bit, INT6 2-bit);
    #: None when the codec has only a nibble plane.
    hi2_i32_off: int | None
    scale_i32_off: int | None
    rank_tile_i32: int
    #: Payload i32 a thread contributes per atom.
    n_words_per_thread: int
    #: Workgroup width this instance was built for.
    block: int = BLOCK

    @property
    def has_scale(self) -> bool:
        return self.scale_i32_off is not None

    @property
    def dec_step(self) -> float:
        return -1.0 / self.bias

    @property
    def scale_min_e5(self) -> int:
        """Smallest scale exponent field for this codec.

        The encode factor ``1 / (scale * |dec_step|)`` is splat to fp16, so the
        scale must stay at or above ``1 / (|dec_step| * 65504)``: field 0
        (``2**-12``) for INT4, field 1 (``2**-11``) for INT5, field 2
        (``2**-10``) for INT6. A group below the floor is quantized against the
        floor scale.
        """
        if not self.has_scale:
            return 0
        floor = 1.0 / (abs(self.dec_step) * _FP16_MAX)
        return max(0, math.ceil(math.log2(floor)) + _SCALE_BIAS)

    @property
    def scale_rounds_up(self) -> bool:
        """Round the group scale up to the next representable value, not to
        nearest.

        Rounding to nearest can round the scale below the extremum, which then
        clips by up to 1/16. That is half an INT4 step, but one INT5 and two
        INT6 steps, so INT5 and INT6 round up: no clip, at a step up to 1/8
        coarser. On a normal fill (fp64 model) INT6 gains 0.7 dB rounding up
        (33.9 vs 33.3 dB), INT5 is within 0.1 dB either way, and INT4 keeps
        rounding to nearest (22.2 dB, vs 21.8 dB rounding up).
        """
        return self.bits > 4

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

    @property
    def hi_bits(self) -> int | None:
        """Width of the extra plane, or ``None`` when there is none."""
        if self.hi2_i32_off is None:
            return None
        return self.bits - 4

    @property
    def hi_share(self) -> int:
        """Threads that share one extra-plane i32 (8 values × ``hi_bits`` each).

        INT6 is 16 bits/thread → 2 threads; INT5 is 8 bits/thread → 4.
        """
        return 32 // (8 * self.hi_bits)

    def plane_slots(self, tid):
        """``[(i32 offset within the rank-tile, store predicate)]``, one per
        payload word, in the order :func:`_codec_quant` returns them.

        The offset is where this thread's word lives; the predicate says
        whether this thread is the one that stores it -- a Python ``True``
        when every thread owns its word (INT4's nibble plane, fp16's four
        dense planes), or the extra-plane leader when a lane pair (INT6) or
        quartet (INT5) shares one slot. A load ignores the predicate: every
        thread reads a plane's word regardless of who wrote it.
        """
        if self.n_words_per_thread == 1:
            return [(tid, True)]
        if self.hi2_i32_off is not None:
            hi_leader, hi_slot = hi_slot_of(tid, self.hi_share)
            return [(tid, True), (fx.Int32(self.hi2_i32_off) + hi_slot, hi_leader)]
        # Dense multi-word codec (fp16): every thread owns every word, at a
        # fixed stride of one rank-tile row (``block`` i32) per word.
        return [
            (fx.Int32(w * self.block) + tid, True)
            for w in range_constexpr(self.n_words_per_thread)
        ]


@functools.cache
def codecs_for_block(block: int = BLOCK) -> dict[str, "Codec"]:
    """The wire formats at a given workgroup width."""

    block = validate_block(block)
    nibble = block  # one i32 of nibbles per thread
    hi2 = block // PAIR  # INT6's dense 2-bit plane, one i32 per lane pair
    # One i32 per GROUP threads (four group scales), padded to a sector.
    scale = _round_up_to_sector(block // GROUP)
    return {
        c.name: c
        for c in (
            # block*4 B nibbles then block/2 B scale: a 4.5*block B rank-tile,
            # 1152 B and 18 sectors at block=256.
            Codec(
                name="int4",
                bits=4,
                bias=8,
                dequant_bias=_K_HALF2_1032,
                hi2_i32_off=None,
                scale_i32_off=nibble,
                rank_tile_i32=nibble + scale,
                n_words_per_thread=1,
                block=block,
            ),
            # Plus a block B 1-bit plane: 5.5*block B before scale padding,
            # 1408 B and 22 sectors at block=256.
            Codec(
                name="int5",
                bits=5,
                bias=16,
                dequant_bias=_K_HALF2_1040,
                hi2_i32_off=nibble,
                scale_i32_off=nibble + (block // 4),
                rank_tile_i32=nibble + (block // 4) + scale,
                n_words_per_thread=2,
                block=block,
            ),
            # Plus a block*2 B 2-bit plane between them: 6.5*block B, 1664 B
            # and 26 sectors at block=256.
            Codec(
                name="int6",
                bits=6,
                bias=32,
                dequant_bias=_K_HALF2_1056,
                hi2_i32_off=nibble,
                scale_i32_off=nibble + hi2,
                rank_tile_i32=nibble + hi2 + scale,
                n_words_per_thread=2,
                block=block,
            ),
            # Four dense fp16x2 planes, no scale region: 16*block B, 4096 B and
            # 64 sectors at block=256. Passthrough wire format.
            Codec(
                name="fp16",
                bits=16,
                bias=None,
                dequant_bias=None,
                hi2_i32_off=None,
                scale_i32_off=None,
                rank_tile_i32=block * 4,
                n_words_per_thread=4,
                block=block,
            ),
        )
    }


CODECS = codecs_for_block(BLOCK)
INT4, INT5, INT6, FP16 = (
    CODECS["int4"],
    CODECS["int5"],
    CODECS["int6"],
    CODECS["fp16"],
)


def thread_lane(tid, block: int = BLOCK):
    """(wave, lane) for *tid* in a ``(block // WAVE) x WAVE`` block.

    ``lane`` is the codec's own required argument to :func:`_codec_quant` /
    :func:`_codec_dequant` (the pairing and shuffle width both key off it);
    ``wave`` is only a byproduct callers use for their own fanout layouts.
    Shared by the mesh, ring and codec-test kernels.

    Every supported *block* is a whole number of waves, so there is no partial
    wave for the shuffles to fall off.
    """
    if block == WAVE:
        return fx.Int32(0), tid
    thread_layout = fx.make_layout((block // WAVE, WAVE), (WAVE, 1))
    return fx.idx2crd(tid, thread_layout).unpack()


def scale_slot_of(tid, block: int = BLOCK):
    """(scale_slot, pair_in_slot) -- this thread's group-16 scale slot, and
    which of the slot's four groups (each a ``PAIR`` of threads) is its own.

    ``scale_slot`` addresses the i32 that holds the slot's scales;
    ``pair_in_slot`` selects this group's scale within it, via
    :func:`_scale_from_word`.
    """
    scale_own_layout = fx.make_layout(
        (block // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
    )
    scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(tid, scale_own_layout).unpack()
    return scale_slot, pair_in_slot


def hi_slot_of(tid, share):
    """(hi_leader, hi_slot) for the extra-bit plane.

    ``share`` threads compact into one i32 (2 for INT6, 4 for INT5; see
    :func:`_compact_hi`). ``hi_leader`` is the lowest-tid thread of that
    group, the one that stores the word; ``hi_slot`` is which i32.
    Meaningless for INT4, so callers that only ever run INT4 need not call
    this at all.
    """
    hi_leader = (tid & fx.Int32(share - 1)) == fx.Int32(0)
    hi_slot = tid.shrui(fx.Int32(share.bit_length() - 1))
    return hi_leader, hi_slot


def _scale_from_word(codec, word, pair_in_slot):
    """This thread's decoding scale, out of a packed group-16 scale word.

    ``None`` for a codec with no scale plane (fp16 passthrough).
    """
    if not codec.has_scale:
        return None
    return _splat_f16x2(_pse_decoding_step(codec, word, pair_in_slot))


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    FlyDSL has no MODE helper; ``llvm.amdgcn.s.setreg`` is the same
    intrinsic ``rocdl.disable_xdl_arb_stall`` uses for a different bit.
    """
    # hwreg(HW_REG_MODE=1, offset=23, size=2): id | (offset << 6) | ((size - 1) << 11)
    imm = fx.Int32(1 | (23 << 6) | ((2 - 1) << 11)).ir_value()
    val = fx.Int32(1).ir_value()
    llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


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
    wmin = fx.min(fx.min(p0, p1), fx.min(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = fx.min(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    """Convert a bf16 atom to fp16. Magnitudes above 65504 saturate, and values
    that fp16 cannot represent as finely as bf16 lose mantissa bits."""
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _atom_bf16_to_f16_scaled(atom, scale):
    """:func:`_atom_bf16_to_f16`, multiplied by the f32 *scale* (an 8-wide splat)
    before the fp16 narrowing, so a bf16 above 65504 survives a divisor."""
    value = fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float32) * scale
    return value.to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16_scaled(atom, scale):
    """:func:`_atom_f16_to_bf16`, multiplied by the f32 *scale* (an 8-wide splat)
    after the fp16 widening, so the restored sum can exceed 65504."""
    value = fx.Vector(atom).bitcast(fx.Float16).to(fx.Float32) * scale
    return value.to(fx.BFloat16).bitcast(fx.Int32)


# The group scale is a pair-shared-exponent (PSE) float. Each group-16 scale is
# ``r = exponent * 8 + mantissa``: a 3-bit mantissa and a 5-bit exponent with
# bias 12, so ``(r + _PSE_R_TO_F32) << 20`` is its f32 bit pattern. A scale
# i32 holds the four groups of its slot (``pair_in_slot``) as two 16-bit
# halves, groups 0 and 1 in the low half and 2 and 3 in the high one. In a
# half, group A is the even one and B the odd one:
#
#   bits  0-7   r_A
#   bits  8-13  r_B - (r_A & ~7) + 32: B's exponent relative to A's, plus 4,
#               then B's mantissa
#   bit  14/15  sign of A / sign of B
#
# B's exponent has to lie in [A - 4, A + 3]. When it does not, the smaller of
# the two scales is raised to the nearest one that fits, which costs that group
# mantissa bits but never clips it.
_PSE_R_TO_F32 = (127 - _SCALE_BIAS) * 8
# r of the largest scale the 5-bit exponent holds, 1.875 * 2**19.
_PSE_R_MAX = 255


def _pse_r_floor(codec):
    return codec.scale_min_e5 * 8


def _pse_scale(codec, ext, lane):
    """``(scale_word, d)`` for this thread's group.

    *ext* is the group's signed extremum (f32 of an fp16). ``scale_word`` is
    the packed PSE i32 of the thread's whole ``GROUP`` -- every lane of it
    computes the same word, and only the leader stores it -- and ``d`` is this
    group's decoding step, ``scale * dec_step``, as f32.

    The scale is rounded on the f32 bits directly: adding half an ulp of the
    3-bit mantissa (or a whole one less one, to round up) and dropping the low
    20 bits leaves ``r`` with any mantissa carry already in the exponent. The
    exponent floor also catches an all-zero group, whose ``r`` is negative.
    """
    rnd = (1 << 20) - 1 if codec.scale_rounds_up else 1 << 19
    bits = ext.bitcast(fx.Int32)
    mag = bits & fx.Int32(0x7FFFFFFF)
    r = (mag + fx.Int32(rnd)).shrui(fx.Int32(20)) - fx.Int32(_PSE_R_TO_F32)
    # The ceiling only binds on NaN or Inf: the largest fp16 rounds to r 224.
    r = fx.min(fx.max(r, fx.Int32(_pse_r_floor(codec))), fx.Int32(_PSE_R_MAX))

    # The partner group's lanes are xor-PAIR away, and the other half of the
    # slot's word xor-2*PAIR away.
    b = lane.shrui(fx.Int32(1)) & fx.Int32(1)
    is_b = b != fx.Int32(0)
    sign_bit = bits.shrui(fx.Int32(31)) << (fx.Int32(14) + b)
    own = r | sign_bit
    other = own.shuffle_xor(PAIR, WAVE)
    r_other = other & fx.Int32(0xFF)
    r_a = is_b.select(r_other, r)
    r_b = is_b.select(r, r_other)
    r_a = fx.max(r_a, (r_b & fx.Int32(~7)) - fx.Int32(3 * 8))
    base = (r_a & fx.Int32(~7)) - fx.Int32(4 * 8)
    r_b = fx.max(r_b, base)
    half = r_a | ((r_b - base) << fx.Int32(8)) | ((own | other) & fx.Int32(0xC000))
    other_half = half.shuffle_xor(2 * PAIR, WAVE)
    is_lo = (lane & fx.Int32(2 * PAIR)) == fx.Int32(0)
    word = is_lo.select(
        half | (other_half << fx.Int32(16)), other_half | (half << fx.Int32(16))
    )
    return word, _pse_step(codec, is_b.select(r_b, r_a), bits)


def _pse_step(codec, r, sign_src):
    """``scale * dec_step`` for scale *r*, with the sign of *sign_src*'s top bit.

    ``dec_step`` is ``-2**-log2(bias)``, so the product is exact and is built
    as bits: the exponent drops by ``log2(bias)`` and the sign flips.
    """
    shift = int(math.log2(codec.bias)) * 8
    mag = (r + fx.Int32(_PSE_R_TO_F32 - shift)) << fx.Int32(20)
    sign = (sign_src & fx.Int32(-0x80000000)) ^ fx.Int32(-0x80000000)
    return (mag | sign).bitcast(fx.Float32)


def _pse_decoding_step(codec, word, pair_in_slot):
    """This thread's ``scale * dec_step`` (f32) out of its slot's PSE word."""
    b = pair_in_slot & fx.Int32(1)
    half = word.shrui((pair_in_slot & fx.Int32(2)) << fx.Int32(3))
    r_a = half & fx.Int32(0xFF)
    r_b = (
        (half & fx.Int32(0xF8))
        + (half.shrui(fx.Int32(8)) & fx.Int32(0x3F))
        - fx.Int32(4 * 8)
    )
    r = (b != fx.Int32(0)).select(r_b, r_a)
    sign = half.shrui(fx.Int32(14) + b) << fx.Int32(31)
    return _pse_step(codec, r, sign)


def _pack_fields(fields, width, mask=None):
    """``fields[i] << (i * width)`` OR-ed together, each masked first if *mask*."""
    out = fields[0] if mask is None else fields[0] & mask
    for i in range_constexpr(1, len(fields)):
        f = fields[i] if mask is None else fields[i] & mask
        out = out | (f << fx.Int32(i * width))
    return out


def _quant_atom_fp16(codec, atom, enc_pk):
    """Quantize 8 fp16 into this codec's planes, as packed i32 words.

    Rounds with the fp16 magic number ``w + (1024 + bias)``: the sum lies in
    ``[1024, 2048)``, where the fp16 ulp is 1, and ``1024 + bias`` is even, so
    the add rounds half to even like ``roundeven`` and leaves
    ``roundeven(w) + bias`` in the low mantissa bits.
    """
    q = []
    lo = _splat_f16x2(fx.Float16(codec.qmin))
    hi = _splat_f16x2(fx.Float16(codec.qmax))
    magic = _splat_f16x2(fx.Float16(1024.0 + codec.bias))
    code_mask = (1 << codec.bits) - 1
    codes = fx.Int32(code_mask | (code_mask << 16))
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(w + magic) & codes)
    if codec.n_words_per_thread == 1:
        return (_pack_fields(q, 4),)
    # Every code is masked here. In INT4 each field already fills its whole
    # nibble, so the shift-or cannot collide; a 5- or 6-bit code would overrun
    # its neighbour's slot if left whole.
    hi_bits = codec.hi_bits
    hi_mask = (1 << hi_bits) - 1
    m4 = fx.Int32(_K_MASK_000F)
    m_hi = fx.Int32(hi_mask | (hi_mask << 16))
    lo4 = [qi & m4 for qi in q]
    hi = [qi.shrui(fx.Int32(4)) & m_hi for qi in q]
    return (_pack_fields(lo4, 4), _pack_fields(hi, hi_bits))


def _compact_hi(packed_hi, lane, hi_bits):
    """``share`` threads' extra-bit planes into the single i32 they share.

    ``packed_hi`` carries its live bits in each f16 half -- the f16x2 pairing
    puts a thread's even elements in the low half of every i32 and its odd
    ones in the high half. Squeeze those to ``8 * hi_bits`` dense bits, then
    butterfly-merge with xor-1 (and xor-2 when four threads share, INT5).
    Every lane of the group computes the same word; only the leader stores
    it, at ``hi2_i32_off + (tid >> log2(share))``.

    The xor-1 step is the same pairing :func:`_pair_signed_ext_f16` already
    uses for the group-16 extremum -- no second convention is introduced.
    """
    n_lo = 4 * hi_bits
    mask_lo = (1 << n_lo) - 1
    c = (packed_hi & fx.Int32(mask_lo)) | (
        packed_hi.shrui(fx.Int32(16 - n_lo)) & fx.Int32(mask_lo << n_lo)
    )
    word = c
    thread_bits = 8 * hi_bits
    share = 4 // hi_bits
    shift = thread_bits
    xor_off = 1
    while xor_off < share:
        other = word.shuffle_xor(xor_off, WAVE)
        is_lo = (lane & fx.Int32(xor_off)) == fx.Int32(0)
        word = is_lo.select(
            word | (other << fx.Int32(shift)),
            other | (word << fx.Int32(shift)),
        )
        shift *= 2
        xor_off *= 2
    return word


def _expand_hi(word, tid, hi_bits):
    """Inverse of :func:`_compact_hi`, for the calling thread's slice."""
    thread_bits = 8 * hi_bits
    share = 4 // hi_bits
    n_lo = 4 * hi_bits
    mask_lo = (1 << n_lo) - 1
    c = word.shrui((tid & fx.Int32(share - 1)) * fx.Int32(thread_bits)) & fx.Int32(
        (1 << thread_bits) - 1
    )
    return (c & fx.Int32(mask_lo)) | (
        (c & fx.Int32(mask_lo << n_lo)) << fx.Int32(16 - n_lo)
    )


def _codec_quant(codec, atom, lane, tid):
    """Quantize one atom. Returns ``(words, scale_word, is_leader)``.

    ``words`` is this codec's payload i32s in wire order, and is opaque to the
    caller: the ring restages received words verbatim on its all-gather lap and
    must not have to know how many there are.

    A codec with no scale plane (fp16 passthrough) skips quantization
    entirely.
    """
    if not codec.has_scale:
        return tuple(atom[i] for i in range_constexpr(4)), None, False
    ext = _pair_signed_ext_f16(atom)
    scale_word, d = _pse_scale(codec, ext, lane)
    # The scale floor keeps d away from zero: 1/|d| peaks at 2**15 for every
    # codec (see Codec.scale_min_e5), so the fp16 splat cannot overflow and an
    # all-zero group needs no special case.
    enc = fx.Float32(1.0) / d
    words = _quant_atom_fp16(codec, atom, _splat_f16x2(enc))
    if codec.hi2_i32_off is not None:
        words = (words[0], _compact_hi(words[1], lane, codec.hi_bits))
    is_leader = (tid % GROUP) == 0
    return words, scale_word, is_leader


def _codec_dequant(codec, words, scale, tid, acc=None):
    """Unpack four codes to f16x2, scale, optionally FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.

    *words* are as they sit on the wire, so the extra-bit plane is still
    compacted and is expanded here -- once, outside the loop.

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
    if codec.hi2_i32_off is not None:
        hi_bits = codec.hi_bits
        hi = _expand_hi(words[1], tid, hi_bits)
        hi_mask = (1 << hi_bits) - 1
        m_hi = fx.Int32(hi_mask | (hi_mask << 16))
    for i in range_constexpr(4):
        code = packed.shrui(fx.Int32(i * 4)) & mask
        if codec.hi2_i32_off is not None:
            code = code | ((hi.shrui(fx.Int32(i * hi_bits)) & m_hi) << fx.Int32(4))
        dq = _f16x2(code | bias_hi) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_load(codec, get, tid, scale_slot):
    """Read one packet through ``get(i32_off_in_tile) -> i32``.

    Returns ``(words, scale_word)`` exactly as they sit on the wire -- the
    extra-bit plane stays compacted -- so a forwarding path can restage them
    byte for byte without decoding. ``scale_word`` is ``None`` for a codec with
    no scale plane (fp16 passthrough).
    """
    words = tuple(get(off) for off, _pred in codec.plane_slots(tid))
    scale_word = (
        get(fx.Int32(codec.scale_i32_off) + scale_slot) if codec.has_scale else None
    )
    return words, scale_word
