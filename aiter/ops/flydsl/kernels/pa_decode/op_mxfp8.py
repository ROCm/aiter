# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""MXFP8 output of the PA decode epilogues: o_proj's group-32 FP8 input.

OCP E4M3 values in the output's layout plus one E8M0 scale byte per 32
consecutive elements of a token's ``[heads * head_dim]`` row, row-major
``[tokens, heads * head_dim // 32]``: the activation pair a native MXFP8 GEMM
(``gemm_a8w8_mx32``) reads. The values are the BF16/FP16 output the same call
stores, quantized as a separate group-32 quant of it (ATOM's ``quantize_fp8``)
would, bit for bit:

    amax = max(max |x| over the group, 1e-10)
    code = ceil_pow2(amax * fp32(1 / 448))    # biased E8M0 exponent, RoundUp
    q    = e4m3(min(max(x * 2**(127 - code), -448), 448))    # nearest even

``x`` is the value rounded to the output dtype first. A group lies inside one
head row (head_dim % 32 == 0), so the lanes of one wave hold all of it and the
group maximum takes only lane shuffles: no LDS, CTA or cluster exchange.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

from ..quant_utils import emit_mx_e8m0_scale

GROUP = 32
FP8_MAX = 448.0
# The amax floor of the separate quant (ATOM ``AMAX_FLOOR``): an all-zero group
# gets the scale of 1e-10, not E8M0 zero.
AMAX_FLOOR = 1e-10
# E8M0 1.0: the scale byte of a padded row.
E8M0_ONE = 127


def group_code(amax):
    """Biased E8M0 exponent (i32) of a group whose largest magnitude is ``amax``."""
    return fx.Int32(
        emit_mx_e8m0_scale(
            fx.maxnumf(fx.Float32(amax), fx.Float32(AMAX_FLOOR)),
            mode=MxScaleRoundModeInt.RoundUp,
            dtype=MxDtypeInt.FP8_E4M3,
        )
    )


def group_reciprocal(code):
    """1 / 2**(code - 127), exact; 0 past the finite range, as the separate quant."""
    return (fx.max(fx.Int32(254) - code, fx.Int32(0)) << 23).bitcast(fx.Float32)


class MxFp8Groups:
    """Store output-dtype values as MXFP8 at element offsets of a contiguous
    output; ``quantized`` and ``scales`` are the FP8 and E8M0 byte pointers."""

    def __init__(self, quantized, scales):
        self.quantized = fx.recast_iter(fx.Int8, quantized)
        self.scales = fx.recast_iter(fx.Int8, scales)

    @staticmethod
    def _floats(value):
        if const_expr(isinstance(value, fx.Vector)):
            floats = value.to(fx.Float32)
            return [fx.Float32(floats[i]) for i in range_constexpr(floats.numel)]
        return [fx.Float32(value)]

    @staticmethod
    def _abs_max(value):
        if const_expr(isinstance(value, fx.Vector)):
            return fmath.absf(value.to(fx.Float32)).reduce(ReductionOp.MAX)
        return fmath.absf(fx.Float32(value))

    def _store_bytes(self, floats, offset, reciprocal):
        """``floats`` scaled, saturated and packed four to a dword (RNE).

        Raw ``cvt_pk_fp8_f32``: FlyDSL 0.5.0 has no typed f32 -> FP8 conversion
        (``quant_utils.emit_f32_to_fp8`` wraps the same op), and its word-select
        operand fills a dword's upper half without a shift and OR."""
        scaled = []
        for value in floats:
            value = value * reciprocal
            value = fx.maxnumf(value, fx.Float32(-FP8_MAX))
            scaled.append(fx.minnumf(value, fx.Float32(FP8_MAX)).ir_value())
        if const_expr(len(scaled) % 4 == 0):
            words = []
            for base in range_constexpr(0, len(scaled), 4):
                word = fx.Int32(0).ir_value()
                for half in range_constexpr(2):
                    word = fx.rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        scaled[base + 2 * half],
                        scaled[base + 2 * half + 1],
                        word,
                        half,
                    )
                words.append(word)
            target = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                fx.add_offset(self.quantized, offset),
            )
            fx.ptr_store(fx.Vector.from_elements(words, dtype=fx.Int32), target)
        elif const_expr(len(scaled) == 2):
            word = fx.Int32(
                fx.rocdl.cvt_pk_fp8_f32(
                    T.i32, scaled[0], scaled[1], fx.Int32(0).ir_value(), 0
                )
            )
            target = fx.recast_iter(
                fx.PointerType.get(fx.Int16.ir_type, fx.AddressSpace.Global, 2),
                fx.add_offset(self.quantized, offset),
            )
            fx.ptr_store(fx.Vector.from_elements([word.to(fx.Int16)]), target)
        else:
            assert len(scaled) == 1, "MXFP8 stores 1, 2 or 4n values a lane"
            word = fx.Int32(
                fx.rocdl.cvt_pk_fp8_f32(
                    T.i32, scaled[0], scaled[0], fx.Int32(0).ir_value(), 0
                )
            )
            self.quantized[offset] = word.to(fx.Int8)

    @flyc.jit
    def store(self, values, offsets, masks, wave):
        """Quantize this lane's ``values`` (output-dtype vectors or scalars, each
        consecutive elements from its element offset in ``offsets``).

        All of a lane's values belong to one 32-element group, whose other
        values sit on the lanes ``lane ^ m`` for ``m`` in the xor closure of
        ``masks``; each of those lanes calls this. The lane holding the group's
        first element writes its scale byte.
        """
        lane_max = self._abs_max(values[0])
        for value in values[1:]:
            lane_max = fx.maxnumf(lane_max, self._abs_max(value))
        for mask in masks:
            lane_max = fx.maxnumf(
                lane_max, lane_max.shuffle_xor(fx.Int32(mask), fx.Int32(wave))
            )
        code = group_code(lane_max)
        reciprocal = group_reciprocal(code)
        for value, offset in zip(values, offsets):
            self._store_bytes(self._floats(value), offset, reciprocal)
        # Local names keep the guarded store's operands SSA values.
        scales, first, byte = self.scales, offsets[0], code.to(fx.Int8)
        if (first & (GROUP - 1)) == 0:
            scales[first // GROUP] = byte


def lane_masks(lanes):
    """Xor shuffles reducing over ``lanes`` adjacent lanes (a power of two)."""
    assert lanes >= 1 and lanes & (lanes - 1) == 0
    return tuple(1 << i for i in reversed(range(lanes.bit_length() - 1)))
