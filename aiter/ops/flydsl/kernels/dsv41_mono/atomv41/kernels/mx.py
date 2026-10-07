# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 / MXFP8 arithmetic shared by the V4.1 stages.

The original path quantizes activations by three different rules, and each stage
must use the one its original kernel does:

- ``atom.mono.device.mx.code_ceil`` (ATOM ``quantize_fp8``, the window row,
  aiter ``rmsnorm_quant`` MXFP8): the smallest power of two >= amax / 448;
- ``code_round`` (aiter Triton ``_mxfp8_quant_op``, the q_lora norm): amax's
  exponent after adding 0x200000 to its bits, minus 8.

Every E8M0 code is a biased exponent (``atom.mono.device.mx.pow2`` its scale);
the E4M3 range and packing are ``atom.mono.device``'s.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

# (bits(amax) + ROUND_BIAS) & EXP_MASK rounds amax to a power of two
ROUND_BIAS = 0x200000
EXP_MASK = -8388608  # 0xFF800000
# log2(448) rounded down: the headroom of the round-to-nearest code
FP8_EXP = 8


def code_round(amax):
    """``_mxfp8_quant_op``'s code: amax rounded to a power of two, minus 2^8,
    clamped to +-127 unbiased."""
    p2 = (amax.bitcast(fx.Int32) + ROUND_BIAS) & fx.Int32(EXP_MASK)
    unbiased = ((p2 >> 23) & 255) - 127 - FP8_EXP
    return fx.max(fx.min(unbiased, 127), -127) + 127


def fp8_value(v):
    """``v`` (in range) rounded to E4M3 and back, as the hardware converts."""
    w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, v, v, fx.Int32(0), False))
    return fx.Float32(rocdl.cvt_f32_fp8(T.f32, w.ir_value(), 0))
