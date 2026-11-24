# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Number format conversions on registers.

f32 -> fp8 (FNUZ) packing for the epilogues, and the fd2 decode of the mxfp4
weights into fp8 MFMA operands with its per-channel scale factor.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T


def _fp8x4_pack(a, b, c, d):
    """Four f32 -> four fp8 (FNUZ) in one i32: saturate to +-240, round to nearest even, and
    map a rounded-to-zero magnitude to +0 (FNUZ has no -0: 0x80 is NaN).

    min / max are minnumf / maxnumf (one v_min / v_max each), so NaN packs to 240 (0x7F).
    NaN reaches the output through the scale instead: the caller's amax must propagate it
    (fx.max, not maxnumf), then the dequantized values are NaN whatever the byte. The
    NaN-propagating minimumf / maximumf (NaN packs to 0, as aiter's
    ``moe_2stage_a16wmix`` ``_pack_activation``) give the same bits on finite inputs, but
    cost a gemm1 kernel 324 more of its 5919 instructions, and with a +-Inf input the
    zero bytes turn the row into 0 * inf, which a maxnumf amax downstream then loses."""
    v = [
        fx.maxnumf(fx.minnumf(fx.Float32(x), fx.Float32(240.0)), fx.Float32(-240.0))
        for x in (a, b, c, d)
    ]
    # cvt_pk_fp8_f32 is unstable (not in fx.rocdl.__all__); the stable conversion,
    # Vector.to(fx.Float8E4M3FNUZ), does not lower on FlyDSL 0.5.0 (arith.truncf has
    # no LLVM translation; gfx942 compile-only probe).
    lo = rocdl.cvt_pk_fp8_f32(T.i32, v[0], v[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, v[2], v[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)


# ---- fp4 -> fp8 decode of the fd2 weights -------------------------------------------
# W stays E2M1/E8M0. Output channel n is normalized by 2**(max_exp[n]-127-5)
# (``_channel_factor``, applied in the epilogue); per 32-K group the remaining
# 2**(5-delta), delta = max_exp - group_exp, is folded into an 8-entry
# E2M1-magnitude -> FP8-FNUZ byte table (``_fp8_lut_predelta``) so each nibble
# decodes with one v_perm.
#
# W and scales are pre-transformed by ``weights.prepare_weights``: each dword holds
# (k_j, k_{j+4}) in byte j (low, high nibble) and no -0 nibble, so even nibbles decode
# straight to K 0..3 and odd nibbles to K 4..7 (no interleave v_perm), and the sign is
# bit 3 of the nibble moved to bit 7 (no sign-table v_perm). The scale byte already
# holds min(delta, 11) * 8; groups below 2**-11 of the channel max (delta > 11) hold
# only zeros (``prepare_weights`` refuses other weights), so they decode to 0.


def _perm_u32(hi, lo, sel):
    # perm_b32 is unstable (not in fx.rocdl.__all__) and FlyDSL 0.5.0 has no stable byte
    # permute: the fd2 decode looks a byte table up with a runtime selector, one
    # v_perm_b32 per 4 nibbles, which shifts and masks cannot express.
    return fx.Uint32(
        rocdl.perm_b32(
            fx.Uint32(hi).ir_value(),
            fx.Uint32(lo).ir_value(),
            fx.Uint32(sel).ir_value(),
        )
    )


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _fp8_lut_predelta(delta):
    """LUT of a 32-K group from its fd2 scale byte (min(max-code, 11)*8, broadcast
    to all 4 bytes)."""
    delta = fx.Uint32(delta)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    return lo, hi, fx.Uint32(0x80808080)


def _channel_factor(exponent):
    """2**(exponent-127-5) as f32 (subnormal-safe)."""
    normal = ((exponent - 5).to(fx.Uint32) << 23).bitcast(fx.Float32)
    subnormal = (fx.Uint32(1) << (exponent + 17).to(fx.Uint32)).bitcast(fx.Float32)
    return (exponent >= 6).select(normal, subnormal)
