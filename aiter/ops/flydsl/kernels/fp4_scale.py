# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 block-scale formats and direct WMMA emission."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_ops
from flydsl.expr import arith, rocdl
from flydsl.expr.arith import ArithValue, _to_raw
from flydsl.expr.typing import T

from aiter.utility.mx_types import MxDtypeInt

from .quant_utils import emit_mx_e8m0_scale

_SCALE_FORMAT_NAMES = ("e8", "e5m3", "e4m3")


def emit_fp4_block_scale(amax, scale_format=0, global_scale=1.0):
    """Return the decoded block scale and its unsigned ISA byte (RNE)."""
    if scale_format == 0:
        byte = emit_mx_e8m0_scale(
            ArithValue(_to_raw(amax)) * (1.0 / global_scale), dtype=MxDtypeInt.FP4_E2M1
        )
        scale = (byte << 23).bitcast(T.f32)
        return (byte == 0).select(arith.constant(1.0, type=T.f32), scale), arith.trunci(
            T.i8, byte
        )
    bias = 15 if scale_format == 1 else 7
    max_value = 114688.0 if scale_format == 1 else 448.0
    value = fx.Float32(arith.minnumf(amax * (1.0 / (6.0 * global_scale)), max_value))
    bits = value.bitcast(fx.Int32)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - ((127 - bias) << 3)
    denorm = fx.Int32(arith.fptoui(T.i32, _to_raw(_rint(value * (2.0 ** (bias + 2))))))
    encoded = (value < 2.0 ** (1 - bias)).select(denorm, normal)
    encoded = (encoded < 0).select(fx.Int32(0), encoded)
    upper = fx.Int32(254 if scale_format == 1 else 126)
    encoded = (encoded > upper).select(upper, encoded)
    exponent, mantissa = encoded >> 3, encoded & 7
    normal_scale = (((exponent + (127 - bias)) << 23) | (mantissa << 20)).bitcast(
        fx.Float32
    )
    denorm_scale = fx.Float32(arith.uitofp(T.f32, _to_raw(mantissa))) * (
        2.0 ** (1 - bias - 3)
    )
    scale = (exponent == 0).select(denorm_scale, normal_scale)
    return (encoded == 0).select(fx.Float32(1.0), scale), arith.trunci(
        T.i8, _to_raw(encoded)
    )


def _rint(value):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, "llvm.rint.f32", [_to_raw(value)], [], [])
    )


def wmma_fp4(a, b, c, sa, sb, *, block_size=32, format_a=0, format_b=0, row_b=0):
    """WMMA matrix A is logical weight B; matrix B is logical activation A.

    fmtScaleA maps to NEG[1:0], fmtScaleB to NEG_HI[1:0]. Row selectors
    are independent SCL_OPSEL fields. SCALE16 consumes i64 scale operands.
    """
    if block_size == 32:
        return rocdl.wmma_scale_f32_32x16x128_f4(
            T.vec(16, T.f32),
            a,
            b,
            c,
            sa,
            sb,
            fmtScaleA=format_a,
            fmtScaleB=format_b,
            scaleBType=row_b,
        )
    return rocdl_ops.wmma_scale16_f32_32x16x128_f4(
        T.vec(16, T.f32),
        a,
        b,
        c,
        sa.ir_value(),
        sb.ir_value(),
        fmtScaleA=ir.Attribute.parse(
            f"#rocdl<wmma_matrix_scale_format {_SCALE_FORMAT_NAMES[format_a]}>"
        ),
        fmtScaleB=ir.Attribute.parse(
            f"#rocdl<wmma_matrix_scale_format {_SCALE_FORMAT_NAMES[format_b]}>"
        ),
        scaleBType=ir.Attribute.parse(f"#rocdl<wmma_matrix_scale row{row_b}>"),
    ).result
