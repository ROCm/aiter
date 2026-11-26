# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T


def rcp_f32(value):
    """Public FlyDSL reciprocal intrinsic, lowering directly to ``v_rcp_f32``."""
    return rocdl.rcp(T.f32, value)


def afn_rcp_f32(value):
    """Approximate reciprocal expressed as afn division.

    This also lowers to ``v_rcp_f32``, but retains the surrounding fast-math
    flags used by schedule-sensitive attention kernels.
    """
    with fx.arith.fastmath("afn"):
        return fx.Float32(1.0) / fx.Float32(value)


def pin_vector(value):
    """Materialize ``value`` here through tied asm.

    FlyDSL 0.5.3 has no optimization-barrier expression. This prevents LLVM
    from sinking schedule-sensitive softmax work across a barrier.
    """
    value = fx.Vector(value)
    words = value.bitcast(fx.Int32) if value.dtype.width < 32 else value
    tied = fx.Vector(
        llvm.inline_asm(
            words.ir_value().type,
            [words.ir_value()],
            "",
            "=v,0",
            has_side_effects=True,
        )
    )
    return tied.bitcast(value.dtype) if value.dtype.width < 32 else tied


def uniform_i32(value):
    """Move lane 0's i32 to an SGPR with FlyDSL's public ROCDL intrinsic."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(value)))


def uniform_i64(value):
    """Move both halves of an i64 to SGPRs."""
    value = fx.Int64(value)
    lo = uniform_i32(fx.Int32(value & fx.Int64(0xFFFFFFFF)))
    hi = uniform_i32(fx.Int32(value >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def permlane16_swap_i32(x, y):
    """Swap 16-lane halves between two i32 values with ``v_permlane16``."""
    swapped = rocdl.permlane16_swap(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        x.ir_value(),
        y.ir_value(),
        False,
        False,
    )
    return fx.Int32(llvm.extractvalue(T.i32, swapped, [0])), fx.Int32(
        llvm.extractvalue(T.i32, swapped, [1])
    )


def opaque_i32(value):
    """Hide a known i32 constant from LLVM without imposing a schedule barrier."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(value).ir_value()],
            "",
            "=v,0",
            has_side_effects=False,
        )
    )


def pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def udiv_pow2(value, divisor: int):
    return value >> fx.Int32(pow2_shift(divisor))


def urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def udiv_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def urem_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return urem_pow2(value, divisor)
    return value % fx.Int32(divisor)
