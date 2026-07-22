# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, rocdl
from flydsl.expr.typing import T


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def exp2_amdgcn_scalar(scalar_value):
    raw = (
        arith.unwrap(scalar_value)
        if hasattr(scalar_value, "ir_value") or hasattr(scalar_value, "type")
        else scalar_value
    )
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def exp2_f32_fast(value):
    from flydsl._mlir.dialects import vector as _vector_dialect

    raw = (
        arith.unwrap(value)
        if hasattr(value, "ir_value") or hasattr(value, "type")
        else value
    )
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        n = ty.shape[0]
        elems = []
        for i in range(n):
            scalar = _vector_dialect.extract(
                raw, static_position=[i], dynamic_position=[]
            )
            elems.append(exp2_amdgcn_scalar(scalar))
        return _vector_dialect.from_elements(ty, elems)
    return exp2_amdgcn_scalar(raw)


def cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


# Alias: several kernels historically spelled this ``ceildiv``.
ceildiv = cdiv


def align_up(value: int, align: int) -> int:
    """Round *value* up to the next multiple of *align* (static ints)."""
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


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


def unflatten_k(k_flat, qkhe_loop: int = 2):
    n = qkhe_loop * 2
    return [[k_flat[td * n + j] for j in range(n)] for td in range(len(k_flat) // n)]
