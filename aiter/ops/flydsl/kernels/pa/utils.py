# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

# Upstream (FlyDSL kernels/common/utils.py) also re-exports five mem_ops helpers
# here for back-compat. The vendored paged-decode path uses none of them, so they
# are dropped rather than pulling kernels/common/mem_ops.py in behind them.


def global_pointer_from_addr(addr, dtype, *, alignment: int):
    ptr_type = fx.PointerType.get(
        elem_ty=dtype.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=alignment,
    )
    return fx.inttoptr(ptr_type, addr)


def copy_load(source, offset, copy_atom, register):
    fx.copy(copy_atom, fx.slice(source, (None, fx.Int32(offset))), register)
    return fx.memref_load_vec(register)


def copy_store(destination, offset, copy_atom, register, value):
    fx.memref_store_vec(value, register)
    fx.copy(copy_atom, register, fx.slice(destination, (None, fx.Int32(offset))))


def load_global_16b(global_ptr, byte_offset, copy_atom, register):
    source = fx.make_view(global_ptr + byte_offset, fx.make_layout(16, 1))
    fx.copy(copy_atom, source, register)
    return fx.memref_load_vec(register).bitcast(fx.Int64)


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def exp2_amdgcn_scalar(scalar_value):
    raw = fx.as_ir_value(scalar_value)
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def exp2_f32_fast(value):
    raw = fx.as_ir_value(value)
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        vec = fx.Vector(raw)
        elems = [exp2_amdgcn_scalar(vec[i]) for i in range(ty.shape[0])]
        return fx.Vector.from_elements(elems, vec.dtype)
    return exp2_amdgcn_scalar(raw)


def cdiv(numer, denom):
    """Ceiling division for host integers and typed DSL integer values."""
    if isinstance(numer, (fx.Numeric, fx.Vector)) or isinstance(denom, (fx.Numeric, fx.Vector)):
        return fx.ceildiv(numer, denom)
    return -(-numer // denom)


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
