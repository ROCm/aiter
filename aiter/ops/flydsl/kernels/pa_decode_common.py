# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared math, layout loads, and native instruction boundaries for PA decode."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

from .dpp_utils import update_dpp_i32


def make_flat_loader(tensor, dtype, width, copy_op):
    atom = fx.make_copy_atom(copy_op, dtype)
    register = fx.make_rmem_tensor(fx.make_layout(width, 1), dtype)
    if isinstance(copy_op, fx.rocdl.CopyOpCDNA3BufferCopyType):
        tensor = fx.rocdl.make_buffer_tensor(tensor, max_size=True)
    flat = fx.make_view(fx.get_iter(tensor), fx.make_layout(1 << 30, 1))
    tiles = fx.logical_divide(flat, fx.make_layout(1, 1))

    def load(element_offset):
        fx.copy(atom, fx.slice(tiles, (None, element_offset)), register)
        return fx.Vector(fx.memref_load_vec(register))

    return load


def load_lds_words(base, byte_offset, words=4):
    # Preserve the 8-/16-byte alignment of each operand's swizzled LDS layout.
    pointer = fx.add_offset(base, fx.make_int_tuple(byte_offset))
    pointer_type = fx.PointerType.get(
        fx.Int32.ir_type, fx.AddressSpace.Shared, words * 4
    )
    return fx.Vector(
        fx.ptr_load(
            fx.recast_iter(pointer_type, pointer),
            result_type=fx.Vector.make_type(words, fx.Int32),
        )
    )


def page_resource(tensor, element_offset, page_elements, element_bytes=1):
    # Rebase at the 64-bit physical page address before using 32-bit DMA offsets.
    base = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element_offset))
    view = fx.make_view(base, fx.make_layout(page_elements, 1))
    buffer = fx.rocdl.make_buffer_tensor(
        view, num_records_bytes=fx.Int64(page_elements * element_bytes).ir_value()
    )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(buffer))


def async_load_lds_nt(source, source_byte_offset, lds_base, wave_byte_offset):
    # The 0.3.2 async copy atom cannot express NT. The raw intrinsic also takes
    # byte offsets and a uniform m0 base; hardware adds lane * 16 to the latter.
    address = fx.Int32(fx.ptrtoint(lds_base)) + wave_byte_offset
    uniform = fx.Int32(fx.rocdl.readfirstlane(T.i32, address))
    pointer_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
    destination = fx.to_llvm_ptr(fx.inttoptr(pointer_type, uniform))
    fx.rocdl.raw_ptr_buffer_load_async_lds(
        source,
        destination,
        fx.Int32(16).ir_value(),
        source_byte_offset.ir_value(),
        fx.Int32(0).ir_value(),
        fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(T.i32, 2),
    )


def swap_lane_pair(a, b, bit=2):
    # Native lane swaps return an LLVM pair, not a vector; unpack only here.
    pair_type = ir.Type.parse("!llvm.struct<(i32, i32)>")
    operation = fx.rocdl.permlane16_swap if bit == 1 else fx.rocdl.permlane32_swap
    pair = operation(pair_type, a.ir_value(), b.ir_value(), False, False)
    return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(
        llvm.extractvalue(T.i32, pair, [1])
    )


def reduce_lane_pair(value, bit=2):
    bits = fx.Float32(value).bitcast(fx.Int32)
    low, high = swap_lane_pair(bits, bits, bit)
    return low.bitcast(fx.Float32), high.bitcast(fx.Float32)


def copy_load(source, offset, copy_atom, register):
    fx.copy(copy_atom, fx.slice(source, (None, fx.Int32(offset))), register)
    return fx.memref_load_vec(register)


def copy_store(destination, offset, copy_atom, register, value):
    fx.memref_store_vec(value, register)
    fx.copy(copy_atom, register, fx.slice(destination, (None, fx.Int32(offset))))


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
    if isinstance(numer, (fx.Numeric, fx.Vector)) or isinstance(
        denom, (fx.Numeric, fx.Vector)
    ):
        return fx.ceildiv(numer, denom)
    return -(-numer // denom)


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


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    src_i32 = fx.Float32(src).bitcast(fx.Int32)
    if offset == 8:
        out_i32 = update_dpp_i32(src_i32, src_i32, 280, 0xF, 0xC, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 264, 0xF, 0x3, False, **kw)
    elif offset == 4:
        out_i32 = update_dpp_i32(src_i32, src_i32, 276, 0xF, 0xA, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 260, 0xF, 0x5, False, **kw)
    elif offset == 2:
        out_i32 = update_dpp_i32(src_i32, src_i32, 78, 0xF, 0xF, False, **kw)
    elif offset == 1:
        out_i32 = update_dpp_i32(src_i32, src_i32, 177, 0xF, 0xF, False, **kw)
    else:
        raise ValueError(
            f"dpp_xor_f32 only supports 16-lane offsets 1, 2, 4, 8; got {offset}"
        )
    return fx.Int32(out_i32).bitcast(fx.Float32).ir_value()
