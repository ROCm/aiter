# SPDX-License-Identifier: MIT

"""Shared numeric and tail primitives for BF16 decode GEMM policies."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from .gemm_decode_config import (
    ContractionMode,
    MFMA_K,
    OutputRounding,
    ReductionMode,
)


def raw(value):
    if isinstance(value, ir.Value):
        return value
    if hasattr(value, "ir_value"):
        return raw(value.ir_value())
    if hasattr(value, "value"):
        return raw(value.value)
    return value


def const_bf16(value: float = 0.0):
    return arith_dialect.ConstantOp(
        ir.BF16Type.get(),
        ir.FloatAttr.get(ir.BF16Type.get(), value),
    ).result


def const_f32(value: float = 0.0):
    return arith_dialect.ConstantOp(
        ir.F32Type.get(),
        ir.FloatAttr.get(ir.F32Type.get(), value),
    ).result


def add_f32(lhs, rhs):
    return arith_dialect.AddFOp(raw(lhs), raw(rhs)).result


def pack_bf16x2(lo, hi):
    lo_i16 = ArithValue(raw(lo)).bitcast(T.i16)
    hi_i16 = ArithValue(raw(hi)).bitcast(T.i16)
    lo_i32 = ArithValue(lo_i16).extui(T.i32)
    hi_i32 = ArithValue(hi_i16).extui(T.i32)
    return ArithValue(lo_i32) | (ArithValue(hi_i32) << fx.Int32(16))


def unpack_bf16x2_f32(packed):
    packed = ArithValue(raw(packed))
    lo_bits = (packed & fx.Int32(0xFFFF)) << fx.Int32(16)
    hi_bits = packed & fx.Int32(0xFFFF0000)
    return (
        raw(ArithValue(lo_bits).bitcast(T.f32)),
        raw(ArithValue(hi_bits).bitcast(T.f32)),
    )


def prepare_pair(packed, contraction: ContractionMode):
    if contraction == ContractionMode.DOT2_BF16:
        return raw(packed)
    expanded = unpack_bf16x2_f32(packed)
    if contraction == ContractionMode.PACKED_F32:
        return vector.from_elements(T.vec(2, T.f32), list(expanded))
    return expanded


def zero_wave_accumulator(contraction: ContractionMode):
    if contraction == ContractionMode.PACKED_F32:
        return arith.constant_vector(0.0, T.vec(2, T.f32))
    return const_f32()


def contract_pair(accumulator, a_pair, b_pair, contraction: ContractionMode):
    if contraction == ContractionMode.DOT2_BF16:
        return llvm.inline_asm(
            ir.F32Type.get(),
            [raw(accumulator), raw(a_pair), raw(b_pair)],
            "v_dot2_f32_bf16 $0, $2, $3, $1",
            "=v,0,v,v",
            has_side_effects=False,
        )
    if contraction == ContractionMode.PACKED_F32:
        return llvm.inline_asm(
            ir.VectorType.get([2], ir.F32Type.get()),
            [raw(accumulator), raw(a_pair), raw(b_pair)],
            "v_pk_fma_f32 $0, $2, $3, $1",
            "=v,0,v,v",
            has_side_effects=False,
        )
    accumulator = llvm.intr_fma(a_pair[0], b_pair[0], raw(accumulator))
    return llvm.intr_fma(a_pair[1], b_pair[1], accumulator)


def dpp_add_f32(value, control: str):
    return llvm.inline_asm(
        ir.F32Type.get(),
        [raw(value), raw(value), raw(value)],
        f"s_nop 3\n\tv_add_f32 $0, $2, $3 {control} bound_ctrl:0",
        "=v,0,v,v",
        has_side_effects=False,
    )


def wavefront_reduce_sum_f32(value):
    for shift in (8, 4, 2, 1):
        value = dpp_add_f32(value, f"row_shr:{shift}")
    value = dpp_add_f32(value, "row_bcast:15")
    return dpp_add_f32(value, "row_bcast:31")


def bpermute_reduce_sum_f32(value, lane):
    value = llvm.inline_asm(
        ir.F32Type.get(),
        [raw(value)],
        "s_nop 3\n\tv_mov_b32 $0, $1",
        "=v,v",
        has_side_effects=False,
    )
    for stage in range_constexpr(6):
        partner = lane ^ fx.Int32(1 << stage)
        value_i32 = ArithValue(raw(value)).bitcast(T.i32)
        peer_i32 = fx.rocdl.ds_bpermute(
            T.i32,
            partner * fx.Int32(4),
            value_i32,
        )
        value = add_f32(value, ArithValue(peer_i32).bitcast(T.f32))
    return value


def reduce_wave_accumulator(accumulator, lane, contraction, reduction):
    use_dpp = reduction == ReductionMode.DPP
    if contraction != ContractionMode.PACKED_F32:
        return (
            wavefront_reduce_sum_f32(accumulator)
            if use_dpp
            else bpermute_reduce_sum_f32(accumulator, lane)
        )
    lo = vector.extract(
        accumulator,
        static_position=[0],
        dynamic_position=[],
    )
    hi = vector.extract(
        accumulator,
        static_position=[1],
        dynamic_position=[],
    )
    if use_dpp:
        lo = wavefront_reduce_sum_f32(lo)
        hi = wavefront_reduce_sum_f32(hi)
    else:
        lo = bpermute_reduce_sum_f32(lo, lane)
        hi = bpermute_reduce_sum_f32(hi, lane)
    return add_f32(lo, hi)


def convert_bf16(value, element, rounding: OutputRounding):
    if rounding == OutputRounding.RNE:
        return arith_dialect.TruncFOp(T.bf16, raw(value)).result
    seed = (
        ArithValue(raw(element)) * fx.Int32(0x45D9F3B)
    ) ^ (ArithValue(raw(element)) << fx.Int32(16)) ^ fx.Int32(0x27D4EB2D)
    converted = llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [raw(value), raw(seed)],
        "v_cvt_sr_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )
    return ArithValue(converted).trunci(T.i16).bitcast(T.bf16)


def store_bf16(value, resource, element, rounding: OutputRounding) -> None:
    output = convert_bf16(value, element, rounding)
    buffer_ops.buffer_store(output, resource, element)


def mfma_4x4x4_bf16(a_fragment, b_fragment, accumulator):
    """Use the shared native atom; FlyDSL has no matching high-level MMA atom."""
    a_i16 = vector.bitcast(T.vec(4, T.i16), a_fragment)
    b_i16 = vector.bitcast(T.vec(4, T.i16), b_fragment)
    return fx.rocdl.mfma_f32_4x4x4bf16_1k_(
        T.vec(4, T.f32),
        raw(a_i16),
        raw(b_i16),
        raw(accumulator),
        0,
        0,
        0,
    )


def bf16x4_slice(fragment, fragment_index: int):
    return vector.extract_strided_slice(
        T.vec(MFMA_K, T.bf16),
        raw(fragment),
        [fragment_index * MFMA_K],
        [MFMA_K],
        [1],
    )


def dpp_move_f32(value, control: int):
    return fx.rocdl.update_dpp(
        T.f32,
        raw(value),
        raw(value),
        control,
        0xF,
        0xF,
        True,
    )


def reduce_mfma_scalar(accumulator):
    components = [
        vector.extract(accumulator, static_position=[i], dynamic_position=[])
        for i in range_constexpr(4)
    ]
    result = components[0]
    result = add_f32(result, dpp_move_f32(components[1], 0x101))
    result = add_f32(result, dpp_move_f32(components[2], 0x102))
    result = add_f32(result, dpp_move_f32(components[3], 0x103))
    result = add_f32(result, dpp_move_f32(result, 0x104))
    result = add_f32(result, dpp_move_f32(result, 0x108))
    result = dpp_move_f32(result, 0x11F)
    result = add_f32(result, dpp_move_f32(result, 0x142))
    return add_f32(result, dpp_move_f32(result, 0x143))


def masked_bf16_vector(
    resource,
    row_base,
    column_base,
    width: int,
    row_size: int,
    cache_modifier: int = 0,
    row_valid=None,
):
    """Load a compile-time BF16 vector with safe N/K tail masking."""
    zero = const_bf16()
    values = []
    for offset in range_constexpr(width):
        column = column_base + fx.Int32(offset)
        valid = column < fx.Int32(row_size)
        if row_valid is not None:
            valid = valid & row_valid
        safe_column = ArithValue(raw(valid)).select(column, fx.Int32(0))
        safe_row_base = row_base
        if row_valid is not None:
            safe_row_base = ArithValue(raw(row_valid)).select(row_base, fx.Int32(0))
        loaded = buffer_ops.buffer_load(
            resource,
            safe_row_base + safe_column,
            vec_width=1,
            dtype=T.bf16,
            cache_modifier=cache_modifier,
        )
        values.append(ArithValue(raw(valid)).select(loaded, zero))
    return vector.from_elements(T.vec(width, T.bf16), values)
