# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Shared memory, wave, and score primitives for gfx950 FP8 attention."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import LOG2E

LN2 = 1.0 / LOG2E
_P_HEADROOM_LOG2 = 8.807354922057604  # log2(448), the largest finite E4M3 value.


def daz_denormal_attr():
    """Return the structured f32 DAZ attribute consumed by LLVM lowering.

    The ``llvm.`` key carries this through GPU-to-ROCDL lowering. The former
    ``denormal-fp-math-f32`` passthrough string reaches IR but does not change
    the hardware denormal mode.
    """
    return ir.Attribute.parse(
        "#llvm.denormal_fpenv<"
        "default_output_mode = ieee, default_input_mode = ieee, "
        "float_output_mode = preservesign, float_input_mode = preservesign>"
    )


def _p_headroom_log2(traits):
    """Exponent bias shared by FP8 probability packing and LSE recovery."""
    headroom = _P_HEADROOM_LOG2
    if traits.DUALWAVE_SWP_LAZY_RESCALE:
        headroom -= traits.DUALWAVE_SWP_RESCALE_THRESHOLD
    return max(0.0, headroom)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


def _score_pair_to_lists(v_s):
    s_lo, s_hi = v_s
    return (
        [Vec(s_lo)[r] for r in range_constexpr(16)],
        [Vec(s_hi)[r] for r in range_constexpr(16)],
    )


def _score_lists_to_vecs(v_s_lists):
    s_lo, s_hi = v_s_lists
    return (
        Vec.from_elements([fx.as_ir_value(v) for v in s_lo], fx.Float32).ir_value(),
        Vec.from_elements([fx.as_ir_value(v) for v in s_hi], fx.Float32).ir_value(),
    )


def _reduction_pair(v_f32):
    v_i32 = fx.Float32(v_f32).bitcast(fx.Int32).ir_value()
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, v_i32, v_i32, False, True)
    lhs_i32 = llvm.extractvalue(T.i32, swapped, [0])
    rhs_i32 = llvm.extractvalue(T.i32, swapped, [1])
    return fx.Int32(lhs_i32).bitcast(fx.Float32), fx.Int32(rhs_i32).bitcast(fx.Float32)


def _lane_pair_reduce(v, reducer, fm_fast):
    lhs, rhs = _reduction_pair(v)
    return reducer(lhs, rhs, fm_fast)


def _score_pair_sum(v_s, zero_f, fm_fast):
    s_lo, s_hi = _score_lists_to_vecs(v_s)
    tile = Vec(s_lo) + Vec(s_hi)
    reducer = lambda a, b, _fm: a + b
    return _lane_pair_reduce(
        tile.reduce("add", init_val=zero_f, fastmath=fm_fast), reducer, fm_fast
    )


def _scale_o_accs(v_o, scale_scalar, traits):
    scale_vec = Vec.from_elements([scale_scalar], fx.Float32).broadcast_to(16)
    for dc in range_constexpr(traits.D_CHUNKS):
        v_o[dc] = Vec(v_o[dc]) * scale_vec


def _init_dualwave_q_row(ctx):
    """Set the live query row and its int32 masking coordinates."""
    traits = ctx.traits
    ctx.q_row_in_block = ctx.wave_q_offset + ctx.lane_mod_32
    ctx.q_start_pos_i32 = fx.Int32(ctx.q_start + ctx.wave_id_uni * traits.ROWS_PER_WAVE)
    ctx.q_row = ctx.q_start + ctx.q_row_in_block
    ctx.q_row_i32 = fx.Int32(ctx.q_row)


def _cu_load(div, idx, cu_atom, cu_v1i32):
    """Load cu_seqlens[idx] into an SGPR. ``idx`` must be wave-uniform."""
    v = fly.copy_atom_call_ssa(
        [cu_v1i32], cu_atom, fx.slice(div, (None, fx.Int32(idx)))
    )
    value = Vec(v, (1,), fx.Int32)[0]
    return fx.Index(rocdl.readfirstlane(T.i32, value.ir_value()))


def _buffer_load_128(elem_index, _load_atom_128, q_div, q_load_i32x4_type):
    """128-bit global->register load (buffer_load_dwordx4) from Q."""
    return fly.copy_atom_call_ssa(
        [q_load_i32x4_type],
        _load_atom_128,
        fx.slice(q_div, (None, fx.Int32(elem_index))),
    )


def _buffer_load_lds_128(
    src_div, lds_byte_addr, src_elem, soffset_elems, _dma_atom, _lds_ptr_ty
):
    """128-bit global->LDS DMA; `src_elem` is voffset, `soffset_elems` is scaled by the atom."""
    lds_ptr = fx.inttoptr(_lds_ptr_ty, fx.Int32(lds_byte_addr))
    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
    src = fx.slice(src_div, (None, fx.Int32(src_elem)))
    fx.copy(_dma_atom, src, dst, soffset=fx.Int32(soffset_elems))


def _buffer_store_128(
    pack_i32_vec, elem_index, _o_store_reg_128, _store_atom_128, o_div
):
    """128-bit register->global store (buffer_store_dwordx4) into O."""
    fx.memref_store_vec(pack_i32_vec, _o_store_reg_128)
    fx.copy(
        _store_atom_128, _o_store_reg_128, fx.slice(o_div, (None, fx.Int32(elem_index)))
    )
