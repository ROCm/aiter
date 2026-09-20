# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Memory and wave primitives for dense and native paged gfx950 FP8 attention."""

import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..kernels_common import LOG2E

LN2 = 1.0 / LOG2E
_P_HEADROOM_LOG2 = 8.807354922057604  # log2(448), the largest finite E4M3 value.


def _p_headroom_log2(traits):
    """Exponent bias shared by FP8 probability packing and LSE recovery."""
    headroom = _P_HEADROOM_LOG2
    if traits.DUALWAVE_SWP_LAZY_RESCALE:
        headroom -= traits.DUALWAVE_SWP_RESCALE_THRESHOLD
    return max(0.0, headroom)


def load(ptr, *, dtype, count):
    """Plain copy on FlyDSL 0.3.2, preserving byte-only pointer alignment."""
    # Dynamic FP8 offsets need not be word-aligned. Copy bytes, then bitcast
    # registers; do not strengthen the pointer alignment to use a wider type.
    byte_count = count * dtype.width // 8
    view = fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(byte_count, 1))
    fragment = fx.make_rmem_tensor(byte_count, fx.Uint8)
    atom = fx.make_copy_atom(fx.UniversalCopy(byte_count * 8), fx.Uint8)
    fx.copy(atom, view, fragment)
    result = fx.Vector(fragment.load()).bitcast(dtype)
    return result[0] if count == 1 else result


def store(ptr, value):
    """Plain scalar/vector store without assuming more than byte alignment."""
    vector = (
        value
        if isinstance(value, fx.Vector)
        else fx.Vector.from_elements([value], type(value))
    )
    packed = vector.bitcast(fx.Uint8)
    view = fx.make_view(fx.recast_iter(fx.Uint8, ptr), fx.make_layout(packed.numel, 1))
    fragment = fx.make_rmem_tensor(packed.numel, fx.Uint8)
    fragment.store(packed)
    atom = fx.make_copy_atom(fx.UniversalCopy(packed.numel * 8), fx.Uint8)
    fx.copy(atom, fragment, view)


def _read_exec_i64():
    """Read the current wave exec mask, matching Clang's builtin lowering."""
    true_i1 = fx.Boolean(True).ir_value()
    return rocdl.ballot(T.i64, true_i1)


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
