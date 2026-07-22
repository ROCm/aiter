# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Single home for kernel-facing memory & atomic helpers.

Consolidates the LLVM-pointer construction, global load/store, and atomic
primitives that were previously scattered across ``kernels_common.py`` and
``utils.py`` (and hand-rolled inline in individual kernels). The original
modules re-export these names, so existing imports keep working unchanged.

Two distinct atomic mechanisms live here and are NOT interchangeable:
  * ``atomic_add``        - LLVM ``atomicrmw`` on an ``!llvm.ptr`` (pointer atomics)
  * ``buffer_atomic_add`` - AMD ``raw.ptr.buffer.atomic.fadd`` on a buffer resource
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith as _expr_arith
from flydsl.expr import buffer_ops, const_expr, rocdl
from flydsl.expr.typing import T

# Public API. Prefer the unified names; the legacy aliases at the bottom of this
# list exist only so pre-consolidation imports keep working (re-exported by
# kernels_common / utils) and should not be used in new code:
#   element_ptr  <- get_llvm_ptr        aligned_ptr <- extract_global_ptr
#   to_llvm_ptr  <- _create_llvm_ptr
__all__ = [
    # pointer construction / extraction (preferred)
    "element_ptr",
    "to_llvm_ptr",
    "aligned_ptr",
    "global_ptr_from_addr",
    # global load / store
    "global_load",
    "global_load_i32",
    "global_load_i64x2",
    "global_store",
    # atomics (two distinct mechanisms)
    "atomic_add",
    "buffer_atomic_add",
    # legacy aliases (back-compat only)
    "get_llvm_ptr",
    "_create_llvm_ptr",
    "extract_global_ptr",
]


# --------------------------------------------------------------------------- #
# Pointer construction / extraction
# --------------------------------------------------------------------------- #
def get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type=None):
    """Build a global (address-space 1) ``!llvm.ptr`` at ``ptr + offset*dtype_bytes``.

    Shared home for the LLVM-ptr arithmetic used by atomic/global accesses
    (previously duplicated in hgemm_splitk.py, small_m_hgemm.py, splitk_hgemm.py
    and rmsnorm_kernel.py).
    """
    if ptr_type is None:
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
    base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, ptr)
    base_ptr = _llvm.PtrToIntOp(T.i64, base_ptr).result
    byte_offset = _expr_arith.index_cast(
        T.i64, fx.Index(offset) * fx.Index(dtype_bytes)
    )
    llvm_ptr = _llvm.AddOp(base_ptr, byte_offset, _llvm.IntegerOverflowFlags(0)).result
    llvm_ptr = _llvm.IntToPtrOp(ptr_type, llvm_ptr).result
    return llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr


# Unified name for the element-pointer helper above.
element_ptr = get_llvm_ptr


def _create_llvm_ptr(value, address_space: int = 1):
    value = buffer_ops._unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = T.i64
        value = buffer_ops._unwrap_value(_std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return _llvm.IntToPtrOp(ptr_type, value).result


# Unified name: int/index scalar -> typed llvm.ptr (any address space).
to_llvm_ptr = _create_llvm_ptr


def extract_global_ptr(tensor):
    raw = (
        tensor.ir_value()
        if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value)
        else tensor
    )
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, raw)


# Unified name: tensor/memref -> aligned global base pointer.
aligned_ptr = extract_global_ptr


def global_ptr_from_addr(addr_i64):
    raw = (
        addr_i64.ir_value()
        if hasattr(addr_i64, "ir_value") and not isinstance(addr_i64, ir.Value)
        else addr_i64
    )
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), raw).result


# --------------------------------------------------------------------------- #
# Global load / store
# --------------------------------------------------------------------------- #
def global_load(global_ptr, byte_offset, result_type, *, alignment):
    ptr = buffer_ops.get_element_ptr(
        global_ptr, byte_offset=fx.Int64(byte_offset), elem_type=T.i8
    )
    return _llvm.LoadOp(result_type, ptr, alignment=alignment).result


def global_load_i64x2(global_ptr, byte_offset_i64):
    return global_load(global_ptr, byte_offset_i64, T.i64x2, alignment=16)


def global_load_i32(global_ptr, elem_offset_i32):
    return global_load(
        global_ptr, fx.Int64(elem_offset_i32) * fx.Int64(4), T.i32, alignment=4
    )


def global_store(global_ptr, byte_offset, value, *, alignment):
    """Store ``value`` at ``global_ptr + byte_offset`` (mirror of ``global_load``).

    Unwraps a FlyDSL expression operand the same way ``atomic_add`` does, so
    callers may pass either a raw ``ir.Value`` or a FlyDSL expression. New in the
    consolidation; the hand-rolled ``llvm.StoreOp`` sites are collected in a later
    phase, so this has no call site yet.
    """
    val = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    ptr = buffer_ops.get_element_ptr(
        global_ptr, byte_offset=fx.Int64(byte_offset), elem_type=T.i8
    )
    _llvm.StoreOp(val, ptr, alignment=alignment)


# --------------------------------------------------------------------------- #
# Atomics -- two distinct mechanisms (see module docstring)
# --------------------------------------------------------------------------- #
def atomic_add(
    dst,
    offset,
    value,
    *,
    dtype_bytes=4,
    syncscope="agent",
    ordering=None,
    alignment=None,
    ptr_type=None,
):
    """Atomically add ``value`` into ``dst[offset]`` in global memory.

    inline (e.g. the rmsnorm backward ``dweight`` accumulation). Selects
    ``fadd`` for a floating-point operand and integer ``add`` otherwise, from
    the operand's IR type, so a single call covers both cases. Returns the
    atomicrmw result (the value previously stored at ``dst[offset]``).

    ``dtype_bytes`` sizes the byte offset and, unless ``alignment`` is given, is
    reused as the access alignment.
    """
    ptr = get_llvm_ptr(dst, offset, dtype_bytes, ptr_type=ptr_type)
    val = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    elem_ty = val.type.element_type if isinstance(val.type, ir.VectorType) else val.type
    bin_op = (
        _llvm.AtomicBinOp.fadd
        if isinstance(elem_ty, ir.FloatType)
        else _llvm.AtomicBinOp.add
    )
    if ordering is None:
        ordering = _llvm.AtomicOrdering.monotonic
    if alignment is None:
        alignment = dtype_bytes
    return _llvm.AtomicRMWOp(
        bin_op,
        ptr,
        val,
        ordering,
        syncscope=syncscope,
        alignment=alignment,
    ).result


def buffer_atomic_add(vdata, rsrc, offset, soffset, aux):
    """Buffer-resource atomic fadd (AMD ``raw.ptr.buffer.atomic.fadd``).

    Single entry point for the buffer-resource atomic accumulation used by the
    MoE gemm/blockscale wrappers (``atomic_add_f16x2 / _f32 / _x2``) and conv3d.
    Distinct from ``atomic_add``: operates on a buffer resource + byte offset,
    not an ``!llvm.ptr``, so the two are not interchangeable.
    """
    return rocdl.raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux)
