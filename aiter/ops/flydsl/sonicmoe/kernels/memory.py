# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Memory helpers used only by the FlyDSL SonicMoE kernels."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr

from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64


def element_ptr(ptr, offset, dtype_bytes, ptr_type=None):
    """Build a global LLVM pointer at ``ptr + offset * dtype_bytes``."""

    if ptr_type is None:
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
    byte_offset = fx.Int64(fx.Index(offset) * fx.Index(dtype_bytes))
    address = buf_base_i64(ptr) + byte_offset
    return llvm.IntToPtrOp(ptr_type, address.ir_value()).result


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
    """Atomically add ``value`` into ``dst[offset]`` and return the old value."""

    ptr = element_ptr(dst, offset, dtype_bytes, ptr_type=ptr_type)
    val = value.ir_value() if const_expr(hasattr(value, "ir_value")) else value
    elem_type = (
        val.type.element_type if isinstance(val.type, ir.VectorType) else val.type
    )
    atomic_op = (
        llvm.AtomicBinOp.fadd
        if isinstance(elem_type, ir.FloatType)
        else llvm.AtomicBinOp.add
    )
    if ordering is None:
        ordering = llvm.AtomicOrdering.monotonic
    if alignment is None:
        alignment = dtype_bytes
    return llvm.AtomicRMWOp(
        atomic_op,
        ptr,
        val,
        ordering,
        syncscope=syncscope,
        alignment=alignment,
    ).result


__all__ = ["atomic_add", "element_ptr"]
