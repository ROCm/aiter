# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Layout-API buffer views with an element-offset calling convention.

These replace the vendored ``buffer_ops`` shim's ``(rsrc, element_offset)``
form. A buffer here is an ordinary FlyDSL buffer tensor, so loads and stores go
through ``fx.copy`` with a ``BufferCopy`` atom and keep hardware OOB checking.

The view layout is ``((width, 1), (1, 1))``. Stride ``(1, 1)`` is the load-bearing
detail: coordinate ``(i, j)`` maps to element ``i + j``, so slicing the last
coordinate counts **elements** and reproduces the shim's offset units exactly.
The obvious-looking ``(1, width)`` counts *groups* of ``width`` and silently
shifts every address once ``width > 1``.

Element type is chosen by the caller, which is how byte-addressed access works:
an i8-typed buffer's element index already is a byte offset, so the shim's
``offset_is_bytes`` flag has no equivalent here and is simply not needed.
"""

import flydsl.expr as fx
from flydsl.expr import as_ir_value
from flydsl.expr.typing import Vector as Vec


def make_buffer_addr(
    addr_i64, elem_ty, width=1, *, max_size=True, num_records_bytes=None
):
    """Build a typed buffer tensor over a computed i64 base address."""
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


def make_buffer(src, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    """Build a typed buffer tensor from a tensor or a raw pointer kernel arg.

    A tensor is dereferenced with ``get_iter`` first; a pointer argument goes
    straight to ``ptrtoint``, matching the byte-address math the shim used.
    """
    ptr = src if isinstance(src, fx.Pointer) else fx.get_iter(src)
    return make_buffer_addr(
        fx.Int64(fx.ptrtoint(ptr)),
        elem_ty,
        width,
        max_size=max_size,
        num_records_bytes=num_records_bytes,
    )


def buffer_load(buffer, index, elem_ty, width=1, cache_modifier=0, raw=False):
    """Load ``width`` elements of ``elem_ty`` starting at element ``index``.

    ``raw=True`` returns an ``ir.Value`` instead of an fx value, for callers
    whose result feeds a raw ``arith.*`` / ``rocdl.*`` sink. Those sinks reject
    ``Numeric`` (``fx.Int32`` and friends are not ``ir.Value`` subclasses), so
    getting this wrong surfaces as ``Operand N ... must be a Value`` one trace
    error at a time.
    """
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, index)), fragment)
    value = Vec(fragment.load())
    value = value[0] if width == 1 else value
    return as_ir_value(value) if raw else value


def buffer_store(buffer, index, value, elem_ty, width=1, cache_modifier=0):
    """Store ``width`` elements of ``elem_ty`` starting at element ``index``."""
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(Vec.from_elements([value], elem_ty) if width == 1 else Vec(value))
    fx.copy(atom, fragment, fx.slice(buffer, (None, index)))
