# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Buffer helpers of the MegaMoEV2 small- and large-input kernels' dispatch and combine code.

Their launch configs live in mega_moe_config (select_fused_config); their GEMM
tiles come from gemm_util, gemm1 and gemm2.
"""

from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

from ..tensor_shim import _fx_value_elem, buf_copy_load, buf_copy_store, ptr_buf_tensor


class AddrBuffer(NamedTuple):
    """A raw device address and an optional hardware OOB bound in bytes."""

    addr: object
    num_records_bytes: object = None


def _addr_view(buf, elem, width):
    # unit_stride=1: a wide access may start at any element offset.
    return ptr_buf_tensor(
        buf.addr,
        elem,
        unit_elems=width,
        unit_stride=1,
        num_records_bytes=buf.num_records_bytes,
    )


def load_at(buf, index, elem=fx.Int32, width=1, cache_modifier=0):
    """Load ``width`` x ``elem`` at element ``index`` of ``buf``."""
    return buf_copy_load(
        _addr_view(buf, elem, width), index, elem, width, cache_modifier
    )


def store_at(value, buf, index, cache_modifier=0, offset_is_bytes=False):
    """Store ``value`` (scalar or vector) at element ``index`` of ``buf``."""
    elem, width = _fx_value_elem(value)
    if offset_is_bytes:
        index = fx.Int32(index) // fx.Int32(elem.width // 8)
    buf_copy_store(
        _addr_view(buf, elem, width), index, value, elem, width, cache_modifier
    )


def _make_buffer(tensor, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


def _make_buffer_from_addr(addr, elem_ty, width=1, *, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(addr))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def _buffer_load(buffer, group_index, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, group_index)), fragment)
    value = Vec(fragment.load())
    return value[0] if width == 1 else value


def _buffer_store(buffer, group_index, value, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(Vec.from_elements([value], elem_ty) if width == 1 else Vec(value))
    fx.copy(atom, fragment, fx.slice(buffer, (None, group_index)))
