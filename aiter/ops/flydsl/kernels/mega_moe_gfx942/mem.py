# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side memory helpers of the kernels: global / LDS pointers, buffer (V#) loads
and stores, i32 buffer views.

The kernels load and store rows found at run time, a wave-uniform base plus a per-lane
offset, which has no layout form; those accesses go through aiter's ``buffer_ops``
(resource descriptor in SGPRs, 32-bit per-lane byte offset, out-of-range lanes read 0
and drop their stores).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from ..buffer_ops import (  # noqa: F401  (re-exported to the kernels)
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    get_element_ptr,
)
from ..mxfp4_gemm_common import (  # noqa: F401  (re-exported to the kernels)
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from ..tensor_shim import _to_raw as _raw


def _pointer(address, dtype, alignment=16):
    """Global fx.Pointer to ``dtype`` (an fx class) over a raw i64 device address."""
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment),
        fx.Int64(address),
    )


def _store(ptr, idx, value):
    """ptr[idx] = value (a call: inside a dynamic if, the AST rewriter would take a
    subscript store to a closure variable for an assignment of it)."""
    ptr[idx] = value


@flyc.jit
def _lds_f32_store_masked(lds_f32, idx, val, ok):
    if ok:
        lds_f32[idx] = val


def _rsrc(addr, nbytes=None):
    """Buffer resource over the i64 address ``addr``; ``nbytes``: bound of the hardware range
    check (None: 4 GB)."""
    if nbytes is None:
        return create_buffer_resource_from_addr(_raw(fx.Int64(addr)))
    return create_buffer_resource_from_addr(
        _raw(fx.Int64(addr)), num_records_bytes=_raw(fx.Int64(nbytes))
    )


def _ld(rsrc, idx, width=1, mask=None):
    """i32 (width 1) / i32 x width at i32 element ``idx`` (masked: 0)."""
    return buffer_load(rsrc, idx, vec_width=width, dtype=T.i32, mask=mask)


# --------------------------------------------------------------------------------------
# i32 buffer views (layout API: BufferCopy atoms)
# --------------------------------------------------------------------------------------


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])
