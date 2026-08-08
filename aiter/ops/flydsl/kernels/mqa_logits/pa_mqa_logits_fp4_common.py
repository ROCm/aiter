# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T


def _pack_i32_pair_to_i64(a_i32, b_i32):
    return fx.Vector.from_elements([a_i32, b_i32], dtype=fx.Int32).bitcast(fx.Int64)[0]


def _pack_lo_i64x2_to_i32x8(x0, x1):
    undef0 = _llvm.mlir_undef(T.i64)
    undef1 = _llvm.mlir_undef(T.i64)
    return fx.Vector.from_elements([x0, x1, undef0, undef1], dtype=fx.Int64).bitcast(
        fx.Int32
    )


def _i32_buffer(ptr, width=1):
    """OOB-checked global i32 buffer-tensor over ``ptr`` (mirrors a max_size V#).

    ``width`` shapes it ``(N, width)`` so a per-``width`` row can be sliced and
    vector-copied; ``width=1`` gives a flat tensor for scalar ``[idx]`` loads.
    """
    src = fx.get_iter(ptr)
    it = fx.recast_iter(fx.PointerType.get(T.i32, src.memspace, 4), src)
    if width == 1:
        lay = fx.make_layout((1 << 30,), (1,))
    else:
        lay = fx.make_layout((1 << 28, width), (width, 1))
    return fx.rocdl.make_buffer_tensor(fx.make_view(it, lay))


def _load_vec4_i32(bt2d, elem_off):
    """Load 4 contiguous i32 at ``elem_off`` (multiple of 4) from a width-4
    buffer-tensor, returning a raw v4i32 (OOB lanes read 0)."""
    row = fx.slice(bt2d, (elem_off // fx.Int32(4), None))
    row_div = fx.logical_divide(row, fx.make_layout(4, 1))
    reg_ty = fx.MemRefType.get(T.i32, fx.LayoutType.get(4, 1), fx.AddressSpace.Register)
    r = fx.memref_alloca(reg_ty, fx.make_layout(4, 1))
    fx.copy(
        fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 4),
        fx.slice(row_div, (None, fx.Int32(0))),
        r,
    )
    return fx.memref_load_vec(r).ir_value()
