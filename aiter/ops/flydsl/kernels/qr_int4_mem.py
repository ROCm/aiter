# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Low-level buffer / IPC pointer helpers used by ``qr_int4_quad_fanout``.

Vendored subset of FlyDSL ``kernels/comm/custom_all_reduce_kernel.py``.
The rest of that all-reduce kernel is not copied.
"""

from __future__ import annotations

from typing import ClassVar

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm, rocdl
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.typing import T

from . import buffer_ops

# AMD GFX942 buffer aux: bit 2 = NT (nontemporal).
_CM_CACHED = 0
_CM_SC1 = 2
_CM_SC0_SC1 = 3
_CM_NT = 4


def _make_rsrc(addr_i64):
    """Create buffer resource descriptor from a wave-uniform i64 base address."""
    return buffer_ops.create_buffer_resource_from_addr(addr_i64)


def _load_v4i32(rsrc, elem_off_i32):
    """Buffer-load vector<4xi32> (16 bytes) with pre-built descriptor."""
    raw = buffer_ops.buffer_load(rsrc, elem_off_i32, vec_width=4, dtype=T.i32)
    return fx.Vector(raw)


def _store_v4i32(rsrc, elem_off_i32, data):
    """Buffer-store vector<4xi32> (16 bytes), cached."""
    buffer_ops.buffer_store(data, rsrc, elem_off_i32, cache_modifier=_CM_CACHED)


def _load_i32_uncached(rsrc):
    """Load i32 bypassing L2 (sc1) via pre-built rsrc descriptor."""
    val = buffer_ops.buffer_load(
        rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=_CM_SC1
    )
    rocdl.s_waitcnt(0)
    return val


def _store_i32_uncached(rsrc, val_i32):
    """Store i32 bypassing L1+L2 (sc0+sc1) via pre-built rsrc descriptor."""
    buffer_ops.buffer_store(val_i32, rsrc, 0, cache_modifier=_CM_SC0_SC1)
    rocdl.s_waitcnt(0)


def _invalidate_l1():
    """Invalidate L1 scalar cache (buffer_inv sc1)."""
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


def _pack_i64_vec(values):
    """Pack preloaded i64 values into vector<Nxi64> for contiguous VGPR storage."""
    return fx.Vector.from_elements(values, dtype=fx.Int64)


def _extract_i64(vec, index):
    """Extract i64 from a packed vector by dynamic index (VGPR-relative)."""
    if not isinstance(vec, fx.Vector):
        vec = fx.Vector(vec)
    return vec[index]


def _load_device_ptr(array_base_i64, index):
    """Load i64 pointer from a device-side pointer array at *index*."""
    rsrc = buffer_ops.create_buffer_resource_from_addr(array_base_i64)
    return buffer_ops.buffer_load(rsrc, index, vec_width=1, dtype=T.i64)


def _raw(v):
    """Unwrap FlyDSL wrapper values when low-level MLIR ops need raw ir.Value."""
    return v.ir_value() if hasattr(v, "ir_value") else v


class _IfOnlyASTRewriter(ASTRewriter):
    """AST rewriter variant that lowers Python if, keeps while untouched."""

    transformers: ClassVar[list] = [
        t for t in ASTRewriter.transformers if t.__name__ != "CanonicalizeWhile"
    ]
    rewrite_globals: ClassVar[dict] = {
        name: value
        for name, value in ASTRewriter.rewrite_globals.items()
        if name not in {"scf_while_gen", "scf_while_init"}
    }


def _dsl_if_only(func):
    """Rewrite helper-level Python if into scf.if without touching while."""
    return _IfOnlyASTRewriter.transform(func)
