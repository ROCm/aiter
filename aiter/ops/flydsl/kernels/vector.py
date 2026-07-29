# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Vector dialect wrappers, vendored into aiter.

flydsl deleted ``flydsl.expr.vector``; callers are now expected to use the raw
``flydsl._mlir.dialects.vector`` and wrap every operand in ``as_ir_value``.
aiter has 500+ vector call sites, so the auto-unwrapping layer is kept here
instead — a missing ``as_ir_value`` or ``kDynamic`` sentinel only surfaces at
trace time.

Re-exports the whole raw dialect (``broadcast``, ``shuffle``, ``insert``,
``extract_strided_slice``, ``reduction``, ...) and rebuilds the five wrappers
that unwrap DSL values: ``from_elements``, ``store``, ``extract``,
``load_op``, ``bitcast``. ``extract`` also supplies the ``kDynamic`` sentinel
for dynamic positions. Only published flydsl APIs are used.

Upstream: FlyDSL ``python/flydsl/expr/vector.py``, deleted before
ROCm/FlyDSL#880; the wrappers use FlyDSL's canonical ``as_ir_value``
converter.
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import vector as _vector

# Re-export the raw dialect so vector.broadcast / vector.shuffle work directly
from flydsl._mlir.dialects.vector import *
from flydsl.expr.meta import dsl_loc_tracing

# Vector and related types, which flydsl.expr.vector also re-exported
from flydsl.expr.typing import (  # noqa: F401
    ReductionOp,
    Vector,
    as_ir_value,
    empty_like,
    full,
    full_like,
    ones_like,
    zeros_like,
)

# ═══════════════════════════════════════════════════════════════════════
# Dialect helper wrappers (legacy, will be deprecated)
# Prefer using Vector methods or _mlir.dialects.vector directly.
# ═══════════════════════════════════════════════════════════════════════


def _as_index_ir_value(value):
    if isinstance(value, int):
        from flydsl.expr import arith as _arith_ext

        return _arith_ext.constant(value, index=True)
    return as_ir_value(value)


@dsl_loc_tracing
def from_elements(*args, **kwargs):
    """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [as_ir_value(v) for v in elems]
        return _vector.from_elements(*args, **kwargs)

    return _vector.from_elements(*args, **kwargs)


@dsl_loc_tracing
def store(value, memref, indices, **kwargs):
    """Vector store wrapper that accepts ArithValue/wrappers for value/indices."""
    return _vector.store(
        as_ir_value(value),
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
        **kwargs,
    )


# -----------------------------------------------------------------------------
# Thin wrappers for common op classes that otherwise require `.result` access.
# -----------------------------------------------------------------------------


@dsl_loc_tracing
def extract(vector, static_position=None, dynamic_position=None):
    """Wrapper around `vector.ExtractOp(...).result`.

    When only ``dynamic_position`` is supplied (without explicit
    ``static_position``), each dynamic index needs a corresponding
    ``kDynamic`` sentinel in the static attribute so the ODS builder
    pairs them correctly.  This wrapper fills in the sentinels
    automatically.
    """
    if static_position is None:
        static_position = []
    if dynamic_position is None:
        dynamic_position = []
    dynamic_position = [_as_index_ir_value(i) for i in dynamic_position]

    n_static = len(static_position)
    n_dynamic = len(dynamic_position)
    if n_dynamic > 0 and n_static < n_dynamic:
        kDynamic = ir.ShapedType.get_dynamic_size()
        static_position = list(static_position) + [kDynamic] * (n_dynamic - n_static)

    return _vector.ExtractOp(
        as_ir_value(vector),
        static_position=static_position,
        dynamic_position=dynamic_position,
    ).result


@dsl_loc_tracing
def load_op(result_type, memref, indices):
    """Wrapper around `vector.LoadOp(...).result`."""
    return _vector.LoadOp(
        result_type,
        as_ir_value(memref),
        [_as_index_ir_value(i) for i in indices],
    ).result


@dsl_loc_tracing
def bitcast(result_type, source):
    """Wrapper around `vector.BitCastOp(...).result`."""
    return _vector.BitCastOp(
        result_type,
        as_ir_value(source),
    ).result
