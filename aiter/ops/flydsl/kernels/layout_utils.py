"""Layout helpers for GEMM kernels.

Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops for static layouts.
Falls back to fly dialect ops for dynamic layouts.
"""

import re
import builtins as _builtins

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import arith
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T


def _wrap(v):
    """Wrap raw ir.Value in ArithValue for operator overloading compatibility."""
    if isinstance(v, ArithValue):
        return v
    if isinstance(v, ir.Value):
        return ArithValue(v)
    return v


def _parse_dim(tok):
    """Parse a single dimension token: '?' → None, otherwise int."""
    tok = tok.strip()
    return None if tok == '?' else int(tok)


def _parse_layout(ly):
    """Parse '(s0,s1,...):(d0,d1,...)' → (shapes, strides) as lists (None for '?')."""
    ly_str = str(ly.type) if hasattr(ly, 'type') else str(ly)
    m = re.search(r'\(([^)]+)\):\(([^)]+)\)', ly_str)
    if not m:
        return None
    shapes = [_parse_dim(s) for s in m.group(1).split(',')]
    strides = [_parse_dim(s) for s in m.group(2).split(',')]
    return shapes, strides


def _has_dynamic_strides(strides):
    """Check if any stride is dynamic (None)."""
    return any(s is None for s in strides)


def idx2crd(idx, layout):
    """Decompose flat index into a list of coordinate values.

    For static layouts, computes coordinates with plain arith ops.
    For dynamic layouts, falls back to fx.idx2crd + fx.get.
    """
    parsed = _parse_layout(layout)

    if parsed is None or _has_dynamic_strides(parsed[1]):
        result = fx.idx2crd(idx, layout)
        ndims = len(parsed[1]) if parsed else 1
        return [_wrap(fx.get(result, i)) for i in range(ndims)]

    if hasattr(idx, 'type') and str(idx.type) != 'index':
        idx = arith.index_cast(T.index, idx)
    shapes, strides = parsed
    ndims = len(strides)

    ordered = sorted(
        [(i, s, sz) for i, s, sz in _builtins.zip(range(ndims), strides, shapes)
         if s != 0],
        key=lambda x: x[1], reverse=True,
    )
    coords = [None] * ndims
    remaining = idx
    for i, stride_val, size_val in ordered:
        c = remaining / arith.index(stride_val)
        if size_val is not None:
            c = c % arith.index(size_val)
        coords[i] = c
    for i in range(ndims):
        if coords[i] is None:
            coords[i] = remaining
    return coords


def crd2idx(crd, layout):
    """Compute flat index from a coordinate tuple/list.

    For static layouts, computes with plain arith ops.
    For dynamic layouts, falls back to fx.crd2idx with fx.make_coord.
    """
    if not isinstance(crd, (list, tuple)):
        crd = [crd]
    parsed = _parse_layout(layout)

    if parsed is None or _has_dynamic_strides(parsed[1]):
        # fly.make_coord requires i32/i64, not index
        crd_i32 = []
        for c in crd:
            cv = c
            if isinstance(cv, ArithValue):
                cv = cv.ir_value() if hasattr(cv, 'ir_value') else cv
            if isinstance(cv, ir.Value) and isinstance(cv.type, ir.IndexType):
                cv = arith.index_cast(T.i32, cv)
            crd_i32.append(cv)
        coord_val = fx.make_coord(*crd_i32)
        result = fx.crd2idx(coord_val, layout)
        scalar = fx.get_scalar(result)
        if isinstance(scalar, ir.Value) and not isinstance(scalar.type, ir.IndexType):
            scalar = arith.index_cast(T.index, scalar)
        return _wrap(scalar)

    _, strides = parsed
    result = None
    for coord_v, stride_v in _builtins.zip(crd, strides):
        if stride_v == 0:
            continue
        term = coord_v if stride_v == 1 else coord_v * arith.index(stride_v)
        result = term if result is None else result + term
    return result if result is not None else arith.index(0)


def get(int_tuple, mode):
    """Extract element at `mode` from a Python list/tuple."""
    return int_tuple[mode]
