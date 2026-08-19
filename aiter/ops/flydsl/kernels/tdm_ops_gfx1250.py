# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx1250 TDM compatibility helpers."""

from __future__ import annotations

import contextlib

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import fly as fly_dialect
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.rocdl import tdm_ops as _tdm_ops
from flydsl.expr.typing import as_ir_value

TDMDescriptor2D = _tdm_ops.TDMDescriptor2D
tensor_load_2d = _tdm_ops.tensor_load_2d
tensor_wait = _tdm_ops.tensor_wait
update_tensor_descriptor_2d_addr64 = _tdm_ops.update_tensor_descriptor_2d_addr64

compute_padding_encoding = _tdm_ops.compute_padding_encoding
compute_warp_distribution = _tdm_ops.compute_warp_distribution

__all__ = [
    "TDMDescriptor2D",
    "make_tensor_descriptor_2d",
    "tensor_load_2d",
    "tensor_wait",
    "update_tensor_descriptor_2d_addr64",
    "update_tensor_descriptor_2d_lds_addr",
]


def _fly_lds_base_index(raw: ir.Value) -> ir.Value:
    """Extract a Fly shared pointer / view as an LDS byte index."""
    ptr_type = ir.Type.parse("!llvm.ptr<3>")
    ptr = fly_dialect.extract_aligned_pointer_as_index(ptr_type, raw)
    i64 = ir.IntegerType.get_signless(64)
    ptr_i64 = llvm_dialect.ptrtoint(i64, ptr)
    return std_arith.IndexCastOp(ir.IndexType.get(), ptr_i64).result


class _FlyAwareMemrefDialect:
    """``memref`` dialect proxy whose pointer extraction also accepts Fly values."""

    def __getattr__(self, name):
        return getattr(memref_dialect, name)

    @staticmethod
    def extract_aligned_pointer_as_index(source):
        raw = as_ir_value(source)
        try:
            ir.MemRefType(raw.type)
        except ValueError:
            return _fly_lds_base_index(raw)
        return memref_dialect.extract_aligned_pointer_as_index(raw)


@contextlib.contextmanager
def _fly_aware_lds_extraction():
    """Let ``tdm_ops`` take a Fly shared view where it expects a memref."""
    saved = _tdm_ops.memref_dialect
    _tdm_ops.memref_dialect = _FlyAwareMemrefDialect()
    try:
        yield
    finally:
        _tdm_ops.memref_dialect = saved


def make_tensor_descriptor_2d(*args, **kwargs) -> TDMDescriptor2D:
    """``tdm_ops.make_tensor_descriptor_2d`` accepting a Fly ``lds_memref``."""
    with _fly_aware_lds_extraction():
        return _tdm_ops.make_tensor_descriptor_2d(*args, **kwargs)


@dsl_loc_tracing
def update_tensor_descriptor_2d_lds_addr(
    desc: TDMDescriptor2D,
    new_lds_addr,
) -> TDMDescriptor2D:
    """Return a 2-D descriptor with its LDS address replaced."""
    return TDMDescriptor2D(
        dgroup0=vector.InsertOp(
            _raw(new_lds_addr),
            _raw(desc.dgroup0),
            static_position=[1],
            dynamic_position=[],
        ).result,
        dgroup1=desc.dgroup1,
    )
