# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Opt-in per-CTA timestamp trace for MegaMoE kernels (AITER_MEGA_TRACE=1).

Each traced kernel owns a row range of one process-wide int64 buffer; lane 0 of a
CTA stores ``s_memrealtime`` (100 MHz) at named points.  The buffer address is
baked into the kernel at compile time, so tracing is a debugging aid only.
"""

import os

import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm as _llvm_d

from ..tensor_shim import ptr_buf_tensor

TRACE_ENABLED = os.environ.get("AITER_MEGA_TRACE", "0") == "1"
TRACE_FIELDS = 8
# Row bases per kernel kind; each kind has room for MAX_ROWS CTAs.
TRACE_MAX_ROWS = 16384
TRACE_KINDS = ("stage1", "stage2", "combine", "prepare")
_BUFFER = None


def trace_buffer():
    global _BUFFER
    if _BUFFER is None:
        _BUFFER = torch.zeros(
            len(TRACE_KINDS) * TRACE_MAX_ROWS * TRACE_FIELDS,
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
    return _BUFFER


def trace_base(kind):
    """Byte address of ``kind``'s rows (0 when tracing is disabled)."""
    if not TRACE_ENABLED:
        return 0
    row0 = TRACE_KINDS.index(kind) * TRACE_MAX_ROWS
    return trace_buffer().data_ptr() + row0 * TRACE_FIELDS * 8


def trace_rows(kind):
    """Host view [TRACE_MAX_ROWS, TRACE_FIELDS] of ``kind``'s rows."""
    row0 = TRACE_KINDS.index(kind) * TRACE_MAX_ROWS
    return trace_buffer().view(-1, TRACE_FIELDS)[row0 : row0 + TRACE_MAX_ROWS]


def now():
    return fx.Int64(
        _llvm_d.call_intrinsic(
            fx.Int64.ir_type, "llvm.amdgcn.s.memrealtime", [], [], []
        )
    )


def record(base, row, field, value):
    """Store ``value`` into ``[row, field]`` of a trace range (lane-0 caller).

    A plain store: a release-ordered system store would write back L2 per call.
    """
    row = fx.Int32(row)
    row = (row < fx.Int32(TRACE_MAX_ROWS)).select(row, fx.Int32(TRACE_MAX_ROWS - 1))
    table = ptr_buf_tensor(fx.Int64(base), fx.Int64)
    table[row * fx.Int32(TRACE_FIELDS) + fx.Int32(field)] = fx.Int64(value)
