# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Device-side pieces the conv3d kernels build on, as
``gemm_a16w16_gfx950_utils.py`` is for the GEMM.

What the machine fixes rather than what this operator chose: the MFMA shape,
the wave width, the vector widths a gfx950 load and an LDS write come in, and
the thin wrappers over the rocdl intrinsics that spell a barrier, a scalar
broadcast or a buffer atomic. The tile sizes, the barrier interval and the
padding modes are the algorithm's own and stay in ``conv3d_implicit_gfx950.py``.

Running a compiled launcher is a host concern and lives in
``../conv_kernels.py`` with the rest of the dispatch; only the stream coercion
both launchers take is here.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import T

# One MFMA instruction's output tile, and how many accumulator values of it a
# lane holds. gfx950's bf16 MFMA is 16x16x16 with 4 values per lane.
MFMA_M = 16
MFMA_N = 16
MFMA_C_VALUES = 4

WARP_SIZE = 64

BF16_BYTES = 2

# Elements per gather/DMA vector: 8 bf16 is the 16 bytes a buffer_load_lds
# moves per lane, and the width ds_write_b128 wants on the far side.
LDG_VEC = 8

# Compile hints applied to both conv3d kernels. Empty by default; a caller that
# needs to pass FlyDSL a hint sets it before the first compile.
CONV_COMPILE_HINTS = {}


def _as_stream(stream):
    return stream if hasattr(stream, "_is_stream_param") else fx.Stream(stream)


def buffer_atomic_add(vdata, rsrc, offset, soffset, aux):
    """Buffer-resource atomic fadd (AMD ``raw.ptr.buffer.atomic.fadd``).

    Upstream lives in flydsl's repo-level ``kernels/common/mem_ops.py``, which
    its wheel does not ship, so aiter keeps this one-line equivalent alongside
    the vendored ``buffer_ops`` / ``vector`` modules. Operates on a buffer
    resource plus byte offset, not an ``!llvm.ptr``.
    """
    return fx.rocdl.raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux)


def barrier(vmcnt=0, lgkmcnt=None):
    """Wait on the named counters, then barrier.

    Not gpu.barrier(): which counters this waits on is the whole point. The
    caller names only the ones it needs, so the DMAs prefetching the next K
    tiles stay in flight across the barrier instead of being drained by it.
    Naming a counter here is a scheduling decision.
    """
    fx.rocdl.s_waitcnt(vmcnt=vmcnt, lgkmcnt=lgkmcnt)
    fx.rocdl.s_barrier()


def sgpr(x):
    """Broadcast lane 0's value into a scalar register."""
    return fx.Int64(fx.rocdl.readfirstlane(T.i64, fx.Int64(x)))


def in_range(v, hi):
    return (v >= 0) & (v < fx.Int64(hi))


def dil(tap, factor):
    """Scale a filter tap by its dilation, folding the factor away when it is 1."""
    return tap * factor if const_expr(factor != 1) else tap


def gather_valid(base, *masks):
    """AND the masks that apply; a None mask is one the caller does not need."""
    for m in masks:
        if const_expr(m is not None):
            base = base & m
    return base
