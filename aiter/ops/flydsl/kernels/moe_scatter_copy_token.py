# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route-gather (scatter-copy) kernel (FlyDSL).

Background
----------
Before the stage1 GEMM, each token's quantized payload (and per-token scale)
must be copied from the flat per-token layout into the grouped per-expert
layout the masked grouped kernel consumes::

    for e in range(E):
        toks = tokens routed to expert e          # n = counts[e] of them
        grouped[e, :n] = a_payload[toks]          # copy each token's row

This kernel does that copy in one pass. Each destination grouped row is mapped
to its source token row via a precomputed ``dst_src`` map (``-1`` => leave the
destination untouched, i.e. an unused padding slot). One thread-block per
destination row copies the whole row; threads stride over the row's elements.

The copy is byte-exact. We pick the widest vectorized buffer op whose access
size divides ``row_bytes`` (which keeps every row base naturally aligned), using
the same fallback ladder as the MoE GEMM X-load path:

    row_bytes % 16 == 0 -> dwordx4 (16B / vec4 i32)   -- fastest
    row_bytes %  8 == 0 -> dwordx2 ( 8B / vec2 i32)
    row_bytes %  4 == 0 -> dword   ( 4B / scalar i32)
    otherwise           -> byte    ( 1B / scalar i8)   -- e.g. a 90B scale row

Source and destination rows share the same byte width.

Grid  : (num_dst_rows, 1, 1)   -- num_dst_rows = E * max_m
Block : (BLOCK_THREADS, 1, 1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import buffer_ops

from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc, MOE_KERNARG_PRELOAD_COUNT

BLOCK_THREADS = 256


def build_moe_scatter_copy_token_module(row_bytes: int):
    """Return a JIT launcher that gathers source rows into destination rows.

    Parameters
    ----------
    row_bytes : int   byte width of one row (same for src and dst).

    Launcher signature: ``(src, dst, dst_src, num_dst, stream=...)`` where
    ``src``/``dst`` are uint8 tensors viewed as (rows, row_bytes) and
    ``dst_src`` is an int32 (num_dst,) map from dst row -> src row (-1 to skip).
    """
    assert row_bytes > 0
    # Fallback ladder (mirrors moe_gemm_2stage X-load): pick the widest buffer op
    # whose access size divides row_bytes. Because each width divides row_bytes,
    # every row base (row * row_bytes) is naturally aligned for that width, and the
    # row is covered by a whole number of units with no in-row remainder.
    if row_bytes % 16 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 4, 16, True, "dwx4"
    elif row_bytes % 8 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 2, 8, True, "dwx2"
    elif row_bytes % 4 == 0:
        vec_width, unit_bytes, use_dword, width_tag = 1, 4, True, "dw"
    else:
        vec_width, unit_bytes, use_dword, width_tag = 1, 1, False, "by"

    # Number of vectorized copy units (buffer ops) per row, and the per-row stride
    # measured in copy-dtype elements (i32 when dword*, i8 otherwise).
    n_units = row_bytes // unit_bytes
    row_stride_elems = row_bytes // 4 if use_dword else row_bytes

    module_name = f"moe_scatter_copy_token_b{row_bytes}_{width_tag}"

    @flyc.kernel(name=module_name)
    def scatter_copy_kernel(
        src: fx.Pointer,  # (num_src, row_bytes) uint8
        dst: fx.Pointer,  # (num_dst, row_bytes) uint8
        dst_src: fx.Pointer,  # (num_dst,) int32  -- src row per dst row, -1=skip
        num_dst: Int32,
    ):
        i32 = T.i32
        cdt = T.i32 if use_dword else T.i8

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        num_dst_i32 = ArithValue(num_dst)
        bid_i32 = ArithValue(bid)
        n_units_i32 = arith.constant(n_units, type=i32)
        row_stride_i32 = arith.constant(row_stride_elems, type=i32)

        dst_valid = arith.cmpi(CmpIPredicate.ult, bid_i32, num_dst_i32)
        _if_dst = scf.IfOp(dst_valid)
        with ir.InsertionPoint(_if_dst.then_block):
            map_rsrc = ptr_rsrc(dst_src)
            srow = ArithValue(
                buffer_ops.buffer_load(map_rsrc, bid_i32, vec_width=1, dtype=i32)
            )
            row_ok = arith.cmpi(CmpIPredicate.sge, srow, arith.constant(0, type=i32))
            _if_row = scf.IfOp(row_ok)
            with ir.InsertionPoint(_if_row.then_block):
                src_rsrc = ptr_rsrc(src)
                dst_rsrc = ptr_rsrc(dst)
                tid_i32 = ArithValue(tid)
                # Element base of each row, in copy-dtype elements (i32 if dword*, i8 otherwise).
                src_base = srow * row_stride_i32
                dst_base = bid_i32 * row_stride_i32

                for it in range_constexpr(
                    (n_units + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    # Each unit copies `vec_width` copy-dtype elements via one buffer op
                    # (dwordx4 / dwordx2 / dword / byte). Threads stride over the row's units.
                    uidx = tid_i32 + arith.constant(it * BLOCK_THREADS, type=i32)
                    u_ok = arith.cmpi(CmpIPredicate.ult, uidx, n_units_i32)
                    _if_u = scf.IfOp(u_ok)
                    with ir.InsertionPoint(_if_u.then_block):
                        # vec_width is a build-time constant, so resolve the unit->element
                        # offset with a Python ternary (avoid a kernel-level `if` statement).
                        eidx = (
                            uidx
                            if vec_width == 1
                            else uidx * arith.constant(vec_width, type=i32)
                        )
                        v = buffer_ops.buffer_load(
                            src_rsrc,
                            src_base + eidx,
                            vec_width=vec_width,
                            dtype=cdt,
                        )
                        buffer_ops.buffer_store(v, dst_rsrc, dst_base + eidx)
                        scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_scatter_copy(
        src: fx.Pointer,
        dst: fx.Pointer,
        dst_src: fx.Pointer,
        num_dst: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_dst = arith.index_cast(T.index, num_dst)
        launcher = scatter_copy_kernel(src, dst, dst_src, num_dst)
        launcher.launch(
            grid=(idx_dst, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_scatter_copy.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_scatter_copy
