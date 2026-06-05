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

The copy is byte-exact: rows whose byte width is a multiple of 4 are copied as
i32 dwords (fast); otherwise (e.g. a per-32 scale row of 90 bytes) they are
copied as i8 bytes. Source and destination rows share the same byte width.

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
    # i32 dword copy when the row is 4-byte aligned, else i8 byte copy.
    if row_bytes % 4 == 0:
        use_dword = True
        n_elems = row_bytes // 4
    else:
        use_dword = False
        n_elems = row_bytes

    @flyc.kernel
    def scatter_copy_kernel(
        src: fx.Tensor,      # (num_src, row_bytes) uint8
        dst: fx.Tensor,      # (num_dst, row_bytes) uint8
        dst_src: fx.Tensor,  # (num_dst,) int32  -- src row per dst row, -1=skip
        num_dst: Int32,
    ):
        i32 = T.i32
        cdt = T.i32 if use_dword else T.i8

        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        num_dst_i32 = ArithValue(num_dst)
        bid_i32 = ArithValue(bid)
        n_elems_i32 = arith.constant(n_elems, type=i32)

        dst_valid = arith.cmpi(CmpIPredicate.ult, bid_i32, num_dst_i32)
        _if_dst = scf.IfOp(dst_valid)
        with ir.InsertionPoint(_if_dst.then_block):
            map_rsrc = buffer_ops.create_buffer_resource(dst_src, max_size=True)
            srow = ArithValue(
                buffer_ops.buffer_load(map_rsrc, bid_i32, vec_width=1, dtype=i32)
            )
            row_ok = arith.cmpi(
                CmpIPredicate.sge, srow, arith.constant(0, type=i32)
            )
            _if_row = scf.IfOp(row_ok)
            with ir.InsertionPoint(_if_row.then_block):
                src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
                dst_rsrc = buffer_ops.create_buffer_resource(dst, max_size=True)
                tid_i32 = ArithValue(tid)
                # Element base of each row (i32 elems if dword, bytes if i8).
                src_base = srow * n_elems_i32
                dst_base = bid_i32 * n_elems_i32

                for it in range_constexpr(
                    (n_elems + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    eidx = tid_i32 + arith.constant(it * BLOCK_THREADS, type=i32)
                    e_ok = arith.cmpi(CmpIPredicate.ult, eidx, n_elems_i32)
                    _if_e = scf.IfOp(e_ok)
                    with ir.InsertionPoint(_if_e.then_block):
                        v = buffer_ops.buffer_load(
                            src_rsrc, src_base + eidx, vec_width=1, dtype=cdt
                        )
                        buffer_ops.buffer_store(v, dst_rsrc, dst_base + eidx)
                        scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_scatter_copy(
        src: fx.Tensor,
        dst: fx.Tensor,
        dst_src: fx.Tensor,
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

    return launch_scatter_copy
