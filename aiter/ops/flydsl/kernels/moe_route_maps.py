# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort.

Computes two maps for the grouped MoE layout:
  topids_to_rows : route ``i = t*topk + k`` -> grouped-GEMM row that token ``t``'s
                   k-th expert occupies = ``expert*max_m + slot``
  rows_to_tokens : its inverse, grouped row -> source token

The within-expert ``slot`` is claimed via a global ``atomicAdd(1)`` on a
per-expert counter initialized to 0; the kernel forms the grouped row in-place
as ``slot + e*max_m`` -- no host-side argsort / nonzero / one-hot, and no
host-side offset array (the final counter is ``counts[e]`` directly).

One thread per route (grid covers ceil(numel/BLOCK)). The within-expert order is
atomic-race order (nondeterministic), which is fine: scatter-copy and
gather-reduce both key off the *same* topids_to_rows, and the grouped GEMM is
order-agnostic within ``[0, counts[e])``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr import const_expr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops

BLOCK_THREADS = 256


def build_moe_route_maps_module(input_i64=False):
    """Return a JIT launcher computing the route maps in one pass.

    Args:
        input_i64: When True, ``topk_ids`` is int64 and the kernel truncates
            to int32 internally. When False (default), ``topk_ids`` must be
            int32.

    Launcher: ``(topk_ids, atomic_buffer, topids_to_rows, rows_to_tokens,
    numel, topk, max_m, grid_blocks, stream=...)``

      topk_ids       : (numel,)   int32/int64  flattened expert ids
      atomic_buffer  : (E,)       int32  per-expert counter (pre-zeroed by caller)
      topids_to_rows : (numel,)   int32  out: grouped row per route
      rows_to_tokens : (E*max_m,) int32  out: source token per grouped row
                       (pre-filled with -1 by caller; kernel overwrites valid rows)
    """

    @flyc.kernel(name="moe_route_maps")
    def route_maps_kernel(
        topk_ids: fx.Tensor,
        atomic_buffer: fx.Tensor,
        topids_to_rows: fx.Tensor,
        rows_to_tokens: fx.Tensor,
        numel: Int32,
        topk: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)

        in_range = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)
            a_rsrc = buffer_ops.create_buffer_resource(rows_to_tokens, max_size=True)

            if const_expr(input_i64):
                e_raw = buffer_ops.buffer_load(topk_rsrc, tid, vec_width=1, dtype=T.i64)
                e = arith.trunci(i32, e_raw)
            else:
                e = buffer_ops.buffer_load(topk_rsrc, tid, vec_width=1, dtype=i32)

            base_idx = buffer_ops.extract_base_index(atomic_buffer, address_space=1)
            e_idx = arith.index_cast(T.index, e)
            addr = fx.Index(base_idx) + fx.Index(e_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            ptr = ptr._value if hasattr(ptr, "_value") else ptr

            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

            row = ArithValue(slot) + ArithValue(e) * ArithValue(max_m)

            buffer_ops.buffer_store(row, c_rsrc, tid)
            token = arith.divui(tid, ArithValue(topk))
            buffer_ops.buffer_store(token, a_rsrc, row)
            scf.YieldOp([])

    @flyc.jit
    def launch_route_maps(
        topk_ids: fx.Tensor,
        atomic_buffer: fx.Tensor,
        topids_to_rows: fx.Tensor,
        rows_to_tokens: fx.Tensor,
        numel: fx.Int32,
        topk: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.index_cast(T.index, grid_blocks)
        route_maps_kernel(
            topk_ids, atomic_buffer, topids_to_rows, rows_to_tokens,
            numel, topk, max_m
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_route_maps
