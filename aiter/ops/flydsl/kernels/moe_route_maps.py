# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort.

Computes topids_to_rows (route -> grouped row) and rows_to_tokens (inverse)
via per-expert atomicAdd. One thread per route, no host-side argsort.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops

BLOCK_THREADS = 256


def build_moe_route_maps_module():
    """JIT launcher: builds topids_to_rows and rows_to_tokens in one pass."""

    @flyc.kernel(name="moe_route_maps")
    def route_maps_kernel(
        topk_ids: fx.Tensor,  # (numel,) int32
        atomic_buffer: fx.Tensor,  # (E,) int32, init 0
        topids_to_rows: fx.Tensor,  # (numel,) int32 out: route -> grouped row
        rows_to_tokens: fx.Tensor,  # (E*max_m,) int32 out: grouped row -> token
        numel: Int32,
        topk: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            c_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)
            a_rsrc = buffer_ops.create_buffer_resource(rows_to_tokens, max_size=True)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)

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
            buffer_ops.buffer_store(row, c_rsrc, route)
            token = arith.divui(route, ArithValue(topk))
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
            topk_ids, atomic_buffer, topids_to_rows, rows_to_tokens, numel, topk, max_m
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_route_maps


def build_moe_topids_to_rows_module():
    """JIT launcher: builds topids_to_rows only (no rows_to_tokens inverse)."""

    @flyc.kernel(name="moe_route")
    def route_kernel(
        topk_ids: fx.Tensor,
        atomic_buffer: fx.Tensor,
        topids_to_rows: fx.Tensor,
        numel: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
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
            buffer_ops.buffer_store(row, out_rsrc, route)
            scf.YieldOp([])

    @flyc.jit
    def launch_topids_to_rows(
        topk_ids: fx.Tensor,
        atomic_buffer: fx.Tensor,
        topids_to_rows: fx.Tensor,
        numel: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        route_kernel(
            topk_ids, atomic_buffer, topids_to_rows, numel, max_m
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topids_to_rows
