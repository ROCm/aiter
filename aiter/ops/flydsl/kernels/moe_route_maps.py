# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE route -> grouped-row map kernel (FlyDSL), atomic-scatter argsort.

Computes topids_to_rows (route -> grouped row) and rows_to_tokens (inverse)
via per-expert atomicAdd. One thread per route, no host-side argsort.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, ptrtoint
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate, _to_raw as _raw
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels.tensor_shim import (
    STensor,
    ptr_rsrc,
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

BLOCK_THREADS = 256
MAX_EXPERTS_PER_BLOCK = 512


def build_moe_route_maps_module():
    """JIT launcher: builds topids_to_rows and rows_to_tokens in one pass."""

    @flyc.kernel(name="moe_route_maps")
    def route_maps_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32
        atomic_buffer: fx.Pointer,  # (E,) int32, init 0
        topids_to_rows: fx.Pointer,  # (numel,) int32 out: route -> grouped row
        rows_to_tokens: fx.Pointer,  # (E*max_m,) int32 out: grouped row -> token
        numel: Int32,
        topk: Int32,
        experts: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = ptr_rsrc(topk_ids)
            c_rsrc = ptr_rsrc(topids_to_rows)
            a_rsrc = ptr_rsrc(rows_to_tokens)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
            ge0 = arith.cmpi(CmpIPredicate.sge, e, arith.constant(0, type=i32))
            ltE = arith.cmpi(CmpIPredicate.slt, e, ArithValue(experts))
            valid_e = arith.andi(ge0, ltE)

            _if_valid = scf.IfOp(valid_e, has_else=True)
            with ir.InsertionPoint(_if_valid.then_block):
                base_idx = arith.index_cast(T.index, ptrtoint(atomic_buffer))
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
            with ir.InsertionPoint(_if_valid.else_block):
                buffer_ops.buffer_store(arith.constant(-1, type=i32), c_rsrc, route)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_route_maps(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        rows_to_tokens: fx.Pointer,
        numel: fx.Int32,
        topk: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.index_cast(T.index, grid_blocks)
        launch = route_maps_kernel(
            topk_ids,
            atomic_buffer,
            topids_to_rows,
            rows_to_tokens,
            numel,
            topk,
            experts,
            max_m,
        )
        launch.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_route_maps.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_maps


def build_moe_topids_to_rows_module():
    """JIT launcher: builds topids_to_rows only (no rows_to_tokens inverse)."""

    @flyc.kernel(name="moe_route")
    def route_kernel(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: Int32,
        experts: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + ArithValue(fx.thread_idx.x)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if = scf.IfOp(in_range)
        with ir.InsertionPoint(_if.then_block):
            topk_rsrc = ptr_rsrc(topk_ids)
            out_rsrc = ptr_rsrc(topids_to_rows)

            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
            ge0 = arith.cmpi(CmpIPredicate.sge, e, arith.constant(0, type=i32))
            ltE = arith.cmpi(CmpIPredicate.slt, e, ArithValue(experts))
            valid_e = arith.andi(ge0, ltE)
            _if_valid = scf.IfOp(valid_e, has_else=True)
            with ir.InsertionPoint(_if_valid.then_block):
                base_idx = arith.index_cast(T.index, ptrtoint(atomic_buffer))
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
            with ir.InsertionPoint(_if_valid.else_block):
                buffer_ops.buffer_store(arith.constant(-1, type=i32), out_rsrc, route)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_topids_to_rows(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gx = arith.index_cast(T.index, grid_blocks)
        launch = route_kernel(
            topk_ids, atomic_buffer, topids_to_rows, numel, experts, max_m
        )
        launch.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_topids_to_rows.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_topids_to_rows


def build_moe_topids_to_rows_blockagg_module():
    """JIT launcher: topids_to_rows with per-block LDS count aggregation.

    Reduces global atomics from one per route to at most one per (block, expert).
    The kernel name is versioned to avoid reusing stale compiled artifacts from
    earlier experiments.
    """

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="moe_route_blockagg_v2_smem"
    )
    count_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = count_off + MAX_EXPERTS_PER_BLOCK * 4
    base_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = base_off + MAX_EXPERTS_PER_BLOCK * 4

    @flyc.kernel(name="moe_route_blockagg_v2", known_block_size=[BLOCK_THREADS, 1, 1])
    def route_kernel(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: Int32,
        experts: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)

        lds_base = allocator.get_base()
        lds_count = STensor(
            SmemPtr(lds_base, count_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds_row_base = STensor(
            SmemPtr(lds_base, base_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )

        tid = ArithValue(fx.thread_idx.x)
        route = ArithValue(fx.block_idx.x) * arith.constant(
            BLOCK_THREADS, type=i32
        ) + tid

        tid_idx = arith.index_cast(T.index, tid)
        experts_idx = arith.index_cast(T.index, ArithValue(experts))
        stride_idx = arith.index(BLOCK_THREADS)

        zero_loop = scf.ForOp(tid_idx, experts_idx, stride_idx)
        with ir.InsertionPoint(zero_loop.body):
            e_i32 = arith.index_cast(i32, zero_loop.induction_variable)
            lds_count[fx.Index(e_i32)] = c0
            lds_row_base[fx.Index(e_i32)] = c0
            scf.YieldOp([])
        gpu.barrier()

        topk_rsrc = ptr_rsrc(topk_ids)
        out_rsrc = ptr_rsrc(topids_to_rows)
        in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        route_if = scf.IfOp(in_range, results_=[i32, i32], has_else=True)
        with ir.InsertionPoint(route_if.then_block):
            e = buffer_ops.buffer_load(topk_rsrc, route, vec_width=1, dtype=i32)
            ge0 = arith.cmpi(CmpIPredicate.sge, e, c0)
            ltE = arith.cmpi(CmpIPredicate.slt, e, ArithValue(experts))
            valid_e = arith.andi(ge0, ltE)
            safe_e = arith.select(valid_e, _raw(e), c0)
            valid_i32 = arith.select(valid_e, c1, c0)
            scf.YieldOp([safe_e, valid_i32])
        with ir.InsertionPoint(route_if.else_block):
            scf.YieldOp([c0, c0])

        expert = ArithValue(route_if.results[0])
        route_valid_i32 = ArithValue(route_if.results[1])
        route_valid = arith.cmpi(CmpIPredicate.ne, route_valid_i32, c0)

        count_base_idx = buffer_ops.extract_base_index(lds_base)
        local_slot_if = scf.IfOp(route_valid, results_=[i32], has_else=True)
        with ir.InsertionPoint(local_slot_if.then_block):
            expert_idx = arith.index_cast(T.index, expert)
            addr = (
                fx.Index(count_base_idx)
                + fx.Index(count_off)
                + fx.Index(expert_idx) * fx.Index(4)
            )
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            ptr = ptr._value if hasattr(ptr, "_value") else ptr
            local_slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                c1,
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            scf.YieldOp([local_slot])
        with ir.InsertionPoint(local_slot_if.else_block):
            scf.YieldOp([c0])
        gpu.barrier()

        global_base_idx = arith.index_cast(T.index, ptrtoint(atomic_buffer))
        publish_loop = scf.ForOp(tid_idx, experts_idx, stride_idx)
        with ir.InsertionPoint(publish_loop.body):
            e_i32 = arith.index_cast(i32, publish_loop.induction_variable)
            cnt = lds_count[fx.Index(e_i32)]
            has_count = arith.cmpi(CmpIPredicate.ugt, cnt, c0)
            publish_if = scf.IfOp(has_count, results_=[i32], has_else=True)
            with ir.InsertionPoint(publish_if.then_block):
                e_idx = publish_loop.induction_variable
                addr = fx.Index(global_base_idx) + fx.Index(e_idx) * fx.Index(4)
                ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
                ptr = ptr._value if hasattr(ptr, "_value") else ptr
                block_base = llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    ptr,
                    _raw(cnt),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
                scf.YieldOp([block_base])
            with ir.InsertionPoint(publish_if.else_block):
                scf.YieldOp([c0])
            lds_row_base[fx.Index(e_i32)] = ArithValue(publish_if.results[0])
            scf.YieldOp([])
        gpu.barrier()

        store_if = scf.IfOp(route_valid, has_else=True)
        with ir.InsertionPoint(store_if.then_block):
            row_base = lds_row_base[fx.Index(expert)]
            row = (
                ArithValue(expert) * ArithValue(max_m)
                + ArithValue(row_base)
                + ArithValue(local_slot_if.results[0])
            )
            buffer_ops.buffer_store(row, out_rsrc, route)
            scf.YieldOp([])
        with ir.InsertionPoint(store_if.else_block):
            range_if = scf.IfOp(in_range)
            with ir.InsertionPoint(range_if.then_block):
                buffer_ops.buffer_store(c0 - c1, out_rsrc, route)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_topids_to_rows_blockagg(
        topk_ids: fx.Pointer,
        atomic_buffer: fx.Pointer,
        topids_to_rows: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx = arith.index_cast(T.index, grid_blocks)
        route_kernel(
            topk_ids, atomic_buffer, topids_to_rows, numel, experts, max_m
        ).launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_topids_to_rows_blockagg.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_topids_to_rows_blockagg
