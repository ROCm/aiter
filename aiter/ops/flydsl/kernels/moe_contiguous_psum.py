# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepGEMM-contiguous M-tile prefix sum (FlyDSL), single-block parallel scan.

Computes tile-aligned exclusive prefix sum of per-expert counts for the
contiguous grouped-GEMM scheduler. Single-block parallel scan replaces
torch.cumsum (avoids rocprim trampoline overhead for small E).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate, _to_raw as _raw

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf

from aiter.ops.flydsl.kernels.tensor_shim import (
    ptr_rsrc,
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

MAX_EXPERTS_PER_BLOCK = 512


@fx.struct
class _PsumStorage:
    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]


@fx.struct
class _RoutePsumStorage:
    cnt: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]


def _lds_load(ptr, idx):
    """Scalar i32 load from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    """Scalar i32 store to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, ptr + fx.Int64(idx))


def build_moe_contiguous_psum_module():
    """JIT launcher: tile-aligned prefix sum over per-expert counts."""

    @flyc.kernel(
        name="moe_contiguous_psum",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_kernel(
        masked_m: fx.Pointer,  # (E,) int32 in
        starts: fx.Pointer,  # (E,) int32 out
        psum: fx.Pointer,  # (E,) int32 out
        contiguous_m: fx.Pointer,  # (1,) int32 out
        experts: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        tile_v = ArithValue(tile_m)
        tile_minus_1 = tile_v - arith.constant(1, type=i32)

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        in_range = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(experts))
        _if_load = scf.IfOp(in_range)
        with ir.InsertionPoint(_if_load.then_block):
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            q = arith.divui(ArithValue(m) + tile_minus_1, tile_v)
            aligned = ArithValue(q) * tile_v
            _lds_store(lds0, aligned, tid)
            scf.YieldOp([])

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_range)
            with ir.InsertionPoint(_if_scan.then_block):
                val = _lds_load(src, tid)
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = _lds_load(src, tid - arith.constant(offset, type=i32))
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                _lds_store(dst, val + fx.Int32(prev_if.results[0]), tid)
                scf.YieldOp([])
            gpu.barrier()
            src, dst = dst, src

        _if_store = scf.IfOp(in_range)
        with ir.InsertionPoint(_if_store.then_block):
            is_first = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
            start_if = scf.IfOp(is_first, results_=[i32], has_else=True)
            with ir.InsertionPoint(start_if.then_block):
                scf.YieldOp([arith.constant(0, type=i32)])
            with ir.InsertionPoint(start_if.else_block):
                prev = _lds_load(src, tid - arith.constant(1, type=i32))
                scf.YieldOp([_raw(prev)])
            start = fx.Int32(start_if.results[0])
            m_tid = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + m_tid, p_rsrc, tid)

            is_last = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                ArithValue(experts) - arith.constant(1, type=i32),
            )
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                final_cur = _lds_load(src, tid)
                gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
                cm = arith.select(gt, _raw(final_cur), _raw(tile_v))
                buffer_ops.buffer_store(cm, c_rsrc, arith.constant(0, type=i32))
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_psum(
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        experts: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        psum_kernel(masked_m, starts, psum, contiguous_m, experts, tile_m).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum


def build_moe_contiguous_psum_remap_module():
    """JIT launcher: contiguous psum + in-place masked-to-contiguous row remap."""

    @flyc.kernel(
        name="moe_contiguous_psum_remap",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_remap_kernel(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: Int32,
        experts: Int32,
        route_max_m: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        tile_v = ArithValue(tile_m)
        tile_minus_1 = tile_v - arith.constant(1, type=i32)

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        m_rsrc = ptr_rsrc(masked_m)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        in_expert = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(experts))
        _if_load = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_load.then_block):
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            q = arith.divui(ArithValue(m) + tile_minus_1, tile_v)
            aligned = ArithValue(q) * tile_v
            _lds_store(lds0, aligned, tid)
            scf.YieldOp([])

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_expert)
            with ir.InsertionPoint(_if_scan.then_block):
                val = _lds_load(src, tid)
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = _lds_load(src, tid - arith.constant(offset, type=i32))
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                _lds_store(dst, val + fx.Int32(prev_if.results[0]), tid)
                scf.YieldOp([])
            gpu.barrier()
            src, dst = dst, src

        _if_store = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_store.then_block):
            is_first = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
            start_if = scf.IfOp(is_first, results_=[i32], has_else=True)
            with ir.InsertionPoint(start_if.then_block):
                scf.YieldOp([arith.constant(0, type=i32)])
            with ir.InsertionPoint(start_if.else_block):
                prev = _lds_load(src, tid - arith.constant(1, type=i32))
                scf.YieldOp([_raw(prev)])
            start = fx.Int32(start_if.results[0])
            m_tid = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + m_tid, p_rsrc, tid)
            is_last = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                ArithValue(experts) - arith.constant(1, type=i32),
            )
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                final_cur = _lds_load(src, tid)
                gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
                cm = arith.select(gt, _raw(final_cur), _raw(tile_v))
                buffer_ops.buffer_store(cm, c_rsrc, arith.constant(0, type=i32))
                scf.YieldOp([])
            scf.YieldOp([])

        gpu.barrier()

        tid_idx = arith.index_cast(T.index, tid)
        numel_idx = arith.index_cast(T.index, ArithValue(numel))
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)
        remap_loop = scf.ForOp(tid_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(remap_loop.body):
            route_i32 = arith.index_cast(i32, remap_loop.induction_variable)
            row = ArithValue(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            m = ArithValue(route_max_m)
            expert = ArithValue(arith.divui(row, m))
            slot = row - expert * m
            start = buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(ArithValue(start) + slot, rows_rsrc, route_i32)
            scf.YieldOp([])

    @flyc.jit
    def launch_psum_remap(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        route_max_m: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        psum_remap_kernel(
            masked_m,
            topids_to_rows,
            starts,
            psum,
            contiguous_m,
            numel,
            experts,
            route_max_m,
            tile_m,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum_remap.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum_remap


def build_moe_route_psum_fused_module():
    """JIT launcher: single-workgroup fused route + atomic + psum + remap.

    For small token counts every route fits in one workgroup, so the three
    pre-GEMM launches (route-maps, contiguous-psum, remap) collapse into one
    kernel. The per-expert atomic counter lives in LDS (workgroup-scope
    atomics, no global round-trip), and the tile-aligned prefix sum + in-place
    masked->contiguous row remap reuse the single-block scan below.

    Outputs match ``topids_to_rows`` (contiguous layout) + ``masked_m`` counts
    + ``psum`` (m_tile_map) of the split-kernel path bit-for-bit.
    """

    @flyc.kernel(
        name="moe_route_psum_fused",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def route_psum_fused_kernel(
        topk_ids: fx.Pointer,  # (numel,) i32 in
        topids_to_rows: fx.Pointer,  # (numel,) i32 out (contiguous rows)
        masked_m: fx.Pointer,  # (E,) i32 out (per-expert counts)
        starts: fx.Pointer,  # (E,) i32 out (contiguous row base per expert)
        psum: fx.Pointer,  # (E,) i32 out (= m_tile_map)
        numel: Int32,
        experts: Int32,
        max_m: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        tile_v = ArithValue(tile_m)
        tile_minus_1 = tile_v - arith.constant(1, type=i32)

        lds = fx.SharedAllocator().allocate(_RoutePsumStorage).peek()
        lds_cnt = lds.cnt.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        topk_rsrc = ptr_rsrc(topk_ids)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)

        in_expert = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(experts))

        # Phase A: zero the LDS per-expert atomic counter.
        _if_zero = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_zero.then_block):
            _lds_store(lds_cnt, arith.constant(0, type=i32), tid)
            scf.YieldOp([])
        gpu.barrier()

        # Phase B: route + workgroup-scope LDS atomic -> masked-layout rows.
        tid_idx = arith.index_cast(T.index, tid)
        numel_idx = arith.index_cast(T.index, ArithValue(numel))
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)
        route_loop = scf.ForOp(tid_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(route_loop.body):
            route_i32 = arith.index_cast(i32, route_loop.induction_variable)
            e = buffer_ops.buffer_load(topk_rsrc, route_i32, vec_width=1, dtype=i32)
            addr = fx.ptrtoint(lds_cnt + fx.Int64(e))
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            ptr = ptr._value if hasattr(ptr, "_value") else ptr
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            row = ArithValue(slot) + ArithValue(e) * ArithValue(max_m)
            buffer_ops.buffer_store(row, rows_rsrc, route_i32)
            scf.YieldOp([])
        gpu.barrier()

        # Phase C: tile-aligned inclusive scan of per-expert counts.
        _if_load = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_load.then_block):
            m = _lds_load(lds_cnt, tid)
            q = arith.divui(_raw(m) + tile_minus_1, tile_v)
            aligned = ArithValue(q) * tile_v
            _lds_store(lds0, aligned, tid)
            buffer_ops.buffer_store(m, m_rsrc, tid)
            scf.YieldOp([])
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_expert)
            with ir.InsertionPoint(_if_scan.then_block):
                val = _lds_load(src, tid)
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = _lds_load(src, tid - arith.constant(offset, type=i32))
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                _lds_store(dst, val + fx.Int32(prev_if.results[0]), tid)
                scf.YieldOp([])
            gpu.barrier()
            src, dst = dst, src

        _if_store = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_store.then_block):
            is_first = arith.cmpi(CmpIPredicate.eq, tid, arith.constant(0, type=i32))
            start_if = scf.IfOp(is_first, results_=[i32], has_else=True)
            with ir.InsertionPoint(start_if.then_block):
                scf.YieldOp([arith.constant(0, type=i32)])
            with ir.InsertionPoint(start_if.else_block):
                prev = _lds_load(src, tid - arith.constant(1, type=i32))
                scf.YieldOp([_raw(prev)])
            start = fx.Int32(start_if.results[0])
            m_tid = _lds_load(lds_cnt, tid)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + m_tid, p_rsrc, tid)
            scf.YieldOp([])
        gpu.barrier()

        # Phase D: in-place masked -> contiguous row remap.
        remap_loop = scf.ForOp(tid_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(remap_loop.body):
            route_i32 = arith.index_cast(i32, remap_loop.induction_variable)
            row = ArithValue(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            m = ArithValue(max_m)
            expert = ArithValue(arith.divui(row, m))
            slot = row - expert * m
            start = buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(ArithValue(start) + slot, rows_rsrc, route_i32)
            scf.YieldOp([])

    @flyc.jit
    def launch_route_psum_fused(
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        max_m: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        route_psum_fused_kernel(
            topk_ids,
            topids_to_rows,
            masked_m,
            starts,
            psum,
            numel,
            experts,
            max_m,
            tile_m,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_route_psum_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_route_psum_fused
