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
from flydsl.expr.arith import ArithValue, CmpIPredicate, CmpFPredicate, _to_raw as _raw
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from aiter.ops.flydsl.kernels.tensor_shim import (
    STensor,
    ptr_rsrc,
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

MAX_EXPERTS_PER_BLOCK = 512


def build_moe_contiguous_psum_module():
    """JIT launcher: tile-aligned prefix sum over per-expert counts."""

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="moe_contiguous_psum_smem"
    )
    lds0_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds0_off + MAX_EXPERTS_PER_BLOCK * 4
    lds1_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds1_off + MAX_EXPERTS_PER_BLOCK * 4

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

        lds_base = allocator.get_base()
        lds0 = STensor(
            SmemPtr(lds_base, lds0_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds1 = STensor(
            SmemPtr(lds_base, lds1_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )

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
            lds0[fx.Index(tid)] = aligned
            scf.YieldOp([])

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_range)
            with ir.InsertionPoint(_if_scan.then_block):
                val = src[fx.Index(tid)]
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = src[fx.Index(tid - arith.constant(offset, type=i32))]
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                dst[fx.Index(tid)] = ArithValue(val) + ArithValue(prev_if.results[0])
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
                prev = src[fx.Index(tid - arith.constant(1, type=i32))]
                scf.YieldOp([_raw(prev)])
            start = ArithValue(start_if.results[0])
            m_tid = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + ArithValue(m_tid), p_rsrc, tid)

            is_last = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                ArithValue(experts) - arith.constant(1, type=i32),
            )
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                final_cur = src[fx.Index(tid)]
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
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
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

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="moe_contiguous_psum_remap_smem"
    )
    lds0_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds0_off + MAX_EXPERTS_PER_BLOCK * 4
    lds1_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds1_off + MAX_EXPERTS_PER_BLOCK * 4

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
        num_valid_routes: fx.Pointer,  # (1,) int32: only remap routes < this (EP dead-tail skip)
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        tile_v = ArithValue(tile_m)
        tile_minus_1 = tile_v - arith.constant(1, type=i32)

        lds_base = allocator.get_base()
        lds0 = STensor(
            SmemPtr(lds_base, lds0_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds1 = STensor(
            SmemPtr(lds_base, lds1_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )

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
            lds0[fx.Index(tid)] = aligned
            scf.YieldOp([])

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_expert)
            with ir.InsertionPoint(_if_scan.then_block):
                val = src[fx.Index(tid)]
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = src[fx.Index(tid - arith.constant(offset, type=i32))]
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                dst[fx.Index(tid)] = ArithValue(val) + ArithValue(prev_if.results[0])
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
                prev = src[fx.Index(tid - arith.constant(1, type=i32))]
                scf.YieldOp([_raw(prev)])
            start = ArithValue(start_if.results[0])
            m_tid = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + ArithValue(m_tid), p_rsrc, tid)
            is_last = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                ArithValue(experts) - arith.constant(1, type=i32),
            )
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                final_cur = src[fx.Index(tid)]
                gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
                cm = arith.select(gt, _raw(final_cur), _raw(tile_v))
                buffer_ops.buffer_store(cm, c_rsrc, arith.constant(0, type=i32))
                scf.YieldOp([])
            scf.YieldOp([])

        gpu.barrier()

        # Only remap valid routes ([0, nvr)); dead-tail routes (>= num_valid_routes)
        # hold unwritten/garbage rows from the route kernel and must NOT be used as
        # a row index (would OOB-read starts[expert]). They are never read
        # downstream. When truncation is disabled the caller passes numel == nvr.
        nvr = ArithValue(
            buffer_ops.buffer_load(
                ptr_rsrc(num_valid_routes),
                arith.constant(0, type=i32),
                vec_width=1,
                dtype=i32,
            )
        )
        tid_idx = arith.index_cast(T.index, tid)
        nvr_idx = arith.index_cast(T.index, ArithValue(nvr))
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)
        remap_loop = scf.ForOp(tid_idx, nvr_idx, stride_idx)
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
        num_valid_routes: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
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
            num_valid_routes,
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


def build_moe_contiguous_psum_remap_ep_module():
    """psum + masked->contiguous remap, FUSED with the gemm2 EP ep_rowmap build.

    Identical prefix-sum + in-place row remap as ``build_moe_contiguous_psum_remap
    _module``, but the single remap pass ALSO scatters, for each valid local route
    (gather_w != 0), the packed dest (origin_pe*slot_stride + origin_lid*topk + k)
    + f32-weight-bits into ep_rowmap[final_contiguous_row]. This folds the separate
    ``moe_build_ep_rowmap`` launch (Opportunity A) into the remap it already runs,
    reusing the final row it just computed (one fewer kernel + no re-read of rows).
    ep_rowmap is a flat (cap_rows+1, 2) i32 tensor; rows untouched by a valid route
    keep the -1 sentinel written in the init loop below.
    """

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="moe_contiguous_psum_remap_ep_smem"
    )
    lds0_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds0_off + MAX_EXPERTS_PER_BLOCK * 4
    lds1_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds1_off + MAX_EXPERTS_PER_BLOCK * 4

    @flyc.kernel(
        name="moe_contiguous_psum_remap_ep",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def psum_remap_ep_kernel(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: Int32,
        experts: Int32,
        route_max_m: Int32,
        tile_m: Int32,
        num_valid_routes: fx.Pointer,
        gather_w: fx.Pointer,   # (numel,) bf16, 0 for dropped/remote
        tis: fx.Pointer,        # (recv_cap,) i32 recv_slot -> origin enc
        ep_rowmap: fx.Pointer,  # (cap_rows+1, 2) i32 flat out
        cap_rows: Int32,
        topk: Int32,
        max_tok: Int32,
        slot_stride: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        tile_v = ArithValue(tile_m)
        tile_minus_1 = tile_v - arith.constant(1, type=i32)

        lds_base = allocator.get_base()
        lds0 = STensor(
            SmemPtr(lds_base, lds0_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds1 = STensor(
            SmemPtr(lds_base, lds1_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )

        m_rsrc = ptr_rsrc(masked_m)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)
        w_rsrc = ptr_rsrc(gather_w)
        tis_rsrc = ptr_rsrc(tis)
        ep_rsrc = ptr_rsrc(ep_rowmap)

        neg1 = arith.constant(-1, type=i32)
        zero = arith.constant(0, type=i32)
        one = arith.constant(1, type=i32)
        two = arith.constant(2, type=i32)

        in_expert = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(experts))
        _if_load = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_load.then_block):
            m = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            q = arith.divui(ArithValue(m) + tile_minus_1, tile_v)
            aligned = ArithValue(q) * tile_v
            lds0[fx.Index(tid)] = aligned
            scf.YieldOp([])

        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_expert)
            with ir.InsertionPoint(_if_scan.then_block):
                val = src[fx.Index(tid)]
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = src[fx.Index(tid - arith.constant(offset, type=i32))]
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                dst[fx.Index(tid)] = ArithValue(val) + ArithValue(prev_if.results[0])
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
                prev = src[fx.Index(tid - arith.constant(1, type=i32))]
                scf.YieldOp([_raw(prev)])
            start = ArithValue(start_if.results[0])
            m_tid = buffer_ops.buffer_load(m_rsrc, tid, vec_width=1, dtype=i32)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + ArithValue(m_tid), p_rsrc, tid)
            is_last = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                ArithValue(experts) - arith.constant(1, type=i32),
            )
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                final_cur = src[fx.Index(tid)]
                gt = arith.cmpi(CmpIPredicate.sgt, final_cur, tile_v)
                cm = arith.select(gt, _raw(final_cur), _raw(tile_v))
                buffer_ops.buffer_store(cm, c_rsrc, arith.constant(0, type=i32))
                scf.YieldOp([])
            scf.YieldOp([])

        gpu.barrier()

        tid_idx = arith.index_cast(T.index, tid)
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)

        # EP Phase 1: init ep_rowmap[0:cap_rows] = (-1, 0) before any scatter.
        cap_idx = arith.index_cast(T.index, ArithValue(cap_rows))
        init_loop = scf.ForOp(tid_idx, cap_idx, stride_idx)
        with ir.InsertionPoint(init_loop.body):
            row_i32 = arith.index_cast(i32, init_loop.induction_variable)
            base = ArithValue(row_i32) * ArithValue(two)
            buffer_ops.buffer_store(neg1, ep_rsrc, _raw(base))
            buffer_ops.buffer_store(zero, ep_rsrc, _raw(base + ArithValue(one)))
            scf.YieldOp([])
        gpu.barrier()

        nvr = ArithValue(
            buffer_ops.buffer_load(
                ptr_rsrc(num_valid_routes),
                arith.constant(0, type=i32),
                vec_width=1,
                dtype=i32,
            )
        )
        nvr_idx = arith.index_cast(T.index, ArithValue(nvr))
        topk_v = ArithValue(topk)
        max_tok_v = ArithValue(max_tok)
        slot_stride_v = ArithValue(slot_stride)
        # Fused remap + ep_rowmap scatter over valid routes ([0, nvr)).
        remap_loop = scf.ForOp(tid_idx, nvr_idx, stride_idx)
        with ir.InsertionPoint(remap_loop.body):
            route_i32 = arith.index_cast(i32, remap_loop.induction_variable)
            row = ArithValue(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            m = ArithValue(route_max_m)
            expert = ArithValue(arith.divui(row, m))
            slot = row - expert * m
            start = buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
            final_row = ArithValue(start) + slot
            buffer_ops.buffer_store(final_row, rows_rsrc, route_i32)
            # ep_rowmap scatter: only kept local routes (gather_w != 0).
            route = ArithValue(route_i32)
            w_bf = buffer_ops.buffer_load(w_rsrc, route_i32, vec_width=1, dtype=T.bf16)
            w_f32 = ArithValue(w_bf).extf(T.f32)
            is_kept = arith.cmpf(
                CmpFPredicate.ONE, _raw(w_f32), arith.constant(0.0, type=T.f32)
            )
            kept_if = scf.IfOp(is_kept)
            with ir.InsertionPoint(kept_if.then_block):
                t = ArithValue(arith.divui(_raw(route), _raw(topk_v)))
                k = route - t * topk_v
                enc = ArithValue(
                    buffer_ops.buffer_load(tis_rsrc, _raw(t), vec_width=1, dtype=i32)
                )
                origin_pe = ArithValue(arith.divui(_raw(enc), _raw(max_tok_v)))
                origin_lid = enc - origin_pe * max_tok_v
                packed = origin_pe * slot_stride_v + origin_lid * topk_v + k
                w_bits = ArithValue(w_f32).bitcast(T.i32)
                ep_base = final_row * ArithValue(two)
                buffer_ops.buffer_store(_raw(packed), ep_rsrc, _raw(ep_base))
                buffer_ops.buffer_store(
                    _raw(w_bits), ep_rsrc, _raw(ep_base + ArithValue(one))
                )
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_psum_remap_ep(
        masked_m: fx.Pointer,
        topids_to_rows: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        route_max_m: fx.Int32,
        tile_m: fx.Int32,
        num_valid_routes: fx.Pointer,
        gather_w: fx.Pointer,
        tis: fx.Pointer,
        ep_rowmap: fx.Pointer,
        cap_rows: fx.Int32,
        topk: fx.Int32,
        max_tok: fx.Int32,
        slot_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        psum_remap_ep_kernel(
            masked_m,
            topids_to_rows,
            starts,
            psum,
            contiguous_m,
            numel,
            experts,
            route_max_m,
            tile_m,
            num_valid_routes,
            gather_w,
            tis,
            ep_rowmap,
            cap_rows,
            topk,
            max_tok,
            slot_stride,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_psum_remap_ep.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_psum_remap_ep


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

    gpu_arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="moe_route_psum_fused_smem"
    )
    cnt_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = cnt_off + MAX_EXPERTS_PER_BLOCK * 4
    lds0_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds0_off + MAX_EXPERTS_PER_BLOCK * 4
    lds1_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds1_off + MAX_EXPERTS_PER_BLOCK * 4

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

        lds_base = allocator.get_base()
        lds_cnt = STensor(
            SmemPtr(lds_base, cnt_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds0 = STensor(
            SmemPtr(lds_base, lds0_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )
        lds1 = STensor(
            SmemPtr(lds_base, lds1_off, T.i32, shape=(MAX_EXPERTS_PER_BLOCK,)),
            dtype=T.i32,
            shape=(MAX_EXPERTS_PER_BLOCK,),
        )

        topk_rsrc = ptr_rsrc(topk_ids)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)

        in_expert = arith.cmpi(CmpIPredicate.ult, tid, ArithValue(experts))

        # Phase A: zero the LDS per-expert atomic counter.
        _if_zero = scf.IfOp(in_expert)
        with ir.InsertionPoint(_if_zero.then_block):
            lds_cnt[fx.Index(tid)] = arith.constant(0, type=i32)
            scf.YieldOp([])
        gpu.barrier()

        # Phase B: route + workgroup-scope LDS atomic -> masked-layout rows.
        cnt_base_idx = buffer_ops.extract_base_index(lds_base)
        tid_idx = arith.index_cast(T.index, tid)
        numel_idx = arith.index_cast(T.index, ArithValue(numel))
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)
        route_loop = scf.ForOp(tid_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(route_loop.body):
            route_i32 = arith.index_cast(i32, route_loop.induction_variable)
            e = buffer_ops.buffer_load(topk_rsrc, route_i32, vec_width=1, dtype=i32)
            e_idx = arith.index_cast(T.index, e)
            addr = (
                fx.Index(cnt_base_idx)
                + fx.Index(cnt_off)
                + fx.Index(e_idx) * fx.Index(4)
            )
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
            m = lds_cnt[fx.Index(tid)]
            q = arith.divui(ArithValue(m) + tile_minus_1, tile_v)
            aligned = ArithValue(q) * tile_v
            lds0[fx.Index(tid)] = aligned
            buffer_ops.buffer_store(ArithValue(m), m_rsrc, tid)
            scf.YieldOp([])
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            _if_scan = scf.IfOp(in_expert)
            with ir.InsertionPoint(_if_scan.then_block):
                val = src[fx.Index(tid)]
                has_prev = arith.cmpi(
                    CmpIPredicate.uge, tid, arith.constant(offset, type=i32)
                )
                prev_if = scf.IfOp(has_prev, results_=[i32], has_else=True)
                with ir.InsertionPoint(prev_if.then_block):
                    prev = src[fx.Index(tid - arith.constant(offset, type=i32))]
                    scf.YieldOp([_raw(prev)])
                with ir.InsertionPoint(prev_if.else_block):
                    scf.YieldOp([arith.constant(0, type=i32)])
                dst[fx.Index(tid)] = ArithValue(val) + ArithValue(prev_if.results[0])
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
                prev = src[fx.Index(tid - arith.constant(1, type=i32))]
                scf.YieldOp([_raw(prev)])
            start = ArithValue(start_if.results[0])
            m_tid = lds_cnt[fx.Index(tid)]
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + ArithValue(m_tid), p_rsrc, tid)
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
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
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


def build_moe_ep_rowmap_module():
    """JIT launcher: build the gemm2-fused EP row->dest/weight map (ep_rowmap).

    Replaces the ~20 per-layer torch ops that used to build ep_rowmap on the host.
    Single block, grid-stride. Phase 1 inits ep_rowmap[0:cap_rows] to (-1, 0).
    Phase 2 scatters, for each valid LOCAL route (gather_w != 0), the packed dest
    (origin_pe*slot_stride + origin_lid*topk + k) + f32-weight-bits into
    ep_rowmap[final_row]. Remote experts (gather_w == 0, route kernel zeroed them)
    and dead-tail routes (route >= num_valid_routes) stay at the -1 sentinel so the
    gemm2 P2P epilogue skips them.

    ep_rowmap is a flat (cap_rows+1, 2) i32 tensor: [2*row]=dst_packed,
    [2*row+1]=f32 weight bits. tis[t] encodes origin = origin_pe*max_tok+origin_lid
    (t = route // topk, k = route % topk).
    """

    @flyc.kernel(
        name="moe_build_ep_rowmap",
        known_block_size=[MAX_EXPERTS_PER_BLOCK, 1, 1],
    )
    def ep_rowmap_kernel(
        topids_to_rows: fx.Pointer,  # (numel,) i32 final rows
        gather_w: fx.Pointer,        # (numel,) bf16 weights (0 for dropped)
        tis: fx.Pointer,             # (recv_cap,) i32 recv_slot -> origin enc
        ep_rowmap: fx.Pointer,       # (cap_rows+1, 2) i32 flat out
        num_valid_routes: fx.Pointer,  # (1,) i32
        cap_rows: Int32,
        topk: Int32,
        max_tok: Int32,
        slot_stride: Int32,
    ):
        i32 = T.i32
        tid = ArithValue(fx.thread_idx.x)
        neg1 = arith.constant(-1, type=i32)
        zero = arith.constant(0, type=i32)
        one = arith.constant(1, type=i32)
        two = arith.constant(2, type=i32)

        rows_rsrc = ptr_rsrc(topids_to_rows)
        w_rsrc = ptr_rsrc(gather_w)
        tis_rsrc = ptr_rsrc(tis)
        ep_rsrc = ptr_rsrc(ep_rowmap)

        tid_idx = arith.index_cast(T.index, tid)
        stride_idx = arith.index(MAX_EXPERTS_PER_BLOCK)

        # Phase 1: init ep_rowmap[0:cap_rows] = (-1, 0)
        cap_idx = arith.index_cast(T.index, ArithValue(cap_rows))
        init_loop = scf.ForOp(tid_idx, cap_idx, stride_idx)
        with ir.InsertionPoint(init_loop.body):
            row_i32 = arith.index_cast(i32, init_loop.induction_variable)
            base = ArithValue(row_i32) * ArithValue(two)
            buffer_ops.buffer_store(neg1, ep_rsrc, _raw(base))
            buffer_ops.buffer_store(
                zero, ep_rsrc, _raw(base + ArithValue(one))
            )
            scf.YieldOp([])

        gpu.barrier()

        # Phase 2: scatter valid local routes ([0, nvr), gather_w != 0)
        nvr = ArithValue(
            buffer_ops.buffer_load(
                ptr_rsrc(num_valid_routes), zero, vec_width=1, dtype=i32
            )
        )
        nvr_idx = arith.index_cast(T.index, nvr)
        topk_v = ArithValue(topk)
        max_tok_v = ArithValue(max_tok)
        slot_stride_v = ArithValue(slot_stride)
        scat_loop = scf.ForOp(tid_idx, nvr_idx, stride_idx)
        with ir.InsertionPoint(scat_loop.body):
            route = ArithValue(arith.index_cast(i32, scat_loop.induction_variable))
            w_bf = buffer_ops.buffer_load(w_rsrc, _raw(route), vec_width=1, dtype=T.bf16)
            w_f32 = ArithValue(w_bf).extf(T.f32)
            is_kept = arith.cmpf(
                CmpFPredicate.ONE, _raw(w_f32), arith.constant(0.0, type=T.f32)
            )
            kept_if = scf.IfOp(is_kept)
            with ir.InsertionPoint(kept_if.then_block):
                row = ArithValue(
                    buffer_ops.buffer_load(rows_rsrc, _raw(route), vec_width=1, dtype=i32)
                )
                t = ArithValue(arith.divui(_raw(route), _raw(topk_v)))
                k = route - t * topk_v
                enc = ArithValue(
                    buffer_ops.buffer_load(tis_rsrc, _raw(t), vec_width=1, dtype=i32)
                )
                origin_pe = ArithValue(arith.divui(_raw(enc), _raw(max_tok_v)))
                origin_lid = enc - origin_pe * max_tok_v
                packed = origin_pe * slot_stride_v + origin_lid * topk_v + k
                w_bits = ArithValue(w_f32).bitcast(T.i32)
                base = row * ArithValue(two)
                buffer_ops.buffer_store(_raw(packed), ep_rsrc, _raw(base))
                buffer_ops.buffer_store(
                    _raw(w_bits), ep_rsrc, _raw(base + ArithValue(one))
                )
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_ep_rowmap(
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        tis: fx.Pointer,
        ep_rowmap: fx.Pointer,
        num_valid_routes: fx.Pointer,
        cap_rows: fx.Int32,
        topk: fx.Int32,
        max_tok: fx.Int32,
        slot_stride: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ep_rowmap_kernel(
            topids_to_rows,
            gather_w,
            tis,
            ep_rowmap,
            num_valid_routes,
            cap_rows,
            topk,
            max_tok,
            slot_stride,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(MAX_EXPERTS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    launch_ep_rowmap.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_ep_rowmap
