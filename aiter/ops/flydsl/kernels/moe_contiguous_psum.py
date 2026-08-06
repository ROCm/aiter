# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepGEMM-contiguous M-tile prefix sum (FlyDSL), single-block parallel scan.

Computes tile-aligned exclusive prefix sum of per-expert counts for the
contiguous grouped-GEMM scheduler. Single-block parallel scan replaces
torch.cumsum (avoids rocprim trampoline overhead for small E).

The block is ``MAX_EXPERTS_PER_BLOCK`` threads wide but E is not bounded by it:
the scan sweeps the experts in block-sized chunks and carries the running offset
between chunks in LDS. Kimi-K3 (E=896) is the first model to exceed one chunk.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

MAX_EXPERTS_PER_BLOCK = 512


@fx.struct
class _PsumStorage:
    """LDS for the prefix-scan kernels.

    ``lds0``/``lds1`` are ping-pong buffers: each Hillis-Steele step reads one
    and writes the other, then the two swap. ``carry`` accumulates the total of
    the chunks already scanned, so an E wider than the block still gets one
    continuous prefix sum. The trailing 16 is the byte alignment of each array.
    """

    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    carry: fx.Array[fx.Int32, 1, 16]


@fx.struct
class _RoutePsumStorage:
    """LDS for the fused route+psum kernel.

    Adds ``cnt`` -- the per-expert counter the route phase bumps with LDS
    atomics -- to the ping-pong scan buffers of :class:`_PsumStorage`.
    """

    cnt: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds0: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]
    lds1: fx.Array[fx.Int32, MAX_EXPERTS_PER_BLOCK, 16]


def _lds_load(ptr, idx):
    """Scalar i32 load from an LDS pointer at element offset ``idx``."""
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, val, idx):
    """Scalar i32 store to an LDS pointer at element offset ``idx``."""
    fx.ptr_store(val, ptr + fx.Int64(idx))


# The chunked scan below is written out in both kernels rather than shared:
# @flyc.kernel AST-transforms only the decorated body, so a dynamic `for`/`if`
# does not survive being factored into a plain helper.


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
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            _lds_store(carry, fx.Int32(0), 0)
        gpu.barrier()

        # One Hillis-Steele scan spans exactly one thread per lane, so a single
        # pass covers at most MAX_EXPERTS_PER_BLOCK experts -- it used to be the
        # whole kernel, which silently left starts/psum unwritten for every
        # expert past 512 (Kimi-K3 has 896: garbage offsets, then a memory fault
        # in the GEMM that indexes with them).
        #
        # So sweep E in block-sized chunks instead. Each chunk scans as before
        # and then adds ``carry``, the tile-aligned total of all chunks already
        # scanned, which is what makes the per-chunk scans one continuous prefix
        # sum. ``carry`` has to be LDS, not a register: it is produced by lane 0
        # and consumed by all of them on the next iteration.
        #
        # Lanes past ``experts`` feed 0 into the scan -- they keep the last lane
        # holding the true chunk total, and write no output.
        for base in range(0, experts, MAX_EXPERTS_PER_BLOCK):
            e = fx.Uint32(base) + tid
            in_expert = e < fx.Uint32(experts)
            m_e = fx.Uint32(0)
            if in_expert:
                m_e = fx.Uint32(
                    buffer_ops.buffer_load(m_rsrc, e, vec_width=1, dtype=i32)
                )
            _lds_store(lds0, fx.Int32((m_e + tile_minus_1) // tile_v * tile_v), tid)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
                gpu.barrier()
                src, dst = dst, src

            base_off = _lds_load(carry, 0)
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = _lds_load(src, tid - 1)
                start = excl + base_off
                buffer_ops.buffer_store(start, s_rsrc, e)
                buffer_ops.buffer_store(start + fx.Int32(m_e), p_rsrc, e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = _lds_load(src, MAX_EXPERTS_PER_BLOCK - 1)
            gpu.barrier()
            if is_lane0:
                _lds_store(carry, base_off + chunk_total, 0)
            gpu.barrier()

        if is_lane0:
            total = _lds_load(carry, 0)
            gt = total > fx.Int32(tile_v)
            buffer_ops.buffer_store(gt.select(total, tile_v), c_rsrc, 0)

    @flyc.jit
    def launch_psum(
        masked_m: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        contiguous_m: fx.Pointer,
        experts: fx.Int32,
        tile_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
        num_valid_routes: fx.Pointer,  # (1,) int32: only remap routes < this (EP dead-tail skip)
    ):
        i32 = T.i32
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_PsumStorage).peek()
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        carry = lds.carry.ptr

        m_rsrc = ptr_rsrc(masked_m)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)
        c_rsrc = ptr_rsrc(contiguous_m)

        is_lane0 = tid == fx.Uint32(0)
        if is_lane0:
            _lds_store(carry, fx.Int32(0), 0)
        gpu.barrier()

        # One Hillis-Steele scan spans exactly one thread per lane, so a single
        # pass covers at most MAX_EXPERTS_PER_BLOCK experts -- it used to be the
        # whole kernel, which silently left starts/psum unwritten for every
        # expert past 512 (Kimi-K3 has 896: garbage offsets, then a memory fault
        # in the GEMM that indexes with them).
        #
        # So sweep E in block-sized chunks instead. Each chunk scans as before
        # and then adds ``carry``, the tile-aligned total of all chunks already
        # scanned, which is what makes the per-chunk scans one continuous prefix
        # sum. ``carry`` has to be LDS, not a register: it is produced by lane 0
        # and consumed by all of them on the next iteration.
        #
        # Lanes past ``experts`` feed 0 into the scan -- they keep the last lane
        # holding the true chunk total, and write no output.
        for base in range(0, experts, MAX_EXPERTS_PER_BLOCK):
            e = fx.Uint32(base) + tid
            in_expert = e < fx.Uint32(experts)
            m_e = fx.Uint32(0)
            if in_expert:
                m_e = fx.Uint32(
                    buffer_ops.buffer_load(m_rsrc, e, vec_width=1, dtype=i32)
                )
            _lds_store(lds0, fx.Int32((m_e + tile_minus_1) // tile_v * tile_v), tid)
            gpu.barrier()

            src = lds0
            dst = lds1
            for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
                gpu.barrier()
                src, dst = dst, src

            base_off = _lds_load(carry, 0)
            if in_expert:
                is_not_first = tid != 0
                excl = fx.Int32(0)
                if is_not_first:
                    excl = _lds_load(src, tid - 1)
                start = excl + base_off
                buffer_ops.buffer_store(start, s_rsrc, e)
                buffer_ops.buffer_store(start + fx.Int32(m_e), p_rsrc, e)

            # Fold this chunk's total in before the next one overwrites lds0.
            chunk_total = _lds_load(src, MAX_EXPERTS_PER_BLOCK - 1)
            gpu.barrier()
            if is_lane0:
                _lds_store(carry, base_off + chunk_total, 0)
            gpu.barrier()

        if is_lane0:
            total = _lds_load(carry, 0)
            gt = total > fx.Int32(tile_v)
            buffer_ops.buffer_store(gt.select(total, tile_v), c_rsrc, 0)

        gpu.barrier()

        # Only remap valid routes ([0, valid_route_count)); dead-tail routes
        # hold unwritten/garbage rows from the route kernel and must NOT be used
        # as a row index (would OOB-read starts[expert]). They are never read
        # downstream. When truncation is disabled the caller passes a null pointer
        # instead of a (1,) tensor, so the load must not run unconditionally.
        num_valid_routes_is_set = fx.Int64(ptrtoint(num_valid_routes)) != 0
        valid_route_count = fx.Uint32(numel)
        if num_valid_routes_is_set:
            valid_route_count = fx.Uint32(
                buffer_ops.buffer_load(
                    ptr_rsrc(num_valid_routes), fx.Uint32(0), vec_width=1, dtype=i32
                )
            )
        for route_i32 in range(tid, valid_route_count, MAX_EXPERTS_PER_BLOCK):
            row_raw = buffer_ops.buffer_load(
                rows_rsrc, route_i32, vec_width=1, dtype=i32
            )
            # An EP route with no grouped row carries moe_route_maps'
            # DROPPED_ROUTE_ROW (negative) sentinel: masked/contiguous row math
            # would turn it into a wild expert index (OOB starts[] read), and the
            # sentinel has to survive for the consumers that check it, so leave
            # the slot untouched.
            row_is_mapped = fx.Int32(row_raw) >= fx.Int32(0)
            if row_is_mapped:
                row = fx.Uint32(row_raw)
                m = fx.Uint32(route_max_m)
                expert = row // m
                slot = row - expert * m
                start = fx.Uint32(
                    buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
                )
                buffer_ops.buffer_store(start + slot, rows_rsrc, route_i32)

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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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

        bid = ArithValue(fx.block_idx.x)
        blk_sz = ArithValue(arith.constant(MAX_EXPERTS_PER_BLOCK, type=i32))
        gtid = bid * blk_sz + tid
        is_blk0 = arith.cmpi(CmpIPredicate.eq, _raw(bid), arith.constant(0, type=i32))
        # Multi-block: every block keeps its exclusive per-expert starts in LDS so the
        # grid-strided scatter reads starts from LDS (never global -> no cross-block
        # barrier). Only block 0 writes the global starts/psum/contiguous_m outputs.
        starts_lds = dst  # spare ping-pong buffer now holds the exclusive starts
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
            starts_lds[fx.Index(tid)] = start
            _if_g = scf.IfOp(is_blk0)
            with ir.InsertionPoint(_if_g.then_block):
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
            scf.YieldOp([])

        gpu.barrier()

        gtid_idx = arith.index_cast(T.index, _raw(gtid))
        gstride_idx = arith.index(EP_REMAP_NBLK * MAX_EXPERTS_PER_BLOCK)

        # EP Phase 1 init (ep_rowmap[:] = (-1, 0)) is now a parallel HW memset the
        # host wrapper issues BEFORE this launch (one int64 fill over the whole
        # (cap_rows+1, 2) map), replacing what used to be an O(cap_rows) serial loop.
        # The memset is ordered before this kernel on the stream, so the grid-strided
        # scatter below only overwrites the kept rows.

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
        remap_loop = scf.ForOp(gtid_idx, nvr_idx, gstride_idx)
        with ir.InsertionPoint(remap_loop.body):
            route_i32 = arith.index_cast(i32, remap_loop.induction_variable)
            row = ArithValue(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            # A route with no grouped row carries moe_route_maps'
            # DROPPED_ROUTE_ROW: the masked->contiguous math would turn it into a
            # wild expert index (OOB LDS read of starts) and there is nothing to
            # scatter. Skipping it also covers the old ``gather_w != 0`` test --
            # the route kernel gives exactly those routes the sentinel.
            is_kept = arith.cmpi(CmpIPredicate.sge, _raw(row), zero)
            kept_if = scf.IfOp(is_kept)
            with ir.InsertionPoint(kept_if.then_block):
                m = ArithValue(route_max_m)
                expert = ArithValue(arith.divui(row, m))
                slot = row - expert * m
                start = ArithValue(starts_lds[fx.Index(expert)])
                final_row = start + slot
                buffer_ops.buffer_store(final_row, rows_rsrc, route_i32)
                # ep_rowmap scatter for this row: packed dest + f32 weight bits.
                route = ArithValue(route_i32)
                w_bf = buffer_ops.buffer_load(
                    w_rsrc, route_i32, vec_width=1, dtype=T.bf16
                )
                w_f32 = ArithValue(w_bf).extf(T.f32)
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
            grid=(arith.index(EP_REMAP_NBLK), 1, 1),
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
        # Uint32: every value here is a non-negative count/index, so `<`, `>=`
        # and `//` lower to ult/uge/divui exactly like the arith.* calls they
        # replace.
        tid = fx.Uint32(fx.thread_idx.x)
        tile_v = fx.Uint32(tile_m)
        tile_minus_1 = tile_v - 1

        lds = fx.SharedAllocator().allocate(_RoutePsumStorage).peek()
        lds_cnt = lds.cnt.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr

        topk_rsrc = ptr_rsrc(topk_ids)
        rows_rsrc = ptr_rsrc(topids_to_rows)
        m_rsrc = ptr_rsrc(masked_m)
        s_rsrc = ptr_rsrc(starts)
        p_rsrc = ptr_rsrc(psum)

        in_expert = tid < fx.Uint32(experts)

        # Phase A: zero the LDS per-expert atomic counter.
        if in_expert:
            _lds_store(lds_cnt, fx.Int32(0), tid)
        gpu.barrier()

        # Phase B: route + workgroup-scope LDS atomic -> masked-layout rows.
        # The atomic needs a raw addrspace(3) pointer, so the counter array's
        # base is taken as an integer here; SharedAllocator has already folded
        # its offset in, leaving only the per-expert element offset to add.
        cnt_base_i64 = fx.Int64(fx.ptrtoint(lds_cnt))
        numel_i32 = fx.Uint32(numel)
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            e = buffer_ops.buffer_load(topk_rsrc, route_i32, vec_width=1, dtype=i32)
            off_i64 = fx.Int64(e) * 4
            ptr = buffer_ops.create_llvm_ptr(
                cnt_base_i64 + fx.Int64(off_i64), address_space=3
            )
            ptr = ptr._value if hasattr(ptr, "_value") else ptr
            slot = llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                ptr,
                arith.constant(1, type=i32),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result
            row = fx.Uint32(slot) + fx.Uint32(e) * fx.Uint32(max_m)
            buffer_ops.buffer_store(row, rows_rsrc, route_i32)
        gpu.barrier()

        # Phase C: tile-aligned inclusive scan of per-expert counts.
        if in_expert:
            m = fx.Uint32(_lds_load(lds_cnt, tid))
            _lds_store(lds0, (m + tile_minus_1) // tile_v * tile_v, tid)
            buffer_ops.buffer_store(m, m_rsrc, tid)
        gpu.barrier()

        src = lds0
        dst = lds1
        for offset in range_constexpr(1, MAX_EXPERTS_PER_BLOCK):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                val = _lds_load(src, tid)
                has_prev = tid >= offset
                prev = fx.Int32(0)
                if has_prev:
                    prev = _lds_load(src, tid - offset)
                _lds_store(dst, val + prev, tid)
            gpu.barrier()
            src, dst = dst, src

        if in_expert:
            is_not_first = tid != 0
            start = fx.Int32(0)
            if is_not_first:
                start = _lds_load(src, tid - 1)
            m_tid = _lds_load(lds_cnt, tid)
            buffer_ops.buffer_store(start, s_rsrc, tid)
            buffer_ops.buffer_store(start + m_tid, p_rsrc, tid)
        gpu.barrier()

        # Phase D: in-place masked -> contiguous row remap.
        for route_i32 in range(tid, numel_i32, MAX_EXPERTS_PER_BLOCK):
            row = fx.Uint32(
                buffer_ops.buffer_load(rows_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            m = fx.Uint32(max_m)
            expert = row // m
            slot = row - expert * m
            start = fx.Uint32(
                buffer_ops.buffer_load(s_rsrc, expert, vec_width=1, dtype=i32)
            )
            buffer_ops.buffer_store(start + slot, rows_rsrc, route_i32)

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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_route_psum_fused
