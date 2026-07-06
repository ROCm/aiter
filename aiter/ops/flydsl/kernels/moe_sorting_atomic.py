# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""atomicAdd-based MoE token sorting (FlyDSL), lazy-index (no data movement).

Algorithm (hpc-ops `count_and_gather` style, but indices only):

    K0 fill     : pre-fill sorted_ids with sentinel, sorted_weights with 0.
    K1 count    : for each (token, k): atomicAdd(ws_count[eid], 1).
    K2 cumsum   : single thread, exclusive prefix-sum of per-expert padded
                  counts -> ws_offset[E+1], ws_cursor[E] = offset, num_valid_ids.
    K2b eids    : grid = E, block e writes its expert id into sorted_expert_ids.
    K3 scatter  : for each (token, k): slot = atomicAdd(ws_cursor[eid], 1);
                  sorted_ids[slot] = (k << 24) | token; sorted_weights[slot] = w.

Packed token ID format matches CK/native: (topk_slot << 24) | token_id.
Padding sentinel: (topk << 24) | M.

Note: token order within an expert is NOT preserved (atomic scatter order is
non-deterministic).  This is functionally correct for MoE GEMM (each token
carries its own packed id) but means element-wise comparison with the
order-preserving CK/native output must be done as a per-expert multiset.
"""

import functools

from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, gpu, range_constexpr, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects import memref as memref_ops
from flydsl._mlir.dialects._llvm_enum_gen import AtomicBinOp, AtomicOrdering
from flydsl._mlir.dialects._arith_enum_gen import AtomicRMWKind
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])

BLOCK_SIZE = 256
UNIT_SIZE = 32


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _idx(i):
    """Convert an i32/index DSL value to a raw index-typed ir.Value."""
    raw = i.ir_value() if hasattr(i, "ir_value") else i
    if not isinstance(raw.type, ir.IndexType):
        raw = ArithValue(i).index_cast(T.index)
        raw = raw.ir_value() if hasattr(raw, "ir_value") else raw
    return raw


def _lds_load(mr, idx):
    return fx.Int32(memref_ops.load(mr, [_idx(idx)]))


def _lds_store(mr, val, idx):
    memref_ops.store(_unwrap(val), mr, [_idx(idx)])


def _lds_atomic_add(mr, idx, val):
    """LDS atomicAdd(&mr[idx], val); returns OLD value as fx.Int32."""
    old = memref_ops.atomic_rmw(AtomicRMWKind.addi, _unwrap(val), mr, [_idx(idx)])
    return fx.Int32(old)


def _atomic_add_i32(base_ptr, elem_idx, val):
    """atomicAdd(&base_ptr[elem_idx], val) on a global i32 array.

    base_ptr : !llvm.ptr<1> at array start (from _global_base_ptr).
    elem_idx : fx.Int32 element index (GEP in i32 units).
    val      : fx.Int32 increment.
    Returns the OLD value as fx.Int32.
    """
    eptr = buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_unwrap(elem_idx), elem_type=T.i32
    )
    old = llvm.atomicrmw(
        AtomicBinOp.add, eptr, _unwrap(val), AtomicOrdering.monotonic
    )
    return fx.Int32(old)


def _global_base_ptr(tensor):
    """Return a !llvm.ptr<1> pointing at the start of a global tensor."""
    base_idx = buffer_ops.extract_base_index(tensor, address_space=1)
    return buffer_ops.create_llvm_ptr(base_idx, address_space=1)


def _grid_stride_niters(total, stride):
    c_one = fx.Int32(1)
    return (total + stride - c_one) // stride


# ---------------------------------------------------------------------------
# Kernel builders (cached by constexpr config)
# ---------------------------------------------------------------------------
ORDERED_SUB = 2048  # LDS sub-chunk for the exact-ordered scatter


@functools.lru_cache(maxsize=128)
def compile_moe_sorting_atomic(*, num_experts: int, topk: int, nblocks: int,
                               unit_size: int = UNIT_SIZE, contig: bool = False,
                               ordered: bool = False, fuse_cumsum: bool = True,
                               fuse_launch: bool = True):
    # fuse_launch=True: cut 5 kernel launches down to 3 by merging the two
    # dependency-free kernel pairs into single launches (cumsum stays alone -- a
    # grid barrier sits on both sides of it):
    #   stage0: count_kernel also does fill (blocks [NB, NB+n_fill) write the
    #           sorted_ids sentinel / zero sorted_weights) and moe_buf zeroing.
    #   stage2: scatter_kernel also does write_eids (blocks [NB, NB+E) write
    #           sorted_expert_ids for expert bid-NB).
    # Output is identical; this only removes launch/scheduling overhead.
    # fuse_cumsum=True: drop cumsum's phase C (the single-block NB*E serial
    # rewrite of partial[b,e] -> base slot). Instead each scatter block computes
    # its own base on the fly: base[bid,e] = ws_offset[e] + sum_{b'<bid} count[b',e],
    # reading the global ws_offset[] + the raw per-block counts (NB<=32, parallel
    # across blocks). Kills the underutilized single-block cumsum tail. partial
    # then stays as RAW counts (not rewritten). Results are identical.
    # ordered=True: exact token-ascending order within each expert (like CK),
    # produced fully on-GPU. Implies contiguous partitioning; the scatter uses
    # a single-thread in-LDS rank pass (deterministic, in token order) + a
    # parallel write, instead of the atomic position. No host argsort. The
    # rank pass is serial-per-block so the SORT is slower, but stage2 gets
    # CK-quality ordering. NOTE: ordered forces contig.
    if ordered:
        contig = True
    # contig=True: each count/scatter block owns a CONTIGUOUS token-routing
    # range [b*chunk, (b+1)*chunk) instead of a grid-stride stripe. This makes
    # the per-expert output coarsely token-ordered (scramble confined to a
    # ~chunk-wide window) -> recovers most of stage2's gather/scatter DRAM
    # locality, entirely on-GPU (no host argsort). Still uses the in-block LDS
    # atomic, so order is NOT byte-exact vs CK, only locality-friendly.
    E = num_experts
    NB = nblocks
    c_topk = topk
    c_unit = unit_size
    OFF = 0              # ws layout: [0:E+1] = per-expert padded offsets

    arch = get_rocm_arch()

    def _alloc_region(alloc, n):
        off = alloc._align(alloc.ptr, 16)
        alloc.ptr = off + n * 4
        return off

    # count kernel LDS: hist[E].
    alloc_cnt = SmemAllocator(None, arch=arch)
    cnt_hist_off = _alloc_region(alloc_cnt, E)

    # cumsum kernel LDS: total[E+1] (dummy slot E for OOB lanes), offsetL[E].
    # PAR_SCAN: when E <= BLOCK_SIZE each thread owns one expert, so phase B can
    # use a parallel (Hillis-Steele) prefix scan instead of a single-thread
    # serial loop -> ~log2(BLOCK_SIZE) steps instead of E.  Needs scan[BLOCK_SIZE].
    PAR_SCAN = E <= BLOCK_SIZE
    alloc_cs = SmemAllocator(None, arch=arch)
    cs_tot_off = _alloc_region(alloc_cs, E + 1)
    cs_off_off = _alloc_region(alloc_cs, E)
    if PAR_SCAN:
        cs_scan_off = _alloc_region(alloc_cs, BLOCK_SIZE)

    # scatter kernel LDS.
    alloc_sc = SmemAllocator(None, arch=arch)
    if ordered:
        # wcur[E+1] absolute cursor (init=base, dummy slot E for padding lanes).
        sc_wcur_off = _alloc_region(alloc_sc, E + 1)
        sc_base_off = sc_wcur_off  # unused in ordered path
    else:
        sc_base_off = _alloc_region(alloc_sc, E + 1)
        sc_wcur_off = _alloc_region(alloc_sc, E)

    # ---- K0: fill sentinel into sorted_ids / zero sorted_weights ----
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fill_kernel(
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        i32_sorted_len: fx.Int32,
        i32_sentinel: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c_zero = fx.Int32(0)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        gid0 = bid * fx.Int32(BLOCK_SIZE) + tid
        stride = gpu.grid_dim.x * fx.Int32(BLOCK_SIZE)
        niters = _grid_stride_niters(i32_sorted_len, stride)
        for _i in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
            idx = gid0 + fx.Int32(_i) * stride
            valid = idx < i32_sorted_len
            safe = valid.select(idx, c_zero)
            buffer_ops.buffer_store(i32_sentinel, ids_rsrc, safe)
            buffer_ops.buffer_store(c_zero, w_rsrc, safe)

    # ---- K1: per-block LDS histogram -> partial[block, E].  Fused with
    #          moe_buf zeroing: blocks [NB, NB+n_zero) clear moe_buf while the
    #          first NB blocks count (overlaps the bandwidth-bound clear with
    #          the cheap counting; count's LDS is tiny so zero-block occupancy
    #          is barely affected). ----
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def count_kernel(
        topk_ids: fx.Tensor,
        partial: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        i32_total: fx.Int32,
        i32_v4_total: fx.Int32,
        i32_sorted_len: fx.Int32,
        i32_sentinel: fx.Int32,
        i32_nfill: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_E = fx.Int32(E)
        c_NB = fx.Int32(NB)
        c_oob = fx.Int32(0x7FFFFFFF)
        # zero blocks start after the count blocks (+ the fill blocks when fused).
        c_fill_end = (c_NB + i32_nfill) if fuse_launch else c_NB

        ifop_c = scf.IfOp(_unwrap(bid < c_NB))
        with _if_then(ifop_c):
            ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            part_rsrc = buffer_ops.create_buffer_resource(partial, max_size=True)
            base = alloc_cnt.get_base()
            hist = SmemPtr(base, cnt_hist_off, T.i32, shape=(E,)).get()

            for _c in range_constexpr(0, E, BLOCK_SIZE):
                e = fx.Int32(_c) + tid
                safe_e = (e < c_E).select(e, c_zero)
                _lds_store(hist, c_zero, safe_e)
            gpu.barrier()

            if contig:
                # block b owns contiguous routings [b*chunk, (b+1)*chunk)
                chunk = (i32_total + c_NB - c_one) // c_NB
                start = bid * chunk
                blk_end = start + chunk
                nit = _grid_stride_niters(chunk, fx.Int32(BLOCK_SIZE))
                for _i in range(fx.Index(0), ArithValue(nit).index_cast(T.index), fx.Index(1)):
                    idx = start + fx.Int32(_i) * fx.Int32(BLOCK_SIZE) + tid
                    valid = (idx < i32_total) & (idx < blk_end)
                    safe = valid.select(idx, c_zero)
                    eid = buffer_ops.buffer_load(ids_rsrc, safe, vec_width=1, dtype=T.i32)
                    inc = valid.select(c_one, c_zero)
                    _lds_atomic_add(hist, eid, inc)
            else:
                gid0 = bid * fx.Int32(BLOCK_SIZE) + tid
                stride = c_NB * fx.Int32(BLOCK_SIZE)
                niters = _grid_stride_niters(i32_total, stride)
                for _i in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
                    idx = gid0 + fx.Int32(_i) * stride
                    valid = idx < i32_total
                    safe = valid.select(idx, c_zero)
                    eid = buffer_ops.buffer_load(ids_rsrc, safe, vec_width=1, dtype=T.i32)
                    inc = valid.select(c_one, c_zero)
                    _lds_atomic_add(hist, eid, inc)
            gpu.barrier()

            row = bid * c_E
            for _m in range_constexpr(0, E, BLOCK_SIZE):
                e = fx.Int32(_m) + tid
                valid = e < c_E
                safe_e = valid.select(e, c_zero)
                c = _lds_load(hist, safe_e)
                dst = valid.select(row + e, c_oob)
                buffer_ops.buffer_store(c, part_rsrc, dst)

        if fuse_launch:
            # blocks [NB, NB+nfill) fill sorted_ids with sentinel / zero weights
            # (the old standalone fill_kernel, folded in to save a launch).
            ifop_f = scf.IfOp(_unwrap((bid >= c_NB) & (bid < c_fill_end)))
            with _if_then(ifop_f):
                sids_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
                sw_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
                fid0 = (bid - c_NB) * fx.Int32(BLOCK_SIZE) + tid
                fstride = i32_nfill * fx.Int32(BLOCK_SIZE)
                fniters = _grid_stride_niters(i32_sorted_len, fstride)
                for _f in range(fx.Index(0), ArithValue(fniters).index_cast(T.index), fx.Index(1)):
                    fidx = fid0 + fx.Int32(_f) * fstride
                    fvalid = fidx < i32_sorted_len
                    fsafe = fvalid.select(fidx, c_oob)
                    buffer_ops.buffer_store(i32_sentinel, sids_rsrc, fsafe)
                    buffer_ops.buffer_store(c_zero, sw_rsrc, fsafe)

        ifop_z = scf.IfOp(_unwrap(bid >= c_fill_end))
        with _if_then(ifop_z):
            mb_rsrc = buffer_ops.create_buffer_resource(moe_buf_i32, max_size=True)
            _z = _unwrap(fx.Int32(0))
            c_zero_v4 = vector.from_elements(T.vec(4, T.i32), [_z, _z, _z, _z])
            c4 = fx.Int32(4)
            zid0 = (bid - c_fill_end) * fx.Int32(BLOCK_SIZE) + tid
            zstride = (gpu.grid_dim.x - c_fill_end) * fx.Int32(BLOCK_SIZE)
            zniters = _grid_stride_niters(i32_v4_total, zstride)
            for _z2 in range(fx.Index(0), ArithValue(zniters).index_cast(T.index), fx.Index(1)):
                zidx = zid0 + fx.Int32(_z2) * zstride
                zvalid = zidx < i32_v4_total
                buffer_ops.buffer_store(c_zero_v4, mb_rsrc, zvalid.select(zidx * c4, c_oob))

    # ---- K2: reduce partial[*, e] -> total, padded exclusive prefix -> offset,
    #          then rewrite partial[b, e] in-place to the per-block base slot ----
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def cumsum_kernel(
        partial: fx.Tensor,
        workspace: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_E = fx.Int32(E)
        c_unit_i = fx.Int32(c_unit)
        c_oob = fx.Int32(0x7FFFFFFF)
        part_rsrc = buffer_ops.create_buffer_resource(partial, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        nv_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        base = alloc_cs.get_base()
        total = SmemPtr(base, cs_tot_off, T.i32, shape=(E + 1,)).get()
        offl = SmemPtr(base, cs_off_off, T.i32, shape=(E,)).get()

        # phase A: each thread e reduces partial[*, e] across blocks
        for _c in range_constexpr(0, E, BLOCK_SIZE):
            e = fx.Int32(_c) + tid
            valid = e < c_E
            safe_e = valid.select(e, c_zero)
            s = c_zero
            for _b in range_constexpr(0, NB):
                off_b = fx.Int32(_b) * c_E + safe_e
                s = s + buffer_ops.buffer_load(part_rsrc, off_b, vec_width=1, dtype=T.i32)
            # OOB lanes dump into the dummy slot E
            _lds_store(total, s, valid.select(e, c_E))
        gpu.barrier()

        # phase B: padded exclusive prefix over experts -> ws_offset[E+1].
        if PAR_SCAN:
            # ---- parallel Hillis-Steele scan (one thread == one expert) ----
            scan = SmemPtr(base, cs_scan_off, T.i32, shape=(BLOCK_SIZE,)).get()
            e = tid
            valid = e < c_E
            safe_e = valid.select(e, c_zero)
            cnt = _lds_load(total, safe_e)
            blocks = (cnt + c_unit_i - c_one) // c_unit_i
            my_pad = valid.select(
                (cnt == c_zero).select(c_zero, blocks * c_unit_i), c_zero
            )
            _lds_store(scan, my_pad, e)
            gpu.barrier()
            nsteps = BLOCK_SIZE.bit_length() - 1  # log2(BLOCK_SIZE) Hillis-Steele steps
            for _s in range_constexpr(0, nsteps):
                d = 1 << _s
                ge = e >= fx.Int32(d)
                src = ge.select(e - fx.Int32(d), c_zero)
                add = ge.select(_lds_load(scan, src), c_zero)
                cur = _lds_load(scan, e)
                gpu.barrier()
                _lds_store(scan, cur + add, e)
                gpu.barrier()
            # scan[e] now = inclusive prefix; exclusive offset = inclusive - my_pad
            excl = _lds_load(scan, e) - my_pad
            buffer_ops.buffer_store(
                excl, ws_rsrc, valid.select(fx.Int32(OFF) + e, c_oob)
            )
            if not fuse_cumsum:
                ifv = scf.IfOp(_unwrap(valid))
                with _if_then(ifv):
                    _lds_store(offl, excl, e)
            ifop_t = scf.IfOp(_unwrap(tid == c_zero))
            with _if_then(ifop_t):
                tot = _lds_load(scan, fx.Int32(E - 1))  # inclusive[E-1] = grand total
                buffer_ops.buffer_store(tot, ws_rsrc, fx.Int32(OFF + E))
                buffer_ops.buffer_store(tot, nv_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, nv_rsrc, c_one)
            gpu.barrier()
        else:
            # ---- fallback: thread 0 serial scan in LDS, then parallel ws write
            # (used when E > BLOCK_SIZE so the one-thread-per-expert map breaks) ----
            ifop = scf.IfOp(_unwrap(tid == c_zero))
            with _if_then(ifop):
                off = c_zero
                for _e in range_constexpr(0, E):
                    cnt = _lds_load(total, fx.Int32(_e))
                    blocks = (cnt + c_unit_i - c_one) // c_unit_i
                    padded = (cnt == c_zero).select(c_zero, blocks * c_unit_i)
                    _lds_store(offl, off, fx.Int32(_e))
                    off = off + padded
                _lds_store(total, off, c_E)  # stash grand total in dummy slot E
                buffer_ops.buffer_store(off, nv_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, nv_rsrc, c_one)
            gpu.barrier()
            for _c in range_constexpr(0, E, BLOCK_SIZE):
                e = fx.Int32(_c) + tid
                valid = e < c_E
                safe_e = valid.select(e, c_zero)
                v = _lds_load(offl, safe_e)
                buffer_ops.buffer_store(v, ws_rsrc, valid.select(fx.Int32(OFF) + e, c_oob))
            ifop_t = scf.IfOp(_unwrap(tid == c_zero))
            with _if_then(ifop_t):
                tot = _lds_load(total, c_E)
                buffer_ops.buffer_store(tot, ws_rsrc, fx.Int32(OFF + E))
            gpu.barrier()

        # phase C: each thread e turns partial[b, e] (a count) into the running
        # base slot for block b's expert-e tokens.  SKIPPED when fuse_cumsum:
        # the scatter kernel then computes each block's base on the fly from the
        # raw counts + ws_offset (avoids this single-block NB*E serial rewrite).
        if not fuse_cumsum:
            for _c in range_constexpr(0, E, BLOCK_SIZE):
                e = fx.Int32(_c) + tid
                valid = e < c_E
                safe_e = valid.select(e, c_zero)
                running = _lds_load(offl, safe_e)
                for _b in range_constexpr(0, NB):
                    off_b = fx.Int32(_b) * c_E + safe_e
                    v = buffer_ops.buffer_load(part_rsrc, off_b, vec_width=1, dtype=T.i32)
                    dst = valid.select(off_b, fx.Int32(0x7FFFFFFF))
                    buffer_ops.buffer_store(running, part_rsrc, dst)
                    running = running + v

    # ---- K2b: write expert ids into sorted_expert_ids (grid = E) ----
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def write_eids_kernel(
        workspace: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
    ):
        eid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c_one = fx.Int32(1)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        se_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        o0 = buffer_ops.buffer_load(ws_rsrc, fx.Int32(OFF) + eid, vec_width=1, dtype=T.i32)
        o1 = buffer_ops.buffer_load(
            ws_rsrc, fx.Int32(OFF) + eid + c_one, vec_width=1, dtype=T.i32
        )
        blk0 = o0 // fx.Int32(c_unit)
        nb = (o1 - o0) // fx.Int32(c_unit)
        niters = _grid_stride_niters(nb, fx.Int32(BLOCK_SIZE))
        for _j in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
            j = fx.Int32(_j) * fx.Int32(BLOCK_SIZE) + tid
            valid = j < nb
            safe = valid.select(blk0 + j, fx.Int32(0x7FFFFFFF))
            buffer_ops.buffer_store(eid, se_rsrc, safe)

    # ---- K3: single-pass scatter. partial[bid, e] holds this block's base
    #          slot for expert e (computed in cumsum).  When fuse_launch, blocks
    #          [NB, NB+E) instead run the write_eids work (no data dependency on
    #          the scatter blocks -- both only read cumsum's outputs). ----
    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def scatter_kernel(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        partial: fx.Tensor,
        workspace: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        i32_total: fx.Int32,
    ):
        bid = gpu.block_idx.x
        tid = gpu.thread_idx.x
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_E = fx.Int32(E)
        c_NB = fx.Int32(NB)
        c_topk_i = fx.Int32(c_topk)
        c_oob = fx.Int32(0x7FFFFFFF)
        ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        w_rsrc = buffer_ops.create_buffer_resource(topk_weights, max_size=True)
        part_rsrc = buffer_ops.create_buffer_resource(partial, max_size=True)
        sids_rsrc = buffer_ops.create_buffer_resource(sorted_ids, max_size=True)
        sw_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ws_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        se_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        base = alloc_sc.get_base()

        def _block_base(safe_e):
            # per-block base slot for expert safe_e.
            if fuse_cumsum:
                # partial holds RAW counts; base = ws_offset[e] + prefix over blocks.
                b = buffer_ops.buffer_load(ws_rsrc, safe_e, vec_width=1, dtype=T.i32)
                for _b in range_constexpr(0, NB):
                    cnt = buffer_ops.buffer_load(
                        part_rsrc, fx.Int32(_b) * c_E + safe_e, vec_width=1, dtype=T.i32
                    )
                    b = b + (fx.Int32(_b) < bid).select(cnt, c_zero)
                return b
            # partial already rewritten to base slots by cumsum's phase C.
            return buffer_ops.buffer_load(
                part_rsrc, bid * c_E + safe_e, vec_width=1, dtype=T.i32
            )

        def _emit_scatter():
            if ordered:
                # ---- fast (~exact) ordered scatter (contiguous partition) ----
                # wcur[e] starts at the block's base slot.  We walk the block's
                # contiguous tokens in BLOCK_SIZE sub-chunks IN ORDER (barrier
                # between sub-chunks), using a parallel LDS atomic cursor.  Across
                # sub-chunks order is preserved; within one 256-token sub-chunk the
                # atomic scrambles only ~1 token/expert -> effectively token order.
                wcur = SmemPtr(base, sc_wcur_off, T.i32, shape=(E + 1,)).get()
                for _c in range_constexpr(0, E, BLOCK_SIZE):
                    e = fx.Int32(_c) + tid
                    valid = e < c_E
                    safe_e = valid.select(e, c_zero)
                    b = _block_base(safe_e)
                    _lds_store(wcur, b, valid.select(e, c_E))
                gpu.barrier()

                chunk = (i32_total + c_NB - c_one) // c_NB
                start = bid * chunk
                blk_end = start + chunk
                nit = _grid_stride_niters(chunk, fx.Int32(BLOCK_SIZE))
                for _i in range(fx.Index(0), ArithValue(nit).index_cast(T.index), fx.Index(1)):
                    idx = start + fx.Int32(_i) * fx.Int32(BLOCK_SIZE) + tid
                    valid = (idx < i32_total) & (idx < blk_end)
                    safe = valid.select(idx, c_zero)
                    eid = buffer_ops.buffer_load(ids_rsrc, safe, vec_width=1, dtype=T.i32)
                    token = safe // c_topk_i
                    kslot = safe % c_topk_i
                    packed = (kslot << fx.Int32(24)) | token
                    w_i32 = buffer_ops.buffer_load(w_rsrc, safe, vec_width=1, dtype=T.i32)
                    inc = valid.select(c_one, c_zero)
                    slot = _lds_atomic_add(wcur, eid, inc)  # = base[eid] + running
                    safe_slot = valid.select(slot, c_oob)
                    buffer_ops.buffer_store(packed, sids_rsrc, safe_slot)
                    buffer_ops.buffer_store(w_i32, sw_rsrc, safe_slot)
                    gpu.barrier()  # serialize sub-chunks -> preserve token order
                return

            bbase = SmemPtr(base, sc_base_off, T.i32, shape=(E + 1,)).get()
            wcur = SmemPtr(base, sc_wcur_off, T.i32, shape=(E,)).get()

            for _c in range_constexpr(0, E, BLOCK_SIZE):
                e = fx.Int32(_c) + tid
                valid = e < c_E
                safe_e = valid.select(e, c_zero)
                b = _block_base(safe_e)
                store_e = valid.select(e, c_E)
                _lds_store(bbase, b, store_e)
                _lds_store(wcur, c_zero, safe_e)
            gpu.barrier()

            def _scatter_one(idx, valid):
                safe = valid.select(idx, c_zero)
                eid = buffer_ops.buffer_load(ids_rsrc, safe, vec_width=1, dtype=T.i32)
                token = safe // c_topk_i
                kslot = safe % c_topk_i
                packed = (kslot << fx.Int32(24)) | token
                w_i32 = buffer_ops.buffer_load(w_rsrc, safe, vec_width=1, dtype=T.i32)
                inc = valid.select(c_one, c_zero)
                lpos = _lds_atomic_add(wcur, eid, inc)
                slot = _lds_load(bbase, eid) + lpos
                safe_slot = valid.select(slot, c_oob)
                buffer_ops.buffer_store(packed, sids_rsrc, safe_slot)
                buffer_ops.buffer_store(w_i32, sw_rsrc, safe_slot)

            if contig:
                # MUST match count's partition so partial[bid,e] base is consistent.
                chunk = (i32_total + c_NB - c_one) // c_NB
                start = bid * chunk
                blk_end = start + chunk
                nit = _grid_stride_niters(chunk, fx.Int32(BLOCK_SIZE))
                for _i in range(fx.Index(0), ArithValue(nit).index_cast(T.index), fx.Index(1)):
                    idx = start + fx.Int32(_i) * fx.Int32(BLOCK_SIZE) + tid
                    _scatter_one(idx, (idx < i32_total) & (idx < blk_end))
            else:
                # only NB blocks scatter -> stride over NB, not the (NB+E) grid.
                gid0 = bid * fx.Int32(BLOCK_SIZE) + tid
                stride = c_NB * fx.Int32(BLOCK_SIZE)
                niters = _grid_stride_niters(i32_total, stride)
                for _i in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
                    idx = gid0 + fx.Int32(_i) * stride
                    _scatter_one(idx, idx < i32_total)

        def _emit_eids(eid):
            # block writes expert id `eid` into its sorted_expert_ids block range.
            o0 = buffer_ops.buffer_load(ws_rsrc, fx.Int32(OFF) + eid, vec_width=1, dtype=T.i32)
            o1 = buffer_ops.buffer_load(
                ws_rsrc, fx.Int32(OFF) + eid + c_one, vec_width=1, dtype=T.i32
            )
            blk0 = o0 // fx.Int32(c_unit)
            nb = (o1 - o0) // fx.Int32(c_unit)
            niters = _grid_stride_niters(nb, fx.Int32(BLOCK_SIZE))
            for _j in range(fx.Index(0), ArithValue(niters).index_cast(T.index), fx.Index(1)):
                j = fx.Int32(_j) * fx.Int32(BLOCK_SIZE) + tid
                valid = j < nb
                safe = valid.select(blk0 + j, c_oob)
                buffer_ops.buffer_store(eid, se_rsrc, safe)

        if fuse_launch:
            # blocks [0, NB) scatter; blocks [NB, NB+E) write expert ids.
            ifs = scf.IfOp(_unwrap(bid < c_NB))
            with _if_then(ifs):
                _emit_scatter()
            ife = scf.IfOp(_unwrap(bid >= c_NB))
            with _if_then(ife):
                _emit_eids(bid - c_NB)
        else:
            _emit_scatter()

    # ---- launch wrappers ----
    @flyc.jit
    def launch_fill(sorted_ids, sorted_weights, i32_sorted_len, i32_sentinel,
                    n_grid: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        launcher = fill_kernel(sorted_ids, sorted_weights, i32_sorted_len, i32_sentinel)
        launcher.launch(grid=(n_grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    @flyc.jit
    def launch_count(topk_ids, partial, moe_buf_i32, sorted_ids, sorted_weights,
                     i32_total, i32_v4_total, i32_sorted_len, i32_sentinel, i32_nfill,
                     n_grid: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        alloc_cnt.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc_cnt.finalize()
        launcher = count_kernel(topk_ids, partial, moe_buf_i32, sorted_ids,
                                sorted_weights, i32_total, i32_v4_total,
                                i32_sorted_len, i32_sentinel, i32_nfill)
        launcher.launch(grid=(n_grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    @flyc.jit
    def launch_cumsum(partial, workspace, num_valid_ids, i32_tokens,
                      stream: fx.Stream = fx.Stream(None)):
        alloc_cs.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc_cs.finalize()
        launcher = cumsum_kernel(partial, workspace, num_valid_ids, i32_tokens)
        launcher.launch(grid=(1, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    @flyc.jit
    def launch_write_eids(workspace, sorted_expert_ids,
                          n_grid: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        launcher = write_eids_kernel(workspace, sorted_expert_ids)
        launcher.launch(grid=(n_grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    @flyc.jit
    def launch_scatter(topk_ids, topk_weights, partial, workspace, sorted_ids,
                       sorted_weights, sorted_expert_ids, i32_total, n_grid: fx.Int32,
                       stream: fx.Stream = fx.Stream(None)):
        alloc_sc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            alloc_sc.finalize()
        launcher = scatter_kernel(
            topk_ids, topk_weights, partial, workspace, sorted_ids, sorted_weights,
            sorted_expert_ids, i32_total
        )
        launcher.launch(grid=(n_grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    return (launch_fill, launch_count, launch_cumsum,
            launch_write_eids, launch_scatter)
