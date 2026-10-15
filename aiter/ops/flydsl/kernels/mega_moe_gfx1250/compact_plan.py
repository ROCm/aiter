# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Send-side compact route plan for gfx1250 TDM dispatch.

A low-LDS persistent-enough grid counts every local route into
``(dest_rank, local_expert)`` buckets, allgathers the histogram, then writes
the destination compact row into ``tok_map``. Dispatch copies payload straight
onto that row, so the receiver never runs ``moe_route_g2l_lds``.
"""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import readlane
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from . import tdm_prims as TDM
from .config import _LANE_MASK as LANE_MASK
from .config import _LOG2_WAVE_SIZE as LOG2_WAVE
from .config import _WAVE_SIZE as WAVE

PLAN_BLOCKS = 1
PLAN_WAVES = 32
PLAN_THREADS = PLAN_WAVES * WAVE
_LDS_ROUTE_CAP = 8192


def compact_plan_waves(*, npes: int, max_routes: int) -> int:
    """Waves for ``tdm_compact_plan``. Must be >= ``npes`` (one warp TDM-stores each peer).

    The kernel is allgather-bound; 32 waves only help the route tally past ~1K
    routes. Decode (tens of routes) pays the extra waves as launch occupancy.
    """
    env = os.environ.get("AITER_TDM_COMPACT_PLAN_WAVES")
    if env:
        return max(int(npes), int(env))
    routes = max(1, int(max_routes))
    if routes <= 96:
        want = 4
    elif routes <= 384:
        want = 8
    else:
        want = 32
    return max(int(npes), want)


def _align32(n: int) -> int:
    return (int(n) + 31) // 32 * 32


def compact_hist_layout(*, npes: int, experts_per_rank: int, max_routes: int):
    """Per-parity histogram row layout for the symmetric arena.

    Returns ``(row_dwords, sparse_cap, use_sparse)``. Dense rows are ``npes*epr``
    dwords. Sparse rows pack ``nnz`` plus ``(seg, cnt)`` pairs when local routes
    cannot fill the dense table (decode).
    """
    segs = int(npes) * int(experts_per_rank)
    max_routes = max(1, int(max_routes))
    sparse_cap = min(segs, _align32(max_routes))
    use_sparse = sparse_cap < segs
    row_dwords = _align32(1 + sparse_cap) if use_sparse else segs
    return row_dwords, sparse_cap, use_sparse


def compact_hist_stride(*, npes: int, experts_per_rank: int, max_routes: int) -> int:
    row_dwords, _, _ = compact_hist_layout(
        npes=npes, experts_per_rank=experts_per_rank, max_routes=max_routes
    )
    return int(npes) * int(row_dwords)


def compact_done_nbytes() -> int:
    """Two parity arrival counters (one i32 each)."""
    return 8


@flyc.jit
def _wave32_inclusive_scan_i32(value, lane):
    """Inclusive sum within one gfx1250 wave32."""
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True)
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x10)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, source16 * fx.Int32(4), value)
    return (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)


def compact_row_capacity(
    *,
    max_recv: int,
    topk: int,
    experts_per_rank: int,
    tile_m: int,
) -> int:
    """Static CUDAGraph-safe bound matching grouped_moe contiguous_m."""
    tile_m = int(tile_m)
    ub = int(max_recv) * int(topk) + int(experts_per_rank) * tile_m - int(topk)
    aligned = ((ub + tile_m - 1) // tile_m) * tile_m
    return max(tile_m, aligned)


def compact_comb_tab_dwords(*, npes: int, experts_per_rank: int) -> int:
    """Dwords of the ``dense_comb`` run table the plan appends after psum.

    Entry ``e*npes + src`` is ``(row_start, delta)``: the first compact row this
    rank's local expert ``e`` holds from ``src``, and ``comb_slot - row`` for
    every row of that run on ``src``'s side.
    """
    return 2 * int(npes) * int(experts_per_rank)


@functools.cache
def compile_tdm_compact_plan(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    topk: int,
    tile_m: int,
    compact_cap: int,
    off_hist: int,
    off_done: int,
    hist_stride: int,
    max_routes: int,
    hist_pingpong: bool = True,
    dense_comb: bool = False,
):
    """Compile the compact-plan kernel. ``hist_stride`` is ``npes * row_dwords``.

    ``hist_pingpong``: the single-buffer protocol indexes the 2-deep hist/done
    arena by ``gen & 1``. Double-buffered callers pass a slot-specific
    ``off_hist`` / ``off_done`` and set this False so two in-flight plans do
    not share a done counter.

    ``dense_comb`` lays this rank's comb_inp out segment-major instead of
    ``tok*topk + k``: the route of segment ``seg`` with intra index ``i`` owns
    slot ``local_base[seg] + i``, ``local_base`` being the exclusive prefix of
    this rank's own histogram. A ``(src, dest, e)`` run then sits contiguous on
    both ends, in the same order, so gemm2 can return it in one plain TDM. The
    plan writes ``comb_slot[route]`` for the reduce, and the run table of
    :func:`compact_comb_tab_dwords` into ``psum[experts_per_rank:]`` for gemm2.
    """
    if WAVE != 32:
        raise ValueError("compact plan requires gfx1250 wave32")
    epr = int(experts_per_rank)
    segs = int(npes) * epr
    if segs > 1024:
        raise ValueError(
            f"compact plan LDS hist supports at most 1024 segments, got {segs}"
        )
    tile_m = int(tile_m)
    compact_cap = int(compact_cap)
    max_routes = max(1, int(max_routes))
    hist_pingpong = bool(hist_pingpong)
    dense_comb = bool(dense_comb)
    plan_blocks = PLAN_BLOCKS
    if dense_comb and plan_blocks != 1:
        raise ValueError("dense_comb reads the hist matrix from block 0's LDS")
    plan_waves = compact_plan_waves(npes=npes, max_routes=max_routes)
    plan_threads = plan_waves * WAVE
    peer_bits = max(1, (int(npes) - 1).bit_length())
    peer_mask = (1 << peer_bits) - 1
    if compact_cap >= (1 << (31 - peer_bits)):
        raise ValueError(
            "compact row encoding exceeds positive i32: "
            f"cap={compact_cap} peers={npes}"
        )
    hist_stride = int(hist_stride)
    row_dwords, _sparse_cap, use_sparse = compact_hist_layout(
        npes=npes, experts_per_rank=epr, max_routes=max_routes
    )
    if hist_stride != npes * row_dwords:
        raise ValueError(
            f"hist_stride={hist_stride} != npes*row_dwords={npes * row_dwords}"
        )
    merge_routes = max_routes <= _LDS_ROUTE_CAP
    if max_routes % 4 == 0:
        route_vec = 4
    elif max_routes % 2 == 0:
        route_vec = 2
    else:
        route_vec = 1
    dropped = -1

    @flyc.kernel(name="tdm_compact_plan", known_block_size=[plan_threads, 1, 1])
    def kernel(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        addr_comb_slot: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        window = cco.Window(arena)

        rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_bhist = create_buffer_resource_from_addr(addr_block_hist)
        rsrc_base = create_buffer_resource_from_addr(addr_send_base)
        rsrc_mm = create_buffer_resource_from_addr(addr_masked_m)
        rsrc_psum = create_buffer_resource_from_addr(addr_psum)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)
        if const_expr(dense_comb):
            rsrc_cslot = create_buffer_resource_from_addr(addr_comb_slot)

        smem = fx.SharedAllocator(static=False)
        hist_ptr = smem.allocate(segs * 4, 128)._ptr
        total_ptr = smem.allocate(segs * 4, 16)._ptr
        pref_ptr = smem.allocate(segs * 4, 16)._ptr
        matrix_ptr = smem.allocate(npes * segs * 4, 128)._ptr
        lds_hist = fx.Int64(fx.ptrtoint(hist_ptr))
        lds_total = fx.Int64(fx.ptrtoint(total_ptr))
        lds_pref = fx.Int64(fx.ptrtoint(pref_ptr))
        lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))
        if const_expr(merge_routes):
            routes_ptr = smem.allocate(max_routes * 4, 16)._ptr
            lds_routes = fx.Int64(fx.ptrtoint(routes_ptr))
        if const_expr(dense_comb):
            estart_ptr = smem.allocate(epr * 4, 16)._ptr
            lds_estart = fx.Int64(fx.ptrtoint(estart_ptr))
        if const_expr(use_sparse):
            pack_ptr = smem.allocate(row_dwords * 4, 128)._ptr
            recv_ptr = smem.allocate(npes * row_dwords * 4, 128)._ptr
            lds_pack = fx.Int64(fx.ptrtoint(pack_ptr))
            lds_recv = fx.Int64(fx.ptrtoint(recv_ptr))

        for s in range(tid, segs, plan_threads):
            comm_ops.store_i32_lds(
                lds_hist + fx.Int64(s) * fx.Int64(4), arith.constant(0)
            )
        fx.barrier()

        n_routes = inp_cur_tok * fx.Int32(topk)

        def _tally_one(route, expert):
            dest_pe = expert // epr
            valid = (expert >= 0) & (dest_pe >= 0) & (dest_pe < npes)
            local_e = expert - dest_pe * fx.Int32(epr)
            segment = dest_pe * fx.Int32(epr) + local_e
            intra = arith.constant(0)
            if valid:
                intra = comm_ops.atomic_add_lds(
                    lds_hist + fx.Int64(segment) * fx.Int64(4), arith.constant(1)
                )
            packed = arith.select(
                valid,
                segment | (intra << arith.constant(16)),
                arith.constant(dropped),
            )
            if const_expr(merge_routes):
                comm_ops.store_i32_lds(
                    lds_routes + fx.Int64(route) * fx.Int64(4), packed
                )
            else:
                buffer_store(packed, rsrc_map, route)

        vec_n = n_routes - (n_routes & fx.Int32(route_vec - 1))
        stride = plan_blocks * plan_threads * route_vec
        for route in range(
            bid * plan_threads * route_vec + tid * route_vec, vec_n, stride
        ):
            if const_expr(route_vec == 1):
                expert = buffer_load(rsrc_idx, route, vec_width=1, dtype=T.i32)
                _tally_one(route, expert)
            else:
                raw = fx.Vector(
                    buffer_load(rsrc_idx, route, vec_width=route_vec, dtype=T.i32)
                )
                for k in range_constexpr(route_vec):
                    _tally_one(route + k, raw[k])
        for route in range(
            vec_n + bid * plan_threads + tid, n_routes, plan_blocks * plan_threads
        ):
            expert = buffer_load(rsrc_idx, route, vec_width=1, dtype=T.i32)
            _tally_one(route, expert)

        fx.barrier()
        if const_expr(plan_blocks > 1):
            for s in range(tid, segs, plan_threads):
                cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4))
                buffer_store(cnt, rsrc_bhist, bid * segs + s)
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
                next_gen = gen + arith.constant(1)
                arrive = comm_ops.atomic_add_system(addr_barrier, arith.constant(1))
                if arrive != plan_blocks - 1:
                    comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(4), next_gen)
                    comm_ops.fence_agent_acquire()
                else:
                    comm_ops.fence_system_release()
                    buffer_store(arith.constant(0), rsrc_bar, 0)
                    buffer_store(next_gen, rsrc_bar, 2)
                    comm_ops.fence_agent_release()
                    buffer_store(next_gen, rsrc_bar, 1)
            fx.barrier()

        if bid == 0:
            if const_expr(plan_blocks > 1):
                for s in range(tid, segs, plan_threads):
                    total = arith.constant(0)
                    for blk in range_constexpr(plan_blocks):
                        cnt = buffer_load(
                            rsrc_bhist, blk * segs + s, vec_width=1, dtype=T.i32
                        )
                        buffer_store(total, rsrc_bhist, blk * segs + s)
                        total = total + cnt
                    buffer_store(total, rsrc_base, s)
                comm_ops.waitcnt_stores()
                fx.barrier()

            if tid == 0:
                gen = buffer_load(
                    rsrc_bar, 2, vec_width=1, dtype=T.i32
                ) + arith.constant(1)
                buffer_store(gen, rsrc_bar, 2)
            fx.barrier()
            gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
            if const_expr(hist_pingpong):
                parity = gen & arith.constant(1)
                hist_off = off_hist + parity * hist_stride * 4
            else:
                hist_off = off_hist
            done_off = off_done

            if const_expr(use_sparse):
                for s in range(tid, row_dwords, plan_threads):
                    comm_ops.store_i32_lds(
                        lds_pack + fx.Int64(s) * fx.Int64(4), arith.constant(0)
                    )
                fx.barrier()
                for s in range(tid, segs, plan_threads):
                    cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4))
                    if cnt != 0:
                        slot = comm_ops.atomic_add_lds(lds_pack, arith.constant(1))
                        packed = fx.Int32(s) | (cnt << arith.constant(16))
                        comm_ops.store_i32_lds(
                            lds_pack + fx.Int64(slot + 1) * fx.Int64(4), packed
                        )
                fx.barrier()
                tdm_rows = row_dwords // 32
                if warp < npes:
                    peer_hist = fx.Int64(window.lsa_ptr(warp, hist_off)) + fx.Int64(
                        rank * row_dwords * 4
                    )
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_pack)), peer_hist
                        ),
                        TDM.tdm_group1(32, tdm_rows, 4),
                    )
                fx.barrier()
                TDM.tdm_wait(0)
            elif const_expr(plan_blocks == 1 and segs % 32 == 0):
                if warp < npes:
                    peer_hist = fx.Int64(window.lsa_ptr(warp, hist_off)) + fx.Int64(
                        rank * segs * 4
                    )
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_hist)), peer_hist
                        ),
                        TDM.tdm_group1(32, segs // 32, 4),
                    )
                fx.barrier()
                TDM.tdm_wait(0)
            else:
                hist_vec = 2 if segs % 2 == 0 else 1
                for peer in range_constexpr(npes):
                    peer_hist = fx.Int64(window.lsa_ptr(peer, hist_off)) + fx.Int64(
                        rank * segs * 4
                    )
                    peer_hist_rsrc = create_buffer_resource_from_addr(peer_hist)
                    for s in range(
                        tid * hist_vec,
                        segs,
                        plan_threads * hist_vec,
                    ):
                        vals = [
                            comm_ops.load_i32_lds(
                                lds_hist + fx.Int64(s + i) * fx.Int64(4)
                            )
                            for i in range_constexpr(hist_vec)
                        ]
                        buffer_store(
                            fx.Vector.from_elements(vals, dtype=fx.Int32),
                            peer_hist_rsrc,
                            s,
                        )
                comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                comm_ops.fence_system_release()
            fx.barrier()
            if tid < npes:
                comm_ops.atomic_add_system(
                    fx.Int64(window.lsa_ptr(tid, done_off)),
                    arith.constant(1),
                )
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                comm_ops.wait_i32_until_equals(
                    fx.Int64(window.lsa_ptr(rank, done_off)),
                    gen * fx.Int32(npes),
                )
                comm_ops.fence_system_acquire()
            fx.barrier()

            if const_expr(use_sparse):
                tdm_rows = (npes * row_dwords) // 32
                TDM.tdm_load(
                    TDM.tdm_group0(
                        arith.trunci(T.i32, arith.unwrap(lds_recv)),
                        fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off)),
                    ),
                    TDM.tdm_group1(32, tdm_rows, 4),
                )
                TDM.tdm_wait(0)
                for s in range(tid, npes * segs, plan_threads):
                    comm_ops.store_i32_lds(
                        lds_matrix + fx.Int64(s) * fx.Int64(4), arith.constant(0)
                    )
                fx.barrier()
                for src in range_constexpr(npes):
                    src_base = fx.Int32(src * row_dwords)
                    nnz = comm_ops.load_i32_lds(
                        lds_recv + fx.Int64(src * row_dwords) * fx.Int64(4)
                    )
                    for i in range(tid, nnz, plan_threads):
                        packed = comm_ops.load_i32_lds(
                            lds_recv + fx.Int64(src_base + i + 1) * fx.Int64(4)
                        )
                        seg = packed & arith.constant(0xFFFF)
                        cnt = packed >> arith.constant(16)
                        comm_ops.store_i32_lds(
                            lds_matrix
                            + fx.Int64(src * segs) * fx.Int64(4)
                            + fx.Int64(seg) * fx.Int64(4),
                            cnt,
                        )
                fx.barrier()
            else:
                matrix_n = npes * segs
                if const_expr(matrix_n % 32 == 0):
                    TDM.tdm_load(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_matrix)),
                            fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off)),
                        ),
                        TDM.tdm_group1(32, matrix_n // 32, 4),
                    )
                    TDM.tdm_wait(0)
                else:
                    local_hist_rsrc = create_buffer_resource_from_addr(
                        fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off))
                    )
                    for s in range(tid, matrix_n, plan_threads):
                        comm_ops.store_i32_lds(
                            lds_matrix + fx.Int64(s) * fx.Int64(4),
                            buffer_load(local_hist_rsrc, s, vec_width=1, dtype=T.i32),
                        )
                fx.barrier()
            for idx in range(tid, segs, plan_threads):
                dest = idx // fx.Int32(epr)
                e = idx - dest * fx.Int32(epr)
                total = arith.constant(0)
                my_prefix = arith.constant(0)
                for src in range_constexpr(npes):
                    cnt = comm_ops.load_i32_lds(
                        lds_matrix
                        + fx.Int64(src * segs + dest * fx.Int32(epr) + e) * fx.Int64(4)
                    )
                    if src == rank:
                        my_prefix = total
                    total = total + cnt
                comm_ops.store_i32_lds(lds_total + fx.Int64(idx) * fx.Int64(4), total)
                comm_ops.store_i32_lds(
                    lds_pref + fx.Int64(idx) * fx.Int64(4), my_prefix
                )
                if dest == rank:
                    buffer_store(total, rsrc_mm, e)
            fx.barrier()
            if warp < npes:
                dest = warp
                carry = arith.constant(0)
                for chunk in range(0, epr, WAVE):
                    e = fx.Int32(chunk) + lane
                    in_expert = e < fx.Int32(epr)
                    safe_e = in_expert.select(e, fx.Int32(0))
                    idx = dest * fx.Int32(epr) + safe_e
                    total = arith.constant(0)
                    my_prefix = arith.constant(0)
                    if in_expert:
                        total = comm_ops.load_i32_lds(
                            lds_total + fx.Int64(idx) * fx.Int64(4)
                        )
                        my_prefix = comm_ops.load_i32_lds(
                            lds_pref + fx.Int64(idx) * fx.Int64(4)
                        )
                    aligned = (total + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
                    aligned = aligned * fx.Int32(tile_m)
                    inclusive = _wave32_inclusive_scan_i32(aligned, lane)
                    expert_start = carry + inclusive - aligned
                    if in_expert:
                        send = expert_start + my_prefix
                        if dest == rank:
                            buffer_store(expert_start + total, rsrc_psum, e)
                            if const_expr(dense_comb):
                                comm_ops.store_i32_lds(
                                    lds_estart + fx.Int64(safe_e) * fx.Int64(4),
                                    expert_start,
                                )
                        buffer_store(send, rsrc_base, idx)
                        comm_ops.store_i32_lds(
                            lds_hist + fx.Int64(idx) * fx.Int64(4), send
                        )
                    carry = carry + readlane(T.i32, inclusive, WAVE - 1)
            comm_ops.waitcnt_stores()
            fx.barrier()

            if const_expr(dense_comb):
                # Each src row of the matrix becomes its inclusive prefix in
                # place, one warp per row. Row src then gives both src's
                # local_base (its entry one segment back) and its count in a
                # segment (the difference of two neighbours).
                if warp < npes:
                    row_base = lds_matrix + fx.Int64(warp) * fx.Int64(segs * 4)
                    carry = arith.constant(0)
                    for chunk in range(0, segs, WAVE):
                        s = fx.Int32(chunk) + lane
                        in_seg = s < fx.Int32(segs)
                        s_addr = row_base + fx.Int64(
                            in_seg.select(s, fx.Int32(0))
                        ) * fx.Int64(4)
                        cnt = arith.constant(0)
                        if in_seg:
                            cnt = comm_ops.load_i32_lds(s_addr)
                        inclusive = _wave32_inclusive_scan_i32(cnt, lane)
                        if in_seg:
                            comm_ops.store_i32_lds(s_addr, carry + inclusive)
                        carry = carry + readlane(T.i32, inclusive, WAVE - 1)
                fx.barrier()

                def incl_at(src, seg):
                    return comm_ops.load_i32_lds(
                        lds_matrix
                        + (fx.Int64(src) * fx.Int64(segs) + fx.Int64(seg)) * fx.Int64(4)
                    )

                def excl_at(src, seg):
                    has_prev = seg > fx.Int32(0)
                    prev = incl_at(src, has_prev.select(seg - 1, fx.Int32(0)))
                    return has_prev.select(prev, fx.Int32(0))

                # Run table for gemm2: src's rows of local expert e start at
                # estart[e] + (rows of lower srcs) and map onto src's slots
                # local_base_src[seg] + i, so their slot is row + delta.
                for item in range(tid, epr * npes, plan_threads):
                    e = item // fx.Int32(npes)
                    src = item - e * fx.Int32(npes)
                    seg = fx.Int32(rank * epr) + e
                    prefix = arith.constant(0)
                    for lower in range_constexpr(npes):
                        cnt = incl_at(lower, seg) - excl_at(lower, seg)
                        prefix = prefix + (fx.Int32(lower) < src).select(
                            cnt, fx.Int32(0)
                        )
                    row_start = (
                        comm_ops.load_i32_lds(lds_estart + fx.Int64(e) * fx.Int64(4))
                        + prefix
                    )
                    tab = fx.Int32(epr) + item * fx.Int32(2)
                    buffer_store(row_start, rsrc_psum, tab)
                    buffer_store(
                        excl_at(src, seg) - row_start, rsrc_psum, tab + fx.Int32(1)
                    )

        if const_expr(plan_blocks > 1):
            if tid == 0:
                gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
                next_gen = gen + arith.constant(1)
                arrive = comm_ops.atomic_add_system(addr_barrier, arith.constant(1))
                if arrive != plan_blocks - 1:
                    comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(4), next_gen)
                    comm_ops.fence_agent_acquire()
                else:
                    comm_ops.fence_system_release()
                    buffer_store(arith.constant(0), rsrc_bar, 0)
                    buffer_store(next_gen, rsrc_bar, 2)
                    comm_ops.fence_agent_release()
                    buffer_store(next_gen, rsrc_bar, 1)
            fx.barrier()

        for route in range(bid * plan_threads + tid, n_routes, plan_threads):
            if const_expr(merge_routes):
                packed = comm_ops.load_i32_lds(
                    lds_routes + fx.Int64(route) * fx.Int64(4)
                )
            else:
                packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
            valid = packed >= 0
            segment = packed & arith.constant(0xFFFF)
            intra = packed >> arith.constant(16)
            dest_pe = segment // fx.Int32(epr)
            send_base = comm_ops.load_i32_lds(
                lds_hist + fx.Int64(segment) * fx.Int64(4)
            )
            dest_row = send_base + intra
            in_cap = dest_row < compact_cap
            flat = (dest_row << fx.Int32(peer_bits)) | (dest_pe & fx.Int32(peer_mask))
            buffer_store(
                arith.select(valid & in_cap, flat, arith.constant(dropped)),
                rsrc_map,
                route,
            )
            if const_expr(dense_comb):
                own_seg = valid.select(segment, fx.Int32(0))
                has_prev = own_seg > fx.Int32(0)
                local_base = comm_ops.load_i32_lds(
                    lds_matrix
                    + (
                        fx.Int64(rank * segs)
                        + fx.Int64(has_prev.select(own_seg - 1, fx.Int32(0)))
                    )
                    * fx.Int64(4)
                )
                local_base = has_prev.select(local_base, fx.Int32(0))
                buffer_store(
                    arith.select(
                        valid & in_cap, local_base + intra, arith.constant(dropped)
                    ),
                    rsrc_cslot,
                    route,
                )

    @flyc.jit
    def launch(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        addr_comb_slot: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        kernel(
            arena,
            addr_inp_idx,
            addr_tok_map,
            addr_block_hist,
            addr_send_base,
            addr_masked_m,
            addr_psum,
            addr_barrier,
            addr_comb_slot,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(plan_blocks, 1, 1),
            block=[plan_threads, 1, 1],
            stream=stream,
        )

    return launch
