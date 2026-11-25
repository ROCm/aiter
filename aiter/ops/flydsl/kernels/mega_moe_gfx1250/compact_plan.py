# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Send-side compact route plan for gfx1250 TDM dispatch.

A low-LDS single-block kernel counts every local route into
``(dest_rank, local_expert)`` buckets and pushes the histogram to every peer
(``tdm_compact_plan_arrive``); a separate one-wave kernel carries the only
cross-rank wait (``tdm_compact_plan_sync``); a final kernel reads the allgathered
histograms and writes the destination compact row into ``tok_map``
(``tdm_compact_plan_finish``). Dispatch copies payload straight onto that row,
so the receiver never runs ``moe_route_g2l_lds``.
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
):
    """Compile the compact plan as three back-to-back kernels on one stream.

    ``tdm_compact_plan_arrive``  tally local routes, push the histogram to every
                                 peer and bump each peer's ``done`` counter.
                                 Never waits on a peer.
    ``tdm_compact_plan_sync``    the only cross-rank wait: one wave spins until
                                 every peer has arrived.
    ``tdm_compact_plan_finish``  read the allgathered histograms, derive the
                                 compact rows and rewrite ``tok_map``.

    The wait lives in its own tiny kernel so the two compute kernels never sit
    spinning on a peer while holding their LDS and waves; a slow peer is charged
    to the sync kernel instead of to the plan. Kernel retirement is
    stream-ordered, so ``finish`` needs no in-kernel handshake with ``sync``.

    LDS does not survive a kernel boundary, so the packed per-route
    ``segment | intra << 16`` word goes through ``tok_map`` and ``finish`` reads
    the histograms back out of the arena.

    ``hist_stride`` is ``npes * row_dwords``. ``hist_pingpong``: the
    single-buffer protocol indexes the 2-deep hist/done arena by ``gen & 1``.
    Double-buffered callers pass a slot-specific ``off_hist`` / ``off_done`` and
    set this False so two in-flight plans do not share a done counter.
    """
    if WAVE != 32:
        raise ValueError("compact plan requires gfx1250 wave32")
    if PLAN_BLOCKS != 1:
        raise ValueError("split compact plan kernels are single-block")
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
    plan_waves = compact_plan_waves(npes=npes, max_routes=max_routes)
    plan_threads = plan_waves * WAVE
    sync_threads = WAVE
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
    if max_routes % 4 == 0:
        route_vec = 4
    elif max_routes % 2 == 0:
        route_vec = 2
    else:
        route_vec = 1
    dropped = -1

    def _hist_off(gen):
        if hist_pingpong:
            return off_hist + (gen & arith.constant(1)) * hist_stride * 4
        return off_hist

    @flyc.kernel(name="tdm_compact_plan_arrive", known_block_size=[plan_threads, 1, 1])
    def arrive_kernel(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_barrier: Int64,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        warp = tid >> LOG2_WAVE
        window = cco.Window(arena)

        rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)

        smem = fx.SharedAllocator(static=False)
        hist_ptr = smem.allocate(segs * 4, 128)._ptr
        lds_hist = fx.Int64(fx.ptrtoint(hist_ptr))
        if const_expr(use_sparse):
            pack_ptr = smem.allocate(row_dwords * 4, 128)._ptr
            lds_pack = fx.Int64(fx.ptrtoint(pack_ptr))

        for s in range(tid, segs, plan_threads):
            comm_ops.store_i32_lds(lds_hist + fx.Int64(s) * 4, arith.constant(0))
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
                    lds_hist + fx.Int64(segment) * 4, arith.constant(1)
                )
            packed = arith.select(
                valid,
                segment | (intra << arith.constant(16)),
                arith.constant(dropped),
            )
            buffer_store(packed, rsrc_map, route)

        vec_n = n_routes - (n_routes & fx.Int32(route_vec - 1))
        stride = plan_threads * route_vec
        for route in range(tid * route_vec, vec_n, stride):
            if const_expr(route_vec == 1):
                expert = buffer_load(rsrc_idx, route, vec_width=1, dtype=T.i32)
                _tally_one(route, expert)
            else:
                raw = fx.Vector(
                    buffer_load(rsrc_idx, route, vec_width=route_vec, dtype=T.i32)
                )
                for k in range_constexpr(route_vec):
                    _tally_one(route + k, raw[k])
        for route in range(vec_n + tid, n_routes, plan_threads):
            expert = buffer_load(rsrc_idx, route, vec_width=1, dtype=T.i32)
            _tally_one(route, expert)
        fx.barrier()

        if tid == 0:
            gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32) + arith.constant(1)
            buffer_store(gen, rsrc_bar, 2)
        fx.barrier()
        gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
        hist_off = _hist_off(gen)

        if const_expr(use_sparse):
            for s in range(tid, row_dwords, plan_threads):
                comm_ops.store_i32_lds(lds_pack + fx.Int64(s) * 4, arith.constant(0))
            fx.barrier()
            for s in range(tid, segs, plan_threads):
                cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * 4)
                if cnt != 0:
                    slot = comm_ops.atomic_add_lds(lds_pack, arith.constant(1))
                    packed = fx.Int32(s) | (cnt << arith.constant(16))
                    comm_ops.store_i32_lds(lds_pack + fx.Int64(slot + 1) * 4, packed)
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
        elif const_expr(segs % 32 == 0):
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
                for s in range(tid * hist_vec, segs, plan_threads * hist_vec):
                    vals = [
                        comm_ops.load_i32_lds(lds_hist + fx.Int64(s + i) * 4)
                        for i in range_constexpr(hist_vec)
                    ]
                    buffer_store(
                        fx.Vector.from_elements(vals, dtype=fx.Int32),
                        peer_hist_rsrc,
                        s,
                    )
            comm_ops.waitcnt_stores()
        fx.barrier()

        # Arrive: publish "my histogram is in your arena" to every peer. The
        # release fence orders the histogram stores above before the increment.
        if tid == 0:
            comm_ops.fence_system_release()
        fx.barrier()
        if tid < npes:
            comm_ops.atomic_add_system(
                fx.Int64(window.lsa_ptr(tid, off_done)),
                arith.constant(1),
            )
        comm_ops.waitcnt_stores()

    @flyc.kernel(name="tdm_compact_plan_sync", known_block_size=[sync_threads, 1, 1])
    def sync_kernel(
        arena: Int64,
        addr_barrier: Int64,
    ):
        """Wait until every peer's histogram has landed in this rank's arena."""
        tid = fx.thread_idx.x
        window = cco.Window(arena)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)
        if tid == 0:
            gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
            comm_ops.wait_i32_until_equals(
                fx.Int64(window.lsa_ptr(rank, off_done)),
                gen * fx.Int32(npes),
            )
            comm_ops.fence_system_acquire()

    @flyc.kernel(name="tdm_compact_plan_finish", known_block_size=[plan_threads, 1, 1])
    def finish_kernel(
        arena: Int64,
        addr_tok_map: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        window = cco.Window(arena)

        rsrc_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_base = create_buffer_resource_from_addr(addr_send_base)
        rsrc_mm = create_buffer_resource_from_addr(addr_masked_m)
        rsrc_psum = create_buffer_resource_from_addr(addr_psum)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)

        smem = fx.SharedAllocator(static=False)
        send_ptr = smem.allocate(segs * 4, 128)._ptr
        total_ptr = smem.allocate(segs * 4, 16)._ptr
        pref_ptr = smem.allocate(segs * 4, 16)._ptr
        matrix_ptr = smem.allocate(npes * segs * 4, 128)._ptr
        lds_send = fx.Int64(fx.ptrtoint(send_ptr))
        lds_total = fx.Int64(fx.ptrtoint(total_ptr))
        lds_pref = fx.Int64(fx.ptrtoint(pref_ptr))
        lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))
        if const_expr(use_sparse):
            recv_ptr = smem.allocate(npes * row_dwords * 4, 128)._ptr
            lds_recv = fx.Int64(fx.ptrtoint(recv_ptr))

        n_routes = inp_cur_tok * fx.Int32(topk)
        gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
        hist_off = _hist_off(gen)

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
                comm_ops.store_i32_lds(lds_matrix + fx.Int64(s) * 4, arith.constant(0))
            fx.barrier()
            for src in range_constexpr(npes):
                src_base = fx.Int32(src * row_dwords)
                nnz = comm_ops.load_i32_lds(lds_recv + fx.Int64(src * row_dwords) * 4)
                for i in range(tid, nnz, plan_threads):
                    packed = comm_ops.load_i32_lds(
                        lds_recv + fx.Int64(src_base + i + 1) * 4
                    )
                    seg = packed & arith.constant(0xFFFF)
                    cnt = packed >> arith.constant(16)
                    comm_ops.store_i32_lds(
                        lds_matrix + fx.Int64(src * segs) * 4 + fx.Int64(seg) * 4,
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
                        lds_matrix + fx.Int64(s) * 4,
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
                    lds_matrix + fx.Int64(src * segs + dest * fx.Int32(epr) + e) * 4
                )
                if src == rank:
                    my_prefix = total
                total = total + cnt
            comm_ops.store_i32_lds(lds_total + fx.Int64(idx) * 4, total)
            comm_ops.store_i32_lds(lds_pref + fx.Int64(idx) * 4, my_prefix)
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
                    total = comm_ops.load_i32_lds(lds_total + fx.Int64(idx) * 4)
                    my_prefix = comm_ops.load_i32_lds(lds_pref + fx.Int64(idx) * 4)
                aligned = (total + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
                aligned = aligned * fx.Int32(tile_m)
                inclusive = _wave32_inclusive_scan_i32(aligned, lane)
                expert_start = carry + inclusive - aligned
                if in_expert:
                    send = expert_start + my_prefix
                    if dest == rank:
                        buffer_store(expert_start + total, rsrc_psum, e)
                    buffer_store(send, rsrc_base, idx)
                    comm_ops.store_i32_lds(lds_send + fx.Int64(idx) * 4, send)
                carry = carry + readlane(T.i32, inclusive, WAVE - 1)
        comm_ops.waitcnt_stores()
        fx.barrier()

        for route in range(tid, n_routes, plan_threads):
            packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
            valid = packed >= 0
            segment = packed & arith.constant(0xFFFF)
            intra = packed >> arith.constant(16)
            dest_pe = segment // fx.Int32(epr)
            send_base = comm_ops.load_i32_lds(lds_send + fx.Int64(segment) * 4)
            dest_row = send_base + intra
            in_cap = dest_row < compact_cap
            flat = (dest_row << fx.Int32(peer_bits)) | (dest_pe & fx.Int32(peer_mask))
            buffer_store(
                arith.select(valid & in_cap, flat, arith.constant(dropped)),
                rsrc_map,
                route,
            )

    @flyc.jit
    def launch(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,  # unused; kept so callers need not change
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        arrive_kernel(
            arena, addr_inp_idx, addr_tok_map, addr_barrier, inp_cur_tok
        ).launch(grid=(1, 1, 1), block=[plan_threads, 1, 1], stream=stream)
        sync_kernel(arena, addr_barrier).launch(
            grid=(1, 1, 1), block=[sync_threads, 1, 1], stream=stream
        )
        finish_kernel(
            arena,
            addr_tok_map,
            addr_send_base,
            addr_masked_m,
            addr_psum,
            addr_barrier,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(grid=(1, 1, 1), block=[plan_threads, 1, 1], stream=stream)

    return launch
