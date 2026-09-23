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
from dataclasses import dataclass

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

    The kernel is allgather-bound. Extra waves past ~8 only help the route
    tally, and a 32-wave block spent the wait parked on a full CU.
    """
    env = os.environ.get("AITER_TDM_COMPACT_PLAN_WAVES")
    if env:
        return max(int(npes), int(env))
    routes = max(1, int(max_routes))
    if routes <= 256:
        want = 4
    elif routes <= 2048:
        want = 8
    else:
        want = 16
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


def _tdm_dword_rows(n_dwords: int) -> tuple[int, int] | None:
    """Split a dense dword run into 128B-aligned TDM rows. Wider rows first."""
    n = int(n_dwords)
    if n <= 0 or n % 32 != 0:
        return None
    for dim0 in (256, 128, 64, 32):
        if n % dim0 == 0:
            return dim0, n // dim0
    return None


def compact_done_stride(*, npes: int) -> int:
    """Dwords in one plan slot's arrival row: one generation flag per source rank."""
    return int(npes)


def compact_done_nbytes(*, npes: int, slots: int = 2) -> int:
    """Per-source generation flags, ``slots`` deep for double-buffered plans."""
    return int(slots) * compact_done_stride(npes=npes) * 4


# ── Tile-ready contract ───────────────────────────────────────────────────────
# The plan already decides, per local expert, where its compact rows start and
# how many arrive (what it writes to ``masked_m`` / ``psum``). Cutting that into
# GEMM-sized M tiles is what turns "wait for the whole dispatch kernel" into
# "wait for this tile", i.e. the dispatch-parallel-GEMM1 overlap.
#
# The block lives in the SYMMETRIC arena because the arrival counters are bumped
# by the SENDER: a peer's dispatch writes a compact row into our window and has
# to tick the tile that row belongs to, so the counters must be peer-addressable.
#
# Per plan slot, all int32, dword offsets from the slot base:
#   [0] gen         generation these expectations belong to. Written LAST with a
#                   release and only ever forward, never cleared: a consumer
#                   waiting for its own generation therefore cannot latch the
#                   previous one's tiles, which is what makes a reused slot safe.
#   [1] n_tiles     tiles the published generation covers.
#   [2] align_m     rows per tile of the published generation. A step's tile
#                   follows its token bucket, so this is not a constant.
#   [3] rows        planned rows, clamped to the arena's compact row capacity.
#   [4] producers   producers (dispatch blocks) that finished pushing this slot.
#   [5] work        tiles a consumer has claimed -- the global task counter.
#   [6] queue_head  consumer cursor into ready_queue.
#   [7] queue_tail  producer cursor into ready_queue.
#   tile_expected[tile_cap]  arrivals a tile needs before it may be computed.
#                            One per ROW, so a producer ticks once per row it
#                            lands. 0 means the tile holds no row this
#                            generation and no producer may publish it.
#   tile_ready[tile_cap]     arrivals so far; a producer atomic-adds here and the
#                            one that closes a tile publishes it.
#   ready_queue[tile_cap]    completion plane, one entry per tile: either the
#                            tile ids in completion order (queue_head/tail) or
#                            the generation stamped at the tile's own id, for a
#                            consumer that claims dense ids off `work` instead.
#
# ``tile_cap`` is cut at ``COMPACT_TILE_MIN_M`` so one arena region serves every
# tile height a step may pick, while the kernel only clears and bounds the
# ``ceil(cap / tile_m)`` tiles the compiled height can actually index.
COMPACT_TILE_MIN_M = 16
COMPACT_TILE_CTRL_DWORDS = 8
(
    COMPACT_TILE_GEN_DW,
    COMPACT_TILE_NTILES_DW,
    COMPACT_TILE_ALIGN_M_DW,
    COMPACT_TILE_ROWS_DW,
    COMPACT_TILE_PRODUCERS_DW,
    COMPACT_TILE_WORK_DW,
    COMPACT_TILE_QHEAD_DW,
    COMPACT_TILE_QTAIL_DW,
) = range(COMPACT_TILE_CTRL_DWORDS)


@dataclass(frozen=True, slots=True)
class CompactTileLayout:
    """Dword geometry of one plan slot's tile-ready block."""

    compact_cap: int
    min_tile_m: int
    tile_cap: int
    ctrl_dwords: int
    expected_dw: int
    ready_dw: int
    queue_dw: int
    slot_dwords: int

    @property
    def slot_nbytes(self) -> int:
        return self.slot_dwords * 4

    def tiles_for(self, tile_m: int) -> int:
        """Tiles a step at ``tile_m`` may index. Never more than ``tile_cap``.

        A step's tile has to be a whole number of ``min_tile_m`` blocks, so that
        one index space (cut at the minimum) bounds every height.
        """
        tile_m = int(tile_m)
        if tile_m < self.min_tile_m or tile_m % self.min_tile_m:
            raise ValueError(
                f"tile_m={tile_m} must be a positive multiple of "
                f"min_tile_m={self.min_tile_m}"
            )
        tiles = (self.compact_cap + tile_m - 1) // tile_m
        if tiles > self.tile_cap:
            raise ValueError(
                f"tile_m={tile_m} needs {tiles} tiles, past the block's {self.tile_cap}"
            )
        return tiles


def compact_tile_layout(
    *, compact_cap: int, min_tile_m: int = COMPACT_TILE_MIN_M
) -> CompactTileLayout:
    """Lay out one plan slot's tile-ready block for ``compact_cap`` rows."""
    compact_cap = int(compact_cap)
    min_tile_m = int(min_tile_m)
    if compact_cap <= 0:
        raise ValueError(f"compact_cap must be positive, got {compact_cap}")
    if min_tile_m <= 0 or min_tile_m & (min_tile_m - 1):
        raise ValueError(
            f"min_tile_m must be a positive power of two, got {min_tile_m}"
        )
    # 32-dword granularity throughout, so every plane starts 128B aligned and a
    # future consumer can TDM whole rows of counters.
    tile_cap = _align32((compact_cap + min_tile_m - 1) // min_tile_m)
    ctrl_dwords = _align32(COMPACT_TILE_CTRL_DWORDS)
    expected_dw = ctrl_dwords
    ready_dw = expected_dw + tile_cap
    queue_dw = ready_dw + tile_cap
    return CompactTileLayout(
        compact_cap=compact_cap,
        min_tile_m=min_tile_m,
        tile_cap=tile_cap,
        ctrl_dwords=ctrl_dwords,
        expected_dw=expected_dw,
        ready_dw=ready_dw,
        queue_dw=queue_dw,
        slot_dwords=queue_dw + tile_cap,
    )


def compact_tile_nbytes(
    *, compact_cap: int, slots: int = 2, min_tile_m: int = COMPACT_TILE_MIN_M
) -> int:
    """Arena bytes for ``slots`` tile-ready blocks."""
    slots = int(slots)
    if slots <= 0:
        raise ValueError(f"slots must be positive, got {slots}")
    layout = compact_tile_layout(compact_cap=compact_cap, min_tile_m=min_tile_m)
    return slots * layout.slot_nbytes


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
    off_tile: int | None = None,
    tile_min_m: int = COMPACT_TILE_MIN_M,
):
    """Compile the compact-plan kernel. ``hist_stride`` is ``npes * row_dwords``.

    ``hist_pingpong``: the single-buffer protocol indexes the 2-deep hist/done
    arena by ``gen & 1``. Double-buffered callers pass a slot-specific
    ``off_hist`` / ``off_done`` and set this False so two in-flight plans do
    not share a done counter.

    ``off_tile``: this slot's tile-ready block in the arena (see
    ``compact_tile_layout``). None leaves the contract out of the kernel
    entirely, which is the old plan, byte for byte.
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
    plan_blocks = PLAN_BLOCKS
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
    hist_tdm = _tdm_dword_rows(segs)
    matrix_tdm = _tdm_dword_rows(npes * segs)
    hist_tdm_dim0, hist_tdm_dim1 = hist_tdm if hist_tdm else (0, 0)
    matrix_tdm_dim0, matrix_tdm_dim1 = matrix_tdm if matrix_tdm else (0, 0)
    tile_ready = off_tile is not None
    tile_count = 0
    tile_exp_dw = tile_rdy_dw = tile_q_dw = 0
    if tile_ready:
        off_tile = int(off_tile)
        # 0 is a real arena offset, but never this block's: the plan's own
        # regions come first, so a 0 here means an unbound region name.
        if off_tile <= 0 or off_tile % 128:
            raise ValueError(
                "off_tile must be a positive 128-byte aligned arena offset, "
                f"got {off_tile}"
            )
        tile_layout = compact_tile_layout(
            compact_cap=compact_cap, min_tile_m=tile_min_m
        )
        tile_count = tile_layout.tiles_for(tile_m)
        tile_exp_dw = tile_layout.expected_dw
        tile_rdy_dw = tile_layout.ready_dw
        tile_q_dw = tile_layout.queue_dw
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

        smem = fx.SharedAllocator(static=False)
        # One dword past the segments is the tally's trash slot: a dropped route
        # bumps that instead of branching around the atomic. Nothing reads it.
        hist_ptr = smem.allocate((segs + 1) * 4, 128)._ptr
        lds_hist = fx.Int64(fx.ptrtoint(hist_ptr))
        if const_expr(merge_routes):
            routes_ptr = smem.allocate(max_routes * 4, 16)._ptr
            lds_routes = fx.Int64(fx.ptrtoint(routes_ptr))
        if const_expr(use_sparse):
            matrix_ptr = smem.allocate(npes * segs * 4, 128)._ptr
            pack_ptr = smem.allocate(row_dwords * 4, 128)._ptr
            recv_ptr = smem.allocate(npes * row_dwords * 4, 128)._ptr
            lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))
            lds_pack = fx.Int64(fx.ptrtoint(pack_ptr))
            lds_recv = fx.Int64(fx.ptrtoint(recv_ptr))
        else:
            matrix_ptr = smem.allocate(npes * segs * 4, 128)._ptr
            lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))
        for s in range(tid, segs + 1, plan_threads):
            comm_ops.store_i32_lds(
                lds_hist + fx.Int64(s) * fx.Int64(4), arith.constant(0)
            )
        fx.barrier()

        n_routes = inp_cur_tok * fx.Int32(topk)

        def _tally_bump(expert):
            """Claim this route's slot in its segment. Returns what emit needs.

            The atomic is unconditional -- an ``if valid`` around it puts every
            unrolled route in its own exec-masked block, and the block boundary
            forces ``ds_add_rtn`` to retire before the next one issues. Dropped
            routes bump the trash slot and are masked out in emit instead, so
            the ``route_vec`` atomics stay in flight together.
            """
            dest_pe = expert // epr
            valid = (expert >= 0) & (dest_pe >= 0) & (dest_pe < npes)
            local_e = expert - dest_pe * fx.Int32(epr)
            segment = dest_pe * fx.Int32(epr) + local_e
            slot = arith.select(valid, segment, arith.constant(segs))
            intra = comm_ops.atomic_add_lds(
                lds_hist + fx.Int64(slot) * fx.Int64(4), arith.constant(1)
            )
            return valid, segment, intra

        def _tally_emit(route, valid, segment, intra):
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

        def _tally_one(route, expert):
            valid, segment, intra = _tally_bump(expert)
            _tally_emit(route, valid, segment, intra)

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
                # Every atomic first, then every store: consuming each
                # ``intra`` right after its own atomic drains the LDS counter
                # once per route instead of once per vector.
                pending = []
                for k in range_constexpr(route_vec):
                    pending.append(_tally_bump(raw[k]))
                for k in range_constexpr(route_vec):
                    valid, segment, intra = pending[k]
                    _tally_emit(route + k, valid, segment, intra)
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
                done_off = off_done + parity * npes * 4
            else:
                hist_off = off_hist
                done_off = off_done

            if const_expr(tile_ready):
                tile_base = fx.Int64(window.lsa_ptr(my_lsa_rank, off_tile))
                rsrc_tile = create_buffer_resource_from_addr(tile_base)
                # Clear this slot's arrival state BEFORE the done flag below.
                # A peer cannot dispatch -- and so cannot bump a counter in our
                # window -- until it has seen that flag from every source, so
                # nothing lands between this clear and the expectations written
                # after the allgather.
                for dw in range(tid, tile_count, plan_threads):
                    zero = arith.constant(0)
                    buffer_store(zero, rsrc_tile, fx.Int32(tile_exp_dw) + dw)
                    buffer_store(zero, rsrc_tile, fx.Int32(tile_rdy_dw) + dw)
                    buffer_store(zero, rsrc_tile, fx.Int32(tile_q_dw) + dw)
                if tid < COMPACT_TILE_CTRL_DWORDS - 1:
                    # Every control dword but [0] gen. The generation is
                    # monotonic; zeroing it would hand a waiting consumer a
                    # generation it has already been through.
                    buffer_store(
                        arith.constant(0),
                        rsrc_tile,
                        fx.Int32(COMPACT_TILE_GEN_DW + 1) + tid,
                    )
                comm_ops.waitcnt_stores()
                fx.barrier()

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
                    TDM.tdm_wait(0)
            elif const_expr(plan_blocks == 1 and hist_tdm_dim0 > 0):
                if warp < npes:
                    peer_hist = fx.Int64(window.lsa_ptr(warp, hist_off)) + fx.Int64(
                        rank * segs * 4
                    )
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_hist)), peer_hist
                        ),
                        TDM.tdm_group1(hist_tdm_dim0, hist_tdm_dim1, 4),
                    )
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
                # One release store per peer, no remote atomic RMW on a shared
                # counter. Each source owns its own flag dword.
                comm_ops.store_i32_system(
                    fx.Int64(window.lsa_ptr(tid, done_off)),
                    arith.constant(rank),
                    gen,
                )
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid < npes:
                # Relaxed poll. An acquire load here compiles to a
                # SCOPE_SYS global_inv on *every* iteration, so the spin keeps
                # flushing the caches it is about to read through. The single
                # acquire fence below is what orders the peer's payload.
                comm_ops.spin_until_ge_i32_system(
                    fx.Int64(window.lsa_ptr(rank, done_off))
                    + fx.Int64(tid) * fx.Int64(4),
                    gen,
                    acquire=False,
                    sleep=False,
                )
            fx.barrier()
            if tid == 0:
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
                if const_expr(matrix_tdm_dim0 > 0):
                    TDM.tdm_load(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_matrix)),
                            fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off)),
                        ),
                        TDM.tdm_group1(matrix_tdm_dim0, matrix_tdm_dim1, 4),
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
                        for src in range_constexpr(npes):
                            cnt = comm_ops.load_i32_lds(
                                lds_matrix
                                + fx.Int64(
                                    src * segs + dest * fx.Int32(epr) + safe_e
                                )
                                * fx.Int64(4)
                            )
                            if src == rank:
                                my_prefix = total
                            total = total + cnt
                    aligned = (total + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
                    aligned = aligned * fx.Int32(tile_m)
                    inclusive = _wave32_inclusive_scan_i32(aligned, lane)
                    expert_start = carry + inclusive - aligned
                    if in_expert:
                        send = expert_start + my_prefix
                        if dest == rank:
                            buffer_store(total, rsrc_mm, e)
                            buffer_store(expert_start + total, rsrc_psum, e)
                            if const_expr(tile_ready):
                                # Rows landing in OUR window, so this is the one
                                # branch that owns a tile expectation.
                                #
                                # The scan aligns every expert_start to tile_m,
                                # so an expert owns whole tiles -- and exactly
                                # ceil(rows/tile_m) of them, which is why the
                                # published tile range stays dense. The full
                                # ones take tile_m and the last takes the
                                # remainder; charging that one a full tile_m
                                # would ask for the alignment padding, which
                                # nobody sends.
                                #
                                # Rows past the arena's capacity are dropped by
                                # _finish_one, so they are not expected either --
                                # counting them would leave the tile short
                                # forever and hang a polling consumer.
                                room = fx.Int32(compact_cap) - expert_start
                                room = (room > fx.Int32(0)).select(room, fx.Int32(0))
                                live = (total < room).select(total, room)
                                first = expert_start // fx.Int32(tile_m)
                                n_tile = (live + fx.Int32(tile_m - 1)) // fx.Int32(
                                    tile_m
                                )
                                for ti in range(n_tile):
                                    left = live - fx.Int32(ti) * fx.Int32(tile_m)
                                    rows_ti = (left < fx.Int32(tile_m)).select(
                                        left, fx.Int32(tile_m)
                                    )
                                    buffer_store(
                                        rows_ti,
                                        rsrc_tile,
                                        fx.Int32(tile_exp_dw) + first + fx.Int32(ti),
                                    )
                        buffer_store(send, rsrc_base, idx)
                        comm_ops.store_i32_lds(
                            lds_hist + fx.Int64(idx) * fx.Int64(4), send
                        )
                    carry = carry + readlane(T.i32, inclusive, WAVE - 1)
                if const_expr(tile_ready):
                    # Publish: geometry first, then `gen` with a release, so a
                    # consumer that has seen this generation also sees every
                    # expectation written above it. `carry` is this dest's total
                    # aligned rows and is wave-uniform (readlane broadcast).
                    #
                    # Clamping the row count is what keeps tiles
                    # [0, n_tiles) dense even when the plan overflows the
                    # arena: rows past the cap belong to experts that got no
                    # tile at all, and the expert straddling the cap still owns
                    # every tile up to it.
                    comm_ops.waitcnt_stores()
                    if (dest == rank) & (lane == 0):
                        rows = (carry < fx.Int32(compact_cap)).select(
                            carry, fx.Int32(compact_cap)
                        )
                        buffer_store(
                            (rows + fx.Int32(tile_m - 1)) // fx.Int32(tile_m),
                            rsrc_tile,
                            COMPACT_TILE_NTILES_DW,
                        )
                        buffer_store(
                            arith.constant(tile_m),
                            rsrc_tile,
                            COMPACT_TILE_ALIGN_M_DW,
                        )
                        buffer_store(rows, rsrc_tile, COMPACT_TILE_ROWS_DW)
                        comm_ops.waitcnt_stores()
                        comm_ops.fence_system_release()
                        comm_ops.store_i32_system(
                            tile_base, arith.constant(COMPACT_TILE_GEN_DW), gen
                        )
            comm_ops.waitcnt_stores()
            fx.barrier()

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

        def _finish_one(route, packed):
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

        vec_n = n_routes - (n_routes & fx.Int32(route_vec - 1))
        wstride = plan_blocks * plan_threads * route_vec
        for route in range(
            bid * plan_threads * route_vec + tid * route_vec, vec_n, wstride
        ):
            if const_expr(route_vec == 1):
                if const_expr(merge_routes):
                    packed = comm_ops.load_i32_lds(
                        lds_routes + fx.Int64(route) * fx.Int64(4)
                    )
                else:
                    packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
                _finish_one(route, packed)
            elif const_expr(merge_routes):
                for k in range_constexpr(route_vec):
                    packed = comm_ops.load_i32_lds(
                        lds_routes + fx.Int64(route + k) * fx.Int64(4)
                    )
                    _finish_one(route + k, packed)
            else:
                raw = fx.Vector(
                    buffer_load(rsrc_map, route, vec_width=route_vec, dtype=T.i32)
                )
                for k in range_constexpr(route_vec):
                    _finish_one(route + k, raw[k])
        for route in range(
            vec_n + bid * plan_threads + tid, n_routes, plan_blocks * plan_threads
        ):
            if const_expr(merge_routes):
                packed = comm_ops.load_i32_lds(
                    lds_routes + fx.Int64(route) * fx.Int64(4)
                )
            else:
                packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
            _finish_one(route, packed)

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
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(plan_blocks, 1, 1),
            block=[plan_threads, 1, 1],
            stream=stream,
        )

    return launch
