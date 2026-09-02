# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 compact EP dispatch emitters for the gfx1250 MegaMoE pipeline.

This is the compact (destination-owned row plan) counterpart of
``mega_moe.dispatch``.  It is deliberately a collection of emitters rather than
a launch wrapper: the persistent kernel in ``mxfp4_preshuffle_gfx1250_tdm.py``
can assign its first arrival to :func:`emit_compact_planner` and subsequent
arrivals to :func:`emit_compact_payload`; every workgroup then rejoins the work
queue as a consumer and gates each tile on :func:`emit_compact_wait_tile`.

Large token buckets use a fixed source-major landing layout. A wire row is sent
once per distinct ``(token, destination peer)`` and destination-local gather
expands it into every final grouped row. Small buckets retain direct-to-grouped
stores because their fixed gather cost exceeds the saved fabric bytes. All
symmetric addresses use ``Window.lsa_ptr``; there is no P2P pointer table.

Synchronization invariants
--------------------------
* ``expected`` is a non-zero generation and ``parity == generation & 1``.
  ``count_done``, ``plan_ready``, ``pair_ready`` and ``landing_done`` carry it.
  ``group_done`` joins payload producers; ``metadata_done`` joins rowmap writers.
* A source publishes its count-matrix stores with a system release store to
  ``count_done[parity, source]``.  A destination acquires every source slot
  before computing row bases.
* The destination clears its current-parity tile-row counters before publishing
  ``plan_ready``. Payload producers cannot race that clear:
  they wait for that destination's plan-ready generation first.
* In dedup mode, source rank S stores token T at fixed landing slot
  ``S*max_tokens+T``. After all source metadata is visible, up to 32 local
  gather workgroups materialize grouped payload/scale rows.
* Gatherers release tile-row counters only after local TDM stores drain, so the
  existing tile counter remains GEMM's only readiness gate.

The rowmap already encodes ``source*max_tokens*topk + token*topk + k_slot``.
Gather divides that value by ``topk`` to recover the landing slot, avoiding a
second per-route map and its remote traffic. Row-major scales ride the TDM
engine with payload; the forced interleaved fallback reads scales from landing
and applies the existing WMMA address transform locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import (
    ballot,
    ds_bpermute,
    readfirstlane,
    readlane,
    update_dpp,
)
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.communication_ops_utils import traced
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _scale_row_dword_base,
)

from . import tdm_prims as TDM

WAVE = 32
LANE_MASK = WAVE - 1
LOG2_WAVE = 5
DEFAULT_WORK_HEADS = 8
DEFAULT_ALIGNMENT = 128
TILE_READY_GRANULARITY = 16


def _dedup_enabled(max_tokens: int) -> bool:
    raw = os.environ.get("AITER_STAGE1_DEDUP", "auto").strip().lower()
    if raw == "auto":
        return max_tokens >= 512
    if raw not in ("0", "1"):
        raise ValueError("AITER_STAGE1_DEDUP must be auto, 0, or 1")
    return raw == "1"


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CompactWorkspaceLayout:
    """Host-side byte layout for one rank's symmetric compact workspace.

    Build with :meth:`make`, allocate ``nbytes`` bytes at the same
    ``arena_offset`` on every rank, and pass the returned offsets unchanged to
    the emitters.  All listed integer arrays are int32 except ``entry_ticket``
    (int64).  ``max_pairs`` is normally ``max_tokens * topk``.

    ``work_heads`` are cache-line separated because they are intended for the
    persistent GEMM work queue.  ``epoch_gate`` is the generation gate paired
    with the never-reset 64-bit ``entry_ticket``.
    """

    total_experts: int
    npes: int
    experts_per_rank: int
    max_pairs: int
    tile_ready_capacity: int
    work_head_count: int
    alignment: int
    local_hist: int
    count_matrix: int
    count_done: int
    task_row_base: int
    pair_base: int
    local_cursor: int
    pair_order: int
    pair_ready: int
    group_done: int
    metadata_done: int
    landing_done: int
    plan_ready: int
    payload_ready: int
    tile_rows_ready: int
    launch_ready: int
    entry_ticket: int
    epoch_gate: int
    work_heads: int
    nbytes: int

    @classmethod
    def make(
        cls,
        *,
        npes: int,
        experts_per_rank: int,
        max_tokens: int,
        topk: int,
        max_rows: int,
        work_head_count: int = DEFAULT_WORK_HEADS,
        alignment: int = DEFAULT_ALIGNMENT,
    ) -> "CompactWorkspaceLayout":
        """Compute all local offsets and the required symmetric allocation size."""
        if npes <= 0 or experts_per_rank <= 0:
            raise ValueError("npes and experts_per_rank must be positive")
        if max_tokens < 0 or topk <= 0:
            raise ValueError("max_tokens must be non-negative and topk positive")
        if max_rows <= 0 or max_rows % TILE_READY_GRANULARITY:
            raise ValueError("max_rows must be positive and 16-row aligned")
        if work_head_count <= 0:
            raise ValueError("work_head_count must be positive")
        if alignment < 8 or alignment & (alignment - 1):
            raise ValueError("alignment must be a power of two and at least 8")

        total_experts = npes * experts_per_rank
        max_pairs = max_tokens * topk
        tile_ready_capacity = max_rows // TILE_READY_GRANULARITY
        cursor = 0
        offsets: dict[str, int] = {}

        def put(name: str, size: int, align: int = alignment) -> None:
            nonlocal cursor
            cursor = _align(cursor, align)
            offsets[name] = cursor
            cursor += size

        put("local_hist", total_experts * 4)
        put("count_matrix", npes * experts_per_rank * 4)
        put("count_done", 2 * npes * 4)
        put("task_row_base", total_experts * 4)
        put("pair_base", total_experts * 4)
        put("local_cursor", total_experts * 4)
        put("pair_order", max_pairs * 4)
        put("pair_ready", 2 * 4)
        # Producers count themselves in here once they have scattered their slice
        # of ``pair_order``; own cache line so the spin does not thrash the
        # neighbouring readiness flags.
        put("group_done", 2 * 4, alignment)
        put("metadata_done", 2 * 4, alignment)
        put("landing_done", 2 * npes * 4, alignment)
        put("plan_ready", 2 * npes * 4)
        put("payload_ready", 2 * experts_per_rank * 4)
        put("tile_rows_ready", 2 * tile_ready_capacity * 4)
        put("launch_ready", npes * 4)
        put("entry_ticket", 8, 8)
        put("epoch_gate", 4)
        # One cache line per queue shard/head.
        put("work_heads", work_head_count * alignment)
        return cls(
            total_experts=total_experts,
            npes=npes,
            experts_per_rank=experts_per_rank,
            max_pairs=max_pairs,
            tile_ready_capacity=tile_ready_capacity,
            work_head_count=work_head_count,
            alignment=alignment,
            nbytes=_align(cursor, alignment),
            **offsets,
        )

    def offset(self, name: str) -> int:
        """Return one named byte offset, useful to generic host binders."""
        offset_names = {
            "local_hist",
            "count_matrix",
            "count_done",
            "task_row_base",
            "pair_base",
            "local_cursor",
            "pair_order",
            "pair_ready",
            "group_done",
            "metadata_done",
            "landing_done",
            "plan_ready",
            "payload_ready",
            "tile_rows_ready",
            "launch_ready",
            "entry_ticket",
            "epoch_gate",
            "work_heads",
        }
        if name not in offset_names:
            raise KeyError(name)
        return getattr(self, name)

    def absolute_offset(self, arena_offset: int, name: str) -> int:
        """Return ``arena_offset + local_offset`` for a symmetric arena region."""
        return arena_offset + self.offset(name)

    def validate_capacity(self, *, capacity_rows: int, tile_m: int) -> None:
        """Validate host-known structural bounds.

        Runtime route counts are checked by the planner and reported through
        ``num_valid[1]``.  A host can additionally guarantee no overflow by
        sizing ``capacity_rows`` for its routing policy.
        """
        if capacity_rows <= 0:
            raise ValueError("capacity_rows must be positive")
        if tile_m <= 0 or tile_m & (tile_m - 1):
            raise ValueError("tile_m must be a positive power of two")
        if capacity_rows % tile_m:
            raise ValueError("capacity_rows must be tile_m aligned")


def compact_workspace_layout(**kwargs) -> CompactWorkspaceLayout:
    """Convenience alias for :meth:`CompactWorkspaceLayout.make`."""
    return CompactWorkspaceLayout.make(**kwargs)


def compact_payload_lds_bytes(*, wire_stride: int, num_waves: int) -> int:
    """LDS bytes required by :func:`emit_compact_payload`: one wire row per wave.

    The producer owns a private LDS region, disjoint from the GEMM A/B/C arena,
    so every extra tile it stages costs GEMM occupancy.  One is enough: the walk
    is token-major, so a staged row feeds all of that token's remote stores
    before the next token overwrites it, and deeper batching bought nothing --
    two tiles per wave measured 912us against 743us.
    """
    if wire_stride <= 0 or num_waves <= 0:
        raise ValueError("wire_stride and num_waves must be positive")
    return _align(wire_stride, 128) * num_waves


def _local(window, rank: int, arena_offset: int, local_offset: int):
    return fx.Int64(window.lsa_ptr(fx.Int32(rank), arena_offset + local_offset))


def _peer(window, peer, arena_offset: int, local_offset: int):
    return fx.Int64(window.lsa_ptr(peer, arena_offset + local_offset))


def _rsrc(addr):
    return buffer_ops.create_buffer_resource_from_addr(addr)


def _load_i32(addr, index):
    return fx.Int32(buffer_ops.buffer_load(_rsrc(addr), index, vec_width=1, dtype=T.i32))


def _store_i32(addr, index, value):
    buffer_ops.buffer_store(fx.Int32(value), _rsrc(addr), index)


def _wave32_inclusive_scan_i32(value, lane):
    """Inclusive i32 scan over exactly one gfx1250 wave."""
    raw = value.ir_value()
    zero = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.Int32(update_dpp(T.i32, zero, raw, dpp, 0xF, 0xF, True))
        value = (lane >= fx.Int32(shift)).select(value + remote, value)
        raw = value.ir_value()
    # DPP scans each 16-lane row independently.  Every upper-half lane adds the
    # completed lower row (lane 15), not its lane-16 peer.
    remote16 = fx.Int32(readlane(T.i32, value, 15))
    return (lane >= fx.Int32(16)).select(value + remote16, value)


def _wave32_reduce_max_i32(value, lane):
    """Broadcast the maximum i32 value over one wave32."""
    for distance in (1, 2, 4, 8, 16):
        peer = fx.Int32(ds_bpermute(T.i32, (lane ^ fx.Int32(distance)) * 4, value))
        value = (peer > value).select(peer, value)
    return value


@traced
def emit_compact_planner(
    *,
    arena_handle,
    arena_offset: int,
    layout: CompactWorkspaceLayout,
    rank: int,
    npes: int,
    experts_per_rank: int,
    topk: int,
    max_tokens: int,
    tile_m: int,
    capacity_rows: int,
    num_waves: int,
    addr_in_idx,
    cur_tokens,
    m_tile_map_offset: int,
    num_valid_offset: int,
    parity,
    expected,
) -> None:
    """Emit compact counting, destination planning, and source route grouping.

    Parameters are compile-time Python integers except ``arena_handle``,
    ``addr_in_idx``, ``cur_tokens``, ``parity`` and ``expected``.  The caller
    must execute this body in exactly one workgroup for this rank.  The
    workgroup must contain ``num_waves * 32`` threads.

    ``addr_in_idx`` is flattened int32 ``[cur_tokens, topk]`` global expert ids.
    ``m_tile_map_offset`` names ``experts_per_rank`` int32 valid-row ends;
    ``num_valid_offset`` names at least two int32 values: padded row count and
    overflow status.  ``capacity_rows`` bounds every destination output array.
    Invalid expert ids are dropped.  ``cur_tokens <= max_tokens`` is a caller
    precondition.  On overflow the plan is still published (preventing peer
    deadlock), payload writes are suppressed, and ``num_valid[1] == 1``; the
    caller must not launch GEMM for that generation.
    """
    if npes != layout.npes or experts_per_rank != layout.experts_per_rank:
        raise ValueError("layout geometry does not match planner geometry")
    if layout.total_experts != npes * experts_per_rank:
        raise ValueError("layout total_experts mismatch")
    if layout.max_pairs < max_tokens * topk:
        raise ValueError("pair_order capacity is smaller than max_tokens*topk")
    if tile_m <= 0 or tile_m & (tile_m - 1):
        raise ValueError("tile_m must be a positive power of two")
    if capacity_rows <= 0 or capacity_rows % tile_m:
        raise ValueError("capacity_rows must be positive and tile_m aligned")
    if num_waves < 2:
        raise ValueError("compact planner needs at least two waves")

    total_experts = npes * experts_per_rank
    block_threads = num_waves * WAVE
    window = cco.Window(fx.Int64(arena_handle))
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(LANE_MASK)
    wave = tid >> fx.Int32(LOG2_WAVE)

    local_hist = _local(window, rank, arena_offset, layout.local_hist)
    count_matrix = _local(window, rank, arena_offset, layout.count_matrix)
    count_done = _local(window, rank, arena_offset, layout.count_done)
    task_row_base = _local(window, rank, arena_offset, layout.task_row_base)
    pair_base = _local(window, rank, arena_offset, layout.pair_base)
    local_cursor = _local(window, rank, arena_offset, layout.local_cursor)
    pair_ready = _local(window, rank, arena_offset, layout.pair_ready)
    group_done = _local(window, rank, arena_offset, layout.group_done)
    metadata_done = _local(window, rank, arena_offset, layout.metadata_done)
    landing_done = _local(window, rank, arena_offset, layout.landing_done)
    plan_ready = _local(window, rank, arena_offset, layout.plan_ready)
    payload_ready = _local(window, rank, arena_offset, layout.payload_ready)
    tile_rows_ready = _local(window, rank, arena_offset, layout.tile_rows_ready)
    map_addr = _local(window, rank, arena_offset, m_tile_map_offset)
    num_valid_addr = _local(window, rank, arena_offset, num_valid_offset)
    idx_rsrc = _rsrc(addr_in_idx)

    for i in range(tid, total_experts, block_threads):
        _store_i32(local_hist, i, 0)
        _store_i32(local_cursor, i, 0)
    if tid < fx.Int32(experts_per_rank):
        _store_i32(
            payload_ready,
            fx.Int32(parity) * fx.Int32(experts_per_rank) + tid,
            0,
        )
    if tid == fx.Int32(0):
        # Producers only touch this after they observe ``pair_ready``, which this
        # workgroup publishes strictly later, so zeroing it here cannot race.
        _store_i32(group_done, fx.Int32(parity), 0)
        _store_i32(metadata_done, fx.Int32(parity), 0)
    if tid < fx.Int32(npes):
        _store_i32(
            landing_done,
            fx.Int32(parity) * fx.Int32(npes) + tid,
            0,
        )
    active_tiles = capacity_rows // tile_m
    for tile in range(tid, active_tiles, block_threads):
        _store_i32(
            tile_rows_ready,
            fx.Int32(parity * layout.tile_ready_capacity) + tile,
            fx.Int32(0),
        )
    comm_ops.waitcnt_all()
    fx.barrier()
    comm_ops.fence_agent_release()

    route_limit = fx.Int32(cur_tokens) * fx.Int32(topk)
    for route in range(tid, route_limit, block_threads):
        expert = fx.Int32(
            buffer_ops.buffer_load(idx_rsrc, route, vec_width=1, dtype=T.i32)
        )
        valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(total_experts))
        if valid:
            comm_ops.atomic_add_agent(local_hist + fx.Int64(expert) * 4, fx.Int32(1))
    comm_ops.waitcnt_all()
    fx.barrier()
    comm_ops.fence_agent_acquire()

    # Source histogram -> every destination's source-major count matrix.
    for ge in range(tid, total_experts, block_threads):
        destination = ge // fx.Int32(experts_per_rank)
        local_expert = ge - destination * fx.Int32(experts_per_rank)
        remote_matrix = _peer(window, destination, arena_offset, layout.count_matrix)
        count = _load_i32(local_hist, ge)
        _store_i32(
            remote_matrix,
            fx.Int32(rank * experts_per_rank) + local_expert,
            count,
        )
    comm_ops.waitcnt_stores()
    fx.barrier()

    # Wave 0 owns destination planning.
    if wave == fx.Int32(0):
        comm_ops.fence_system_release()
        done_index = fx.Int32(parity) * fx.Int32(npes) + fx.Int32(rank)
        for destination in range(lane, npes, WAVE):
            remote_done = _peer(window, destination, arena_offset, layout.count_done)
            comm_ops.store_i32_system(remote_done, done_index, fx.Int32(expected))
        for source in range(lane, npes, WAVE):
            slot = fx.Int32(parity) * fx.Int32(npes) + source
            comm_ops.spin_until_eq_i32(count_done + fx.Int64(slot) * 4, expected)
        comm_ops.fence_system_acquire()

        row_carry = fx.Int32(0)
        for chunk in range_constexpr((experts_per_rank + WAVE - 1) // WAVE):
            local_expert = fx.Int32(chunk * WAVE) + lane
            live = local_expert < fx.Int32(experts_per_rank)
            safe_expert = live.select(local_expert, fx.Int32(0))
            global_expert = fx.Int32(rank * experts_per_rank) + local_expert
            source_counts = []
            total_count = fx.Int32(0)
            for source in range_constexpr(npes):
                count = _load_i32(
                    count_matrix,
                    fx.Int32(source * experts_per_rank) + safe_expert,
                )
                count = live.select(count, fx.Int32(0))
                source_counts.append(count)
                total_count = total_count + count
            tiles = (total_count + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
            padded = tiles * fx.Int32(tile_m)
            inclusive = _wave32_inclusive_scan_i32(padded, lane)
            row_base = row_carry + inclusive - padded

            source_prefix = fx.Int32(0)
            for source in range_constexpr(npes):
                if live:
                    remote_bases = _peer(
                        window, fx.Int32(source), arena_offset, layout.task_row_base
                    )
                    _store_i32(
                        remote_bases,
                        global_expert,
                        row_base + source_prefix,
                    )
                source_prefix = source_prefix + source_counts[source]
            if live:
                # Valid end (not padded end), matching contiguous-M m_tile_map.
                _store_i32(map_addr, local_expert, row_base + total_count)

            last_lane = min(WAVE - 1, experts_per_rank - chunk * WAVE - 1)
            row_carry = row_carry + fx.Int32(readlane(T.i32, inclusive, last_lane))

        if lane == fx.Int32(0):
            overflow = (row_carry > fx.Int32(capacity_rows)).select(
                fx.Int32(1), fx.Int32(0)
            )
            _store_i32(num_valid_addr, fx.Int32(0), row_carry)
            _store_i32(num_valid_addr, fx.Int32(1), overflow)
        comm_ops.waitcnt_stores()
        comm_ops.fence_system_release()
        for source in range(lane, npes, WAVE):
            remote_ready = _peer(
                window, fx.Int32(source), arena_offset, layout.plan_ready
            )
            ready_index = fx.Int32(parity) * fx.Int32(npes) + fx.Int32(rank)
            comm_ops.store_i32_system(remote_ready, ready_index, fx.Int32(expected))

    # Wave 1 builds the source-global exclusive prefix over every
    # (destination, expert) pair, one wave32 scan per 32 pairs.  Done serially on
    # a single lane this was ``total_experts`` dependent iterations -- 384 of them
    # at 4 ranks and 96 experts each, measured at 24us of pure latency that every
    # producer then waited on, since the payload walk cannot start until
    # ``pair_base`` exists.
    if wave == fx.Int32(1):
        carry = fx.Int32(0)
        for chunk in range_constexpr((total_experts + WAVE - 1) // WAVE):
            ge = fx.Int32(chunk * WAVE) + lane
            live = ge < fx.Int32(total_experts)
            count = _load_i32(local_hist, live.select(ge, fx.Int32(0)))
            count = live.select(count, fx.Int32(0))
            inclusive = _wave32_inclusive_scan_i32(count, lane)
            if live:
                exclusive = carry + inclusive - count
                _store_i32(pair_base, ge, exclusive)
                _store_i32(local_cursor, ge, exclusive)
            # Dead lanes contributed zero, so the last lane always carries the
            # chunk total whether or not the chunk is full.
            carry = carry + fx.Int32(readlane(T.i32, inclusive, WAVE - 1))
        comm_ops.waitcnt_stores()
        if lane == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(pair_ready, parity, expected)

    # The route grouping itself now runs on the producers (see
    # ``emit_compact_route_group``).  Scattering all ``cur_tokens * topk`` routes
    # from this one workgroup cost 48us that every producer waited through, while
    # the producers between them have two orders of magnitude more threads idle at
    # that moment.


@traced
def emit_compact_payload(
    *,
    arena_handle,
    arena_offset: int,
    layout: CompactWorkspaceLayout,
    rank: int,
    npes: int,
    experts_per_rank: int,
    topk: int,
    max_tokens_per_rank: int,
    num_waves: int,
    producer_blocks: int,
    producer_slot,
    lds_base_i32,
    addr_in_wire,
    addr_in_weights,
    addr_in_idx,
    cur_tokens,
    payload_offset: int,
    grouped_scale_offset: int,
    rowmap_offset: int,
    landing_offset: int,
    m_tile_map_offset: int,
    num_valid_offset: int,
    wire_stride: int,
    payload_bytes: int,
    scale_bytes: int,
    tile_m: int,
    wmma_rep: int,
    parity,
    expected,
    scale_rowmajor: bool = False,
) -> None:
    """Emit one compact payload-producer workgroup.

    ``lds_base_i32`` is a caller-owned, 128-byte-aligned raw LDS byte address
    with at least ``compact_payload_lds_bytes(...)`` bytes.  Accepting it rather
    than creating another ``SharedAllocator`` is what makes this emitter safe
    to inline into the grouped GEMM, which already owns the workgroup's LDS.

    ``addr_in_wire`` has ``max_tokens_per_rank`` rows of
    ``[payload_bytes | scale_bytes | optional wire padding]``.
    ``addr_in_weights`` is flattened f32 ``[tokens, topk]``.

    The first pass is token-major and groups routes while sending one wire row
    per distinct destination. The second is ``(destination, expert)``-major and
    writes coalesced rowmap metadata. After source generations arrive, up to 32
    workgroups gather local landing rows into grouped payload/scale, then publish
    tile readiness. Below the automatic dedup knee, the first pass writes final
    grouped rows directly and the second pass publishes readiness as before.

    ``srcmap[row]`` is ``(rank*max_tokens_per_rank + token) | (k_slot << 24)``.
    This is the format decoded by MegaMoE gemm2
    (low 24 bits source token, high 8 bits top-k slot); host validation must
    enforce ``npes*max_tokens_per_rank <= 2**24`` and ``topk <= 256``.
    """
    if npes != layout.npes or experts_per_rank != layout.experts_per_rank:
        raise ValueError("layout geometry does not match payload geometry")
    if producer_blocks < npes:
        raise ValueError("compact payload needs at least one producer per destination")
    if producer_blocks % npes:
        raise ValueError("producer_blocks must be divisible by npes")
    if wire_stride < payload_bytes + scale_bytes:
        raise ValueError("wire_stride is smaller than payload plus scales")
    if payload_bytes <= 0 or scale_bytes <= 0:
        raise ValueError("payload_bytes and scale_bytes must be positive")
    if scale_bytes % 4:
        raise ValueError("scale_bytes must be dword aligned")
    if wire_stride % 4 or payload_bytes % 4:
        raise ValueError("wire_stride and payload_bytes must be dword aligned")
    if wmma_rep <= 0 or tile_m % (wmma_rep * 16):
        raise ValueError("tile_m must be divisible by wmma_rep*16")
    if npes * max_tokens_per_rank > 1 << 24:
        raise ValueError("source-token encoding exceeds 24 bits")
    if max_tokens_per_rank <= 0 or (
        max_tokens_per_rank & (max_tokens_per_rank - 1)
    ):
        raise ValueError(
            "max_tokens_per_rank must be a positive power of two for gemm2 decode"
        )
    if topk > 1 << 8:
        raise ValueError("top-k slot encoding exceeds 8 bits")
    dedup_on = _dedup_enabled(max_tokens_per_rank)

    window = cco.Window(fx.Int64(arena_handle))
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(LANE_MASK)
    wave = tid >> fx.Int32(LOG2_WAVE)
    destination = fx.Int32(producer_slot) % fx.Int32(npes)
    destination_group = fx.Int32(producer_slot) // fx.Int32(npes)
    producers_per_destination = (producer_blocks + npes - 1) // npes

    pair_base = _local(window, rank, arena_offset, layout.pair_base)
    local_hist = _local(window, rank, arena_offset, layout.local_hist)
    local_cursor = _local(window, rank, arena_offset, layout.local_cursor)
    task_row_base = _local(window, rank, arena_offset, layout.task_row_base)
    pair_order = _local(window, rank, arena_offset, layout.pair_order)
    plan_ready = _local(window, rank, arena_offset, layout.plan_ready)
    pair_ready = _local(window, rank, arena_offset, layout.pair_ready)
    group_done = _local(window, rank, arena_offset, layout.group_done)
    metadata_done = _local(window, rank, arena_offset, layout.metadata_done)
    landing_done = _local(window, rank, arena_offset, layout.landing_done)
    landing_wire = _local(window, rank, arena_offset, landing_offset)
    local_rowmap = _local(window, rank, arena_offset, rowmap_offset)

    # A token's routes can land on any peer, so this block needs every
    # destination's row plan, not just the one it publishes readiness for.
    if tid == fx.Int32(0):
        for d in range_constexpr(npes):
            comm_ops.spin_until_eq_i32(
                plan_ready + fx.Int64(fx.Int32(parity) * fx.Int32(npes) + d) * 4,
                expected,
            )
        comm_ops.spin_until_eq_i32(pair_ready + fx.Int64(parity) * 4, expected)
    fx.barrier()
    comm_ops.fence_system_acquire()

    # One complete wire row per wave-tile.  128B alignment preserves
    # descriptor/LDS alignment even when wire_stride itself is not a power of
    # two.  One tile per wave is enough because the walk is token-major: every
    # store a staged row feeds is issued before it is reloaded, so the tile is
    # reused rather than double-buffered.
    tile_bytes = _align(wire_stride, 128)
    wire_desc = TDM.tdm_group1(wire_stride, 1, 1)
    payload_desc = TDM.tdm_group1(payload_bytes, 1, 1)
    # Row-major scales are one contiguous run per grouped row, so they can ride
    # the same TDM engine as the payload instead of the vector-memory path. That
    # engine choice, not the byte count, was the cost: a 224B ``buffer_store``
    # measured 154us against the 7168B TDM payload store's 74us, because the
    # vector store occupies the memory pipeline and has to be drained, while a
    # descriptor is handed to the mover and forgotten. The scales already sit in
    # LDS at ``tile + payload_bytes`` -- ``wire_desc`` loaded the whole wire row.
    scale_desc = TDM.tdm_group1(scale_bytes, 1, 1)
    # Scales are scattered (not TDM-stored) directly into the WMMA-interleaved
    # layout, folding ``wmma_rep`` consecutive rows into each scale tile.  This
    # must match ``moe_fused_route_quant_scatter`` byte-for-byte.
    rows_per_tile = wmma_rep * 16
    dst_scale_dwords_per_row = (scale_bytes // 4) * wmma_rep

    def wave_tile():
        """LDS byte address of this wave's wire tile."""
        return fx.Int32(lds_base_i32) + readfirstlane(T.i32, wave) * fx.Int32(
            tile_bytes
        )

    def emit_load(tile, source_token):
        TDM.tdm_load(
            TDM.tdm_group0(
                tile,
                fx.Int64(addr_in_wire)
                + fx.Int64(source_token) * fx.Int64(wire_stride),
            ),
            wire_desc,
        )

    def emit_landing(tile, source_token, dest_pe):
        """Store one wire row once for this (token, destination) pair."""
        slot = fx.Int32(rank * max_tokens_per_rank) + source_token
        TDM.tdm_store(
            TDM.tdm_group0(
                tile,
                _peer(window, dest_pe, arena_offset, landing_offset)
                + fx.Int64(slot) * fx.Int64(wire_stride),
            ),
            wire_desc,
        )

    def emit_direct_store(tile, source_token, destination_row, dest_pe):
        """Legacy direct-to-grouped row path used below the dedup knee."""
        TDM.tdm_store(
            TDM.tdm_group0(
                tile,
                _peer(window, dest_pe, arena_offset, payload_offset)
                + fx.Int64(destination_row) * fx.Int64(payload_bytes),
            ),
            payload_desc,
        )
        dst_scale = _peer(window, dest_pe, arena_offset, grouped_scale_offset)
        if const_expr(scale_rowmajor):
            TDM.tdm_store(
                TDM.tdm_group0(
                    tile + fx.Int32(payload_bytes),
                    dst_scale + fx.Int64(destination_row) * fx.Int64(scale_bytes),
                ),
                scale_desc,
            )
        else:
            scale_dwords = scale_bytes // 4
            wire_scale_dw_base = fx.Int32(source_token) * fx.Int32(
                wire_stride // 4
            ) + fx.Int32(payload_bytes // 4)
            for dw in range_constexpr(0, scale_dwords, WAVE):
                dword = fx.Int32(dw) + lane
                if dword < fx.Int32(scale_dwords):
                    value = buffer_ops.buffer_load(
                        _rsrc(fx.Int64(addr_in_wire)),
                        wire_scale_dw_base + dword,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    row_base = _scale_row_dword_base(
                        fx.Uint32(destination_row),
                        c_rows_per_tile=fx.Int32(rows_per_tile),
                        c_dst_scale_dwords_per_row=fx.Int32(dst_scale_dwords_per_row),
                        c16_i32=fx.Int32(16),
                    )
                    dst_dword = row_base + fx.Uint32(dword) * fx.Uint32(rows_per_tile)
                    buffer_ops.buffer_store(
                        value, _rsrc(dst_scale), fx.Int32(dst_dword)
                    )

    def emit_gather_store(tile, landing_slot, destination_row):
        """Materialize one local grouped row from its deduplicated landing row."""
        TDM.tdm_store(
            TDM.tdm_group0(
                tile,
                _local(window, rank, arena_offset, payload_offset)
                + fx.Int64(destination_row) * fx.Int64(payload_bytes),
            ),
            payload_desc,
        )
        # The e8m0 scales trail the payload in the wire row, so they arrive in LDS
        # with it and only the destination layout decides how they leave.
        dst_scale = _local(window, rank, arena_offset, grouped_scale_offset)
        scale_dwords = scale_bytes // 4
        if const_expr(scale_rowmajor):
            # Grouped row r's e8m0 run is contiguous at ``r*scale_bytes``, which
            # is exactly the shape already staged in LDS, so one descriptor moves
            # it.  The GEMM strided-reads element ``r*(scale_bytes//4)+k128``
            # back, so this lands where the WMMA-interleaved LDS load expects it.
            TDM.tdm_store(
                TDM.tdm_group0(
                    tile + fx.Int32(payload_bytes),
                    dst_scale + fx.Int64(destination_row) * fx.Int64(scale_bytes),
                ),
                scale_desc,
            )
        else:
            # The interleaved layout scatters with a ``rows_per_tile`` stride, so
            # it cannot be one contiguous run and stays on the vector path.
            wire_scale_dw_base = fx.Int32(landing_slot) * fx.Int32(
                wire_stride // 4
            ) + fx.Int32(payload_bytes // 4)
            for dw in range_constexpr(0, scale_dwords, WAVE):
                dword = fx.Int32(dw) + lane
                if dword < fx.Int32(scale_dwords):
                    value = buffer_ops.buffer_load(
                        _rsrc(landing_wire),
                        wire_scale_dw_base + dword,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    row_base = _scale_row_dword_base(
                        fx.Uint32(destination_row),
                        c_rows_per_tile=fx.Int32(rows_per_tile),
                        c_dst_scale_dwords_per_row=fx.Int32(dst_scale_dwords_per_row),
                        c16_i32=fx.Int32(16),
                    )
                    dst_dword = row_base + fx.Uint32(dword) * fx.Uint32(rows_per_tile)
                    buffer_ops.buffer_store(
                        value,
                        _rsrc(dst_scale),
                        fx.Int32(dst_dword),
                    )

    def emit_segment_rowmap(segment_row, segment_count, source_base, destination_base):
        """Write one segment's rowmap entries with the whole workgroup at once.

        A segment's destination rows are consecutive, so spreading them across
        threads turns what used to be one 8B store per row -- issued by lane 0 of
        whichever wave owned that row, 1/32 lane efficiency and a separate remote
        transaction each -- into a single coalesced burst. The rowmap does not
        depend on the payload bytes, so it only has to land before the segment's
        readiness publish, which the existing drain and fence already cover.
        """
        idx = tid
        while idx < segment_count:
            source_row = segment_row + idx
            route = _load_i32(pair_order, source_base + source_row)
            source_token = route // fx.Int32(topk)
            topk_slot = route - source_token * fx.Int32(topk)
            weight = buffer_ops.buffer_load(
                _rsrc(addr_in_weights), route, vec_width=1, dtype=T.f32
            )
            weight_bits = fx.Float32(weight).bitcast(fx.Int32)
            # Current gfx1250 GEMM2 consumes ep_rowmap directly:
            # (destination combine slot, route weight bits). The two fields are
            # adjacent, so they ship as one 8B element.
            source_encoding = (
                fx.Int32(rank * max_tokens_per_rank * topk)
                + source_token * fx.Int32(topk)
                + topk_slot
            )
            buffer_ops.buffer_store(
                vector.from_elements(
                    T.vec(2, T.i32), [source_encoding, weight_bits]
                ),
                _rsrc(_peer(window, destination, arena_offset, rowmap_offset)),
                (destination_base + source_row) * fx.Int32(2),
            )
            idx = idx + fx.Int32(num_waves * WAVE)

    # A destination whose plan overflowed its row capacity is already a failed
    # run that the host detects through ``num_valid[1]``, so the payload walk is
    # skipped wholesale rather than per destination: the walk is token-major and
    # a token's routes span peers, which would otherwise cost a peer load per
    # route just to re-derive a flag that is uniformly zero on every good run.
    overflow_any = fx.Int32(0)
    for d in range_constexpr(npes):
        overflow_any = overflow_any | _load_i32(
            _peer(window, fx.Int32(d), arena_offset, num_valid_offset), fx.Int32(1)
        )

    # Phase 1: group this rank's routes by global expert and move their payload,
    # in one token-major pass, one token per wave.
    #
    # The wire row is staged once per token and sent once per distinct
    # destination peer. Multiple routes of that token to experts on the same
    # peer share the fixed [source, token] landing row; phase 3 expands it into
    # the final grouped rows locally.
    #
    # Grouping is fused in rather than run as its own pass so that ``position``
    # stays in a register: the store address needs the route's rank within its
    # expert run, which is exactly what the grouping atomic returns. Splitting
    # them would need both an inverse ``route -> slot`` array and a second
    # ``producer_blocks`` barrier.
    total_experts = npes * experts_per_rank
    idx_rsrc = _rsrc(addr_in_idx)
    tile = wave_tile()
    token = fx.Int32(producer_slot) * fx.Int32(num_waves) + wave
    while token < fx.Int32(cur_tokens):
        route = token * fx.Int32(topk) + lane
        in_range = lane < fx.Int32(topk)
        expert = fx.Int32(
            buffer_ops.buffer_load(
                idx_rsrc,
                in_range.select(route, fx.Int32(0)),
                vec_width=1,
                dtype=T.i32,
            )
        )
        live = in_range & (expert >= fx.Int32(0)) & (expert < fx.Int32(total_experts))
        # Dead lanes add zero to expert 0's cursor instead of being branched
        # around, so ``position`` stays a plain wave value that ``readlane`` can
        # broadcast when the stores are issued.
        safe_expert = live.select(expert, fx.Int32(0))
        position = fx.Int32(
            comm_ops.atomic_add_agent(
                local_cursor + fx.Int64(safe_expert) * 4,
                live.select(fx.Int32(1), fx.Int32(0)),
            )
        )
        if live:
            _store_i32(pair_order, position, route)
        destination_row = (
            _load_i32(task_row_base, safe_expert)
            + position
            - _load_i32(pair_base, safe_expert)
        )
        dest_pe = safe_expert // fx.Int32(experts_per_rank)
        live_i = live.select(fx.Int32(1), fx.Int32(0))
        if overflow_any == fx.Int32(0):  # noqa: SIM102 - device predicates
            if ballot(T.i32, live) != fx.Int32(0):
                emit_load(tile, token)
                TDM.tdm_wait(0)
                if const_expr(dedup_on):
                    for l in range_constexpr(topk):
                        live_l = fx.Int32(readlane(T.i32, live_i, l))
                        dest_l = fx.Int32(readlane(T.i32, dest_pe, l))
                        first_for_peer = live_l != fx.Int32(0)
                        for earlier in range_constexpr(l):
                            earlier_live = fx.Int32(
                                readlane(T.i32, live_i, earlier)
                            )
                            earlier_dest = fx.Int32(
                                readlane(T.i32, dest_pe, earlier)
                            )
                            first_for_peer = first_for_peer & (
                                (earlier_live == fx.Int32(0))
                                | (earlier_dest != dest_l)
                            )
                        if first_for_peer:
                            emit_landing(tile, token, dest_l)
                else:
                    for l in range_constexpr(topk):
                        if readlane(T.i32, live_i, l) != fx.Int32(0):
                            emit_direct_store(
                                tile,
                                token,
                                fx.Int32(readlane(T.i32, destination_row, l)),
                                fx.Int32(readlane(T.i32, dest_pe, l)),
                            )
                # The next token reloads into this same tile.
                TDM.tdm_wait(0)
        token = token + fx.Int32(producer_blocks * num_waves)

    # Phase 2 cannot start until every block's phase 1 is done: a segment's
    # readiness covers rows whose payload and ``pair_order`` entry were produced
    # by whichever block happened to own that token. The TDM stores are already
    # drained, so reaching ``producer_blocks`` here means the bytes have landed
    # at their destination and the publish below can only be observed after them.
    comm_ops.waitcnt_all()
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_system_release()
        comm_ops.atomic_add_agent(group_done + fx.Int64(parity) * 4, fx.Int32(1))
        comm_ops.spin_until_eq_i32(
            group_done + fx.Int64(parity) * 4, fx.Int32(producer_blocks)
        )
    fx.barrier()
    comm_ops.fence_system_acquire()

    # Phase 2: write rowmap and grouped-row -> landing-row metadata remotely.
    # This stays destination-major so each workgroup emits coalesced metadata.
    def walk_segments(emit):
        """Call ``emit(tile_id, segment_row, segment_count, base, dst_base)``.

        A source's rows for one expert are contiguous in the destination but can
        cross an M-tile boundary, and readiness is per tile, so the run is cut at
        those boundaries.
        """
        task = destination_group
        while task < fx.Int32(experts_per_rank):
            ge = destination * fx.Int32(experts_per_rank) + task
            source_count = _load_i32(local_hist, ge)
            source_base = _load_i32(pair_base, ge)
            destination_base = _load_i32(task_row_base, ge)
            segment_row = fx.Int32(0)
            while segment_row < source_count:
                segment_dst = destination_base + segment_row
                tile_id = segment_dst // fx.Int32(tile_m)
                tile_end = (tile_id + fx.Int32(1)) * fx.Int32(tile_m)
                rows_left = source_count - segment_row
                room = tile_end - segment_dst
                segment_count = (rows_left < room).select(rows_left, room)
                emit(
                    tile_id,
                    segment_row,
                    segment_count,
                    source_base,
                    destination_base,
                )
                segment_row = segment_row + segment_count
            task = task + fx.Int32(producers_per_destination)

    walk_segments(
        lambda tile_id, segment_row, segment_count, source_base, destination_base: (
            emit_segment_rowmap(
                segment_row, segment_count, source_base, destination_base
            )
        )
    )

    remote_tile_ready = _peer(
        window, destination, arena_offset, layout.tile_rows_ready
    )

    def publish_direct(tile_id, segment_row, segment_count, source_base, dst_base):
        if tid == fx.Int32(0):
            slot = fx.Int32(parity) * fx.Int32(layout.tile_ready_capacity) + tile_id
            comm_ops.atomic_add_system(
                remote_tile_ready + fx.Int64(slot) * 4, segment_count
            )

    if const_expr(not dedup_on):
        comm_ops.waitcnt_all()
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
        walk_segments(publish_direct)

    # Every source publishes one completion generation after all of its landing
    # rows and metadata are system-visible. Destination gatherers consume only
    # local memory after acquiring all source generations.
    if const_expr(dedup_on):
        comm_ops.waitcnt_all()
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
            comm_ops.atomic_add_agent(
                metadata_done + fx.Int64(parity) * 4, fx.Int32(1)
            )
            comm_ops.spin_until_eq_i32(
                metadata_done + fx.Int64(parity) * 4, fx.Int32(producer_blocks)
            )
            if fx.Int32(producer_slot) == fx.Int32(0):
                for d in range_constexpr(npes):
                    remote_done = _peer(
                        window, fx.Int32(d), arena_offset, layout.landing_done
                    )
                    slot = fx.Int32(parity) * fx.Int32(npes) + fx.Int32(rank)
                    comm_ops.store_i32_system(
                        remote_done, slot, fx.Int32(expected)
                    )
        fx.barrier()

    # Phase 3: destination-local expansion. Wait until every source has landed
    # its deduplicated rows and row mapping, then materialize grouped contiguous
    # M. Each expert is owned by one workgroup, so its tile-ready values can be
    # released with stores rather than contended atomics.
    m_tile_map = _local(window, rank, arena_offset, m_tile_map_offset)
    tile_rows_ready = _local(
        window, rank, arena_offset, layout.tile_rows_ready
    )
    gather_blocks = min(32, producer_blocks) if dedup_on else 0
    gather_active = fx.Int32(producer_slot) < fx.Int32(gather_blocks)
    if gather_active:
        if tid == fx.Int32(0):
            for source in range_constexpr(npes):
                slot = fx.Int32(parity) * fx.Int32(npes) + fx.Int32(source)
                comm_ops.spin_until_eq_i32(
                    landing_done + fx.Int64(slot) * 4, expected
                )
        fx.barrier()
        comm_ops.fence_system_acquire()

    expert = gather_active.select(
        fx.Int32(producer_slot), fx.Int32(experts_per_rank)
    )
    while expert < fx.Int32(experts_per_rank):
        prev_index = (expert == fx.Int32(0)).select(
            fx.Int32(0), expert - fx.Int32(1)
        )
        prev_end = _load_i32(m_tile_map, prev_index)
        expert_start = (expert == fx.Int32(0)).select(
            fx.Int32(0),
            ((prev_end + fx.Int32(tile_m - 1)) // fx.Int32(tile_m))
            * fx.Int32(tile_m),
        )
        expert_end = _load_i32(m_tile_map, expert)
        expert_rows = expert_end - expert_start

        row = wave
        while row < expert_rows:
            destination_row = expert_start + row
            source_encoding = _load_i32(
                local_rowmap, destination_row * fx.Int32(2)
            )
            landing_slot = source_encoding // fx.Int32(topk)
            TDM.tdm_load(
                TDM.tdm_group0(
                    tile,
                    landing_wire + fx.Int64(landing_slot) * fx.Int64(wire_stride),
                ),
                wire_desc,
            )
            TDM.tdm_wait(0)
            emit_gather_store(tile, landing_slot, destination_row)
            TDM.tdm_wait(0)
            row = row + fx.Int32(num_waves)
        expert = expert + fx.Int32(gather_blocks)

    if gather_active:
        comm_ops.waitcnt_all()
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
        fx.barrier()

    # Publish after the block has materialized all of its experts. This pays one
    # system release per gather block rather than one per expert.
    expert = gather_active.select(
        fx.Int32(producer_slot), fx.Int32(experts_per_rank)
    )
    while expert < fx.Int32(experts_per_rank):
        if tid == fx.Int32(0):
            prev_index = (expert == fx.Int32(0)).select(
                fx.Int32(0), expert - fx.Int32(1)
            )
            prev_end = _load_i32(m_tile_map, prev_index)
            expert_start = (expert == fx.Int32(0)).select(
                fx.Int32(0),
                ((prev_end + fx.Int32(tile_m - 1)) // fx.Int32(tile_m))
                * fx.Int32(tile_m),
            )
            expert_end = _load_i32(m_tile_map, expert)
            segment_start = expert_start
            while segment_start < expert_end:
                tile_id = segment_start // fx.Int32(tile_m)
                tile_end = (tile_id + fx.Int32(1)) * fx.Int32(tile_m)
                segment_end = (expert_end < tile_end).select(expert_end, tile_end)
                ready_slot = (
                    fx.Int32(parity) * fx.Int32(layout.tile_ready_capacity)
                    + tile_id
                )
                comm_ops.store_i32_system(
                    tile_rows_ready,
                    ready_slot,
                    segment_end - segment_start,
                )
                segment_start = segment_end
        expert = expert + fx.Int32(gather_blocks)


@traced
def emit_compact_wait_tile(
    *,
    arena_handle,
    arena_offset: int,
    layout: CompactWorkspaceLayout,
    rank: int,
    m_tile,
    rows_needed,
    parity,
) -> None:
    """Acquire one destination M-tile as soon as all of its rows have landed.

    Called by every consumer (planner/producers rejoin too) right after it
    Producers add the number of completed rows contributed to this tile after
    draining their payload, scale and rowmap stores. The consumer waits for the
    exact valid-row count rather than for every source to finish the expert.
    """
    window = cco.Window(fx.Int64(arena_handle))
    tile_rows_ready = _local(window, rank, arena_offset, layout.tile_rows_ready)
    if fx.thread_idx.x == fx.Int32(0):
        slot = (
            fx.Int32(parity) * fx.Int32(layout.tile_ready_capacity)
            + fx.Int32(m_tile)
        )
        comm_ops.spin_until_eq_i32(
            tile_rows_ready + fx.Int64(slot) * 4, fx.Int32(rows_needed)
        )
    fx.barrier()
    comm_ops.fence_system_acquire()


__all__ = [
    "CompactWorkspaceLayout",
    "compact_payload_lds_bytes",
    "compact_workspace_layout",
    "emit_compact_payload",
    "emit_compact_planner",
    "emit_compact_wait_tile",
]
