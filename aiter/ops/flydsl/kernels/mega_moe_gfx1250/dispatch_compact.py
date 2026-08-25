# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 compact EP dispatch emitters for the gfx1250 MegaMoE pipeline.

This is the compact (destination-owned row plan) counterpart of
``mega_moe.dispatch``.  It is deliberately a collection of emitters rather than
a launch wrapper: the persistent kernel in ``mxfp4_preshuffle_gfx1250_tdm.py``
can assign its first arrival to :func:`emit_compact_planner` and subsequent
arrivals to :func:`emit_compact_payload`; every workgroup then rejoins the work
queue as a consumer and gates each tile on :func:`emit_compact_wait_expert`.

There is no pull phase and no receive-slot reservation.  A route is copied
straight to the final tile-aligned row assigned by the destination planner.
All symmetric addresses are formed with ``Window.lsa_ptr(peer, arena_offset)``;
there is intentionally no P2P pointer table.

Synchronization invariants
--------------------------
* ``expected`` is a non-zero, monotonically changing generation value and
  ``parity == generation & 1``.  ``count_done``, ``plan_ready``,
  ``pair_ready`` and ``pair_order_ready`` are generation values, never counters.
* A source publishes its count-matrix stores with a system release store to
  ``count_done[parity, source]``.  A destination acquires every source slot
  before computing row bases.
* The destination clears only its current-parity ``payload_ready`` counters
  before publishing ``plan_ready``.  Payload producers cannot race that clear:
  they wait for that destination's plan-ready generation first.
* One source publishes exactly once to every ``(destination, local_expert)``,
  including zero-count tasks.  Therefore ``payload_ready[parity, expert] ==
  npes`` means all rows (and metadata) for that expert are visible.
* TDM tensor counters are drained (and remote scale scatters completed) before
  payload-ready is incremented, so ``payload_ready[parity, expert] == npes``
  is a sufficient GEMM-readiness gate on its own -- there is no separate
  launch barrier.

The input wire scale is row-major.  It must not be exposed as the grouped GEMM
scale: that layout is ``(Mtile, K//128, wmma_rep, 16, 4)``.  Payload producers
split one whole-row LDS load into a tightly packed payload store and scatter
the trailing e8m0 scales straight into that WMMA-interleaved layout using the
exact address helper from ``moe_fused_route_quant_scatter``, so each expert is
GEMM-readable the instant its ``payload_ready`` fires.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import ds_bpermute, readfirstlane, readlane, update_dpp
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
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
    pair_order_ready: int
    plan_ready: int
    payload_ready: int
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
        work_head_count: int = DEFAULT_WORK_HEADS,
        alignment: int = DEFAULT_ALIGNMENT,
    ) -> "CompactWorkspaceLayout":
        """Compute all local offsets and the required symmetric allocation size."""
        if npes <= 0 or experts_per_rank <= 0:
            raise ValueError("npes and experts_per_rank must be positive")
        if max_tokens < 0 or topk <= 0:
            raise ValueError("max_tokens must be non-negative and topk positive")
        if work_head_count <= 0:
            raise ValueError("work_head_count must be positive")
        if alignment < 8 or alignment & (alignment - 1):
            raise ValueError("alignment must be a power of two and at least 8")

        total_experts = npes * experts_per_rank
        max_pairs = max_tokens * topk
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
        put("pair_order_ready", 2 * 4)
        put("plan_ready", 2 * npes * 4)
        put("payload_ready", 2 * experts_per_rank * 4)
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
            "pair_order_ready",
            "plan_ready",
            "payload_ready",
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


def compact_payload_lds_bytes(
    *, wire_stride: int, num_waves: int, pipe_depth: int = 1
) -> int:
    """LDS bytes required by :func:`emit_compact_payload`.

    ``pipe_depth`` whole-wire-row tiles per wave.  ``pipe_depth == 1`` fully
    drains TDM every iteration, so a multi-row expert reuses the one tile.
    ``pipe_depth == 2`` double-buffers so each row's remote payload store
    overlaps the next row's load (the producer runs inside the register/LDS-heavy
    GEMM at low occupancy, so this latency is otherwise fully exposed).
    """
    if wire_stride <= 0 or num_waves <= 0:
        raise ValueError("wire_stride and num_waves must be positive")
    if pipe_depth not in (1, 2):
        raise ValueError("pipe_depth must be 1 or 2")
    return _align(wire_stride, 128) * num_waves * pipe_depth


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
    pair_order = _local(window, rank, arena_offset, layout.pair_order)
    pair_ready = _local(window, rank, arena_offset, layout.pair_ready)
    pair_order_ready = _local(window, rank, arena_offset, layout.pair_order_ready)
    plan_ready = _local(window, rank, arena_offset, layout.plan_ready)
    payload_ready = _local(window, rank, arena_offset, layout.payload_ready)
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

    # Wave 1 builds the source-global exclusive prefix serially.  This is tiny
    # (normally <=512 entries), deterministic, and avoids wave64 assumptions.
    if wave == fx.Int32(1):
        if lane == fx.Int32(0):
            prefix = fx.Int32(0)
            for ge in range(fx.Int32(0), fx.Int32(total_experts), 1):
                count = _load_i32(local_hist, ge)
                _store_i32(pair_base, ge, prefix)
                _store_i32(local_cursor, ge, prefix)
                prefix = prefix + count
            comm_ops.waitcnt_stores()
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(pair_ready, parity, expected)

    fx.barrier()
    if (wave > fx.Int32(0)) & (lane == fx.Int32(0)):
        comm_ops.spin_until_eq_i32(pair_ready + fx.Int64(parity) * 4, expected)
    fx.barrier()
    if wave > fx.Int32(0):
        group_tid = (wave - fx.Int32(1)) * fx.Int32(WAVE) + lane
        group_threads = fx.Int32((num_waves - 1) * WAVE)
        for route in range(group_tid, route_limit, group_threads):
            expert = fx.Int32(
                buffer_ops.buffer_load(idx_rsrc, route, vec_width=1, dtype=T.i32)
            )
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(total_experts))
            if valid:
                position = fx.Int32(
                    comm_ops.atomic_add_agent(
                        local_cursor + fx.Int64(expert) * 4, fx.Int32(1)
                    )
                )
                _store_i32(pair_order, position, route)
    comm_ops.waitcnt_all()
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        comm_ops.store_i32_system(pair_order_ready, parity, expected)


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
    payload_offset: int,
    grouped_scale_offset: int,
    rowmap_offset: int,
    num_valid_offset: int,
    wire_stride: int,
    payload_bytes: int,
    scale_bytes: int,
    tile_m: int,
    wmma_rep: int,
    parity,
    expected,
    pipe_depth: int = 1,
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
    ``producer_slot`` partitions tasks by destination first; consequently at
    least ``npes`` producer blocks are required.  Each task waits for the
    destination's row plan, reads this source's ``task_row_base`` and grouped
    ``pair_order``, and sends rows directly to final destination row numbers.

    The full wire row is loaded by one TDM descriptor into each wave's LDS tile.
    A TDM store lands the tightly packed grouped payload; the e8m0 scales that
    trail it in the wire row are scattered straight into the destination's
    WMMA-interleaved ``(Mtile, K//128, wmma_rep, 16, 4)`` layout via the shared
    ``_scale_row_dword_base`` helper, so no separate finalize pass is needed and
    each expert becomes GEMM-readable the moment its ``payload_ready`` fires.

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
    if pipe_depth not in (1, 2):
        raise ValueError("pipe_depth must be 1 or 2")
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

    window = cco.Window(fx.Int64(arena_handle))
    tid = fx.Int32(fx.thread_idx.x)
    lane = tid & fx.Int32(LANE_MASK)
    wave = tid >> fx.Int32(LOG2_WAVE)
    destination = fx.Int32(producer_slot) % fx.Int32(npes)
    destination_group = fx.Int32(producer_slot) // fx.Int32(npes)
    producers_per_destination = (producer_blocks + npes - 1) // npes

    pair_base = _local(window, rank, arena_offset, layout.pair_base)
    local_hist = _local(window, rank, arena_offset, layout.local_hist)
    task_row_base = _local(window, rank, arena_offset, layout.task_row_base)
    pair_order = _local(window, rank, arena_offset, layout.pair_order)
    plan_ready = _local(window, rank, arena_offset, layout.plan_ready)
    pair_order_ready = _local(window, rank, arena_offset, layout.pair_order_ready)

    ready_index = fx.Int32(parity) * fx.Int32(npes) + destination
    if tid == fx.Int32(0):
        comm_ops.spin_until_eq_i32(
            plan_ready + fx.Int64(ready_index) * 4, expected
        )
        comm_ops.spin_until_eq_i32(
            pair_order_ready + fx.Int64(parity) * 4, expected
        )
    fx.barrier()
    comm_ops.fence_system_acquire()

    # One complete wire row per wave.  128B alignment preserves descriptor/LDS
    # alignment even when wire_stride itself is not a power of two.  With
    # ``pipe_depth == 2`` each wave owns two such tiles in separate LDS banks so
    # a row's remote payload store overlaps the next row's load.
    tile_bytes = _align(wire_stride, 128)
    bank_stride = num_waves * tile_bytes
    wire_desc = TDM.tdm_group1(wire_stride, 1, 1)
    payload_desc = TDM.tdm_group1(payload_bytes, 1, 1)
    # Scales are scattered (not TDM-stored) directly into the WMMA-interleaved
    # layout, folding ``wmma_rep`` consecutive rows into each scale tile.  This
    # must match ``moe_fused_route_quant_scatter`` byte-for-byte.
    rows_per_tile = wmma_rep * 16
    dst_scale_dwords_per_row = (scale_bytes // 4) * wmma_rep

    def wave_tile(buf):
        """LDS byte address of this wave's tile in double-buffer bank ``buf``."""
        return (
            fx.Int32(lds_base_i32)
            + buf * fx.Int32(bank_stride)
            + readfirstlane(T.i32, wave) * fx.Int32(tile_bytes)
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

    def emit_flush(tile, source_token, destination_row, route):
        """Store one gathered row: payload TDM + WMMA-interleaved scale + rowmap."""
        topk_slot = route - source_token * fx.Int32(topk)
        TDM.tdm_store(
            TDM.tdm_group0(
                tile,
                _peer(window, destination, arena_offset, payload_offset)
                + fx.Int64(destination_row) * fx.Int64(payload_bytes),
            ),
            payload_desc,
        )
        # The e8m0 scales trail the payload in the wire row.  Scatter them into
        # the destination's WMMA-interleaved scale buffer with the same helper
        # finalize used, so each expert is GEMM-readable as soon as its
        # ``payload_ready`` counter reaches ``npes``.
        dst_scale = _peer(window, destination, arena_offset, grouped_scale_offset)
        # Move the e8m0 scales one dword per lane, not one byte.  The scales are
        # dword-aligned (scale_bytes % 4 == 0) and, in the WMMA-interleaved
        # layout, the four bytes of one source dword share a destination dword
        # (``_scale_row_dword_base`` addresses dwords, and consecutive bytes only
        # differ in ``byte_in_dword``), so a whole dword lands with a single
        # store.  The wire read stays coalesced across lanes and the destination
        # store count drops 4x -- that store count is the bulk of the fused
        # producer's non-payload cost.
        scale_dwords = scale_bytes // 4
        wire_scale_dw_base = fx.Int32(source_token) * fx.Int32(wire_stride // 4) + fx.Int32(
            payload_bytes // 4
        )
        for dw in range_constexpr(0, scale_dwords, WAVE):
            dword = fx.Int32(dw) + lane
            if dword < fx.Int32(scale_dwords):
                value = buffer_ops.buffer_load(
                    _rsrc(fx.Int64(addr_in_wire)),
                    wire_scale_dw_base + dword,
                    vec_width=1,
                    dtype=T.i32,
                )
                if const_expr(scale_rowmajor):
                    # Row-major coalesced: grouped row r's e8m0 scales are a
                    # contiguous ``scale_bytes`` run at ``r*scale_bytes``.
                    # Consecutive lanes write consecutive dwords, so each wave
                    # step is one coalesced remote burst.  The GEMM strided-reads
                    # element ``r*(scale_bytes//4)+k128`` back, so this lands
                    # where the WMMA-interleaved LDS load expects it.
                    buffer_ops.buffer_store(
                        value,
                        _rsrc(dst_scale),
                        destination_row * fx.Int32(scale_dwords) + dword,
                    )
                else:
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
        if lane == fx.Int32(0):
            weight = buffer_ops.buffer_load(
                _rsrc(addr_in_weights), route, vec_width=1, dtype=T.f32
            )
            weight_bits = fx.Float32(weight).bitcast(fx.Int32)
            # Current gfx1250 GEMM2 consumes ep_rowmap directly:
            # (destination combine slot, route weight bits).
            source_encoding = (
                fx.Int32(rank * max_tokens_per_rank * topk)
                + source_token * fx.Int32(topk)
                + topk_slot
            )
            _store_i32(
                _peer(window, destination, arena_offset, rowmap_offset),
                destination_row * fx.Int32(2),
                source_encoding,
            )
            _store_i32(
                _peer(window, destination, arena_offset, rowmap_offset),
                destination_row * fx.Int32(2) + fx.Int32(1),
                weight_bits,
            )

    task = destination_group
    while task < fx.Int32(experts_per_rank):
        local_expert = task
        ge = destination * fx.Int32(experts_per_rank) + local_expert
        source_count = _load_i32(local_hist, ge)
        source_base = _load_i32(pair_base, ge)
        destination_base = _load_i32(task_row_base, ge)
        overflow = _load_i32(
            _peer(window, destination, arena_offset, num_valid_offset),
            fx.Int32(1),
        )

        if const_expr(pipe_depth > 1):
            # Software-pipelined: issue this row's load, then flush the row
            # loaded last iteration so its remote store overlaps the load.  The
            # load/store tensor counter retires in issue order, so ``tdm_wait(1)``
            # after issuing the current load guarantees the previous row's load
            # (and every store before it) has landed while leaving the current
            # load in flight.
            row = wave
            have_prev = fx.Int32(0)
            p_tok = fx.Int32(0)
            p_dst = fx.Int32(0)
            p_route = fx.Int32(0)
            p_buf = fx.Int32(1)
            while row < source_count:
                if overflow == fx.Int32(0):
                    cur_buf = fx.Int32(1) - p_buf
                    route = _load_i32(pair_order, source_base + row)
                    source_token = route // fx.Int32(topk)
                    destination_row = destination_base + row
                    emit_load(wave_tile(cur_buf), source_token)
                    if have_prev != fx.Int32(0):
                        TDM.tdm_wait(1)
                        emit_flush(wave_tile(p_buf), p_tok, p_dst, p_route)
                    have_prev = fx.Int32(1)
                    p_tok = source_token
                    p_dst = destination_row
                    p_route = route
                    p_buf = cur_buf
                row = row + fx.Int32(num_waves)
            if overflow == fx.Int32(0):
                if have_prev != fx.Int32(0):
                    TDM.tdm_wait(0)
                    emit_flush(wave_tile(p_buf), p_tok, p_dst, p_route)
                    TDM.tdm_wait(0)
            comm_ops.waitcnt_all()
        else:
            row = wave
            while row < source_count:
                if overflow == fx.Int32(0):
                    route = _load_i32(pair_order, source_base + row)
                    source_token = route // fx.Int32(topk)
                    destination_row = destination_base + row
                    tile = wave_tile(fx.Int32(0))
                    emit_load(tile, source_token)
                    TDM.tdm_wait(0)
                    emit_flush(tile, source_token, destination_row, route)
                    TDM.tdm_wait(0)
                    comm_ops.waitcnt_all()
                row = row + fx.Int32(num_waves)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
            remote_payload_ready = _peer(
                window, destination, arena_offset, layout.payload_ready
            )
            slot = fx.Int32(parity) * fx.Int32(experts_per_rank) + local_expert
            comm_ops.atomic_add_system(
                remote_payload_ready + fx.Int64(slot) * 4, fx.Int32(1)
            )
        fx.barrier()
        task = task + fx.Int32(producers_per_destination)


@traced
def emit_compact_wait_expert(
    *,
    arena_handle,
    arena_offset: int,
    layout: CompactWorkspaceLayout,
    rank: int,
    npes: int,
    experts_per_rank: int,
    expert,
    parity,
) -> None:
    """Acquire the local destination's operands for one expert's M-tile.

    Called by every consumer (planner/producers rejoin too) right after it
    resolves the expert owning its claimed work-queue tile.  It waits only for
    that expert's ``payload_ready`` counter to reach ``npes`` -- i.e. all source
    ranks delivered its payload and WMMA-interleaved scales -- so consumers on
    ready experts start computing while producers are still shipping later ones.
    This replaces the old global ``launch_ready`` barrier and the separate scale
    finalize phase with gfx950-style per-tile payload waits.
    """
    window = cco.Window(fx.Int64(arena_handle))
    payload_ready = _local(window, rank, arena_offset, layout.payload_ready)
    if fx.thread_idx.x == fx.Int32(0):
        slot = fx.Int32(parity) * fx.Int32(experts_per_rank) + fx.Int32(expert)
        # atomicAdd contributes one increment per source, hence equality to npes.
        comm_ops.spin_until_eq_i32(
            payload_ready + fx.Int64(slot) * 4, fx.Int32(npes)
        )
    fx.barrier()
    comm_ops.fence_system_acquire()


__all__ = [
    "CompactWorkspaceLayout",
    "compact_payload_lds_bytes",
    "compact_workspace_layout",
    "emit_compact_payload",
    "emit_compact_planner",
    "emit_compact_wait_expert",
]
