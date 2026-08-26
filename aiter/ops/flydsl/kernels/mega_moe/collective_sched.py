# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Collective scheduling and synchronisation helpers for the TP MoE kernels.

These are TP's own copies of four pieces of MegaMoE's scheduler. MegaMoE's
source files are deliberately NOT modified and NOT imported from -- see section
6 of docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md for
why sharing was rejected. Only ~35 lines are genuinely identical between the
two; the rest diverges because TP has no expert-major routing, no capacity
overflow, and no per-tile payload readiness.

Do not "deduplicate" these against dispatch.py / mega_moe_stage1.py. The two
sides are allowed to evolve independently, and MegaMoEV2 is frozen.

These are trace-time helpers, not device functions: ``@flyc.jit`` bodies are
inlined during tracing, so factoring code in here emits no ``func.call``.
"""

# fmt: off

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from .gemm_util import _buffer_load, _buffer_store, _make_buffer_from_addr


@flyc.jit
def copy_row(source_rsrc, destination_rsrc, lane, *, safe_end_i32, n_i32):
    """Copy one row global->global, dwordx4 per lane.

    Verbatim from MegaMoE's dispatch.py::_copy_token_row. ``n_i32`` is the row
    length in i32 words; ``safe_end_i32`` is the 512-word-aligned main body,
    i.e. ``(n_i32 // 512) * 512``. Handles both the 7168-byte activation row and
    the 224-byte scale row -- the latter has ``safe_end_i32 == 0`` and takes only
    the tail loop.
    """
    lane_offset = lane * fx.Int32(4)
    if const_expr(safe_end_i32 > 0):
        for column in range(lane_offset, safe_end_i32, 512):
            value0 = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            value1 = buffer_ops.buffer_load(source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value0, destination_rsrc, column)
            buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(safe_end_i32 < n_i32):
        for column in range(lane_offset + safe_end_i32, n_i32, 256):
            value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value, destination_rsrc, column)


@flyc.jit
def emit_ticket_and_roles(*, tid, lds_scratch, a_entry_count, a_epoch_gate,
        epoch_slot, launch_grid_x, producer_blocks):
    """Take this CTA's launch ticket and derive its role for the round.

    One atomic on thread 0, broadcast to the block through byte 0 of the LDS
    scratch. ``a_entry_count`` is a monotonically increasing i64 counter that is
    never reset: dividing by ``launch_grid_x`` recovers which launch this CTA
    belongs to, and the remainder is its role index within that launch.

    Returns ``(gate_addr, gate_epoch, is_owner, is_producer, producer_slot)``.
    """
    ticket_scratch = fx.recast_iter(fx.Int64, lds_scratch.ptr)
    ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
    if tid == fx.Int32(0):
        ticket64 = fx.Int64(
            comm_ops.atomic_add_agent(a_entry_count + fx.Int64(epoch_slot * 8), fx.Int64(1))
        )
        fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
    fx.barrier()
    ticket64 = Vec(ticket_view.load())[0]
    generation = ticket64 // fx.Int64(launch_grid_x)
    ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
    gate_addr = a_epoch_gate + fx.Int64(epoch_slot * 4)
    gate_epoch = fx.Int32(generation + fx.Int64(1))
    is_owner = ticket == fx.Int32(0)
    is_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(producer_blocks))
    producer_slot = ticket - fx.Int32(1)
    return gate_addr, gate_epoch, is_owner, is_producer, producer_slot


@flyc.jit
def emit_launch_rendezvous(*, tid, is_owner, p_launch_ready, a_launch_ready,
        a_reset_counters, reset_count, gate_addr, gate_epoch, launch_epoch,
        npes, rank):
    """Rendezvous with every peer, reset per-launch counters, open the gate.

    Same as emit_epoch_rendezvous minus the parity flip: the caller derives
    parity and launch_epoch on the host, which removes a GPU->CPU sync from the
    per-call path. The peer wait still does the real work -- it is what stops
    rank A's round N+1 push from landing in a buffer rank B is still reading in
    round N, because on a single stream B entering round N+1 means B's round-N
    kernel retired.
    """
    if is_owner:
        if tid < fx.Int32(npes):
            peer = (tid + fx.Int32(rank)) % fx.Int32(npes)
            comm_ops.fence_system_release()
            launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
            remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
            comm_ops.store_i32_system(remote_launch_ready, fx.Int32(rank), launch_epoch)
            mori_shmem.int32_wait_until_greater_than(
                a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
            )
            comm_ops.fence_system_acquire()
        if tid == fx.Int32(0):
            reset_rsrc = _make_buffer_from_addr(a_reset_counters, fx.Int32)
            for slot in range_constexpr(reset_count):
                _buffer_store(reset_rsrc, fx.Int32(slot * 16), fx.Int32(0), fx.Int32)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
    else:
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
            comm_ops.fence_agent_acquire()
        fx.barrier()


@flyc.jit
def emit_epoch_rendezvous(*, tid, is_owner, parity_rsrc, expected_rsrc,
        p_launch_ready, a_launch_ready, a_reset_counters, reset_count,
        gate_addr, gate_epoch, npes, rank):
    """Flip the epoch, rendezvous with every peer, reset local state, open the gate.

    One indivisible if/else, copied from MegaMoE with its three EP-only
    const_expr branches removed. The owner CTA flips parity and expected,
    publishes its launch epoch to every peer and waits for theirs, zeroes the
    per-launch counters, then stores the gate; every other CTA waits on the gate.

    The peer wait is what stops rank A's round N+1 push from landing in a buffer
    rank B is still reading in round N: on a single stream, B entering round N+1
    means B's round-N kernel retired.

    ``next_parity_lane`` / ``launch_epoch_lane`` are rebound inside the nested
    ``if tid == 0`` and read after it; the readfirstlane pair must stay in this
    function or the SSA merge point moves.

    ``a_reset_counters`` is an i32 array of ``reset_count`` per-launch counters
    (this kernel uses one: push_done) that the owner zeroes each round.
    """
    if is_owner:
        next_parity_lane = fx.Int32(0)
        launch_epoch_lane = fx.Int32(0)
        if tid == fx.Int32(0):
            old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
            next_parity_lane = old_parity ^ fx.Int32(1)
            previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
            next_expected = previous_expected + fx.Int32(npes)
            _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
            launch_epoch_lane = (
                (next_expected // fx.Int32(npes)) * fx.Int32(2) - next_parity_lane
            )
        next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
        launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
        if tid < fx.Int32(npes):
            peer = (tid + fx.Int32(rank)) % fx.Int32(npes)
            comm_ops.fence_system_release()
            launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
            remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
            comm_ops.store_i32_system(remote_launch_ready, fx.Int32(rank), launch_epoch)
            mori_shmem.int32_wait_until_greater_than(
                a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
            )
            comm_ops.fence_system_acquire()
        if tid == fx.Int32(0):
            reset_rsrc = _make_buffer_from_addr(a_reset_counters, fx.Int32)
            for slot in range_constexpr(reset_count):
                _buffer_store(reset_rsrc, fx.Int32(slot * 16), fx.Int32(0), fx.Int32)
        fx.barrier()
        if tid == fx.Int32(0):
            fx.rocdl.s_waitcnt(0)
            comm_ops.fence_agent_release()
            _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
            fx.rocdl.s_waitcnt(0)
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
    else:
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
            comm_ops.fence_agent_acquire()
        fx.barrier()
