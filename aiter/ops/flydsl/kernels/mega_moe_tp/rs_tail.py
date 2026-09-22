# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The ReduceScatter tail, as something any kernel can grow.

Whatever kernel last writes this rank's ``[M, H]`` partial can also finish the
layer: wait for the peers, pull their rows, and sum.  Which kernel that is
depends on the tuned GEMM2 -- an ``atomic`` row accumulates the partial itself
(:mod:`.stage2_rs`), a ``reduce`` row stages per-route output and a reduction
kernel produces the partial (:mod:`.reduce_rs`) -- so the tail lives here
rather than in either of them.

Shape
-----
:func:`emit_rs_tail` is called once, at the very end of the host kernel, by
*every* CTA::

    drain          every CTA: s_waitcnt, then arrive at fan-in counter
                   `cta % service`
    fan-in         the high `service` CTAs each drain the counter they own,
                   then meet at a root counter -- at which point the whole
                   local partial has landed
    writeback      the first WRITEBACK_BLOCKS of them issue one L2 writeback
                   each, one per XCD, and the last publishes the epoch to
                   every peer
    peer wait      one CTA polls all TP flags and releases the rest locally
    pull           the service CTAs read this rank's row range out of every
                   peer's partial and sum in FP32

Four things here are load-bearing, each of them measured rather than assumed
(kimi3 TP8, gfx950, a ~25k-CTA host grid):

* **A bare drain, not a fence.**  ``fence_agent_release()`` per CTA is a cache
  writeback: 1995 us of tail against 23 us for ``s_waitcnt vmcnt(0)``.
* **Fan-in counters in ordinary device memory.**  They are local-only, and
  atomics on IPC-exported arena pages bypass L2 for a fabric round trip.
* **The writeback count is capped independently of ``service``.**  Paying one
  per service CTA made a 256-wide group 2.3x slower than a 32-wide one, even
  though the pull wants to be wide.
* **One peer poller.**  Peer flags are system-scope reads over the fabric; all
  service CTAs polling all TP flags floods the very links the peers are still
  draining their own writes over.

The service CTAs are the *high* ids: dispatched last, so they begin waiting as
the host grid drains rather than holding a CU idle through it.  Reading the
epoch at kernel entry is correct at any id, because the bump cannot happen
until every CTA has arrived and a CTA must be dispatched before it can arrive.
"""

from __future__ import annotations


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl

from .. import communication_ops_utils as comm
from ..mxfp4_gemm_common import _udiv, _umod
from .p2p import (
    DESC_FLAGS,
    desc_peer_base,
    desc_slot,
    flat_buffer,
    load_words,
    store_words,
)
from .reduce_scatter import (
    ARRIVE_STRIDE_BYTES,
    PHASE_CTR_SLOT,
    PHASE_GATE_SLOT,
    MAX_SERVICE_BLOCKS,
    RS_DESC_EPOCH,
    RS_DESC_FANIN,
    RS_DESC_OUTPUT,
    RS_DESC_PARTIAL,
    RS_UNIT_ELEMS,
    WRITEBACK_BLOCKS,
    rs_desc_slot,
)

__all__ = ["emit_phase_barrier", "emit_rs_tail", "read_epoch", "rs_tail_slots"]

_VEC_WORDS = RS_UNIT_ELEMS // 2


def rs_service_low() -> bool:
    """Kept for the hosting kernel's module-name hash; always False now."""
    return False



def rs_tail_slots(tp_size: int):
    """Descriptor slot indices the tail reads, resolved once on the host."""
    return {
        "epoch": rs_desc_slot(tp_size, RS_DESC_EPOCH),
        "fanin": rs_desc_slot(tp_size, RS_DESC_FANIN),
        "partial": rs_desc_slot(tp_size, RS_DESC_PARTIAL),
        "output": rs_desc_slot(tp_size, RS_DESC_OUTPUT),
    }


def read_epoch(arg_desc, slots):
    """The epoch this launch will publish.

    Must be called at kernel entry, before any CTA can arrive: the bump happens
    only after every CTA has, so an entry read always observes the pre-launch
    value no matter when the hardware dispatched this CTA.
    """
    epoch_addr = desc_slot(arg_desc, slots["epoch"])
    return epoch_addr, fx.Int32(comm.load_i32_global_agent(epoch_addr)) + fx.Int32(1)


@flyc.jit
def emit_phase_barrier(arg_desc, epoch, cta, ctas, tid, *, tp_size: int):
    """Grid-wide barrier with a device-visible handoff, for a multi-GEMM kernel.

    Separates two compute phases inside one kernel when the second reads what
    the first wrote. A plain ``s_waitcnt`` is *not* enough here, unlike in
    :func:`emit_rs_tail`: MI355X L2 is per-XCD, so a store that reached the
    writer's L2 is not visible to a reader on another XCD. Between kernels the
    dispatch boundary handles that; inside one kernel it has to be an explicit
    agent-scope release/acquire.

    The generation is the arena epoch, so nothing needs resetting between
    launches, and the counter is reset by the last CTA in.

    **Requires a resident grid.** Every CTA waits here, so a queued CTA that
    cannot be scheduled deadlocks the ones that arrived -- size the grid to what
    the device holds at once.
    """
    base = desc_slot(arg_desc, rs_tail_slots(tp_size)["fanin"])
    ctr = base + fx.Int64(PHASE_CTR_SLOT * ARRIVE_STRIDE_BYTES)
    gate = base + fx.Int64(PHASE_GATE_SLOT * ARRIVE_STRIDE_BYTES)
    comm.fence_agent_release()
    gpu.barrier()
    if tid == fx.Int32(0):
        previous = fx.Int32(comm.atomic_add_agent(ctr, fx.Int32(1)))
        if previous == ctas - fx.Int32(1):
            comm.store_i32_global_agent_release(ctr, fx.Int32(0))
            comm.store_i32_global_agent_release(gate, epoch)
        comm.spin_until_ge_i32_agent(gate, epoch)
    gpu.barrier()
    comm.fence_agent_acquire()


@flyc.jit
def emit_rs_tail(
    arg_desc,
    i32_rank,
    i32_rows,
    epoch_addr,
    epoch,
    cta,
    ctas,
    tid,
    *,
    tp_size: int,
    model_dim: int,
    block: int,
    service_blocks: int,
):
    """Emit the arrival, the publish and the pull-reduce at a kernel's end.

    ``cta``/``ctas`` are a *linear* block id and block count, so a 2D host grid
    just passes ``bx + by * gx`` and ``gx * gy``.

    ``@flyc.jit`` is required, not decorative: the divergent ``if``s below are
    rewritten into ``scf.if`` by the tracer, and a plain Python function would
    try to evaluate them as bools instead.
    """
    units_per_row = model_dim // RS_UNIT_ELEMS
    row_bytes = model_dim * 2
    slots = rs_tail_slots(tp_size)
    arrive_base = desc_slot(arg_desc, slots["fanin"])
    flags_offset = desc_slot(arg_desc, DESC_FLAGS)
    # How many CTAs run the collective. Capped by the *work*, not just by
    # ``service_blocks``: the pull-reduce has ``i32_rows * units_per_row``
    # units to move, and at decode sizes that is a handful -- kimi3 M=8 is 448
    # units, which 2 CTAs cover. Handing it 128 CTAs means 126 of them spin
    # through the whole rendezvous and then issue peer reads for nothing.
    # Worth +2.3% at kimi3 M=8; saturates back to ``service_blocks`` as soon as
    # the rows grow, so large M is untouched.
    units_total = i32_rows * fx.Int32(units_per_row)
    work_ctas = (units_total + fx.Int32(block - 1)) // fx.Int32(block)
    service = fx.min(fx.min(ctas, fx.Int32(service_blocks)), fx.max(work_ctas, fx.Int32(1)))
    # The *last* block ids run the collective. Taking the lowest instead --
    # which is what the AllGather push does, and for the apparently good reason
    # that low ids are dispatched first and are therefore resident from the
    # start -- was measured and is **worse**: -1.1% at kimi3 M=8, -1.4% at
    # M=64. The dispatch-wave latency this was meant to remove is not there.
    first_service = ctas - service

    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    gpu.barrier()
    if tid == fx.Int32(0):
        lane_idx = _umod(cta, service)
        comm.atomic_add_agent(
            arrive_base + fx.Int64(lane_idx) * fx.Int64(ARRIVE_STRIDE_BYTES),
            fx.Int32(1),
        )

    if cta >= first_service:
        local_base = desc_peer_base(arg_desc, i32_rank)
        own = cta - first_service
        if tid == fx.Int32(0):
            per_lane = _udiv(ctas, service)
            remainder = ctas - per_lane * service
            expected = per_lane + (own < remainder).select(fx.Int32(1), fx.Int32(0))
            own_addr = arrive_base + fx.Int64(own) * fx.Int64(ARRIVE_STRIDE_BYTES)
            comm.spin_until_ge_i32_agent(own_addr, expected)
            comm.fence_agent_acquire()
            # The stream serializes launches, so resetting here cannot race the
            # next call's arrivals.
            comm.store_i32_global_agent_release(own_addr, fx.Int32(0))

            root_addr = arrive_base + fx.Int64(
                MAX_SERVICE_BLOCKS * ARRIVE_STRIDE_BYTES
            )
            gate_addr = arrive_base + fx.Int64(
                (MAX_SERVICE_BLOCKS + 1) * ARRIVE_STRIDE_BYTES
            )
            wb_root_addr = arrive_base + fx.Int64(
                (MAX_SERVICE_BLOCKS + 2) * ARRIVE_STRIDE_BYTES
            )
            peers_addr = arrive_base + fx.Int64(
                (MAX_SERVICE_BLOCKS + 3) * ARRIVE_STRIDE_BYTES
            )
            seen = fx.Int32(comm.atomic_add_agent(root_addr, fx.Int32(1)))
            if seen == service - fx.Int32(1):
                comm.store_i32_global_agent_release(root_addr, fx.Int32(0))
                comm.store_i32_global_agent_release(gate_addr, epoch)
            comm.spin_until_ge_i32_agent(gate_addr, epoch)
            comm.fence_agent_acquire()

            writebacks = fx.min(service, fx.Int32(WRITEBACK_BLOCKS))
            if own < writebacks:
                comm.fence_system_release()
                wb_seen = fx.Int32(comm.atomic_add_agent(wb_root_addr, fx.Int32(1)))
                if wb_seen == writebacks - fx.Int32(1):
                    comm.store_i32_global_agent_release(wb_root_addr, fx.Int32(0))
                    comm.store_i32_global_agent_release(epoch_addr, epoch)
                    for slot in range_constexpr(tp_size):
                        peer = (i32_rank + fx.Int32(slot)) % fx.Int32(tp_size)
                        comm.store_i32_global_system_release(
                            desc_peer_base(arg_desc, peer)
                            + flags_offset
                            + fx.Int64(i32_rank) * fx.Int64(4),
                            epoch,
                        )
                    for source in range_constexpr(tp_size):
                        comm.spin_until_ge_i32_system(
                            local_base + flags_offset + fx.Int64(source * 4),
                            epoch,
                            acquire=True,
                            sleep=True,
                        )
                    comm.fence_system_acquire()
                    comm.store_i32_global_agent_release(peers_addr, epoch)
            comm.spin_until_ge_i32_agent(peers_addr, epoch)
            comm.fence_system_acquire()
        gpu.barrier()

        gid = own * fx.Int32(block) + tid
        stride = service * fx.Int32(block)
        units = i32_rows * fx.Int32(units_per_row)
        partial_offset = desc_slot(arg_desc, slots["partial"])
        total_bytes = fx.Int64(i32_rows) * fx.Int64(tp_size) * fx.Int64(row_bytes)
        # Stagger by rank *and* by block: with a rank-only rotation every CTA on
        # this GPU pulls from the same peer at the same instant, so one xGMI
        # link carries the whole grid while the other seven idle.
        sources = []
        for slot in range_constexpr(tp_size):
            peer = (i32_rank + own + fx.Int32(slot)) % fx.Int32(tp_size)
            sources.append(
                flat_buffer(
                    desc_peer_base(arg_desc, peer) + partial_offset,
                    fx.Int32,
                    total_bytes,
                )
            )
        output = flat_buffer(
            desc_slot(arg_desc, slots["output"]),
            fx.Int32,
            fx.Int64(i32_rows) * fx.Int64(row_bytes),
        )
        shard_base = i32_rank * units
        # No tail guard: both views are bounded by the same num_records, so the
        # hardware drops an over-run store and zero-fills its load.
        for unit in range(gid, units, stride):
            index = fx.Int32(unit)
            acc = fx.Vector.filled(RS_UNIT_ELEMS, 0.0, fx.Float32)
            for slot in range_constexpr(tp_size):
                words = load_words(
                    sources[slot],
                    shard_base + index,
                    width=_VEC_WORDS,
                    cache_modifier=2,
                )
                acc = acc + words.bitcast(fx.BFloat16).to(fx.Float32)
            store_words(
                output,
                index,
                acc.to(fx.BFloat16).bitcast(fx.Int32),
                width=_VEC_WORDS,
            )
