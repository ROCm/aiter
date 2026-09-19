# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""One-launch AllGather for the fused TP MoE layer.

Every rank pushes its own token shard straight into all peers' arenas and then
waits for theirs, replacing the NCCL ``all_gather_into_tensor`` chain (payload,
scale, routing metadata) plus the unpack copies with a single kernel.

Why a push works here
---------------------
Under sequence parallelism rank ``p`` owns global rows ``[p*m, (p+1)*m)``, so
the destination offset is ``p * m * row_bytes``.  The histogram /
count-exchange / dynamic-base machinery an EP all-to-all needs collapses into a
fixed-stride flat copy, which is why each region is a plain contiguous memcpy
into every peer.

Regions are described once at init (source address, arena offset, row width);
only the row count changes per call.  The epoch is device-side -- the barrier
reads and bumps a counter in the arena rather than taking a number from the
host, so a captured graph replays correctly instead of re-using a flag value
that is already satisfied.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .. import communication_ops_utils as comm
from .p2p import (
    DESC_ARRIVE,
    DESC_FLAGS,
    desc_peer_base,
    desc_region_offset,
    desc_region_src,
    desc_size,
    desc_slot,
    flat_buffer,
    load_words,
    store_words,
)

__all__ = [
    "AG_DESC_DONE",
    "AG_DESC_EPOCH",
    "PUSH_MIN_BYTES",
    "PUSH_UNROLL",
    "PUSH_VEC_BYTES",
    "ag_desc_size",
    "ag_desc_slot",
    "compile_allgather_push",
    "push_units",
]

# Descriptor slot beyond the shared prefix and the per-region entries.
AG_DESC_EPOCH = 0  # local address of the monotone epoch counter (i32)
#: Bumped *after* the cross-rank wait completes, unlike the epoch which is
#: bumped before it. A kernel whose CTAs continue past the AllGather waits on
#: this: the epoch alone would let them through while peer data is still in
#: flight. See ``emit_ag_barrier(gate_all=True)``.
AG_DESC_DONE = 1

_BLOCK = int(os.environ.get("AITER_TP_AG_BLOCK", "256"))
# 16 B/lane is one buffer_load_dwordx4; a wave then moves 1 KiB per instruction.
_VEC_WORDS = 4
PUSH_VEC_BYTES = _VEC_WORDS * 4
#: Smallest addressable push unit. A region whose row is not a whole number of
#: :data:`PUSH_VEC_BYTES` (topk*4 for topk=6 or 9, say) drops to dword copies
#: for that region alone; the wide payload regions keep their dwordx4.
PUSH_MIN_BYTES = 4
# Units a thread issues before waiting on any of them. One 16 B access per
# thread leaves xGMI badly under-subscribed at prefill sizes -- the grid is
# capped at the CU count (the closing barrier needs every CTA resident), so
# depth has to come from unrolling rather than from more CTAs.
PUSH_UNROLL = 4


@functools.cache
def compile_allgather_push(
    tp_size: int,
    row_bytes: tuple[int, ...],
    *,
    block: int = _BLOCK,
):
    """Build the push-AllGather launcher for one set of region row widths.

    ``row_bytes[r]`` is region ``r``'s bytes per token.  A region that is a
    whole number of :data:`PUSH_VEC_BYTES` copies with ``dwordx4``; anything
    else (a narrow routing row, say) falls back to dword copies for that region
    only, so ``topk`` need not be a multiple of four.
    """
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if not row_bytes:
        raise ValueError("at least one region is required")
    for width in row_bytes:
        if width <= 0 or width % PUSH_MIN_BYTES:
            raise ValueError(
                f"region row width {width} must be a positive multiple of "
                f"{PUSH_MIN_BYTES}"
            )
    regions = len(row_bytes)
    epoch_index = desc_size(tp_size, regions) + AG_DESC_EPOCH
    tag = "_".join(str(w) for w in row_bytes)
    name = f"mega_moe_tp_ag_push_tp{tp_size}_b{block}_{tag}"

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def ag_push_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        gid = bid * fx.Int32(block) + tid
        stride = i32_grid * fx.Int32(block)

        for slot in range_constexpr(tp_size):
            # Stagger by rank *and* by block: with a rank-only rotation every
            # CTA on this GPU drives the same peer at the same instant, so one
            # xGMI link carries the whole grid while the other seven idle.
            peer = (i32_rank + bid + fx.Int32(slot)) % fx.Int32(tp_size)
            peer_base = desc_peer_base(arg_desc, peer)
            for region in range_constexpr(regions):
                width = row_bytes[region]
                vec_words = _VEC_WORDS if width % PUSH_VEC_BYTES == 0 else 1
                unit_bytes = vec_words * 4
                nbytes = fx.Int64(i32_rows) * fx.Int64(width)
                src = flat_buffer(
                    desc_slot(arg_desc, desc_region_src(tp_size, region)),
                    fx.Int32,
                    nbytes,
                )
                dst = flat_buffer(
                    peer_base
                    + desc_slot(arg_desc, desc_region_offset(tp_size, region))
                    + fx.Int64(i32_rank) * nbytes,
                    fx.Int32,
                    nbytes,
                )
                units = i32_rows * fx.Int32(width // unit_bytes)
                # No tail guard: both views are bounded by the same
                # num_records, so the hardware drops an over-run store and
                # returns zero for its load.
                for unit in range(gid, units, stride * fx.Int32(PUSH_UNROLL)):
                    staged = [
                        load_words(
                            src,
                            fx.Int32(unit) + fx.Int32(step) * stride,
                            width=vec_words,
                            cache_modifier=1,
                        )
                        for step in range_constexpr(PUSH_UNROLL)
                    ]
                    for step in range_constexpr(PUSH_UNROLL):
                        store_words(
                            dst,
                            fx.Int32(unit) + fx.Int32(step) * stride,
                            staged[step],
                            width=vec_words,
                            cache_modifier=0,
                        )

        # Publish this CTA's stores, then let the last CTA run the rank barrier.
        # Agent release per thread is just s_waitcnt; the system release that
        # follows is a full L2 writeback, so only one thread per CTA pays for
        # it. Every XCD is still covered, because the grid spans all of them.
        comm.fence_agent_release()
        gpu.barrier()
        if tid == fx.Int32(0):
            comm.fence_system_release()
            arrive_addr = desc_slot(arg_desc, DESC_ARRIVE)
            previous = fx.Int32(comm.atomic_add_agent(arrive_addr, fx.Int32(1)))
            if previous == i32_grid - fx.Int32(1):
                # Every CTA has drained its stores. Reset the counter for the
                # next launch (the stream serializes launches, so nothing can
                # race this), hand each peer our epoch, and wait for theirs.
                comm.store_i32_global_agent_release(arrive_addr, fx.Int32(0))
                epoch_addr = desc_slot(arg_desc, epoch_index)
                epoch = fx.Int32(comm.load_i32_global_agent(epoch_addr)) + fx.Int32(1)
                comm.store_i32_global_agent_release(epoch_addr, epoch)
                flags_offset = desc_slot(arg_desc, DESC_FLAGS)
                for slot in range_constexpr(tp_size):
                    peer = (i32_rank + fx.Int32(slot)) % fx.Int32(tp_size)
                    comm.store_i32_global_system_release(
                        desc_peer_base(arg_desc, peer)
                        + flags_offset
                        + fx.Int64(i32_rank) * fx.Int64(4),
                        epoch,
                    )
                local_base = desc_peer_base(arg_desc, i32_rank)
                for source in range_constexpr(tp_size):
                    comm.spin_until_ge_i32_system(
                        local_base + flags_offset + fx.Int64(source * 4),
                        epoch,
                        acquire=True,
                    )
                comm.fence_system_acquire()

    @flyc.jit
    def launch(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        ag_push_kernel(arg_desc, i32_rank, i32_rows, i32_grid).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.block = block
    return launch


def push_units(rows: int, row_bytes) -> int:
    """Grid-stride work items one push covers, across every region."""
    total = 0
    for width in row_bytes:
        unit = PUSH_VEC_BYTES if width % PUSH_VEC_BYTES == 0 else PUSH_MIN_BYTES
        total += rows * (width // unit)
    return total


def ag_desc_size(tp_size: int, regions: int) -> int:
    return desc_size(tp_size, regions) + 2


def ag_desc_slot(tp_size: int, regions: int, which: int) -> int:
    return desc_size(tp_size, regions) + which
