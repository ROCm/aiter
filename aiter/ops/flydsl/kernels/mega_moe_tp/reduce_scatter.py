# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReduceScatter for the fused TP MoE layer.

GEMM2 leaves every rank holding a full ``[M, H]`` partial -- its own
``inter_dim`` shard's contribution to every global token.  Rank ``r`` owns
global rows ``[r*m, (r+1)*m)``, so the layer needs
``y_local = sum_over_ranks(partial)[r*m : (r+1)*m]``.

Shape
-----
1. :func:`compile_reduce_scatter_publish` -- a data-free kernel that writes
   back L2 and hands every peer a monotone epoch, announcing that this rank's
   GEMM2 output is readable.
2. :func:`compile_reduce_scatter_pull` -- every rank reads its own row range out
   of all ``TP`` partials and sums them in FP32 (strictly more accurate than a
   BF16 tree reduction), writing the local ``[m, H]`` output.

Why pull, and why two launches
------------------------------
The alternative -- push each peer's row range into a staging slot, then reduce
locally -- turns the cross-GPU traffic into fire-and-forget writes, but it also
adds a full ``M*H*2`` local read/write pass and a second buffer.  Measured on
TP8 / H=3584 the staged push is ~1.3x *slower* than pulling at prefill sizes, so
the round-trip cost of the remote reads is the smaller of the two prices.

The split into two launches is what lets both grids be sized purely for
bandwidth: a single kernel would have to hold every CTA at a grid-wide barrier,
capping the grid at what the device holds at once, and that cap is exactly what
starves the pull of memory parallelism.  Here the publish kernel's CTAs only
fetch-and-add (the last one publishes and leaves) and every pull CTA spins on
flags its peers have already written, so no CTA ever waits on a sibling.

Epochs are device-side: the publish reads and bumps a counter in the arena
rather than taking a number from the host, so a captured graph replays correctly
instead of re-using a flag value that is already satisfied.
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
    DESC_PEER_BASE,
    desc_peer_base,
    desc_slot,
    flat_buffer,
    load_words,
    store_words,
)

__all__ = [
    "ARRIVE_STRIDE_BYTES",
    "PHASE_CTR_SLOT",
    "PHASE_GATE_SLOT",
    "WRITEBACK_BLOCKS",
    "RS_DESC_FANIN",
    "MAX_SERVICE_BLOCKS",
    "RS_ARRIVE_SLOTS",
    "RS_DESC_EPOCH",
    "RS_DESC_OUTPUT",
    "RS_DESC_PARTIAL",
    "RS_PULL_UNROLL",
    "RS_UNIT_ELEMS",
    "compile_reduce_scatter_publish",
    "compile_reduce_scatter_pull",
    "rs_desc_size",
    "rs_desc_slot",
]

_PULL_BLOCK = int(os.environ.get("AITER_TP_RS_BLOCK", "1024"))
# CTAs in the publish kernel. It moves no data; the only requirement is that it
# spans every XCD, because one thread's buffer_wbl2 writes back just its own
# XCD's L2 and a peer must not be able to read a stale GEMM2 line.
_PUBLISH_BLOCKS = 64
_PUBLISH_BLOCK = 64
# 8 BF16 = 16 B = one dwordx4 per lane.
RS_UNIT_ELEMS = 8
_VEC_WORDS = RS_UNIT_ELEMS // 2
# Accesses a thread issues before waiting on any of them.
RS_PULL_UNROLL = 1

# Descriptor slots beyond the shared prefix (arrive, flags, peer bases).
RS_DESC_PARTIAL = 0  # arena byte offset of the [M, H] GEMM2 partial
RS_DESC_OUTPUT = 1  # local address of the [m, H] output
RS_DESC_EPOCH = 2  # local address of the monotone epoch counter (i32)
RS_DESC_FANIN = 3  # local address of the fused tail's fan-in counters (i32[])
_RS_EXTRA = 4

#: Fan-in counters the fused GEMM2 tail arrives at, plus one second-level
#: counter after them.
#:
#: These live in ordinary device memory, *not* in the symmetric arena. No peer
#: ever reads them -- they only say "this rank's GEMM2 tiles are all done" --
#: and arena pages are IPC-exported, so atomics on them bypass L2 and pay a
#: fabric round trip each. That is invisible for the standalone publish kernel's
#: 64 CTAs and ruinous for a fused tail, where every one of a prefill GEMM2
#: grid's ~25k CTAs arrives: measured at 1.9 ms, versus 0.4 us for the same
#: kernel with the tail compiled out.
MAX_SERVICE_BLOCKS = 256
#: Each counter gets its own 128-byte cache line. Packing them adjacently puts
#: all of them on two lines, which is no better than a single counter: the line
#: ping-pongs between XCDs on every arrival and a prefill GEMM2 grid arrives
#: tens of thousands of times. One line per counter, with a counter index of
#: ``block % service`` and ``service`` a multiple of the eight XCDs, keeps each
#: line resident in exactly one XCD's L2.
ARRIVE_STRIDE_DW = 32
ARRIVE_STRIDE_BYTES = ARRIVE_STRIDE_DW * 4
#: One line per fan-in counter, then six more: the whole-grid fan-in root, the
#: flag that releases every service CTA once that root fills, the writeback
#: fan-in root, the flag that says every peer has published, and a
#: counter/gate pair for the inter-phase barrier a kernel hosting more than one
#: GEMM needs.
RS_ARRIVE_SLOTS = (MAX_SERVICE_BLOCKS + 6) * ARRIVE_STRIDE_DW
#: Line indices of the phase-barrier pair, past the four the RS tail owns.
PHASE_CTR_SLOT = MAX_SERVICE_BLOCKS + 4
PHASE_GATE_SLOT = MAX_SERVICE_BLOCKS + 5
#: L2 is per XCD and ``buffer_wbl2`` writes back only the issuing XCD's, so this
#: many service CTAs -- consecutive block ids, which round-robin the XCDs -- is
#: exactly enough to cover the device. Letting *every* service CTA do it instead
#: is what made a 256-CTA service group 2.3x slower than a 32-CTA one.
WRITEBACK_BLOCKS = 8


def rs_desc_size(tp_size: int) -> int:
    return DESC_PEER_BASE + tp_size + _RS_EXTRA


def rs_desc_slot(tp_size: int, which: int) -> int:
    return DESC_PEER_BASE + tp_size + which


def _validate(tp_size: int, model_dim: int) -> int:
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if model_dim <= 0 or model_dim % RS_UNIT_ELEMS:
        raise ValueError(
            f"model_dim must be a positive multiple of {RS_UNIT_ELEMS}, "
            f"got {model_dim}"
        )
    return model_dim // RS_UNIT_ELEMS


@functools.cache
def compile_reduce_scatter_publish(tp_size: int, *, block: int = _PUBLISH_BLOCK):
    """Announce that this rank's GEMM2 partial is visible to its peers."""
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    name = f"mega_moe_tp_rs_publish_tp{tp_size}"

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def rs_publish_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        if tid == fx.Int32(0):
            # One writeback per CTA, and the grid spans every XCD, so all of
            # GEMM2's dirty lines reach memory where the peers can read them.
            comm.fence_system_release()
            arrive_addr = desc_slot(arg_desc, DESC_ARRIVE)
            previous = fx.Int32(comm.atomic_add_agent(arrive_addr, fx.Int32(1)))
            if previous == i32_grid - fx.Int32(1):
                # The stream serializes launches, so resetting here cannot race
                # the next call's arrivals.
                comm.store_i32_global_agent_release(arrive_addr, fx.Int32(0))
                epoch_addr = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_EPOCH))
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

    launcher = _wrap_launch(rs_publish_kernel, block, 0, name)
    launcher.blocks = _PUBLISH_BLOCKS
    return launcher


@functools.cache
def compile_reduce_scatter_pull(
    tp_size: int,
    model_dim: int,
    *,
    block: int = _PULL_BLOCK,
):
    """Sum this rank's row range across every peer's partial."""
    units_per_row = _validate(tp_size, model_dim)
    row_bytes = model_dim * 2
    name = f"mega_moe_tp_rs_pull_tp{tp_size}_h{model_dim}_b{block}"

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def rs_pull_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        gid = bid * fx.Int32(block) + tid
        stride = i32_grid * fx.Int32(block)
        units = i32_rows * fx.Int32(units_per_row)

        local_base = desc_peer_base(arg_desc, i32_rank)
        # Every CTA spins on its own GPU's flags, which the peers' publish
        # kernels write, so no CTA waits on a sibling and the grid is unbounded.
        if tid == fx.Int32(0):
            epoch_addr = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_EPOCH))
            epoch = fx.Int32(comm.load_i32_global_agent(epoch_addr))
            flags_offset = desc_slot(arg_desc, DESC_FLAGS)
            for source in range_constexpr(tp_size):
                comm.spin_until_ge_i32_system(
                    local_base + flags_offset + fx.Int64(source * 4),
                    epoch,
                    acquire=True,
                    sleep=False,
                )
            comm.fence_system_acquire()
        gpu.barrier()

        partial_offset = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_PARTIAL))
        total_bytes = fx.Int64(i32_rows) * fx.Int64(tp_size) * fx.Int64(row_bytes)
        # Stagger by rank *and* by block: with a rank-only rotation every CTA on
        # this GPU pulls from the same peer at the same instant, so one xGMI
        # link carries the whole grid while the other seven idle.
        sources = []
        for slot in range_constexpr(tp_size):
            peer = (i32_rank + bid + fx.Int32(slot)) % fx.Int32(tp_size)
            sources.append(
                flat_buffer(
                    desc_peer_base(arg_desc, peer) + partial_offset,
                    fx.Int32,
                    total_bytes,
                )
            )
        output = flat_buffer(
            desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_OUTPUT)),
            fx.Int32,
            fx.Int64(i32_rows) * fx.Int64(row_bytes),
        )
        shard_base = i32_rank * units
        # No tail guard: the views are bounded by num_records, so the hardware
        # drops an over-run store and zero-fills its load.
        for unit in range(gid, units, stride * fx.Int32(RS_PULL_UNROLL)):
            for step in range_constexpr(RS_PULL_UNROLL):
                index = fx.Int32(unit) + fx.Int32(step) * stride
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

    return _wrap_launch(rs_pull_kernel, block, units_per_row, name)


def _wrap_launch(kernel, block, units_per_row, name):
    @flyc.jit
    def launch(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_desc, i32_rank, i32_rows, i32_grid).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.block = block
    launch.units_per_row = units_per_row
    launch.__name__ = f"launch_{name}"
    return launch
