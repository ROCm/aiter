# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Latency-tuned MoonEP remote weight prefetch.

Drop-in replacement for ``moonep_weight_prefetch``: identical launch ABI and
byte-identical output (it is a pure copy), so the two can be A/B'd against the
same buffers.  The reference version stays untouched.

Why
---
Measured on MI355X (8 ranks, H=7168 I=2048, one weight matrix = 29.4 MB,
busiest-rank bytes over slowest-rank time, upstream ``bench_comm`` methodology):

=====================  ==========  ===========
routing                    us          GB/s
=====================  ==========  ===========
uniform (1 slot)          706.0         41.6
profile (6 slots)        3177.4         55.4
=====================  ==========  ===========

``moonep_link_probe`` puts this machine's remote *read* ceiling at 235 GB/s, so
prefetch runs at 18-24% of what the link allows.  That is unlike combine, which
already sits at 99% of the ceiling -- here there is a real implementation gap.
Upstream reaches 640-755 GB/s on B300 at the same 29.4 MB matrix size; roughly
2.8x of that lead is its link, the rest is the two structural differences copied
here.

What upstream does (``MoonEP/moonep/prefetch.py``)
--------------------------------------------------
1. **Compacts the valid slots** into a shared-memory ``slot_tab`` (``:165``,
   ``:212``) and then only walks live entries.
2. **Splits load from store across warps** behind a 2-6 stage shared-memory ring
   (``:196-198`` pipeline, ``:227`` producer TMA g2s, ``:253`` consumer TMA
   s2g), so a store never waits on the load that produced it.
3. 2D TMA with 128x128 tiles -- Hopper/Blackwell only, and not needed here.

What this file does instead
---------------------------
1. **Slot-outer loop nest.**  The reference does a single flat grid-stride over
   the whole ``prefetch_slots * weight`` address space, so with B=48 slots and
   one live slot each thread runs 336 iterations of which 329 only load a slot
   id and branch away.  Walking slots on the outside and chunks on the inside
   gives the same effect as upstream's compaction -- dead slots cost one
   wave-uniform branch for the whole block -- without a shared-memory table.
   Every block still works on every live slot, so each one keeps full grid
   parallelism.
2. **Batched issue instead of a warp split.**  The reference pairs each remote
   ``buffer_load`` with the ``buffer_store`` that consumes it, exposing one full
   remote latency per 16 B.  Here a whole batch of loads is issued before any
   store.  Upstream needs shared memory for this because TMA lands there; we do
   not, because a copy has **no data reuse** -- the only thing staging buys is
   register pressure relief, and 16 vec4 is 64 VGPRs, which is affordable.  The
   same restructuring took the dispatch epilogue from 298 us to 160 us.
3. The slot metadata (expert id, owner, peer base) is resolved once per slot
   rather than once per 16 B.

Tail handling
-------------
Both buffer resources are built with ``num_records_bytes = weight_bytes``, so
out-of-range lanes in the last batch are dropped by the hardware bounds check on
both the load and the store side.  That removes the tail predicate entirely; the
batch size is still clamped to what the weight actually needs so the overshoot
stays under one batch.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, range_constexpr
from flydsl.expr.typing import Stream

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

VEC_I32 = 4
# Remote loads kept in flight per thread before the matching stores are issued.
# 16 vec4 == 64 VGPRs of payload, which leaves plenty of occupancy headroom.
DEFAULT_LOADS_IN_FLIGHT = 16


def make_moonep_weight_prefetch_fast_jit(
    *,
    experts_per_rank: int,
    prefetch_slots: int,
    weight_numel: int,
    elem_bytes: int = 2,
    block_num: int = 128,
    block_threads: int = 256,
    loads_in_flight: int = DEFAULT_LOADS_IN_FLIGHT,
):
    """Build the tuned prefetch launcher (same ABI as the reference builder).

    The copy is untyped -- only ``weight_numel * elem_bytes`` matters -- so fp8
    weights and their scale blocks go through unchanged.
    """

    if experts_per_rank <= 0 or prefetch_slots <= 0:
        raise ValueError("expert and slot counts must be positive")
    if weight_numel <= 0 or elem_bytes <= 0:
        raise ValueError("weight size and element size must be positive")
    weight_bytes = weight_numel * elem_bytes
    if weight_bytes % 16 != 0:
        raise ValueError(
            f"expert weight must be 16-byte aligned, got {weight_bytes} bytes"
        )
    if block_num <= 0 or block_threads <= 0 or block_threads % 64 != 0:
        raise ValueError("launch geometry must be positive and wave-aligned")
    if loads_in_flight <= 0:
        raise ValueError("loads_in_flight must be positive")

    weight_i32 = weight_bytes // 4
    stride = block_num * block_threads * VEC_I32
    # One grid pass covers ``stride`` dwords, so a weight needs this many passes.
    passes = (weight_i32 + stride - 1) // stride
    batch = min(loads_in_flight, passes)
    span = stride * batch
    name = (
        f"moonep_weight_prefetch_fast_epr{experts_per_rank}_b{prefetch_slots}"
        f"_n{weight_numel}x{elem_bytes}_g{block_num}_t{block_threads}_f{batch}"
    )

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def prefetch_kernel(
        addr_experts_to_copy: fx.Int64,  # INT32 [B], global expert ids, -1 = idle
        addr_peer_home_weight_ptrs: fx.Int64,  # INT64 [world_size]
        addr_prefetched_weights: fx.Int64,  # BF16 [B, weight_numel]
    ):
        tid = fx.Int32(fx.thread_idx.x)
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(block_threads) + tid
        lane_base = gid * fx.Int32(VEC_I32)
        experts_rsrc = create_buffer_resource_from_addr(addr_experts_to_copy)
        peer_ptrs_rsrc = create_buffer_resource_from_addr(
            addr_peer_home_weight_ptrs
        )

        # Slots on the outside: an idle slot costs one wave-uniform branch for
        # the whole block instead of 1/B of every thread's iterations.
        for slot in range(0, prefetch_slots, 1):
            expert = buffer_load(experts_rsrc, slot, vec_width=1, dtype=T.i32)
            # Depends only on the slot, so this is block-uniform and compiles to
            # a scalar branch -- an idle slot issues no memory traffic at all.
            if expert >= fx.Int32(0):
                owner = expert // fx.Int32(experts_per_rank)
                local_expert = expert % fx.Int32(experts_per_rank)
                owner_base = buffer_load(
                    peer_ptrs_rsrc, owner, vec_width=1, dtype=T.i64
                )
                # num_records bounds both sides to one weight, so the final
                # batch's out-of-range lanes are dropped in hardware and no tail
                # predicate is needed.
                src_rsrc = create_buffer_resource_from_addr(
                    owner_base + fx.Int64(local_expert) * weight_bytes,
                    num_records_bytes=weight_bytes,
                )
                dst_rsrc = create_buffer_resource_from_addr(
                    addr_prefetched_weights + fx.Int64(slot) * weight_bytes,
                    num_records_bytes=weight_bytes,
                )
                for base in range(lane_base, weight_i32, span):
                    # Issue the whole batch before consuming any of it: this is
                    # the register-resident equivalent of upstream's
                    # producer/consumer warp split through shared memory.
                    values = []
                    for j in range_constexpr(batch):
                        values.append(
                            buffer_load(
                                src_rsrc,
                                base + fx.Int32(j * stride),
                                vec_width=VEC_I32,
                                dtype=T.i32,
                            )
                        )
                    for j in range_constexpr(batch):
                        buffer_store(
                            values[j],
                            dst_rsrc,
                            base + fx.Int32(j * stride),
                        )

    @flyc.jit
    def launch(
        addr_experts_to_copy: fx.Int64,
        addr_peer_home_weight_ptrs: fx.Int64,
        addr_prefetched_weights: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        prefetch_kernel(
            addr_experts_to_copy,
            addr_peer_home_weight_ptrs,
            addr_prefetched_weights,
        ).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_weight_prefetch_fast_jit", "DEFAULT_LOADS_IN_FLIGHT"]
