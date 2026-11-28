# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Peer link bandwidth probe: how fast *can* this GPU read from its peers?

Combine reads remote rows at 232 GB/s while dispatch writes remote rows at 359
GB/s (MI355X, S=8192 H=7168 K=8 E=384 R=8, upstream ``bench_comm`` methodology).
Upstream MoonEP hides its own read latency behind a 16-stage TMA pipeline into
shared memory and reaches 650 GB/s -- on B300 over NVLink5, which is not our
link.  Before building an AMD analogue of that pipeline it has to be settled
whether depth is the lever at all, because a back-of-envelope says it is not:

    ~8 loads in flight/thread x 16 B x 64 lanes x ~32 waves/CU x 256 CU
        ~= 65 MB in flight

At 232 GB/s that would imply a 280 us memory latency.  Real remote latency is
1-3 us, so the wave-level pipeline is already ~100x deeper than the
bandwidth-delay product needs, and the limit must be somewhere else -- the link
itself, or a queue between the CU and it.

This module is the decisive experiment: the simplest possible streaming access
(fully coalesced, contiguous, no scatter, no unpack, no accumulate chain beyond
one add) against peer memory, with the in-flight depth as a free parameter.

* ``direction="read"``  -- upper bound for combine.
* ``direction="write"`` -- upper bound for dispatch, and the control that says
  whether the read/write asymmetry is real on this link.
* peers are rotated by block index, so all links are driven concurrently the
  way combine drives them, rather than one link at a time.

Nothing here is part of the MoonEP pipeline; it exists to size the next step.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, range_constexpr
from flydsl.expr.typing import Stream

from aiter.ops.flydsl.kernels import vector as fly_vector
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

VEC_I32 = 4
# Any value the summed payload will not produce, so the sink store is never
# taken but the compiler cannot prove the loads dead.
SINK_SENTINEL = 0x7ADBEEF


def probe_stride_dwords(*, block_num: int, block_threads: int, depth: int) -> int:
    """Dword span one full grid pass covers at this geometry.

    ``covered_dwords`` must be a multiple of this so the kernel never runs a
    partial tail, and callers sweeping ``depth`` should size it with the sweep's
    largest depth so every point moves exactly the same bytes.
    """
    return block_num * block_threads * VEC_I32 * depth


def make_link_probe_jit(
    *,
    covered_dwords: int,
    num_peers: int,
    depth: int,
    block_num: int,
    block_threads: int = 256,
    direction: str = "read",
    cache_modifier: int = 0,
):
    """Build a streaming peer-memory probe.

    Args:
        covered_dwords: dwords touched per peer buffer; must be a multiple of
            ``probe_stride_dwords(...)``.
        num_peers: length of the device-side pointer table.  The table holds
            whatever the caller wants measured -- remote peers only, the local
            buffer repeated, or a mix.
        depth: loads (or stores) issued back to back before any is consumed.
        direction: ``"read"`` or ``"write"``.
        cache_modifier: raw aux/cachepolicy bits on the buffer instruction.
            On gfx940+ these are bit0=sc0, bit1=nt, bit4=sc1.  Peer traffic has
            no reuse, so the default policy may be paying for cache lookups and
            coherence it cannot use; 0 is the current production setting and
            everything else here is the experiment.
    """

    if direction not in ("read", "write"):
        raise ValueError("direction must be 'read' or 'write'")
    if covered_dwords <= 0 or num_peers <= 0 or depth <= 0:
        raise ValueError("covered_dwords, num_peers, and depth must be positive")
    if block_num <= 0 or block_threads <= 0 or block_threads % 64 != 0:
        raise ValueError("block_num must be positive and block_threads a wave multiple")

    stride_one = block_num * block_threads * VEC_I32
    stride_total = stride_one * depth
    if covered_dwords % stride_total != 0:
        raise ValueError(
            f"covered_dwords={covered_dwords} is not a multiple of "
            f"stride_total={stride_total}"
        )

    name = (
        f"moonep_link_probe_{direction}_p{num_peers}_d{depth}"
        f"_b{block_num}_t{block_threads}_c{covered_dwords}_cm{cache_modifier}"
    )

    # Two whole kernels rather than one with a ``direction`` branch: inside a
    # kernel body FlyDSL's AST rewriter turns every ``if`` into ``scf.if``, so a
    # Python-level predicate there is a compile error, not a trace-time choice.
    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def probe_read(
        addr_peer_ptrs: fx.Int64,  # INT64 [num_peers], device-side pointer table
        addr_sink: fx.Int64,  # INT32 [block_num], keeps the loads alive
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        ptrs_rsrc = create_buffer_resource_from_addr(addr_peer_ptrs)
        sink_rsrc = create_buffer_resource_from_addr(addr_sink)
        start = (bid * fx.Int32(block_threads) + tid) * fx.Int32(VEC_I32)
        acc = fx.Int32(0)

        for p in range_constexpr(num_peers):
            # Rotate by block so every link is driven at once.  Walking the
            # table in order would measure one link at a time, which is not how
            # combine uses them.
            slot = (bid + fx.Int32(p)) % fx.Int32(num_peers)
            base = buffer_load(ptrs_rsrc, slot, vec_width=1, dtype=T.i64)
            rsrc = create_buffer_resource_from_addr(base)
            for off in range(start, covered_dwords, stride_total):
                vals = []
                for d in range_constexpr(depth):
                    vals.append(
                        buffer_load(
                            rsrc,
                            off + fx.Int32(d * stride_one),
                            vec_width=VEC_I32,
                            dtype=T.i32,
                            cache_modifier=cache_modifier,
                        )
                    )
                for d in range_constexpr(depth):
                    for pos in range_constexpr(VEC_I32):
                        acc = acc + fx.Int32(
                            fly_vector.extract(
                                vals[d],
                                static_position=[pos],
                                dynamic_position=[],
                            )
                        )

        # Never taken; exists so the accumulator -- and therefore every load --
        # is live.  One slot per block, so no contention.
        if acc == fx.Int32(SINK_SENTINEL):
            buffer_store(acc, sink_rsrc, bid)

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def probe_write(
        addr_peer_ptrs: fx.Int64,
        addr_sink: fx.Int64,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        ptrs_rsrc = create_buffer_resource_from_addr(addr_peer_ptrs)
        start = (bid * fx.Int32(block_threads) + tid) * fx.Int32(VEC_I32)
        payload = fly_vector.from_elements(
            T.vec(VEC_I32, T.i32), [fx.Int32(0)] * VEC_I32
        )

        for p in range_constexpr(num_peers):
            slot = (bid + fx.Int32(p)) % fx.Int32(num_peers)
            base = buffer_load(ptrs_rsrc, slot, vec_width=1, dtype=T.i64)
            rsrc = create_buffer_resource_from_addr(base)
            for off in range(start, covered_dwords, stride_total):
                for d in range_constexpr(depth):
                    buffer_store(
                        payload,
                        rsrc,
                        off + fx.Int32(d * stride_one),
                        cache_modifier=cache_modifier,
                    )

    kernel = probe_read if direction == "read" else probe_write

    @flyc.jit
    def launch(
        addr_peer_ptrs: fx.Int64,
        addr_sink: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        kernel(addr_peer_ptrs, addr_sink).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_link_probe_jit", "probe_stride_dwords", "VEC_I32"]
