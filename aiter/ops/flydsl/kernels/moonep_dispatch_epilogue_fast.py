# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Latency-tuned local epilogue for MoonEP dispatch.

Drop-in replacement for ``moonep_dispatch_epilogue``: identical launch ABI and
identical output, so the two can be A/B'd against the same correctness test.
The reference version stays untouched.

Why
---
Measured on MI355X at S=8192 H=7168 K=8 E=384 R=8 (block_num=256), isolated by
sweeping ``token_padding`` (which changes only the zero-fill work):

===================  ========  =========  ==========  ===================
kernel                     us      bytes  achieved    share of ~8 TB/s HBM
===================  ========  =========  ==========  ===================
expand_duplicates       298.4   636.7 MB   2.13 TB/s   27%
zero_padding             69.2    38.5 MB   0.56 TB/s    7%
===================  ========  =========  ==========  ===================

``expand_duplicates`` is **latency**-bound, not bandwidth-bound: the reference
copies a row with 14 iterations of ``buffer_load`` immediately followed by the
dependent ``buffer_store``, so every one of those 14 memory latencies is
exposed.  1024 waves x 76 rows/wave x 14 x ~800 cycles is ~340 us at 2.5 GHz,
which is the number that was measured.  A row is only 14 vec4 registers per
lane, so this version issues a whole batch of loads before any store and pays
one latency per batch instead of one per vec4.

``zero_padding`` had two separate problems: it stored **four scalar i32 values**
where the expand kernel next to it stores one vec4 (4x the store instructions),
and it gave one wavefront to a whole group and walked that group's padding rows
serially, so only ``num_groups`` waves had work.  Here every wave grid-strides
over the padding rows of every group.

Not addressed here: the primary row is still re-read once per duplicate.
Upstream instead builds a compact ``(primary_loff, dup_start, dup_n)`` group
table inside dispatch and reads each primary once, which is ~12% less traffic;
that is a plan/dispatch-side change, not an epilogue one.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, range_constexpr
from flydsl.expr.typing import Stream

from flydsl.expr import rocdl as fly_rocdl

from aiter.ops.flydsl.kernels import vector as fly_vector
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

WAVE_SIZE = 64
# vec4 of i32 == 16 B per lane, so one wave moves 1024 B per instruction.
VEC_I32 = 4
LANE_STRIDE_I32 = WAVE_SIZE * VEC_I32

# Loads kept in flight per wave before the matching stores are issued.  A whole
# 7168-wide bf16 row is 14 vec4 per lane, so the default covers a full row while
# staying well inside the register budget.
DEFAULT_LOADS_IN_FLIGHT = 16


def make_moonep_dispatch_epilogue_fast_jit(
    *,
    hidden_dim: int,
    num_dispatch_rows: int,
    num_groups: int,
    block_num: int = 256,
    warp_num_per_block: int = 4,
    loads_in_flight: int = DEFAULT_LOADS_IN_FLIGHT,
):
    """Build the tuned epilogue launcher (same ABI as the reference builder)."""

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if num_dispatch_rows <= 0 or num_groups <= 0:
        raise ValueError("row and group counts must be positive")
    if block_num <= 0 or warp_num_per_block <= 0:
        raise ValueError("launch geometry must be positive")
    if loads_in_flight <= 0:
        raise ValueError("loads_in_flight must be positive")

    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    block_threads = warp_num_per_block * WAVE_SIZE
    global_warps = block_num * warp_num_per_block

    # hidden_dim % 8 == 0 makes hidden_i32 a multiple of 4, so the tail is a
    # whole number of vec4 and only needs a lane bound, never a byte mask.
    full_chunks = hidden_i32 // LANE_STRIDE_I32
    tail_vecs = (hidden_i32 - full_chunks * LANE_STRIDE_I32) // VEC_I32
    batch = min(loads_in_flight, full_chunks) if full_chunks else 0
    num_batches = (full_chunks + batch - 1) // batch if batch else 0
    tail_off = full_chunks * LANE_STRIDE_I32

    suffix = (
        f"h{hidden_dim}_nvs{num_dispatch_rows}_g{num_groups}"
        f"_b{block_num}_w{warp_num_per_block}_f{batch}"
    )

    @flyc.kernel(
        name=f"moonep_expand_duplicates_fast_{suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def expand_duplicates(
        addr_hidden: fx.Int64,  # BF16 [NvS, H]
        addr_duplicate_src: fx.Int64,  # INT32 [NvS], -1 for primary/padding
    ):
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        warp = tid >> fx.Int32(6)
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        duplicate_src_rsrc = create_buffer_resource_from_addr(addr_duplicate_src)
        lane_off = lane * fx.Int32(VEC_I32)

        # Per-row grid-stride.  A cooperative ballot scan (one vectorized
        # duplicate_src load per 64 rows) was measured and is SLOWER: giving a
        # wave a whole 64-row window forces it to walk that window's duplicates
        # serially, and duplicates are not spread evenly across windows, so the
        # worst wave ends up with more copies than the per-row split gives it
        # (240.5 us vs 206.2 us end to end).  The scan loads are cheap because
        # successive iterations are independent and overlap with the copies.
        for dst_row in range(global_warp, num_dispatch_rows, global_warps):
            src_row = buffer_load(
                duplicate_src_rsrc, dst_row, vec_width=1, dtype=T.i32
            )
            # One wave owns one row, so this predicate is wave-uniform and the
            # whole copy is skipped with a scalar branch on non-duplicate rows.
            if src_row >= fx.Int32(0):
                src_rsrc = create_buffer_resource_from_addr(
                    addr_hidden + fx.Int64(src_row) * hidden_bytes
                )
                dst_rsrc = create_buffer_resource_from_addr(
                    addr_hidden + fx.Int64(dst_row) * hidden_bytes
                )
                # range_constexpr, not range: inside a kernel body FlyDSL
                # rewrites every ``for`` into scf.for even with constant
                # bounds, which would put the loads back on the critical path.
                for b in range_constexpr(num_batches):
                    base = b * batch
                    n = min(batch, full_chunks - base)
                    values = []
                    for j in range_constexpr(n):
                        values.append(
                            buffer_load(
                                src_rsrc,
                                lane_off
                                + fx.Int32((base + j) * LANE_STRIDE_I32),
                                vec_width=VEC_I32,
                                dtype=T.i32,
                            )
                        )
                    for j in range_constexpr(n):
                        buffer_store(
                            values[j],
                            dst_rsrc,
                            lane_off + fx.Int32((base + j) * LANE_STRIDE_I32),
                        )
                # Tail chunk.  Guarded only by the lane predicate: when
                # tail_vecs == 0 this folds to ``lane < 0`` and never runs, so
                # no Python-level ``if`` is needed inside the kernel body.
                if lane < fx.Int32(tail_vecs):
                    off = lane_off + fx.Int32(tail_off)
                    buffer_store(
                        buffer_load(
                            src_rsrc, off, vec_width=VEC_I32, dtype=T.i32
                        ),
                        dst_rsrc,
                        off,
                    )

    @flyc.kernel(
        name=f"moonep_zero_padding_fast_{suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def zero_padding(
        addr_hidden: fx.Int64,  # BF16 [NvS, H]
        addr_weights: fx.Int64,  # FP32 [NvS]
        addr_zero_fill_ranges: fx.Int64,  # INT32 [G, 2]
    ):
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        warp = tid >> fx.Int32(6)
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        ranges_rsrc = create_buffer_resource_from_addr(addr_zero_fill_ranges)
        weights_rsrc = create_buffer_resource_from_addr(addr_weights)
        lane_off = lane * fx.Int32(VEC_I32)
        zero_vec = fly_vector.from_elements(
            T.vec(VEC_I32, T.i32), [fx.Int32(0)] * VEC_I32
        )

        # Every wave walks every group and grid-strides that group's padding
        # rows, so all waves get work instead of only the first ``num_groups``.
        # The two range loads are at wave-uniform addresses and stay in cache.
        for group in range(0, num_groups, 1):
            start = buffer_load(
                ranges_rsrc, group * 2, vec_width=1, dtype=T.i32
            )
            count = buffer_load(
                ranges_rsrc, group * 2 + 1, vec_width=1, dtype=T.i32
            )
            for pad_idx in range(global_warp, count, global_warps):
                row = start + pad_idx
                row_rsrc = create_buffer_resource_from_addr(
                    addr_hidden + fx.Int64(row) * hidden_bytes
                )
                for c in range_constexpr(full_chunks):
                    buffer_store(
                        zero_vec,
                        row_rsrc,
                        lane_off + fx.Int32(c * LANE_STRIDE_I32),
                    )
                if lane < fx.Int32(tail_vecs):
                    buffer_store(
                        zero_vec, row_rsrc, lane_off + fx.Int32(tail_off)
                    )
                if lane == fx.Int32(0):
                    buffer_store(fx.Int32(0), weights_rsrc, row)

    @flyc.jit
    def launch(
        addr_hidden: fx.Int64,
        addr_weights: fx.Int64,
        addr_duplicate_src: fx.Int64,
        addr_zero_fill_ranges: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        expand_duplicates(addr_hidden, addr_duplicate_src).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        zero_padding(addr_hidden, addr_weights, addr_zero_fill_ranges).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_dispatch_epilogue_fast_jit"]
