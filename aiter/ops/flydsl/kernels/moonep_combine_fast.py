# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

# NOTE: no ``from __future__ import annotations`` needed here, but keep the
# module free of Python-level control flow inside kernel bodies -- FlyDSL's AST
# rewriter owns every ``for``/``if`` there, including ones with constant bounds.

"""Latency-tuned MoonEP combine.

Drop-in replacement for ``moonep_combine``: identical launch ABI and identical
output, so the two can be A/B'd against the same correctness check.  The
reference version stays untouched.

Why
---
Measured on MI355X at S=8192 H=7168 K=8 E=384 R=8, cross-rank mean, with the
upstream ``bench_comm.time_gpu_op`` methodology:

=========================  =========  ===========
op                                us    wire GB/s
=========================  =========  ===========
dispatch (remote writes)      1509.2        359.1
dispatch, idealised addr      2142.7        383.7
combine (remote reads)        3531.8        232.8
=========================  =========  ===========

Dispatch sits within 6.5% of what the same mechanism reaches with perfectly
sequential addressing, so 360-384 GB/s is this path's ceiling.  Combine reads
over the same link at 233 GB/s -- 65% of that -- so it is not link-bound, it is
read-latency bound.

The reference kernel makes every payload load wait on a two-deep chain:

    encoded   = load(dst[token*K + k])          # loop invariant, reloaded
    peer_base = load(peer_ptrs[peer])           # loop invariant, reloaded
    raw_vec   = load(peer_base + row*H*2 + dw)  # the actual data

and both invariants are re-fetched on every iteration of the ``dw_base`` loop
(4 iterations at H=7168).  This version resolves the K row base addresses once
per token, then issues all K payload loads for a chunk before touching any of
them, so K loads are in flight instead of one.  That is the portable half of
what upstream does with a 16-stage TMA/mbarrier pipeline into shared memory
(``MoonEP/moonep/combine.py``: a dedicated LOAD warp feeds four ACC warps).

Still missing versus upstream: upstream's combine **skips duplicate entries**
entirely (``combine.py:320`` ``load_token = dst_val >= 0``) because its
``combine_prologue`` has already reduced each duplicate row into its primary on
the destination rank -- local HBM work instead of link traffic.  We pull all K
rows across the link, which is 8/5.29 = 1.51x the traffic at uniform routing.
Closing that needs a prologue kernel plus the group table, not a combine-side
change.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Stream, T

from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.moonep_combine import (
    _pack_bf16_pair,
    _unpack_bf16_pair,
)

VEC_DWORDS = 4


def make_moonep_combine_fast_jit(
    *,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    block_threads: int = 256,
    gather_route_weights: bool = True,
    apply_route_weights: bool = False,
    skip_duplicates: bool = False,
):
    """Build the tuned combine launcher (same ABI as the reference builder).

    ``skip_duplicates`` matches upstream ``combine.py:320``: duplicate entries
    load no payload because ``moonep_combine_prologue`` already folded them
    into their primary on the destination rank.  It is only correct when that
    prologue has run on the staged input.
    """

    if num_tokens <= 0 or hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("token count and 16-byte-aligned hidden_dim are required")
    if top_k <= 0 or num_dispatch_rows <= 0 or block_threads <= 0:
        raise ValueError("top_k, row count, and block size must be positive")

    row_dwords = hidden_dim // 2
    row_bytes = hidden_dim * 2
    name = (
        f"moonep_combine_fast_bf16_s{num_tokens}_h{hidden_dim}_k{top_k}"
        f"_nvs{num_dispatch_rows}_t{block_threads}"
        f"_gw{int(gather_route_weights)}_aw{int(apply_route_weights)}"
        f"_sd{int(skip_duplicates)}"
    )

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def combine_kernel(
        addr_dst: fx.Int64,  # INT32 [S,K]
        addr_peer_expert_output_ptrs: fx.Int64,  # INT64 [world_size]
        addr_peer_route_weight_ptrs: fx.Int64,  # INT64 [world_size]
        addr_output: fx.Int64,  # BF16 [S,H]
        addr_gathered_route_weights: fx.Int64,  # FP32 [S,K]
    ):
        token = fx.block_idx.x
        tid = fx.thread_idx.x
        dst_rsrc = create_buffer_resource_from_addr(addr_dst)
        peer_ptrs_rsrc = create_buffer_resource_from_addr(
            addr_peer_expert_output_ptrs
        )
        peer_weight_ptrs_rsrc = create_buffer_resource_from_addr(
            addr_peer_route_weight_ptrs
        )
        output_rsrc = create_buffer_resource_from_addr(addr_output)
        gathered_weights_rsrc = create_buffer_resource_from_addr(
            addr_gathered_route_weights
        )

        if tid < top_k:
            route_idx = token * top_k + tid
            encoded = buffer_load(dst_rsrc, route_idx, vec_width=1, dtype=T.i32)
            raw = (encoded >= 0).select(encoded, -encoded - 1)
            peer = raw // num_dispatch_rows
            row = raw % num_dispatch_rows
            peer_weight_rsrc = create_buffer_resource_from_addr(
                buffer_load(peer_weight_ptrs_rsrc, peer, vec_width=1, dtype=T.i64)
            )
            route_weight = buffer_load(
                peer_weight_rsrc, row, vec_width=1, dtype=T.f32
            )
            if gather_route_weights:
                buffer_store(route_weight, gathered_weights_rsrc, route_idx)

        # Resolve the K remote row descriptors once per token.  The reference
        # redoes this inside the chunk loop, which puts two dependent loads in
        # front of every payload load.
        row_rsrcs = []
        weights = []
        take_row = []
        for k_idx in range_constexpr(top_k):
            route_idx = token * top_k + fx.Int32(k_idx)
            encoded = buffer_load(dst_rsrc, route_idx, vec_width=1, dtype=T.i32)
            raw = (encoded >= 0).select(encoded, -encoded - 1)
            peer = raw // num_dispatch_rows
            row = raw % num_dispatch_rows
            peer_base = buffer_load(
                peer_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
            )
            row_rsrcs.append(
                create_buffer_resource_from_addr(
                    peer_base + fx.Int64(row) * row_bytes
                )
            )
            weight = fx.Float32(1.0)
            if apply_route_weights:
                peer_weight_rsrc = create_buffer_resource_from_addr(
                    buffer_load(
                        peer_weight_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
                    )
                )
                weight = fx.Float32(
                    buffer_load(peer_weight_rsrc, row, vec_width=1, dtype=T.f32)
                )
            weights.append(weight)
            take_row.append(encoded >= 0)

        zero_vec = vector.from_elements(
            T.vec(VEC_DWORDS, T.i32), [fx.Int32(0)] * VEC_DWORDS
        )

        for dw_base in range(
            tid * VEC_DWORDS, row_dwords, block_threads * VEC_DWORDS
        ):
            # Issue every remote load for this chunk before consuming any of
            # them, so K loads are in flight instead of a serial chain.
            raw_vecs = []
            for k_idx in range_constexpr(top_k):
                if skip_duplicates:
                    # Block-uniform predicate (token == block_idx), so this is a
                    # scalar branch and the remote load is genuinely skipped.
                    contrib = zero_vec
                    if take_row[k_idx]:
                        contrib = buffer_load(
                            row_rsrcs[k_idx],
                            dw_base,
                            vec_width=VEC_DWORDS,
                            dtype=T.i32,
                        )
                    raw_vecs.append(contrib)
                else:
                    raw_vecs.append(
                        buffer_load(
                            row_rsrcs[k_idx],
                            dw_base,
                            vec_width=VEC_DWORDS,
                            dtype=T.i32,
                        )
                    )

            acc = [fx.Float32(0.0) for _ in range(2 * VEC_DWORDS)]
            for k_idx in range_constexpr(top_k):
                for pos in range_constexpr(VEC_DWORDS):
                    raw_dw = vector.extract(
                        raw_vecs[k_idx], static_position=[pos], dynamic_position=[]
                    )
                    lo, hi = _unpack_bf16_pair(raw_dw)
                    acc[2 * pos] = acc[2 * pos] + weights[k_idx] * lo
                    acc[2 * pos + 1] = acc[2 * pos + 1] + weights[k_idx] * hi

            packed = [
                _pack_bf16_pair(acc[2 * pos], acc[2 * pos + 1])
                for pos in range(VEC_DWORDS)
            ]
            out_vec = vector.from_elements(T.vec(VEC_DWORDS, T.i32), packed)
            buffer_store(out_vec, output_rsrc, token * row_dwords + dw_base)

    @flyc.jit
    def launch(
        addr_dst: fx.Int64,
        addr_peer_expert_output_ptrs: fx.Int64,
        addr_peer_route_weight_ptrs: fx.Int64,
        addr_output: fx.Int64,
        addr_gathered_route_weights: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        combine_kernel(
            addr_dst,
            addr_peer_expert_output_ptrs,
            addr_peer_route_weight_ptrs,
            addr_output,
            addr_gathered_route_weights,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_combine_fast_jit"]
