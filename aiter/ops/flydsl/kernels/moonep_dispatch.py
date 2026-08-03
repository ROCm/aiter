# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Preplanned direct-P2P dispatch for the MoonEP gfx950 prototype.

The host reference planner has already assigned every routed entry an exact
destination row.  This kernel therefore does no allocation and no atomics: it
decodes ``dst[S, K]`` and writes directly through MORI peer-pointer tables.

``dst >= 0`` is the primary route for a token on a destination rank.  It writes
both the BF16 hidden row and the FP32 route weight.  ``dst < 0`` encodes
``-raw_dst - 1`` for another expert on the same destination rank; it writes the
route weight but deliberately skips the duplicate hidden payload.  A later
local duplicate-expansion epilogue fills those hidden rows without another
cross-GPU transfer.

This file contains only the data-plane kernel and launcher.  Symmetric-memory
allocation, peer-pointer construction, and inter-rank completion are owned by
the caller so this prototype cannot change the existing FlyDSL all2all path.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import T, arith, range_constexpr
from flydsl.expr.typing import Stream

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

_JIT_SCHEMA_VERSION = "v1-bf16-primary-direct-p2p"


def make_moonep_preplanned_dispatch_kernel(
    *,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    block_num: int,
    warp_num_per_block: int,
):
    """Build the BF16 preplanned dispatch kernel.

    One wave owns one flattened ``(token, k)`` route entry.  Lanes cooperate
    on the hidden-row copy while lane 0 publishes that entry's route weight.
    ``num_dispatch_rows`` is the planner's ``NvS`` stride used by the raw
    destination encoding ``dest_rank * NvS + local_row``.
    """

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_dispatch_rows <= 0:
        raise ValueError("num_dispatch_rows must be positive")
    if block_num <= 0 or warp_num_per_block <= 0:
        raise ValueError("launch geometry must be positive")

    # BF16 only for the first gfx950 milestone.  A vec4 i32 operation moves
    # 16 bytes, and hidden_dim % 8 keeps every row and tail vector aligned.
    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    kernel_name = (
        f"moonep_preplanned_dispatch_bf16_h{hidden_dim}_k{top_k}"
        f"_nvs{num_dispatch_rows}_b{block_num}_w{warp_num_per_block}"
    )

    @flyc.kernel(
        name=kernel_name,
        known_block_size=[warp_num_per_block * 64, 1, 1],
    )
    def dispatch_kernel(
        addr_hidden: fx.Int64,  # BF16 [S, H]
        addr_route_weights: fx.Int64,  # FP32 [S, K]
        addr_dst: fx.Int64,  # INT32 [S, K]
        addr_peer_hidden_ptrs: fx.Int64,  # INT64 [world_size]
        addr_peer_weight_ptrs: fx.Int64,  # INT64 [world_size]
        addr_peer_duplicate_src_ptrs: fx.Int64,  # INT64 [world_size]
        num_tokens: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        global_warps = block_num * warp_num_per_block
        route_count = num_tokens * top_k

        dst_rsrc = create_buffer_resource_from_addr(addr_dst)
        weight_in_rsrc = create_buffer_resource_from_addr(addr_route_weights)
        peer_hidden_rsrc = create_buffer_resource_from_addr(addr_peer_hidden_ptrs)
        peer_weight_rsrc = create_buffer_resource_from_addr(addr_peer_weight_ptrs)
        peer_duplicate_src_rsrc = create_buffer_resource_from_addr(
            addr_peer_duplicate_src_ptrs
        )

        for route_idx in range(global_warp, route_count, global_warps):
            encoded_dst = buffer_load(
                dst_rsrc, route_idx, vec_width=1, dtype=T.i32()
            )
            is_primary = encoded_dst >= 0
            raw_dst = is_primary.select(encoded_dst, -encoded_dst - 1)
            dest_rank = raw_dst // num_dispatch_rows
            dest_row = raw_dst % num_dispatch_rows

            # For a negative entry, recover the earlier primary row belonging
            # to this source token and destination rank.  Publishing this tiny
            # row map lets the destination duplicate hidden locally after the
            # cross-rank barrier instead of sending the payload twice.
            src_token = route_idx // top_k
            k_slot = route_idx % top_k
            primary_row = fx.Int32(-1)
            for prior_k in range_constexpr(top_k):
                prior_encoded = buffer_load(
                    dst_rsrc,
                    src_token * top_k + prior_k,
                    vec_width=1,
                    dtype=T.i32(),
                )
                prior_is_primary = prior_encoded >= 0
                prior_raw = prior_is_primary.select(
                    prior_encoded, -prior_encoded - 1
                )
                prior_matches = (k_slot > prior_k) & prior_is_primary & (
                    prior_raw // num_dispatch_rows == dest_rank
                )
                primary_row = prior_matches.select(
                    prior_raw % num_dispatch_rows, primary_row
                )

            # Every route owns a distinct destination row, including a
            # negative-encoded duplicate route.  Keep its scalar weight there.
            if lane == 0:
                route_weight = buffer_load(
                    weight_in_rsrc, route_idx, vec_width=1, dtype=T.f32()
                )
                remote_weight_addr = buffer_load(
                    peer_weight_rsrc, dest_rank, vec_width=1, dtype=T.i64()
                )
                remote_weight_rsrc = create_buffer_resource_from_addr(
                    remote_weight_addr
                )
                buffer_store(
                    arith.bitcast(T.i32(), route_weight),
                    remote_weight_rsrc,
                    dest_row,
                )
                remote_duplicate_src_addr = buffer_load(
                    peer_duplicate_src_rsrc,
                    dest_rank,
                    vec_width=1,
                    dtype=T.i64(),
                )
                remote_duplicate_src_rsrc = create_buffer_resource_from_addr(
                    remote_duplicate_src_addr
                )
                duplicate_src = is_primary.select(-1, primary_row)
                buffer_store(
                    duplicate_src, remote_duplicate_src_rsrc, dest_row
                )

            # Lanes copy 16 B apiece and stride by one wave (64 * 16 B).
            # For duplicates copy_end == lane_i32_off, making the loop empty.
            remote_hidden_addr = (
                buffer_load(
                    peer_hidden_rsrc, dest_rank, vec_width=1, dtype=T.i64()
                )
                + fx.Int64(dest_row) * hidden_bytes
            )
            local_hidden_addr = addr_hidden + fx.Int64(src_token) * hidden_bytes
            local_hidden_rsrc = create_buffer_resource_from_addr(local_hidden_addr)
            remote_hidden_rsrc = create_buffer_resource_from_addr(remote_hidden_addr)
            lane_i32_off = lane * 4
            copy_end = is_primary.select(hidden_i32, lane_i32_off)
            for i32_off in range(lane_i32_off, copy_end, 64 * 4):
                hidden_vec = buffer_load(
                    local_hidden_rsrc, i32_off, vec_width=4, dtype=T.i32()
                )
                buffer_store(hidden_vec, remote_hidden_rsrc, i32_off)

    return dispatch_kernel


def make_moonep_preplanned_dispatch_jit(
    *,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    block_num: int = 128,
    warp_num_per_block: int = 4,
):
    """Return a FlyDSL JIT launcher for preplanned direct-P2P dispatch."""

    kernel = make_moonep_preplanned_dispatch_kernel(
        hidden_dim=hidden_dim,
        top_k=top_k,
        num_dispatch_rows=num_dispatch_rows,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )

    # Make every IR-affecting value visible to FlyDSL's closure-based cache key.
    _key_hidden_dim = hidden_dim
    _key_top_k = top_k
    _key_num_dispatch_rows = num_dispatch_rows
    _key_block_num = block_num
    _key_warp_num_per_block = warp_num_per_block
    _key_schema_version = _JIT_SCHEMA_VERSION

    @flyc.jit
    def launch(
        addr_hidden: fx.Int64,
        addr_route_weights: fx.Int64,
        addr_dst: fx.Int64,
        addr_peer_hidden_ptrs: fx.Int64,
        addr_peer_weight_ptrs: fx.Int64,
        addr_peer_duplicate_src_ptrs: fx.Int64,
        num_tokens: fx.Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = (
            _key_hidden_dim,
            _key_top_k,
            _key_num_dispatch_rows,
            _key_block_num,
            _key_warp_num_per_block,
            _key_schema_version,
        )
        kernel(
            addr_hidden,
            addr_route_weights,
            addr_dst,
            addr_peer_hidden_ptrs,
            addr_peer_weight_ptrs,
            addr_peer_duplicate_src_ptrs,
            num_tokens,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return launch


__all__ = [
    "make_moonep_preplanned_dispatch_jit",
    "make_moonep_preplanned_dispatch_kernel",
]
