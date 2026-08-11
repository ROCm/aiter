# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Single-launch MoonEP token dispatch plus remote weight prefetch.

The grid is statically partitioned: the first CTA range performs preplanned
direct-P2P token scatter, while the second range loads selected remote expert
weights into local slots.  Both work domains therefore overlap inside one
kernel launch on one stream; no multi-stream API or ordering contract exists.
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


def make_moonep_dispatch_prefetch_jit(
    *,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    dispatch_block_num: int,
    warp_num_per_block: int,
    experts_per_rank: int,
    prefetch_slots: int,
    weight_numel: int,
    prefetch_block_num: int,
):
    """Build one launcher containing disjoint dispatch/prefetch CTA ranges."""

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if top_k <= 0 or num_dispatch_rows <= 0:
        raise ValueError("top_k and num_dispatch_rows must be positive")
    if dispatch_block_num <= 0 or prefetch_block_num <= 0:
        raise ValueError("CTA partitions must be positive")
    if warp_num_per_block <= 0:
        raise ValueError("warp_num_per_block must be positive")
    if experts_per_rank <= 0 or prefetch_slots <= 0:
        raise ValueError("expert and slot counts must be positive")
    if weight_numel <= 0 or weight_numel % 8 != 0:
        raise ValueError("BF16 expert weight size must be 16-byte aligned")

    block_threads = warp_num_per_block * 64
    total_block_num = dispatch_block_num + prefetch_block_num
    dispatch_global_warps = dispatch_block_num * warp_num_per_block
    prefetch_global_threads = prefetch_block_num * block_threads
    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    weight_bytes = weight_numel * 2
    weight_i32 = weight_bytes // 4
    total_weight_i32 = prefetch_slots * weight_i32
    name = (
        f"moonep_dispatch_prefetch_h{hidden_dim}_k{top_k}"
        f"_nvs{num_dispatch_rows}_db{dispatch_block_num}"
        f"_pb{prefetch_block_num}_w{warp_num_per_block}"
        f"_epr{experts_per_rank}_b{prefetch_slots}_wn{weight_numel}"
    )

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def combined_kernel(
        addr_hidden: fx.Int64,
        addr_route_weights: fx.Int64,
        addr_dst: fx.Int64,
        addr_peer_hidden_ptrs: fx.Int64,
        addr_peer_weight_ptrs: fx.Int64,
        addr_peer_duplicate_src_ptrs: fx.Int64,
        num_tokens: fx.Int32,
        addr_experts_to_copy: fx.Int64,
        addr_peer_home_weight_ptrs: fx.Int64,
        addr_prefetched_weights: fx.Int64,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        if bid < dispatch_block_num:
            lane = tid & 63
            warp = tid >> 6
            global_warp = bid * warp_num_per_block + warp
            route_count = num_tokens * top_k
            dst_rsrc = create_buffer_resource_from_addr(addr_dst)
            weight_in_rsrc = create_buffer_resource_from_addr(addr_route_weights)
            peer_hidden_rsrc = create_buffer_resource_from_addr(
                addr_peer_hidden_ptrs
            )
            peer_weight_rsrc = create_buffer_resource_from_addr(
                addr_peer_weight_ptrs
            )
            peer_duplicate_src_rsrc = create_buffer_resource_from_addr(
                addr_peer_duplicate_src_ptrs
            )

            for route_idx in range(
                global_warp, route_count, dispatch_global_warps
            ):
                encoded_dst = buffer_load(
                    dst_rsrc, route_idx, vec_width=1, dtype=T.i32
                )
                is_primary = encoded_dst >= 0
                raw_dst = is_primary.select(encoded_dst, -encoded_dst - 1)
                dest_rank = raw_dst // num_dispatch_rows
                dest_row = raw_dst % num_dispatch_rows
                src_token = route_idx // top_k
                k_slot = route_idx % top_k

                primary_row = fx.Int32(-1)
                for prior_k in range_constexpr(top_k):
                    prior_encoded = buffer_load(
                        dst_rsrc,
                        src_token * top_k + prior_k,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    prior_is_primary = prior_encoded >= 0
                    prior_raw = prior_is_primary.select(
                        prior_encoded, -prior_encoded - 1
                    )
                    prior_matches = (
                        (k_slot > prior_k)
                        & prior_is_primary
                        & (prior_raw // num_dispatch_rows == dest_rank)
                    )
                    primary_row = prior_matches.select(
                        prior_raw % num_dispatch_rows, primary_row
                    )

                if lane == 0:
                    route_weight = buffer_load(
                        weight_in_rsrc,
                        route_idx,
                        vec_width=1,
                        dtype=T.f32,
                    )
                    remote_weight_addr = buffer_load(
                        peer_weight_rsrc,
                        dest_rank,
                        vec_width=1,
                        dtype=T.i64,
                    )
                    remote_weight_rsrc = create_buffer_resource_from_addr(
                        remote_weight_addr
                    )
                    buffer_store(
                        arith.bitcast(T.i32, route_weight),
                        remote_weight_rsrc,
                        dest_row,
                    )
                    remote_duplicate_src_addr = buffer_load(
                        peer_duplicate_src_rsrc,
                        dest_rank,
                        vec_width=1,
                        dtype=T.i64,
                    )
                    remote_duplicate_src_rsrc = create_buffer_resource_from_addr(
                        remote_duplicate_src_addr
                    )
                    duplicate_src = is_primary.select(-1, primary_row)
                    buffer_store(
                        duplicate_src, remote_duplicate_src_rsrc, dest_row
                    )

                remote_hidden_addr = (
                    buffer_load(
                        peer_hidden_rsrc,
                        dest_rank,
                        vec_width=1,
                        dtype=T.i64,
                    )
                    + fx.Int64(dest_row) * hidden_bytes
                )
                local_hidden_addr = (
                    addr_hidden + fx.Int64(src_token) * hidden_bytes
                )
                local_hidden_rsrc = create_buffer_resource_from_addr(
                    local_hidden_addr
                )
                remote_hidden_rsrc = create_buffer_resource_from_addr(
                    remote_hidden_addr
                )
                lane_i32_off = lane * 4
                copy_end = is_primary.select(hidden_i32, lane_i32_off)
                for i32_off in range(lane_i32_off, copy_end, 64 * 4):
                    hidden_vec = buffer_load(
                        local_hidden_rsrc,
                        i32_off,
                        vec_width=4,
                        dtype=T.i32,
                    )
                    buffer_store(hidden_vec, remote_hidden_rsrc, i32_off)
        else:
            prefetch_bid = bid - dispatch_block_num
            global_thread = prefetch_bid * block_threads + tid
            experts_rsrc = create_buffer_resource_from_addr(addr_experts_to_copy)
            peer_home_rsrc = create_buffer_resource_from_addr(
                addr_peer_home_weight_ptrs
            )
            prefetched_rsrc = create_buffer_resource_from_addr(
                addr_prefetched_weights
            )
            for i32_off in range(
                global_thread * 4,
                total_weight_i32,
                prefetch_global_threads * 4,
            ):
                slot = i32_off // weight_i32
                expert = buffer_load(
                    experts_rsrc, slot, vec_width=1, dtype=T.i32
                )
                if expert >= 0:
                    owner = expert // experts_per_rank
                    local_expert = expert % experts_per_rank
                    owner_base = buffer_load(
                        peer_home_rsrc, owner, vec_width=1, dtype=T.i64
                    )
                    src_addr = (
                        owner_base + fx.Int64(local_expert) * weight_bytes
                    )
                    src_rsrc = create_buffer_resource_from_addr(src_addr)
                    expert_i32_off = i32_off - slot * weight_i32
                    value = buffer_load(
                        src_rsrc,
                        expert_i32_off,
                        vec_width=4,
                        dtype=T.i32,
                    )
                    buffer_store(value, prefetched_rsrc, i32_off)

    @flyc.jit
    def launch(
        addr_hidden: fx.Int64,
        addr_route_weights: fx.Int64,
        addr_dst: fx.Int64,
        addr_peer_hidden_ptrs: fx.Int64,
        addr_peer_weight_ptrs: fx.Int64,
        addr_peer_duplicate_src_ptrs: fx.Int64,
        num_tokens: fx.Int32,
        addr_experts_to_copy: fx.Int64,
        addr_peer_home_weight_ptrs: fx.Int64,
        addr_prefetched_weights: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        combined_kernel(
            addr_hidden,
            addr_route_weights,
            addr_dst,
            addr_peer_hidden_ptrs,
            addr_peer_weight_ptrs,
            addr_peer_duplicate_src_ptrs,
            num_tokens,
            addr_experts_to_copy,
            addr_peer_home_weight_ptrs,
            addr_prefetched_weights,
        ).launch(
            grid=(total_block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_dispatch_prefetch_jit"]
