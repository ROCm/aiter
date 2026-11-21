# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MoonEP BF16 direct peer gather and top-k weighted combine."""

from __future__ import annotations

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


def _unpack_bf16_pair(raw_dw):
    raw = fx.Uint32(raw_dw)
    lo = ((raw & 0xFFFF) << 16).bitcast(fx.Float32)
    hi = (((raw >> 16) & 0xFFFF) << 16).bitcast(fx.Float32)
    return lo, hi


def _pack_bf16_pair(lo_f32, hi_f32):
    lo = fx.Uint32(lo_f32.to(fx.BFloat16).bitcast(fx.Uint16))
    hi = fx.Uint32(hi_f32.to(fx.BFloat16).bitcast(fx.Uint16))
    return lo | (hi << 16)


def make_moonep_combine_jit(
    *,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    block_threads: int = 256,
    gather_route_weights: bool = True,
    apply_route_weights: bool = False,
):
    """Build source-rank combine: peer gather, optional weighting, top-k sum."""

    if num_tokens <= 0 or hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("token count and 16-byte-aligned hidden_dim are required")
    if top_k <= 0 or num_dispatch_rows <= 0 or block_threads <= 0:
        raise ValueError("top_k, row count, and block size must be positive")

    vec_dwords = 4
    row_dwords = hidden_dim // 2
    name = (
        f"moonep_combine_bf16_s{num_tokens}_h{hidden_dim}_k{top_k}"
        f"_nvs{num_dispatch_rows}_t{block_threads}_gw{int(gather_route_weights)}"
        f"_aw{int(apply_route_weights)}"
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
            encoded = buffer_load(
                dst_rsrc, route_idx, vec_width=1, dtype=T.i32
            )
            raw = (encoded >= 0).select(encoded, -encoded - 1)
            peer = raw // num_dispatch_rows
            row = raw % num_dispatch_rows
            peer_weight_base = buffer_load(
                peer_weight_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
            )
            peer_weight_rsrc = create_buffer_resource_from_addr(
                peer_weight_base
            )
            route_weight = buffer_load(
                peer_weight_rsrc, row, vec_width=1, dtype=T.f32
            )
            if gather_route_weights:
                buffer_store(
                    route_weight,
                    gathered_weights_rsrc,
                    route_idx,
                )

        for dw_base in range(tid * vec_dwords, row_dwords, block_threads * vec_dwords):
            acc = [fx.Float32(0.0) for _ in range(2 * vec_dwords)]
            for k_idx in range_constexpr(top_k):
                route_idx = token * top_k + k_idx
                encoded = buffer_load(
                    dst_rsrc, route_idx, vec_width=1, dtype=T.i32
                )
                raw = (encoded >= 0).select(encoded, -encoded - 1)
                peer = raw // num_dispatch_rows
                row = raw % num_dispatch_rows
                peer_base = buffer_load(
                    peer_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
                )
                row_addr = peer_base + fx.Int64(row) * hidden_dim * 2
                row_rsrc = create_buffer_resource_from_addr(row_addr)
                weight = fx.Float32(1.0)
                if apply_route_weights:
                    peer_weight_base = buffer_load(
                        peer_weight_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
                    )
                    peer_weight_rsrc = create_buffer_resource_from_addr(
                        peer_weight_base
                    )
                    weight = fx.Float32(
                        buffer_load(
                            peer_weight_rsrc, row, vec_width=1, dtype=T.f32
                        )
                    )
                raw_vec = buffer_load(
                    row_rsrc, dw_base, vec_width=vec_dwords, dtype=T.i32
                )
                for lane in range_constexpr(vec_dwords):
                    raw_dw = vector.extract(
                        raw_vec, static_position=[lane], dynamic_position=[]
                    )
                    lo, hi = _unpack_bf16_pair(raw_dw)
                    acc[2 * lane] = acc[2 * lane] + weight * lo
                    acc[2 * lane + 1] = acc[2 * lane + 1] + weight * hi

            packed = [
                _pack_bf16_pair(acc[2 * lane], acc[2 * lane + 1])
                for lane in range(vec_dwords)
            ]
            out_vec = vector.from_elements(T.vec(vec_dwords, T.i32), packed)
            output_dw = token * row_dwords + dw_base
            buffer_store(out_vec, output_rsrc, output_dw)

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


__all__ = ["make_moonep_combine_jit"]
