# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Push-based MoonEP combine: remote writes instead of remote reads.

Why this exists
---------------
``moonep_link_probe`` measured what this machine can actually do, with the
simplest possible streaming access (contiguous, coalesced, all links driven at
once, MI355X x8, 256 MiB per peer, cross-rank mean):

===============  =======  =======  =======  =======  =======  =======
in-flight depth        1        2        4        8       16       32
===============  =======  =======  =======  =======  =======  =======
remote read        235.8    231.6    229.8    225.2    224.6    228.3
remote write       447.3    447.2    447.9    448.6    447.9        -
===============  =======  =======  =======  =======  =======  =======

(GB/s, bn=2048 for reads / bn=256 for writes; the same kernel against local HBM
does 6044 GB/s, so the kernel shape is not the limit.)

Two things follow, and they decide the design:

1. **Remote read bandwidth is flat at ~235 GB/s from depth 1 to depth 32.**  Our
   pull-based combine already achieves 232.8 GB/s, i.e. 99% of the best point
   the probe ever reached.  A deeper software pipeline -- the AMD analogue of
   upstream's 16-stage TMA/mbarrier staging into shared memory -- cannot help,
   because in-flight depth is provably not what limits this path.  Upstream's
   650 GB/s is a property of B300/NVLink5, not of a better kernel.

2. **Remote writes run 1.90x faster than remote reads** (448 vs 235).  Reads are
   round trips that hold requester state; writes are posted.  So the way to
   make combine faster on this hardware is not to read better, it is to stop
   reading: have the rank that *holds* the expert output write it home, and let
   the owning rank reduce out of its own HBM.

Structure
---------
Three kernels, same wire bytes as the deduplicated pull path, but the link
traffic changes direction:

``publish_src_slots``  source rank -> peers, once per plan.  For every primary
                       entry ``(token, k)`` it writes ``rank * S*K + token*K+k``
                       into the destination's ``src_slot[local_offset]``.  That
                       is the reverse of ``plan.dst``, which the destination has
                       no other way to know.  S*K*4 = 262 KB per rank, ~0.06% of
                       the payload, and it does not repeat per combine.
                       Rows left at -1 (duplicates, padding) are never pushed.

``push_rows``          destination rank -> peers, per combine.  One wave per
                       local row: if ``src_slot[row] >= 0``, write the row of
                       expert output into ``staging[slot]`` on the owning rank.
                       Remote writes only.

``reduce_local``       owning rank, per combine.  Identical arithmetic to
                       ``moonep_combine``, except every payload row comes from
                       local ``staging`` instead of a peer, and duplicate
                       entries are skipped exactly as upstream does
                       (``MoonEP/moonep/combine.py:320``).  ~621 MB of local HBM
                       reads at ~6 TB/s is ~110 us.

Route weights are deliberately left on the original remote-scalar-gather path so
``gathered_route_weights`` stays bit-identical to the reference; it is 262 KB.

Ordering: ``prologue -> push_rows -> shmem barrier -> reduce_local``.  The
prologue (``moonep_combine_prologue``) still runs first so only primaries are
pushed; without it this would move 822 MB instead of 544 MB.

Cost
----
``staging`` is a symmetric ``[S*K, H]`` bf16 buffer -- 940 MB at S=8192 H=7168
K=8.  Only ``primaries`` of those rows are ever written (~43.3k of 65.5k here),
so a prefix-sum over ``is_primary`` would compact it to ~621 MB; that is a plan
change and is not done here.

Numerics
--------
Same as the deduplicated pull path: folding duplicates in the prologue rounds to
bf16 once more than summing all K in fp32 inside combine, so compare against the
reference with a tolerance, not ``torch.equal``.
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
from aiter.ops.flydsl.kernels.moonep_combine import (
    _pack_bf16_pair,
    _unpack_bf16_pair,
)

WAVE_SIZE = 64
VEC_I32 = 4
LANE_STRIDE_I32 = WAVE_SIZE * VEC_I32


def make_moonep_publish_src_slots_jit(
    *,
    num_tokens: int,
    top_k: int,
    num_dispatch_rows: int,
    rank: int,
    block_threads: int = 256,
    block_num: int = 256,
):
    """Build the reverse map publisher (source rank -> peers, once per plan).

    ``src_slot`` must be filled with -1 before the first call and after any plan
    change; only primary entries write, so stale non-negative values would push
    rows that no longer exist.
    """

    if num_tokens <= 0 or top_k <= 0 or num_dispatch_rows <= 0:
        raise ValueError("shape parameters must be positive")
    if block_threads % WAVE_SIZE != 0 or block_num <= 0:
        raise ValueError("launch geometry must be positive and wave-aligned")

    n_entries = num_tokens * top_k
    global_threads = block_num * block_threads
    rank_tag_base = rank * n_entries
    suffix = f"s{num_tokens}_k{top_k}_nvs{num_dispatch_rows}_r{rank}_b{block_num}"

    @flyc.kernel(
        name=f"moonep_publish_src_slots_{suffix}",
        known_block_size=[block_threads, 1, 1],
    )
    def publish(
        addr_dst: fx.Int64,  # INT32 [S, K]
        addr_peer_src_slot_ptrs: fx.Int64,  # INT64 [world_size]
    ):
        tid = fx.Int32(fx.thread_idx.x)
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(block_threads) + tid
        dst_rsrc = create_buffer_resource_from_addr(addr_dst)
        ptrs_rsrc = create_buffer_resource_from_addr(addr_peer_src_slot_ptrs)

        for entry in range(gid, n_entries, global_threads):
            encoded = buffer_load(dst_rsrc, entry, vec_width=1, dtype=T.i32)
            # Primaries only.  A duplicate's destination row is folded into its
            # primary by the prologue and must never be pushed.
            if encoded >= fx.Int32(0):
                dest = encoded // fx.Int32(num_dispatch_rows)
                loff = encoded % fx.Int32(num_dispatch_rows)
                peer_base = buffer_load(ptrs_rsrc, dest, vec_width=1, dtype=T.i64)
                slot_rsrc = create_buffer_resource_from_addr(peer_base)
                buffer_store(
                    fx.Int32(rank_tag_base) + entry, slot_rsrc, loff
                )

    @flyc.jit
    def launch(
        addr_dst: fx.Int64,
        addr_peer_src_slot_ptrs: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        publish(addr_dst, addr_peer_src_slot_ptrs).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def make_moonep_push_rows_jit(
    *,
    hidden_dim: int,
    num_dispatch_rows: int,
    num_tokens: int,
    top_k: int,
    block_num: int = 256,
    warp_num_per_block: int = 4,
):
    """Build the destination-side row pusher (the remote-write half of combine).

    One wave owns one local row, so the ``src_slot >= 0`` predicate is
    wave-uniform and skipped rows cost a scalar branch.
    """

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if num_dispatch_rows <= 0 or num_tokens <= 0 or top_k <= 0:
        raise ValueError("shape parameters must be positive")
    if block_num <= 0 or warp_num_per_block <= 0:
        raise ValueError("launch geometry must be positive")

    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    block_threads = warp_num_per_block * WAVE_SIZE
    global_warps = block_num * warp_num_per_block
    n_entries = num_tokens * top_k
    full_chunks = hidden_i32 // LANE_STRIDE_I32
    tail_vecs = (hidden_i32 - full_chunks * LANE_STRIDE_I32) // VEC_I32
    tail_off = full_chunks * LANE_STRIDE_I32
    suffix = (
        f"h{hidden_dim}_nvs{num_dispatch_rows}_s{num_tokens}_k{top_k}"
        f"_b{block_num}_w{warp_num_per_block}"
    )

    @flyc.kernel(
        name=f"moonep_push_rows_{suffix}", known_block_size=[block_threads, 1, 1]
    )
    def push_rows(
        addr_expert_output: fx.Int64,  # BF16 [NvS, H], local
        addr_src_slot: fx.Int64,  # INT32 [NvS], -1 where nothing to push
        addr_peer_staging_ptrs: fx.Int64,  # INT64 [world_size]
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        warp = tid >> fx.Int32(6)
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        slot_rsrc = create_buffer_resource_from_addr(addr_src_slot)
        ptrs_rsrc = create_buffer_resource_from_addr(addr_peer_staging_ptrs)
        lane_off = lane * fx.Int32(VEC_I32)

        for row in range(global_warp, num_dispatch_rows, global_warps):
            tag = buffer_load(slot_rsrc, row, vec_width=1, dtype=T.i32)
            if tag >= fx.Int32(0):
                src_rank = tag // fx.Int32(n_entries)
                slot = tag % fx.Int32(n_entries)
                peer_base = buffer_load(
                    ptrs_rsrc, src_rank, vec_width=1, dtype=T.i64
                )
                src_rsrc = create_buffer_resource_from_addr(
                    addr_expert_output + fx.Int64(row) * hidden_bytes
                )
                dst_rsrc = create_buffer_resource_from_addr(
                    peer_base + fx.Int64(slot) * hidden_bytes
                )
                # Load the whole row before storing any of it: the local reads
                # run at HBM speed and the remote writes are posted, so there is
                # no reason to interleave them one vec4 at a time the way the
                # original epilogue did.
                values = []
                for c in range_constexpr(full_chunks):
                    values.append(
                        buffer_load(
                            src_rsrc,
                            lane_off + fx.Int32(c * LANE_STRIDE_I32),
                            vec_width=VEC_I32,
                            dtype=T.i32,
                        )
                    )
                for c in range_constexpr(full_chunks):
                    buffer_store(
                        values[c],
                        dst_rsrc,
                        lane_off + fx.Int32(c * LANE_STRIDE_I32),
                    )
                # Folds to a never-taken branch when tail_vecs == 0.
                if lane < fx.Int32(tail_vecs):
                    off = lane_off + fx.Int32(tail_off)
                    buffer_store(
                        buffer_load(
                            src_rsrc, off, vec_width=VEC_I32, dtype=T.i32
                        ),
                        dst_rsrc,
                        off,
                    )

    @flyc.jit
    def launch(
        addr_expert_output: fx.Int64,
        addr_src_slot: fx.Int64,
        addr_peer_staging_ptrs: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        push_rows(
            addr_expert_output, addr_src_slot, addr_peer_staging_ptrs
        ).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def make_moonep_reduce_local_jit(
    *,
    num_tokens: int,
    hidden_dim: int,
    top_k: int,
    num_dispatch_rows: int,
    block_threads: int = 256,
    gather_route_weights: bool = True,
    apply_route_weights: bool = False,
):
    """Build the owning-rank reduction over locally staged rows.

    Same arithmetic and same ABI shape as ``moonep_combine``, but the payload
    comes from ``staging[token*K + k]`` in local HBM.  Route weights keep the
    original remote scalar gather so ``gathered_route_weights`` is unchanged.
    """

    if num_tokens <= 0 or hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("token count and 16-byte-aligned hidden_dim are required")
    if top_k <= 0 or num_dispatch_rows <= 0 or block_threads <= 0:
        raise ValueError("top_k, row count, and block size must be positive")

    row_dwords = hidden_dim // 2
    row_bytes = hidden_dim * 2
    name = (
        f"moonep_reduce_local_bf16_s{num_tokens}_h{hidden_dim}_k{top_k}"
        f"_nvs{num_dispatch_rows}_t{block_threads}"
        f"_gw{int(gather_route_weights)}_aw{int(apply_route_weights)}"
    )

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def reduce_kernel(
        addr_dst: fx.Int64,  # INT32 [S,K]
        addr_staging: fx.Int64,  # BF16 [S*K, H], local
        addr_peer_route_weight_ptrs: fx.Int64,  # INT64 [world_size]
        addr_output: fx.Int64,  # BF16 [S,H]
        addr_gathered_route_weights: fx.Int64,  # FP32 [S,K]
    ):
        token = fx.block_idx.x
        tid = fx.thread_idx.x
        dst_rsrc = create_buffer_resource_from_addr(addr_dst)
        peer_weight_ptrs_rsrc = create_buffer_resource_from_addr(
            addr_peer_route_weight_ptrs
        )
        output_rsrc = create_buffer_resource_from_addr(addr_output)
        gathered_weights_rsrc = create_buffer_resource_from_addr(
            addr_gathered_route_weights
        )

        # Unchanged from the reference: a remote scalar per (token, k), 262 KB
        # in total, kept on the original path so the gathered weights stay
        # bit-identical.
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

        row_rsrcs = []
        weights = []
        take_row = []
        for k_idx in range_constexpr(top_k):
            route_idx = token * top_k + fx.Int32(k_idx)
            encoded = buffer_load(dst_rsrc, route_idx, vec_width=1, dtype=T.i32)
            # Staging is indexed by the source-side entry, so the row address is
            # just the entry index -- no peer table, no decode.
            row_rsrcs.append(
                create_buffer_resource_from_addr(
                    addr_staging + fx.Int64(route_idx) * row_bytes
                )
            )
            weight = fx.Float32(1.0)
            if apply_route_weights:
                raw = (encoded >= 0).select(encoded, -encoded - 1)
                peer = raw // num_dispatch_rows
                prow = raw % num_dispatch_rows
                peer_weight_rsrc = create_buffer_resource_from_addr(
                    buffer_load(
                        peer_weight_ptrs_rsrc, peer, vec_width=1, dtype=T.i64
                    )
                )
                weight = fx.Float32(
                    buffer_load(peer_weight_rsrc, prow, vec_width=1, dtype=T.f32)
                )
            weights.append(weight)
            # Duplicates were folded into their primary by the prologue, so the
            # only rows pushed -- and the only ones staged -- are primaries.
            take_row.append(encoded >= 0)

        zero_vec = fly_vector.from_elements(
            T.vec(VEC_I32, T.i32), [fx.Int32(0)] * VEC_I32
        )

        for dw_base in range(tid * VEC_I32, row_dwords, block_threads * VEC_I32):
            raw_vecs = []
            for k_idx in range_constexpr(top_k):
                # token == block_idx makes this block-uniform, so it is a scalar
                # branch and the load is genuinely skipped.
                contrib = zero_vec
                if take_row[k_idx]:
                    contrib = buffer_load(
                        row_rsrcs[k_idx],
                        dw_base,
                        vec_width=VEC_I32,
                        dtype=T.i32,
                    )
                raw_vecs.append(contrib)

            acc = [fx.Float32(0.0) for _ in range(2 * VEC_I32)]
            for k_idx in range_constexpr(top_k):
                for pos in range_constexpr(VEC_I32):
                    raw_dw = fly_vector.extract(
                        raw_vecs[k_idx], static_position=[pos], dynamic_position=[]
                    )
                    lo, hi = _unpack_bf16_pair(raw_dw)
                    acc[2 * pos] = acc[2 * pos] + weights[k_idx] * lo
                    acc[2 * pos + 1] = acc[2 * pos + 1] + weights[k_idx] * hi

            packed = [
                _pack_bf16_pair(acc[2 * pos], acc[2 * pos + 1])
                for pos in range(VEC_I32)
            ]
            buffer_store(
                fly_vector.from_elements(T.vec(VEC_I32, T.i32), packed),
                output_rsrc,
                token * row_dwords + dw_base,
            )

    @flyc.jit
    def launch(
        addr_dst: fx.Int64,
        addr_staging: fx.Int64,
        addr_peer_route_weight_ptrs: fx.Int64,
        addr_output: fx.Int64,
        addr_gathered_route_weights: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        reduce_kernel(
            addr_dst,
            addr_staging,
            addr_peer_route_weight_ptrs,
            addr_output,
            addr_gathered_route_weights,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = [
    "make_moonep_publish_src_slots_jit",
    "make_moonep_push_rows_jit",
    "make_moonep_reduce_local_jit",
]
