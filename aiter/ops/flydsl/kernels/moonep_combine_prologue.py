# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Local duplicate reduction before MoonEP combine.

Upstream MoonEP runs ``combine_prologue`` on the destination rank to fold every
duplicate row into its primary *before* combine, so combine only pulls primary
rows across the link (``MoonEP/moonep/combine.py:320``:
``load_token = dst_val >= Int32(0)``).  We had no such step, so our combine
pulled all K rows per token: 8 rows instead of 5.29 at uniform routing, i.e.
1.51x the link traffic on the most expensive stage.

Measured before this change (MI355X, S=8192 H=7168 K=8 E=384 R=8, cross-rank
mean, upstream ``bench_comm`` methodology):

=========================  =========  =============  ===========
op                                us   remote bytes    read GB/s
=========================  =========  =============  ===========
upstream prologue+combine      973.7        543.6 MB          650
ours combine                  3541.8        822.0 MB          232
=========================  =========  =============  ===========

so the 4.23x splits into 1.51x of avoidable traffic (this file) and 2.80x of
read bandwidth (upstream hides it with a 16-stage TMA pipeline into shared
memory; not addressed here).

Grouping without the upstream group table
-----------------------------------------
The hazard is that several duplicates share one primary, so a naive
"add my row into my primary" races.  Upstream avoids it with a
``(primary_loff, dup_start, dup_n)`` table built on the source rank inside
dispatch.  The destination can derive an equivalent grouping locally from the
``duplicate_src`` map it already receives, and because a token contributes at
most ``K - 1`` duplicates to one rank, a fixed stride removes the need for any
prefix sum:

    pass 1  slot = atomicAdd(dup_count[primary], 1)          # int32, local
            dup_list[primary * (K - 1) + slot] = my_row
    pass 2  one wavefront per primary: read the primary and its listed
            duplicates, sum in fp32, write the primary once

No remote atomics, no generation counter, and dispatch is untouched.

Numerics
--------
Folding duplicates here rounds to bf16 once more than summing all K rows in
fp32 inside combine, so results are close but **not** bit-identical to the
reference path; compare with a tolerance, not ``torch.equal``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import T, range_constexpr
from flydsl.expr.typing import Stream

from aiter.ops.flydsl.kernels import vector as fly_vector
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
    create_llvm_ptr,
)
from aiter.ops.flydsl.kernels.moonep_combine import (
    _pack_bf16_pair,
    _unpack_bf16_pair,
)

WAVE_SIZE = 64
VEC_I32 = 4
LANE_STRIDE_I32 = WAVE_SIZE * VEC_I32


def _unwrap(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _global_atomic_add_i32(base_i64, idx, val):
    """Device-scope ``atomicrmw add`` on an i32 global slot; returns the old value."""
    ptr = create_llvm_ptr(base_i64 + fx.Int64(idx) * 4, address_space=1)
    raw_ptr = ptr._value if hasattr(ptr, "_value") else ptr
    return llvm.AtomicRMWOp(
        llvm.AtomicBinOp.add,
        raw_ptr,
        _unwrap(val),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=4,
    ).result


def make_moonep_combine_prologue_jit(
    *,
    hidden_dim: int,
    num_dispatch_rows: int,
    top_k: int,
    block_num: int = 1024,
    warp_num_per_block: int = 4,
):
    """Build the two-pass local duplicate reduction launcher."""

    if hidden_dim <= 0 or hidden_dim % 8 != 0:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    if num_dispatch_rows <= 0 or top_k <= 1:
        raise ValueError("row count must be positive and top_k must exceed 1")
    if block_num <= 0 or warp_num_per_block <= 0:
        raise ValueError("launch geometry must be positive")

    hidden_bytes = hidden_dim * 2
    hidden_i32 = hidden_bytes // 4
    block_threads = warp_num_per_block * WAVE_SIZE
    global_threads = block_num * block_threads
    global_warps = block_num * warp_num_per_block
    dup_stride = top_k - 1
    full_chunks = hidden_i32 // LANE_STRIDE_I32
    tail_vecs = (hidden_i32 - full_chunks * LANE_STRIDE_I32) // VEC_I32
    tail_off = full_chunks * LANE_STRIDE_I32
    suffix = f"h{hidden_dim}_nvs{num_dispatch_rows}_k{top_k}_b{block_num}"

    @flyc.kernel(
        name=f"moonep_dup_index_{suffix}", known_block_size=[block_threads, 1, 1]
    )
    def build_dup_lists(
        addr_duplicate_src: fx.Int64,  # INT32 [NvS], -1 for primary/padding
        addr_dup_count: fx.Int64,  # INT32 [NvS], pre-zeroed
        addr_dup_list: fx.Int64,  # INT32 [NvS, K-1]
    ):
        tid = fx.Int32(fx.thread_idx.x)
        gid = fx.Int32(fx.block_idx.x) * fx.Int32(block_threads) + tid
        src_rsrc = create_buffer_resource_from_addr(addr_duplicate_src)
        list_rsrc = create_buffer_resource_from_addr(addr_dup_list)

        for row in range(gid, num_dispatch_rows, global_threads):
            primary = buffer_load(src_rsrc, row, vec_width=1, dtype=T.i32)
            if primary >= fx.Int32(0):
                slot = fx.Int32(
                    _global_atomic_add_i32(addr_dup_count, primary, fx.Int32(1))
                )
                buffer_store(
                    row, list_rsrc, primary * fx.Int32(dup_stride) + slot
                )

    @flyc.kernel(
        name=f"moonep_dup_reduce_{suffix}", known_block_size=[block_threads, 1, 1]
    )
    def reduce_duplicates(
        addr_hidden: fx.Int64,  # BF16 [NvS, H]
        addr_dup_count: fx.Int64,  # INT32 [NvS]
        addr_dup_list: fx.Int64,  # INT32 [NvS, K-1]
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid & fx.Int32(WAVE_SIZE - 1)
        warp = tid >> fx.Int32(6)
        global_warp = fx.block_idx.x * warp_num_per_block + warp
        count_rsrc = create_buffer_resource_from_addr(addr_dup_count)
        list_rsrc = create_buffer_resource_from_addr(addr_dup_list)
        lane_off = lane * fx.Int32(VEC_I32)

        for primary in range(global_warp, num_dispatch_rows, global_warps):
            n_dups = buffer_load(count_rsrc, primary, vec_width=1, dtype=T.i32)
            # Wave-uniform: one wave owns one primary row.
            if n_dups > fx.Int32(0):
                # Self-clearing: pass 2 is the only reader, so resetting the
                # counter here leaves the array zeroed for the next call and
                # removes the memset that pass 1 would otherwise need.
                buffer_store(fx.Int32(0), count_rsrc, primary)
                prim_rsrc = create_buffer_resource_from_addr(
                    addr_hidden + fx.Int64(primary) * hidden_bytes
                )
                # Accumulate the whole row in fp32 registers.  full_chunks is 14
                # at H=7168, so this is 28 f32 per lane.
                acc_lo = []
                acc_hi = []
                for c in range_constexpr(full_chunks):
                    raw = buffer_load(
                        prim_rsrc,
                        lane_off + fx.Int32(c * LANE_STRIDE_I32),
                        vec_width=VEC_I32,
                        dtype=T.i32,
                    )
                    los = []
                    his = []
                    for pos in range_constexpr(VEC_I32):
                        dw = fly_vector.extract(
                            raw, static_position=[pos], dynamic_position=[]
                        )
                        lo, hi = _unpack_bf16_pair(dw)
                        los.append(lo)
                        his.append(hi)
                    acc_lo.append(los)
                    acc_hi.append(his)

                for j in range(fx.Int32(0), n_dups, 1):
                    dup_row = buffer_load(
                        list_rsrc,
                        primary * fx.Int32(dup_stride) + j,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    dup_rsrc = create_buffer_resource_from_addr(
                        addr_hidden + fx.Int64(dup_row) * hidden_bytes
                    )
                    for c in range_constexpr(full_chunks):
                        raw = buffer_load(
                            dup_rsrc,
                            lane_off + fx.Int32(c * LANE_STRIDE_I32),
                            vec_width=VEC_I32,
                            dtype=T.i32,
                        )
                        for pos in range_constexpr(VEC_I32):
                            dw = fly_vector.extract(
                                raw, static_position=[pos], dynamic_position=[]
                            )
                            lo, hi = _unpack_bf16_pair(dw)
                            acc_lo[c][pos] = acc_lo[c][pos] + lo
                            acc_hi[c][pos] = acc_hi[c][pos] + hi

                for c in range_constexpr(full_chunks):
                    packed = [
                        _pack_bf16_pair(acc_lo[c][pos], acc_hi[c][pos])
                        for pos in range(VEC_I32)
                    ]
                    buffer_store(
                        fly_vector.from_elements(T.vec(VEC_I32, T.i32), packed),
                        prim_rsrc,
                        lane_off + fx.Int32(c * LANE_STRIDE_I32),
                    )

                # Tail chunk; folds to a never-taken branch when tail_vecs == 0.
                if lane < fx.Int32(tail_vecs):
                    off = lane_off + fx.Int32(tail_off)
                    raw = buffer_load(
                        prim_rsrc, off, vec_width=VEC_I32, dtype=T.i32
                    )
                    t_lo = []
                    t_hi = []
                    for pos in range_constexpr(VEC_I32):
                        dw = fly_vector.extract(
                            raw, static_position=[pos], dynamic_position=[]
                        )
                        lo, hi = _unpack_bf16_pair(dw)
                        t_lo.append(lo)
                        t_hi.append(hi)
                    for j in range(fx.Int32(0), n_dups, 1):
                        dup_row = buffer_load(
                            list_rsrc,
                            primary * fx.Int32(dup_stride) + j,
                            vec_width=1,
                            dtype=T.i32,
                        )
                        dup_rsrc = create_buffer_resource_from_addr(
                            addr_hidden + fx.Int64(dup_row) * hidden_bytes
                        )
                        raw_d = buffer_load(
                            dup_rsrc, off, vec_width=VEC_I32, dtype=T.i32
                        )
                        for pos in range_constexpr(VEC_I32):
                            dw = fly_vector.extract(
                                raw_d, static_position=[pos], dynamic_position=[]
                            )
                            lo, hi = _unpack_bf16_pair(dw)
                            t_lo[pos] = t_lo[pos] + lo
                            t_hi[pos] = t_hi[pos] + hi
                    packed = [
                        _pack_bf16_pair(t_lo[pos], t_hi[pos])
                        for pos in range(VEC_I32)
                    ]
                    buffer_store(
                        fly_vector.from_elements(T.vec(VEC_I32, T.i32), packed),
                        prim_rsrc,
                        off,
                    )

    @flyc.jit
    def launch(
        addr_hidden: fx.Int64,
        addr_duplicate_src: fx.Int64,
        addr_dup_count: fx.Int64,
        addr_dup_list: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        build_dup_lists(addr_duplicate_src, addr_dup_count, addr_dup_list).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )
        reduce_duplicates(addr_hidden, addr_dup_count, addr_dup_list).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


__all__ = ["make_moonep_combine_prologue_jit"]
