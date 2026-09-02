# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-tensor intranode push all-to-all kernel."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import T, const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.arith import FastMathFlags
from flydsl.expr.rocdl import readfirstlane
from flydsl.expr.typing import ReductionOp, Stream

from .buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from .communication_ops_utils import (
    atomic_add_global_at,
    fence_system_acquire,
    store_i64_global_system,
)

_JIT_SCHEMA_VERSION = "v9-transport-only-qkv"
_PUSH_PIPELINE_DEPTH = 16
_OUT_CHANNEL_COUNT = 8
_OUT_CHANNEL_DEPTH = 1


def make_fused_a2a_kernel(
    *,
    rank,
    npes,
    heads,
    seq_len,
    head_dim,
    block_num,
    warp_num_per_block,
    fuse_norm_rope,
):
    row_nbytes = head_dim * 2
    if row_nbytes % 16 != 0:
        raise ValueError(f"head row must be 16-byte aligned, got {row_nbytes}")

    heads_local = heads // npes
    seq_full = seq_len * npes
    chunks_per_row = row_nbytes // 16
    total_chunks = heads * seq_len * chunks_per_row
    vec = 8
    block_threads = 64
    hd = heads * head_dim
    tile = block_threads * vec
    if hd % tile != 0 or tile % head_dim != 0:
        raise ValueError(f"unsupported Q/K norm tiling for H={heads}, D={head_dim}")
    n_tiles = hd // tile
    shared_storage = fx.struct(
        type(
            "_SharedStorage",
            (),
            {
                "__annotations__": {
                    "p2p_bases_q": fx.Array[fx.Int64, npes, 16],
                    "p2p_bases_k": fx.Array[fx.Int64, npes, 16],
                    "p2p_bases_v": fx.Array[fx.Int64, npes, 16],
                }
            },
        )
    )

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def fused_a2a_push(
        addr_input_q: fx.Int64,
        addr_input_k: fx.Int64,
        addr_input_v: fx.Int64,
        addr_norm_q: fx.Int64,
        addr_norm_k: fx.Int64,
        addr_cos: fx.Int64,
        addr_sin: fx.Int64,
        addr_p2p_output_q: fx.Int64,
        addr_p2p_output_k: fx.Int64,
        addr_p2p_output_v: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_grid_barrier: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        grid_thread_id = bid * (warp_num_per_block * 64) + tid

        rsrc_p2p_output_q = create_buffer_resource_from_addr(addr_p2p_output_q)
        rsrc_p2p_output_k = create_buffer_resource_from_addr(addr_p2p_output_k)
        rsrc_p2p_output_v = create_buffer_resource_from_addr(addr_p2p_output_v)
        rsrc_p2p_xdb = create_buffer_resource_from_addr(addr_p2p_xdb_mem)
        rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
        rsrc_grid_barrier = create_buffer_resource_from_addr(addr_grid_barrier)

        shared = fx.SharedAllocator().allocate(shared_storage).peek()
        p2p_bases_q = shared.p2p_bases_q.view(fx.make_layout(npes, 1))
        p2p_bases_k = shared.p2p_bases_k.view(fx.make_layout(npes, 1))
        p2p_bases_v = shared.p2p_bases_v.view(fx.make_layout(npes, 1))
        if lane < npes:
            peer_base_q = buffer_load(rsrc_p2p_output_q, lane, vec_width=1, dtype=T.i64)
            peer_base_k = buffer_load(rsrc_p2p_output_k, lane, vec_width=1, dtype=T.i64)
            peer_base_v = buffer_load(rsrc_p2p_output_v, lane, vec_width=1, dtype=T.i64)
            fx.memref_store(peer_base_q, p2p_bases_q, lane)
            fx.memref_store(peer_base_k, p2p_bases_k, lane)
            fx.memref_store(peer_base_v, p2p_bases_v, lane)
        fx.barrier()

        rsrc_input_q = create_buffer_resource_from_addr(addr_input_q)
        rsrc_input_k = create_buffer_resource_from_addr(addr_input_k)
        rsrc_input_v = create_buffer_resource_from_addr(addr_input_v)
        rsrc_norm_q = create_buffer_resource_from_addr(addr_norm_q)
        rsrc_norm_k = create_buffer_resource_from_addr(addr_norm_k)
        rsrc_cos = create_buffer_resource_from_addr(addr_cos)
        rsrc_sin = create_buffer_resource_from_addr(addr_sin)
        fm_fast = FastMathFlags.fast

        def wave_reduce_add(value):
            result = fx.Float32(value)
            for shift in (32, 16, 8, 4, 2, 1):
                result = result.addf(
                    result.shuffle_xor(shift, block_threads), fastmath=fm_fast
                )
            return result

        def process_qk(rsrc_input, rsrc_norm, p2p_bases):
            for seq in range(global_warp_id, seq_len, global_warp_num):
                tiles = []
                sq_acc = fx.Float32(0.0)
                row_base = seq * hd
                head_offset = (lane * vec) % head_dim
                freq_offset = seq * head_dim + head_offset
                cos_f = fx.Vector(
                    buffer_load(rsrc_cos, freq_offset, vec_width=4, dtype=T.f32)
                )
                cos_f_hi = fx.Vector(
                    buffer_load(rsrc_cos, freq_offset + 4, vec_width=4, dtype=T.f32)
                )
                sin_f = fx.Vector(
                    buffer_load(rsrc_sin, freq_offset, vec_width=4, dtype=T.f32)
                )
                sin_f_hi = fx.Vector(
                    buffer_load(rsrc_sin, freq_offset + 4, vec_width=4, dtype=T.f32)
                )
                for tile_idx in range_constexpr(n_tiles):
                    element_offset = row_base + tile_idx * tile + lane * vec
                    values = fx.Vector(
                        buffer_load(
                            rsrc_input, element_offset, vec_width=vec, dtype=T.bf16
                        )
                    )
                    tiles.append(values)
                    values_f = values.to(fx.Float32)
                    sq_acc = sq_acc.addf(
                        fx.Float32(
                            (values_f * values_f).reduce(
                                ReductionOp.ADD, fastmath=fm_fast
                            )
                        ),
                        fastmath=fm_fast,
                    )
                rstd = fmath.rsqrt(
                    wave_reduce_add(sq_acc) * (1.0 / hd) + 1.0e-6,
                    fastmath=fm_fast,
                )
                for batch_start in range_constexpr(0, n_tiles, _PUSH_PIPELINE_DEPTH):
                    outputs = []
                    destinations = []
                    batch_size = min(_PUSH_PIPELINE_DEPTH, n_tiles - batch_start)
                    for batch_idx in range_constexpr(batch_size):
                        tile_idx = batch_start + batch_idx
                        col = tile_idx * tile + lane * vec
                        weights = fx.Vector(
                            buffer_load(rsrc_norm, col, vec_width=vec, dtype=T.bf16)
                        ).to(fx.Float32)
                        values_f = tiles[tile_idx].to(fx.Float32)
                        scaled = [
                            values_f[i] * rstd * weights[i]
                            for i in range_constexpr(vec)
                        ]
                        rotated = [None] * vec
                        for pair in range_constexpr(vec // 2):
                            even = scaled[2 * pair]
                            odd = scaled[2 * pair + 1]
                            cos_even = (
                                cos_f[2 * pair] if pair < 2 else cos_f_hi[2 * pair - 4]
                            )
                            sin_odd = (
                                sin_f[2 * pair + 1]
                                if pair < 2
                                else sin_f_hi[2 * pair - 3]
                            )
                            rotated[2 * pair] = even * cos_even - odd * sin_odd
                            rotated[2 * pair + 1] = even * sin_odd + odd * cos_even
                        outputs.append(
                            fx.Vector.from_elements(
                                [value.ir_value() for value in rotated],
                                dtype=fx.Float32,
                            ).to(fx.BFloat16)
                        )
                        destinations.append(tile_idx)
                    lane_group = lane >> 4
                    lane_in_group = lane & 15
                    for batch_idx in range_constexpr(batch_size):
                        tile_idx = destinations[batch_idx]
                        for group in range_constexpr(4):
                            if lane_group == group:
                                head = tile_idx * 4 + group
                                dest_pe = head // heads_local
                                local_head = head % heads_local
                                dst_row = local_head * seq_full + rank * seq_len + seq
                                peer_base = fx.memref_load(p2p_bases, dest_pe)
                                dst_addr = fx.Uint64(
                                    peer_base + fx.Int64(dst_row * row_nbytes)
                                )
                                dst_addr_lo = readfirstlane(T.i32, fx.Uint32(dst_addr))
                                dst_addr_hi = readfirstlane(
                                    T.i32, fx.Uint32(dst_addr >> 32)
                                )
                                uniform_dst_addr = (
                                    fx.Uint64(dst_addr_hi) << 32
                                ) | fx.Uint64(dst_addr_lo)
                                rsrc_dst = create_buffer_resource_from_addr(
                                    uniform_dst_addr, num_records_bytes=row_nbytes
                                )
                                buffer_store(
                                    outputs[batch_idx], rsrc_dst, lane_in_group * 8
                                )

        def transport(input_rsrc, p2p_bases):
            peer_chunks = total_chunks // npes
            peer_group_count = (peer_chunks + 63) // 64
            peer_warp_num = global_warp_num // npes
            dest_pe = global_warp_id % npes
            peer_warp_id = global_warp_id // npes
            peer_base = fx.Uint64(fx.memref_load(p2p_bases, dest_pe))
            peer_base_lo = readfirstlane(T.i32, fx.Uint32(peer_base))
            peer_base_hi = readfirstlane(T.i32, fx.Uint32(peer_base >> 32))
            uniform_peer_base = (fx.Uint64(peer_base_hi) << 32) | fx.Uint64(
                peer_base_lo
            )
            rsrc_dst = create_buffer_resource_from_addr(
                uniform_peer_base, num_records_bytes=total_chunks * 16
            )
            group_step = peer_warp_num * _PUSH_PIPELINE_DEPTH
            for group_base in range(peer_warp_id, peer_group_count, group_step):
                values = []
                destinations = []
                valid_values = []
                for batch_idx in range_constexpr(_PUSH_PIPELINE_DEPTH):
                    group_idx = group_base + batch_idx * peer_warp_num
                    dest_chunk = group_idx * 64 + lane
                    valid = dest_chunk < peer_chunks
                    safe_dest_chunk = valid.select(dest_chunk, 0)
                    local_head = safe_dest_chunk // (seq_len * chunks_per_row)
                    seq_chunk = safe_dest_chunk % (seq_len * chunks_per_row)
                    seq = seq_chunk // chunks_per_row
                    row_chunk = seq_chunk % chunks_per_row
                    head = dest_pe * heads_local + local_head
                    src_chunk = (seq * heads + head) * chunks_per_row + row_chunk
                    values.append(
                        buffer_load(
                            input_rsrc,
                            src_chunk * 4,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    dst_chunk = (
                        local_head * seq_full * chunks_per_row
                        + (rank * seq_len + seq) * chunks_per_row
                        + row_chunk
                    )
                    destinations.append(dst_chunk * 4)
                    valid_values.append(valid)
                for batch_idx in range_constexpr(_PUSH_PIPELINE_DEPTH):
                    if valid_values[batch_idx]:
                        buffer_store(
                            values[batch_idx], rsrc_dst, destinations[batch_idx]
                        )

        if const_expr(fuse_norm_rope):
            process_qk(rsrc_input_q, rsrc_norm_q, p2p_bases_q)
            process_qk(rsrc_input_k, rsrc_norm_k, p2p_bases_k)
        else:
            transport(rsrc_input_q, p2p_bases_q)
            transport(rsrc_input_k, p2p_bases_k)
        transport(rsrc_input_v, p2p_bases_v)

        fx.rocdl.s_waitcnt(vmcnt=0)

        # All blocks must be resident: this is a grid-wide software barrier.
        fx.barrier()
        if tid == 0:
            atomic_add_global_at(addr_grid_barrier, 1)

        xdb_cur_flag = buffer_load(rsrc_xdb_flag, 0, vec_width=1, dtype=T.i64)
        if grid_thread_id < npes:
            mori_shmem.int32_wait_until_equals(addr_grid_barrier, block_num)
            fence_system_acquire()
            buffer_store(fx.Int32(0), rsrc_grid_barrier, 0)
            xdb_remote_addr = (
                buffer_load(rsrc_p2p_xdb, grid_thread_id, vec_width=1, dtype=T.i64)
                + fx.Int64(rank) * 8
            )
            store_i64_global_system(xdb_remote_addr, xdb_cur_flag)

        if grid_thread_id == 0:
            atomic_add_global_at(addr_xdb_flag, fx.Int64(1))

        if tid < npes:
            peer_slot = addr_xdb_mem + fx.Int64(tid) * 8
            mori_shmem.uint64_wait_until_equals(peer_slot, xdb_cur_flag)
            fence_system_acquire()
        fx.barrier()

    return fused_a2a_push


def make_fused_a2a_jit(
    *,
    rank,
    npes,
    heads,
    seq_len,
    head_dim,
    block_num,
    warp_num_per_block,
    fuse_norm_rope,
):
    kernel = make_fused_a2a_kernel(
        rank=rank,
        npes=npes,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
        fuse_norm_rope=fuse_norm_rope,
    )
    key = (
        rank,
        npes,
        heads,
        seq_len,
        head_dim,
        block_num,
        warp_num_per_block,
        fuse_norm_rope,
        _JIT_SCHEMA_VERSION,
    )

    @flyc.jit
    def launch(
        addr_input_q: fx.Int64,
        addr_input_k: fx.Int64,
        addr_input_v: fx.Int64,
        addr_norm_q: fx.Int64,
        addr_norm_k: fx.Int64,
        addr_cos: fx.Int64,
        addr_sin: fx.Int64,
        addr_p2p_output_q: fx.Int64,
        addr_p2p_output_k: fx.Int64,
        addr_p2p_output_v: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_grid_barrier: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        kernel(
            addr_input_q,
            addr_input_k,
            addr_input_v,
            addr_norm_q,
            addr_norm_k,
            addr_cos,
            addr_sin,
            addr_p2p_output_q,
            addr_p2p_output_k,
            addr_p2p_output_v,
            addr_xdb_mem,
            addr_p2p_xdb_mem,
            addr_xdb_flag,
            addr_grid_barrier,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return launch


def make_fused_a2a_out_kernel(
    *, rank, npes, heads_local, seq_local, head_dim, block_num, warp_num_per_block
):
    row_nbytes = head_dim * 2
    seq_full = seq_local * npes
    chunks_per_row = row_nbytes // 16
    peer_chunks = seq_local * heads_local * chunks_per_row
    shared_storage = fx.struct(
        type(
            "_OutSharedStorage",
            (),
            {"__annotations__": {"p2p_bases": fx.Array[fx.Int64, npes, 16]}},
        )
    )

    @flyc.kernel(known_block_size=[warp_num_per_block * 64, 1, 1])
    def fused_a2a_out_push(
        addr_input: fx.Int64,
        addr_p2p_output: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_grid_barrier: fx.Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & 63
        warp = tid >> 6
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        grid_thread_id = bid * (warp_num_per_block * 64) + tid

        rsrc_p2p_output = create_buffer_resource_from_addr(addr_p2p_output)
        rsrc_p2p_xdb = create_buffer_resource_from_addr(addr_p2p_xdb_mem)
        rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
        rsrc_grid_barrier = create_buffer_resource_from_addr(addr_grid_barrier)

        shared = fx.SharedAllocator().allocate(shared_storage).peek()
        p2p_bases = shared.p2p_bases.view(fx.make_layout(npes, 1))
        if lane < npes:
            peer_base = buffer_load(rsrc_p2p_output, lane, vec_width=1, dtype=T.i64)
            fx.memref_store(peer_base, p2p_bases, lane)
        fx.barrier()

        rsrc_input = create_buffer_resource_from_addr(addr_input)
        channel_warp_num = global_warp_num // _OUT_CHANNEL_COUNT
        channel_id = global_warp_id % _OUT_CHANNEL_COUNT
        channel_warp_id = global_warp_id // _OUT_CHANNEL_COUNT
        group_step = channel_warp_num * _OUT_CHANNEL_DEPTH
        peer_base = fx.memref_load(p2p_bases, channel_id)
        dst_addr = peer_base + fx.Int64(rank * peer_chunks * 16)
        rsrc_dst = create_buffer_resource_from_addr(
            dst_addr, num_records_bytes=peer_chunks * 16
        )
        peer_groups = (peer_chunks + 63) // 64
        for group_base in range(channel_warp_id, peer_groups, group_step):
            values = []
            destinations = []
            for batch_idx in range_constexpr(_OUT_CHANNEL_DEPTH):
                group_idx = group_base + batch_idx * channel_warp_num
                dest_chunk = group_idx * 64 + lane
                valid = dest_chunk < peer_chunks
                safe_dest_chunk = valid.select(dest_chunk, 0)
                seq = safe_dest_chunk // (heads_local * chunks_per_row)
                head_chunk = safe_dest_chunk % (heads_local * chunks_per_row)
                local_head = head_chunk // chunks_per_row
                row_chunk = head_chunk % chunks_per_row
                src_row = local_head * seq_full + channel_id * seq_local + seq
                src_offset = src_row * chunks_per_row * 4 + row_chunk * 4
                values.append(
                    buffer_load(rsrc_input, src_offset, vec_width=4, dtype=T.i32)
                )
                destinations.append(dest_chunk * 4)
            for batch_idx in range_constexpr(_OUT_CHANNEL_DEPTH):
                if destinations[batch_idx] < peer_chunks * 4:
                    buffer_store(values[batch_idx], rsrc_dst, destinations[batch_idx])

        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.barrier()
        if tid == 0:
            atomic_add_global_at(addr_grid_barrier, 1)

        xdb_cur_flag = buffer_load(rsrc_xdb_flag, 0, vec_width=1, dtype=T.i64)
        if grid_thread_id < npes:
            mori_shmem.int32_wait_until_equals(addr_grid_barrier, block_num)
            fence_system_acquire()
            buffer_store(fx.Int32(0), rsrc_grid_barrier, 0)
            xdb_remote_addr = (
                buffer_load(rsrc_p2p_xdb, grid_thread_id, vec_width=1, dtype=T.i64)
                + fx.Int64(rank) * 8
            )
            store_i64_global_system(xdb_remote_addr, xdb_cur_flag)

        if grid_thread_id == 0:
            atomic_add_global_at(addr_xdb_flag, fx.Int64(1))

        if tid < npes:
            peer_slot = addr_xdb_mem + fx.Int64(tid) * 8
            mori_shmem.uint64_wait_until_equals(peer_slot, xdb_cur_flag)
            fence_system_acquire()
        fx.barrier()

    return fused_a2a_out_push


def make_fused_a2a_out_jit(
    *, rank, npes, heads_local, seq_local, head_dim, block_num, warp_num_per_block
):
    kernel = make_fused_a2a_out_kernel(
        rank=rank,
        npes=npes,
        heads_local=heads_local,
        seq_local=seq_local,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )
    key = (
        rank,
        npes,
        heads_local,
        seq_local,
        head_dim,
        block_num,
        warp_num_per_block,
        _JIT_SCHEMA_VERSION,
    )

    @flyc.jit
    def launch(
        addr_input: fx.Int64,
        addr_p2p_output: fx.Int64,
        addr_xdb_mem: fx.Int64,
        addr_p2p_xdb_mem: fx.Int64,
        addr_xdb_flag: fx.Int64,
        addr_grid_barrier: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        _ = key
        kernel(
            addr_input,
            addr_p2p_output,
            addr_xdb_mem,
            addr_p2p_xdb_mem,
            addr_xdb_flag,
            addr_grid_barrier,
        ).launch(
            grid=(block_num, 1, 1),
            block=(warp_num_per_block * 64, 1, 1),
            stream=stream,
        )

    return launch
