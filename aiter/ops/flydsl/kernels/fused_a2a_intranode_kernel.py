# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-tensor intranode push all-to-all kernel."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import T
from flydsl.expr.typing import Stream

from .buffer_ops import buffer_load, buffer_store, create_buffer_resource_from_addr
from .communication_ops_utils import (
    atomic_add_global_at,
    fence_system_acquire,
    store_i64_global_system,
)

_JIT_SCHEMA_VERSION = "v3-fused-in-hop-pack-qkv"


def make_fused_a2a_kernel(
    *, rank, npes, heads, seq_len, head_dim, block_num, warp_num_per_block
):
    row_nbytes = head_dim * 2
    if row_nbytes % 16 != 0:
        raise ValueError(f"head row must be 16-byte aligned, got {row_nbytes}")

    heads_local = heads // npes
    seq_full = seq_len * npes
    chunks_per_row = row_nbytes // 16
    total_chunks = heads * seq_len * chunks_per_row
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
        for chunk_idx in range(global_warp_id, total_chunks, global_warp_num):
            row = chunk_idx // chunks_per_row
            row_chunk = chunk_idx % chunks_per_row
            seq = row // heads
            head = row % heads
            dest_pe = head // heads_local
            local_head = head % heads_local
            peer_base_q = fx.memref_load(p2p_bases_q, dest_pe)
            peer_base_k = fx.memref_load(p2p_bases_k, dest_pe)
            peer_base_v = fx.memref_load(p2p_bases_v, dest_pe)
            dst_row = local_head * seq_full + rank * seq_len + seq
            dst_offset = fx.Int64(dst_row * row_nbytes + row_chunk * 16)
            rsrc_dst_q = create_buffer_resource_from_addr(peer_base_q + dst_offset)
            rsrc_dst_k = create_buffer_resource_from_addr(peer_base_k + dst_offset)
            rsrc_dst_v = create_buffer_resource_from_addr(peer_base_v + dst_offset)
            i32_offset = chunk_idx * 4
            value_q = buffer_load(rsrc_input_q, i32_offset, vec_width=4, dtype=T.i32)
            value_k = buffer_load(rsrc_input_k, i32_offset, vec_width=4, dtype=T.i32)
            value_v = buffer_load(rsrc_input_v, i32_offset, vec_width=4, dtype=T.i32)
            buffer_store(value_q, rsrc_dst_q, 0)
            buffer_store(value_k, rsrc_dst_k, 0)
            buffer_store(value_v, rsrc_dst_v, 0)

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
    *, rank, npes, heads, seq_len, head_dim, block_num, warp_num_per_block
):
    kernel = make_fused_a2a_kernel(
        rank=rank,
        npes=npes,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )
    key = (
        rank,
        npes,
        heads,
        seq_len,
        head_dim,
        block_num,
        warp_num_per_block,
        _JIT_SCHEMA_VERSION,
    )

    @flyc.jit
    def launch(
        addr_input_q: fx.Int64,
        addr_input_k: fx.Int64,
        addr_input_v: fx.Int64,
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
    total_chunks = npes * peer_chunks
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
        for chunk_idx in range(global_warp_id, total_chunks, global_warp_num):
            dest_pe = chunk_idx // peer_chunks
            dest_chunk = chunk_idx % peer_chunks
            seq = dest_chunk // (heads_local * chunks_per_row)
            head_chunk = dest_chunk % (heads_local * chunks_per_row)
            local_head = head_chunk // chunks_per_row
            row_chunk = head_chunk % chunks_per_row
            src_row = local_head * seq_full + dest_pe * seq_local + seq
            src_offset = src_row * chunks_per_row * 4 + row_chunk * 4
            value = buffer_load(rsrc_input, src_offset, vec_width=4, dtype=T.i32)
            peer_base = fx.memref_load(p2p_bases, dest_pe)
            dst_addr = peer_base + fx.Int64(rank * peer_chunks * 16 + dest_chunk * 16)
            buffer_store(value, create_buffer_resource_from_addr(dst_addr), 0)

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
        "v1-fused-out-hop",
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
