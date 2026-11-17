# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Receiver-driven FP4 dispatch into the grouped GEMM's packed A layout."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import tdm_ops

from aiter.ops.flydsl.kernels.moe_apreshuffled_quant_scatter import (
    make_apre_payload_ops,
)

from .tdm_gather_shim import make_tensor_gather_descriptor


@functools.cache
def compile_pull_dispatch(
    *,
    rank,
    world,
    experts,
    max_routes,
    model_dim,
    wire_row,
    off_wire,
    off_ready,
    off_rowmap,
    align_m,
    topk,
    blocks=256,
    a_preshuffle=True,
    tdm_gather=False,
):
    """Compile an M16 gather; each workgroup owns complete A tiles.

    The producer plan publishes route identities in ep_rowmap. Quantized source
    rows live in the symmetric arena until the existing combine barrier, which
    prevents the next layer overwriting them before every peer finishes pull.

    With a_preshuffle=False, write row-major payload storage. The caller must
    also select the row-major GEMM input contract.

    Each wave in tdm_gather mode issues two eight-row gathers for one M16
    tile before waiting. Four waves each process one tile, with no persistent
    loop. Disjoint LDS slices and output vectors avoid workgroup barriers;
    the source-ready protocol and initial metadata barriers remain shared.
    """
    payload_bytes = model_dim // 2
    scale_bytes = model_dim // 32
    threads = 128 if tdm_gather else 512
    tiles_per_block = threads // 32 if tdm_gather else 1
    if tdm_gather and wire_row % 128:
        raise ValueError("TDM gather requires a 128B wire stride")
    payload_lds_dwords, copy_payload, store_payload = make_apre_payload_ops(
        model_dim,
        lds_row_stride_bytes=wire_row if tdm_gather else 0,
        output_preshuffled=a_preshuffle if tdm_gather else True,
    )
    if tdm_gather:
        payload_lds_dwords = tiles_per_block * 16 * wire_row // 4
    if not a_preshuffle and not tdm_gather:
        payload_lds_dwords = 1
    if payload_lds_dwords * 4 + experts * 4 > 256 * 1024:
        raise ValueError("pull dispatch LDS tile exceeds 256 KiB")

    @fx.struct
    class Storage:
        rows: fx.Array[fx.Int32, payload_lds_dwords, 16]
        ends: fx.Array[fx.Int32, experts]

    @flyc.kernel(
        name=(
            f"ep_dispatch_pull_{'apre' if a_preshuffle else 'rowmajor'}"
            f"_h{model_dim}_a{align_m}"
            f"{'_tdmgather_t128_g2x8_grid' if tdm_gather else ''}"
        ),
        known_block_size=(threads, 1, 1),
    )
    def kernel(
        arena: fx.Int64,
        psum: fx.Tensor,
        generation: fx.Tensor,
        output: fx.Tensor,
        scales: fx.Tensor,
        rows: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 32
        warp = tid // 32
        window = cco.CachedWindow(arena)
        global_i32 = fx.PointerType.get(
            fx.Int32.ir_type, fx.AddressSpace.Global, 4
        )
        local_ready = window.base + fx.Int64(rank) * window.stride + off_ready
        epoch = generation[2]
        # Atomic system release/acquire publishes all earlier plan/quant writes.
        # Every block can publish the same epoch; none waits on a queued block.
        # LLVM AMDGPU spells system scope as the empty string.
        if tid == 0:
            fx.generic_store(
                fx.inttoptr(global_i32, local_ready),
                epoch,
                memory_order=fx.AtomicOrdering.Release,
                syncscope="",
            )
        fx.barrier()
        if tid < world:
            address = window.base + fx.Int64(tid) * window.stride + off_ready
            ready = fx.generic_load(
                fx.inttoptr(global_i32, address),
                memory_order=fx.AtomicOrdering.Acquire,
                syncscope="",
            )
            while ready != epoch:
                ready = fx.generic_load(
                    fx.inttoptr(global_i32, address),
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="",
                )
        fx.barrier()
        storage = fx.SharedAllocator().allocate(Storage).peek()
        output_address = fx.Int64(fx.ptrtoint(fx.get_iter(output)))
        ends = storage.ends.ptr
        for expert in range(tid, experts, threads):
            ends[expert] = psum[expert]
        fx.barrier()
        local_map = (
            window.base + fx.Int64(rank) * window.stride + fx.Int64(off_rowmap)
        )
        # psum has unpadded ends. Work begins only inside an expert, so stale
        # mapping entries in padding or an empty expert are never dereferenced.
        live_rows = fx.min(rows, ends[experts - 1])
        if const_expr(tdm_gather):
            tile = fx.block_idx.x * tiles_per_block + warp
            if tile * 16 < live_rows:
                row0 = tile * 16
                # Find the first unpadded expert end after this tile. Binary search
                # bounds metadata traffic independently of the expert count.
                lower = fx.Int32(0)
                upper = fx.Int32(experts - 1)
                while lower < upper:
                    middle = (lower + upper) // 2
                    if ends[middle] <= row0:
                        lower = middle + 1
                    else:
                        upper = middle
                expert = lower
                end = ends[expert]
                prev = (expert > 0).select(expert - 1, 0)
                start = ((ends[prev] + align_m - 1) // align_m) * align_m
                start = (expert > 0).select(start, 0)
                if (row0 >= start) & (row0 < end):
                    gather_row = row0 + lane % 16
                    mapped_row = (gather_row < end).select(gather_row, row0)
                    packed = fx.generic_load(
                        fx.inttoptr(
                            global_i32, local_map + fx.Int64(mapped_row) * 8
                        )
                    )
                    source = packed // max_routes
                    token = (packed % max_routes) // topk
                    # Rank windows need not be divisible by wire_row. Gather
                    # indices use 128B address units across the flat VA.
                    source_index = fx.Int32(
                        (fx.Int64(source) * window.stride
                         + fx.Int64(token) * wire_row) // 128
                    )
                    wire_bound = fx.Int32(window.stride * world // 128)
                    wire_pointer = fx.inttoptr(
                        fx.PointerType.get(
                            fx.Int8.ir_type, fx.AddressSpace.Global, 128
                        ),
                        window.base + off_wire,
                    )
                    wire_view = fx.make_view(
                        wire_pointer,
                        fx.make_layout((wire_bound, wire_row), (128, 1)),
                    )
                    lds_view = fx.make_view(
                        fx.recast_iter(fx.Int8, storage.rows.ptr),
                        fx.make_layout(tiles_per_block * 16 * wire_row, 1),
                    )
                    for gather in range_constexpr(2):
                        row_indices = []
                        # TDM descriptors require SGPR indices. No typed
                        # wave-uniform cast exists; readlane supplies each index.
                        for index in range_constexpr(8):
                            row_indices.append(fx.Int32(fx.rocdl.readlane(
                                fx.Int32.ir_type, source_index.ir_value(),
                                fx.Int32(gather * 8 + index).ir_value(),
                            )))
                        descriptor = make_tensor_gather_descriptor(
                            global_ptr=wire_view,
                            lds_memref=lds_view,
                            row_indices=row_indices,
                            row_width=wire_row,
                            tensor_dim0=wire_row,
                            tensor_dim1=wire_bound.ir_value(),
                            stride=128,
                            elem_bytes=1,
                            index_size=32,
                            lds_byte_offset=(warp * 16 + gather * 8) * wire_row,
                        )
                        tdm_ops.tensor_load_gather(descriptor)
                    # Both gathers are in flight before this wave waits. Its
                    # LDS tile is private, so no workgroup barrier is needed.
                    tdm_ops.tensor_wait(0)
                    for item in range(lane, 16 * (scale_bytes // 4), 32):
                        tile_row = item // (scale_bytes // 4)
                        word = item % (scale_bytes // 4)
                        scales[row0 * (scale_bytes // 4) + item] = (
                            storage.rows.ptr[
                                (warp * 16 + tile_row) * (wire_row // 4)
                                + payload_bytes // 4 + word
                            ]
                        )
                    lds_address = fx.Int64(fx.ptrtoint(storage.rows.ptr))
                    lds_vector = fx.PointerType.get(
                        fx.Int32.ir_type, fx.AddressSpace.Shared, 16
                    )
                    global_vector = fx.PointerType.get(
                        fx.Int32.ir_type, fx.AddressSpace.Global, 16
                    )
                    # Consecutive lanes own consecutive 16B output vectors.
                    # Packed A is [M/16, K/32, M%16, 16 payload bytes].
                    for item in range(lane, payload_bytes, 32):
                        if const_expr(a_preshuffle):
                            tile_row = item % 16
                            vector = item // 16
                        else:
                            tile_row = item // (payload_bytes // 16)
                            vector = item % (payload_bytes // 16)
                        source_offset = (
                            (warp * 16 + tile_row) * wire_row + vector * 16
                        )
                        payload = fx.generic_load(
                            fx.inttoptr(
                                lds_vector, lds_address + fx.Int64(source_offset)
                            ),
                            count=4,
                        )
                        output_offset = (
                            fx.Int64(tile) * (16 * payload_bytes)
                            + fx.Int64(item) * 16
                        )
                        fx.generic_store(
                            fx.inttoptr(
                                global_vector, output_address + output_offset
                            ),
                            payload,
                        )
        else:
            for tile in range(fx.block_idx.x, (live_rows + 15) // 16, blocks):
                row0 = tile * 16
                # Find the first unpadded expert end after this tile. Binary search
                # bounds metadata traffic independently of the expert count.
                lower = fx.Int32(0)
                upper = fx.Int32(experts - 1)
                while lower < upper:
                    middle = (lower + upper) // 2
                    if ends[middle] <= row0:
                        lower = middle + 1
                    else:
                        upper = middle
                expert = lower
                end = ends[expert]
                prev = (expert > 0).select(expert - 1, 0)
                start = ((ends[prev] + align_m - 1) // align_m) * align_m
                start = (expert > 0).select(start, 0)
                if (row0 >= start) & (row0 < end):
                    row = row0 + warp
                    # As in the original scatter, padding uses a valid source row;
                    # only live expert rows are consumed by GEMM and combine.
                    mapped_row = (row < end).select(row, row0)
                    packed = fx.generic_load(
                        fx.inttoptr(global_i32, local_map + fx.Int64(mapped_row) * 8)
                    )
                    source = packed // max_routes
                    token = (packed % max_routes) // topk
                    address = (
                        window.base
                        + fx.Int64(source) * window.stride
                        + off_wire
                        + fx.Int64(token) * wire_row
                    )
                    if const_expr(a_preshuffle):
                        copy_payload(storage.rows.ptr, address)
                    else:
                        vector_pointer = fx.PointerType.get(
                            fx.Int32.ir_type, fx.AddressSpace.Global, 16
                        )
                        payload_vectors = []
                        for chunk in range_constexpr(payload_bytes // 512):
                            vector = lane + chunk * 32
                            offset = fx.Int64(vector) * 16
                            payload_vectors.append(
                                fx.generic_load(
                                    fx.inttoptr(vector_pointer, address + offset),
                                    count=4,
                                )
                            )
                        for chunk in range_constexpr(payload_bytes // 512):
                            vector = lane + chunk * 32
                            offset = fx.Int64(vector) * 16
                            destination = (
                                output_address + fx.Int64(row) * payload_bytes + offset
                            )
                            fx.generic_store(
                                fx.inttoptr(vector_pointer, destination),
                                payload_vectors[chunk],
                            )
                    # Scales retain the ordinary compact GEMM's row-major layout.
                    for word in range(lane, scale_bytes // 4, 32):
                        scale_address = address + payload_bytes + fx.Int64(word) * 4
                        scales[row * (scale_bytes // 4) + word] = fx.generic_load(
                            fx.inttoptr(global_i32, scale_address)
                        )
                    if const_expr(a_preshuffle):
                        fx.rocdl.s_wait_asynccnt(0)
                        fx.barrier()
                        store_payload(storage.rows.ptr, output_address, fx.Uint32(tile))
                        fx.barrier()

    @flyc.jit
    def launch(
        arena: fx.Int64,
        psum: fx.Tensor,
        generation: fx.Tensor,
        output: fx.Tensor,
        scales: fx.Tensor,
        rows: fx.Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        grid_blocks = blocks
        if const_expr(tdm_gather):
            # Use the existing graph-safe row capacity. Device psum guards
            # unused tiles without a host readback or balanced-routing assumption.
            grid_blocks = fx.max(1, (rows + 63) // 64)
        kernel(arena, psum, generation, output, scales, rows).launch(
            grid=(grid_blocks, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch
