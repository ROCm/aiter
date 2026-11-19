# SPDX-License-Identifier: MIT
"""Pull each source token once per destination and scatter locally from LDS."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

from .dispatch_pull import pull_lds_row_pitch
from .tdm_gather_shim import make_tensor_gather_descriptor


def make_unique_producer(
    *,
    rank,
    world,
    model_dim,
    wire_row,
    max_routes,
    topk,
    producers,
    off_wire,
    off_routes,
    off_counts,
    off_tile_flag,
    flag_stride
):
    """Build a source-token scheduler with row-counted destination readiness.

    Every wave owns 16 unique source tokens. Payload and scale remain in its
    LDS slice while it writes all matching local routes. The last completed
    row of each M64 publishes the existing S1F flag after an acquire/release
    counter joins every writer. No cross-device completion atomic is added.
    """
    max_tokens = max_routes // topk
    source_tiles = (max_tokens + 15) // 16
    peer_bits = max(1, (world - 1).bit_length())
    peer_mask = (1 << peer_bits) - 1
    payload_bytes, scale_bytes = model_dim // 2, model_dim // 32
    lds_pitch = pull_lds_row_pitch(wire_row)

    @flyc.jit
    def producer(
        storage: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        psum: fx.Pointer,
        arena: fx.Int64,
        addr_tile_info: fx.Int64,
        rows: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int32,
    ):
        lane, wave = fx.thread_idx.x % 32, fx.thread_idx.x // 32
        window = cco.CachedWindow(arena)
        local = window.base + fx.Int64(rank) * window.stride
        i32ptr = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)
        lds_rows = fx.recast_iter(fx.Int32, storage)
        out_scales = fx.recast_iter(fx.Int32, scales)
        info = fx.inttoptr(i32ptr, addr_tile_info)
        counts = fx.inttoptr(i32ptr, local + off_counts)
        for task in range(slot * 4 + wave, world * source_tiles, producers * 4):
            source = task // source_tiles
            token0 = (task % source_tiles) * 16
            token = token0 + lane % 16
            source_base = window.base + fx.Int64(source) * window.stride
            route_map = fx.inttoptr(i32ptr, source_base + off_routes)
            token_count = fx.generic_load(fx.add_offset(route_map, max_routes))
            safe_token = fx.min(token, max_tokens - 1)
            mapped = [
                fx.generic_load(fx.add_offset(route_map, safe_token * topk + k))
                for k in range_constexpr(topk)
            ]
            keeps = [
                (token < token_count)
                & (mapped[k] >= 0)
                & ((mapped[k] & peer_mask) == rank)
                for k in range_constexpr(topk)
            ]
            needed = keeps[0]
            for k in range_constexpr(1, topk):
                needed = needed | keeps[k]
            active = needed.select(fx.Int32(1), fx.Int32(0))
            for shift in range_constexpr(5):
                active += active.shuffle_xor(fx.Int32(1 << shift), fx.Int32(32))
            if active > 0:
                wire_bound = fx.Int32(window.stride * world // 128)
                source_index = fx.Int32(
                    (fx.Int64(source) * window.stride + fx.Int64(token) * wire_row)
                    // 128
                )
                source_index = needed.select(source_index, wire_bound)
                wire_pointer = fx.inttoptr(
                    fx.PointerType.get(T.i8, fx.AddressSpace.Global, 128),
                    window.base + off_wire,
                )
                wire_view = fx.make_view(
                    wire_pointer, fx.make_layout((wire_bound, wire_row), (128, 1))
                )
                lds_view = fx.make_view(
                    fx.recast_iter(fx.Int8, storage), fx.make_layout(64 * lds_pitch, 1)
                )
                for gather in range_constexpr(2):
                    # No typed SGPR lane-extraction API; descriptors require
                    # scalar indices even though each lane owns a token.
                    indices = [
                        fx.Int32(
                            fx.rocdl.readlane(
                                T.i32,
                                source_index.ir_value(),
                                fx.Int32(gather * 8 + i).ir_value(),
                            )
                        )
                        for i in range_constexpr(8)
                    ]
                    descriptor = make_tensor_gather_descriptor(
                        global_ptr=wire_view,
                        lds_memref=lds_view,
                        row_indices=indices,
                        row_width=lds_pitch,
                        tensor_dim0=wire_row,
                        tensor_dim1=wire_bound.ir_value(),
                        stride=128,
                        elem_bytes=1,
                        index_size=32,
                        lds_byte_offset=(wave * 16 + gather * 8) * lds_pitch,
                    )
                    tdm_ops.tensor_load_gather(descriptor)
                tdm_ops.tensor_wait(0)
                # Indexed TDM stores reuse the loaded rows in LDS. The local
                # row-major output makes each payload transfer contiguous in K.
                payload_view = fx.make_view(
                    fx.recast_iter(fx.Int8, payload),
                    fx.make_layout((rows, payload_bytes), (payload_bytes, 1)),
                )
                for k in range_constexpr(topk):
                    destination = fx.Int32(
                        keeps[k].select(mapped[k] >> peer_bits, rows)
                    )
                    for group in range_constexpr(2):
                        # TDM descriptors require SGPR indices; readlane is the
                        # pinned API for extracting a wave-uniform scalar index.
                        indices = [
                            fx.Int32(
                                fx.rocdl.readlane(
                                    T.i32,
                                    destination.ir_value(),
                                    fx.Int32(group * 8 + i).ir_value(),
                                )
                            )
                            for i in range_constexpr(8)
                        ]
                        descriptor = make_tensor_gather_descriptor(
                            global_ptr=payload_view,
                            lds_memref=lds_view,
                            row_indices=indices,
                            row_width=lds_pitch,
                            tensor_dim0=payload_bytes,
                            tensor_dim1=rows.ir_value(),
                            stride=payload_bytes,
                            elem_bytes=1,
                            index_size=32,
                            lds_byte_offset=(wave * 16 + group * 8) * lds_pitch,
                        )
                        tdm_ops.tensor_store_gather(descriptor)
                # Small scale rows use coalesced vector-lane stores while the
                # payload TDM stores are in flight, reading the same LDS safely.
                for tile_row in range(16):
                    routes = [
                        fx.gpu.shuffle_idx(mapped[k], tile_row, 32)
                        for k in range_constexpr(topk)
                    ]
                    destinations = [
                        (token0 + tile_row < token_count)
                        & (routes[k] >= 0)
                        & ((routes[k] & peer_mask) == rank)
                        for k in range_constexpr(topk)
                    ]
                    for word in range(lane, scale_bytes // 4, 32):
                        value = lds_rows[
                            (wave * 16 + tile_row) * (lds_pitch // 4)
                            + payload_bytes // 4
                            + word
                        ]
                        for k in range_constexpr(topk):
                            if destinations[k]:
                                row = routes[k] >> peer_bits
                                out_scales[row * (scale_bytes // 4) + word] = value
                tdm_ops.tensor_wait(0)
                # Release each writer's complete payload/scale stores before
                # joining the completion chain. The last RMW acquires all prior
                # releases before publishing the original consumer-ready flag.
                comm_ops.waitcnt_stores()
                comm_ops.fence_agent_release()
                if lane < 16:
                    for k in range_constexpr(topk):
                        if keeps[k]:
                            tile = (mapped[k] >> peer_bits) // 64
                            expected = (info[tile] >> 16) & 0xFFFF
                            previous = fx.atomic_add(
                                fx.add_offset(counts, tile),
                                fx.Int32(1),
                                ordering=fx.AtomicOrdering.AcqRel,
                                syncscope="agent",
                            )
                            if previous + 1 == expected:
                                comm_ops.store_i32_global_system_release(
                                    local
                                    + off_tile_flag
                                    + fx.Int64(tile) * flag_stride
                                    + rank * 4,
                                    generation,
                                )

    return producer
