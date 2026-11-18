# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Pull/shuffle producer adapter for the original S1F consumer."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import range_constexpr

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

from .dispatch_pull import make_pull_tile, pull_lds_row_pitch
from .s1f_stage1 import (
    CTL_CLUSTER_TICKETS, CTL_PHEAD, FLAG_M, FLAG_STRIDE, _trap, make_s1f_stage1,
)


def resolve_pull_cluster(
    cluster_m, cluster_n, workers, producers, n_tiles, tile_m, align_m,
):
    """Keep complete initial-consumer clusters, or return a 1x1 fallback."""
    size = cluster_m * cluster_n
    reason = ""
    if cluster_m not in (1, 2, 4) or cluster_n not in (1, 2, 3, 4) or size > 16:
        reason = "unsupported physical cluster geometry"
    elif n_tiles % cluster_n:
        reason = "N tiles do not form complete clusters"
    elif workers % size or (workers - producers) % size:
        reason = "initial consumers would form mixed producer/consumer clusters"
    elif cluster_m > 1 and align_m != tile_m:
        reason = "M alignment holes cannot participate in cross-M multicast"
    elif CTL_CLUSTER_TICKETS + 2 * workers * 4 > CTL_PHEAD:
        reason = "cluster tickets exceed reserved S1F control storage"
    return (1, 1, reason) if reason else (cluster_m, cluster_n, "")


@functools.cache
def compile_s1f_pull(
    *,
    rank,
    world,
    experts,
    max_routes,
    compact_cap,
    model_dim,
    inter_dim,
    wire_row,
    off_wire,
    off_ready,
    off_rowmap,
    off_tile_flag,
    tile_m,
    topk,
    workers,
    producers,
    stage1_act,
    quant_wmma_rep,
    align_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    num_buffers,
    next_stage_prefetch,
    waves_per_tensor_tdm,
    tdm_b_th,
    lds_soa_load_interleave,
    rolled_drain,
    cluster_m=1,
    cluster_n=1,
):
    """Build source-ready/metadata setup and S1F with a pull producer.

    A flag tile belongs to one producer wave. That wave gathers and shuffles
    four M16 chunks, retires its stores, then publishes the original S1F flag.
    No cross-wave wait or fence is needed to publish a tile. Consumer scheduling,
    entry tickets, and drain live in make_s1f_stage1. The 1x1 path retains task
    prefetch hooks; optional consumer clusters use group readiness and claims.
    """
    if tile_m not in (64, 128, 256):
        raise ValueError("shared-consumer pull S1F requires M64/128/256")
    if model_dim % 256 or inter_dim % 128:
        raise ValueError("unsupported pull S1F geometry")
    if align_m % tile_m:
        raise ValueError("plan alignment must be a multiple of GEMM1 tile_m")
    tile_cap = (compact_cap + FLAG_M - 1) // FLAG_M

    @flyc.kernel(name=f"s1f_pull_ready_ep{world}", known_block_size=(128, 1, 1))
    def ready_kernel(
        arena: fx.Int64,
        generation: fx.Tensor,
        psum: fx.Tensor,
        tile_info: fx.Tensor,
        rows: fx.Int32,
    ):
        tid = fx.thread_idx.x
        win = cco.CachedWindow(arena)
        ptr = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        epoch = generation[2]
        if fx.block_idx.x == 0:
            if tid == 0:
                fx.generic_store(
                    fx.inttoptr(ptr, win.base + rank * win.stride + off_ready),
                    epoch,
                    memory_order=fx.AtomicOrdering.Release,
                    syncscope="",
                )
            if tid < world:
                address = win.base + fx.Int64(tid) * win.stride + off_ready
                value = fx.generic_load(
                    fx.inttoptr(ptr, address),
                    memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="",
                )
                spins = fx.Int32(0)
                while (value != epoch) & (spins < 10000000):
                    fx.rocdl.s_sleep(1)
                    value = fx.generic_load(
                        fx.inttoptr(ptr, address),
                        memory_order=fx.AtomicOrdering.Acquire,
                        syncscope="",
                    )
                    spins += 1
                if value != epoch:
                    _trap()
        tile = fx.block_idx.x * 128 + tid
        if tile < tile_cap:
            row = tile * FLAG_M
            info = fx.Int32(0)
            if row < fx.min(rows, psum[experts - 1]):
                lower = fx.Int32(0)
                upper = fx.Int32(experts - 1)
                while lower < upper:
                    middle = (lower + upper) // 2
                    if psum[middle] <= row:
                        lower = middle + 1
                    else:
                        upper = middle
                previous = fx.max(lower - 1, 0)
                start = ((psum[previous] + align_m - 1) // align_m) * align_m
                start = (lower > 0).select(start, 0)
                # The plan may align to GEMM2's larger tile; padding must not
                # become live GEMM1 rows or dereference stale inverse metadata.
                count = (row >= start).select(fx.min(FLAG_M, psum[lower] - row), 0)
                # Pull has one local publisher per flag tile, regardless of
                # the input rows' source ranks. The consumer mask names that
                # publisher, using the original per-rank flag slot.
                source_mask = (count > 0).select(1 << rank, 0)
                info = source_mask | (lower << 8) | (count << 16)
            tile_info[tile] = info

    @flyc.jit
    def ready(
        arena: fx.Int64,
        generation: fx.Tensor,
        psum: fx.Tensor,
        tile_info: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        ready_kernel(arena, generation, psum, tile_info, rows).launch(
            grid=((tile_cap + 127) // 128, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    copy_tile = make_pull_tile(
        model_dim=model_dim,
        wire_row=wire_row,
        max_routes=max_routes,
        topk=topk,
        world=world,
        off_wire=off_wire,
        tiles_per_block=4,
    )

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
        lane = fx.thread_idx.x % 32
        wave = fx.thread_idx.x // 32
        window = cco.CachedWindow(arena)
        local = window.base + rank * window.stride
        global_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        shared_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
        tile_info = fx.inttoptr(global_i32, addr_tile_info)
        lds_rows = fx.recast_iter(shared_i32, storage)
        scale_rows = fx.recast_iter(global_i32, scales)
        output = fx.Int64(fx.ptrtoint(payload))
        ends = fx.recast_iter(global_i32, psum)
        active_rows = fx.min(rows, ends[experts - 1])
        for tile in range(
            slot * 4 + wave,
            (active_rows + FLAG_M - 1) // FLAG_M,
            producers * 4,
        ):
            # Each wave owns the entire flag tile. Its four M16 copies use
            # its private LDS slice; other waves can publish independently.
            count = (tile_info[tile] >> 16) & 0xFFFF
            end = tile * FLAG_M + count
            for chunk in range_constexpr(FLAG_M // 16):
                row0 = tile * FLAG_M + chunk * 16
                if row0 < end:
                    copy_tile(
                        lds_rows,
                        output,
                        scale_rows,
                        local + off_rowmap,
                        window.base,
                        window.stride,
                        row0,
                        end,
                    )
            # Match the push publication contract: retire this wave's data
            # stores, then publish a relaxed system-scope generation flag.
            # No WG barrier: all data covered by this flag belongs to us.
            comm_ops.waitcnt_stores()
            if (lane == 0) & (count > 0):
                comm_ops.store_i32_global_system_monotonic(
                    local + off_tile_flag + fx.Int64(tile) * FLAG_STRIDE + rank * 4,
                    generation,
                )

    fused = make_s1f_stage1(
        rank=rank,
        npes=world,
        experts_per_rank=experts,
        topk=topk,
        payload_bytes=model_dim // 2,
        scale_bytes=model_dim // 32,
        row_stride=wire_row,
        compact_cap=compact_cap,
        max_tok_per_rank=max_routes // topk,
        max_routes=max_routes,
        off_out_tok=0,
        off_ep_rowmap=off_rowmap,
        off_tile_flag=off_tile_flag,
        model_dim=model_dim,
        inter_dim=inter_dim,
        stage1_act=stage1_act,
        quant_wmma_rep=quant_wmma_rep,
        workers=workers,
        producers=producers,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        next_stage_prefetch=next_stage_prefetch,
        waves_per_tensor_tdm=waves_per_tensor_tdm,
        tdm_b_th=tdm_b_th,
        lds_soa_load_interleave=lds_soa_load_interleave,
        rolled_drain=rolled_drain,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        ctas_per_cu=1,
        pull_producer=producer,
        pull_lds_bytes=64 * pull_lds_row_pitch(wire_row),
    )
    return ready, fused
