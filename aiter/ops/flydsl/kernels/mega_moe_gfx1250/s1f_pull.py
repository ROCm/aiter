# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc.

"""Pull/shuffle producer adapter for the original S1F consumer."""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

from . import ready_queue as rq
from .dispatch_pull import make_pull_tile, pull_lds_row_pitch
from .s1f_producer_group import make_group_producer, validate_producer_group
from .s1f_stage1 import (
    FLAG_M, FLAG_STRIDE, _trap, make_s1f_stage1,
)


def pull_affinity_geometry(workers, producers):
    """Reduce the global role ratio into complete physical clusters."""
    if not 0 < producers < workers:
        raise ValueError("affinity requires 0 < producers < workers")
    clusters = math.gcd(workers, producers)
    size = workers // clusters
    if size not in (2, 4, 8, 16):
        raise ValueError("affinity role ratio needs a 2/4/8/16-TG cluster")
    return size, producers // clusters


def resolve_pull_cluster(
    cluster_m, cluster_n, workers, producers, n_tiles, tile_m, align_m,
):
    """Keep complete initial-consumer clusters, or return a 1x1 fallback."""
    size = cluster_n
    reason = ""
    if cluster_m not in (1, 2, 4) or cluster_n not in (1, 2, 3, 4):
        reason = "only 1xN leader clusters are supported"
    elif n_tiles % cluster_n:
        reason = "N tiles do not form complete clusters"
    elif workers % size or (workers - producers) % size:
        reason = "initial consumers would form mixed producer/consumer clusters"
    if reason:
        return 1, 1, reason
    return 1, cluster_n, ("M clustering disabled; using 1xN" if cluster_m != 1 else "")


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
    affinity_size=1,
    affinity_producers=0,
    ready_first=False,
    ready_queue=False,
    cluster_local=False,
    producer_group=1,
    tile_ready=False,
    unique_pull=False,
    off_unique_routes=0,
    off_unique_counts=0,
    packed_input=False,
    off_packed=0,
    src_rank_shuffle=False,
    packed_wave_load=False,
    packed_load_k_bytes=1024,
):
    """Build source-ready/metadata setup and S1F with a pull producer.

    With tile_ready, one TG retires all four M64 pieces and publishes one
    M256 flag after a workgroup barrier. Otherwise each wave independently
    publishes its original M64 flag. Consumer scheduling,
    entry tickets, and drain live in make_s1f_stage1. Both clustered and solo
    consumers retain the same readiness and task-prefetch hooks.
    """
    validate_producer_group(producer_group, workers, producers, tile_m)
    if tile_ready and (
        tile_m != 256 or ready_queue or cluster_local or producer_group != 1
    ):
        raise ValueError("M256 ready requires the one-TG pull producer")
    if producer_group == 4 and (
        ready_queue or cluster_local or cluster_m != 1 or cluster_n != 1
        or affinity_size != 1
    ):
        raise ValueError("four-TG producers require independent GEMM consumers")
    if tile_m not in (64, 128, 256):
        raise ValueError("shared-consumer pull S1F requires M64/128/256")
    if model_dim % 256 or inter_dim % 128:
        raise ValueError("unsupported pull S1F geometry")
    if align_m % tile_m:
        raise ValueError("plan alignment must be a multiple of GEMM1 tile_m")
    if cluster_local and (
        not ready_first or ready_queue or affinity_size <= 1
        or align_m != tile_m or cluster_m != 1 or cluster_n != 1
    ):
        raise ValueError("cluster-local S1F requires aligned local-ready affinity")
    if ready_queue and (not ready_first or align_m != tile_m):
        raise ValueError("ready queue requires local-ready and aligned GEMM tiles")
    if unique_pull and (packed_input or ready_first or tile_ready or producer_group != 1):
        raise ValueError("unique pull requires ordinary external-ready M64 flags")
    tile_cap = (compact_cap + FLAG_M - 1) // FLAG_M
    affinity_clusters = producers // 4 if producer_group == 4 else workers // affinity_size
    queue_affinity = affinity_size > 1 or producer_group == 4
    ready_extent = max(tile_cap, affinity_clusters if queue_affinity else 0)
    ready_name = f"s1f_pull_ready_ep{world}"
    if src_rank_shuffle:
        if not packed_input or ready_first or tile_m != 256:
            raise ValueError("source shuffle requires external-ready packed M256")
        ready_name += "_src256"
    if affinity_size > 1:
        ready_name += f"_affinity{affinity_size}"
    if producer_group == 4:
        ready_name += "_producer4"

    @flyc.kernel(name=ready_name, known_block_size=(128, 1, 1))
    def ready_kernel(
        arena: fx.Int64,
        generation: fx.Tensor,
        psum: fx.Tensor,
        tile_info: fx.Tensor,
        rows: fx.Int32,
        queue_heads: fx.Pointer,
    ):
        tid = fx.thread_idx.x
        if const_expr(queue_affinity):
            index = fx.block_idx.x * 128 + tid
            if index < affinity_clusters:
                fx.recast_iter(fx.Int32, queue_heads)[index * 32] = 0
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
        if const_expr(unique_pull):
            tile = fx.block_idx.x * 128 + tid
            if tile < tile_cap:
                pointer = fx.inttoptr(
                    ptr, win.base + rank * win.stride + off_unique_counts
                )
                fx.generic_store(fx.add_offset(pointer, tile), fx.Int32(0))
        if const_expr(not src_rank_shuffle):
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
        queue_heads: fx.Int64,
        stream: fx.Stream,
    ):
        ready_kernel(
            arena, generation, psum, tile_info, rows,
            fx.inttoptr(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4),
                queue_heads,
            ),
        ).launch(
            grid=((ready_extent + 127) // 128, 1, 1),
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

    if packed_input:
        from .packed_copy import make_packed_copy
        copy_tile = make_packed_copy(
            model_dim=model_dim, wire_row=wire_row, max_routes=max_routes,
            world=world, rank=rank, cap=compact_cap, off_packed=off_packed,
        )

    producer_lds_bytes = 64 * pull_lds_row_pitch(wire_row)
    producer_lds_gap = None
    if packed_wave_load:
        if not src_rank_shuffle or not packed_input or ready_first:
            raise ValueError("wave loads require source-shuffled external-ready pull")
        from .packed_wave_copy import make_packed_wave_copy, packed_wave_lds_bytes
        from .s1f_stage1 import PLANAR_LDS

        producer_lds_bytes = packed_wave_lds_bytes(model_dim, packed_load_k_bytes)
        reserve_gap = bool(
            num_buffers >= 4
            and (PLANAR_LDS or lds_soa_load_interleave)
            and producer_lds_bytes > 196352
        )
        if reserve_gap:
            wave_bytes = producer_lds_bytes // 4
            producer_lds_gap = (2 * wave_bytes, 3 * wave_bytes)
            producer_lds_bytes += wave_bytes
        if producer_lds_bytes > 327680:
            raise ValueError("packed wave load exceeds320KiB LDS; reduce K bytes")
        copy_wave = make_packed_wave_copy(
            model_dim=model_dim,
            max_routes=max_routes,
            rank=rank,
            cap=compact_cap,
            off_packed=off_packed,
            k_bytes=packed_load_k_bytes,
            control_gap=reserve_gap,
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
        first = slot * 4 + wave
        stride = producers * 4
        live_wave = fx.Int32(1)
        if const_expr(affinity_size > 1):
            # Each producer owns complete M tiles from its physical cluster.
            # Different producers may prepare subsequent M tiles concurrently.
            cluster_id = slot // affinity_producers
            member = slot % affinity_producers
            first = (cluster_id + member * affinity_clusters) * (tile_m // FLAG_M) + wave
            stride = affinity_clusters * affinity_producers * (tile_m // FLAG_M)
            live_wave = fx.Int32(wave < tile_m // FLAG_M)
        tile_count = (active_rows + FLAG_M - 1) // FLAG_M
        loop_first = live_wave.select(first, tile_count)
        loop_end = tile_count
        loop_stride = stride
        if const_expr(tile_ready):
            # Uniform trip counts keep all four waves at the publication
            # barrier, including empty experts and a partial final M256.
            loop_first = first // 4
            loop_end = (tile_count + 3) // 4
            loop_stride = stride // 4
        for item in range(loop_first, loop_end, loop_stride):
            tile = item * 4 + wave if const_expr(tile_ready) else item
            count = fx.Int32(0)
            if tile < tile_count:
                count = (tile_info[tile] >> 16) & 0xFFFF
            end = tile * FLAG_M + count
            if const_expr(packed_wave_load):
                if count > 0:
                    copy_wave(
                        lds_rows,
                        output,
                        scale_rows,
                        local + off_rowmap,
                        window.base,
                        window.stride,
                        tile * FLAG_M,
                        end,
                    )
            else:
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
            comm_ops.waitcnt_stores()
            if const_expr(tile_ready):
                # Every writer releases before rendezvous. The leader acquires
                # their completion before publishing the original SYS flag.
                comm_ops.fence_agent_release()
                fx.gpu.barrier()
                comm_ops.fence_agent_acquire()
                if fx.thread_idx.x == 0:
                    comm_ops.store_i32_global_system_release(
                        local + off_tile_flag + fx.Int64(item * 4)
                        * FLAG_STRIDE + rank * 4,
                        generation,
                    )
            else:
                if (lane == 0) & (count > 0):
                    comm_ops.store_i32_global_system_monotonic(
                        local + off_tile_flag + fx.Int64(tile)
                        * FLAG_STRIDE + rank * 4,
                        generation,
                    )

    @flyc.jit
    def producer_ready_first(
        storage: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        psum: fx.Pointer,
        arena: fx.Int64,
        addr_tile_info: fx.Int64,
        rows: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int32,
        addr_queue: fx.Int64 = 0,
    ):
        lane = fx.thread_idx.x % 32
        wave = fx.thread_idx.x // 32
        window = cco.CachedWindow(arena)
        local = window.base + rank * window.stride
        global_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        shared_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
        info = fx.inttoptr(global_i32, addr_tile_info)
        lds_rows = fx.recast_iter(shared_i32, storage)
        scale_rows = fx.recast_iter(global_i32, scales)
        output = fx.Int64(fx.ptrtoint(payload))
        ends = fx.recast_iter(global_i32, psum)
        tile_count = (fx.min(rows, ends[experts - 1]) + FLAG_M - 1) // FLAG_M
        first = slot * 4 + wave
        stride = producers * 4
        if const_expr(affinity_size > 1):
            cluster_id = slot // affinity_producers
            member = slot % affinity_producers
            first = (
                (cluster_id + member * affinity_clusters) * (tile_m // FLAG_M)
                + wave
            )
            first = (wave < tile_m // FLAG_M).select(first, tile_count)
            stride = affinity_clusters * affinity_producers * (tile_m // FLAG_M)

        loop_first, loop_end, loop_stride = first, tile_count, stride
        if const_expr(tile_ready):
            loop_first = first // 4
            loop_end = (tile_count + 3) // 4
            loop_stride = stride // 4

        # This kernel follows the completed local plan/quant launch, so its
        # coordinator may now publish source readiness without a setup kernel.
        if (slot == 0) & (fx.thread_idx.x == 0):
            epoch = fx.generic_load(
                fx.inttoptr(global_i32, local + off_ready + 8)
            )
            fx.generic_store(
                fx.inttoptr(global_i32, local + off_ready), epoch,
                memory_order=fx.AtomicOrdering.Release, syncscope="",
            )

        epoch = fx.generic_load(
            fx.inttoptr(global_i32, local + off_ready + 8)
        )
        all_ready = fx.inttoptr(global_i32, local + off_ready + 4)
        # The initial producer leader collects readiness immediately, then
        # becomes an ordinary producer. Other TGs can copy local rows meanwhile.
        if slot == 0:
            if (fx.thread_idx.x < world) & (fx.thread_idx.x != rank):
                peer = fx.inttoptr(
                    global_i32,
                    window.base + fx.Int64(fx.thread_idx.x) * window.stride
                    + off_ready,
                )
                value = fx.generic_load(
                    peer, memory_order=fx.AtomicOrdering.Acquire,
                    syncscope="",
                )
                spins = fx.Int32(0)
                while (value != epoch) & (spins < 10000000):
                    fx.rocdl.s_sleep(1)
                    value = fx.generic_load(
                        peer, memory_order=fx.AtomicOrdering.Acquire,
                        syncscope="",
                    )
                    spins += 1
                if value != epoch:
                    _trap()
            fx.gpu.barrier()
            if fx.thread_idx.x == 0:
                fx.generic_store(
                    all_ready, epoch,
                    memory_order=fx.AtomicOrdering.Release,
                    syncscope="agent",
                )

        # Share one gather/shuffle body. The leader is already synchronized and
        # copies all its chunks in phase 1; followers copy local chunks first.
        first_phase = (slot == 0).select(1, 0)
        for phase in range(first_phase, fx.Int32(2)):
            if (phase == 1) & (slot != 0):
                # Other TGs only poll local memory. The second WG barrier
                # propagates this acquire to every wave before remote TDM.
                if fx.thread_idx.x == 0:
                    value = fx.generic_load(
                        all_ready, memory_order=fx.AtomicOrdering.Acquire,
                        syncscope="agent",
                    )
                    spins = fx.Int32(0)
                    while (value != epoch) & (spins < 10000000):
                        fx.rocdl.s_sleep(1)
                        value = fx.generic_load(
                            all_ready, memory_order=fx.AtomicOrdering.Acquire,
                            syncscope="agent",
                        )
                        spins += 1
                    if value != epoch:
                        _trap()
                fx.gpu.barrier()
            for item in range(loop_first, loop_end, loop_stride):
                tile = item * 4 + wave if const_expr(tile_ready) else item
                count = fx.Int32(0)
                bounds = fx.Int32(0)
                if tile < tile_count:
                    count = (info[tile] >> 16) & 0xFFFF
                    bounds = info[tile_cap + tile]
                local_first = (slot == 0).select(4, bounds & 0xFFFF)
                local_end = (slot == 0).select(0, (bounds >> 16) & 0xFFFF)
                all_local = (local_first == 0) & (
                    local_end == (count + 15) // 16
                )
                first_chunk = (phase == 0).select(local_first, 0)
                end_chunk = (phase == 0).select(local_end, (count + 15) // 16)
                # Each copy retires its TDM before returning; unrolling cannot
                # overlap chunks here. Keep one descriptor/shuffle body.
                for chunk in range(first_chunk, end_chunk):
                    own = (chunk >= local_first) & (chunk < local_end)
                    if own == (phase == 0):
                        copy_tile(
                            lds_rows, output, scale_rows, local + off_rowmap,
                            window.base, window.stride, tile * FLAG_M + chunk * 16,
                            tile * FLAG_M + count,
                        )
                # Local-only tiles publish early. Mixed/remote tiles publish
                # once, after their remaining chunks have been copied.
                if const_expr(tile_ready):
                    # Uniform M256 loops include empty waves in the rendezvous.
                    # Phase 0 may publish only if every live M64 is local;
                    # otherwise phase 1 completes it after the leader's ready.
                    comm_ops.waitcnt_stores()
                    comm_ops.fence_agent_release()
                    fx.gpu.barrier()
                    comm_ops.fence_agent_acquire()
                    if fx.thread_idx.x == 0:
                        tile_local = fx.Int32(slot != 0)
                        for part in range_constexpr(4):
                            sub = item * 4 + part
                            if sub < tile_count:
                                live = (info[sub] >> 16) & 0xFFFF
                                sub_bounds = info[tile_cap + sub]
                                local_only = ((sub_bounds & 0xFFFF) == 0) & (
                                    (sub_bounds >> 16) == (live + 15) // 16
                                )
                                tile_local &= fx.Int32((live == 0) | local_only)
                        if tile_local == fx.Int32(phase == 0):
                            comm_ops.store_i32_global_system_release(
                                local + off_tile_flag + fx.Int64(item * 4)
                                * FLAG_STRIDE + rank * 4,
                                generation,
                            )
                elif (count > 0) & (all_local == (phase == 0)):
                    comm_ops.waitcnt_stores()
                    if lane == 0:
                        comm_ops.store_i32_global_system_monotonic(
                            local + off_tile_flag + fx.Int64(tile) * FLAG_STRIDE
                            + rank * 4,
                            generation,
                        )
                        if const_expr(ready_queue):
                            # Each live M64 publishes exactly once. AcqRel RMWs
                            # collect every wave's completed payload/scale stores
                            # before the last wave release-publishes the A tile.
                            m = tile // (tile_m // FLAG_M)
                            needed = fx.Int32(0)
                            for part in range_constexpr(tile_m // FLAG_M):
                                sub = m * (tile_m // FLAG_M) + part
                                if sub < tile_cap:
                                    needed += fx.Int32((info[sub] >> 16) != 0)
                            base = addr_queue + rq.AFFINITY_WORDS * 4
                            count_ptr = fx.inttoptr(
                                global_i32, base + (rq.COUNTS + fx.Int64(m)) * 4
                            )
                            previous = fx.atomic_add(
                                count_ptr, fx.Int32(1), syncscope="agent",
                                ordering=fx.AtomicOrdering.AcqRel,
                            )
                            if previous + 1 == needed:
                                position = fx.atomic_add(
                                    fx.inttoptr(global_i32, base + rq.TAIL * 4),
                                    fx.Int32(1), syncscope="agent",
                                )
                                if position >= tile_cap:
                                    _trap()
                                entry = fx.inttoptr(
                                    global_i32, base + (
                                        rq.COUNTS + tile_cap + fx.Int64(position)
                                    ) * 4,
                                )
                                fx.generic_store(
                                    entry, m + 1,
                                    memory_order=fx.AtomicOrdering.Release,
                                    syncscope="agent",
                                )

    selected_producer = producer_ready_first if ready_first else producer
    if unique_pull:
        from .unique_pull import make_unique_producer
        selected_producer = make_unique_producer(
            rank=rank, world=world, model_dim=model_dim, wire_row=wire_row,
            max_routes=max_routes, topk=topk, producers=producers,
            off_wire=off_wire, off_routes=off_unique_routes,
            off_counts=off_unique_counts, off_tile_flag=off_tile_flag,
            flag_stride=FLAG_STRIDE,
        )
    if producer_group == 4:
        selected_producer = make_group_producer(
            copy_tile=copy_tile, rank=rank, world=world, experts=experts,
            producers=producers, tile_cap=tile_cap, off_ready=off_ready,
            off_rowmap=off_rowmap, off_tile_flag=off_tile_flag,
            ready_first=ready_first,
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
        affinity_size=affinity_size,
        affinity_producers=affinity_producers,
        ctas_per_cu=1,
        pull_producer=selected_producer,
        pull_unique=unique_pull,
        pull_producer_group=producer_group,
        pull_tile_ready=tile_ready,
        pull_ready_queue=ready_queue,
        pull_cluster_local=cluster_local,
        pull_cluster_config=(
            (copy_tile, off_ready, off_rowmap, wire_row) if cluster_local else None
        ),
        pull_lds_bytes=producer_lds_bytes,
        pull_lds_control_gap=producer_lds_gap,
    )
    return (None if ready_first else ready), fused
