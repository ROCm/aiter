# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Experimental cluster-local S1F rounds without a global atomic scheduler."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl.cluster import (
    cluster_signal_once_per_wg,
    cluster_wait,
)

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops

from .dispatch_pull import pull_lds_row_pitch
from .s1f_stage1 import (
    INST_PREFETCH, LEGACY_S1F_PREFETCH, _prefetch_cold_code, _trap,
    prefetch_kernel_code,
)


def make_cluster_local_kernel(
    *, name, block, arena_lds, lds_holder, meta_holder, prologue_holder,
    g1_task, copy_tile, off_ready, off_rowmap, wire_row, rank, world,
    experts, tile_m, n_tiles, n1, tile_cap, m_cap, cluster_size,
    producers, cluster_count, launch_n,
):
    """Build rounds with one M tile per producer and cluster-wide consumption.

    Physical cluster ownership is fixed so metadata is available for B prefetch.
    Phase 1 publishes the batch's A tiles; phase 2 finishes consumption before
    any TG advances to the next phase-1 generation. No task or ready RMW is used.
    Cross-device quant readiness retains one leader and release/acquire stores.
    """
    consumers = cluster_size - producers
    subtiles = tile_m // 64
    if not 0 < producers < cluster_size or subtiles not in (1, 2, 4):
        raise ValueError("invalid cluster-local producer or tile geometry")
    if 64 * pull_lds_row_pitch(wire_row) > arena_lds:
        raise ValueError("producer scratch exceeds GEMM LDS allocation")

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def kernel(
        arg_a2: fx.Pointer,
        arg_a1: fx.Pointer,
        arg_w1: fx.Pointer,
        arg_a1_scale: fx.Pointer,
        arg_w1_scale: fx.Pointer,
        arg_a2_scale: fx.Pointer,
        arg_psum: fx.Pointer,
        arena: fx.Int64,
        inp_tok: fx.Int64,
        inp_scale: fx.Int64,
        inp_wts: fx.Int64,
        addr_send: fx.Int64,
        addr_runs: fx.Int64,
        addr_tile_src: fx.Int64,
        addr_ctl: fx.Int64,
        i32_m: fx.Int32,
        inp_cur_tok: fx.Int32,
        f32_swiglu_limit: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // 32
        if const_expr(INST_PREFETCH):
            prefetch_kernel_code(block // 32)
        elif const_expr(LEGACY_S1F_PREFETCH):
            if wave == 0:
                _prefetch_cold_code()
        cluster = fx.Int32(fx.block_idx.x) // launch_n
        member = (
            fx.Int32(fx.block_idx.y) * launch_n
            + fx.Int32(fx.block_idx.x) % launch_n
        )
        is_producer = member < producers
        smem = fx.SharedAllocator(static=False)
        storage = smem.allocate(arena_lds, 128)._ptr
        lds_holder[0] = storage
        global_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        shared_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
        info = fx.inttoptr(global_i32, addr_tile_src)
        psum = fx.recast_iter(global_i32, arg_psum)
        rows = fx.min(i32_m, psum[experts - 1])
        m_tiles = fx.min(m_cap, (rows + tile_m - 1) // tile_m)
        window = cco.CachedWindow(arena)
        local = window.base + rank * window.stride

        if is_producer:
            # Consumers may already prefetch B while source readiness is pending.
            epoch = fx.generic_load(fx.inttoptr(global_i32, local + off_ready + 8))
            all_ready = fx.inttoptr(global_i32, local + off_ready + 4)
            if (cluster == 0) & (member == 0):
                if tid == 0:
                    fx.generic_store(
                        fx.inttoptr(global_i32, local + off_ready), epoch,
                        memory_order=fx.AtomicOrdering.Release, syncscope="",
                    )
                if (tid < world) & (tid != rank):
                    peer = fx.inttoptr(
                        global_i32, window.base + fx.Int64(tid) * window.stride
                        + off_ready,
                    )
                    value = fx.generic_load(
                        peer, memory_order=fx.AtomicOrdering.Acquire, syncscope="",
                    )
                    spins = fx.Int32(0)
                    while (value != epoch) & (spins < 10000000):
                        fx.rocdl.s_sleep(1)
                        value = fx.generic_load(
                            peer, memory_order=fx.AtomicOrdering.Acquire, syncscope="",
                        )
                        spins += 1
                    if value != epoch:
                        _trap()
                fx.gpu.barrier()
                if tid == 0:
                    fx.generic_store(
                        all_ready, epoch, memory_order=fx.AtomicOrdering.Release,
                        syncscope="agent",
                    )
            else:
                if tid == 0:
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

        def bind_task(expert, count, first):
            def wait_for_a(first=first):
                if first != 0:
                    # All waves execute this after the B-owner prologue issues.
                    cluster_wait()
                    comm_ops.fence_agent_acquire()

            meta_holder[0] = (expert, count)
            prologue_holder[0] = wait_for_a

        for batch in range(cluster, m_tiles, cluster_count * producers):
            live_m = fx.min(
                producers, (m_tiles - batch + cluster_count - 1) // cluster_count
            )
            if is_producer:
                m = batch + member * cluster_count
                if (m < m_tiles) & (wave < subtiles) & (
                    m * subtiles + wave < tile_cap
                ):
                    part = m * subtiles + wave
                    count = (info[part] >> 16) & 0xFFFF
                    for chunk in range_constexpr(4):
                        start = part * 64 + chunk * 16
                        if start < part * 64 + count:
                            copy_tile(
                                fx.recast_iter(shared_i32, storage),
                                fx.Int64(fx.ptrtoint(arg_a1)),
                                fx.recast_iter(global_i32, arg_a1_scale),
                                local + off_rowmap, window.base, window.stride,
                                start, part * 64 + count,
                            )
                comm_ops.waitcnt_stores()
                comm_ops.fence_agent_release()
            fx.gpu.barrier()
            cluster_signal_once_per_wg()

            # Consumers get the first tasks, retaining B-load/producer overlap.
            # Producers join computation as soon as their own batch is ready.
            task_first = is_producer.select(consumers + member, member - producers)
            first_wait = fx.Int32(1)
            if is_producer | (task_first >= live_m * n_tiles):
                cluster_wait()
                comm_ops.fence_agent_acquire()
                first_wait = fx.Int32(0)
            for task in range(task_first, live_m * n_tiles, cluster_size):
                m = batch + (task // n_tiles) * cluster_count
                n = task % n_tiles
                metadata = info[m * subtiles]
                count = fx.Int32(0)
                for part in range_constexpr(subtiles):
                    if m * subtiles + part < tile_cap:
                        count += (info[m * subtiles + part] >> 16) & 0xFFFF
                bind_task((metadata >> 8) & 0xFF, count, first_wait)
                g1_task(
                    arg_a2, arg_a1, arg_w1, arg_a1_scale, arg_w1_scale,
                    arg_psum, arg_a1, arg_a2_scale, arg_a1, i32_m,
                    fx.Int32(n1), f32_swiglu_limit,
                    fx.Float32(1.0), fx.Float32(1.0), m, n,
                )
                first_wait = fx.Int32(0)
            # No member can reuse the cluster barrier for a new batch until
            # all members have completed phase 1 and their assigned GEMMs.
            fx.rocdl.tdm_ops.tensor_wait(0)
            comm_ops.waitcnt_stores()
            fx.gpu.barrier()
            cluster_signal_once_per_wg()
            cluster_wait()

    return kernel
