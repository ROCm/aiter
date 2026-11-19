# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""Four producer TGs cooperate on A256, retaining independent GEMM consumers."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl.cluster import cluster_signal_once_per_wg, cluster_wait

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops


def validate_producer_group(
    group: int, workers: int, producers: int, tile_m: int,
) -> None:
    """Reject partial physical producer clusters before compiling or launching.

    Args:
        group: Number of TGs cooperating on one A tile, either 1 or 4.
        workers: Total resident persistent TGs.
        producers: TGs initially assigned to dispatch.
        tile_m: GEMM1 tile height.

    Raises:
        ValueError: The group size or cooperative launch geometry is invalid.
    """
    if group not in (1, 4):
        raise ValueError("AITER_MOE_S1F_PRODUCER_GROUP must be 1 or 4")
    if group == 4:
        if producers % 4 or not 0 < producers < workers:
            raise ValueError(
                "four-TG producers require 0 < producers < workers "
                "and producers % 4 == 0"
            )
        if workers % 4:
            raise ValueError("four-TG producers require workers % 4 == 0")
        if tile_m != 256:
            raise ValueError("four-TG producers currently require GEMM1 tile_m=256")


def make_group_producer(
    *, copy_tile, rank, world, experts, producers, tile_cap, off_ready,
    off_rowmap, off_tile_flag, ready_first,
):
    """Return an adapter with one M64 per TG and one M16 per wave.

    Cluster g owns A256 indices g + iteration * (producers // 4). All four
    members traverse identical loops, including empty/padded work. Only TG0
    publishes the four original M64 generation flags, after every member's
    payload/scale stores complete and the cluster barrier joins them.
    """
    from .s1f_stage1 import FLAG_M, FLAG_STRIDE, _trap

    groups = producers // 4

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
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // 32
        group = slot // 4
        member = slot % 4
        window = cco.CachedWindow(arena)
        local = window.base + rank * window.stride
        global_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
        shared_i32 = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4)
        info = fx.inttoptr(global_i32, addr_tile_info)
        ends = fx.recast_iter(global_i32, psum)
        lds_rows = fx.recast_iter(shared_i32, storage)
        scale_rows = fx.recast_iter(global_i32, scales)
        output = fx.Int64(fx.ptrtoint(payload))
        m_tiles = (fx.min(rows, ends[experts - 1]) + 255) // 256
        first_phase = fx.Int32(1)
        if const_expr(ready_first):
            epoch = fx.generic_load(
                fx.inttoptr(global_i32, local + off_ready + 8)
            )
            all_ready = fx.inttoptr(global_i32, local + off_ready + 4)
            if (slot == 0) & (tid == 0):
                fx.generic_store(
                    fx.inttoptr(global_i32, local + off_ready), epoch,
                    memory_order=fx.AtomicOrdering.Release, syncscope="",
                )
            if slot == 0:
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
            # All members of the coordinator's cluster skip phase0 together.
            first_phase = (group == 0).select(1, 0)

        for phase in range(first_phase, fx.Int32(2)):
            if const_expr(ready_first):
                if phase == 1:
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
            for m in range(group, m_tiles, groups):
                tile = m * 4 + member
                count = fx.Int32(0)
                local_first = fx.Int32(0)
                local_end = fx.Int32(0)
                if tile < tile_cap:
                    count = (info[tile] >> 16) & 0xFFFF
                    if const_expr(ready_first):
                        bounds = info[tile_cap + tile]
                        local_first = (group == 0).select(4, bounds & 0xFFFF)
                        local_end = (group == 0).select(0, (bounds >> 16) & 0xFFFF)
                own = (wave >= local_first) & (wave < local_end)
                if (wave * 16 < count) & (own == (phase == 0)):
                    copy_tile(
                        lds_rows, output, scale_rows, local + off_rowmap,
                        window.base, window.stride, tile * FLAG_M + wave * 16,
                        tile * FLAG_M + count,
                    )

                # Every wave retires its own stores before its TG signals.
                # Empty members participate: no member-local loop/exit may
                # bypass this cluster-wide generation.
                comm_ops.waitcnt_stores()
                comm_ops.fence_agent_release()
                fx.gpu.barrier()
                cluster_signal_once_per_wg()
                cluster_wait()
                comm_ops.fence_agent_acquire()
                if member == 0:
                    all_local = fx.Int32(0)
                    if const_expr(ready_first):
                        all_local = fx.Int32(group != 0)
                        for part in range_constexpr(4):
                            sub = m * 4 + part
                            if sub < tile_cap:
                                live = (info[sub] >> 16) & 0xFFFF
                                bounds = info[tile_cap + sub]
                                local_only = ((bounds & 0xFFFF) == 0) & (
                                    ((bounds >> 16) & 0xFFFF) == (live + 15) // 16
                                )
                                all_local &= fx.Int32((live == 0) | local_only)
                    if all_local == fx.Int32(phase == 0):
                        if tid < 4:
                            sub = m * 4 + tid
                            if sub < tile_cap:
                                if ((info[sub] >> 16) & 0xFFFF) > 0:
                                    comm_ops.store_i32_global_system_monotonic(
                                        local + off_tile_flag + fx.Int64(sub)
                                        * FLAG_STRIDE + rank * 4,
                                        generation,
                                    )

    return producer
