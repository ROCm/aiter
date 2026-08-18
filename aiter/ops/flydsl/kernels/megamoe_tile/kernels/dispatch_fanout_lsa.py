# SPDX-License-Identifier: MIT
"""Preplanned node-local aligned-record fan-out over CCO LSA pointers."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops

from ..cco import lsa_ptr
from ..runtime import HierCcoArenaLayout


FANOUT_RECORD_BYTES = 2048  # backward-compatible default


@dataclass(frozen=True)
class DispatchFanoutModule:
    launch: object
    launch_broadcast: object
    capacity: int
    node_ranks: int
    record_bytes: int

    def broadcast_grid_blocks(self, record_count: int) -> int:
        record_count = int(record_count)
        if record_count <= 0:
            raise ValueError("record_count must be positive")
        return 2 * self.node_ranks * record_count

    def validate_broadcast_plan(
        self,
        record_count: int,
        local_slot_base: int,
        remote_slot_base: int,
    ) -> None:
        """Validate the two disjoint contiguous destination-slot ranges."""

        record_count = int(record_count)
        local_slot_base = int(local_slot_base)
        remote_slot_base = int(remote_slot_base)
        if record_count <= 0:
            raise ValueError("record_count must be positive")
        local_end = local_slot_base + record_count
        remote_end = remote_slot_base + record_count
        if local_slot_base < 0 or local_end > self.capacity:
            raise ValueError("local broadcast slot range exceeds capacity")
        if remote_slot_base < 0 or remote_end > self.capacity:
            raise ValueError("remote broadcast slot range exceeds capacity")
        if max(local_slot_base, remote_slot_base) < min(local_end, remote_end):
            raise ValueError("local and remote broadcast slot ranges overlap")


def build_dispatch_fanout_lsa(
    layout: HierCcoArenaLayout, *, node_ranks: int = 8
) -> DispatchFanoutModule:
    """Build one-entry/one-record fan-out with host-preallocated slots."""

    if node_ranks <= 0 or node_ranks > 64:
        raise ValueError("node_ranks must be in [1,64]")
    inbox = layout.region("fanout_inbox")
    ready = layout.region("fanout_ready")
    count = layout.region("fanout_count")
    capacity = inbox.shape[0]
    record_bytes = inbox.shape[1]
    if inbox.shape != (capacity, record_bytes) or record_bytes % 16:
        raise ValueError("fanout inbox must use 16-byte-aligned records")
    if ready.shape != (2, capacity) or ready.dtype != torch.int64:
        raise ValueError("fanout ready layout mismatch")

    inbox_base = inbox.offset
    ready_base = ready.offset
    ready_parity_bytes = ready.nbytes // 2
    count_base = count.offset
    count_parity_bytes = count.nbytes // 2
    record_dwords = record_bytes // 4
    tag = f"n{node_ranks}_c{capacity}_r{record_bytes}"

    @flyc.kernel(
        name=f"megamoe_dispatch_fanout_lsa_{tag}",
        known_block_size=[256, 1, 1],
    )
    def fanout_kernel(
        records: fx.Int64,
        window: fx.Int64,
        dest_lsa_rank: fx.Int64,
        dest_slot: fx.Int64,
        valid_plan: fx.Int64,
        plan_count: fx.Int32,
        generation: fx.Int64,
        parity: fx.Int32,
        error_flag: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        entry = fx.Int32(gpu.block_id("x"))
        if entry < plan_count:
            records_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
            rank_rsrc = buffer_ops.create_buffer_resource_from_addr(dest_lsa_rank)
            slot_rsrc = buffer_ops.create_buffer_resource_from_addr(dest_slot)
            valid_rsrc = buffer_ops.create_buffer_resource_from_addr(valid_plan)
            dest = buffer_ops.buffer_load(rank_rsrc, entry, vec_width=1, dtype=T.i32)
            slot = buffer_ops.buffer_load(slot_rsrc, entry, vec_width=1, dtype=T.i32)
            planned = buffer_ops.buffer_load(
                valid_rsrc, entry, vec_width=1, dtype=T.i32
            ) != fx.Int32(0)
            bounds = (dest >= fx.Int32(0)) & (dest < fx.Int32(node_ranks)) & (
                slot >= fx.Int32(0)
            ) & (slot < fx.Int32(capacity))
            active = planned & bounds
            safe_dest = bounds.select(dest, fx.Int32(0))
            safe_slot = bounds.select(slot, fx.Int32(0))
            remote = lsa_ptr(
                window,
                safe_dest,
                fx.Int64(inbox_base)
                + fx.Int64(safe_slot) * fx.Int64(record_bytes),
            )
            remote_rsrc = buffer_ops.create_buffer_resource_from_addr(remote)
            src_base = entry * fx.Int32(record_dwords)
            for dword in range(
                tx * fx.Int32(4), record_dwords, fx.Int32(1024)
            ):
                if active:
                    value = buffer_ops.buffer_load(
                        records_rsrc, src_base + dword, vec_width=4, dtype=T.i32
                    )
                    buffer_ops.buffer_store(value, remote_rsrc, dword)

            fx.rocdl.s_waitcnt(0)
            fx.gpu.barrier()
            if tx == fx.Int32(0):
                if planned:
                    if bounds:
                        comm_ops.fence_system_release()
                        ready_offset = (
                            fx.Int64(ready_base)
                            + fx.Int64(parity) * fx.Int64(ready_parity_bytes)
                            + fx.Int64(safe_slot) * fx.Int64(8)
                        )
                        remote_ready = lsa_ptr(window, safe_dest, ready_offset)
                        comm_ops.store_i64_global_system(remote_ready, generation)
                        count_offset = (
                            fx.Int64(count_base)
                            + fx.Int64(parity) * fx.Int64(count_parity_bytes)
                        )
                        remote_count = lsa_ptr(window, safe_dest, count_offset)
                        comm_ops.atomic_add_system(remote_count, fx.Int32(1))
                    else:
                        comm_ops.atomic_add_system(error_flag, fx.Int32(1))

    @flyc.kernel(
        name=f"megamoe_dispatch_fanout_broadcast_lsa_{tag}",
        known_block_size=[256, 1, 1],
    )
    def fanout_broadcast_kernel(
        local_records: fx.Int64,
        remote_records: fx.Int64,
        window: fx.Int64,
        record_count: fx.Int32,
        local_slot_base: fx.Int32,
        remote_slot_base: fx.Int32,
        generation: fx.Int64,
        parity: fx.Int32,
        error_flag: fx.Int64,
    ):
        """Broadcast two contiguous record slabs to every node-local rank."""

        tx = fx.Int32(gpu.thread_id("x"))
        flat = fx.Int32(gpu.block_id("x"))
        safe_count = (record_count > fx.Int32(0)).select(
            record_count, fx.Int32(1)
        )
        blocks_per_side = fx.Int32(node_ranks) * safe_count
        side = flat // blocks_per_side
        side_flat = flat - side * blocks_per_side
        dest = side_flat // safe_count
        record = side_flat - dest * safe_count
        is_local = side == fx.Int32(0)
        records = is_local.select(local_records, remote_records)
        slot_base = is_local.select(local_slot_base, remote_slot_base)
        slot = slot_base + record
        bounds = (
            (record_count > fx.Int32(0))
            & (side >= fx.Int32(0))
            & (side < fx.Int32(2))
            & (dest >= fx.Int32(0))
            & (dest < fx.Int32(node_ranks))
            & (record >= fx.Int32(0))
            & (record < record_count)
            & (slot >= fx.Int32(0))
            & (slot < fx.Int32(capacity))
            & (parity >= fx.Int32(0))
            & (parity < fx.Int32(2))
        )
        safe_dest = bounds.select(dest, fx.Int32(0))
        safe_slot = bounds.select(slot, fx.Int32(0))
        records_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
        remote = lsa_ptr(
            window,
            safe_dest,
            fx.Int64(inbox_base)
            + fx.Int64(safe_slot) * fx.Int64(record_bytes),
        )
        remote_rsrc = buffer_ops.create_buffer_resource_from_addr(remote)
        src_base = record * fx.Int32(record_dwords)
        for dword in range(
            tx * fx.Int32(4), record_dwords, fx.Int32(1024)
        ):
            if bounds:
                value = buffer_ops.buffer_load(
                    records_rsrc, src_base + dword, vec_width=4, dtype=T.i32
                )
                buffer_ops.buffer_store(value, remote_rsrc, dword)

        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            if bounds:
                comm_ops.fence_system_release()
                ready_offset = (
                    fx.Int64(ready_base)
                    + fx.Int64(parity) * fx.Int64(ready_parity_bytes)
                    + fx.Int64(safe_slot) * fx.Int64(8)
                )
                remote_ready = lsa_ptr(window, safe_dest, ready_offset)
                comm_ops.store_i64_global_system(remote_ready, generation)
                count_offset = (
                    fx.Int64(count_base)
                    + fx.Int64(parity) * fx.Int64(count_parity_bytes)
                )
                remote_count = lsa_ptr(window, safe_dest, count_offset)
                comm_ops.atomic_add_system(remote_count, fx.Int32(1))
            else:
                comm_ops.atomic_add_system(error_flag, fx.Int32(1))

    @flyc.jit
    def launch(
        records: fx.Int64,
        window: fx.Int64,
        dest_lsa_rank: fx.Int64,
        dest_slot: fx.Int64,
        valid_plan: fx.Int64,
        plan_count: fx.Int32,
        generation: fx.Int64,
        parity: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        fanout_kernel(
            records,
            window,
            dest_lsa_rank,
            dest_slot,
            valid_plan,
            plan_count,
            generation,
            parity,
            error_flag,
        ).launch(grid=(plan_count, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_broadcast(
        local_records: fx.Int64,
        remote_records: fx.Int64,
        window: fx.Int64,
        record_count: fx.Int32,
        local_slot_base: fx.Int32,
        remote_slot_base: fx.Int32,
        generation: fx.Int64,
        parity: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        fanout_broadcast_kernel(
            local_records,
            remote_records,
            window,
            record_count,
            local_slot_base,
            remote_slot_base,
            generation,
            parity,
            error_flag,
        ).launch(
            grid=(fx.Int32(2 * node_ranks) * record_count, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    launch.capacity = capacity
    launch.node_ranks = node_ranks
    launch.record_bytes = record_bytes
    launch_broadcast.capacity = capacity
    launch_broadcast.node_ranks = node_ranks
    launch_broadcast.record_bytes = record_bytes
    launch_broadcast.kernel_name = (
        f"megamoe_dispatch_fanout_broadcast_lsa_{tag}"
    )
    launch_broadcast.sides = 2
    launch_broadcast.grid_blocks_per_record = 2 * node_ranks
    launch_broadcast.requires_disjoint_slot_ranges = True
    return DispatchFanoutModule(
        launch=launch,
        launch_broadcast=launch_broadcast,
        capacity=capacity,
        node_ranks=node_ranks,
        record_bytes=record_bytes,
    )
