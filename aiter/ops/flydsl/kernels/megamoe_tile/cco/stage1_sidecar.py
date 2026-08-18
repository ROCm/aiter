# SPDX-License-Identifier: MIT
"""Production-oriented CCO Stage-1 communication sidecar skeleton.

This code object deliberately contains no MFMA. It moves a bounded batch of
already-packed dispatch segments, publishes a trailing per-QP generation, and
retains each flushAsync request until the remote credit retires the ring slot.

The receive publisher only exposes the readiness seam consumed by the H1
compute kernel. Real token packing, destination planning, expert sort and
intra-node fan-out remain upstream work and are not emulated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops

from ..runtime import HierCcoArenaLayout
from .ops import (
    TEAM_RAIL,
    TEAM_WORLD,
    flush_async,
    put,
    put_value,
    wait_ready,
    wait_request,
)


@dataclass(frozen=True)
class Stage1SidecarModule:
    launch_send: object
    launch_reclaim: object
    launch_publish_plan_expected: object
    launch_mark_chunk_ready: object
    launch_enqueue_prepacked: object
    launch_return_credit: object
    batch_per_qp: int
    segment_bytes: int
    payload_bytes: int
    team: str


def build_stage1_sidecar_module(
    layout: HierCcoArenaLayout,
    *,
    batch_per_qp: int,
    segment_bytes: int,
    team: str = TEAM_RAIL,
) -> Stage1SidecarModule:
    """Compile the standalone CCO Stage-1 sidecar code object."""

    if batch_per_qp <= 0:
        raise ValueError("batch_per_qp must be positive")
    if segment_bytes <= 0 or segment_bytes % 8:
        raise ValueError("segment_bytes must be positive and 8-byte aligned")
    if team not in (TEAM_WORLD, TEAM_RAIL):
        raise ValueError("team must be world or rail")
    payload_bytes = layout.num_qp * int(batch_per_qp) * int(segment_bytes)
    if payload_bytes > layout.chunk_bytes:
        raise ValueError("bounded dispatch batch exceeds one ring chunk")

    threads = layout.num_qp * 64
    if threads > 1024:
        raise ValueError("one-QP-per-wave sidecar exceeds one workgroup")

    tx_base = layout.region("dispatch_tx").offset
    rx_base = layout.region("dispatch_rx").offset
    ready_base = layout.region("dispatch_ready").offset
    credit_base = layout.region("dispatch_credit").offset
    request_base = layout.region("dispatch_request").offset
    tag = (
        f"{team}_q{layout.num_qp}_r{layout.ring_depth}_"
        f"b{batch_per_qp}_s{segment_bytes}"
    )

    @flyc.kernel(
        name=f"megamoe_cco_h1_send_{tag}",
        known_block_size=[threads, 1, 1],
    )
    def send_kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx % fx.Int32(64)
        qp = tx // fx.Int32(64)
        slot_byte = fx.Int64(slot) * fx.Int64(layout.chunk_bytes)

        for item in range_constexpr(batch_per_qp):
            segment = qp * fx.Int32(batch_per_qp) + fx.Int32(item)
            payload_byte = fx.Int64(segment) * fx.Int64(segment_bytes)
            put(
                dev_comm,
                qp,
                peer,
                arena_win,
                fx.Int64(rx_base) + slot_byte + payload_byte,
                arena_win,
                fx.Int64(tx_base) + slot_byte + payload_byte,
                fx.Int64(segment_bytes),
                aggregate=True,
                scope="warp",
                team=team,
            )

        ready_byte = (
            fx.Int64(ready_base)
            + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
            * fx.Int64(8)
        )
        put_value(
            dev_comm,
            qp,
            peer,
            arena_win,
            ready_byte,
            generation,
            aggregate=True,
            scope="warp",
            team=team,
        )
        request = flush_async(
            dev_comm, qp, peer, scope="warp", team=team
        )

        # Only the QP-owner lane has a meaningful request token. Do not wait
        # here: reclaim_kernel waits after the remote credit permits slot reuse.
        if lane == fx.Int32(0):
            request_addr = (
                arena_ptr
                + fx.Int64(request_base)
                + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
                * fx.Int64(8)
            )
            comm_ops.store_i64_global_system(request_addr, request)

    @flyc.kernel(
        name=f"megamoe_cco_h1_reclaim_{tag}",
        known_block_size=[threads, 1, 1],
    )
    def reclaim_kernel(
        dev_comm: fx.Int64,
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx % fx.Int32(64)
        qp = tx // fx.Int32(64)
        index_byte = (
            fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp)
        ) * fx.Int64(8)
        if lane == fx.Int32(0):
            wait_ready(arena_ptr + fx.Int64(credit_base) + index_byte, generation)
        fx.gpu.barrier()
        comm_ops.fence_system_acquire()
        request = fx.Int64(
            comm_ops.load_i64_global(
                arena_ptr + fx.Int64(request_base) + index_byte
            )
        )
        wait_request(dev_comm, qp, request, scope="warp")
        if lane == fx.Int32(0):
            comm_ops.store_i64_global_system(
                arena_ptr + fx.Int64(request_base) + index_byte, fx.Int64(0)
            )

    @flyc.kernel(
        name=f"megamoe_cco_h1_plan_expected_{tag}",
        known_block_size=[256, 1, 1],
    )
    def publish_plan_expected_kernel(
        generation: fx.Int64,
        plan_ready: fx.Int64,
        input_expected: fx.Int64,
        active_m_tiles: fx.Int32,
        expected_per_tile: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        expected_rsrc = buffer_ops.create_buffer_resource_from_addr(input_expected)
        for tile in range(tx, active_m_tiles, fx.Int32(256)):
            buffer_ops.buffer_store(expected_per_tile, expected_rsrc, tile)
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            comm_ops.fence_system_release()
            comm_ops.store_i64_global_system(plan_ready, generation)

    @flyc.kernel(
        name=f"megamoe_cco_h1_mark_chunk_ready_{tag}",
        known_block_size=[256, 1, 1],
    )
    def mark_chunk_ready_kernel(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        input_ready: fx.Int64,
        first_m_tile: fx.Int32,
        tile_count: fx.Int32,
        delta: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        if tx < fx.Int32(layout.num_qp):
            ready_byte = (
                fx.Int64(ready_base)
                + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(tx))
                * fx.Int64(8)
            )
            wait_ready(arena_ptr + ready_byte, generation)
        fx.gpu.barrier()
        comm_ops.fence_system_acquire()
        comm_ops.fence_system_release()
        for item in range(tx, tile_count, fx.Int32(256)):
            tile = first_m_tile + item
            comm_ops.atomic_add_system(
                input_ready + fx.Int64(tile) * fx.Int64(4), delta
            )

    @flyc.kernel(
        name=f"megamoe_cco_h1_enqueue_prepacked_{tag}",
        known_block_size=[256, 1, 1],
    )
    def enqueue_prepacked_kernel(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        queue_header: fx.Int64,
        ready_queue: fx.Int64,
        total_work: fx.Int32,
        first_flat_tile: fx.Int32,
        tile_count: fx.Int32,
        tail_before: fx.Int32,
        final_batch: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        if tx < fx.Int32(layout.num_qp):
            ready_byte = (
                fx.Int64(ready_base)
                + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(tx))
                * fx.Int64(8)
            )
            wait_ready(arena_ptr + ready_byte, generation)
        fx.gpu.barrier()
        comm_ops.fence_system_acquire()

        queue_rsrc = buffer_ops.create_buffer_resource_from_addr(ready_queue)
        for item in range(tx, tile_count, fx.Int32(256)):
            buffer_ops.buffer_store(
                first_flat_tile + item, queue_rsrc, tail_before + item
            )
        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            next_tail = tail_before + tile_count
            comm_ops.fence_system_release()
            if tail_before == fx.Int32(0):
                # Header: [epoch, total_work, tail, done_generation]. Epoch is
                # the init publication consumed by H1, so write it last after
                # total/tail/done and after the separate plan publication.
                comm_ops.store_i64_global_system(
                    queue_header + fx.Int64(8), fx.Int64(total_work)
                )
                comm_ops.store_i64_global_system(
                    queue_header + fx.Int64(16), fx.Int64(0)
                )
                comm_ops.store_i64_global_system(
                    queue_header + fx.Int64(24), fx.Int64(0)
                )
                comm_ops.store_i64_global_system(queue_header, generation)
            comm_ops.store_i64_global_system(
                queue_header + fx.Int64(16), fx.Int64(next_tail)
            )
            if final_batch != fx.Int32(0):
                comm_ops.store_i64_global_system(
                    queue_header + fx.Int64(24), generation
                )

    @flyc.kernel(
        name=f"megamoe_cco_h1_credit_{tag}",
        known_block_size=[threads, 1, 1],
    )
    def return_credit_kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        qp = tx // fx.Int32(64)
        credit_byte = (
            fx.Int64(credit_base)
            + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
            * fx.Int64(8)
        )
        put_value(
            dev_comm,
            qp,
            peer,
            arena_win,
            credit_byte,
            generation,
            aggregate=True,
            scope="warp",
            team=team,
        )
        request = flush_async(
            dev_comm, qp, peer, scope="warp", team=team
        )
        wait_request(dev_comm, qp, request, scope="warp")

    @flyc.jit
    def launch_send(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        arena_ptr: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        send_kernel(
            dev_comm, arena_win, arena_ptr, peer, slot, generation
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    @flyc.jit
    def launch_reclaim(
        dev_comm: fx.Int64,
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        reclaim_kernel(dev_comm, arena_ptr, slot, generation).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_publish_plan_expected(
        generation: fx.Int64,
        plan_ready: fx.Int64,
        input_expected: fx.Int64,
        active_m_tiles: fx.Int32,
        expected_per_tile: fx.Int32,
        stream: fx.Stream,
    ):
        publish_plan_expected_kernel(
            generation,
            plan_ready,
            input_expected,
            active_m_tiles,
            expected_per_tile,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_mark_chunk_ready(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        input_ready: fx.Int64,
        first_m_tile: fx.Int32,
        tile_count: fx.Int32,
        delta: fx.Int32,
        stream: fx.Stream,
    ):
        mark_chunk_ready_kernel(
            arena_ptr,
            slot,
            generation,
            input_ready,
            first_m_tile,
            tile_count,
            delta,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_enqueue_prepacked(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        queue_header: fx.Int64,
        ready_queue: fx.Int64,
        total_work: fx.Int32,
        first_flat_tile: fx.Int32,
        tile_count: fx.Int32,
        tail_before: fx.Int32,
        final_batch: fx.Int32,
        stream: fx.Stream,
    ):
        enqueue_prepacked_kernel(
            arena_ptr,
            slot,
            generation,
            queue_header,
            ready_queue,
            total_work,
            first_flat_tile,
            tile_count,
            tail_before,
            final_batch,
        ).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_return_credit(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        slot: fx.Int32,
        generation: fx.Int64,
        stream: fx.Stream,
    ):
        return_credit_kernel(
            dev_comm, arena_win, peer, slot, generation
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return Stage1SidecarModule(
        launch_send=launch_send,
        launch_reclaim=launch_reclaim,
        launch_publish_plan_expected=launch_publish_plan_expected,
        launch_mark_chunk_ready=launch_mark_chunk_ready,
        launch_enqueue_prepacked=launch_enqueue_prepacked,
        launch_return_credit=launch_return_credit,
        batch_per_qp=int(batch_per_qp),
        segment_bytes=int(segment_bytes),
        payload_bytes=payload_bytes,
        team=team,
    )


@dataclass
class CcoStage1Sidecar:
    """Checked host lifecycle for dispatch ring requests and credits."""

    layout: HierCcoArenaLayout
    module: Stage1SidecarModule
    _slot_generation: list[int] = field(init=False)
    _slot_reclaimed: list[bool] = field(init=False)
    _receive_generation: int = field(default=0, init=False)
    _receive_tail: int = field(default=0, init=False)
    _receive_total_work: int = field(default=0, init=False)
    _receive_done: bool = field(default=True, init=False)
    _plan_generation: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.module.team not in (TEAM_WORLD, TEAM_RAIL):
            raise ValueError("invalid sidecar team")
        self._slot_generation = [0] * self.layout.ring_depth
        self._slot_reclaimed = [True] * self.layout.ring_depth

    @classmethod
    def create(
        cls,
        layout: HierCcoArenaLayout,
        *,
        batch_per_qp: int,
        segment_bytes: int,
        team: str = TEAM_RAIL,
    ) -> "CcoStage1Sidecar":
        return cls(
            layout,
            build_stage1_sidecar_module(
                layout,
                batch_per_qp=batch_per_qp,
                segment_bytes=segment_bytes,
                team=team,
            ),
        )

    def post_dispatch(
        self,
        dev_comm,
        arena_win,
        arena_ptr,
        peer,
        slot: int,
        generation: int,
        *,
        stream,
    ) -> None:
        self._validate_slot_generation(slot, generation)
        if not self._slot_reclaimed[slot]:
            raise RuntimeError("dispatch ring slot must be credited/reclaimed before reuse")
        self.module.launch_send(
            dev_comm,
            arena_win,
            arena_ptr,
            peer,
            slot,
            generation,
            stream=stream,
        )
        self._slot_generation[slot] = int(generation)
        self._slot_reclaimed[slot] = False

    def reclaim_dispatch(
        self, dev_comm, arena_ptr, slot: int, generation: int, *, stream
    ) -> None:
        self._validate_slot_generation(slot, generation)
        if self._slot_generation[slot] != int(generation):
            raise RuntimeError("reclaim generation does not own this ring slot")
        if self._slot_reclaimed[slot]:
            raise RuntimeError("dispatch ring slot is already reclaimed")
        self.module.launch_reclaim(
            dev_comm, arena_ptr, slot, generation, stream=stream
        )
        self._slot_reclaimed[slot] = True

    def publish_plan_expected(
        self,
        generation: int,
        epoch_pointers,
        active_m_tiles: int,
        *,
        expected_per_tile: int,
        stream,
    ) -> None:
        if int(generation) <= 0:
            raise ValueError("generation must be positive")
        if int(epoch_pointers.generation) != int(generation):
            raise ValueError("epoch pointers do not match generation")
        if not 0 <= int(active_m_tiles) <= self.layout.max_m_tiles:
            raise ValueError("active_m_tiles exceeds the arena capacity")
        if int(expected_per_tile) <= 0:
            raise ValueError("expected_per_tile must be positive")
        if int(generation) <= self._plan_generation:
            raise ValueError("plan generation must be strictly increasing")
        self.module.launch_publish_plan_expected(
            generation,
            epoch_pointers.plan_ready,
            epoch_pointers.h1_input_expected,
            active_m_tiles,
            expected_per_tile,
            stream=stream,
        )
        self._plan_generation = int(generation)

    def mark_chunk_ready(
        self,
        arena_ptr,
        slot: int,
        generation: int,
        epoch_pointers,
        first_m_tile: int,
        tile_count: int,
        *,
        delta: int = 1,
        stream,
    ) -> None:
        """Experimental queue producer; direct expected/ready remains default."""

        self._validate_slot_generation(slot, generation)
        if int(epoch_pointers.generation) != int(generation):
            raise ValueError("epoch pointers do not match generation")
        if self._plan_generation != int(generation):
            raise RuntimeError("plan must be published before ready-queue entries")
        if self._plan_generation != int(generation):
            raise RuntimeError("plan/expected must be published before chunk readiness")
        if first_m_tile < 0 or tile_count < 0:
            raise ValueError("M-tile range must be non-negative")
        if int(first_m_tile) + int(tile_count) > self.layout.max_m_tiles:
            raise ValueError("M-tile range exceeds the arena capacity")
        if int(delta) <= 0:
            raise ValueError("ready delta must be positive")
        self.module.launch_mark_chunk_ready(
            arena_ptr,
            slot,
            generation,
            epoch_pointers.h1_input_ready,
            first_m_tile,
            tile_count,
            delta,
            stream=stream,
        )

    def enqueue_prepacked_tiles(
        self,
        arena_ptr,
        slot: int,
        generation: int,
        epoch_pointers,
        *,
        total_work: int,
        first_flat_tile: int,
        tile_count: int,
        final_batch: bool,
        stream,
    ) -> None:
        self._validate_slot_generation(slot, generation)
        if int(epoch_pointers.generation) != int(generation):
            raise ValueError("epoch pointers do not match generation")
        capacity = self.layout.max_m_tiles * self.layout.max_h1_n_blocks
        total_work = int(total_work)
        first_flat_tile = int(first_flat_tile)
        tile_count = int(tile_count)
        if not 0 <= total_work <= capacity:
            raise ValueError("total_work exceeds the ready-queue capacity")
        if first_flat_tile < 0 or tile_count < 0:
            raise ValueError("flat tile range must be non-negative")
        if first_flat_tile + tile_count > total_work:
            raise ValueError("flat tile range exceeds total_work")

        if self._receive_generation != int(generation):
            if self._receive_generation and not self._receive_done:
                raise RuntimeError("previous receive generation has no final queue batch")
            self._receive_generation = int(generation)
            self._receive_tail = 0
            self._receive_total_work = total_work
            self._receive_done = False
        elif self._receive_total_work != total_work:
            raise ValueError("total_work changed within one receive generation")
        if self._receive_done:
            raise RuntimeError("receive queue generation is already complete")
        next_tail = self._receive_tail + tile_count
        if next_tail > total_work:
            raise ValueError("queue append exceeds total_work")
        if final_batch and next_tail != total_work:
            raise ValueError("final queue batch must publish the immutable final tail")

        self.module.launch_enqueue_prepacked(
            arena_ptr,
            slot,
            generation,
            epoch_pointers.h1_queue_header,
            epoch_pointers.h1_ready_queue,
            total_work,
            first_flat_tile,
            tile_count,
            self._receive_tail,
            int(final_batch),
            stream=stream,
        )
        self._receive_tail = next_tail
        self._receive_done = bool(final_batch)

    def return_credit(
        self,
        dev_comm,
        arena_win,
        peer,
        slot: int,
        generation: int,
        *,
        stream,
    ) -> None:
        self._validate_slot_generation(slot, generation)
        self.module.launch_return_credit(
            dev_comm, arena_win, peer, slot, generation, stream=stream
        )

    @property
    def outstanding_slots(self) -> tuple[int, ...]:
        return tuple(
            slot for slot, reclaimed in enumerate(self._slot_reclaimed) if not reclaimed
        )

    def _validate_slot_generation(self, slot: int, generation: int) -> None:
        if not 0 <= int(slot) < self.layout.ring_depth:
            raise ValueError("ring slot is outside the arena")
        if int(generation) <= 0:
            raise ValueError("generation must be positive")
