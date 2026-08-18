# SPDX-License-Identifier: MIT
"""Standalone CCO Stage-2 node-partial return sidecar.

The sidecar transports a bounded batch of already-formed node-partial records.
It does not run GMM2, reduce expert routes, or unpack records into a final
combine buffer. Those producers/consumers remain separate code objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

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


DEFAULT_PARTIAL_RECORD_BYTES = 7424


@dataclass(frozen=True)
class Stage2SidecarModule:
    launch_send: object
    launch_reclaim: object
    launch_publish_received: object
    launch_return_credit: object
    batch_per_qp: int
    record_bytes: int
    payload_bytes: int
    team: str


def build_stage2_sidecar_module(
    layout: HierCcoArenaLayout,
    *,
    batch_per_qp: int = 2,
    record_bytes: int = DEFAULT_PARTIAL_RECORD_BYTES,
    team: str = TEAM_RAIL,
) -> Stage2SidecarModule:
    """Compile the independent node-partial return sidecar code object."""

    if batch_per_qp <= 0:
        raise ValueError("batch_per_qp must be positive")
    if record_bytes <= 0 or record_bytes % 8:
        raise ValueError("record_bytes must be positive and 8-byte aligned")
    if team not in (TEAM_WORLD, TEAM_RAIL):
        raise ValueError("team must be world or rail")
    payload_bytes = layout.num_qp * int(batch_per_qp) * int(record_bytes)
    if payload_bytes > layout.chunk_bytes:
        raise ValueError("bounded partial batch exceeds one ring chunk")

    threads = layout.num_qp * 64
    if threads > 1024:
        raise ValueError("one-QP-per-wave sidecar exceeds one workgroup")

    tx_base = layout.region("partial_tx").offset
    rx_base = layout.region("partial_rx").offset
    ready_base = layout.region("partial_ready").offset
    credit_base = layout.region("partial_credit").offset
    request_base = layout.region("partial_request").offset
    tag = (
        f"{team}_q{layout.num_qp}_r{layout.ring_depth}_"
        f"b{batch_per_qp}_s{record_bytes}"
    )

    @flyc.kernel(
        name=f"megamoe_cco_h2_return_send_{tag}",
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
            record = qp * fx.Int32(batch_per_qp) + fx.Int32(item)
            payload_byte = fx.Int64(record) * fx.Int64(record_bytes)
            put(
                dev_comm,
                qp,
                peer,
                arena_win,
                fx.Int64(rx_base) + slot_byte + payload_byte,
                arena_win,
                fx.Int64(tx_base) + slot_byte + payload_byte,
                fx.Int64(record_bytes),
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
        if lane == fx.Int32(0):
            request_addr = (
                arena_ptr
                + fx.Int64(request_base)
                + (fx.Int64(slot) * fx.Int64(layout.num_qp) + fx.Int64(qp))
                * fx.Int64(8)
            )
            comm_ops.store_i64_global_system(request_addr, request)

    @flyc.kernel(
        name=f"megamoe_cco_h2_return_reclaim_{tag}",
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
        name=f"megamoe_cco_h2_publish_received_{tag}",
        known_block_size=[256, 1, 1],
    )
    def publish_received_kernel(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        node_partial_ready: fx.Int64,
        first_source_token: fx.Int32,
        token_count: fx.Int32,
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

        # This only publishes records already in their consumer-visible layout.
        # A future unpack kernel must run before this point when wire != compute.
        for item in range(tx, token_count, fx.Int32(256)):
            token = first_source_token + item
            comm_ops.store_i64_global_system(
                node_partial_ready + fx.Int64(token) * fx.Int64(8), generation
            )

    @flyc.kernel(
        name=f"megamoe_cco_h2_return_credit_{tag}",
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
    def launch_publish_received(
        arena_ptr: fx.Int64,
        slot: fx.Int32,
        generation: fx.Int64,
        node_partial_ready: fx.Int64,
        first_source_token: fx.Int32,
        token_count: fx.Int32,
        stream: fx.Stream,
    ):
        publish_received_kernel(
            arena_ptr,
            slot,
            generation,
            node_partial_ready,
            first_source_token,
            token_count,
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

    return Stage2SidecarModule(
        launch_send=launch_send,
        launch_reclaim=launch_reclaim,
        launch_publish_received=launch_publish_received,
        launch_return_credit=launch_return_credit,
        batch_per_qp=int(batch_per_qp),
        record_bytes=int(record_bytes),
        payload_bytes=payload_bytes,
        team=team,
    )


@dataclass
class CcoStage2ReturnSidecar:
    """Checked host lifecycle for node-partial return ring slots."""

    layout: HierCcoArenaLayout
    module: Stage2SidecarModule
    _slot_generation: list[int] = field(init=False)
    _slot_reclaimed: list[bool] = field(init=False)

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
        batch_per_qp: int = 2,
        record_bytes: int = DEFAULT_PARTIAL_RECORD_BYTES,
        team: str = TEAM_RAIL,
    ) -> "CcoStage2ReturnSidecar":
        return cls(
            layout,
            build_stage2_sidecar_module(
                layout,
                batch_per_qp=batch_per_qp,
                record_bytes=record_bytes,
                team=team,
            ),
        )

    def post_partial_return(
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
            raise RuntimeError("partial ring slot must be credited/reclaimed before reuse")
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

    def reclaim_partial(
        self, dev_comm, arena_ptr, slot: int, generation: int, *, stream
    ) -> None:
        self._validate_slot_generation(slot, generation)
        if self._slot_generation[slot] != int(generation):
            raise RuntimeError("reclaim generation does not own this partial slot")
        if self._slot_reclaimed[slot]:
            raise RuntimeError("partial ring slot is already reclaimed")
        self.module.launch_reclaim(
            dev_comm, arena_ptr, slot, generation, stream=stream
        )
        self._slot_reclaimed[slot] = True

    def publish_received_partials(
        self,
        arena_ptr,
        slot: int,
        generation: int,
        epoch_pointers,
        first_source_token: int,
        token_count: int,
        *,
        stream,
    ) -> None:
        self._validate_slot_generation(slot, generation)
        if int(epoch_pointers.generation) != int(generation):
            raise ValueError("epoch pointers do not match generation")
        if first_source_token < 0 or token_count < 0:
            raise ValueError("source-token range must be non-negative")
        if int(first_source_token) + int(token_count) > self.layout.max_source_tokens:
            raise ValueError("source-token range exceeds the arena capacity")
        if int(token_count) > self.module.batch_per_qp * self.layout.num_qp:
            raise ValueError("published tokens exceed the bounded return batch")
        self.module.launch_publish_received(
            arena_ptr,
            slot,
            generation,
            epoch_pointers.node_partial_ready,
            first_source_token,
            token_count,
            stream=stream,
        )

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
