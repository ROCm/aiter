# SPDX-License-Identifier: MIT
"""Reusable device protocol for CCO ready/credit transport smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

from .. import comm_ops

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
class TransportSmoke:
    num_qp: int
    batch: int
    chunk_bytes: int
    team: str
    threads: int
    payload_bytes: int
    send_base: int
    recv_base: int
    ready_base: int
    credit_base: int
    required_bytes: int
    launch_send: object
    launch_credit: object
    launch_wait_words: object


def build_transport_smoke(
    *, num_qp: int, batch: int, chunk_bytes: int, team: str
) -> TransportSmoke:
    """Build the common multi-QP payload→ready→credit device protocol."""

    if num_qp not in (1, 2, 4, 8):
        raise ValueError("num_qp must be one of 1,2,4,8")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if chunk_bytes <= 0 or chunk_bytes % 8:
        raise ValueError("chunk_bytes must be positive and 8-byte aligned")
    if team not in (TEAM_WORLD, TEAM_RAIL):
        raise ValueError("team must be world or rail")

    threads = num_qp * 64
    payload_bytes = num_qp * batch * chunk_bytes
    send_base = 0
    recv_base = payload_bytes
    ready_base = 2 * payload_bytes
    credit_base = ready_base + num_qp * 8
    required_bytes = credit_base + num_qp * 8
    if ready_base % 8 or credit_base % 8:
        raise ValueError("ready/credit words must be 8-byte aligned")

    tag = f"{team}_q{num_qp}_b{batch}_c{chunk_bytes}"

    @flyc.kernel(
        name=f"megamoe_cco_smoke_send_{tag}",
        known_block_size=[threads, 1, 1],
    )
    def send_kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        generation: fx.Int64,
    ):
        tx = fx.Int32(fx.thread_idx.x)
        qp = tx // fx.Int32(64)

        for i in range_constexpr(batch):
            chunk = qp * fx.Int32(batch) + fx.Int32(i)
            offset = fx.Int64(chunk) * fx.Int64(chunk_bytes)
            put(
                dev_comm,
                qp,
                peer,
                arena_win,
                fx.Int64(recv_base) + offset,
                arena_win,
                fx.Int64(send_base) + offset,
                fx.Int64(chunk_bytes),
                aggregate=True,
                scope="warp",
                team=team,
            )

        put_value(
            dev_comm,
            qp,
            peer,
            arena_win,
            fx.Int64(ready_base) + fx.Int64(qp) * fx.Int64(8),
            generation,
            aggregate=True,
            scope="warp",
            team=team,
        )
        request = flush_async(
            dev_comm, qp, peer, scope="warp", team=team
        )
        wait_request(dev_comm, qp, request, scope="warp")

    @flyc.kernel(
        name=f"megamoe_cco_smoke_credit_{tag}",
        known_block_size=[threads, 1, 1],
    )
    def credit_kernel(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        generation: fx.Int64,
    ):
        tx = fx.Int32(fx.thread_idx.x)
        qp = tx // fx.Int32(64)
        put_value(
            dev_comm,
            qp,
            peer,
            arena_win,
            fx.Int64(credit_base) + fx.Int64(qp) * fx.Int64(8),
            generation,
            aggregate=True,
            scope="warp",
            team=team,
        )
        request = flush_async(
            dev_comm, qp, peer, scope="warp", team=team
        )
        wait_request(dev_comm, qp, request, scope="warp")

    @flyc.kernel(
        name=f"megamoe_cco_smoke_wait_{tag}",
        known_block_size=[64, 1, 1],
    )
    def wait_words_kernel(
        arena_ptr: fx.Int64, base: fx.Int64, generation: fx.Int64
    ):
        lane = fx.Int32(fx.thread_idx.x)
        if lane < fx.Int32(num_qp):
            address = arena_ptr + base + fx.Int64(lane) * fx.Int64(8)
            wait_ready(address, generation)
        fx.gpu.barrier()
        comm_ops.fence_system_acquire()

    @flyc.jit
    def launch_send(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        generation: fx.Int64,
        stream=fx.Stream(None),
    ):
        send_kernel(dev_comm, arena_win, peer, generation).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_credit(
        dev_comm: fx.Int64,
        arena_win: fx.Int64,
        peer: fx.Int32,
        generation: fx.Int64,
        stream=fx.Stream(None),
    ):
        credit_kernel(dev_comm, arena_win, peer, generation).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_wait_words(
        arena_ptr: fx.Int64,
        base: fx.Int64,
        generation: fx.Int64,
        stream=fx.Stream(None),
    ):
        wait_words_kernel(arena_ptr, base, generation).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return TransportSmoke(
        num_qp=num_qp,
        batch=batch,
        chunk_bytes=chunk_bytes,
        team=team,
        threads=threads,
        payload_bytes=payload_bytes,
        send_base=send_base,
        recv_base=recv_base,
        ready_base=ready_base,
        credit_base=credit_base,
        required_bytes=required_bytes,
        launch_send=launch_send,
        launch_credit=launch_credit,
        launch_wait_words=launch_wait_words,
    )
