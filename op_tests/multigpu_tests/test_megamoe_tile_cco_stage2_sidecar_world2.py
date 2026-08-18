# SPDX-License-Identifier: MIT
"""WORLD2 lifecycle smoke for the CCO Stage-2 return sidecar."""

from __future__ import annotations

import os
import sys
import time

import torch

from mori.cco import Communicator, UniqueId

from aiter.ops.flydsl.kernels.megamoe_tile import cco
from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout


NUM_QP = 4
BATCH_PER_QP = 2
RECORD_BYTES = 7424
CHUNK_BYTES = 64 * 1024
RECORDS = NUM_QP * BATCH_PER_QP
EPOCHS = 2
SLOT = 0


def _bootstrap():
    rank = int(os.environ["CCO_RANK"])
    world = int(os.environ["CCO_WORLD"])
    path = os.environ["CCO_UID_FILE"]
    while not os.path.exists(path) or os.path.getsize(path) != 128:
        time.sleep(0.05)
    with open(path, "rb") as f:
        return rank, world, UniqueId.from_bytes(f.read())


def _pattern(rank: int, epoch: int, words: int):
    base = rank * 1_000_000 + epoch * 100_000
    return tuple(base + index for index in range(1, words + 1))


def main() -> int:
    rank, world, uid = _bootstrap()
    if world != 2:
        raise ValueError("WORLD2 H2 sidecar smoke requires exactly two ranks")

    gpu = int(os.environ.get("CCO_GPU", "0"))
    torch.cuda.set_device(gpu)
    peer = 1 - rank
    layout = HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=NUM_QP,
        chunk_bytes=CHUNK_BYTES,
        max_m_tiles=8,
        max_source_tokens=64,
        max_h1_n_blocks=3,
    )
    sidecar = cco.CcoStage2ReturnSidecar.create(
        layout,
        batch_per_qp=BATCH_PER_QP,
        record_bytes=RECORD_BYTES,
        team=cco.TEAM_WORLD,
    )
    payload_bytes = RECORDS * RECORD_BYTES
    if sidecar.module.payload_bytes != payload_bytes or payload_bytes > CHUNK_BYTES:
        raise AssertionError("invalid bounded partial-return geometry")

    failures = 0
    with Communicator.init(world, rank, uid, per_rank_vmm=64 * 1024 * 1024) as comm:
        resources = cco.create_transport_resources(
            comm,
            layout.total_bytes,
            num_qp=NUM_QP,
            team=cco.TEAM_WORLD,
        )
        arena_win = resources.window
        dc = resources.dev_comm
        cco.zero_window(arena_win.local_ptr, layout.total_bytes)

        tx_offset = layout.ring_chunk_offset("partial_tx", SLOT)
        rx_offset = layout.ring_chunk_offset("partial_rx", SLOT)
        tx_ptr = arena_win.local_ptr + tx_offset
        rx_ptr = arena_win.local_ptr + rx_offset
        payload_words = payload_bytes // 8
        ready_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "partial_ready", SLOT, 0
        )
        credit_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "partial_credit", SLOT, 0
        )
        request_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "partial_request", SLOT, 0
        )

        for epoch in range(1, EPOCHS + 1):
            ptrs = layout.epoch_pointers(arena_win.local_ptr, epoch)
            cco.write_window_u64(
                tx_ptr, _pattern(rank, epoch, payload_words)
            )
            cco.zero_window(rx_ptr, payload_bytes)
            torch.cuda.synchronize()
            comm.barrier()

            sidecar.post_partial_return(
                dc.ptr,
                arena_win.handle,
                arena_win.local_ptr,
                peer,
                SLOT,
                epoch,
                stream=torch.cuda.current_stream(),
            )
            sidecar.publish_received_partials(
                arena_win.local_ptr,
                SLOT,
                epoch,
                ptrs,
                0,
                RECORDS,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()

            expected = _pattern(peer, epoch, payload_words)
            got = cco.read_window_u64(rx_ptr, payload_words)
            failures += sum(
                int(lhs != rhs) for lhs, rhs in zip(got, expected)
            )
            partial_ready = cco.read_window_u64(
                ptrs.node_partial_ready, RECORDS
            )
            failures += sum(int(value != epoch) for value in partial_ready)
            ready_words = cco.read_window_u64(ready_ptr, NUM_QP)
            request_words = cco.read_window_u64(request_ptr, NUM_QP)
            failures += sum(int(value != epoch) for value in ready_words)
            failures += sum(int(value == 0) for value in request_words)

            sidecar.return_credit(
                dc.ptr,
                arena_win.handle,
                peer,
                SLOT,
                epoch,
                stream=torch.cuda.current_stream(),
            )
            sidecar.reclaim_partial(
                dc.ptr,
                arena_win.local_ptr,
                SLOT,
                epoch,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()

            credit_words = cco.read_window_u64(credit_ptr, NUM_QP)
            request_words = cco.read_window_u64(request_ptr, NUM_QP)
            failures += sum(int(value != epoch) for value in credit_words)
            failures += sum(int(value != 0) for value in request_words)
            failures += int(bool(sidecar.outstanding_slots))
            comm.barrier()

        print(
            f"MEGAMOE_CCO_H2_SIDECAR_{'PASS' if failures == 0 else 'FAIL'} "
            f"rank={rank} peer={peer} records={RECORDS} bytes={payload_bytes} "
            f"epochs={EPOCHS} failures={failures}",
            flush=True,
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
