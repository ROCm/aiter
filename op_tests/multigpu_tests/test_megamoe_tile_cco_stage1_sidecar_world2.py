# SPDX-License-Identifier: MIT
"""WORLD2 lifecycle smoke for ``CcoStage1Sidecar``.

Both ranks execute the same two-way protocol and reuse ring slot 0 in epoch 2:

    post_dispatch (flushAsync only)
      -> remote ready/acquire + payload validation
      -> return_credit
      -> sender reclaim (wait credit + retained request)
      -> request token == 0

The launcher is the same UID/file mode used by the other two-node CCO smokes.
"""

from __future__ import annotations

import os
import sys
import time

import torch

from mori.cco import Communicator, UniqueId

from aiter.ops.flydsl.kernels.megamoe_tile import cco
from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout


NUM_QP = 4
BATCH_PER_QP = 8
SEGMENT_BYTES = 2048
CHUNK_BYTES = 64 * 1024
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
        raise ValueError("WORLD2 sidecar smoke requires exactly two ranks")

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
    sidecar = cco.CcoStage1Sidecar.create(
        layout,
        batch_per_qp=BATCH_PER_QP,
        segment_bytes=SEGMENT_BYTES,
        team=cco.TEAM_WORLD,
    )
    if sidecar.module.payload_bytes != CHUNK_BYTES:
        raise AssertionError("WORLD2 smoke must fill exactly one dispatch chunk")

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

        tx_offset = layout.ring_chunk_offset("dispatch_tx", SLOT)
        rx_offset = layout.ring_chunk_offset("dispatch_rx", SLOT)
        tx_ptr = arena_win.local_ptr + tx_offset
        rx_ptr = arena_win.local_ptr + rx_offset
        payload_words = CHUNK_BYTES // 8
        ready_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "dispatch_ready", SLOT, 0
        )
        credit_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "dispatch_credit", SLOT, 0
        )
        request_ptr = arena_win.local_ptr + layout.ring_qp_offset(
            "dispatch_request", SLOT, 0
        )

        for epoch in range(1, EPOCHS + 1):
            ptrs = layout.epoch_pointers(arena_win.local_ptr, epoch)
            cco.write_window_u64(
                tx_ptr, _pattern(rank, epoch, payload_words)
            )
            cco.zero_window(rx_ptr, CHUNK_BYTES)
            torch.cuda.synchronize()
            comm.barrier()

            # Count/metadata planning is separate from payload arrival.
            sidecar.publish_plan_expected(
                epoch,
                ptrs,
                1,
                expected_per_tile=1,
                stream=torch.cuda.current_stream(),
            )
            sidecar.post_dispatch(
                dc.ptr,
                arena_win.handle,
                arena_win.local_ptr,
                peer,
                SLOT,
                epoch,
                stream=torch.cuda.current_stream(),
            )
            sidecar.mark_chunk_ready(
                arena_win.local_ptr,
                SLOT,
                epoch,
                ptrs,
                0,
                1,
                delta=1,
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()

            expected_payload = _pattern(peer, epoch, payload_words)
            got = cco.read_window_u64(rx_ptr, payload_words)
            failures += sum(
                int(lhs != rhs) for lhs, rhs in zip(got, expected_payload)
            )
            failures += int(cco.read_window_u32(ptrs.h1_input_expected, 1)[0] != 1)
            failures += int(cco.read_window_u32(ptrs.h1_input_ready, 1)[0] != 1)
            failures += int(cco.read_window_u64(ptrs.plan_ready, 1)[0] != epoch)
            ready_words = cco.read_window_u64(ready_ptr, NUM_QP)
            failures += sum(int(value != epoch) for value in ready_words)
            # post_dispatch deliberately did not wait/reclaim its request.
            requests = cco.read_window_u64(request_ptr, NUM_QP)
            failures += sum(int(value == 0) for value in requests)

            # Payload/fan-out consumption is complete, so the receiver may
            # return credit. Reclaim waits both credit and local CQ completion.
            sidecar.return_credit(
                dc.ptr,
                arena_win.handle,
                peer,
                SLOT,
                epoch,
                stream=torch.cuda.current_stream(),
            )
            sidecar.reclaim_dispatch(
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
            f"MEGAMOE_CCO_H1_SIDECAR_{'PASS' if failures == 0 else 'FAIL'} "
            f"rank={rank} peer={peer} qps={NUM_QP} slot={SLOT} "
            f"epochs={EPOCHS} failures={failures}",
            flush=True,
        )

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
