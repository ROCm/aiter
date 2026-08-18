# SPDX-License-Identifier: MIT
"""EP8 arbitrary preplanned CCO-LSA dispatch-record fan-out smoke."""

from __future__ import annotations

from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist

from mori.cco import (
    Communicator,
    CCODevCommRequirements,
    GDA_CONNECTION_NONE,
    UniqueId,
)

from aiter.ops.flydsl.kernels.megamoe_tile import cco
from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_dispatch_fanout_lsa
from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout


WORLD = 8
RECORD_BYTES = 2048
RECORDS_PER_PROXY = 4
GENERATION = 7
PARITY = GENERATION & 1


def _broadcast_cco_uid(rank: int) -> UniqueId:
    obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    payload = obj[0]
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise RuntimeError("invalid CCO unique id broadcast")
    return UniqueId.from_bytes(payload)


def _destination(source_rank: int, record: int) -> int:
    return (source_rank * 3 + record * 2 + 1) % WORLD


def _valid(source_rank: int, record: int) -> bool:
    return (source_rank + record) % 5 != 0


def _slot(source_rank: int, record: int) -> int:
    return source_rank * RECORDS_PER_PROXY + record


def _record(source_rank: int, record: int):
    values = torch.arange(RECORD_BYTES, dtype=torch.int32)
    return ((values + source_rank * 37 + record * 19) % 251).to(torch.uint8)


def _record_words(source_rank: int, record: int):
    return [
        int(value) & 0xFFFFFFFFFFFFFFFF
        for value in _record(source_rank, record).view(torch.int64).tolist()
    ]


def main() -> int:
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD or rank != local_rank:
        raise ValueError("fan-out smoke requires one node with 8 node-major ranks")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    uid = _broadcast_cco_uid(rank)
    layout = HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=4,
        chunk_bytes=64 * 1024,
        max_m_tiles=8,
        max_source_tokens=64,
        max_h1_n_blocks=3,
        max_fanout_records=WORLD * RECORDS_PER_PROXY,
    )
    module = build_dispatch_fanout_lsa(layout, node_ranks=WORLD)
    failures = 0

    with Communicator.init(world, rank, uid, per_rank_vmm=64 * 1024 * 1024) as comm:
        memory = comm.alloc_mem(layout.total_bytes)
        window = comm.register_window(memory.ptr, memory.size)
        reqs = CCODevCommRequirements()
        reqs.gda_connection_type = GDA_CONNECTION_NONE
        reqs.gda_context_count = 1
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.lsa_barrier_count = 0
        reqs.rail_gda_barrier_count = 0
        reqs.barrier_count = 0
        dc = comm.create_dev_comm(reqs)
        if dc.lsa_size != WORLD or dc.lsa_rank != local_rank:
            raise RuntimeError("unexpected CCO LSA rank mapping")

        cco.zero_window(window.local_ptr, layout.total_bytes)
        rx_offset = layout.ring_chunk_offset("dispatch_rx", 0)
        records_ptr = window.local_ptr + rx_offset
        for item in range(RECORDS_PER_PROXY):
            cco.write_window_u64(
                records_ptr + item * RECORD_BYTES,
                _record_words(rank, item),
            )

        dest = torch.tensor(
            [_destination(rank, i) for i in range(RECORDS_PER_PROXY)],
            dtype=torch.int32,
            device=device,
        )
        slots = torch.tensor(
            [_slot(rank, i) for i in range(RECORDS_PER_PROXY)],
            dtype=torch.int32,
            device=device,
        )
        valid = torch.tensor(
            [int(_valid(rank, i)) for i in range(RECORDS_PER_PROXY)],
            dtype=torch.int32,
            device=device,
        )
        error = torch.zeros(1, dtype=torch.int32, device=device)
        torch.cuda.synchronize()
        comm.barrier()

        module.launch(
            records_ptr,
            window.handle,
            dest.data_ptr(),
            slots.data_ptr(),
            valid.data_ptr(),
            RECORDS_PER_PROXY,
            GENERATION,
            PARITY,
            error.data_ptr(),
            stream=torch.cuda.current_stream(device),
        )
        torch.cuda.synchronize()
        comm.barrier()

        expected_count = 0
        for source_rank in range(WORLD):
            for item in range(RECORDS_PER_PROXY):
                if not _valid(source_rank, item):
                    continue
                if _destination(source_rank, item) != rank:
                    continue
                expected_count += 1
                slot = _slot(source_rank, item)
                got = cco.read_window_u64(
                    window.local_ptr
                    + layout.offset("fanout_inbox")
                    + slot * RECORD_BYTES,
                    RECORD_BYTES // 8,
                )
                failures += int(got != tuple(_record_words(source_rank, item)))
                ready = cco.read_window_u64(
                    layout.pointer(
                        window.local_ptr, "fanout_ready", parity=PARITY
                    )
                    + slot * 8,
                    1,
                )[0]
                failures += int(ready != GENERATION)

        count = cco.read_window_u32(
            layout.pointer(window.local_ptr, "fanout_count", parity=PARITY),
            1,
        )[0]
        ready_all = cco.read_window_u64(
            layout.pointer(window.local_ptr, "fanout_ready", parity=PARITY),
            module.capacity,
        )
        failures += int(count != expected_count)
        failures += int(sum(int(value != 0) for value in ready_all) != expected_count)
        failures += int(error.item() != 0)

    status = torch.tensor([failures], dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.SUM)
    global_failures = int(status.item())
    dist.barrier()
    print(
        f"MEGAMOE_CCO_EP8_FANOUT_{'PASS' if global_failures == 0 else 'FAIL'} "
        f"rank={rank} expected={expected_count} local_failures={failures} "
        f"global_failures={global_failures}",
        flush=True,
    )
    dist.destroy_process_group()
    return 0 if global_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
