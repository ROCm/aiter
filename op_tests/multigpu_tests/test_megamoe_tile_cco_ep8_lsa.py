# SPDX-License-Identifier: MIT
"""EP8 CCO registered-window LSA pointer visibility smoke.

Launch on one 8-GPU node::

    GLOO_SOCKET_IFNAME=enp193s0f1np1 \
    MORI_SOCKET_IFNAME=enp193s0f1np1 \
    torchrun --standalone --nproc-per-node=8 \
      op_tests/multigpu_tests/test_megamoe_tile_cco_ep8_lsa.py

Every rank writes a unique row in its local window. Every rank then resolves all
eight peer bases with ``cco.lsa_ptr`` and reads its own column from each peer.
No MORI SHMEM allocation or device function is used.
"""

from __future__ import annotations

from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from mori.cco import (
    Communicator,
    CCODevCommRequirements,
    GDA_CONNECTION_NONE,
    UniqueId,
)

from aiter.ops.flydsl.kernels.megamoe_tile import cco
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.megamoe_tile import comm_ops


WORLD = 8
WINDOW_BYTES = 4096
WORDS_PER_ROW = 8
EPOCHS = 2
EPOCH_READY_OFFSET = 128
PARTIAL_OFFSET = 256
PARTIAL_SOURCES = 3
PARTIAL_HIDDEN = 128


@flyc.kernel(known_block_size=[64, 1, 1])
def read_all_peers_kernel(
    window: fx.Int64,
    output: fx.Int64,
    word_index: fx.Int32,
):
    lane = fx.Int32(fx.thread_idx.x)
    output_rsrc = buffer_ops.create_buffer_resource_from_addr(output)
    if lane < fx.Int32(WORLD):
        comm_ops.fence_system_acquire()
        address = cco.lsa_ptr(
            window, lane, fx.Int64(word_index) * fx.Int64(8)
        )
        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(address)
        value = buffer_ops.buffer_load(
            peer_rsrc, fx.Int32(0), vec_width=1, dtype=T.i64
        )
        buffer_ops.buffer_store(value, output_rsrc, lane)


@flyc.jit
def launch_read_all_peers(
    window: fx.Int64,
    output: fx.Int64,
    word_index: fx.Int32,
    stream=fx.Stream(None),
):
    read_all_peers_kernel(window, output, word_index).launch(
        grid=(1, 1, 1), block=(64, 1, 1), stream=stream
    )


def _broadcast_cco_uid(rank: int) -> UniqueId:
    obj = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(obj, src=0)
    payload = obj[0]
    if not isinstance(payload, bytes) or len(payload) != 128:
        raise RuntimeError("invalid CCO unique id broadcast")
    return UniqueId.from_bytes(payload)


def _expected(epoch: int, word_index: int, device):
    peers = torch.arange(WORLD, dtype=torch.int64, device=device)
    return peers * 1_000_000 + epoch * 10_000 + word_index


def _rank_partial(rank: int, epoch: int):
    values = torch.arange(
        PARTIAL_SOURCES * PARTIAL_HIDDEN, dtype=torch.float32
    ).reshape(PARTIAL_SOURCES, PARTIAL_HIDDEN)
    return (values * 0.001 + rank + epoch * 0.25).to(torch.bfloat16)


def main() -> int:
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != WORLD:
        raise ValueError(f"requires one-node world_size=8, got {world}")
    if rank != local_rank:
        raise ValueError("EP8 LSA smoke requires one node with rank == LOCAL_RANK")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    uid = _broadcast_cco_uid(rank)
    failures = 0

    with Communicator.init(world, rank, uid, per_rank_vmm=64 * 1024 * 1024) as comm:
        memory = comm.alloc_mem(WINDOW_BYTES)
        window = comm.register_window(memory.ptr, memory.size)

        # DevComm is used only to assert the CCO LSA rank mapping. GDA and every
        # optional resource pool stay disabled; lsa_ptr itself needs only window.
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
            raise RuntimeError(
                f"unexpected LSA mapping size={dc.lsa_size} rank={dc.lsa_rank}"
            )

        output = torch.empty(WORLD, dtype=torch.int64, device=device)
        from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
            compile_node_partial_reduce_lsa,
            compile_rank_partial_epoch_gate_lsa,
        )

        reduced = torch.zeros(
            (PARTIAL_SOURCES, PARTIAL_HIDDEN),
            dtype=torch.float32,
            device=device,
        )
        reduce_expected = torch.full(
            (PARTIAL_SOURCES,), WORLD, dtype=torch.int32, device=device
        )
        reduce_ready = reduce_expected.clone()
        node_ready = torch.zeros(
            PARTIAL_SOURCES, dtype=torch.int64, device=device
        )
        reducer = compile_node_partial_reduce_lsa(
            D_HIDDEN=PARTIAL_HIDDEN,
            NUM_RANKS=WORLD,
            output_dtype="fp32",
        )
        epoch_gate = compile_rank_partial_epoch_gate_lsa(NUM_RANKS=WORLD)
        assert reducer.requires_registered_window_handle is True
        assert "no-tp-reduction" in reducer.output_contract
        assert epoch_gate.publish_before_wait is True
        cco.zero_window(window.local_ptr, WINDOW_BYTES)

        for epoch in range(1, EPOCHS + 1):
            cco.write_window_u64(
                window.local_ptr,
                (
                    rank * 1_000_000 + epoch * 10_000 + index
                    for index in range(WORDS_PER_ROW)
                ),
            )
            local_partial = _rank_partial(rank, epoch).contiguous()
            packed_partial = local_partial.view(torch.int32).flatten().tolist()
            cco.write_window_u32(
                window.local_ptr + PARTIAL_OFFSET, packed_partial
            )
            output.fill_(-1)
            reduced.zero_()
            node_ready.zero_()
            epoch_gate(
                window.handle,
                window.local_ptr + EPOCH_READY_OFFSET,
                EPOCH_READY_OFFSET,
                epoch,
                stream=torch.cuda.current_stream(device),
            )

            launch_read_all_peers(
                window.handle,
                output.data_ptr(),
                local_rank,
                stream=torch.cuda.current_stream(device),
            )
            torch.cuda.synchronize()
            failures += int(
                torch.count_nonzero(
                    output != _expected(epoch, local_rank, device)
                ).item()
            )
            reducer(
                window.handle,
                PARTIAL_OFFSET,
                reduce_ready.data_ptr(),
                reduce_expected.data_ptr(),
                reduced.data_ptr(),
                node_ready.data_ptr(),
                epoch,
                PARTIAL_SOURCES,
                stream=torch.cuda.current_stream(device),
            )
            torch.cuda.synchronize()
            reference = sum(
                (_rank_partial(peer, epoch).float() for peer in range(WORLD)),
                torch.zeros(
                    (PARTIAL_SOURCES, PARTIAL_HIDDEN), dtype=torch.float32
                ),
            ).to(device)
            failures += int(torch.count_nonzero(reduced != reference).item())
            failures += int(torch.count_nonzero(node_ready != epoch).item())
            comm.barrier()

    status = torch.tensor([failures], dtype=torch.int64)
    dist.all_reduce(status, op=dist.ReduceOp.SUM)
    global_failures = int(status.item())
    dist.barrier()
    print(
        f"MEGAMOE_CCO_EP8_LSA_{'PASS' if global_failures == 0 else 'FAIL'} "
        f"rank={rank} local_rank={local_rank} local_failures={failures} "
        f"global_failures={global_failures}",
        flush=True,
    )
    dist.destroy_process_group()
    return 0 if global_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
