# SPDX-License-Identifier: MIT
"""Eight-GPU 2x4 logical-node MORI transport roundtrip smoke."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _pattern(rank: int, nbytes: int, device: torch.device) -> torch.Tensor:
    return (
        (torch.arange(nbytes, dtype=torch.int32, device=device) + rank * 17)
        .remainder(251)
        .to(torch.uint8)
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise RuntimeError("this smoke requires world_size=8")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")
    torch._C._distributed_c10d._register_process_group("megamoe_tile", dist.group.WORLD)

    from mori import shmem as ms
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_mori_eos_module,
        build_mori_put_signal_module,
        build_mori_quiet_module,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.topology import LogicalTopology

    ms.shmem_torch_process_group_init("megamoe_tile")
    topology = LogicalTopology(8, 4)
    peer = topology.proxy_pe(1 - topology.node_of(rank), rank)
    if ms.shmem_mype() != rank:
        raise RuntimeError("EP rank -> SHMEM PE identity assumption is false")

    nbytes = 4096
    src = dst = ready = credit = None
    try:
        # All PEs allocate in the same order and with the same shape.
        src = ms.mori_shmem_create_tensor((nbytes,), torch.uint8)
        dst = ms.mori_shmem_create_tensor((nbytes,), torch.uint8)
        ready = ms.mori_shmem_create_tensor((1,), torch.int64)
        credit = ms.mori_shmem_create_tensor((1,), torch.int64)
        src.copy_(_pattern(rank, nbytes, device))
        dst.zero_()
        ready.zero_()
        credit.zero_()
        torch.cuda.synchronize()
        dist.barrier()

        put = build_mori_put_signal_module()
        quiet = build_mori_quiet_module()
        control_put = build_mori_eos_module()
        stream = torch.cuda.current_stream()

        # Cross-logical-node traffic stays inside one host and therefore exercises
        # MORI's P2P/copy backend with the exact future RDMA ABI.
        put(
            dst.data_ptr(),
            src.data_ptr(),
            nbytes,
            ready.data_ptr(),
            rank + 1,
            peer,
            1,
            stream=stream,
        )
        quiet(peer, 1, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()

        data_ok = torch.equal(dst, _pattern(peer, nbytes, device))
        ready_ok = ready.item() == peer + 1

        # Receiver-consumed credit travels back through a real non-zero 8-byte WQE.
        control_put(credit.data_ptr(), rank + 101, peer, 0, stream=stream)
        quiet(peer, 0, stream=stream)
        torch.cuda.synchronize()
        dist.barrier()
        credit_ok = credit.item() == peer + 101

        local_ok = torch.tensor(
            [int(data_ok and ready_ok and credit_ok)], dtype=torch.int32
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        if rank == 0:
            print(
                "MEGAMOE_TILE_2X4_RING_PASS",
                bool(local_ok.item()),
                "bytes_per_rank=",
                nbytes,
                flush=True,
            )
        if not local_ok.item():
            raise AssertionError(
                f"rank={rank} peer={peer} data={data_ok} ready={ready_ok} credit={credit_ok}"
            )
    finally:
        dist.barrier()
        for tensor in (credit, ready, dst, src):
            if tensor is not None:
                ms.mori_shmem_free_tensor(tensor)
        ms.shmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
