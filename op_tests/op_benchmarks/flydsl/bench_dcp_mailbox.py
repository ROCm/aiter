# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph time of the FlyDSL DCP mailbox combine and query gather.

Run on one node with 8 GPUs:

    WORLD=8 python3 op_tests/op_benchmarks/flydsl/bench_dcp_mailbox.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.ops.flydsl.dcp_mailbox import (
    CombineWorkspace,
    GatherWorkspace,
    dcp_lse_combine,
    dcp_query_gather,
)


class _Group:
    def __init__(self, cpu_group, rank, world):
        self.cpu_group = cpu_group
        self.rank_in_group = rank
        self.world_size = world


def _graph_us(fn, reps=29):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    # Every rank waits on its peers, so align the ranks before the timed replay.
    dist.barrier()
    g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / reps


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29742")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = _Group(dist.new_group(backend="gloo"), rank, world)
    try:
        tokens, hl = 16, 12
        for dim in (128, 512, 576):
            cws = CombineWorkspace(group, torch.device("cuda"), hl, dim, 64)
            out = torch.randn(
                tokens, world * hl, dim, device="cuda", dtype=torch.bfloat16
            )
            lse = torch.randn(tokens, world * hl, device="cuda", dtype=torch.float32)
            combine_us = _graph_us(
                lambda w=cws, o=out, s=lse: dcp_lse_combine(w, o, s, return_lse=True)
            )
            if rank == 0:
                print(
                    f"DCP_MAILBOX_BENCH combine ranks={world} tokens={tokens} "
                    f"heads={hl} dim={dim} {combine_us:.2f} us",
                    flush=True,
                )
        dim = 512
        gws = GatherWorkspace(
            group, torch.device("cuda"), hl, dim, torch.bfloat16, world * hl, 64
        )
        query = torch.randn(tokens, hl, dim, device="cuda", dtype=torch.bfloat16)
        gather_us = _graph_us(lambda: dcp_query_gather(gws, query))
        if rank == 0:
            print(
                f"DCP_MAILBOX_BENCH gather ranks={world} tokens={tokens} "
                f"heads={hl} dim={dim} {gather_us:.2f} us",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world = int(os.environ.get("WORLD", "8"))
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
