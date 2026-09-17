# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Measure dest-sharded TP all-gather launch overhead (the ~38 µs intercept).

Idea: force a **one-shot** kernel (C=1) by setting chunk_rows >= m_local.
Sweep a few tiny row counts. Copy of 4 rows is ~1 µs, so T(4) ≈ T_fixed.
A line fit on 4/8/16/32 is the same intercept; slope is µs/row copy.

    MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G PYTHONPATH="$PWD" \\
      torchrun --standalone --nproc-per-node=8 \\
      op_tests/multigpu_tests/test_tp_chunk_launch_us.py

Same number from the existing BW sweep (one chunk because 4096 > m_local)::

    torchrun --standalone --nproc-per-node=8 \\
      op_tests/multigpu_tests/test_tp_chunk_bw.py \\
      -b 4 8 16 32 64 128 256 --chunk-rows 4096 --iters 30
"""

from __future__ import annotations

import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")

import flydsl.expr as fx
import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_chunk_payload import (
    compile_tp_chunk_push,
    run_tp_chunk_push,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_push import (
    TpIncrementalWorkspace,
)

MODEL_DIM = 7168
# Launch-dominated. 64+ starts to show XGMI; keep them for the table only.
FIT_ROWS = (4, 8, 16, 32)
ALL_ROWS = FIT_ROWS + (64, 128, 256)
ITERS = 30


def _setup():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _time_one_shot(workspace, x_fp8, x_scale, m_local, iters, device):
    def body():
        run_tp_chunk_push(
            workspace,
            x_fp8,
            x_scale,
            m_local,
            fx.Stream(torch.cuda.current_stream().cuda_stream),
            chunk_rows=m_local,  # C=1: one wave, launch paid every call
        )

    _barrier()
    body()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    cap = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=cap):
        body()
    for _ in range(5):
        graph.replay()
    _barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3
    buf = torch.tensor([local_us], dtype=torch.float64, device=device)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float(buf.item() / workspace.npes)


def _fit_line(xs, ys):
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    intercept = (sy - slope * sx) / n
    return intercept, slope


def main():
    rank, world, device = _setup()
    try:
        if world < 2:
            if rank == 0:
                print("need >=2 GPUs")
            return
        workspace = TpIncrementalWorkspace(
            rank=rank,
            npes=world,
            max_m_local=max(ALL_ROWS),
            model_dim=MODEL_DIM,
            num_experts=1,
            device=device,
        )
        compile_tp_chunk_push(
            npes=world, model_dim=MODEL_DIM, row_major=False, num_producers=32
        )
        times = {}
        for rows in ALL_ROWS:
            workspace.reset()
            gen = torch.Generator(device=device).manual_seed(rank)
            x = torch.randn(
                (rows, MODEL_DIM), dtype=torch.bfloat16, device=device, generator=gen
            )
            x_fp8, x_scale = per_1x32_mx_quant(x, quant_mode="fp8")
            times[rows] = _time_one_shot(
                workspace, x_fp8, x_scale, rows, ITERS, device
            )
            if rank == 0:
                print(f"one-shot rows={rows:3d}  {times[rows]:6.1f} us", flush=True)

        if rank == 0:
            t_fixed, us_per_row = _fit_line(
                list(FIT_ROWS), [times[r] for r in FIT_ROWS]
            )
            print()
            print(f"fit on {list(FIT_ROWS)}:  T(us) = {t_fixed:.1f} + {us_per_row:.2f} * rows")
            print(f"T(4)={times[4]:.1f} us  ≈ launch; extra copy vs 4 rows:")
            for r in ALL_ROWS:
                print(f"  {r:3d}  {times[r]:6.1f} us  delta={times[r] - times[4]:+5.1f}")
            print()
            print("T_fixed is the intercept (~38 us). First fused wave ≈ T_fixed + copy(32).")
            print("Later waves in the same kernel do not pay T_fixed again.")
    finally:
        try:
            ms.shmem_finalize()
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
