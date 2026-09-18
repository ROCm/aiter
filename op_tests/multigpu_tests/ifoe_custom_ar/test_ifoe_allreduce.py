# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""IFOE cross-node custom all-reduce test / benchmark (gfx1250).

Driver only -- the kernels and host API live in
``csrc/{include,kernels}/custom_all_reduce_ifoe.*`` and are exposed as the
``module_custom_all_reduce_ifoe`` JIT module via
``aiter.dist.device_communicators.ifoe_custom_all_reduce.IfoeCustomAllreduce``.

Launch with torchrun (fabric handles are node independent, so TP4 and TP8 use
the same code path):

  # TP4, single node (4 GPUs):
  torchrun --nproc_per_node=4 test_ifoe_allreduce.py --mb 256

  # TP8, two nodes x 4 GPUs (run on each node; NODE_RANK 0 then 1):
  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
      --master_addr=<node0-ip> --master_port=29500 test_ifoe_allreduce.py --mb 256

  # latency sweep over a range of element counts (prints us/iter per size+mode):
  torchrun --nproc_per_node=4 test_ifoe_allreduce.py --sweep --bench
"""

import argparse
import os

import torch
import torch.distributed as dist

from aiter.dist.device_communicators.ifoe_custom_all_reduce import IfoeCustomAllreduce


def _bench(comm, x, mode, iters=100, warmup=20):
    for _ in range(warmup):
        comm.all_reduce(x, mode=mode)
    torch.cuda.synchronize()
    dist.barrier(group=comm.group)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        comm.all_reduce(x, mode=mode)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


# element-count sweep (fp32 elements): 1K up to 4096*8192 = 32M
_SWEEP = [1 << k for k in range(10, 14)] + [
    m * 8192 for m in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
]


def main():
    ap = argparse.ArgumentParser(description="IFOE custom all-reduce test")
    ap.add_argument("--mb", type=int, default=256, help="tensor size in MiB (fp32)")
    ap.add_argument("--modes", nargs="+", default=["fp32", "bf16", "fp8"])
    ap.add_argument(
        "--bench", action="store_true", help="also time and report us/iter + busbw"
    )
    ap.add_argument(
        "--sweep", action="store_true", help="latency sweep over the default size range"
    )
    ap.add_argument(
        "--elems", type=int, nargs="+", help="explicit element counts to benchmark"
    )
    args = ap.parse_args()

    if args.elems:
        sizes = args.elems  # element counts
    elif args.sweep:
        sizes = _SWEEP
    else:
        sizes = [(args.mb << 20) // 4]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.group.WORLD
    rank, world = dist.get_rank(), dist.get_world_size()

    max_bytes = max(sizes) * 4
    comm = IfoeCustomAllreduce(
        group, torch.device("cuda", local_rank), max_bytes=max_bytes
    )
    expected = world * (world + 1) / 2.0

    ok = True
    for numel in sizes:
        bytes_ = numel * 4
        x = torch.full((numel,), float(rank + 1), dtype=torch.float32, device="cuda")
        for mode in args.modes:
            out = comm.all_reduce(x, mode=mode)
            torch.cuda.synchronize()
            mism = int((out != expected).sum().item())
            passed = mism == 0
            ok = ok and passed
            line = f"[world={world} {numel:>10} elem {mode:>4}] {'PASS' if passed else f'FAIL(mism={mism})'}"
            if args.bench:
                ms = _bench(comm, x, mode)
                busbw = 2.0 * (world - 1) / world * bytes_ / (ms / 1e3) / 1e9
                line += f"  {ms * 1e3:8.2f} us/iter  busbw {busbw:6.1f} GB/s"
            if rank == 0:
                print(line, flush=True)

    comm.dispose()
    dist.barrier(group=group)
    dist.destroy_process_group()
    assert ok, "IFOE all-reduce correctness failed"


if __name__ == "__main__":
    main()
