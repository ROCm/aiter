# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Sync latency test for gfx1250 (MI450).
# Measures start_sync, end_sync, and two_sync (start+end) latency
# across grid sizes 1,2,4,8,16,32,64,128,256 with block=256.

import argparse
import os
import sys
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter import logger
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import perftest

set_start_method("spawn", force=True)

_INIT_TIMEOUT_SEC = 120

_GRIDS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
_KERNELS = ["start_sync", "end_sync", "two_sync"]


def _worker(
    tp_size: int,
    rank: int,
    grid: int,
    kernel: str,
    distributed_init_method: str,
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)

    init_distributed_environment(
        world_size=tp_size,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    ca_comm = get_tp_group().device_communicator.ca_comm
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("Custom allreduce not initialized")

    import aiter as ops

    _ptr = ca_comm._ptr

    fn_map = {
        "start_sync": ops.start_sync_latency_gfx1250,
        "end_sync": ops.end_sync_latency_gfx1250,
        "two_sync": ops.two_sync_latency_gfx1250,
    }
    fn = fn_map[kernel]

    @perftest(use_cuda_event=True)
    def run_sync():
        fn(_ptr, grid)

    _, latency_us = run_sync()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return latency_us


def run_one(tp_size: int, grid: int, kernel: str, distributed_init_method: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "49376")
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            _worker,
            args=(tp_size, r, grid, kernel, distributed_init_method),
        )
        for r in range(tp_size)
    ]
    pool.close()

    results = []
    try:
        for r in rets:
            results.append(r.get(timeout=_INIT_TIMEOUT_SEC))
    except Exception:
        pool.terminate()
        pool.join()
        raise
    pool.join()

    return max(results)


parser = argparse.ArgumentParser(
    description="gfx1250 multi-GPU sync latency test "
    "(start_sync / end_sync / two_sync)"
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4],
    default=2,
    help="tensor-parallel size (default: 2)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    tp_size = args.tp_size

    num_gpus = torch.cuda.device_count()
    if num_gpus < tp_size:
        logger.error("Need %d GPUs, have %d", tp_size, num_gpus)
        sys.exit(1)

    rows = []
    for grid in _GRIDS:
        row = {"grid": grid}
        for kernel in _KERNELS:
            init_method = get_distributed_init_method(get_ip(), get_open_port())
            try:
                latency_us = run_one(tp_size, grid, kernel, init_method)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(
                    "grid=%d kernel=%s FAILED: %s",
                    grid,
                    kernel,
                    e,
                )
                latency_us = float("nan")
            row[kernel] = latency_us
        rows.append(row)

    header = f"{'grid':>6s} | {'start_sync(us)':>14s} | {'end_sync(us)':>14s} | {'two_sync(us)':>14s}"
    sep = "-" * len(header)
    print(f"\ngfx1250 sync latency (tp={tp_size}, block=256)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['grid']:>6d} | "
            f"{r['start_sync']:>14.2f} | "
            f"{r['end_sync']:>14.2f} | "
            f"{r['two_sync']:>14.2f}"
        )
    print(sep)
