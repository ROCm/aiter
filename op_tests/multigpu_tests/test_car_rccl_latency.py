# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
import torch.distributed as dist
from multiprocessing import Pool, freeze_support, set_start_method

from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)

NUM_WARMUP = 5
NUM_ITERS = 20
TP_SIZE = 8
HIDDEN = 8192
DTYPE = torch.bfloat16
ELEM_BYTES = 2  # bf16

SIZES_MB = [128, 256, 512, 1024]


def _measure_per_iter_us(fn, num_warmup=NUM_WARMUP, num_iters=NUM_ITERS):
    """Return a list of per-iteration latencies (us) after warmup."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    latencies = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end) * 1000.0)  # ms -> us
    return latencies


def bench_worker(rank_id, tp_size, distributed_init_method):
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, 1)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    results = {}
    for size_mb in SIZES_MB:
        num_elems = size_mb * 1024 * 1024 // ELEM_BYTES
        rows = num_elems // HIDDEN
        shape = (rows, HIDDEN)

        x_aiter = torch.randn(shape, dtype=DTYPE, device=device)
        aiter_lats = _measure_per_iter_us(
            lambda: tensor_model_parallel_all_reduce(x_aiter)  # noqa: F821
        )

        dist.barrier(group=group)

        x_rccl = torch.randn(shape, dtype=DTYPE, device=device)
        rccl_lats = _measure_per_iter_us(lambda: dist.all_reduce(x_rccl, group=group))  # noqa: F821

        dist.barrier(group=group)

        del x_aiter, x_rccl
        torch.cuda.empty_cache()

        results[size_mb] = (aiter_lats, rccl_lats)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return rank_id, results


if __name__ == "__main__":
    freeze_support()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    pool = Pool(processes=TP_SIZE)

    futures = []
    for i in range(TP_SIZE):
        futures.append(pool.apply_async(bench_worker, args=(i, TP_SIZE, init_method)))
    pool.close()
    pool.join()

    all_results = [f.get() for f in futures]

    hdr = f"{'Size':>8} | {'Shape':>16} | {'AITER (us)':>12} | {'RCCL (us)':>12}"
    sep = "-" * len(hdr)

    print()
    print(sep)
    print(hdr)
    print(sep)

    for size_mb in SIZES_MB:
        rows = size_mb * 1024 * 1024 // ELEM_BYTES // HIDDEN
        shape_str = f"({rows}, {HIDDEN})"

        aiter_per_iter = [
            max(r[1][size_mb][0][i] for r in all_results) for i in range(NUM_ITERS)
        ]
        rccl_per_iter = [
            max(r[1][size_mb][1][i] for r in all_results) for i in range(NUM_ITERS)
        ]
        avg_aiter = sum(aiter_per_iter) / NUM_ITERS
        avg_rccl = sum(rccl_per_iter) / NUM_ITERS

        label = f"{size_mb} MB" if size_mb < 1024 else f"{size_mb // 1024} GB"
        print(
            f"{label:>8} | {shape_str:>16} | " f"{avg_aiter:>12.1f} | {avg_rccl:>12.1f}"
        )

    print(sep)
    print(f"  dtype=bf16  hidden={HIDDEN}  tp={TP_SIZE}  mode=eager")
    print(
        f"  warmup={NUM_WARMUP}  iters={NUM_ITERS}  "
        f"metric=avg of per-iter max latency across {TP_SIZE} GPUs"
    )
    print()
