# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Benchmark for gfx1250 (MI450) allgather kernels: naive vs warpsplit.
# TP2 only. Reports latency (us) and XGMI bandwidth (GB/s).
#
# XGMI bandwidth = input_size_bytes / latency
#   (each rank reads input_size bytes from the remote GPU via XGMI)

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

_SIZES = [
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
]


def _size_str(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    return f"{n // 1024}K"


def _worker(
    rank: int,
    size: int,
    kernel_type: int,
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
        world_size=2,
        rank=rank,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(2, 1)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    ca_comm = get_tp_group().device_communicator.ca_comm
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("Custom allreduce not initialized")

    import aiter as ops

    _ptr = ca_comm._ptr
    inp = torch.randn(size, dtype=torch.bfloat16, device=device)
    out = torch.empty(size * 2, dtype=torch.bfloat16, device=device)

    reg_inp = ca_comm._pool["input"].data_ptr
    reg_inp_bytes = ca_comm._pool["input"].max_size

    @perftest(use_cuda_event=True)
    def run_ag():
        ops.all_gather_gfx1250(_ptr, inp, out, kernel_type, reg_inp, reg_inp_bytes)

    _, latency_us = run_ag()

    # Correctness check: gather all inputs and compare
    all_inputs = [torch.empty_like(inp) for _ in range(2)]
    dist.all_gather(all_inputs, inp, group=group)
    ref = torch.cat(all_inputs, dim=0)
    max_err = (out.float() - ref.float()).abs().max().item()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return latency_us, max_err


def run_one(size: int, kernel_type: int, distributed_init_method: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49374"
    pool = Pool(processes=2)
    rets = []
    for rank in range(2):
        rets.append(
            pool.apply_async(
                _worker,
                args=(rank, size, kernel_type, distributed_init_method),
            )
        )
    pool.close()

    results = []
    try:
        for r in rets:
            results.append(r.get(timeout=_INIT_TIMEOUT_SEC))
    except Exception as e:
        pool.terminate()
        pool.join()
        raise
    pool.join()

    latencies = [r[0] for r in results]
    errors = [r[1] for r in results]
    return max(latencies), max(errors)


if __name__ == "__main__":
    kernel_names = {0: "naive", 1: "warpsplit"}

    rows = []
    for size in _SIZES:
        for kt in [0, 1]:
            init_method = get_distributed_init_method(get_ip(), get_open_port())
            try:
                latency_us, max_err = run_one(size, kt, init_method)
            except Exception as e:
                logger.error(
                    "size=%s kernel=%s FAILED: %s", _size_str(size), kernel_names[kt], e
                )
                continue

            input_bytes = size * 2  # bf16 = 2 bytes
            bw_gbs = input_bytes / (latency_us * 1e-6) / 1e9

            rows.append(
                {
                    "size": _size_str(size),
                    "kernel": kernel_names[kt],
                    "latency_us": f"{latency_us:.1f}",
                    "xgmi_bw_gbs": f"{bw_gbs:.2f}",
                    "err": f"{max_err:.2e}",
                }
            )
            logger.info(
                "size=%s kernel=%-10s latency=%7.1f us  xgmi_bw=%6.2f GB/s  err=%s",
                _size_str(size),
                kernel_names[kt],
                latency_us,
                bw_gbs,
                f"{max_err:.2e}",
            )

    # Print final table
    header = f"{'size':>8s} | {'kernel':>10s} | {'latency(us)':>11s} | {'XGMI BW(GB/s)':>13s} | {'err':>10s}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['size']:>8s} | {r['kernel']:>10s} | {r['latency_us']:>11s} | {r['xgmi_bw_gbs']:>13s} | {r['err']:>10s}"
        )
    print(sep)
