# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# XGMI P2P bandwidth test for gfx1250 (MI450).
# Pure remote-read + local-write kernel. TP2 only, bf16.
# Tests unroll = 2, 4, 8 across a range of input sizes.
#
# XGMI bandwidth = input_size_bytes / latency
#   (each rank reads input_size bytes from the remote GPU)

import os
from multiprocessing import Pool, set_start_method

import torch
import torch.distributed as dist

from aiter import logger
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import perftest

set_start_method("spawn", force=True)

_INIT_TIMEOUT_SEC = 120
_CASE_TIMEOUT_SEC = 10

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

_UNROLLS = [2, 4, 8]


def _size_str(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    return f"{n // 1024}K"


def _run_p2p_bw_case(ca_comm, group, size: int, unroll: int, threads: int):
    """Time the remote-read kernel for one ``(size, unroll)`` on an
    initialized rank. Returns ``latency_us``."""
    import aiter as ops

    _ptr = ca_comm._ptr
    inp = torch.randn(size, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(size, dtype=torch.bfloat16, device="cuda")

    reg_inp = ca_comm._pool["input"].data_ptr
    reg_inp_bytes = ca_comm._pool["input"].max_size

    d = 16 // 2  # bf16 pack_size
    packed = size // d
    blocks = min(512, (packed + threads * unroll - 1) // (threads * unroll))

    # Align all ranks so no rank starts timing while a peer is still setting up.
    dist.all_reduce(torch.zeros(1, device="cuda"), group=group)
    torch.cuda.synchronize()

    @perftest(use_cuda_event=True)
    def run_bw():
        ops.p2p_bw_test_gfx1250(
            _ptr,
            inp,
            out,
            unroll,
            threads,
            blocks,
            reg_inp,
            reg_inp_bytes,
        )

    _, latency_us = run_bw()
    return latency_us


def _p2p_bw_sweep(rank: int, cases: list, threads: int, distributed_init_method: str):
    """Run every ``(size, unroll)`` case on one rank inside a single
    distributed init.

    Setting up the TP group costs seconds per init vs. microseconds of kernel
    time, so the group is created and torn down once. A failing case is
    recorded as its error string instead of raised, so every rank still walks
    the full case list in lockstep.
    """
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

    results = []
    for size, unroll in cases:
        try:
            results.append(_run_p2p_bw_case(ca_comm, group, size, unroll, threads))
        except Exception as e:  # noqa: BLE001
            results.append(str(e))

    if dist.is_initialized():
        # Drain and align all ranks before freeing IPC buffers / groups.
        torch.cuda.synchronize()
        get_tp_group().barrier()
        torch.cuda.synchronize()
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    return results


def run_sweep(cases: list, threads: int, distributed_init_method: str):
    """Run all cases on one TP2 group; return one result list per rank."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    pool = Pool(processes=2)
    rets = []
    for rank in range(2):
        rets.append(
            pool.apply_async(
                _p2p_bw_sweep,
                args=(rank, cases, threads, distributed_init_method),
            )
        )
    pool.close()

    timeout = _INIT_TIMEOUT_SEC + _CASE_TIMEOUT_SEC * len(cases)
    results = []
    try:
        for r in rets:
            results.append(r.get(timeout=timeout))
    except Exception:
        pool.terminate()
        pool.join()
        raise
    pool.join()
    return results


if __name__ == "__main__":
    threads = 256

    cases = [(size, unroll) for size in _SIZES for unroll in _UNROLLS]
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    try:
        per_rank = run_sweep(cases, threads, init_method)
    except Exception as e:  # noqa: BLE001
        logger.error("p2p bw sweep FAILED: %s", e)
        per_rank = [["sweep failed"] * len(cases)]

    rows = []
    for i, (size, unroll) in enumerate(cases):
        rank_results = [results[i] for results in per_rank]
        failures = [r for r in rank_results if isinstance(r, str)]
        if failures:
            logger.error(
                "size=%s unroll=%d FAILED: %s",
                _size_str(size),
                unroll,
                failures[0],
            )
            continue
        latency_us = max(rank_results)

        input_bytes = size * 2  # bf16
        bw_gbs = input_bytes / (latency_us * 1e-6) / 1e9

        rows.append(
            {
                "size": _size_str(size),
                "unroll": str(unroll),
                "latency_us": f"{latency_us:.1f}",
                "xgmi_bw_gbs": f"{bw_gbs:.2f}",
            }
        )
        logger.info(
            "size=%s unroll=%d  latency=%7.1f us  xgmi_bw=%6.2f GB/s",
            _size_str(size),
            unroll,
            latency_us,
            bw_gbs,
        )

    # Print final table
    header = (
        f"{'size':>8s} | {'unroll':>6s} | {'latency(us)':>11s} | {'XGMI BW(GB/s)':>13s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['size']:>8s} | {r['unroll']:>6s} | {r['latency_us']:>11s} | {r['xgmi_bw_gbs']:>13s}"
        )
    print(sep)
