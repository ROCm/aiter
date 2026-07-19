# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Stress custom all-reduce input publication across ranks."""

import argparse
import time
from multiprocessing import Pool, freeze_support, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)

_EAGER_SHAPES = ((4096, 7168), (1, 7168), (4096, 7168), (1, 7168))
_VALUES = (1, -2, 3, -4)
_GRAPH_SHAPE = (4, 7168)


def _check_across_ranks(message, group, device):
    failed = torch.tensor(message is not None, dtype=torch.int32, device=device)
    dist.all_reduce(failed, group=group)
    if failed.item():
        raise AssertionError(message or "custom all-reduce failed on a peer rank")


def _run_eager_reuse(ca_comm, group, device, rank, world_size, iters, skew_s):
    inputs = {
        shape: torch.empty(shape, dtype=torch.bfloat16, device=device)
        for shape in set(_EAGER_SHAPES)
    }
    for shape, input_ in inputs.items():
        assert ca_comm.should_custom_ar(
            input_
        ), f"custom all-reduce does not accept eager regression {shape=}"

    rank_sum = world_size * (world_size + 1) // 2
    for iteration in range(iters):
        pending = []
        for call_idx, shape in enumerate(_EAGER_SHAPES):
            value = _VALUES[(iteration + call_idx) % len(_VALUES)]
            if rank == (iteration + call_idx) % world_size:
                time.sleep(skew_s)

            input_ = inputs[shape]
            input_.fill_(value * (rank + 1))
            output = ca_comm.custom_all_reduce(input_)
            assert output is not None
            pending.append((output, value * rank_sum, call_idx, shape))

        torch.cuda.synchronize()
        message = None
        for output, expected, call_idx, shape in pending:
            bad = torch.count_nonzero(output != expected).item()
            if bad:
                message = (
                    "eager custom all-reduce IPC publication mismatch: "
                    f"iteration={iteration} call={call_idx} rank={rank} "
                    f"shape={shape} expected={expected} bad={bad} "
                    f"min={output.min().item()} max={output.max().item()}"
                )
                break
        _check_across_ranks(message, group, device)


def _run_graph_reuse(ca_comm, group, device, rank, world_size, iters, skew_s):
    input_ = torch.empty(_GRAPH_SHAPE, dtype=torch.bfloat16, device=device)
    assert ca_comm.should_custom_ar(input_)
    input_value = rank + 1
    clobber_value = -(rank + 17)
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture:
        with torch.cuda.graph(graph, stream=capture.stream):
            input_.fill_(input_value)
            output = ca_comm.custom_all_reduce(input_)
            assert output is not None
            input_.fill_(clobber_value)

    expected = world_size * (world_size + 1) // 2
    for iteration in range(iters):
        if rank == iteration % world_size:
            time.sleep(skew_s)
        graph.replay()
        torch.cuda.synchronize()

        bad_output = torch.count_nonzero(output != expected).item()
        bad_clobber = torch.count_nonzero(input_ != clobber_value).item()
        message = None
        if bad_output:
            message = (
                "graph custom all-reduce publication mismatch: "
                f"iteration={iteration} rank={rank} expected={expected} "
                f"bad={bad_output} min={output.min().item()} "
                f"max={output.max().item()}"
            )
        elif bad_clobber:
            message = (
                "graph input clobber did not execute: "
                f"iteration={iteration} rank={rank} bad={bad_clobber}"
            )
        _check_across_ranks(message, group, device)


def _worker(rank, world_size, iters, skew_s, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(world_size, 1)

    try:
        tp_group = get_tp_group()
        group = tp_group.device_group
        dist.all_reduce(torch.zeros(1, device=device), group=group)
        torch.cuda.synchronize()

        device_communicator = tp_group.device_communicator
        assert device_communicator is not None
        ca_comm = device_communicator.ca_comm
        assert ca_comm is not None and not ca_comm.disabled

        _run_eager_reuse(ca_comm, group, device, rank, world_size, iters, skew_s)
        if ca_comm.enable_register_for_capturing:
            _run_graph_reuse(ca_comm, group, device, rank, world_size, iters, skew_s)
        return {"rank": rank, "iters": iters}
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rank-skew-ms", type=float, default=2.0)
    parser.add_argument("--tp-size", type=int, choices=(2, 4, 6, 8), default=8)
    args = parser.parse_args()
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.rank_skew_ms < 0:
        parser.error("--rank-skew-ms must be non-negative")

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=args.tp_size) as pool:
        results = [
            pool.apply_async(
                _worker,
                args=(
                    rank,
                    args.tp_size,
                    args.iters,
                    args.rank_skew_ms / 1000.0,
                    init_method,
                ),
            )
            for rank in range(args.tp_size)
        ]
        print([result.get() for result in results])


if __name__ == "__main__":
    freeze_support()
    main()
