# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for gfx1250 custom-allreduce input publication.

The regular correctness test synchronizes after each invocation. That hides a
barrier bug because the registered input pool is not reused while peer reads
are still in flight. This test queues many collectives with changing inputs
before synchronizing, forcing repeated reuse of the same registered pool.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool, set_start_method

os.environ.setdefault("ENABLE_CK", "0")

import torch

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

set_start_method("spawn", force=True)


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    iterations: int,
    numel: int,
    with_graph: bool,
) -> tuple[float, int]:
    torch.cuda.set_device(rank)

    from aiter.dist.communication_op import tensor_model_parallel_all_reduce
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=init_method,
    )
    ensure_model_parallel_initialized(world_size, 1)

    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled

    inp = torch.empty(numel, dtype=torch.float16, device=f"cuda:{rank}")
    assert ca.should_custom_ar(inp)
    assert inp.nbytes > 4 * 1024 * 1024, "test input must bypass the LL protocol"

    graph = None
    graph_out = None
    if with_graph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as capture, torch.cuda.graph(graph, stream=capture.stream):
            graph_out = tensor_model_parallel_all_reduce(inp)
        torch.cuda.synchronize()

    outputs: list[tuple[torch.Tensor, float]] = []
    for iteration in range(iterations):
        local_value = float((iteration * 7 + rank * 13) % 97)
        expected = sum(
            float((iteration * 7 + peer_rank * 13) % 97)
            for peer_rank in range(world_size)
        )
        inp.fill_(local_value)
        if graph is None:
            out = tensor_model_parallel_all_reduce(inp)
        else:
            graph.replay()
            # The captured graph reuses one output address. Queue a D2D snapshot
            # after every replay so every result remains independently testable.
            assert graph_out is not None
            out = torch.empty_like(graph_out)
            out.copy_(graph_out)
        outputs.append((out, expected))

    # Synchronize only after all launches. Retaining every output prevents
    # allocator reuse and lets us verify every queued collective independently.
    torch.cuda.synchronize()
    errors = [(out.float() - expected).abs().max() for out, expected in outputs]
    error_values = torch.stack(errors).cpu()
    max_error = error_values.max().item()
    failures = torch.count_nonzero(error_values).item()

    destroy_model_parallel()
    destroy_distributed_environment()
    print(
        f"rank={rank} iterations={iterations} "
        f"max_error={max_error} failures={failures}",
        flush=True,
    )
    return max_error, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--with-graph", action="store_true")
    parser.add_argument(
        "--numel",
        type=int,
        default=512 * 8192,
        help="FP16 elements; default is 8 MiB and bypasses the 4 MiB LL path",
    )
    args = parser.parse_args()

    world_size = 2
    if torch.cuda.device_count() < world_size:
        print("SKIP: two GPUs are required")
        return 0

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=world_size) as pool:
        jobs = [
            pool.apply_async(
                _worker,
                (
                    rank,
                    world_size,
                    init_method,
                    args.iterations,
                    args.numel,
                    args.with_graph,
                ),
            )
            for rank in range(world_size)
        ]
        results = [job.get(timeout=600) for job in jobs]

    max_error = max(error for error, _ in results)
    failures = sum(count for _, count in results)
    print(f"max_error={max_error} total_failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
