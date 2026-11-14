# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Two-rank same-stream benchmark for MoonEP dispatch/prefetch overlap.

Run on one gfx950 node:

    torchrun --standalone --nproc-per-node=2 \
        op_tests/benchmark_moonep_gfx950.py
"""

from __future__ import annotations

import os

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.moonep_dispatch_op import (
    MoonEPPreplannedDispatchOp,
)
from aiter.ops.flydsl.kernels.moonep_weight_prefetch import (
    MoonEPWeightPrefetchOp,
)
from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_reference_plan


def _share_shmem_unique_id(rank: int) -> bytes:
    objects = [ms.shmem_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objects, src=0)
    unique_id = objects[0]
    assert isinstance(unique_id, bytes) and len(unique_id) == 128
    return unique_id


def _measure_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")
    status = ms.shmem_init_attr(
        ms.MORI_SHMEM_INIT_WITH_UNIQUEID,
        rank,
        world_size,
        _share_shmem_unique_id(rank),
    )
    assert status == 0

    dispatch_op = None
    prefetch_op = None
    try:
        tokens = int(os.environ.get("MOONEP_BENCH_TOKENS", "1024"))
        hidden_dim = int(os.environ.get("MOONEP_BENCH_H", "2048"))
        iterations = int(os.environ.get("MOONEP_BENCH_ITERS", "20"))
        default_blocks = os.environ.get("MOONEP_BENCH_BLOCKS", "64")
        dispatch_blocks = int(
            os.environ.get("MOONEP_BENCH_DISPATCH_BLOCKS", default_blocks)
        )
        prefetch_blocks = int(
            os.environ.get("MOONEP_BENCH_PREFETCH_BLOCKS", default_blocks)
        )
        config = MoonEPPlanConfig(
            rank=rank,
            world_size=world_size,
            num_tokens=tokens,
            top_k=1,
            num_experts=4,
            prefetch_slots=2,
        )
        topk = torch.zeros((tokens, 1), dtype=torch.int32, device="cuda")
        tokens_per_expert = torch.tensor(
            [[tokens, 0, 0, 0], [tokens, 0, 0, 0]],
            dtype=torch.int32,
            device="cuda",
        )
        plan = build_reference_plan(config, topk, tokens_per_expert)
        hidden = torch.randn(
            tokens, hidden_dim, dtype=torch.bfloat16, device="cuda"
        )
        route_weights = torch.ones(
            tokens, 1, dtype=torch.float32, device="cuda"
        )

        dispatch_op = MoonEPPreplannedDispatchOp(
            config, hidden_dim, block_num=dispatch_blocks
        )
        prefetch_op = MoonEPWeightPrefetchOp(
            rank=rank,
            world_size=world_size,
            num_experts=config.num_experts,
            prefetch_slots=int(config.prefetch_slots),
            weight_shape=(hidden_dim, hidden_dim),
            block_num=prefetch_blocks,
            block_threads=256,
        )
        home_weights = torch.full(
            (config.experts_per_rank, hidden_dim, hidden_dim),
            rank + 1,
            dtype=torch.bfloat16,
            device="cuda",
        )
        prefetch_op.load_home_weights(home_weights)

        def sequential() -> None:
            dispatch_op.dispatch(hidden, route_weights, plan)
            prefetch_op.prefetch(plan.experts_to_copy)

        def combined() -> None:
            dispatch_op.dispatch_and_prefetch(
                hidden, route_weights, plan, prefetch_op
            )

        for _ in range(3):
            sequential()
            combined()
        torch.cuda.synchronize()

        sequential_ms = _measure_ms(sequential, iterations)
        combined_ms = _measure_ms(combined, iterations)
        timings = torch.tensor(
            [sequential_ms, combined_ms], dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX)
        if rank == 0:
            sequential_ms, combined_ms = timings.cpu().tolist()
            hidden_mib = tokens * hidden_dim * 2 / (1 << 20)
            weight_mib = hidden_dim * hidden_dim * 2 / (1 << 20)
            print(
                f"tokens={tokens} H={hidden_dim} "
                f"dispatch_blocks={dispatch_blocks} "
                f"prefetch_blocks={prefetch_blocks} "
                f"hidden={hidden_mib:.1f}MiB remote_weight={weight_mib:.1f}MiB"
            )
            print(f"same_stream_sequential_ms={sequential_ms:.4f}")
            print(f"single_grid_combined_ms={combined_ms:.4f}")
            print(f"speedup={sequential_ms / combined_ms:.3f}x")
    finally:
        if prefetch_op is not None:
            prefetch_op.close()
        if dispatch_op is not None:
            dispatch_op.close()
        ms.shmem_barrier_all()
        ms.shmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
