# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Test script to verify RECORD_PARAM_COMMS instrumentation works for all collective ops.

This script profiles all custom collective operations and exports a chrome trace
to verify that 'record_param_comms' events appear in the profiler output.

Tested operations:
    1. tensor_model_parallel_all_reduce
    2. tensor_model_parallel_fused_allreduce_rmsnorm
    3. tensor_model_parallel_reduce_scatter
    4. tensor_model_parallel_all_gather (with use_custom=True)

Usage (supports both methods):
    # Method 1: Direct run (automatically spawns processes)
    python test_collective_profile.py

    # Method 2: Using torchrun
    torchrun --nproc_per_node=8 test_collective_profile.py

After running, check the generated trace file (e.g., trace_rank0.json)
and search for "record_param_comms" events.
"""

import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity

from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_reduce_scatter,
    tensor_model_parallel_all_gather,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)


def run_worker(local_rank, world_size):
    """Worker function for each GPU process"""
    # Set environment variables for this worker
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Initialize process group
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(world_size, 1)

    # Create initial test tensors
    shape = (128, 8192)
    hidden_size = shape[1]

    # Warmup
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    # Profile all collective operations
    trace_file = f"trace_rank{local_rank}.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(3):  # Run 3 iterations
            # 1. AllReduce
            x1 = torch.randn(shape, dtype=torch.float16, device=device)
            out1 = tensor_model_parallel_all_reduce(x1)
            torch.cuda.synchronize()

            # 2. Fused AllReduce + RMSNorm
            # 直接使用 out1 作为输入
            residual = torch.randn(shape, dtype=torch.float16, device=device)
            weight = torch.randn(hidden_size, dtype=torch.float16, device=device)
            eps = 1e-5
            out2, residual_out = tensor_model_parallel_fused_allreduce_rmsnorm(
                out1, residual, weight, eps
            )
            torch.cuda.synchronize()

            # 3. Reduce-Scatter
            # 直接使用 out2 作为输入
            out3 = tensor_model_parallel_reduce_scatter(out2, use_custom=True)
            torch.cuda.synchronize()

            # 4. All-Gather
            # 直接使用 out3 作为输入
            out4 = tensor_model_parallel_all_gather(out3, use_custom=True)
            torch.cuda.synchronize()

    # Export chrome trace
    prof.export_chrome_trace(trace_file)

    if local_rank == 0:
        print(f"\nTrace exported to: {trace_file}")

        # Check if record_param_comms events are present
        with open(trace_file, "r") as f:
            trace_data = json.load(f)

        record_param_comms_events = [
            e
            for e in trace_data.get("traceEvents", [])
            if e.get("name") == "record_param_comms"
        ]

        if record_param_comms_events:
            print(
                f"\n✓ SUCCESS: Found {len(record_param_comms_events)} 'record_param_comms' events!"
            )

            # Count events by operation type
            op_counts = {}
            for event in record_param_comms_events:
                args = event.get("args", {})
                coll_name = args.get("collective_name", "unknown")
                op_counts[coll_name] = op_counts.get(coll_name, 0) + 1

            print("\nEvents by operation type:")
            for op_name, count in sorted(op_counts.items()):
                print(f"  {op_name}: {count}")

            print("\nSample event metadata:")
            sample = record_param_comms_events[0]
            print(json.dumps(sample, indent=2))
        else:
            print("\n✗ WARNING: No 'record_param_comms' events found in trace.")
            print("  This may indicate the instrumentation is not working.")

    # Cleanup
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()


def main():
    # Check if we're already in a torchrun environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running under torchrun, use environment variables directly
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        run_worker(local_rank, world_size)
    else:
        # Not in torchrun, spawn processes manually
        world_size = torch.cuda.device_count()
        if world_size < 2:
            print(f"Error: Need at least 2 GPUs, found {world_size}")
            return

        print(f"Spawning {world_size} processes for {world_size} GPUs...")
        mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
