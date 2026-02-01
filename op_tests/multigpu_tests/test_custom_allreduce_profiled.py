# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Test script to verify RECORD_PARAM_COMMS instrumentation works.

This script profiles the custom allreduce operation and exports a chrome trace
to verify that 'record_param_comms' events appear in the profiler output.

Usage (requires multi-GPU setup):
    torchrun --nproc_per_node=2 test_custom_allreduce_profiled.py

After running, check the generated trace file (e.g., trace_rank0.json) 
and search for "record_param_comms" events.
"""

import os
import json
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity

from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)


def main():
    # Initialize distributed environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=local_rank,
        distributed_init_method="env://",
    )
    ensure_model_parallel_initialized(world_size, 1)
    
    # Create test tensor
    shape = (128, 8192)
    x = torch.randn(shape, dtype=torch.float16, device=device)
    
    # Warmup
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()
    
    # Profile the allreduce operation
    trace_file = f"trace_rank{local_rank}.json"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(5):
            out = tensor_model_parallel_all_reduce(x)
            torch.cuda.synchronize()
    
    # Export chrome trace
    prof.export_chrome_trace(trace_file)
    
    if local_rank == 0:
        print(f"\nTrace exported to: {trace_file}")
        
        # Check if record_param_comms events are present
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
        
        record_param_comms_events = [
            e for e in trace_data.get("traceEvents", [])
            if e.get("name") == "record_param_comms"
        ]
        
        if record_param_comms_events:
            print(f"\n✓ SUCCESS: Found {len(record_param_comms_events)} 'record_param_comms' events!")
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
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
