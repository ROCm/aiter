# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test script for Iris-based communication in AITER.

NOTE: This test is currently deprecated as all_reduce_iris has been removed.
The aiter.ops.triton.comms module now only provides:
- reduce_scatter_iris
- all_gather_iris  
- reduce_scatter_rmsnorm_quant_all_gather (fused kernel)

This file is kept for reference but will fail if run.
"""

import torch
import torch.distributed as dist
import os
import multiprocessing as mp
import sys
import traceback
import logging

import aiter
from aiter.test_common import checkAllclose, perftest
from aiter.dist.parallel_state import graph_capture
from aiter import dtypes
from aiter.ops.triton.comms import IrisCommContext, all_reduce_iris

logger = logging.getLogger("aiter")


def run_iris_all_reduce(tp_size, gpuID, input_tensor, heap_size=1 << 30):
    """
    Run Iris-based all-reduce on a single GPU.

    Args:
        tp_size: Tensor parallel size (number of GPUs)
        gpuID: GPU ID for this process
        input_tensor: Input tensor to reduce
        heap_size: Iris heap size in bytes

    Returns:
        Tuple of (output_tensor, time_in_us)
    """
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        input_tensor = input_tensor.to(device)
        torch.cuda.synchronize()
        dist.barrier()

        # Use Iris context for communication
        with IrisCommContext(heap_size=heap_size) as ctx:

            @perftest()
            def run_all_reduce(x):
                return all_reduce_iris(x, ctx=ctx)

            output, us = run_all_reduce(input_tensor)

        torch.cuda.synchronize()
        print(f"GPU {gpuID} finished in {us:.2f} us")
        return output.cpu(), us

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpuID}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        aiter.destroy_dist_env()


def test_iris_all_reduce(tp_size, shape, dtype, heap_size=1 << 30):
    """
    Test Iris-based all-reduce across multiple GPUs.

    Args:
        tp_size: Number of GPUs to use
        shape: Shape of input tensors
        dtype: Data type
        heap_size: Iris heap size in bytes
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    mp.set_start_method("spawn", force=True)

    # Create reference (sum of all inputs)
    ref = torch.zeros(shape, dtype=dtype)
    inputs = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        inputs.append(x)
        ref += x

    # Run all-reduce on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_iris_all_reduce, args=(tp_size, i, inputs[i], heap_size)
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get()[0] for r in results]
    times = [r.get()[1] for r in results]

    for i, (out, us) in enumerate(zip(outputs, times)):
        msg = f"test_iris_all_reduce: GPU {i}, shape={shape}, dtype={dtype}, time={us:>8.2f} us"
        checkAllclose(ref, out, msg=msg)

    avg_time = sum(times) / len(times)
    print(
        f"✓ test_iris_all_reduce passed: tp_size={tp_size}, shape={shape}, "
        f"dtype={dtype}, avg_time={avg_time:.2f} us\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Iris-based communication in AITER"
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        default=(128, 8192),
        help="Tensor shape (e.g., 128,8192)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type",
    )
    parser.add_argument(
        "-n", "--num_gpus", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--heap_size",
        type=int,
        default=1 << 30,
        help="Iris heap size in bytes (default: 1GB)",
    )

    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    mp.freeze_support()

    print("=" * 80)
    print("Testing Iris-based All-Reduce in AITER")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Shape: {args.shape}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Heap size: {args.heap_size / (1 << 30):.2f} GB")
    print("=" * 80)

    test_iris_all_reduce(
        tp_size=args.num_gpus, shape=args.shape, dtype=dtype, heap_size=args.heap_size
    )

    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)
