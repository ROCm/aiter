# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test script for fused reduce-scatter + RMSNorm + quantization + all-gather.

This script tests the fully fused Triton kernel that combines multiple operations
in a single kernel launch for optimal performance.
"""

import torch
import torch.distributed as dist
import os
import multiprocessing as mp
import sys
import traceback
import logging

import aiter
from aiter.test_common import checkAllclose, perftest, ensure_spawn_method
from aiter.ops.triton.comms import (
    IrisCommContext,
    reduce_scatter_rmsnorm_quant_all_gather,
)

logger = logging.getLogger("aiter")


def run_fused_kernel(
    tp_size,
    gpuID,
    input_data,
    gamma,
    M,
    N,
    heap_size=1 << 30,
    quant_mode="none",
    do_allgather=True,
):
    """
    Run fused reduce-scatter + RMSNorm + quant + all-gather on a single GPU.

    Args:
        tp_size: Tensor parallel size (number of GPUs)
        gpuID: GPU ID for this process
        input_data: Input tensor data (numpy or torch)
        gamma: RMSNorm weight
        M: Number of rows (must be divisible by tp_size)
        N: Number of columns
        heap_size: Iris heap size in bytes
        quant_mode: Quantization mode ("none" or "fp8_per_token")
        do_allgather: Whether to perform all-gather stage

    Returns:
        Tuple of (norm_output, quant_output, gather_output, time_in_us)
    """
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        # Move data to GPU
        input_tensor_host = input_data.to(device)
        gamma = gamma.to(device)

        torch.cuda.synchronize()
        dist.barrier()

        # Use Iris context for communication
        with IrisCommContext(heap_size=heap_size) as ctx:
            # Allocate input in Iris shared memory (required for reduce-scatter)
            input_tensor = ctx.iris_ctx.empty((M, N), dtype=input_tensor_host.dtype)
            input_tensor.copy_(input_tensor_host)

            # Pre-allocate buffers for reuse across iterations
            M_shard = M // tp_size
            dtype = input_tensor_host.dtype
            rs_buffer = ctx.iris_ctx.zeros((M_shard, N), dtype=dtype)
            norm_buffer = torch.empty((M_shard, N), dtype=torch.float32, device=device)

            if quant_mode == "fp8_per_token":
                fp8_dtype = getattr(torch, "float8_e4m3fn", torch.int8)
                fp8_out = ctx.iris_ctx.empty((M_shard, N), dtype=fp8_dtype)
                gather_dtype = fp8_dtype
            else:
                fp8_out = None
                gather_dtype = dtype

            if do_allgather:
                gather_out = ctx.iris_ctx.zeros((M, N), dtype=gather_dtype)
            else:
                gather_out = None

            # Ensure data is synchronized across GPUs
            ctx.iris_ctx.barrier()

            # Use fixed iterations for distributed tests to prevent deadlocks
            # (dynamic iteration count would differ across ranks due to different free memory)
            @perftest(num_rotate_args=10)
            def run_fused():
                return reduce_scatter_rmsnorm_quant_all_gather(
                    input_tensor,
                    gamma,
                    epsilon=1e-6,
                    ctx=ctx,
                    quant_mode=quant_mode,
                    do_allgather=do_allgather,
                    # Pass pre-allocated buffers for reuse
                    rs_buffer=rs_buffer,
                    norm_buffer=norm_buffer,
                    fp8_out=fp8_out,
                    gather_out=gather_out,
                )

            (norm_out, quant_out, gather_out), us = run_fused()

            # Copy results to CPU
            norm_cpu = norm_out.cpu() if norm_out is not None else None
            quant_cpu = quant_out.cpu() if quant_out is not None else None
            gather_cpu = gather_out.cpu() if gather_out is not None else None

        torch.cuda.synchronize()
        print(f"GPU {gpuID} finished in {us:.2f} us")
        return norm_cpu, quant_cpu, gather_cpu, us

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpuID}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        aiter.destroy_dist_env()


def reference_reduce_scatter_rmsnorm(input_tensors, gamma, epsilon, M_shard):
    """
    Reference implementation: reduce-scatter + RMSNorm.

    Args:
        input_tensors: List of input tensors from all ranks
        gamma: RMSNorm weight
        epsilon: RMSNorm epsilon
        M_shard: Shard size per rank

    Returns:
        List of normalized shards (one per rank)
    """
    # Reduce (sum all inputs)
    reduced = torch.zeros_like(input_tensors[0], dtype=torch.float32)
    for tensor in input_tensors:
        reduced += tensor.to(torch.float32)

    # Scatter (split into shards)
    world_size = len(input_tensors)
    shards = []
    for rank in range(world_size):
        shard = reduced[rank * M_shard : (rank + 1) * M_shard, :].to(
            input_tensors[0].dtype
        )
        shards.append(shard)

    # RMSNorm on each shard
    normed_shards = []
    for shard in shards:
        rmsnorm_layer = torch.nn.RMSNorm(
            shard.shape[1], eps=epsilon, dtype=shard.dtype, device=shard.device
        )
        rmsnorm_layer.weight.data.copy_(gamma)
        normed = rmsnorm_layer(shard)
        normed_shards.append(normed)

    return normed_shards


def test_fused_without_quant(
    tp_size, M, N, dtype, heap_size=1 << 30, do_allgather=True
):
    """
    Test fused kernel without quantization.

    Args:
        tp_size: Number of GPUs to use
        M: Number of rows (must be divisible by tp_size)
        N: Number of columns
        dtype: Data type
        heap_size: Iris heap size in bytes
        do_allgather: Whether to perform all-gather stage
    """
    if M % tp_size != 0:
        raise ValueError(f"M ({M}) must be divisible by tp_size ({tp_size})")

    M_shard = M // tp_size

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49374"
    ensure_spawn_method()

    # Create input data
    torch.manual_seed(42)
    gamma = torch.ones(N, dtype=dtype)
    inputs = []
    for i in range(tp_size):
        torch.manual_seed(42 + i)
        x = torch.randn(M, N, dtype=dtype) * (i + 1)
        inputs.append(x)

    # Compute reference
    epsilon = 1e-6
    ref_normed_shards = reference_reduce_scatter_rmsnorm(
        inputs, gamma, epsilon, M_shard
    )

    # Run fused kernel on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_fused_kernel,
                args=(
                    tp_size,
                    i,
                    inputs[i],
                    gamma,
                    M,
                    N,
                    heap_size,
                    "none",
                    do_allgather,
                ),
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get() for r in results]
    times = [out[3] for out in outputs]

    print(f"\n{'='*80}")
    print(f"Checking Results (without quantization, do_allgather={do_allgather})")
    print(f"{'='*80}")

    for i, (norm_out, quant_out, gather_out, us) in enumerate(outputs):
        # Check normalized shard
        # Note: norm_out is in float32 (for numerical accuracy), convert to input dtype for comparison
        ref_shard = ref_normed_shards[i]
        norm_out_converted = norm_out.to(dtype)
        msg = f"GPU {i}: Normalized shard, M={M}, N={N}, dtype={dtype}, time={us:>8.2f} us"
        checkAllclose(ref_shard, norm_out_converted, msg=msg, atol=1e-3, rtol=1e-3)

        # Quantization output should be None
        assert quant_out is None, f"GPU {i}: Expected quant_out=None"

        # Check all-gather output
        if do_allgather:
            assert (
                gather_out is not None
            ), f"GPU {i}: Expected gather_out to be not None"
            # All-gather should collect all shards
            ref_full = torch.cat(ref_normed_shards, dim=0)
            msg = f"GPU {i}: All-gather output, M={M}, N={N}, dtype={dtype}"
            checkAllclose(ref_full, gather_out, msg=msg, atol=1e-3, rtol=1e-3)
        else:
            assert gather_out is None, f"GPU {i}: Expected gather_out=None"

    avg_time = sum(times) / len(times)
    print(
        f"\n? test_fused_without_quant passed: tp_size={tp_size}, M={M}, N={N}, "
        f"dtype={dtype}, do_allgather={do_allgather}, avg_time={avg_time:.2f} us\n"
    )


def test_fused_with_fp8_quant(tp_size, M, N, dtype, heap_size=1 << 30):
    """
    Test fused kernel with FP8 quantization.

    Args:
        tp_size: Number of GPUs to use
        M: Number of rows (must be divisible by tp_size)
        N: Number of columns
        dtype: Data type
        heap_size: Iris heap size in bytes
    """
    if M % tp_size != 0:
        raise ValueError(f"M ({M}) must be divisible by tp_size ({tp_size})")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    ensure_spawn_method()

    # Create input data
    torch.manual_seed(42)
    gamma = torch.ones(N, dtype=dtype)
    inputs = []
    for i in range(tp_size):
        torch.manual_seed(42 + i)
        x = torch.randn(M, N, dtype=dtype) * (i + 1)
        inputs.append(x)

    # Run fused kernel on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_fused_kernel,
                args=(
                    tp_size,
                    i,
                    inputs[i],
                    gamma,
                    M,
                    N,
                    heap_size,
                    "fp8_per_token",
                    True,  # do_allgather
                ),
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get() for r in results]
    times = [out[3] for out in outputs]

    print(f"\n{'='*80}")
    print("Checking Results (with FP8 quantization)")
    print(f"{'='*80}")

    for i, (norm_out, quant_out, gather_out, us) in enumerate(outputs):
        # Norm output should be None (only quant output is available)
        assert norm_out is None, f"GPU {i}: Expected norm_out=None with quantization"

        # Check quantization output exists
        assert (
            quant_out is not None
        ), f"GPU {i}: Expected quant_out to be not None with fp8_per_token"

        # Check that quantization is actually done (dtype should be FP8 or int8)
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.int8)
        assert quant_out.dtype in [
            fp8_dtype,
            torch.int8,
        ], f"GPU {i}: Expected FP8 dtype, got {quant_out.dtype}"

        # Convert to float32 for min/max operations (float8 doesn't support these ops)
        quant_float = quant_out.to(torch.float32)
        print(
            f"GPU {i}: Quantized shard shape={quant_out.shape}, dtype={quant_out.dtype}, "
            f"range=[{quant_float.min().item():.2f}, {quant_float.max().item():.2f}], time={us:>8.2f} us"
        )

        # Check all-gather output
        assert gather_out is not None, f"GPU {i}: Expected gather_out to be not None"
        assert (
            gather_out.shape[0] == M
        ), f"GPU {i}: Expected gather shape [M, N], got {gather_out.shape}"

    avg_time = sum(times) / len(times)
    print(
        f"\n? test_fused_with_fp8_quant passed: tp_size={tp_size}, M={M}, N={N}, "
        f"dtype={dtype}, avg_time={avg_time:.2f} us\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test fused reduce-scatter + RMSNorm + quantization + all-gather"
    )
    parser.add_argument(
        "-M", "--num_rows", type=int, default=8192, help="Number of rows (M)"
    )
    parser.add_argument(
        "-N", "--num_cols", type=int, default=7168, help="Number of columns (N)"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="fp16",
        help="Data type",
    )
    parser.add_argument(
        "-n", "--num_gpus", type=int, default=2, help="Number of GPUs to use"
    )
    parser.add_argument(
        "--heap_size",
        type=int,
        default=2**31,
        help="Iris heap size in bytes (default: 2GB)",
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        choices=["all", "no_quant", "fp8_quant"],
        default="all",
        help="Test mode",
    )

    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Ensure M is divisible by tp_size
    if args.num_rows % args.num_gpus != 0:
        print(
            f"Warning: M ({args.num_rows}) is not divisible by tp_size ({args.num_gpus})"
        )
        args.num_rows = (args.num_rows // args.num_gpus) * args.num_gpus
        print(f"Adjusting M to {args.num_rows}")

    mp.freeze_support()

    print("=" * 80)
    print("Testing Fused Reduce-Scatter + RMSNorm + Quantization + All-Gather")
    print("=" * 80)
    print("Configuration:")
    print(f"  M (rows): {args.num_rows}")
    print(f"  N (cols): {args.num_cols}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Heap size: {args.heap_size / (1 << 30):.2f} GB")
    print(f"  Test mode: {args.test_mode}")
    print("=" * 80)

    try:
        if args.test_mode in ["all", "no_quant"]:
            # Test without quantization, with all-gather
            test_fused_without_quant(
                tp_size=args.num_gpus,
                M=args.num_rows,
                N=args.num_cols,
                dtype=dtype,
                heap_size=args.heap_size,
                do_allgather=True,
            )

            # Test without quantization, without all-gather
            test_fused_without_quant(
                tp_size=args.num_gpus,
                M=args.num_rows,
                N=args.num_cols,
                dtype=dtype,
                heap_size=args.heap_size,
                do_allgather=False,
            )

        if args.test_mode in ["all", "fp8_quant"]:
            # Test with FP8 quantization
            test_fused_with_fp8_quant(
                tp_size=args.num_gpus,
                M=args.num_rows,
                N=args.num_cols,
                dtype=dtype,
                heap_size=args.heap_size,
            )

        print("=" * 80)
        print("All tests passed!")
        print("=" * 80)

    except Exception as e:
        print("=" * 80)
        print(f"TEST FAILED: {str(e)}")
        print("=" * 80)
        traceback.print_exc()
        sys.exit(1)
