#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test script for Iris-based reduce-scatter and all-gather communication primitives.

This script demonstrates how to use the reduce_scatter and all_gather
functions for multi-GPU communication in aiter.

Usage:
    # Run with default settings (2 GPUs)
    python test_reduce_scatter_all_gather.py

    # Run with 4 GPUs
    python test_reduce_scatter_all_gather.py -n 4

    # Run specific test
    python test_reduce_scatter_all_gather.py --test reduce_scatter
"""

import os
import torch
import torch.distributed as dist
import multiprocessing as mp
import sys
import traceback
import logging

import aiter
from aiter.test_common import ensure_spawn_method

# Import AITER Iris communication
try:
    from aiter.ops.triton.comms import (
        IrisCommContext,
        reduce_scatter,
        all_gather,
    )

    IRIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Iris communication: {e}")
    IRIS_AVAILABLE = False

logger = logging.getLogger("aiter")


def run_reduce_scatter_test(tp_size, gpuID, M, N, heap_size=1 << 30):
    """
    Run reduce-scatter test on a single GPU.

    Args:
        tp_size: Number of GPUs (world size)
        gpuID: GPU ID for this process
        M: Number of rows
        N: Number of columns
        heap_size: Iris heap size in bytes

    Returns:
        Tuple of (output_shard, expected_mean, actual_mean)
    """
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(
            f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
        )

        torch.cuda.synchronize()
        dist.barrier()

        # Create Iris communication context
        with IrisCommContext(heap_size=heap_size) as ctx:
            # Create input tensor on GPU first, then copy to Iris shared memory
            # Each rank creates the same shape but different values
            input_tensor_host = torch.ones(
                (M, N), dtype=torch.float32, device=device
            ) * (rank + 1)
            input_tensor = ctx.iris_ctx.empty((M, N), dtype=torch.float32)
            input_tensor.copy_(input_tensor_host)

            torch.cuda.synchronize()
            dist.barrier()

            print(
                f"[Rank {rank}] Input tensor shape: {input_tensor.shape}, mean: {input_tensor.mean():.2f}"
            )

            # Perform reduce-scatter
            output_shard = reduce_scatter(input_tensor, ctx)

            M_shard = M // world_size
            print(
                f"[Rank {rank}] Output shard shape: {output_shard.shape}, expected: ({M_shard}, {N})"
            )

            # Verify the result
            # Expected: sum of all ranks = 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
            expected_mean = world_size * (world_size + 1) / 2
            actual_mean = output_shard.mean().item()

            print(
                f"[Rank {rank}] Output shard mean: {actual_mean:.2f}, expected: {expected_mean:.2f}"
            )

            # Copy result to CPU
            output_cpu = output_shard.cpu()

        torch.cuda.synchronize()
        return output_cpu, expected_mean, actual_mean

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpuID}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        aiter.destroy_dist_env()


def run_all_gather_test(tp_size, gpuID, M_shard, N, heap_size=1 << 30):
    """
    Run all-gather test on a single GPU.

    Args:
        tp_size: Number of GPUs (world size)
        gpuID: GPU ID for this process
        M_shard: Number of rows per shard
        N: Number of columns
        heap_size: Iris heap size in bytes

    Returns:
        Tuple of (full_tensor, rank_means)
    """
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(
            f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
        )

        torch.cuda.synchronize()
        dist.barrier()

        # Create Iris communication context
        with IrisCommContext(heap_size=heap_size) as ctx:
            # Create input shard on GPU first, then copy to Iris shared memory
            # Each rank has different values
            input_shard_host = torch.ones(
                (M_shard, N), dtype=torch.float32, device=device
            ) * (rank + 1)
            input_shard = ctx.iris_ctx.empty((M_shard, N), dtype=torch.float32)
            input_shard.copy_(input_shard_host)

            torch.cuda.synchronize()
            dist.barrier()

            print(
                f"[Rank {rank}] Input shard shape: {input_shard.shape}, mean: {input_shard.mean():.2f}"
            )

            # Perform all-gather
            full_tensor = all_gather(input_shard, ctx)

            M = M_shard * world_size
            print(
                f"[Rank {rank}] Full tensor shape: {full_tensor.shape}, expected: ({M}, {N})"
            )

            # Verify the result
            print(f"[Rank {rank}] Full tensor mean: {full_tensor.mean():.2f}")

            # Check each segment
            rank_means = []
            for r in range(world_size):
                segment = full_tensor[r * M_shard : (r + 1) * M_shard, :]
                actual_mean = segment.mean().item()
                rank_means.append(actual_mean)
                print(f"[Rank {rank}] Segment from rank {r}: mean={actual_mean:.2f}")

            # Copy result to CPU
            full_cpu = full_tensor.cpu()

        torch.cuda.synchronize()
        return full_cpu, rank_means

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpuID}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        aiter.destroy_dist_env()


def run_round_trip_test(tp_size, gpuID, M, N, heap_size=1 << 30):
    """
    Run reduce-scatter + all-gather round trip test on a single GPU.

    Args:
        tp_size: Number of GPUs (world size)
        gpuID: GPU ID for this process
        M: Number of rows
        N: Number of columns
        heap_size: Iris heap size in bytes

    Returns:
        Tuple of (reconstructed_tensor, expected_mean, actual_mean)
    """
    try:
        device = torch.device(f"cuda:{gpuID}")
        torch.cuda.set_device(device)
        aiter.init_dist_env(tp_size, gpuID)

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(
            f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
        )

        torch.cuda.synchronize()
        dist.barrier()

        # Create Iris communication context
        with IrisCommContext(heap_size=heap_size) as ctx:
            # Create input tensor on GPU first, then copy to Iris shared memory
            input_tensor_host = torch.ones(
                (M, N), dtype=torch.float32, device=device
            ) * (rank + 1)
            input_tensor = ctx.iris_ctx.empty((M, N), dtype=torch.float32)
            input_tensor.copy_(input_tensor_host)

            torch.cuda.synchronize()
            dist.barrier()

            original_mean = input_tensor.mean().item()

            print(
                f"[Rank {rank}] Original tensor shape: {input_tensor.shape}, mean: {original_mean:.2f}"
            )

            # Step 1: Reduce-scatter
            output_shard = reduce_scatter(input_tensor, ctx)
            print(
                f"[Rank {rank}] After reduce-scatter, shard shape: {output_shard.shape}"
            )

            # Step 2: All-gather
            reconstructed_tensor = all_gather(output_shard, ctx)
            print(
                f"[Rank {rank}] After all-gather, tensor shape: {reconstructed_tensor.shape}"
            )

            # Verify: all ranks should have the same tensor now
            # Expected: sum of all ranks = world_size * (world_size + 1) / 2
            expected_mean = world_size * (world_size + 1) / 2
            actual_mean = reconstructed_tensor.mean().item()

            print(
                f"[Rank {rank}] Final tensor mean: {actual_mean:.2f}, expected: {expected_mean:.2f}"
            )

            # Copy result to CPU
            reconstructed_cpu = reconstructed_tensor.cpu()

        torch.cuda.synchronize()
        return reconstructed_cpu, expected_mean, actual_mean

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpuID}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        aiter.destroy_dist_env()


def test_reduce_scatter(tp_size, M=8192, N=7168, heap_size=1 << 30):
    """Test reduce-scatter operation."""
    if M % tp_size != 0:
        raise ValueError(f"M ({M}) must be divisible by tp_size ({tp_size})")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49376"
    ensure_spawn_method()

    # Run on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_reduce_scatter_test,
                args=(tp_size, i, M, N, heap_size),
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get() for r in results]

    print(f"\n{'='*80}")
    print("Checking Reduce-Scatter Results")
    print(f"{'='*80}")

    all_passed = True
    for i, (output_shard, expected_mean, actual_mean) in enumerate(outputs):
        if abs(actual_mean - expected_mean) < 0.01:
            print(f"? GPU {i}: Reduce-scatter PASSED (mean={actual_mean:.2f})")
        else:
            print(
                f"? GPU {i}: Reduce-scatter FAILED (mean={actual_mean:.2f}, expected={expected_mean:.2f})"
            )
            all_passed = False

    if all_passed:
        print(f"\n? test_reduce_scatter passed: tp_size={tp_size}, M={M}, N={N}\n")
    else:
        raise AssertionError("Reduce-scatter test failed")


def test_all_gather(tp_size, M_shard=1024, N=7168, heap_size=1 << 30):
    """Test all-gather operation."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49377"
    ensure_spawn_method()

    # Run on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_all_gather_test,
                args=(tp_size, i, M_shard, N, heap_size),
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get() for r in results]

    print(f"\n{'='*80}")
    print("Checking All-Gather Results")
    print(f"{'='*80}")

    all_passed = True
    for i, (full_tensor, rank_means) in enumerate(outputs):
        # Check each segment
        segment_passed = True
        for r in range(tp_size):
            expected_value = r + 1
            actual_mean = rank_means[r]
            if abs(actual_mean - expected_value) < 0.01:
                print(
                    f"? GPU {i}: Segment from rank {r} correct (mean={actual_mean:.2f})"
                )
            else:
                print(
                    f"? GPU {i}: Segment from rank {r} incorrect (mean={actual_mean:.2f}, expected={expected_value})"
                )
                segment_passed = False

        if not segment_passed:
            all_passed = False

    if all_passed:
        print(
            f"\n? test_all_gather passed: tp_size={tp_size}, M_shard={M_shard}, N={N}\n"
        )
    else:
        raise AssertionError("All-gather test failed")


def test_round_trip(tp_size, M=8192, N=7168, heap_size=1 << 30):
    """Test reduce-scatter + all-gather round trip."""
    if M % tp_size != 0:
        raise ValueError(f"M ({M}) must be divisible by tp_size ({tp_size})")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49378"
    ensure_spawn_method()

    # Run on each GPU
    pool = mp.Pool(processes=tp_size)
    results = []
    for i in range(tp_size):
        results.append(
            pool.apply_async(
                run_round_trip_test,
                args=(tp_size, i, M, N, heap_size),
            )
        )
    pool.close()
    pool.join()

    # Check results
    outputs = [r.get() for r in results]

    print(f"\n{'='*80}")
    print("Checking Round-Trip Results")
    print(f"{'='*80}")

    all_passed = True
    for i, (reconstructed, expected_mean, actual_mean) in enumerate(outputs):
        if abs(actual_mean - expected_mean) < 0.01 and reconstructed.shape == (M, N):
            print(f"? GPU {i}: Round-trip PASSED (mean={actual_mean:.2f})")
        else:
            print(
                f"? GPU {i}: Round-trip FAILED (mean={actual_mean:.2f}, expected={expected_mean:.2f})"
            )
            all_passed = False

    if all_passed:
        print(f"\n? test_round_trip passed: tp_size={tp_size}, M={M}, N={N}\n")
    else:
        raise AssertionError("Round-trip test failed")


if __name__ == "__main__":
    import argparse

    print("Script started!", flush=True)

    parser = argparse.ArgumentParser(
        description="Test Iris-based reduce-scatter and all-gather"
    )
    parser.add_argument(
        "--test",
        choices=["reduce_scatter", "all_gather", "round_trip", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "-M", "--num_rows", type=int, default=8192, help="Number of rows (M)"
    )
    parser.add_argument(
        "-N", "--num_cols", type=int, default=7168, help="Number of columns (N)"
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

    args = parser.parse_args()

    if not IRIS_AVAILABLE:
        print("ERROR: Iris library not available. Please install iris:")
        print('  pip install -e ".[triton_comms]"')
        sys.exit(1)

    # Ensure M is divisible by tp_size for reduce_scatter and round_trip tests
    if args.num_rows % args.num_gpus != 0:
        print(
            f"Warning: M ({args.num_rows}) is not divisible by tp_size ({args.num_gpus})"
        )
        args.num_rows = (args.num_rows // args.num_gpus) * args.num_gpus
        print(f"Adjusting M to {args.num_rows}")

    mp.freeze_support()

    print("=" * 80)
    print("Testing Iris-Based Reduce-Scatter and All-Gather")
    print("=" * 80)
    print("Configuration:")
    print(f"  M (rows): {args.num_rows}")
    print(f"  N (cols): {args.num_cols}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Heap size: {args.heap_size / (1 << 30):.2f} GB")
    print(f"  Test mode: {args.test}")
    print("=" * 80)

    try:
        if args.test == "reduce_scatter" or args.test == "all":
            print("\n" + "=" * 80)
            print("TEST: Reduce-Scatter")
            print("=" * 80)
            test_reduce_scatter(
                tp_size=args.num_gpus,
                M=args.num_rows,
                N=args.num_cols,
                heap_size=args.heap_size,
            )

        if args.test == "all_gather" or args.test == "all":
            print("\n" + "=" * 80)
            print("TEST: All-Gather")
            print("=" * 80)
            M_shard = args.num_rows // args.num_gpus
            test_all_gather(
                tp_size=args.num_gpus,
                M_shard=M_shard,
                N=args.num_cols,
                heap_size=args.heap_size,
            )

        if args.test == "round_trip" or args.test == "all":
            print("\n" + "=" * 80)
            print("TEST: Reduce-Scatter + All-Gather (Round Trip)")
            print("=" * 80)
            test_round_trip(
                tp_size=args.num_gpus,
                M=args.num_rows,
                N=args.num_cols,
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
