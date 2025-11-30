#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test script for Iris-based reduce-scatter and all-gather communication primitives.

This script demonstrates how to use the reduce_scatter_iris and all_gather_iris
functions for multi-GPU communication in AITER.

Usage:
    # Run with torchrun for 8 GPUs
    torchrun --nproc_per_node=8 test_reduce_scatter_all_gather.py

    # For 4 GPUs
    torchrun --nproc_per_node=4 test_reduce_scatter_all_gather.py
"""

import os
import torch
import torch.distributed as dist

# Import AITER Iris communication
try:
    from aiter.ops.triton.comms import (
        IrisCommContext,
        reduce_scatter_iris,
        all_gather_iris,
    )

    IRIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Iris communication: {e}")
    IRIS_AVAILABLE = False


def test_reduce_scatter(M=8192, N=7168):
    """
    Test reduce-scatter operation.

    Each rank starts with an M×N tensor.
    After reduce-scatter, each rank has (M/world_size)×N.
    """
    if not IRIS_AVAILABLE:
        print("Iris not available, skipping test")
        return

    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
    )

    # Create Iris communication context
    with IrisCommContext(heap_size=2**30) as ctx:
        shmem = ctx.iris_ctx.shmem

        # Create input tensor in Iris shared memory
        # Each rank creates the same shape but different values
        input_tensor = shmem.ones((M, N), dtype=torch.float32) * (rank + 1)

        print(
            f"[Rank {rank}] Input tensor shape: {input_tensor.shape}, mean: {input_tensor.mean():.2f}"
        )

        # Perform reduce-scatter
        output_shard = reduce_scatter_iris(input_tensor, ctx)

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

        # Check if the result is correct (allow small numerical error)
        if abs(actual_mean - expected_mean) < 0.01:
            print(f"[Rank {rank}] ✓ Reduce-scatter test PASSED")
        else:
            print(f"[Rank {rank}] ✗ Reduce-scatter test FAILED")

    # Clean up
    dist.destroy_process_group()


def test_all_gather(M_shard=1024, N=7168):
    """
    Test all-gather operation.

    Each rank starts with an (M_shard)×N tensor.
    After all-gather, each rank has (M_shard * world_size)×N.
    """
    if not IRIS_AVAILABLE:
        print("Iris not available, skipping test")
        return

    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
    )

    # Create Iris communication context
    with IrisCommContext(heap_size=2**30) as ctx:
        shmem = ctx.iris_ctx.shmem

        # Create input shard in Iris shared memory
        # Each rank has different values
        input_shard = shmem.ones((M_shard, N), dtype=torch.float32) * (rank + 1)

        print(
            f"[Rank {rank}] Input shard shape: {input_shard.shape}, mean: {input_shard.mean():.2f}"
        )

        # Perform all-gather
        full_tensor = all_gather_iris(input_shard, ctx)

        M = M_shard * world_size
        print(
            f"[Rank {rank}] Full tensor shape: {full_tensor.shape}, expected: ({M}, {N})"
        )

        # Verify the result
        # Each rank should have the full tensor with segments from all ranks
        print(f"[Rank {rank}] Full tensor mean: {full_tensor.mean():.2f}")

        # Check each segment
        all_correct = True
        for r in range(world_size):
            segment = full_tensor[r * M_shard : (r + 1) * M_shard, :]
            expected_value = r + 1
            actual_mean = segment.mean().item()
            if abs(actual_mean - expected_value) < 0.01:
                print(
                    f"[Rank {rank}] ✓ Segment from rank {r} correct (mean: {actual_mean:.2f})"
                )
            else:
                print(
                    f"[Rank {rank}] ✗ Segment from rank {r} incorrect (mean: {actual_mean:.2f}, expected: {expected_value})"
                )
                all_correct = False

        if all_correct:
            print(f"[Rank {rank}] ✓ All-gather test PASSED")
        else:
            print(f"[Rank {rank}] ✗ All-gather test FAILED")

    # Clean up
    dist.destroy_process_group()


def test_reduce_scatter_all_gather_round_trip(M=8192, N=7168):
    """
    Test reduce-scatter followed by all-gather (round trip).

    This ensures that the two operations work correctly together.
    """
    if not IRIS_AVAILABLE:
        print("Iris not available, skipping test")
        return

    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    print(
        f"[Rank {rank}] Initialized distributed environment (world_size={world_size})"
    )

    # Create Iris communication context
    with IrisCommContext(heap_size=2**30) as ctx:
        shmem = ctx.iris_ctx.shmem

        # Create input tensor in Iris shared memory
        input_tensor = shmem.ones((M, N), dtype=torch.float32) * (rank + 1)
        original_mean = input_tensor.mean().item()

        print(
            f"[Rank {rank}] Original tensor shape: {input_tensor.shape}, mean: {original_mean:.2f}"
        )

        # Step 1: Reduce-scatter
        output_shard = reduce_scatter_iris(input_tensor, ctx)
        print(f"[Rank {rank}] After reduce-scatter, shard shape: {output_shard.shape}")

        # Step 2: All-gather
        reconstructed_tensor = all_gather_iris(output_shard, ctx)
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

        if abs(actual_mean - expected_mean) < 0.01 and reconstructed_tensor.shape == (
            M,
            N,
        ):
            print(f"[Rank {rank}] ✓ Round-trip test PASSED")
        else:
            print(f"[Rank {rank}] ✗ Round-trip test FAILED")

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Iris-based reduce-scatter and all-gather"
    )
    parser.add_argument(
        "--test",
        choices=["reduce_scatter", "all_gather", "round_trip", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--M", type=int, default=8192, help="Number of rows (M)")
    parser.add_argument("--N", type=int, default=7168, help="Number of columns (N)")

    args = parser.parse_args()

    if not IRIS_AVAILABLE:
        print("ERROR: Iris library not available. Please install iris:")
        print("  pip install git+https://github.com/ROCm/iris.git")
        exit(1)

    # Check environment variables
    if "RANK" not in os.environ:
        print("ERROR: This script must be run with torchrun")
        print("Usage: torchrun --nproc_per_node=8 test_reduce_scatter_all_gather.py")
        exit(1)

    if args.test == "reduce_scatter" or args.test == "all":
        print("\n" + "=" * 80)
        print("TEST: Reduce-Scatter")
        print("=" * 80)
        test_reduce_scatter(M=args.M, N=args.N)

    if args.test == "all_gather" or args.test == "all":
        print("\n" + "=" * 80)
        print("TEST: All-Gather")
        print("=" * 80)
        M_shard = args.M // int(os.environ["WORLD_SIZE"])
        test_all_gather(M_shard=M_shard, N=args.N)

    if args.test == "round_trip" or args.test == "all":
        print("\n" + "=" * 80)
        print("TEST: Reduce-Scatter + All-Gather (Round Trip)")
        print("=" * 80)
        test_reduce_scatter_all_gather_round_trip(M=args.M, N=args.N)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
