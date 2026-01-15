#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit test for two-stage allreduce: sdma_copy + part_reduce

Usage:
  python test_twostage_allreduce.py
  python test_twostage_allreduce.py --shape 128,8192 --num-chunks 8
"""

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing import Optional

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_sdma_copy,
    tensor_model_parallel_part_reduce,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

logger = logging.getLogger("test_twostage")
logging.basicConfig(level=logging.INFO)

set_start_method("spawn", force=True)


def twostage_allreduce_worker(
    tp_size: int,
    pp_size: int,
    rank_id: int,
    shape: tuple,
    dtype: torch.dtype,
    num_chunks: int = 4,
    seed: int = 42,
    distributed_init_method: Optional[str] = None,
):
    """
    Worker function to test two-stage allreduce
    
    Phase 1: SDMA Copy - copy data to buffer via SDMA
    Phase 2: Part Reduce - reduce and all_gather
    """
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)
    
    # Initialize
    logger.info(f"RANK {rank_id}: Initializing with tp_size={tp_size}...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    
    # Create test data - random integers for thorough testing
    # Use rank-specific seed to ensure different data per rank
    torch.manual_seed(seed + rank_id)
    x = torch.randint(1, 16, shape, dtype=torch.int32, device=device).to(dtype)
    x_original = x.clone()  # Keep original for verification
    
    logger.info(f"RANK {rank_id}: Created tensor with shape {shape}")
    logger.info(f"RANK {rank_id}: Input stats - min={x.min().item():.2f}, "
                f"max={x.max().item():.2f}, mean={x.mean().item():.2f}")
    
    # Warmup and align all GPUs
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()
    
    logger.info(f"RANK {rank_id}: Starting two-stage allreduce with {num_chunks} chunks...")
    
    # Phase 1: SDMA Copy
    logger.info(f"RANK {rank_id}: Phase 1 - SDMA Copy")
    for chunk_id in range(num_chunks):
        logger.info(f"RANK {rank_id}: SDMA copy chunk {chunk_id}/{num_chunks}")
        tensor_model_parallel_sdma_copy(x, chunk_num=num_chunks, chunk_id=chunk_id)
    
    torch.cuda.synchronize()
    logger.info(f"RANK {rank_id}: Phase 1 completed")
    
    # Phase 2: Part Reduce - collect results from each chunk
    logger.info(f"RANK {rank_id}: Phase 2 - Part Reduce")
    chunk_results = []
    chunk_size = (shape[0] + num_chunks - 1) // num_chunks
    
    for chunk_id in range(num_chunks):
        logger.info(f"RANK {rank_id}: Part reduce chunk {chunk_id}/{num_chunks}")
        chunk_result = tensor_model_parallel_part_reduce(x, chunk_num=num_chunks, chunk_id=chunk_id)
        
        # Calculate this chunk's row range
        start_idx = chunk_id * chunk_size
        end_idx = min(start_idx + chunk_size, shape[0])
        actual_chunk_size = end_idx - start_idx
        
        # part_reduce returns the reduced chunk
        logger.info(f"RANK {rank_id}: Chunk {chunk_id} result shape: {chunk_result.shape}, "
                    f"expected rows: {actual_chunk_size}")
        
        chunk_results.append(chunk_result)
    
    # Concatenate all chunks to get final result
    logger.info(f"RANK {rank_id}: Concatenating {len(chunk_results)} chunks...")
    result = torch.cat(chunk_results, dim=0)
    
    torch.cuda.synchronize()
    logger.info(f"RANK {rank_id}: Phase 2 completed, final shape: {result.shape}")
    
    # Log result statistics
    logger.info(f"RANK {rank_id}: Result tensor - min={result.min().item():.2f}, "
                f"max={result.max().item():.2f}, mean={result.mean().item():.2f}")
    
    # Cleanup
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    
    # Return both original input and result for verification
    return x_original.cpu(), result.cpu()


def test_twostage_allreduce(
    tp_size: int = 8,
    pp_size: int = 1,
    shape: tuple = (4, 7168),
    dtype: torch.dtype = torch.bfloat16,
    num_chunks: int = 4,
    seed: int = 42,
):
    """Main test function"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49374"
    
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    
    print("=" * 80)
    print("Two-Stage AllReduce Test (sdma_copy + part_reduce)")
    print("=" * 80)
    print(f"TP Size: {tp_size}")
    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Num Chunks: {num_chunks}")
    print(f"Input: Random integers [1, 16)")
    print(f"Expected output: Sum of all ranks' inputs")
    print("=" * 80)
    
    # Launch workers
    pool = Pool(processes=tp_size)
    rets = []
    
    for i in range(tp_size):
        ret = pool.apply_async(
            twostage_allreduce_worker,
            args=(tp_size, pp_size, i, shape, dtype, num_chunks, seed, distributed_init_method)
        )
        rets.append(ret)
    
    pool.close()
    pool.join()
    
    # Collect results (now returns tuple: (input, output))
    results = [ret.get() for ret in rets]
    inputs = [inp for inp, _ in results]
    outputs = [out for _, out in results]
    
    # Calculate expected result: sum of all ranks' inputs
    print("\nCalculating expected result (sum of all inputs)...")
    expected = torch.zeros(shape, dtype=dtype)
    for inp in inputs:
        expected += inp
    
    print(f"Expected result stats - min={expected.min().item():.2f}, "
          f"max={expected.max().item():.2f}, mean={expected.mean().item():.2f}")
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("Verification Results:")
    print("=" * 80)
    
    all_correct = True
    for rank_id, result in enumerate(outputs):
        print(f"\n{'='*80}")
        print(f"RANK {rank_id}:")
        print(f"{'='*80}")
        
        # Get some statistics
        min_val = result.min().item()
        max_val = result.max().item()
        mean_val = result.mean().item()
        
        print(f"Statistics:")
        print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}, Mean: {mean_val:.6f}")
        print(f"  Expected min: {expected.min().item():.6f}, "
              f"max: {expected.max().item():.6f}, "
              f"mean: {expected.mean().item():.6f}")
        result.resize_(expected.shape)
        
        # Check if result is close to expected
        is_correct = torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
        
        if is_correct:
            print(f"Status: ✓ PASS - All values correct!")
        else:
            all_correct = False
            print(f"Status: ✗ FAIL - Found mismatches!")
            
            # Find mismatched elements
            diff = (result - expected).abs()
            mismatch_mask = diff > (1e-2 * expected.abs() + 1e-2)  # rtol=1e-2, atol=1e-2
            mismatch_indices = torch.nonzero(mismatch_mask, as_tuple=False)
            
            num_mismatches = mismatch_indices.shape[0]
            total_elements = result.numel()
            
            print(f"\nMismatch Summary:")
            print(f"  Total elements: {total_elements}")
            print(f"  Mismatched elements: {num_mismatches} ({100*num_mismatches/total_elements:.2f}%)")
            print(f"  Max difference: {diff.max().item():.6f}")
            
            # Show first 20 mismatched locations
            max_show = min(20, num_mismatches)
            print(f"\nFirst {max_show} mismatched locations:")
            print(f"{'Index':<20} {'Expected':<12} {'Actual':<12} {'Diff':<12}")
            print("-" * 60)
            
            for i in range(max_show):
                idx = tuple(mismatch_indices[i].tolist())
                expected_val = expected[idx].item()
                actual_val = result[idx].item()
                diff_val = abs(actual_val - expected_val)
                
                idx_str = str(idx)
                print(f"{idx_str:<20} {expected_val:<12.6f} {actual_val:<12.6f} {diff_val:<12.6f}")
            
            if num_mismatches > max_show:
                print(f"... and {num_mismatches - max_show} more mismatches")
            
            # Analyze error distribution by chunks (if 2D tensor)
            if len(shape) == 2:
                print(f"\nError distribution by chunks (assuming {num_chunks} chunks along dim 0):")
                chunk_size = (shape[0] + num_chunks - 1) // num_chunks
                
                for chunk_id in range(num_chunks):
                    start_idx = chunk_id * chunk_size
                    end_idx = min(start_idx + chunk_size, shape[0])
                    
                    chunk_result = result[start_idx:end_idx, :]
                    chunk_expected = expected[start_idx:end_idx, :]
                    chunk_diff = (chunk_result - chunk_expected).abs()
                    chunk_mismatch_mask = chunk_diff > (1e-2 * chunk_expected.abs() + 1e-2)
                    chunk_num_mismatches = chunk_mismatch_mask.sum().item()
                    chunk_total = chunk_result.numel()
                    
                    print(f"  Chunk {chunk_id} [rows {start_idx}:{end_idx}]: "
                          f"{chunk_num_mismatches}/{chunk_total} errors "
                          f"({100*chunk_num_mismatches/chunk_total:.2f}%), "
                          f"max_diff={chunk_diff.max().item():.6f}")
            
            # Show unique actual values (to identify patterns)
            unique_vals = torch.unique(result)
            if len(unique_vals) <= 20:
                print(f"\nUnique values in result: {unique_vals.tolist()}")
            else:
                print(f"\nNumber of unique values: {len(unique_vals)}")
                print(f"  Min unique: {unique_vals.min().item():.6f}")
                print(f"  Max unique: {unique_vals.max().item():.6f}")
    
    print("\n" + "=" * 80)
    if all_correct:
        print("✓ All ranks PASSED")
    else:
        print("✗ Some ranks FAILED - Check detailed error information above")
    print("=" * 80)


if __name__ == "__main__":
    freeze_support()
    
    parser = argparse.ArgumentParser(
        description="Test two-stage allreduce (sdma_copy + part_reduce)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default test
  python test_twostage_allreduce.py
  
  # Custom shape and chunks
  python test_twostage_allreduce.py --shape 128,8192 --num-chunks 8
  
  # Different TP size
  python test_twostage_allreduce.py --tp-size 4 --num-chunks 2
        """
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8)"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="4,7168",
        help="Tensor shape, e.g. '4,7168' (default: 4,7168)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type (default: bf16)"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=4,
        help="Number of chunks (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Parse shape
    shape = tuple(map(int, args.shape.split(",")))
    
    # Parse dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    test_twostage_allreduce(
        tp_size=args.tp_size,
        pp_size=1,
        shape=shape,
        dtype=dtype,
        num_chunks=args.num_chunks,
        seed=args.seed,
    )


