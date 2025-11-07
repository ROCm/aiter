# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test CK MOE 2-stage functions (ck_moe_stage1_fwd and ck_moe_stage2_fwd).

This test can be run in two ways:

1. Using pytest (for automated testing):
   pytest test_moe_ck_2stage.py -v

2. Using command line arguments (for benchmarking with summary table):
   python test_moe_ck_2stage.py --tokens 1024,2048 --topk 2,4 --dtype bf16,fp8_blockscale
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch
import aiter
from aiter import dtypes, ActivationType, QuantType
from aiter.test_common import checkAllclose, run_perftest
from aiter.utility.dtypes import str2tuple
from aiter.fused_moe import moe_sorting, torch_moe_stage1, torch_moe_stage2, get_2stage_cfgs
from aiter.ops.shuffle import shuffle_weight


def run_2stage_moe(
    num_tokens: int,
    hidden_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_blockscale: bool = False,
    activation: ActivationType = ActivationType.Silu,
    exclude_activations_from_bw: bool = True,
):
    """Run both stages of MOE and return performance metrics.
    
    Args:
        exclude_activations_from_bw: If True (default), excludes input/intermediate/output 
            tensors from bandwidth calculations, assuming they stay in cache (weights only).
    """
    torch.manual_seed(42)
    
    # Generate input data
    input = 0.1 * torch.randn((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    
    # Generate weights
    if use_fp8_blockscale:
        # For FP8 blockscale, we need quantized weights and scales
        w1_fp32 = torch.randn((num_experts, inter_dim * 2, hidden_dim), dtype=torch.float32, device="cuda")
        w2_fp32 = torch.randn((num_experts, hidden_dim, inter_dim), dtype=torch.float32, device="cuda")
        
        # Quantize to FP8
        input = input.to(dtypes.fp8)
        w1 = w1_fp32.to(dtypes.fp8)
        w2 = w2_fp32.to(dtypes.fp8)
        
        # Create block scales (128x128 blocks)
        # block_size = 128
        # num_blocks_w1_n = (inter_dim * 2 + block_size - 1) // block_size
        # num_blocks_w1_k = (hidden_dim + block_size - 1) // block_size
        # num_blocks_w2_n = (hidden_dim + block_size - 1) // block_size
        # num_blocks_w2_k = (inter_dim + block_size - 1) // block_size
        
        # w1_scale = torch.ones((num_experts, num_blocks_w1_n, num_blocks_w1_k), dtype=torch.float32, device="cuda") * 0.01
        # w2_scale = torch.ones((num_experts, num_blocks_w2_n, num_blocks_w2_k), dtype=torch.float32, device="cuda") * 0.01
        # a1_scale = torch.ones((num_tokens, (hidden_dim + block_size - 1) // block_size), dtype=torch.float32, device="cuda")
        # a2_scale = torch.ones((num_tokens, topk, (inter_dim + block_size - 1) // block_size), dtype=torch.float32, device="cuda")
        
        # quant_type = QuantType.per_128x128
        quant_type = QuantType.per_Tensor
        w1_scale = torch.ones((num_experts,), dtype=torch.float32, device="cuda")
        w2_scale = torch.ones((num_experts,), dtype=torch.float32, device="cuda")
        a1_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
        a2_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    else:
        # BF16 case
        w1 = 0.1 * torch.randn((num_experts, inter_dim * 2, hidden_dim), dtype=dtype, device="cuda")
        w2 = 0.1 * torch.randn((num_experts, hidden_dim, inter_dim), dtype=dtype, device="cuda")
        w1_scale = None
        w2_scale = None
        a1_scale = None
        a2_scale = None
        quant_type = QuantType.No
    w1_shuffle = shuffle_weight(w1.clone())
    w2_shuffle = shuffle_weight(w2.clone())
    
    # Generate routing scores and compute topk using torch (to avoid module loading issues)
    score = torch.randn((num_tokens, num_experts), dtype=dtype, device="cuda")
    topk_weights, topk_ids = torch.topk(score, topk, dim=-1)
    # Normalize weights
    topk_weights = torch.softmax(topk_weights.float(), dim=-1)
    topk_ids = topk_ids.to(torch.int32)
    
    # Get metadata for kernel selection and configuration
    q_dtype_w = w1.dtype
    # For FP8 quantized weights, use torch.float8_e4m3fnuz as the dtype
    q_dtype_a = w1.dtype if w1.dtype != torch.uint32 else torch.float8_e4m3fnuz
    isG1U1 = True  # Always using gate+up configuration
    
    metadata = get_2stage_cfgs(
        min(1024, num_tokens),  # consider token_num > 1024 as prefill
        hidden_dim,
        inter_dim,
        num_experts,
        topk,
        dtype,
        q_dtype_a,
        q_dtype_w,
        quant_type,
        isG1U1,
        activation,
        False,  # doweight_stage1
        0,  # hidden_pad
        0,  # intermediate_pad
        None,  # bias1
        None,  # bias2
    )
    
    block_size = metadata.block_m
    
    # Sort for MOE execution
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, num_experts, hidden_dim, dtype, block_size
    )
    
    # ========== Stage 1 ==========
    # Reference implementation using the function from fused_moe.py
    inter_ref = torch_moe_stage1(
        input,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtype,
        activation=activation,
        quant_type=quant_type,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        # w1_bias=None,
        doweight=False,
    )
    
    # CK implementation using metadata.stage1
    inter_out = torch.empty((num_tokens, topk, inter_dim), dtype=dtype, device="cuda")
    
    inter_ck, us_stage1 = run_perftest(
        metadata.stage1,
        input,
        w1_shuffle,
        w2_shuffle,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        inter_out,
        topk,
        block_m=block_size,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        sorted_weights=None,
        num_iters=10,
        num_warmup=2,
    )
    
    # Check correctness for stage 1
    stage1_error = checkAllclose(inter_ref.to(dtype), inter_ck, tol_err_ratio=0.1 if use_fp8_blockscale else 0.01)
    
    # ========== Stage 2 ==========
    # Reference implementation using the function from fused_moe.py
    # NOTE: Both stage2 implementations should use the same intermediate tensor
    # to isolate stage 2 correctness from any stage 1 numerical differences
    output_ref = torch_moe_stage2(
        inter_ref,
        w1,
        w2,
        topk_weights,
        topk_ids,
        dtype=dtype,
        quant_type=quant_type,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        # w2_bias=None,
        doweight=True,
    )
    
    # CK implementation using metadata.stage2
    # Use inter_ref (not inter_ck) to test stage 2 in isolation
    # NOTE: Stage 2 kernel uses AtomicAdd, so we need fresh output buffers for run_perftest
    # to avoid accumulation across iterations. We'll call it once for validation.
    output_out_validation = torch.zeros((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    output_ck = metadata.stage2(
        inter_ref,
        w1_shuffle,
        w2_shuffle,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        output_out_validation,
        topk,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_size,
        sorted_weights=sorted_weights,
    )
    
    # For performance measurement, create fresh buffers for each iteration
    output_out_perf = torch.zeros((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    _, us_stage2 = run_perftest(
        metadata.stage2,
        inter_ref,
        w1_shuffle,
        w2_shuffle,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        output_out_perf,
        topk,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_size,
        sorted_weights=sorted_weights,
        num_iters=10,
        num_warmup=2,
        num_rotate_args=12,  # Force creating 12 separate output buffers
        needTrace=False,
    )
    
    # Check correctness for stage 2
    stage2_error = checkAllclose(output_ref.to(dtype), output_ck, tol_err_ratio=0.1 if use_fp8_blockscale else 0.01)

    # Get flops and BW estimates
    effective_experts = min(num_experts, topk * num_tokens)

    # Calculate FLOPS for each stage
    # Stage 1: input @ w1.T -> (num_tokens, hidden_dim) @ (num_experts, inter_dim*2, hidden_dim).T
    # Effective computation: (num_tokens * topk) * (inter_dim * 2) * hidden_dim * 2 (for matmul)
    # Plus activation (not counted in FLOPS here)
    stage1_flops = num_tokens * topk * (inter_dim * 2) * hidden_dim * 2
    
    # Stage 2: inter @ w2.T -> (num_tokens, topk, inter_dim) @ (num_experts, hidden_dim, inter_dim).T
    # Effective computation: (num_tokens * topk) * hidden_dim * inter_dim * 2 (for matmul)
    stage2_flops = num_tokens * topk * hidden_dim * inter_dim * 2
    
    total_flops = stage1_flops + stage2_flops
    
    # Calculate TFLOPS (time in microseconds, so divide by 1e6 to get seconds)
    stage1_tflops = (stage1_flops / (us_stage1 * 1e-6)) / 1e12
    stage2_tflops = (stage2_flops / (us_stage2 * 1e-6)) / 1e12
    total_tflops = (total_flops / ((us_stage1 + us_stage2) * 1e-6)) / 1e12
    
    # Calculate Bandwidth (GB/s)
    # Bytes per element based on dtype
    if use_fp8_blockscale:
        bytes_per_elem = 1  # FP8
    else:
        bytes_per_elem = 2  # BF16/FP16
    
    # Stage 1: Read input (num_tokens, hidden_dim) + Read w1 weights for effective_experts
    #          + Write output (num_tokens, topk, inter_dim)
    # We can't read more expert weights than exist, so use effective_experts
    if exclude_activations_from_bw:
        # Only count weight reads (assuming activations stay in cache)
        stage1_read_bytes = effective_experts * inter_dim * 2 * hidden_dim * bytes_per_elem  # w1 only
        stage1_write_bytes = 0
    else:
        stage1_read_bytes = (num_tokens * hidden_dim * bytes_per_elem +  # input
                            effective_experts * inter_dim * 2 * hidden_dim * bytes_per_elem)  # w1 (effective experts)
        stage1_write_bytes = num_tokens * topk * inter_dim * bytes_per_elem  # output
    stage1_total_bytes = stage1_read_bytes + stage1_write_bytes
    stage1_bw = (stage1_total_bytes / (us_stage1 * 1e-6)) / 1e9  # GB/s
    
    # Stage 2: Read inter (num_tokens, topk, inter_dim) + Read w2 weights for effective_experts
    #          + Write output (num_tokens, hidden_dim)
    # We can't read more expert weights than exist, so use effective_experts
    if exclude_activations_from_bw:
        # Only count weight reads (assuming activations stay in cache)
        stage2_read_bytes = effective_experts * hidden_dim * inter_dim * bytes_per_elem  # w2 only
        stage2_write_bytes = 0
    else:
        stage2_read_bytes = (num_tokens * topk * inter_dim * bytes_per_elem +  # inter
                            effective_experts * hidden_dim * inter_dim * bytes_per_elem)  # w2 (effective experts)
        stage2_write_bytes = num_tokens * hidden_dim * bytes_per_elem  # output
    stage2_total_bytes = stage2_read_bytes + stage2_write_bytes
    stage2_bw = (stage2_total_bytes / (us_stage2 * 1e-6)) / 1e9  # GB/s
    
    # Combined bandwidth
    total_bytes = stage1_total_bytes + stage2_total_bytes
    total_bw = (total_bytes / ((us_stage1 + us_stage2) * 1e-6)) / 1e9  # GB/s
    
    # Collect metrics
    result = {
        'num_tokens': num_tokens,
        'hidden_dim': hidden_dim,
        'inter_dim': inter_dim,
        'num_experts': num_experts,
        'topk': topk,
        'dtype': 'fp8_blockscale' if use_fp8_blockscale else str(dtype).split('.')[-1],
        'activation': str(activation).split('.')[-1],
        'stage1_us': us_stage1,
        'stage2_us': us_stage2,
        'total_us': us_stage1 + us_stage2,
        'stage1_tflops': stage1_tflops,
        'stage2_tflops': stage2_tflops,
        'total_tflops': total_tflops,
        'stage1_bw_gb_s': stage1_bw,
        'stage2_bw_gb_s': stage2_bw,
        'total_bw_gb_s': total_bw,
        'stage1_error': stage1_error,
        'stage2_error': stage2_error,
    }
    
    return result


# Pytest-parametrized test functions
@pytest.mark.parametrize("dtype_str", ["bf16"])
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("num_tokens", [512, 1024])
@pytest.mark.parametrize("num_experts", [64, 128])
@pytest.mark.parametrize("hidden_dim", [4096])
@pytest.mark.parametrize("inter_dim", [14336])
def test_moe_ck_2stage_correctness(num_experts, num_tokens, topk, hidden_dim, inter_dim, dtype_str):
    """Pytest test for correctness of CK MOE 2-stage operations."""
    torch.random.manual_seed(42)
    
    # Map dtype string
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
    dtype = dtype_map[dtype_str]
    use_fp8 = dtype_str == "fp8_blockscale"
    
    result = run_2stage_moe(
        num_tokens=num_tokens,
        hidden_dim=hidden_dim,
        inter_dim=inter_dim,
        num_experts=num_experts,
        topk=topk,
        dtype=dtype,
        use_fp8_blockscale=use_fp8,
    )
    
    # Assert correctness
    tol = 0.1 if use_fp8 else 0.01
    assert result['stage1_error'] <= tol, f"Stage1 error {result['stage1_error']} exceeds tolerance {tol}"
    assert result['stage2_error'] <= tol, f"Stage2 error {result['stage2_error']} exceeds tolerance {tol}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test CK MOE 2-stage operations with various configurations'
    )
    parser.add_argument(
        '--tokens',
        type=str2tuple,
        default=[8192],
        help='Comma-separated list of number of tokens (default: 1024)'
    )
    parser.add_argument(
        '--topk',
        type=str2tuple,
        default=[8],
        help='Comma-separated list of topk values (default: 8)'
    )
    parser.add_argument(
        '--num-experts',
        type=str2tuple,
        default=[128],
        help='Comma-separated list of number of experts (default: 6128)'
    )
    parser.add_argument(
        '--hidden-dim',
        type=str2tuple,
        default=[4096],
        help='Comma-separated list of hidden dimensions (default: 4096)'
    )
    parser.add_argument(
        '--inter-dim',
        type=str2tuple,
        default=[192],
        help='Comma-separated list of intermediate dimensions (default: 192)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='bf16',
        choices=['bf16', 'fp16', 'fp8_blockscale'],
        help='Data type: bf16, fp16, or fp8_blockscale (default: bf16)'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='silu',
        choices=['silu', 'gelu'],
        help='Activation function: silu or gelu (default: silu)'
    )
    parser.add_argument(
        '--include-activations-in-bw',
        action='store_true',
        help='Include input/intermediate/output tensors in bandwidth calculations (default: weights only)'
    )
    
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp8_blockscale": torch.bfloat16,  # Use bf16 as base, but with fp8 flag
    }
    dtype = dtype_map[args.dtype]
    use_fp8 = args.dtype == "fp8_blockscale"
    
    # Parse activation
    activation_map = {
        "silu": ActivationType.Silu,
        "gelu": ActivationType.Gelu,
    }
    activation = activation_map[args.activation]
    
    # Get parameter lists
    tokens_list = args.tokens
    topk_list = args.topk
    num_experts_list = args.num_experts
    hidden_dim_list = args.hidden_dim
    inter_dim_list = args.inter_dim
    
    # Run all combinations (cartesian product)
    configs = list(itertools.product(
        tokens_list,
        hidden_dim_list,
        inter_dim_list,
        num_experts_list,
        topk_list,
    ))
    
    print(f"Running {len(configs)} configuration(s):")
    print(f"  tokens:                      {tokens_list}")
    print(f"  hidden_dim:                  {hidden_dim_list}")
    print(f"  inter_dim:                   {inter_dim_list}")
    print(f"  num_experts:                 {num_experts_list}")
    print(f"  topk:                        {topk_list}")
    print(f"  dtype:                       {args.dtype}")
    print(f"  activation:                  {args.activation}")
    print(f"  include_activations_in_bw:   {args.include_activations_in_bw}")
    print("=" * 100)
    
    # Collect results from all configurations
    collected = []
    for i, (num_tokens, hidden_dim, inter_dim, num_experts, topk) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: "
              f"tokens={num_tokens}, hidden={hidden_dim}, inter={inter_dim}, "
              f"experts={num_experts}, topk={topk}", end=" ... ")
        
        result = run_2stage_moe(
            num_tokens=num_tokens,
            hidden_dim=hidden_dim,
            inter_dim=inter_dim,
            num_experts=num_experts,
            topk=topk,
            dtype=dtype,
            use_fp8_blockscale=use_fp8,
            activation=activation,
            exclude_activations_from_bw=not args.include_activations_in_bw,
        )
        collected.append(result)
        print("Done")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Create and print DataFrame
    df = pd.DataFrame(collected)
    print(df.to_string(index=False))
    
    # Print additional statistics
    print("\n" + "=" * 100)
    print("PERFORMANCE STATISTICS")
    print("=" * 100)
    print(f"Average stage1 time:      {df['stage1_us'].mean():.2f} us")
    print(f"Average stage2 time:      {df['stage2_us'].mean():.2f} us")
    print(f"Average total time:       {df['total_us'].mean():.2f} us")
    print(f"\nAverage stage1 TFLOPS:    {df['stage1_tflops'].mean():.2f}")
    print(f"Average stage2 TFLOPS:    {df['stage2_tflops'].mean():.2f}")
    print(f"Average total TFLOPS:     {df['total_tflops'].mean():.2f}")
    bw_note = "" if args.include_activations_in_bw else " (weights only)"
    print(f"\nAverage stage1 BW:        {df['stage1_bw_gb_s'].mean():.2f} GB/s{bw_note}")
    print(f"Average stage2 BW:        {df['stage2_bw_gb_s'].mean():.2f} GB/s{bw_note}")
    print(f"Average total BW:         {df['total_bw_gb_s'].mean():.2f} GB/s{bw_note}")
    
    # Check for any errors
    tol = 0.1 if use_fp8 else 0.01
    errors = df[(df['stage1_error'] > tol) | (df['stage2_error'] > tol)]
    print("\n" + "=" * 100)
    if len(errors) > 0:
        print(f"WARNING: {len(errors)} configuration(s) had errors exceeding tolerance {tol}!")
        print(errors.to_string(index=False))
    else:
        print(f"All tests passed with errors within tolerance ({tol})!")
    print("=" * 100)
