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


def run_2stage_moe(
    num_tokens: int,
    hidden_dim: int,
    inter_dim: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_blockscale: bool = False,
    activation: ActivationType = ActivationType.Silu,
):
    """Run both stages of MOE and return performance metrics."""
    torch.manual_seed(42)
    
    # Generate input data
    input = torch.randn((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    
    # Generate weights
    if use_fp8_blockscale:
        # For FP8 blockscale, we need quantized weights and scales
        w1_fp32 = torch.randn((num_experts, inter_dim * 2, hidden_dim), dtype=torch.float32, device="cuda")
        w2_fp32 = torch.randn((num_experts, hidden_dim, inter_dim), dtype=torch.float32, device="cuda")
        
        # Quantize to FP8
        w1 = w1_fp32.to(dtypes.fp8)
        w2 = w2_fp32.to(dtypes.fp8)
        
        # Create block scales (128x128 blocks)
        block_size = 128
        num_blocks_w1_n = (inter_dim * 2 + block_size - 1) // block_size
        num_blocks_w1_k = (hidden_dim + block_size - 1) // block_size
        num_blocks_w2_n = (hidden_dim + block_size - 1) // block_size
        num_blocks_w2_k = (inter_dim + block_size - 1) // block_size
        
        w1_scale = torch.ones((num_experts, num_blocks_w1_n, num_blocks_w1_k), dtype=torch.float32, device="cuda") * 0.01
        w2_scale = torch.ones((num_experts, num_blocks_w2_n, num_blocks_w2_k), dtype=torch.float32, device="cuda") * 0.01
        a1_scale = torch.ones((num_tokens, (hidden_dim + block_size - 1) // block_size), dtype=torch.float32, device="cuda")
        a2_scale = torch.ones((num_tokens, topk, (inter_dim + block_size - 1) // block_size), dtype=torch.float32, device="cuda")
        
        quant_type = QuantType.per_128x128
    else:
        # BF16 case
        w1 = torch.randn((num_experts, inter_dim * 2, hidden_dim), dtype=dtype, device="cuda")
        w2 = torch.randn((num_experts, hidden_dim, inter_dim), dtype=dtype, device="cuda")
        w1_scale = None
        w2_scale = None
        a1_scale = None
        a2_scale = None
        quant_type = QuantType.No
    
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
        w1,
        w2,
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
    output_out = torch.empty((num_tokens, hidden_dim), dtype=dtype, device="cuda")
    output_out.fill_(0)
    
    output_ck, us_stage2 = run_perftest(
        metadata.stage2,
        inter_ck,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        output_out,
        topk,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        block_m=block_size,
        sorted_weights=sorted_weights,
        num_iters=10,
        num_warmup=2,
        needTrace=True,
    )
    
    # Check correctness for stage 2
    stage2_error = checkAllclose(output_ref.to(dtype), output_ck, tol_err_ratio=0.1 if use_fp8_blockscale else 0.01)
    
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
    print(f"  tokens:      {tokens_list}")
    print(f"  hidden_dim:  {hidden_dim_list}")
    print(f"  inter_dim:   {inter_dim_list}")
    print(f"  num_experts: {num_experts_list}")
    print(f"  topk:        {topk_list}")
    print(f"  dtype:       {args.dtype}")
    print(f"  activation:  {args.activation}")
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
    print(f"Average stage1 time: {df['stage1_us'].mean():.2f} us")
    print(f"Average stage2 time: {df['stage2_us'].mean():.2f} us")
    print(f"Average total time:  {df['total_us'].mean():.2f} us")
    
    # Check for any errors
    tol = 0.1 if use_fp8 else 0.01
    errors = df[(df['stage1_error'] > tol) | (df['stage2_error'] > tol)]
    if len(errors) > 0:
        print(f"\nWARNING: {len(errors)} configuration(s) had errors exceeding tolerance {tol}!")
        print(errors.to_string(index=False))
    else:
        print(f"\nAll tests passed with errors within tolerance ({tol})!")
    print("=" * 100)
