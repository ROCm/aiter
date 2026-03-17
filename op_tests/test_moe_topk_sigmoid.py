# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test topk_sigmoid operation with various configurations.

This test can be run in two ways:

1. Using pytest (for automated testing):
   pytest test_moe_topk_sigmoid.py -v

2. Using command line arguments (for benchmarking with summary table):
   python test_moe_topk_sigmoid.py --num-experts 64,128 --topk 2,4,8 --dtype fp16
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    perftest,
)
from aiter.utility.dtypes import str2Dtype, str2tuple


@perftest(num_iters=10, num_warmup=1)
def run_torch(gating_output: torch.Tensor, topk: int):
    # llama4 maverick custom routing function
    router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores, router_indices.to(torch.int32)


@perftest(num_iters=10, num_warmup=1)
def run_fused(gating_output: torch.Tensor, topk: int):
    tokens, _ = gating_output.shape
    router_scores = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    router_indices = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    aiter.topk_sigmoid(router_scores, router_indices, gating_output)
    return router_scores, router_indices


def benchmark_topk_sigmoid(
    num_experts: int = 128,
    num_tokens: int = 1024,
    topk: int = 4,
    dtype: torch.dtype = torch.float16,
):
    # generate data - each row has only unique values
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    gating_output = torch.gather(gating_output, dim=-1, index=permutation)
    assert gating_output.is_contiguous()
    # run benchmarks
    (scores_torch, indices_torch), avg_torch = run_torch(gating_output.clone(), topk)
    (scores_fused, indices_fused), avg_fused = run_fused(gating_output.clone(), topk)
    # check correctness
    score_errors = checkAllclose(scores_torch, scores_fused, tol_err_ratio=0.01)
    index_errors = checkAllclose(indices_torch, indices_fused, tol_err_ratio=0.01)

    # Collect results for summary
    result = {
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
        "dtype": str(dtype).split(".")[-1],
        "torch_us": avg_torch,
        "fused_us": avg_fused,
        "uplift": avg_torch / avg_fused,
        "score_errors": score_errors,
        "index_errors": index_errors,
    }

    # print some failed rows if errors are significant
    if score_errors > 0.01 or index_errors > 0.01:
        failed_rows = (indices_torch != indices_fused).sum(dim=-1) > 0
        print(
            f"\n[ERROR] Configuration: num_experts={num_experts}, num_tokens={num_tokens}, topk={topk}, dtype={str(dtype).split('.')[-1]}"
        )
        print("Wrong scores:")
        print(scores_torch[failed_rows][:5])
        print(scores_fused[failed_rows][:5])
        print("Wrong indices:")
        print(indices_torch[failed_rows][:5])
        print(indices_fused[failed_rows][:5])
        print("Gating outputs:")
        failed_values = gating_output[failed_rows][:5]
        failed_values, _ = failed_values.sort(dim=-1, descending=True)
        print(failed_values[:, :10])
        print(
            f"Number of wrong tokens: {sum(failed_rows)} / {len(failed_rows)}, {100 * sum(failed_rows) / len(failed_rows):.2f} %"
        )

    return result


# Pytest-parametrized test functions
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
def test_topk_sigmoid_correctness(num_experts, num_tokens, topk, dtype):
    _run_correctness_test(num_experts, num_tokens, topk, dtype)

def _run_correctness_test(num_experts, num_tokens, topk, dtype):
    """Pytest test for correctness of topk_sigmoid operation."""
    torch.random.manual_seed(0)

    # generate data - each row has only unique values
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    gating_output = torch.gather(gating_output, dim=-1, index=permutation)
    assert gating_output.is_contiguous()

    # run both implementations
    (scores_torch, indices_torch), _ = run_torch(gating_output.clone(), topk)
    (scores_fused, indices_fused), _ = run_fused(gating_output.clone(), topk)

    # check correctness
    score_errors = checkAllclose(scores_torch, scores_fused, tol_err_ratio=0.01)
    index_errors = checkAllclose(indices_torch, indices_fused, tol_err_ratio=0.01)

    # Assert correctness
    assert score_errors <= 0.01, f"Score errors {score_errors} exceed tolerance"
    assert index_errors <= 0.01, f"Index errors {index_errors} exceed tolerance"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test topk_sigmoid operation with various configurations"
    )
    parser.add_argument(
        "--num-experts",
        type=str2tuple,
        default=[128],
        help="Comma-separated list of number of experts (default: 16,128)",
    )
    parser.add_argument(
        "--num-tokens",
        type=str2tuple,
        default=[1024],
        help="Comma-separated list of number of tokens (default: 1024)",
    )
    parser.add_argument(
        "--topk",
        type=str2tuple,
        default=[8],
        help="Comma-separated list of topk values (default: 1,2,8)",
    )
    parser.add_argument(
        "--dtype",
        type=str2Dtype,
        default=[torch.float16, torch.bfloat16],
        help="Comma-separated list of dtypes: fp16, bf16 (default: fp16,bf16)",
    )

    args = parser.parse_args()

    # Get parsed parameter lists
    num_experts_list = args.num_experts
    num_tokens_list = args.num_tokens
    topk_list = args.topk
    dtype_list = args.dtype

    # Run all combinations (cartesian product)
    configs = list(
        itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
    )

    print(f"Running {len(configs)} configuration(s):")
    print(f"  num_experts: {num_experts_list}")
    print(f"  num_tokens:  {num_tokens_list}")
    print(f"  topk:        {topk_list}")
    print(f"  dtype:       {[str(dt).split('.')[-1] for dt in dtype_list]}")
    print("=" * 80)

    # Collect results from all configurations
    collected = []
    for i, (num_experts, num_tokens, topk, dtype) in enumerate(configs, 1):
        result = benchmark_topk_sigmoid(
            num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
        )
        collected.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Create and print DataFrame
    df = pd.DataFrame(collected)
    print(df.to_string(index=False))

    # Print additional statistics
    print("\n" + "=" * 80)
    print(f"Average uplift: {df['uplift'].mean():.2f}x")
    print(f"Max uplift:     {df['uplift'].max():.2f}x")
    print(f"Min uplift:     {df['uplift'].min():.2f}x")

    # Check for any errors
    errors = df[(df["score_errors"] > 0.01) | (df["index_errors"] > 0.01)]
    if len(errors) > 0:
        print(
            f"\nWARNING: {len(errors)} configuration(s) had errors exceeding tolerance!"
        )
        print(errors.to_string(index=False))
    else:
        print("\nAll tests passed with errors within tolerance!")
    print("=" * 80)
