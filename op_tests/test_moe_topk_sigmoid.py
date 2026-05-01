# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

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


# -- topk_softplus (DeepSeek V4-Pro sqrtsoftplus routing) --------------


@perftest(num_iters=10, num_warmup=1)
def run_torch_softplus(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    route_scale: float,
):
    scores = torch.nn.functional.softplus(gating_output.float()).sqrt()
    scores_biased = scores + bias.float()
    topk_ids = scores_biased.topk(topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * route_scale
    return topk_weights, topk_ids.to(torch.int32)


@perftest(num_iters=10, num_warmup=1)
def run_fused_softplus(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    route_scale: float,
):
    tokens, _ = gating_output.shape
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    aiter.topk_softplus(
        topk_weights, topk_ids, gating_output, bias, renormalize, route_scale
    )
    return topk_weights, topk_ids


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


def benchmark_topk_softplus(
    num_experts: int = 256,
    num_tokens: int = 1024,
    topk: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    renormalize: bool = True,
    route_scale: float = 2.5,
):
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    gating_output = torch.gather(gating_output, dim=-1, index=permutation)
    bias = torch.randn(num_experts, dtype=dtype, device="cuda") * 0.1

    (w_torch, i_torch), avg_torch = run_torch_softplus(
        gating_output.clone(), bias, topk, renormalize, route_scale
    )
    (w_fused, i_fused), avg_fused = run_fused_softplus(
        gating_output.clone(), bias, topk, renormalize, route_scale
    )

    # compare by matching expert ids per token
    id_match = 0
    max_w_err = 0.0
    for t in range(num_tokens):
        kern_set = set(i_fused[t].tolist())
        ref_set = set(i_torch[t].tolist())
        if kern_set == ref_set:
            id_match += 1
            for k in range(topk):
                kid = i_fused[t, k].item()
                ref_k = (i_torch[t] == kid).nonzero(as_tuple=True)[0]
                if len(ref_k) > 0:
                    err = abs(w_fused[t, k].item() - w_torch[t, ref_k[0]].item())
                    max_w_err = max(max_w_err, err)

    id_err = 1.0 - id_match / num_tokens

    result = {
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
        "dtype": str(dtype).split(".")[-1],
        "torch_us": avg_torch,
        "fused_us": avg_fused,
        "uplift": avg_torch / avg_fused,
        "id_errors": id_err,
        "max_weight_err": max_w_err,
    }

    if id_err > 0.01:
        print(
            f"\n[ERROR] softplus: num_experts={num_experts}, num_tokens={num_tokens}, "
            f"topk={topk}, dtype={str(dtype).split('.')[-1]}, id_err={id_err:.4f}"
        )

    return result


# Pytest-parametrized test functions -- topk_softplus
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
def test_topk_softplus_correctness(num_experts, num_tokens, topk, dtype):
    """Pytest test for correctness of topk_softplus (sqrtsoftplus) operation."""
    torch.random.manual_seed(0)
    route_scale = 2.5

    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    gating_output = torch.gather(gating_output, dim=-1, index=permutation)
    bias = torch.randn(num_experts, dtype=dtype, device="cuda") * 0.1

    (w_torch, i_torch), _ = run_torch_softplus(
        gating_output.clone(), bias, topk, True, route_scale
    )
    (w_fused, i_fused), _ = run_fused_softplus(
        gating_output.clone(), bias, topk, True, route_scale
    )

    # compare ids per token (order may differ)
    for t in range(num_tokens):
        kern_set = set(i_fused[t].tolist())
        ref_set = set(i_torch[t].tolist())
        assert (
            kern_set == ref_set
        ), f"Token {t}: ID mismatch kernel={sorted(kern_set)} ref={sorted(ref_set)}"

    # compare weights (match by expert id)
    for t in range(num_tokens):
        for k in range(topk):
            kid = i_fused[t, k].item()
            ref_k = (i_torch[t] == kid).nonzero(as_tuple=True)[0]
            assert len(ref_k) > 0
            torch.testing.assert_close(
                w_fused[t, k], w_torch[t, ref_k[0]], atol=1e-5, rtol=1e-4
            )


# Pytest-parametrized test functions -- topk_sigmoid
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128])
def test_topk_sigmoid_correctness(num_experts, num_tokens, topk, dtype):
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
        description="Test topk_sigmoid and topk_softplus operations"
    )
    parser.add_argument(
        "--num-experts",
        type=str2tuple,
        default=[128],
        help="Comma-separated list of number of experts (default: 128)",
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
        help="Comma-separated list of topk values (default: 8)",
    )
    parser.add_argument(
        "--dtype",
        type=str2Dtype,
        default=[torch.float16, torch.bfloat16],
        help="Comma-separated list of dtypes: fp16, bf16 (default: fp16,bf16)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["sigmoid", "softplus", "all"],
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    num_experts_list = to_list(args.num_experts)
    num_tokens_list = to_list(args.num_tokens)
    topk_list = to_list(args.topk)
    dtype_list = to_list(args.dtype)

    configs = list(
        itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
    )

    if args.test in ("sigmoid", "all"):
        print("=" * 80)
        print("topk_sigmoid benchmark")
        print("=" * 80)
        collected = []
        for num_experts, num_tokens, topk, dtype in configs:
            result = benchmark_topk_sigmoid(
                num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
            )
            collected.append(result)
        df = pd.DataFrame(collected)
        print(df.to_string(index=False))
        print(f"\nAverage uplift: {df['uplift'].mean():.2f}x")

    if args.test in ("softplus", "all"):
        print("\n" + "=" * 80)
        print("topk_softplus benchmark")
        print("=" * 80)
        collected = []
        for num_experts, num_tokens, topk, dtype in configs:
            result = benchmark_topk_softplus(
                num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
            )
            collected.append(result)
        df = pd.DataFrame(collected)
        print(df.to_string(index=False))
        print(f"\nAverage uplift: {df['uplift'].mean():.2f}x")
        errors = df[df["id_errors"] > 0.01]
        if len(errors) > 0:
            print(f"\nWARNING: {len(errors)} config(s) had id errors > 1%!")
            print(errors.to_string(index=False))
        else:
            print("All tests passed!")
    print("=" * 80)
