# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for MHC (Multi-Head Channel) Layer operations.

Tests include:
1. Sinkhorn-Knopp normalization
2. Stream aggregate operation
3. Stream distribute mix add operation
4. Full MHC layer forward (static H)
5. Full MHC layer forward (dynamic H)
"""

import torch
import torch.nn.functional as F
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
import aiter
from aiter import dtypes
from aiter.ops.mhc import (
    mhc_layer_forward,
    mhc_layer_forward_dynamic,
    sinkhorn_knopp_forward,
    stream_aggregate_forward,
    stream_distribute_mix_add_forward,
)
import argparse
import pandas as pd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


# ============================================================================
# Reference implementations
# ============================================================================

def sinkhorn_knopp_ref(inp: torch.Tensor, num_iters: int = 20, eps: float = 1e-5) -> torch.Tensor:
    """Reference Sinkhorn-Knopp implementation in PyTorch."""
    M = inp.clone()
    for _ in range(num_iters):
        # Row normalization
        row_sums = M.sum(dim=-1, keepdim=True)
        M = M / (row_sums + eps)
        # Column normalization
        col_sums = M.sum(dim=-2, keepdim=True)
        M = M / (col_sums + eps)
    return M


def stream_aggregate_ref(x_expanded: torch.Tensor, H_pre: torch.Tensor) -> torch.Tensor:
    """Reference stream aggregate: out[b,c] = sum_i(H_pre[i] * x[b,i,c])"""
    # x_expanded: [B, n, C], H_pre: [n] or [B, n]
    if H_pre.dim() == 1:
        # [B, n, C] * [n, 1] -> [B, n, C] -> sum over n -> [B, C]
        return (x_expanded * H_pre.unsqueeze(-1)).sum(dim=1)
    else:
        # [B, n, C] * [B, n, 1] -> [B, n, C] -> sum over n -> [B, C]
        return (x_expanded * H_pre.unsqueeze(-1)).sum(dim=1)


def stream_distribute_mix_add_ref(
    x_expanded: torch.Tensor,
    y: torch.Tensor,
    H_post: torch.Tensor,
    M: torch.Tensor,
) -> torch.Tensor:
    """Reference: out[b,i,c] = H_post[i]*y[b,c] + sum_j(M[i,j]*x[b,j,c])"""
    B, n, C = x_expanded.shape
    
    if M.dim() == 2:
        # M: [n, n], x: [B, n, C] -> [B, n, C]
        # Einsum: M[i,j] * x[b,j,c] -> [b,i,c]
        mix = torch.einsum('ij,bjc->bic', M, x_expanded)
    else:
        # M: [B, n, n], x: [B, n, C] -> [B, n, C]
        mix = torch.einsum('bij,bjc->bic', M, x_expanded)
    
    if H_post.dim() == 1:
        # H_post: [n], y: [B, C] -> distribute to [B, n, C]
        dist = H_post.view(1, n, 1) * y.unsqueeze(1)
    else:
        # H_post: [B, n], y: [B, C] -> [B, n, C]
        dist = H_post.unsqueeze(-1) * y.unsqueeze(1)
    
    return mix + dist


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Reference RMSNorm implementation."""
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def mhc_layer_ref(
    x_expanded: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    H_pre: torch.Tensor,
    H_post: torch.Tensor,
    H_res: torch.Tensor,
    eps: float = 1e-5,
    sinkhorn_iters: int = 20,
) -> torch.Tensor:
    """Reference MHC layer forward pass."""
    B, n, C = x_expanded.shape
    
    # Apply sigmoid to H_pre
    H_pre_act = torch.sigmoid(H_pre)
    
    # Stream aggregate
    x_aggregated = stream_aggregate_ref(x_expanded, H_pre_act)
    
    # Sinkhorn-Knopp on exp(H_res)
    M = sinkhorn_knopp_ref(torch.exp(H_res), sinkhorn_iters, eps)
    
    # RMSNorm
    y_norm = rmsnorm_ref(x_aggregated, rmsnorm_weight, eps)
    
    # Apply 2*sigmoid to H_post
    H_post_act = 2.0 * torch.sigmoid(H_post)
    
    # Stream distribute mix add
    output = stream_distribute_mix_add_ref(x_expanded, y_norm, H_post_act, M)
    
    return output


# ============================================================================
# Test functions
# ============================================================================

@benchmark()
def test_sinkhorn_knopp(n: int, num_iters: int = 20, dtype=torch.float32):
    """Test Sinkhorn-Knopp normalization."""
    ret = {}
    
    # Random positive matrix
    inp = torch.rand(n, n, dtype=dtype, device="cuda") + 0.1
    
    # Reference
    ref, us_ref = run_perftest(sinkhorn_knopp_ref, inp, num_iters, num_iters=100, num_warmup=10)
    
    # Aiter
    out, us_aiter = run_perftest(sinkhorn_knopp_forward, inp, num_iters, num_iters=100, num_warmup=10)
    
    # Check doubly stochastic property
    row_sums = out.sum(dim=-1)
    col_sums = out.sum(dim=-2)
    row_err = (row_sums - 1.0).abs().max().item()
    col_err = (col_sums - 1.0).abs().max().item()
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-4, msg="sinkhorn_knopp [ref vs aiter]")
    
    ret["n"] = n
    ret["num_iters"] = num_iters
    ret["row_err"] = row_err
    ret["col_err"] = col_err
    ret["err"] = err
    ret["us_ref"] = us_ref
    ret["us_aiter"] = us_aiter
    ret["speedup"] = us_ref / us_aiter if us_aiter > 0 else 0
    
    return ret


@benchmark()
def test_sinkhorn_knopp_batched(B: int, n: int, num_iters: int = 20, dtype=torch.float32):
    """Test batched Sinkhorn-Knopp normalization."""
    ret = {}
    
    # Random positive matrices
    inp = torch.rand(B, n, n, dtype=dtype, device="cuda") + 0.1
    
    # Reference (loop over batch)
    ref_list = []
    for b in range(B):
        ref_list.append(sinkhorn_knopp_ref(inp[b], num_iters))
    ref = torch.stack(ref_list)
    
    # Aiter batched
    out, us_aiter = run_perftest(sinkhorn_knopp_forward, inp, num_iters, num_iters=100, num_warmup=10)
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-4, msg="sinkhorn_knopp_batched [ref vs aiter]")
    
    ret["B"] = B
    ret["n"] = n
    ret["num_iters"] = num_iters
    ret["err"] = err
    ret["us_aiter"] = us_aiter
    
    return ret


@benchmark()
def test_stream_aggregate(B: int, n: int, C: int, dtype=torch.float32):
    """Test stream aggregate operation."""
    ret = {}
    
    x_expanded = torch.randn(B, n, C, dtype=dtype, device="cuda")
    H_pre = torch.randn(n, dtype=dtype, device="cuda")
    
    # Reference
    ref, us_ref = run_perftest(stream_aggregate_ref, x_expanded, H_pre, num_iters=100, num_warmup=10)
    
    # Aiter
    out, us_aiter = run_perftest(stream_aggregate_forward, x_expanded, H_pre, num_iters=100, num_warmup=10)
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-4, msg="stream_aggregate [ref vs aiter]")
    
    ret["B"] = B
    ret["n"] = n
    ret["C"] = C
    ret["err"] = err
    ret["us_ref"] = us_ref
    ret["us_aiter"] = us_aiter
    ret["speedup"] = us_ref / us_aiter if us_aiter > 0 else 0
    
    return ret


@benchmark()
def test_stream_distribute_mix_add(B: int, n: int, C: int, dtype=torch.float32):
    """Test stream distribute mix add operation."""
    ret = {}
    
    x_expanded = torch.randn(B, n, C, dtype=dtype, device="cuda")
    y = torch.randn(B, C, dtype=dtype, device="cuda")
    H_post = torch.randn(n, dtype=dtype, device="cuda")
    M = torch.rand(n, n, dtype=dtype, device="cuda")
    M = M / M.sum(dim=-1, keepdim=True)  # Row normalize
    M = M / M.sum(dim=-2, keepdim=True)  # Col normalize (approx doubly stochastic)
    
    # Reference
    ref, us_ref = run_perftest(stream_distribute_mix_add_ref, x_expanded, y, H_post, M, num_iters=100, num_warmup=10)
    
    # Aiter
    out, us_aiter = run_perftest(stream_distribute_mix_add_forward, x_expanded, y, H_post, M, num_iters=100, num_warmup=10)
    
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-4, msg="stream_distribute_mix_add [ref vs aiter]")
    
    ret["B"] = B
    ret["n"] = n
    ret["C"] = C
    ret["err"] = err
    ret["us_ref"] = us_ref
    ret["us_aiter"] = us_aiter
    ret["speedup"] = us_ref / us_aiter if us_aiter > 0 else 0
    
    return ret


@benchmark()
def test_mhc_layer_forward(B: int, n: int, C: int, sinkhorn_iters: int = 20, dtype=torch.float32):
    """Test full MHC layer forward pass with static H parameters."""
    ret = {}
    
    x_expanded = torch.randn(B, n, C, dtype=dtype, device="cuda")
    rmsnorm_weight = torch.ones(C, dtype=dtype, device="cuda")
    H_pre = torch.randn(n, dtype=dtype, device="cuda")
    H_post = torch.randn(n, dtype=dtype, device="cuda")
    H_res = torch.randn(n, n, dtype=dtype, device="cuda")
    eps = 1e-5
    
    # Reference
    ref, us_ref = run_perftest(
        mhc_layer_ref, x_expanded, rmsnorm_weight, H_pre, H_post, H_res, eps, sinkhorn_iters,
        num_iters=100, num_warmup=10
    )
    
    # Aiter
    out, us_aiter = run_perftest(
        mhc_layer_forward, x_expanded, rmsnorm_weight, H_pre, H_post, H_res, eps, sinkhorn_iters,
        num_iters=100, num_warmup=10
    )
    
    err = checkAllclose(ref, out, rtol=1e-2, atol=1e-3, msg="mhc_layer_forward [ref vs aiter]")
    
    ret["B"] = B
    ret["n"] = n
    ret["C"] = C
    ret["sinkhorn_iters"] = sinkhorn_iters
    ret["err"] = err
    ret["us_ref"] = us_ref
    ret["us_aiter"] = us_aiter
    ret["speedup"] = us_ref / us_aiter if us_aiter > 0 else 0
    
    return ret


# ============================================================================
# Main test runner
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MHC Layer unit tests",
    )
    parser.add_argument(
        "-t", "--test",
        type=str,
        choices=["sinkhorn", "aggregate", "distribute", "layer", "all"],
        default="all",
        help="Test to run (default: all)",
    )
    parser.add_argument(
        "-B", "--batch_size",
        type=int,
        nargs="*",
        default=[1, 4, 16, 64],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "-n", "--expansion_rate",
        type=int,
        nargs="*",
        default=[4, 8, 16],
        help="Expansion rates (n) to test",
    )
    parser.add_argument(
        "-C", "--hidden_dim",
        type=int,
        nargs="*",
        default=[256, 512, 1024],
        help="Hidden dimensions to test",
    )
    args = parser.parse_args()
    
    run_sinkhorn = args.test in ["sinkhorn", "all"]
    run_aggregate = args.test in ["aggregate", "all"]
    run_distribute = args.test in ["distribute", "all"]
    run_layer = args.test in ["layer", "all"]
    
    # Test Sinkhorn-Knopp
    if run_sinkhorn:
        print("\n" + "=" * 60)
        print("Testing Sinkhorn-Knopp Normalization")
        print("=" * 60)
        
        df = []
        for n in args.expansion_rate:
            for num_iters in [10, 20]:
                print(f"\nTesting: n={n}, num_iters={num_iters}")
                ret = test_sinkhorn_knopp(n, num_iters)
                df.append({
                    "n": ret["n"],
                    "num_iters": ret["num_iters"],
                    "error": ret["err"],
                    "row_err": ret["row_err"],
                    "col_err": ret["col_err"],
                    "time_us (ref)": ret["us_ref"],
                    "time_us (aiter)": ret["us_aiter"],
                })
        
        df = pd.DataFrame(df)
        df["speedup"] = df["time_us (ref)"] / df["time_us (aiter)"]
        df_md = df.to_markdown(index=False)
        aiter.logger.info("Sinkhorn-Knopp summary (markdown):\n%s", df_md)
        
        # Test batched
        print("\n" + "-" * 40)
        print("Testing Batched Sinkhorn-Knopp")
        print("-" * 40)
        
        df = []
        for B in args.batch_size:
            for n in args.expansion_rate:
                print(f"\nTesting: B={B}, n={n}")
                ret = test_sinkhorn_knopp_batched(B, n)
                df.append({
                    "B": ret["B"],
                    "n": ret["n"],
                    "num_iters": ret["num_iters"],
                    "error": ret["err"],
                    "time_us (aiter)": ret["us_aiter"],
                })
        
        df = pd.DataFrame(df)
        df_md = df.to_markdown(index=False)
        aiter.logger.info("Batched Sinkhorn-Knopp summary (markdown):\n%s", df_md)
    
    # Test Stream Aggregate
    if run_aggregate:
        print("\n" + "=" * 60)
        print("Testing Stream Aggregate")
        print("=" * 60)
        
        df = []
        for B in args.batch_size:
            for n in args.expansion_rate:
                for C in args.hidden_dim:
                    print(f"\nTesting: B={B}, n={n}, C={C}")
                    ret = test_stream_aggregate(B, n, C)
                    df.append({
                        "B": ret["B"],
                        "n": ret["n"],
                        "C": ret["C"],
                        "error": ret["err"],
                        "time_us (ref)": ret["us_ref"],
                        "time_us (aiter)": ret["us_aiter"],
                    })
        
        df = pd.DataFrame(df)
        df["speedup"] = df["time_us (ref)"] / df["time_us (aiter)"]
        df_md = df.to_markdown(index=False)
        aiter.logger.info("Stream Aggregate summary (markdown):\n%s", df_md)
    
    # Test Stream Distribute Mix Add
    if run_distribute:
        print("\n" + "=" * 60)
        print("Testing Stream Distribute Mix Add")
        print("=" * 60)
        
        df = []
        for B in args.batch_size:
            for n in args.expansion_rate:
                for C in args.hidden_dim:
                    print(f"\nTesting: B={B}, n={n}, C={C}")
                    ret = test_stream_distribute_mix_add(B, n, C)
                    df.append({
                        "B": ret["B"],
                        "n": ret["n"],
                        "C": ret["C"],
                        "error": ret["err"],
                        "time_us (ref)": ret["us_ref"],
                        "time_us (aiter)": ret["us_aiter"],
                    })
        
        df = pd.DataFrame(df)
        df["speedup"] = df["time_us (ref)"] / df["time_us (aiter)"]
        df_md = df.to_markdown(index=False)
        aiter.logger.info("Stream Distribute Mix Add summary (markdown):\n%s", df_md)
    
    # Test Full MHC Layer
    if run_layer:
        print("\n" + "=" * 60)
        print("Testing Full MHC Layer Forward")
        print("=" * 60)
        
        df = []
        for B in args.batch_size:
            for n in args.expansion_rate:
                for C in args.hidden_dim:
                    print(f"\nTesting: B={B}, n={n}, C={C}")
                    ret = test_mhc_layer_forward(B, n, C)
                    df.append({
                        "B": ret["B"],
                        "n": ret["n"],
                        "C": ret["C"],
                        "sinkhorn_iters": ret["sinkhorn_iters"],
                        "error": ret["err"],
                        "time_us (ref)": ret["us_ref"],
                        "time_us (aiter)": ret["us_aiter"],
                    })
        
        df = pd.DataFrame(df)
        df["speedup"] = df["time_us (ref)"] / df["time_us (aiter)"]
        df_md = df.to_markdown(index=False)
        aiter.logger.info("MHC Layer Forward summary (markdown):\n%s", df_md)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
