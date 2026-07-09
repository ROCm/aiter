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
import os
import sys

import pandas as pd
import pytest
import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    perftest,
)
from aiter.utility.dtypes import str2Dtype, str2tuple

# NOTE on correctness metrics by score function:
# - sigmoid uses element-wise comparison (score_errors/index_errors) because
#   both torch/topk and fused paths return sorted top-K.
# - softplus/softmax use set-based ID matching (id_errors/max_weight_err)
#   because torch references intentionally use `topk(..., sorted=False)` to
#   mirror routing behavior where top-K order is not semantically required.
#
# Tie-aware selection: the fused kernel scores experts with hardware-approximate
# math (exp2f/log2f, ~1e-6 ULP), while the torch reference uses exact libm. When
# two experts straddle the top-K cutoff with biased selection scores closer than
# this noise, which one wins is a genuine tie and the choice is semantically
# irrelevant (the swapped experts carry near-identical weights). We must NOT flag
# such boundary ties as errors, otherwise tiny token counts (e.g. 64) make a
# single harmless flip exceed the 1% threshold. `_count_routing_mismatches`
# excuses a token iff every kernel-only expert sits within `tol` below the cutoff
# and every reference-only expert sits within `tol` above it.

# Tolerance for boundary ties: ~1e-4 is ~100x the kernel's score-approximation
# noise (~1e-6 on O(1) scores), so genuine routing bugs (gaps >> 1e-4) are still
# caught while harmless tie flips are excused.
_TIE_TOL = 1e-4


def _selection_scores(
    gating_output: torch.Tensor, bias: torch.Tensor, score_func: str
) -> torch.Tensor:
    """Reference biased selection scores [num_tokens, num_experts] in fp32.

    These mirror exactly what the torch reference (and the kernel) sort by to
    pick the top-K: sqrt(softplus(x))+bias for softplus, softmax(x)+bias for
    softmax (bias is added AFTER softmax normalization, matching the kernel).
    """
    g = gating_output.float()
    if score_func == "softplus":
        scores = torch.nn.functional.softplus(g).sqrt()
    elif score_func == "softmax":
        scores = torch.softmax(g, dim=-1)
    else:
        raise ValueError(f"unsupported score_func: {score_func}")
    if bias is not None and bias.numel() > 0:
        scores = scores + bias.float()
    return scores


def _count_routing_mismatches(
    i_fused: torch.Tensor,
    i_torch: torch.Tensor,
    sel_scores: torch.Tensor,
    topk: int,
    tol: float = _TIE_TOL,
    *,
    bias: torch.Tensor = None,
    label: str = "",
) -> int:
    """Number of tokens whose selected expert set differs from the reference in
    a way NOT explained by a near-tie at the top-K selection boundary.

    A token is excused when every kernel-only expert has a selection score within
    `tol` below the reference cutoff and every reference-only expert is within
    `tol` above it (i.e. all disagreements sit on the cutoff and are ties).

    Set env TOPK_TIE_DEBUG=1 to print, for every disagreeing token, the boundary
    experts with their unbiased score f(x), bias, biased selection score f(x)+bias
    and the gap to the cutoff -- evidence that disagreements are genuine ties
    created by the bias bringing two experts' biased scores nearly equal.
    """
    debug = os.environ.get("TOPK_TIE_DEBUG", "0") != "0"
    # Copy small tensors to CPU once to avoid per-token CUDA syncs.
    i_fused_cpu = i_fused.cpu()
    i_torch_cpu = i_torch.cpu()
    sel_cpu = sel_scores.cpu()
    sorted_sel, _ = sel_cpu.sort(dim=-1, descending=True)
    cutoff = sorted_sel[:, topk - 1]  # k-th largest selection score per token
    has_bias = bias is not None and bias.numel() > 0
    bias_cpu = bias.cpu() if has_bias else None
    mism = 0
    for t in range(i_fused_cpu.shape[0]):
        fused_row = i_fused_cpu[t].tolist()
        torch_row = i_torch_cpu[t].tolist()
        kset = set(fused_row)
        rset = set(torch_row)
        # Duplicate expert IDs collapse in set(); treat as real mismatch.
        if kset == rset and len(kset) == topk:
            continue
        thr = cutoff[t].item()
        extra = kset - rset  # kernel picked, ref didn't -> must be >= thr - tol
        missing = rset - kset  # ref picked, kernel didn't -> must be <= thr + tol
        excused = (
            len(kset) == topk
            and len(rset) == topk
            and all(sel_cpu[t, e].item() >= thr - tol for e in extra)
            and all(sel_cpu[t, e].item() <= thr + tol for e in missing)
        )
        if not excused:
            mism += 1

        if debug:

            def _fmt(e):
                sel = sel_cpu[t, e].item()
                b = float(bias_cpu[e]) if has_bias else 0.0
                return (
                    f"      expert {e:4d}: f(x)={sel - b:+.7f}  bias={b:+.7f}  "
                    f"f(x)+bias={sel:+.7f}  gap_to_cutoff={sel - thr:+.2e}"
                )

            tag = "TIE (excused)" if excused else "REAL MISMATCH"
            print(
                f"[TIE_DEBUG]{(' ' + label) if label else ''} token {t}: {tag}  "
                f"cutoff(k={topk})={thr:+.7f}"
            )
            print("    kernel-only (picked by fused, not ref):")
            for e in sorted(extra):
                print(_fmt(e))
            print("    ref-only (picked by torch, not fused):")
            for e in sorted(missing):
                print(_fmt(e))
            for ek in sorted(extra):
                for er in sorted(missing):
                    d = abs(sel_cpu[t, ek].item() - sel_cpu[t, er].item())
                    print(
                        f"    |f(x)+bias gap| between kernel#{ek} and ref#{er} = {d:.2e}"
                    )
    return mism


@perftest(num_iters=10, num_warmup=1)
def run_torch(gating_output: torch.Tensor, topk: int):
    # llama4 maverick custom routing function
    router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores, router_indices.to(torch.int32)


@perftest(num_iters=100, num_warmup=1)
def run_fused(gating_output: torch.Tensor, topk: int):
    tokens, num_experts = gating_output.shape
    router_scores = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    router_indices = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    aiter.topk_gating(
        router_scores,
        router_indices,
        gating_output,
        score_func="sigmoid",
        need_renorm=False,
    )
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


@perftest(num_iters=100, num_warmup=1)
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


# -- topk_softmax ( classic MoE softmax routing) --------------
@perftest(num_iters=10, num_warmup=1)
def run_torch_softmax(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
):
    scores = torch.softmax(gating_output.float(), dim=-1)
    scores_biased = scores + bias.float() if bias.numel() > 0 else scores
    topk_ids = scores_biased.topk(topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_ids) * route_scale
    return topk_weights, topk_ids.to(torch.int32)


@perftest(num_iters=100, num_warmup=1)
def run_fused_softmax(
    gating_output: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
):
    tokens, _ = gating_output.shape
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    aiter.topk_gating(
        topk_weights,
        topk_ids,
        gating_output,
        bias,
        need_renorm=False,  # softmax is already normalized
        routed_scaling_factor=route_scale,
        score_func="softmax",
    )
    return topk_weights, topk_ids


@perftest(num_iters=100, num_warmup=1)
def run_vllm_softmax(
    gating_output: torch.Tensor,
    topk: int,
    route_scale: float,
):
    """vLLM-adapted topkGatingSoftmax kernel (topk_softmax_kernels.cu)."""
    tokens, _ = gating_output.shape
    topk_weights = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    token_expert_indices = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    # need_renorm=True: renorm among top-K (matches softmax-route convention)
    aiter.topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output, need_renorm=False
    )
    if route_scale != 1.0:
        topk_weights.mul_(route_scale)
    return topk_weights, topk_ids


def benchmark_topk_sigmoid(
    num_experts: int = 128,
    num_tokens: int = 1024,
    topk: int = 4,
    dtype: torch.dtype = torch.float16,
):
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
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


def _max_weight_error(w_fused, i_fused, w_torch, i_torch):
    """Max absolute weight error across tokens for matched expert ids."""
    max_err = 0.0
    for t in range(w_fused.shape[0]):
        for k in range(w_fused.shape[1]):
            kid = i_fused[t, k].item()
            ref_k = (i_torch[t] == kid).nonzero(as_tuple=True)[0]
            if len(ref_k) > 0:
                max_err = max(
                    max_err,
                    abs(w_fused[t, k].item() - w_torch[t, ref_k[0]].item()),
                )
    return max_err


def _assert_weights_close(w_fused, i_fused, w_torch, i_torch):
    """Assert matched expert weights are close; skip tie-swapped experts."""
    for t in range(w_fused.shape[0]):
        for k in range(w_fused.shape[1]):
            kid = i_fused[t, k].item()
            ref_k = (i_torch[t] == kid).nonzero(as_tuple=True)[0]
            if len(ref_k) == 0:
                continue
            torch.testing.assert_close(
                w_fused[t, k], w_torch[t, ref_k[0]], atol=1e-5, rtol=1e-4
            )


def _make_gating(num_experts, num_tokens, dtype):
    """Shuffled uniform gating output -- each row has unique values."""
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    return torch.gather(gating_output, dim=-1, index=permutation).contiguous()


def benchmark_topk_softplus(
    num_experts: int = 256,
    num_tokens: int = 1024,
    topk: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    renormalize: bool = True,
    route_scale: float = 2.5,
):
    torch.random.manual_seed(1)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = torch.randn(num_experts, dtype=dtype, device="cuda") * 0.1

    (w_torch, i_torch), avg_torch = run_torch_softplus(
        gating_output.clone(), bias, topk, renormalize, route_scale
    )
    (w_fused, i_fused), avg_fused = run_fused_softplus(
        gating_output.clone(), bias, topk, renormalize, route_scale
    )

    sel = _selection_scores(gating_output, bias, "softplus")
    id_err = (
        _count_routing_mismatches(
            i_fused,
            i_torch,
            sel,
            topk,
            bias=bias,
            label=f"softplus E={num_experts} T={num_tokens} k={topk} {dtype}",
        )
        / num_tokens
    )

    result = {
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
        "dtype": str(dtype).split(".")[-1],
        "torch_us": avg_torch,
        "fused_us": avg_fused,
        "uplift": avg_torch / avg_fused,
        "id_errors": id_err,
        "max_weight_err": _max_weight_error(w_fused, i_fused, w_torch, i_torch),
    }
    if id_err > 0.01:
        print(
            f"\n[ERROR] softplus: num_experts={num_experts}, num_tokens={num_tokens}, "
            f"topk={topk}, dtype={str(dtype).split('.')[-1]}, id_err={id_err:.4f}"
        )
    return result


def benchmark_topk_softmax(
    num_experts: int = 256,
    num_tokens: int = 1024,
    topk: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    route_scale: float = 1.0,
    use_bias: bool = False,
):
    torch.random.manual_seed(2)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (
        torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1
        if use_bias
        else torch.empty(0, device="cuda")
    )

    (w_torch, i_torch), avg_torch = run_torch_softmax(
        gating_output.clone(), bias, topk, route_scale
    )
    (w_fused, i_fused), avg_fused = run_fused_softmax(
        gating_output.clone(), bias, topk, route_scale
    )
    # vLLM kernel: no bias support, pass gating_output directly
    (w_vllm, i_vllm), avg_vllm = run_vllm_softmax(
        gating_output.clone(), topk, route_scale
    )

    sel = _selection_scores(gating_output, bias, "softmax")
    id_err_fused = (
        _count_routing_mismatches(
            i_fused,
            i_torch,
            sel,
            topk,
            bias=bias,
            label=f"softmax/fused E={num_experts} T={num_tokens} k={topk} {dtype}",
        )
        / num_tokens
    )
    # vLLM kernel compared against torch (no bias, sel_scores == softmax(x))
    sel_nobias = _selection_scores(gating_output, torch.empty(0), "softmax")
    id_err_vllm = (
        _count_routing_mismatches(
            i_vllm,
            i_torch,
            sel_nobias,
            topk,
            label=f"softmax/vllm E={num_experts} T={num_tokens} k={topk} {dtype}",
        )
        / num_tokens
    )

    result = {
        "num_experts": num_experts,
        "num_tokens": num_tokens,
        "topk": topk,
        "dtype": str(dtype).split(".")[-1],
        "torch_us": avg_torch,
        "fused_us": avg_fused,
        "vllm_us": avg_vllm,
        "fused_uplift": avg_torch / avg_fused,
        "vllm_uplift": avg_torch / avg_vllm,
        "id_err_fused": id_err_fused,
        "id_err_vllm": id_err_vllm,
    }
    for label, id_err in (("fused", id_err_fused), ("vllm", id_err_vllm)):
        if id_err > 0.01:
            print(
                f"\n[ERROR] softmax/{label}: num_experts={num_experts}, num_tokens={num_tokens}, "
                f"topk={topk}, dtype={str(dtype).split('.')[-1]}, id_err={id_err:.4f}"
            )
    return result


# Pytest-parametrized test functions -- topk_softplus
# Mirrors DeepSeek-V4 model integration: gating fp32 + bias fp32 is the default.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_softplus_correctness(num_experts, num_tokens, topk, dtype, bias_dtype):
    """Pytest test for correctness of topk_softplus (sqrtsoftplus) operation.

    Covers the DeepSeek-V4-Pro use case: router_logits=fp32, bias=fp32.
    Also covers fp16/bf16 gating with mixed bias dtypes.
    """
    torch.random.manual_seed(0)
    route_scale = 2.5

    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(
        bias_dtype
    )

    (w_torch, i_torch), _ = run_torch_softplus(
        gating_output.clone(), bias, topk, True, route_scale
    )
    (w_fused, i_fused), _ = run_fused_softplus(
        gating_output.clone(), bias, topk, True, route_scale
    )

    sel = _selection_scores(gating_output, bias, "softplus")
    n_mism = _count_routing_mismatches(
        i_fused,
        i_torch,
        sel,
        topk,
        bias=bias,
        label=f"softplus gating={dtype} bias={bias_dtype} E={num_experts} k={topk}",
    )
    assert n_mism == 0, (
        f"gating={dtype},bias={bias_dtype},E={num_experts},topk={topk}: "
        f"{n_mism}/{num_tokens} tokens have non-tie ID mismatches"
    )

    _assert_weights_close(w_fused, i_fused, w_torch, i_torch)


# Pytest-parametrized test functions -- topk_sigmoid
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_sigmoid_correctness(num_experts, num_tokens, topk, dtype):
    """Pytest test for correctness of topk_sigmoid operation."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)

    # run both implementations
    (scores_torch, indices_torch), _ = run_torch(gating_output.clone(), topk)
    (scores_fused, indices_fused), _ = run_fused(gating_output.clone(), topk)

    # check correctness
    score_errors = checkAllclose(scores_torch, scores_fused, tol_err_ratio=0.01)
    index_errors = checkAllclose(indices_torch, indices_fused, tol_err_ratio=0.01)

    # Assert correctness
    assert score_errors <= 0.01, f"Score errors {score_errors} exceed tolerance"
    assert index_errors <= 0.01, f"Index errors {index_errors} exceed tolerance"


# Pytest-parametrized test functions -- topk_softmax (via topk_gating)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_softmax_correctness(num_experts, num_tokens, topk, dtype):
    """Pytest test for correctness of topk_gating with score_func='softmax'."""
    torch.random.manual_seed(0)
    route_scale = 1.0

    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

    (w_torch, i_torch), _ = run_torch_softmax(
        gating_output.clone(), bias, topk, route_scale
    )
    (w_fused, i_fused), _ = run_fused_softmax(
        gating_output.clone(), bias, topk, route_scale
    )

    sel = _selection_scores(gating_output, bias, "softmax")
    n_mism = _count_routing_mismatches(
        i_fused,
        i_torch,
        sel,
        topk,
        bias=bias,
        label=f"softmax E={num_experts} k={topk} dtype={dtype}",
    )
    assert n_mism == 0, (
        f"E={num_experts},topk={topk},dtype={dtype}: "
        f"{n_mism}/{num_tokens} tokens have non-tie ID mismatches"
    )

    _assert_weights_close(w_fused, i_fused, w_torch, i_torch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test topk_sigmoid and topk_softplus operations"
    )
    parser.add_argument(
        "--num-experts",
        type=str2tuple,
        default=[64, 128, 256, 384],
        help="Comma-separated list of number of experts (default: 64,128,256,384)",
    )
    parser.add_argument(
        "--num-tokens",
        type=str2tuple,
        default=[16384, 4096, 1024, 256, 64, 1],
        help="Comma-separated list of number of tokens (default: 64,1024,2048)",
    )
    parser.add_argument(
        "--topk",
        type=str2tuple,
        default=[1, 2, 4, 6, 8],
        help="Comma-separated list of topk values (default: 1,2,4,6,8)",
    )
    parser.add_argument(
        "--dtype",
        type=str2Dtype,
        default=[torch.float16, torch.bfloat16, torch.float32],
        help="Comma-separated list of dtypes: fp16, bf16, fp32 (default: fp16,bf16,fp32)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["sigmoid", "softplus", "softmax", "all"],
        help="Which test to run (default: all)",
    )

    args = parser.parse_args()

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    num_experts_list = to_list(args.num_experts)
    num_tokens_list = to_list(args.num_tokens)
    topk_list = to_list(args.topk)
    dtype_list = to_list(args.dtype)

    # Track whether any benchmark section saw a correctness regression
    # (id_errors > 1%); exit non-zero at the end so CI catches it.
    failed_sections: list[str] = []

    if args.test in ("sigmoid", "all"):
        sigmoid_experts = [e for e in num_experts_list]
        sigmoid_dtypes = [d for d in dtype_list if d != torch.float32]
        sigmoid_configs = list(
            itertools.product(
                sigmoid_experts, num_tokens_list, topk_list, sigmoid_dtypes
            )
        )
        print("=" * 80)
        print("topk_sigmoid benchmark")
        print("=" * 80)
        collected = []
        for num_experts, num_tokens, topk, dtype in sigmoid_configs:
            result = benchmark_topk_sigmoid(
                num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
            )
            collected.append(result)
        df = pd.DataFrame(collected)
        print(df.to_string(index=False))
        print(f"\nAverage uplift: {df['uplift'].mean():.2f}x")
        # benchmark_topk_sigmoid uses {score,index}_errors columns
        errors = df[(df["index_errors"] > 0.01) | (df["score_errors"] > 0.01)]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} sigmoid config(s) had errors > 1%!")
            print(errors.to_string(index=False))
            failed_sections.append("sigmoid")

    if args.test in ("softplus", "all"):
        softplus_configs = list(
            itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
        )
        print("\n" + "=" * 80)
        print("topk_softplus benchmark")
        print("=" * 80)
        collected = []
        for num_experts, num_tokens, topk, dtype in softplus_configs:
            result = benchmark_topk_softplus(
                num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
            )
            collected.append(result)
        df = pd.DataFrame(collected)
        print(df.to_string(index=False))
        print(f"\nAverage uplift: {df['uplift'].mean():.2f}x")
        errors = df[df["id_errors"] > 0.01]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} softplus config(s) had id errors > 1%!")
            print(errors.to_string(index=False))
            failed_sections.append("softplus")
        else:
            print("All softplus tests passed!")

    if args.test in ("softmax", "all"):
        softmax_configs = list(
            itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
        )
        print("\n" + "=" * 80)
        print("topk_softmax benchmark: topk_gating (fused) vs topk_softmax (vLLM)")
        print("=" * 80)
        collected = []
        for num_experts, num_tokens, topk, dtype in softmax_configs:
            result = benchmark_topk_softmax(
                num_experts=num_experts, num_tokens=num_tokens, topk=topk, dtype=dtype
            )
            collected.append(result)
        df = pd.DataFrame(collected)
        print(df.to_string(index=False))
        print(f"\nAverage fused uplift: {df['fused_uplift'].mean():.2f}x")
        print(f"Average vllm  uplift: {df['vllm_uplift'].mean():.2f}x")
        errors = df[(df["id_err_fused"] > 0.01) | (df["id_err_vllm"] > 0.01)]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} softmax config(s) had id errors > 1%!")
            print(errors.to_string(index=False))
            failed_sections.append("softmax")
        else:
            print("All softmax tests passed!")
    print("=" * 80)

    if failed_sections:
        print(
            f"FAIL: correctness regression in section(s): "
            f"{', '.join(failed_sections)}",
            file=sys.stderr,
        )
        sys.exit(1)
