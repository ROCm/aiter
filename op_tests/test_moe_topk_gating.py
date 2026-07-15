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
    benchmark,
    checkAllclose,
    perftest,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx
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
    # Fully vectorized on-device (no per-token Python loop): build boolean
    # [T, E] expert masks for the fused and reference selections and evaluate
    # the tie condition with tensor ops.
    T, E = sel_scores.shape
    dev = sel_scores.device
    sel = sel_scores.to(torch.float32)
    i_fused = i_fused.long()
    i_torch = i_torch.long()

    # Cutoff = k-th largest selection score per token.
    cutoff = sel.topk(topk, dim=-1).values.amin(dim=-1, keepdim=True)  # [T, 1]

    fused_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    fused_mask.scatter_(1, i_fused, True)
    ref_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    ref_mask.scatter_(1, i_torch, True)

    # Duplicate ids collapse in the mask; a full selection covers topk experts.
    fused_full = fused_mask.sum(dim=1) == topk
    ref_full = ref_mask.sum(dim=1) == topk
    match = (fused_mask == ref_mask).all(dim=1) & fused_full

    extra = fused_mask & ~ref_mask   # kernel-only -> must be >= cutoff - tol
    missing = ref_mask & ~fused_mask  # ref-only    -> must be <= cutoff + tol
    extra_ok = ((~extra) | (sel >= (cutoff - tol))).all(dim=1)
    missing_ok = ((~missing) | (sel <= (cutoff + tol))).all(dim=1)
    excused = fused_full & ref_full & extra_ok & missing_ok

    bad = (~match) & (~excused)
    mism = int(bad.sum().item())

    if os.environ.get("TOPK_TIE_DEBUG", "0") != "0":
        has_bias = bias is not None and bias.numel() > 0
        bias_cpu = bias.float().cpu() if has_bias else None
        sel_cpu = sel.cpu()
        cut_cpu = cutoff.squeeze(1).cpu()
        extra_cpu, missing_cpu, bad_cpu = extra.cpu(), missing.cpu(), bad.cpu()
        for t in (~match).cpu().nonzero(as_tuple=True)[0].tolist():
            thr = float(cut_cpu[t])

            def _fmt(e):
                s = float(sel_cpu[t, e])
                b = float(bias_cpu[e]) if has_bias else 0.0
                return (
                    f"      expert {e:4d}: f(x)={s - b:+.7f}  bias={b:+.7f}  "
                    f"f(x)+bias={s:+.7f}  gap_to_cutoff={s - thr:+.2e}"
                )

            tag = "REAL MISMATCH" if bool(bad_cpu[t]) else "TIE (excused)"
            print(
                f"[TIE_DEBUG]{(' ' + label) if label else ''} token {t}: {tag}  "
                f"cutoff(k={topk})={thr:+.7f}"
            )
            print("    kernel-only (picked by fused, not ref):")
            for e in extra_cpu[t].nonzero(as_tuple=True)[0].tolist():
                print(_fmt(e))
            print("    ref-only (picked by torch, not fused):")
            for e in missing_cpu[t].nonzero(as_tuple=True)[0].tolist():
                print(_fmt(e))
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


def _torch_weight_aligned_to_fused(w_fused, i_fused, w_torch, i_torch):
    """Scatter the torch (ref) weights into a dense [T, E] map, then gather them
    back in the fused id order. Returns (ref_w_aligned, matched_mask) so callers
    can compare fused vs ref weights for the experts both selected -- fully
    vectorized, no per-token Python loop."""
    T = w_fused.shape[0]
    dev = w_fused.device
    E = int(max(int(i_fused.max()), int(i_torch.max())) + 1)
    dense = torch.zeros((T, E), dtype=torch.float32, device=dev)
    mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    dense.scatter_(1, i_torch.long(), w_torch.to(torch.float32))
    mask.scatter_(1, i_torch.long(), True)
    ref = dense.gather(1, i_fused.long())
    matched = mask.gather(1, i_fused.long())
    return ref, matched


def _max_weight_error(w_fused, i_fused, w_torch, i_torch):
    """Max absolute weight error across tokens for matched expert ids."""
    ref, matched = _torch_weight_aligned_to_fused(w_fused, i_fused, w_torch, i_torch)
    if not bool(matched.any()):
        return 0.0
    diff = (w_fused.to(torch.float32) - ref).abs()
    return float(diff[matched].max())


def _assert_weights_close(w_fused, i_fused, w_torch, i_torch):
    """Assert matched expert weights are close; skip tie-swapped experts."""
    ref, matched = _torch_weight_aligned_to_fused(w_fused, i_fused, w_torch, i_torch)
    torch.testing.assert_close(
        w_fused.to(torch.float32)[matched], ref[matched], atol=1e-5, rtol=1e-4
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


# Regression test for the softmax + correction_bias path.
#
# The prefill kernel (topk_softplus_kernel_prefill) must add bias AFTER softmax
# normalization: softmax is computed over the raw logits and bias is only added
# to the selection score. A previous version added bias in the vectorized-load
# phase (missing the `if constexpr(SCORE_FUNC != SCORE_SOFTMAX)` guard the smem
# kernels have), which normalized over (logit+bias) and double-counted bias,
# corrupting both the routing and the reported unbiased weights.
#
# This also exercises the type-erased bias path (bias dtype is a runtime tag,
# not a template arg) for score_func="softmax" across all supported bias dtypes.
# num_tokens covers both the decode/TPW=1 prefill path (64) and the higher-TPW
# multi-token prefill path (1024).
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024])
@pytest.mark.parametrize("num_experts", [128, 256])
def test_topk_softmax_bias_correctness(num_experts, num_tokens, topk, dtype, bias_dtype):
    """topk_gating softmax with correction_bias: routing + unbiased weights must
    match the fp32 reference across gating/bias dtype combinations."""
    torch.random.manual_seed(0)
    route_scale = 1.0

    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(
        bias_dtype
    )

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
        label=f"softmax+bias E={num_experts} k={topk} gating={dtype} bias={bias_dtype}",
    )
    assert n_mism == 0, (
        f"E={num_experts},topk={topk},gating={dtype},bias={bias_dtype}: "
        f"{n_mism}/{num_tokens} tokens have non-tie ID mismatches"
    )

    # The reported weights must be the *unbiased* softmax weights; the old bug
    # made these normalize over (logit+bias) and could even exceed max softmax.
    _assert_weights_close(w_fused, i_fused, w_torch, i_torch)


def _ref_selection_with_nan(gating_output, bias, score_func):
    """fp32 reference selection score matching the kernel's non-finite handling.
    Reference only -- not timed, not in the table.

    Non-finite semantics (per score function), mirroring the kernel:
    - NaN: always excluded (never selected).
    - +Inf: sigmoid saturates to 1 (selectable); sqrt(softplus) clamps the logit
      to 1e30 (selectable, top-ranked, finite); softmax excludes it (its exp is
      mapped to -inf so it neither poisons the sum nor gets selected).
    - -Inf: score -> 0 (low, not selected) for every function.
    """
    gf = gating_output.float()
    nan = torch.isnan(gf)
    posinf = torch.isposinf(gf)
    b = bias.float() if (bias is not None and bias.numel() > 0) else 0.0
    if score_func == "softmax":
        # NaN and +Inf are excluded from the normalization and from selection.
        s = torch.softmax(gf.masked_fill(nan | posinf, float("-inf")), dim=-1)
        sel = s + b
        exclude = nan | posinf
    elif score_func == "sigmoid":
        sel = torch.sigmoid(gf) + b  # sigmoid(+inf)=1, sigmoid(-inf)=0
        exclude = nan
    else:  # sqrtsoftplus: kernel clamps the logit to 1e30 before softplus
        sel = torch.sqrt(torch.nn.functional.softplus(torch.clamp(gf, max=1.0e30))) + b
        exclude = nan
    return sel.masked_fill(exclude, float("-inf"))


@benchmark()
def bench_topk_gating_nan(num_experts, num_tokens, topk, score_func, dtype):
    """NaN/Inf robustness as an aiter-standard benchmark.

    Injects NaN, +Inf and -Inf experts scattered per token, times the fused
    topk_gating kernel with run_perftest (preallocated output buffers, as the
    model calls it), and checks the routed top-k SET against a reference that
    mirrors the kernel's non-finite handling (see _ref_selection_with_nan).
    Routing is a set match (with tie tolerance), so ``err`` is the fraction of
    tokens whose selected expert set differs; ``nan_leak`` flags any NaN in the
    output weights. Memory-bound op -> TB/s (no meaningful FLOPs).
    """
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(dtype)

    # Scatter NaN across token-dependent positions (so it lands anywhere in a
    # lane's sorted partition) plus a -Inf per token.
    tok = torch.arange(num_tokens, device="cuda")
    for j in range(4):
        gating_output[tok, (tok * (7 * j + 3) + j) % num_experts] = float("nan")
    gating_output[tok, (tok * 11 + 2) % num_experts] = float("-inf")
    # +Inf is a valid extreme logit for the per-element scores (sigmoid saturates
    # to 1; sqrt(softplus) clamps to a finite top-ranked score). Softmax is
    # excluded here: a +Inf makes the row-max +Inf so every exp(finite-inf)=0 and
    # the row collapses -- excluding +Inf from the softmax max/selection is a
    # separate hardening not covered by this test.
    if score_func != "softmax":
        gating_output[tok, (tok * 5 + 1) % num_experts] = float("inf")

    # Preallocated output buffers, matching the real model call.
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    need_renorm = score_func != "softmax"

    _, us = run_perftest(
        aiter.topk_gating,
        topk_weights,
        topk_ids,
        gating_output,
        bias,
        need_renorm=need_renorm,
        routed_scaling_factor=2.5,
        score_func=score_func,
    )

    # Correctness: routed set vs the NaN-excluding fp32 reference (tie-tolerant).
    sel = _ref_selection_with_nan(gating_output, bias, score_func)
    i_ref = sel.topk(topk, dim=-1, sorted=False)[1].to(torch.int32)
    n_mism = _count_routing_mismatches(
        topk_ids,
        i_ref,
        sel,
        topk,
        bias=bias,
        label=f"nan {score_func} E={num_experts} T={num_tokens} k={topk}",
    )
    nan_leak = bool(topk_weights.isnan().any().item())

    # Memory-bound: reads the [T, E] gating matrix, writes T*topk ids + weights.
    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    ret["fused us"] = us
    ret["fused TB/s"] = nbytes / us / 1e6
    ret["fused err"] = n_mism / num_tokens
    ret["nan_leak"] = nan_leak
    return ret


# pytest wrapper: NaN experts must never be selected (top-k set matches the
# NaN-excluding reference) and must never leak into the output weights. Covers
# decode (T=64) and prefill (T=2048) dispatch paths.
@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "sigmoid", "softmax"])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_tokens", [64, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
def test_topk_gating_nan(num_experts, num_tokens, topk, score_func):
    row = bench_topk_gating_nan(
        num_experts, num_tokens, topk, score_func, torch.bfloat16
    )
    assert not row["nan_leak"], (
        f"{score_func} E={num_experts} T={num_tokens} k={topk}: NaN leaked into weights"
    )
    assert row["fused err"] == 0.0, (
        f"{score_func} E={num_experts} T={num_tokens} k={topk}: routed top-k set "
        f"differs from the NaN-excluding reference (err={row['fused err']})"
    )


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
        choices=["sigmoid", "softplus", "softmax", "nan", "all"],
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

    if args.test in ("nan", "all"):
        nan_dtypes = [d for d in dtype_list if d != torch.float32]
        nan_configs = list(
            itertools.product(
                num_experts_list,
                num_tokens_list,
                topk_list,
                ["sqrtsoftplus", "sigmoid", "softmax"],
                nan_dtypes,
            )
        )
        print("\n" + "=" * 80)
        print("topk_gating NaN/Inf robustness")
        print("=" * 80)
        collected = [
            bench_topk_gating_nan(num_experts, num_tokens, topk, score_func, dtype)
            for num_experts, num_tokens, topk, score_func, dtype in nan_configs
        ]
        df = pd.DataFrame(collected)
        aiter.logger.info(
            "topk_gating NaN/Inf robustness summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        errors = df[(df["fused err"] > 0) | (df["nan_leak"])]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} nan config(s) failed (err>0 or nan_leak)!")
            print(errors.to_string(index=False))
            failed_sections.append("nan")
        else:
            print("All nan robustness tests passed!")
    print("=" * 80)

    if failed_sections:
        print(
            f"FAIL: correctness regression in section(s): "
            f"{', '.join(failed_sections)}",
            file=sys.stderr,
        )
        sys.exit(1)
