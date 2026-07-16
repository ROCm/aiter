# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Test topk_gating (topk_sigmoid / topk_softplus / topk_softmax) operations with
various configurations.

This test can be run in two ways:

1. Using pytest (for automated testing):
   pytest test_moe_topk_gating.py -v

2. Using command line arguments (for benchmarking with summary table):
   python test_moe_topk_gating.py --num-experts 64,128 --topk 2,4,8 --dtype fp16
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
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility.dtypes import str2Dtype, str2tuple

# NOTE on correctness metrics by score function:
# - sigmoid uses element-wise comparison (score_err/idx_err) because both
#   torch and the fused kernel return sorted top-K.
# - softplus/softmax use set-based ID matching (err/max_weight_err) because
#   torch references intentionally use `topk(..., sorted=False)` to mirror
#   routing behavior where top-K order is not semantically required.
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

# Max abs weight error for matched expert ids (softplus/softmax). ~100x the
# kernel's exp2f/log2f approximation noise on O(1) weights.
_WEIGHT_TOL = 1e-4


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

    extra = fused_mask & ~ref_mask  # kernel-only -> must be >= cutoff - tol
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


def _make_gating(num_experts, num_tokens, dtype):
    """Shuffled uniform gating output -- each row has unique values."""
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    return torch.gather(gating_output, dim=-1, index=permutation).contiguous()


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
    """Max absolute weight error, restricted to tokens whose fused and torch
    selected SETS are identical.

    With renormalization the weight of even a commonly-selected expert depends on
    the whole selected set (shared denominator = sum over the top-k). A benign
    boundary tie-swap in one slot -- the exp/log HW approximation ranking two
    near-equal experts differently from exact libm, already accepted by
    _count_routing_mismatches -- therefore shifts the *other* experts' weights by
    a non-trivial amount. Comparing weights across a swapped set is a false
    mismatch, so only same-set tokens are measured here."""
    T = w_fused.shape[0]
    dev = w_fused.device
    E = int(max(int(i_fused.max()), int(i_torch.max())) + 1)
    fused_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    fused_mask.scatter_(1, i_fused.long(), True)
    torch_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    torch_mask.scatter_(1, i_torch.long(), True)
    same_set = (fused_mask == torch_mask).all(dim=1)  # [T]

    ref, matched = _torch_weight_aligned_to_fused(w_fused, i_fused, w_torch, i_torch)
    use = matched & same_set.unsqueeze(1)
    if not bool(use.any()):
        return 0.0
    diff = (w_fused.to(torch.float32) - ref).abs()
    return float(diff[use].max())


# ---------------------------------------------------------------------------
# torch references (fp32, untimed -- never enter the perf table, see
# .claude/skills/aiter-op-test/SKILL.md rule 4)
# ---------------------------------------------------------------------------


def ref_sigmoid(gating_output: torch.Tensor, topk: int):
    """Llama4 routing: select top-K by raw logit, weight = sigmoid(selected)."""
    scores, indices = torch.topk(gating_output, topk, dim=-1)
    return torch.sigmoid(scores.float()), indices.to(torch.int32)


def ref_softplus(
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


def ref_softmax(
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


# ---------------------------------------------------------------------------
# topk_sigmoid (Llama4 routing, via topk_gating score_func="sigmoid")
# ---------------------------------------------------------------------------


@benchmark()
def bench_topk_sigmoid(num_experts, num_tokens, topk, dtype):
    """Single fused candidate. Both torch and the fused kernel return
    sorted-descending top-K here, so scores/indices compare element-wise."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    ref_scores, ref_idx = ref_sigmoid(gating_output, topk)

    def run_fused():
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device="cuda"
        )
        topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
        aiter.topk_gating(
            topk_weights,
            topk_ids,
            gating_output,
            score_func="sigmoid",
            need_renorm=False,
        )
        return topk_weights, topk_ids

    candidates = {"fused": run_fused}

    # Memory-bound: reads the [T, E] gating matrix, writes T*topk ids + weights.
    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (w, ids), us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} score_err"] = checkAllclose(
            ref_scores,
            w.to(torch.float32),
            tol_err_ratio=0.01,
            msg=f"{name}: sigmoid scores",
        )
        ret[f"{name} idx_err"] = checkAllclose(
            ref_idx, ids, tol_err_ratio=0.01, msg=f"{name}: sigmoid indices"
        )
    return ret


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_sigmoid_correctness(num_experts, num_tokens, topk, dtype):
    row = bench_topk_sigmoid(num_experts, num_tokens, topk, dtype)
    assert row["fused score_err"] <= 0.01, (
        f"E={num_experts},T={num_tokens},topk={topk},{dtype}: "
        f"score error {row['fused score_err']} exceeds tolerance"
    )
    assert row["fused idx_err"] <= 0.01, (
        f"E={num_experts},T={num_tokens},topk={topk},{dtype}: "
        f"index error {row['fused idx_err']} exceeds tolerance"
    )


# ---------------------------------------------------------------------------
# topk_softplus (DeepSeek V4-Pro sqrtsoftplus routing, via topk_gating)
# ---------------------------------------------------------------------------


@benchmark()
def bench_topk_softplus(
    num_experts,
    num_tokens,
    topk,
    dtype,
    bias_dtype=torch.float32,
    renormalize=True,
    route_scale=2.5,
):
    """Single fused candidate (topk_softplus and topk_gating(score_func=
    "sqrtsoftplus") share the same underlying kernel). Default bias_dtype=fp32
    matches the DeepSeek-V4 real model: bf16 router logits + fp32 correction
    bias."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(
        bias_dtype
    )

    w_torch, i_torch = ref_softplus(gating_output, bias, topk, renormalize, route_scale)

    def run_fused():
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device="cuda"
        )
        topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
        aiter.topk_gating(
            topk_weights,
            topk_ids,
            gating_output,
            bias,
            need_renorm=renormalize,
            routed_scaling_factor=route_scale,
            score_func="sqrtsoftplus",
        )
        return topk_weights, topk_ids

    candidates = {"fused": run_fused}

    sel = _selection_scores(gating_output, bias, "softplus")
    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (w, ids), us = run_perftest(fn)
        n_mism = _count_routing_mismatches(
            ids,
            i_torch,
            sel,
            topk,
            bias=bias,
            label=f"softplus {name} E={num_experts} T={num_tokens} k={topk} {dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = n_mism / num_tokens
        ret[f"{name} max_weight_err"] = _max_weight_error(w, ids, w_torch, i_torch)
    return ret


# Mirrors DeepSeek-V4 model integration: gating fp32 + bias fp32 is the default,
# swept here against fp16/bf16 gating with mixed bias dtypes.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_softplus_correctness(num_experts, num_tokens, topk, dtype, bias_dtype):
    row = bench_topk_softplus(
        num_experts, num_tokens, topk, dtype, bias_dtype=bias_dtype
    )
    assert row["fused err"] == 0.0, (
        f"gating={dtype},bias={bias_dtype},E={num_experts},topk={topk}: "
        f"{row['fused err'] * num_tokens:.0f}/{num_tokens} tokens have non-tie ID mismatches"
    )
    assert row["fused max_weight_err"] < _WEIGHT_TOL, (
        f"gating={dtype},bias={bias_dtype},E={num_experts},topk={topk}: "
        f"max weight error {row['fused max_weight_err']:.2e} exceeds tolerance"
    )


# sqrtsoftplus token sweep (DeepSeek-V4 default): exercise every dispatch tier
# from decode to large prefill, which test_topk_softplus_correctness (T in
# {64,1024,2048}) does not fully cover:
#   T=1/64/256  -> reg opt (TPW=1) decode
#   T=1024/2048 -> smem_n / opt_n mid
#   T>=4096     -> opt_n (E=64 TPW=8) and prefill_n large-prefill paths
@pytest.mark.parametrize("num_tokens", [1, 64, 256, 1024, 4096, 8192, 16384])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_softplus_token_sweep(num_experts, num_tokens, topk):
    row = bench_topk_softplus(num_experts, num_tokens, topk, torch.bfloat16)
    assert row["fused err"] == 0.0, (
        f"E={num_experts},topk={topk},T={num_tokens}: "
        f"{row['fused err'] * num_tokens:.0f}/{num_tokens} tokens have non-tie ID mismatches"
    )
    assert row["fused max_weight_err"] < _WEIGHT_TOL, (
        f"E={num_experts},topk={topk},T={num_tokens}: "
        f"max weight error {row['fused max_weight_err']:.2e} exceeds tolerance"
    )


# ---------------------------------------------------------------------------
# topk_softmax (classic MoE softmax routing, via topk_gating + vLLM-adapted
# topk_softmax kernel as a second candidate)
# ---------------------------------------------------------------------------


@benchmark()
def bench_topk_softmax(
    num_experts,
    num_tokens,
    topk,
    dtype,
    bias_dtype=torch.float32,
    use_bias=False,
    route_scale=1.0,
):
    """Two candidates: aiter's fused topk_gating (bias-capable) and the
    vLLM-adapted topk_softmax kernel (no bias support -- always compared
    against the no-bias reference, regardless of use_bias)."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (
        (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(
            bias_dtype
        )
        if use_bias
        else torch.empty(0, device="cuda")
    )

    w_torch, i_torch = ref_softmax(gating_output, bias, topk, route_scale)
    w_torch_nobias, i_torch_nobias = ref_softmax(
        gating_output, torch.empty(0, device="cuda"), topk, route_scale
    )

    def run_fused():
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device="cuda"
        )
        topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
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

    def run_vllm():
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device="cuda"
        )
        topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
        token_expert_indices = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device="cuda"
        )
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            need_renorm=False,
        )
        if route_scale != 1.0:
            topk_weights.mul_(route_scale)
        return topk_weights, topk_ids

    candidates = {"fused": run_fused, "vllm": run_vllm}
    # vllm ignores bias entirely, so it is always graded against the no-bias
    # reference; fused is graded against the (possibly biased) reference.
    refs = {
        "fused": (
            w_torch,
            i_torch,
            bias,
            _selection_scores(gating_output, bias, "softmax"),
        ),
        "vllm": (
            w_torch_nobias,
            i_torch_nobias,
            None,
            _selection_scores(gating_output, torch.empty(0, device="cuda"), "softmax"),
        ),
    }

    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        w_ref, i_ref, ref_bias, sel = refs[name]
        (w, ids), us = run_perftest(fn)
        n_mism = _count_routing_mismatches(
            ids,
            i_ref,
            sel,
            topk,
            bias=ref_bias,
            label=f"softmax/{name} E={num_experts} T={num_tokens} k={topk} {dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = n_mism / num_tokens
        ret[f"{name} max_weight_err"] = _max_weight_error(w, ids, w_ref, i_ref)
    return ret


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 384])
def test_topk_softmax_correctness(num_experts, num_tokens, topk, dtype):
    """Pytest test for correctness of topk_gating with score_func='softmax'
    (fused candidate) and the vLLM-adapted topk_softmax kernel."""
    row = bench_topk_softmax(num_experts, num_tokens, topk, dtype, use_bias=False)
    for name in ("fused", "vllm"):
        assert row[f"{name} err"] == 0.0, (
            f"{name}: E={num_experts},topk={topk},dtype={dtype}: "
            f"{row[f'{name} err'] * num_tokens:.0f}/{num_tokens} tokens have non-tie ID mismatches"
        )
        assert row[f"{name} max_weight_err"] < _WEIGHT_TOL, (
            f"{name}: E={num_experts},topk={topk},dtype={dtype}: "
            f"max weight error {row[f'{name} max_weight_err']:.2e} exceeds tolerance"
        )


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
# multi-token prefill path (1024). Only the fused candidate is checked here --
# the vLLM kernel has no bias support.
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_tokens", [64, 1024])
@pytest.mark.parametrize("num_experts", [128, 256])
def test_topk_softmax_bias_correctness(
    num_experts, num_tokens, topk, dtype, bias_dtype
):
    row = bench_topk_softmax(
        num_experts, num_tokens, topk, dtype, bias_dtype=bias_dtype, use_bias=True
    )
    assert row["fused err"] == 0.0, (
        f"E={num_experts},topk={topk},gating={dtype},bias={bias_dtype}: "
        f"{row['fused err'] * num_tokens:.0f}/{num_tokens} tokens have non-tie ID mismatches"
    )
    # The reported weights must be the *unbiased* softmax weights; the old bug
    # made these normalize over (logit+bias) and could even exceed max softmax.
    assert row["fused max_weight_err"] < _WEIGHT_TOL, (
        f"E={num_experts},topk={topk},gating={dtype},bias={bias_dtype}: "
        f"max weight error {row['fused max_weight_err']:.2e} exceeds tolerance"
    )


# ---------------------------------------------------------------------------
# NaN/Inf robustness (topk_gating, all score functions)
# ---------------------------------------------------------------------------


def _ref_selection_with_nan(gating_output, bias, score_func):
    """fp32 reference selection score matching the kernel's non-finite handling.
    Reference only -- not timed, not in the table.

    Non-finite semantics (per score function), mirroring the kernel:
    - NaN: always excluded (never selected).
    - +Inf: sigmoid saturates to 1 (selectable); sqrt(softplus) clamps the logit
      to 1e30 (selectable, top-ranked, finite); softmax treats a dominant +Inf
      logit as the limit x->+inf: prob 1.0 for the +Inf expert(s) (split evenly
      on ties), 0 for the rest.
    - -Inf: score -> 0 (low, not selected) for every function.

    NOTE: torch.softmax does NOT compute the +Inf case this way -- its
    max-subtraction still evaluates exp(+inf - +inf) = exp(nan) = nan, so
    torch.softmax(row_with_inf) is all-NaN, not [0, ..., 1, ..., 0]. The
    softmax branch below is hand-rolled (max-subtract, exp, but treat a NaN
    diff -- which only occurs at the position(s) equal to +Inf -- as exp=1)
    to mirror the kernel's actual (intentional, PyTorch-diverging) behaviour.
    """
    gf = gating_output.float()
    nan = torch.isnan(gf)
    b = bias.float() if (bias is not None and bias.numel() > 0) else 0.0
    if score_func == "softmax":
        gf_masked = gf.masked_fill(nan, float("-inf"))  # NaN excluded from max/sum
        row_max = gf_masked.max(dim=-1, keepdim=True).values
        diff = gf_masked - row_max
        exp = torch.where(torch.isnan(diff), torch.ones_like(diff), torch.exp(diff))
        row_sum = exp.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        s = exp / row_sum
        sel = s + b
        exclude = nan
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
    # fp32 bias (not cast to `dtype`): when a +Inf logit collapses softmax to
    # an exact 0.0/1.0 split, ranking among the zero-probability experts is
    # driven entirely by bias, and bf16's ~2-3 significant digits produce
    # frequent exact collisions among ~128 random values -- creating a mass of
    # genuine ties beyond what the tie-tolerant comparison can disambiguate.
    # fp32 (or fp16) bias avoids this test artifact; it also matches the real
    # DeepSeek-V4 usage (bias kept at higher precision than gating logits).
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

    # Scatter NaN across token-dependent positions (so it lands anywhere in a
    # lane's sorted partition) plus a -Inf and +Inf per token.
    tok = torch.arange(num_tokens, device="cuda")
    for j in range(4):
        gating_output[tok, (tok * (7 * j + 3) + j) % num_experts] = float("nan")
    gating_output[tok, (tok * 11 + 2) % num_experts] = float("-inf")
    # +Inf is a valid extreme logit for all scoring functions: sigmoid saturates
    # to 1, sqrt(softplus) clamps to a finite top-ranked score, and softmax gives
    # +Inf experts prob 1.0 (the kernel's intentional limit-case handling; see
    # _ref_selection_with_nan -- this diverges from plain torch.softmax, which
    # produces NaN for a row containing +Inf).
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
    assert not row[
        "nan_leak"
    ], f"{score_func} E={num_experts} T={num_tokens} k={topk}: NaN leaked into weights"
    assert row["fused err"] == 0.0, (
        f"{score_func} E={num_experts} T={num_tokens} k={topk}: routed top-k set "
        f"differs from the NaN-excluding reference (err={row['fused err']})"
    )


# Softmax + +Inf correctness test.
#
# The kernel treats a dominant +Inf logit as the limit x->+inf, NOT via plain
# torch.softmax (which produces NaN for a row containing +Inf, since its
# max-subtraction still evaluates exp(+inf - +inf) = exp(nan) = nan):
#   +Inf expert: softmax prob = 1.0 (NaN logits are pre-excluded, then
#     +Inf - +Inf = NaN in the diff is recognized as "logit == max" -> exp = 1.0)
#   finite experts: exp(finite - +Inf) = 0 -> softmax prob = 0
# So the +Inf expert(s) should be selected and carry weight 1.0. The reference
# (_ref_selection_with_nan) hand-rolls this instead of calling torch.softmax.
# This test verifies both safety (no NaN/Inf leak) and routing correctness.
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("num_tokens", [64, 2048])
@pytest.mark.parametrize("num_experts", [64, 128, 256])
def test_topk_softmax_posinf(num_experts, num_tokens, topk):
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, torch.bfloat16)
    # fp32 bias: see the comment in bench_topk_gating_nan -- a +Inf-collapsed
    # softmax ranks the zero-probability experts by bias alone, and bf16
    # precision creates spurious exact collisions among ~128 random values.
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1
    tok = torch.arange(num_tokens, device="cuda")
    gating_output[tok, (tok * 5 + 1) % num_experts] = float("inf")

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    aiter.topk_gating(
        topk_weights,
        topk_ids,
        gating_output,
        bias,
        need_renorm=False,
        routed_scaling_factor=1.0,
        score_func="softmax",
    )
    assert not topk_weights.isnan().any(), "softmax + +Inf: NaN leaked into weights"
    assert not topk_weights.isinf().any(), "softmax + +Inf: Inf leaked into weights"
    assert (topk_ids >= 0).all() and (
        topk_ids < num_experts
    ).all(), "softmax + +Inf: expert ID out of range"

    # Routing correctness: selection should match the NaN-excluding reference.
    sel = _ref_selection_with_nan(gating_output, bias, "softmax")
    i_ref = sel.topk(topk, dim=-1, sorted=False)[1].to(torch.int32)
    n_mism = _count_routing_mismatches(
        topk_ids,
        i_ref,
        sel,
        topk,
        bias=bias,
        label=f"softmax+posinf E={num_experts} T={num_tokens} k={topk}",
    )
    assert n_mism == 0, (
        f"softmax+posinf E={num_experts} T={num_tokens} k={topk}: "
        f"{n_mism}/{num_tokens} tokens have non-tie routing mismatches"
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
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
        help="Comma-separated list of number of tokens (default: 16384,4096,1024,256,64,1)",
    )
    parser.add_argument(
        "--topk",
        type=str2tuple,
        default=[1, 2, 4, 6, 8],
        help="Comma-separated list of topk values (default: 1,2,4,6,8)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str2Dtype,
        nargs="*",
        default=[torch.float16, torch.bfloat16, torch.float32],
        help="Comma-separated list of dtypes: fp16, bf16, fp32 (default: fp16,bf16,fp32)",
    )
    args = parser.parse_args()

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    num_experts_list = to_list(args.num_experts)
    num_tokens_list = to_list(args.num_tokens)
    topk_list = to_list(args.topk)
    dtype_list = to_list(args.dtype)

    # Track whether any benchmark section saw a correctness regression;
    # exit non-zero at the end so CI catches it.
    failed_sections: list[str] = []

    # -- topk_sigmoid --------------------------------------------------
    sigmoid_dtypes = [d for d in dtype_list if d != torch.float32]
    sigmoid_configs = list(
        itertools.product(num_experts_list, num_tokens_list, topk_list, sigmoid_dtypes)
    )
    df = [bench_topk_sigmoid(*cfg) for cfg in sigmoid_configs]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "topk_sigmoid summary (markdown):\n%s", df.to_markdown(index=False)
    )
    errors = df[(df["fused score_err"] > 0.01) | (df["fused idx_err"] > 0.01)]
    if len(errors) > 0:
        print(f"\nERROR: {len(errors)} sigmoid config(s) had errors > 1%!")
        print(errors.to_string(index=False))
        failed_sections.append("sigmoid")

    # -- topk_softplus ---------------------------------------------------
    softplus_configs = list(
        itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
    )
    df = [bench_topk_softplus(*cfg) for cfg in softplus_configs]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "topk_softplus summary (markdown):\n%s", df.to_markdown(index=False)
    )
    errors = df[(df["fused err"] > 0.01) | (df["fused max_weight_err"] > _WEIGHT_TOL)]
    if len(errors) > 0:
        print(f"\nERROR: {len(errors)} softplus config(s) had errors!")
        print(errors.to_string(index=False))
        failed_sections.append("softplus")

    # -- topk_softmax: topk_gating (fused) vs topk_softmax (vLLM) --------
    softmax_configs = list(
        itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
    )
    df = [bench_topk_softmax(*cfg) for cfg in softmax_configs]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "topk_softmax summary (markdown):\n%s", df.to_markdown(index=False)
    )
    errors = df[
        (df["fused err"] > 0.01)
        | (df["vllm err"] > 0.01)
        | (df["fused max_weight_err"] > _WEIGHT_TOL)
        | (df["vllm max_weight_err"] > _WEIGHT_TOL)
    ]
    if len(errors) > 0:
        print(f"\nERROR: {len(errors)} softmax config(s) had errors!")
        print(errors.to_string(index=False))
        failed_sections.append("softmax")

    # -- topk_gating NaN/Inf robustness -----------------------------------
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
    df = [bench_topk_gating_nan(*cfg) for cfg in nan_configs]
    df = pd.DataFrame(df)
    aiter.logger.info(
        "topk_gating NaN/Inf robustness summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    errors = df[(df["fused err"] > 0) | (df["nan_leak"])]
    if len(errors) > 0:
        print(f"\nERROR: {len(errors)} nan config(s) failed (err>0 or nan_leak)!")
        print(errors.to_string(index=False))
        failed_sections.append("nan")

    if failed_sections:
        print(
            f"FAIL: correctness regression in section(s): {', '.join(failed_sections)}",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("All topk_gating benchmarks passed!")


if __name__ == "__main__":
    main()
