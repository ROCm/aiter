# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Test topk_gating (topk_sigmoid / topk_softplus / topk_softmax) operations with
various configurations.

Usage:
  python test_moe_topk_gating.py --num-experts 64,128 --topk 2,4,8 --dtype fp16
  python test_moe_topk_gating.py --section softmax --topk 8
  python test_moe_topk_gating.py --section sigmoid,softmax --num-tokens 64,1024
  python test_moe_topk_gating.py --section ties,eplb
"""

import argparse
import itertools
import math
import os
import sys

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility.dtypes import str2Dtype, str2tuple

torch.set_default_device("cuda")

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

_TIE_TOL = 1e-4

_WEIGHT_TOL = 1e-4


def _renorm_err(topk_weights):
    """Max |sum(weights) - 1| per row, with non-finite mapped to a failing value.

    A NaN weight makes the raw error NaN, and `NaN > _WEIGHT_TOL` is false, so
    reporting it unmapped would let invalid weights pass as success.
    """
    err = float((topk_weights.sum(-1) - 1.0).abs().max())
    return err if math.isfinite(err) else float("inf")


# EP world size assumed by the eplb section; experts are sharded contiguously,
# so rank r owns [r * num_experts / _EPLB_EP_SIZE, (r + 1) * ...).
_EPLB_EP_SIZE = 4

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def _selection_scores(
    gating_output: torch.Tensor, bias: torch.Tensor, score_func: str
) -> torch.Tensor:
    """Reference biased selection scores [num_tokens, num_experts] in fp32."""
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
    a way NOT explained by a near-tie at the top-K selection boundary."""
    T, E = sel_scores.shape
    dev = sel_scores.device
    sel = sel_scores.to(torch.float32)
    i_fused = i_fused.long()
    i_torch = i_torch.long()

    cutoff = sel.topk(topk, dim=-1).values.amin(dim=-1, keepdim=True)

    fused_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    fused_mask.scatter_(1, i_fused, True)
    ref_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    ref_mask.scatter_(1, i_torch, True)

    fused_full = fused_mask.sum(dim=1) == topk
    ref_full = ref_mask.sum(dim=1) == topk
    match = (fused_mask == ref_mask).all(dim=1) & fused_full

    extra = fused_mask & ~ref_mask
    missing = ref_mask & ~fused_mask
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

            def _fmt(e, t=t, thr=thr):
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
    back in the fused id order."""
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
    selected SETS are identical."""
    T = w_fused.shape[0]
    dev = w_fused.device
    E = int(max(int(i_fused.max()), int(i_torch.max())) + 1)
    fused_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    fused_mask.scatter_(1, i_fused.long(), True)
    torch_mask = torch.zeros((T, E), dtype=torch.bool, device=dev)
    torch_mask.scatter_(1, i_torch.long(), True)
    same_set = (fused_mask == torch_mask).all(dim=1)

    ref, matched = _torch_weight_aligned_to_fused(w_fused, i_fused, w_torch, i_torch)
    use = matched & same_set.unsqueeze(1)
    if not bool(use.any()):
        return 0.0
    diff = (w_fused.to(torch.float32) - ref).abs()
    return float(diff[use].max())


# ---------------------------------------------------------------------------
# torch references (fp32, untimed -- never enter the perf table)
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
    renormalize: bool = False,
):
    scores = torch.softmax(gating_output.float(), dim=-1)
    scores_biased = scores + bias.float() if bias.numel() > 0 else scores
    topk_ids = scores_biased.topk(topk, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * route_scale
    return topk_weights, topk_ids.to(torch.int32)


def _ref_selection_with_nan(gating_output, bias, score_func):
    """fp32 reference selection score matching the kernel's non-finite handling."""
    gf = gating_output.float()
    nan = torch.isnan(gf)
    b = bias.float() if (bias is not None and bias.numel() > 0) else 0.0
    if score_func == "softmax":
        gf_masked = gf.masked_fill(nan, float("-inf"))
        row_max = gf_masked.max(dim=-1, keepdim=True).values
        diff = gf_masked - row_max
        exp = torch.where(torch.isnan(diff), torch.ones_like(diff), torch.exp(diff))
        row_sum = exp.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        s = exp / row_sum
        sel = s + b
        exclude = nan
    elif score_func == "sigmoid":
        sel = torch.sigmoid(gf) + b
        exclude = nan
    else:  # sqrtsoftplus
        sel = torch.sqrt(torch.nn.functional.softplus(torch.clamp(gf, max=1.0e30))) + b
        exclude = nan
    return sel.masked_fill(exclude, float("-inf"))


def _starved_rows_ok(num_experts, num_tokens, topk, score_func, dtype):
    """Does a row the kernel cannot fill degrade the documented way?

    Such a row cannot fill every slot, so the kernel elects a sentinel for the
    remainder, and what that sentinel is worth decides how far the damage goes.
    Reaching the renorm sum, it drops the sum to RENORM_SUM_FLOOR and rescales
    the row's *valid* weights by ~1e20, so one starved row corrupts the experts
    it did resolve. Keeping a weight of its own is worse: the slot becomes a
    real routing decision for a token that has none to make.

    Three poisoned rows. Rows 0 and 2 have no valid expert at all and must come
    out all-zero. They reach that state by different routes -- a NaN is scrubbed
    out of the selection score, a -Inf is already the bottom of it -- and since
    masking an expert out with -Inf is a normal thing for a caller to do, both
    are real inputs rather than one being a theoretical variant of the other.
    Row 1 is short by exactly one slot and must weight precisely its valid
    experts: an id set, so a sentinel that either takes a slot or duplicates a
    real expert fails. Only weights are checked on the dead rows, because the
    sentinel ids there are allowed to repeat -- a zero weight makes them inert.
    Reuses the sweep's token count to keep the same dispatch path as the
    measured case.
    """
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    gating_output[0, :] = float("nan")  # no valid expert at all
    dead_rows = [0]
    starved_row = 1 if (num_tokens > 1 and topk > 1) else None
    valid_experts = []
    if starved_row is not None:
        # Short by one slot. Distinct logits, so no tie-break is involved.
        valid_experts = [(5 + i) % num_experts for i in range(topk - 1)]
        gating_output[starved_row, :] = float("nan")
        for i, e in enumerate(valid_experts):
            gating_output[starved_row, e] = 1.0 + 0.5 * i
    if num_tokens > 2:
        gating_output[2, :] = float("-inf")  # masked out rather than invalid
        dead_rows.append(2)

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")
    aiter.topk_gating(
        topk_weights,
        topk_ids,
        gating_output,
        None,
        need_renorm=score_func != "softmax",
        routed_scaling_factor=2.5,
        score_func=score_func,
    )

    if not topk_weights.isfinite().all().item():
        return False
    if ((topk_ids < 0) | (topk_ids >= num_experts)).any().item():
        return False
    for row in dead_rows:
        if (topk_weights[row] != 0.0).any().item():
            return False
    if starved_row is not None:
        held = topk_ids[starved_row][topk_weights[starved_row] != 0.0]
        if sorted(held.tolist()) != sorted(valid_experts):
            return False
    return True


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
    """Single fused candidate. Default bias_dtype=fp32 matches DeepSeek-V4."""
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
    renormalize=False,
    route_scale=1.0,
):
    """Two candidates: aiter's fused topk_gating (bias-capable) and the
    vLLM-adapted topk_softmax kernel (no bias support)."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = (
        (torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1).to(
            bias_dtype
        )
        if use_bias
        else torch.empty(0, device="cuda")
    )

    w_torch, i_torch = ref_softmax(gating_output, bias, topk, route_scale, renormalize)
    w_torch_nobias, i_torch_nobias = ref_softmax(
        gating_output, torch.empty(0, device="cuda"), topk, route_scale, renormalize
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
            need_renorm=renormalize,
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
            False,
        )
        if renormalize:
            topk_weights.div_(topk_weights.sum(dim=-1, keepdim=True))
        if route_scale != 1.0:
            topk_weights.mul_(route_scale)
        return topk_weights, topk_ids

    candidates = {"fused": run_fused, "vllm": run_vllm}
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


# ---------------------------------------------------------------------------
# NaN/Inf robustness (topk_gating, all score functions)
# ---------------------------------------------------------------------------


@benchmark()
def bench_topk_gating_nan(num_experts, num_tokens, topk, score_func, dtype):
    """NaN/Inf robustness benchmark. Injects NaN, +Inf and -Inf experts
    scattered per token and checks the routed top-k SET against a reference."""
    torch.random.manual_seed(0)
    gating_output = _make_gating(num_experts, num_tokens, dtype)
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda") * 0.1

    tok = torch.arange(num_tokens, device="cuda")
    for j in range(4):
        gating_output[tok, (tok * (7 * j + 3) + j) % num_experts] = float("nan")
    gating_output[tok, (tok * 11 + 2) % num_experts] = float("-inf")
    gating_output[tok, (tok * 5 + 1) % num_experts] = float("inf")

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
    inf_leak = bool(topk_weights.isinf().any().item())
    starved_ok = _starved_rows_ok(num_experts, num_tokens, topk, score_func, dtype)

    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    ret["fused us"] = us
    ret["fused TB/s"] = nbytes / us / 1e6
    ret["fused err"] = n_mism / num_tokens
    ret["nan_leak"] = nan_leak
    ret["inf_leak"] = inf_leak
    ret["starved ok"] = starved_ok
    return ret


@benchmark()
def bench_topk_gating_ties(num_experts, num_tokens, topk, score_func, dtype):
    """Exact-tie benchmark. The other cases build rows of distinct values, but
    real gating output repeats, and lanes that disagree about which of two equal
    experts won consume two selection slots while recording one.

    Scored on dropped slots rather than set equality: any expert whose score
    equals the reference's k-th pick is a valid choice, so the failure to catch
    is picking one strictly below it.
    """
    torch.random.manual_seed(0)
    # Quantise hard so each row holds many exactly-equal values.
    gating_output = (torch.randn(num_tokens, num_experts, device="cuda") * 2).round()
    gating_output = gating_output.to(dtype)

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    _, us = run_perftest(
        aiter.topk_gating,
        topk_weights,
        topk_ids,
        gating_output,
        None,
        need_renorm=True,
        score_func=score_func,
    )

    g = gating_output.float()
    if score_func == "softmax":
        sel = torch.softmax(g, dim=-1)
    elif score_func == "sigmoid":
        sel = torch.sigmoid(g)
    else:
        sel = torch.nn.functional.softplus(g).sqrt()

    kth = sel.topk(topk, dim=-1).values.amin(dim=-1, keepdim=True)
    dropped = int(((kth - sel.gather(1, topk_ids.long())) > _TIE_TOL).any(dim=1).sum())
    ids_sorted = topk_ids.sort(dim=-1).values
    dup = int((ids_sorted.diff(dim=-1) == 0).any(dim=1).sum())

    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    ret["fused us"] = us
    ret["fused TB/s"] = nbytes / us / 1e6
    ret["fused err"] = dropped / num_tokens
    ret["dup ids"] = dup
    ret["renorm err"] = _renorm_err(topk_weights)
    return ret


# ---------------------------------------------------------------------------
# EP load balance on fake-eplb balance-router logits
# ---------------------------------------------------------------------------


def _eplb_gating(num_experts, num_tokens, topk, ep_size, dtype, logit=10.0):
    """Synthetic balance-router logits, as fake-eplb substitutes them.

    A flat ring walks position p = t * topk + j across EP ranks, so consecutive
    picks step the rank by one and rank load is even over the batch. The chosen
    experts sit at +logit and everything else at -logit, which makes the top-K
    set unique while every pair inside it -- and every pair outside -- ties
    exactly. That is the densest tie pattern a router can hand the kernel.
    """
    experts_per_rank = num_experts // ep_size
    p = torch.arange(num_tokens, device="cuda").unsqueeze(1) * topk + torch.arange(
        topk, device="cuda"
    ).unsqueeze(0)
    expert_ids = (p % ep_size) * experts_per_rank + (p // ep_size) % experts_per_rank
    gating_output = torch.full(
        (num_tokens, num_experts), -logit, dtype=dtype, device="cuda"
    )
    gating_output.scatter_(1, expert_ids, logit)
    return gating_output, expert_ids.to(torch.int32)


@benchmark()
def bench_topk_gating_eplb(
    num_experts, num_tokens, topk, score_func, dtype, ep_size=_EPLB_EP_SIZE
):
    """EP load-balance benchmark on tie-dense balance-router logits.

    The ties section perturbs otherwise-distinct rows; here every selected
    expert ties with every other one, which is what turns a slot drop into a
    routing collapse: retiring several tied experts while recording one frees
    slots that refill from the rejected pool, and those sit low in the index
    space, so the batch piles onto the first EP rank instead of spreading.

    Scored on exact set equality -- unlike the ties section, the reference set
    here is unambiguous, since every selected expert outscores every rejected
    one -- plus the resulting EP rank load.
    """
    torch.random.manual_seed(0)
    gating_output, i_ref = _eplb_gating(num_experts, num_tokens, topk, ep_size, dtype)

    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device="cuda")
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device="cuda")

    _, us = run_perftest(
        aiter.topk_gating,
        topk_weights,
        topk_ids,
        gating_output,
        None,
        need_renorm=True,
        score_func=score_func,
    )

    ids_sorted = topk_ids.sort(dim=-1).values
    mism = int((ids_sorted != i_ref.sort(dim=-1).values).any(dim=1).sum())
    dup = int((ids_sorted.diff(dim=-1) == 0).any(dim=1).sum())

    experts_per_rank = num_experts // ep_size

    def max_rank_share(ids):
        ranks = ids.reshape(-1).long() // experts_per_rank
        counts = torch.bincount(ranks, minlength=ep_size)
        return float(counts.max()) / ids.numel()

    nbytes = (
        num_tokens * num_experts * gating_output.element_size()
        + num_tokens * topk * (4 + 4)
    )
    ret = {"gfx": get_gfx()}
    ret["fused us"] = us
    ret["fused TB/s"] = nbytes / us / 1e6
    ret["fused err"] = mism / num_tokens
    ret["dup ids"] = dup
    ret["ep max share"] = max_rank_share(topk_ids)
    ret["ref ep max share"] = max_rank_share(i_ref)
    ret["renorm err"] = _renorm_err(topk_weights)
    return ret


# ---------------------------------------------------------------------------
# main() -- argparse + sweep, one table per score function
# ---------------------------------------------------------------------------


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("topk_gating unsupported on %s; skipping", get_gfx())
        return

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
    parser.add_argument(
        "--section",
        "--score-func",
        dest="section",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["sigmoid", "softplus", "softmax", "nan", "ties", "eplb"],
        help="Comma-separated list of sections to run: "
        "sigmoid,softplus,softmax,nan,ties,eplb (default: all). "
        "The first three are named after a score function; nan, ties and eplb "
        "each sweep all score functions. --score-func is a deprecated alias.",
    )
    args = parser.parse_args()

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    num_experts_list = to_list(args.num_experts)
    num_tokens_list = to_list(args.num_tokens)
    topk_list = to_list(args.topk)
    dtype_list = to_list(args.dtype)
    sections = args.section

    failed_sections: list[str] = []

    # -- topk_sigmoid --------------------------------------------------
    if "sigmoid" in sections:
        sigmoid_dtypes = [d for d in dtype_list if d != torch.float32]
        sigmoid_configs = list(
            itertools.product(
                num_experts_list, num_tokens_list, topk_list, sigmoid_dtypes
            )
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
    if "softplus" in sections:
        softplus_configs = list(
            itertools.product(num_experts_list, num_tokens_list, topk_list, dtype_list)
        )
        df = [bench_topk_softplus(*cfg) for cfg in softplus_configs]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "topk_softplus summary (markdown):\n%s", df.to_markdown(index=False)
        )
        errors = df[
            (df["fused err"] > 0.01) | (df["fused max_weight_err"] > _WEIGHT_TOL)
        ]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} softplus config(s) had errors!")
            print(errors.to_string(index=False))
            failed_sections.append("softplus")

    # -- topk_softmax: topk_gating (fused) vs topk_softmax (vLLM) --------
    if "softmax" in sections:
        softmax_configs = list(
            itertools.product(
                num_experts_list, num_tokens_list, topk_list, dtype_list, [False, True]
            )
        )
        df = [
            bench_topk_softmax(E, T, k, dt, renormalize=rn)
            for E, T, k, dt, rn in softmax_configs
        ]
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
    if "nan" in sections:
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
        errors = df[
            (df["fused err"] > 0)
            | (df["nan_leak"])
            | (df["inf_leak"])
            | (~df["starved ok"])
        ]
        if len(errors) > 0:
            print(
                f"\nERROR: {len(errors)} nan config(s) failed "
                f"(err>0, nan/inf leak, or a mis-weighted starved row)!"
            )
            print(errors.to_string(index=False))
            failed_sections.append("nan")

    # -- topk_gating exact-tie handling ------------------------------------
    if "ties" in sections:
        tie_configs = list(
            itertools.product(
                num_experts_list,
                num_tokens_list,
                topk_list,
                ["sqrtsoftplus", "sigmoid", "softmax"],
                dtype_list,
            )
        )
        df = [bench_topk_gating_ties(*cfg) for cfg in tie_configs]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "topk_gating exact-tie summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        errors = df[
            (df["fused err"] > 0)
            | (df["dup ids"] > 0)
            | (df["renorm err"] > _WEIGHT_TOL)
        ]
        if len(errors) > 0:
            print(f"\nERROR: {len(errors)} tie config(s) failed!")
            print(errors.to_string(index=False))
            failed_sections.append("ties")

    # -- topk_gating EP load balance (fake-eplb balance logits) -------------
    if "eplb" in sections:
        eplb_configs = [
            cfg
            for cfg in itertools.product(
                num_experts_list,
                num_tokens_list,
                topk_list,
                ["sqrtsoftplus", "sigmoid", "softmax"],
                dtype_list,
            )
            if cfg[0] % _EPLB_EP_SIZE == 0
        ]
        # Experts are sharded across ranks, so only counts divisible by the EP
        # size can run. If the caller picked none that qualify (e.g. --section
        # eplb --num-experts 2) there is nothing to measure, and an empty frame
        # has none of the columns the checks below index.
        if not eplb_configs:
            print(
                f"SKIP: eplb needs num_experts divisible by {_EPLB_EP_SIZE}, "
                f"none of {num_experts_list} qualify"
            )
        else:
            df = [bench_topk_gating_eplb(*cfg) for cfg in eplb_configs]
            df = pd.DataFrame(df)
            aiter.logger.info(
                "topk_gating fake-eplb EP balance summary (markdown):\n%s",
                df.to_markdown(index=False),
            )
            errors = df[
                (df["fused err"] > 0)
                | (df["dup ids"] > 0)
                | (df["ep max share"] > df["ref ep max share"] + 0.05)
                | (df["renorm err"] > _WEIGHT_TOL)
            ]
            if len(errors) > 0:
                print(f"\nERROR: {len(errors)} eplb config(s) failed!")
                print(errors.to_string(index=False))
                failed_sections.append("eplb")

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
