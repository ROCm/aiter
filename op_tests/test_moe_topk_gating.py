# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Test topk_gating (topk_sigmoid / topk_softplus / topk_softmax) operations with
various configurations.

Usage:
  python test_moe_topk_gating.py --num-experts 64,128 --topk 2,4,8 --dtype fp16
  python test_moe_topk_gating.py --score-func softmax --topk 8
  python test_moe_topk_gating.py --score-func sigmoid,softmax --num-tokens 64,1024
"""

import argparse
import itertools
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

SUPPORTED_GFX = ["gfx942", "gfx950"]


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
        "--score-func",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["sigmoid", "softplus", "softmax", "nan"],
        help="Comma-separated list of sections to run: sigmoid,softplus,softmax,nan (default: all)",
    )
    args = parser.parse_args()

    def to_list(x):
        return x if isinstance(x, (list, tuple)) else [x]

    num_experts_list = to_list(args.num_experts)
    num_tokens_list = to_list(args.num_tokens)
    topk_list = to_list(args.topk)
    dtype_list = to_list(args.dtype)
    score_funcs = args.score_func

    failed_sections: list[str] = []

    # -- topk_sigmoid --------------------------------------------------
    if "sigmoid" in score_funcs:
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
    if "softplus" in score_funcs:
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
    if "softmax" in score_funcs:
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
    if "nan" in score_funcs:
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
        errors = df[(df["fused err"] > 0) | (df["nan_leak"]) | (df["inf_leak"])]
        if len(errors) > 0:
            print(
                f"\nERROR: {len(errors)} nan config(s) failed (err>0 or nan/inf leak)!"
            )
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
