# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance of ``flydsl_router_sqrtsoftplus_topk`` (gfx950).

The DeepSeek-V4.1-Flash MoE router for decode rows, E = 384 experts with
k = 6 (MoE layers) and E = 128 with k = 3 (DSpark draft layers):

    logits = x [M, 5120] @ gate [E, 5120]^T        (bf16, fp32 accumulate)
    score  = sqrt(softplus(logits))                (logits above 20 pass through)
    ids    = top-k of score + bias                 (bias fp32 [E])
    w      = score[ids] * 1.5 / sum(score[ids])

Every table runs for each expert count (``-e``).

Four tables:

1. ``test_router_sqrtsoftplus_topk`` (perf + correctness, ``-m``): the FlyDSL op
   with the row-major gate, with the load-time shuffled gate
   (``shuffle_router_gate``), and as the model calls it (shuffled gate, an
   ``is_padding`` mask, caller-allocated outputs filled with NaN / -7), plus
   AITER's ``topk_gating`` on a ``torch.mm`` (hipBLASLt) fp32 GEMM, each against
   an fp32 torch reference (rows whose k-th and (k+1)-th candidates tie within
   fp32 noise excluded).
2. ``test_router_logits`` (``-r`` x ``-l``): at every row bucket and beyond, the
   op's own logits (its fp32 partials summed in slice order) against an fp32
   torch GEMM, ids and weights against the reference top-k of those logits,
   and bitwise repeatability.
3. ``test_router_vllm`` (``-v``, when vLLM is importable): ids and weights bit
   for bit against vLLM's ``topk_hash_softplus_sqrt`` on the op's logits, and
   against vLLM's whole router (its GEMM + top-k) off near-ties.
4. Edge cases: exact ties, logits above the softplus switch with padding rows,
   row strides (incl. declined zero / overlapping strides), an offset
   ``is_padding`` view, NaN / inf inputs, the gate shuffle being a
   permutation, CUDA-graph replay into caller buffers poisoned before every
   replay after ``router_sqrtsoftplus_topk_warmup``, and the eligibility helper
   / ``ValueError`` on unsupported inputs.

Run:
    python3 op_tests/test_flydsl_router_sqrtsoftplus_topk.py
    python3 op_tests/test_flydsl_router_sqrtsoftplus_topk.py -m 16 192 -r 7 768
    python3 op_tests/test_flydsl_router_sqrtsoftplus_topk.py -e 128
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
SCALE = 1.5
HIDDEN = 5120
# Routed experts -> top-k.
SHAPES = {384: 6, 128: 3}
# k-th / (k+1)-th ``score + bias`` closer than this (relative) may pick either.
MARGIN = 1e-5
LAYOUTS = ("row_major", "shuffled")


def score_ref(logits):
    return torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits))).sqrt()


def run_torch_router(logits, bias, is_padding=None):
    """fp32 reference: weights, ids, and the rows without a k-th / (k+1)-th
    tie."""
    k = SHAPES[logits.shape[1]]
    score = score_ref(logits)
    top = torch.topk(score + bias, k + 1, dim=-1)
    ids = top.indices[:, :k]
    w = score.gather(1, ids)
    w = w * (SCALE / w.sum(-1, keepdim=True))
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    gap = top.values[:, k - 1] - top.values[:, k]
    return w, ids, gap > MARGIN * top.values[:, k - 1].abs().clamp_min(1.0)


def topk_ref(logits, bias, is_padding=None):
    """The op's top-k decoding of given logits: weight ``(score + bias) - bias``
    summed in slot order (vLLM's ``topk_softplus_sqrt``)."""
    score = score_ref(logits.float())
    ids = torch.topk(score + bias, SHAPES[logits.shape[1]], dim=-1)[1]
    w = (score + bias).gather(1, ids) - bias[ids]
    total = w[:, 0]
    for k in range(1, ids.shape[1]):
        total = total + w[:, k]
    w = w * (SCALE / total[:, None])
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    return w, ids


def router_inputs(m, seed, wscale=0.02, experts=384):
    """x rows at per-row scales 2^-2..2^2, so a row read with another row's
    data or scale shows up."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, HIDDEN), generator=g) * torch.exp2(
        torch.randint(-2, 3, (m, 1), generator=g).float()
    )
    w = torch.randn((experts, HIDDEN), generator=g) * wscale
    bias = torch.randn(experts, generator=g) * 0.1
    return x.to(dtypes.bf16), w.to(dtypes.bf16), bias


def route(x, w, bias, **kw):
    """Run the op with a caller scratch; also return its own logits."""
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_workspace_shape,
    )

    shape = router_sqrtsoftplus_topk_workspace_shape(
        x.shape[0], kw.get("weight_shuffled", False), w.shape[0]
    )
    partials = torch.empty(shape, dtype=dtypes.fp32)
    out = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, partials=partials, **kw)
    torch.cuda.synchronize()
    logits = partials[0].clone()
    for p in partials[1:]:
        logits = logits + p
    return out, logits


def close_rows(logits, bias):
    """Rows whose k-th and (k+1)-th ``score + bias`` tie within MARGIN."""
    k = SHAPES[logits.shape[1]]
    top = torch.topk(score_ref(logits) + bias, k + 1, dim=-1)[0]
    gap = (top[:, k - 1] - top[:, k]).abs()
    return gap < MARGIN * top[:, k - 1].abs().clamp_min(1.0)


def check_route(out, logits, bias, msg, is_padding=None):
    """ids exact and weights within 2e-6 relative of ``topk_ref(logits)`` on
    rows without a near-tie; live rows' weights sum to SCALE. Returns the
    largest error fraction."""
    tw, ti = out
    rw, ri = topk_ref(logits, bias, is_padding)
    ok = ~close_rows(logits, bias)[:, None]
    err_ids = checkAllclose(
        ri.float(), ti.float(), rtol=0, atol=0, tol_err_ratio=0, mask=ok,
        msg=f"{msg}: ids",
    )  # fmt: skip
    err_w = checkAllclose(
        rw, tw, rtol=2e-6, atol=2e-9, tol_err_ratio=0, mask=ok,
        msg=f"{msg}: weights",
    )  # fmt: skip
    live = ok[:, 0] if is_padding is None else ok[:, 0] & ~is_padding
    total = tw[live].sum(-1)
    err_sum = checkAllclose(
        torch.full_like(total, SCALE), total, rtol=1e-5, atol=0, tol_err_ratio=0,
        msg=f"{msg}: weight sums",
    )  # fmt: skip
    err = max(err_ids, err_w, err_sum)
    assert err == 0, f"{msg}: mismatch"
    return err


@benchmark()
def test_router_sqrtsoftplus_topk(m, experts):
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk, shuffle_router_gate
    from aiter.ops.topk import topk_gating

    topk = SHAPES[experts]
    x, w, bias = router_inputs(m, m, experts=experts)
    ws = shuffle_router_gate(w)
    logits = torch.mm(x.float(), w.float().t())
    ref = run_torch_router(logits, bias)
    # The model passes a padding mask and its own output buffers.
    pad = torch.zeros(m, dtype=torch.bool)
    pad[1::5] = True
    ref_pad = run_torch_router(logits, bias, pad)
    out_w = torch.full((m, topk), float("nan"))
    out_ids = torch.full((m, topk), -7, dtype=dtypes.i32)

    def mm_topk_gating():
        tw = torch.empty(m, topk)
        ti = torch.empty(m, topk, dtype=dtypes.i32)
        lg = torch.mm(x, w.t(), out_dtype=dtypes.fp32)
        topk_gating(tw, ti, lg, bias, True, SCALE, "sqrtsoftplus")
        return tw, ti

    candidates = {
        "flydsl": (lambda: flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE), ref),
        "flydsl_shuffled": (
            lambda: flydsl_router_sqrtsoftplus_topk(
                x, ws, bias, SCALE, weight_shuffled=True
            ),
            ref,
        ),
        "flydsl_shuffled_model": (
            lambda: flydsl_router_sqrtsoftplus_topk(
                x,
                ws,
                bias,
                SCALE,
                weight_shuffled=True,
                is_padding=pad,
                topk_weights=out_w,
                topk_ids=out_ids,
            ),
            ref_pad,
        ),
        "mm+topk_gating": (mm_topk_gating, ref),
    }
    flops = 2 * m * HIDDEN * experts
    nbytes = (m + experts) * HIDDEN * 2 + m * topk * 8
    ret = {"gfx": get_gfx_runtime()}
    for name, (fn, (ref_w, ref_ids, clear)) in candidates.items():
        (tw, ti), us = run_perftest(fn)
        if name == "flydsl_shuffled_model":
            assert tw is out_w and ti is out_ids, f"{name}: caller buffers"
        rows = clear[:, None]
        err_ids = checkAllclose(
            ref_ids.float(), ti.float(), rtol=0, atol=0, tol_err_ratio=0,
            mask=rows, msg=f"{name}: ids",
        )  # fmt: skip
        err_w = checkAllclose(
            ref_w, tw, rtol=1e-4, atol=1e-6, tol_err_ratio=0, mask=rows,
            msg=f"{name}: weights",
        )  # fmt: skip
        assert max(err_ids, err_w) == 0, f"{name}: mismatch"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err_ids, err_w)
    return ret


@benchmark()
def test_router_logits(m, layout, experts):
    """Logits, ids / weights and repeatability at one row count and layout."""
    from aiter.ops.flydsl import shuffle_router_gate

    shuffled = layout == "shuffled"
    x, w, bias = router_inputs(m, m + 11 * shuffled, experts=experts)
    wk = shuffle_router_gate(w) if shuffled else w
    out, logits = route(x, wk, bias, weight_shuffled=shuffled)
    ref = torch.mm(x.float(), w.float().t())
    # fp32 accumulation over k = 5120 in another order.
    logits_err = ((logits - ref).abs().max() / ref.abs().max()).item()
    assert logits_err <= 1e-5, f"{m} {layout}: logits {logits_err:.2e}"
    err = check_route(out, logits, bias, f"{m} {layout}")
    again, _ = route(x, wk, bias, weight_shuffled=shuffled)
    repeat = torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])
    assert repeat, f"{m} {layout}: not bitwise repeatable"
    return {
        "logits rel err": logits_err,
        "near-tie rows": int(close_rows(logits, bias).sum().item()),
        "err": err,
        "repeatable": repeat,
    }


@benchmark()
def test_router_vllm(m, experts):
    """Bitwise against vLLM's top-k on the op's logits (normal and above the
    softplus switch), and vLLM's GEMM + top-k off near-ties."""
    import importlib

    from vllm import _custom_ops as ops

    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    topk = SHAPES[experts]
    ret = {}
    for seed, wscale in ((m, 0.02), (m + 1, 0.05), (m + 2, 0.2)):
        x, w, bias = router_inputs(m, seed, wscale, experts)
        out, logits = route(x, w, bias)
        bw = torch.empty((m, topk), dtype=dtypes.fp32)
        bi = torch.empty((m, topk), dtype=dtypes.i32)
        ops.topk_hash_softplus_sqrt(
            bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, *[None] * 4, 0
        )
        assert torch.equal(out[1], bi), f"{m} wscale {wscale}: ids vs vLLM"
        assert torch.equal(out[0], bw), f"{m} wscale {wscale}: weights vs vLLM"
        ret[f"bitwise wscale {wscale}"] = True
    # Registers torch.ops.vllm.fp32_router_gemm_dispatch.
    importlib.import_module("vllm.model_executor.layers.fused_moe.router.gate_linear")
    x, w, bias = router_inputs(m, m + 3, experts=experts)
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    logits = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
    bw = torch.empty((m, topk), dtype=dtypes.fp32)
    bi = torch.empty((m, topk), dtype=dtypes.i32)
    ops.topk_hash_softplus_sqrt(
        bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, *[None] * 4, 0
    )
    ok = ~close_rows(logits, bias)[:, None]
    err_ids = checkAllclose(
        bi.float(), ti.float(), rtol=0, atol=0, tol_err_ratio=0, mask=ok,
        msg=f"{m}: ids vs vLLM router",
    )  # fmt: skip
    err_w = checkAllclose(
        bw, tw, rtol=2e-6, atol=0, tol_err_ratio=0, mask=ok,
        msg=f"{m}: weights vs vLLM router",
    )  # fmt: skip
    assert max(err_ids, err_w) == 0, f"{m}: mismatch vs vLLM router"
    ret["vLLM router err"] = max(err_ids, err_w)
    return ret


def case_shuffle_is_permutation():
    from aiter.ops.flydsl import shuffle_router_gate

    for experts in SHAPES:
        _, w, _ = router_inputs(1, 0, experts=experts)
        back = shuffle_router_gate(w).reshape(experts // 16, HIDDEN // 32, 4, 16, 8)
        back = back.permute(0, 3, 1, 2, 4).reshape(experts, HIDDEN)
        assert torch.equal(back, w), f"{experts} experts"


def case_ties_lowest_id():
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    for experts, topk in SHAPES.items():
        x = torch.zeros((16, HIDDEN), dtype=dtypes.bf16)
        w = torch.zeros((experts, HIDDEN), dtype=dtypes.bf16)
        bias = torch.zeros(experts)
        tied = (experts - 1, 5, 100, 7, experts // 2 + 8, 9, experts - 20, 11)
        for i in tied:
            bias[i] = 1.0
        tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
        want = torch.tensor(sorted(tied)[:topk], dtype=dtypes.i32)
        assert (ti == want).all(), f"{experts} experts, biased ties: lowest ids"
        err = checkAllclose(
            torch.full_like(tw, SCALE / topk), tw, rtol=1e-6, atol=0,
            tol_err_ratio=0, msg=f"{experts} experts, ties: weights",
        )  # fmt: skip
        assert err == 0
        bias.zero_()
        tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
        want = torch.arange(topk, dtype=dtypes.i32)
        assert (ti == want).all(), f"{experts} experts, all tie: ids 0..{topk - 1}"


def case_large_logits_and_padding():
    for experts, m in itertools.product(SHAPES, (16, 192)):
        x, w, bias = router_inputs(m, 9, wscale=0.2, experts=experts)
        pad = torch.zeros(m, dtype=torch.bool)
        pad[::3] = True
        out, logits = route(x, w, bias, is_padding=pad)
        assert (logits > 20).any(), "no logit above the softplus switch"
        check_route(out, logits, bias, f"large logits {experts} {m}", pad)
        assert (out[1][pad] == -1).all() and (out[0][pad] == 0).all()
        assert (out[1][~pad] >= 0).all()


def case_offset_padding_view():
    """is_padding at an odd byte offset (a view into a larger mask)."""
    m = 37
    for experts in SHAPES:
        x, w, bias = router_inputs(m, 4, experts=experts)
        big = torch.zeros(m + 3, dtype=torch.bool)
        big[1::4] = True
        pad = big[1 : m + 1]
        assert pad.data_ptr() % 4 == 1
        out, logits = route(x, w, bias, is_padding=pad)
        check_route(out, logits, bias, f"offset is_padding {experts}", pad)


def case_row_strides():
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_supported,
    )

    m = 96
    for experts, topk in SHAPES.items():
        x, w, bias = router_inputs(m, 8, experts=experts)
        wide = torch.zeros((m, HIDDEN + 64), dtype=dtypes.bf16)
        wide[:, :HIDDEN] = x
        a, _ = route(wide[:, :HIDDEN], w, bias)
        b, _ = route(x, w, bias)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "row stride"
        # One row: its stride is irrelevant, even 0 (an expanded row).
        one = x[3].as_strided((1, HIDDEN), (0, 1))
        assert one.stride(0) == 0
        assert router_sqrtsoftplus_topk_supported(one, w, bias, topk)
        c, _ = route(one, w, bias)
        d, _ = route(x[3:4].contiguous(), w, bias)
        assert torch.equal(c[0], d[0]) and torch.equal(c[1], d[1]), "expanded row"
        # Several rows with a zero or overlapping row stride are declined.
        for bad in (x[3:4].expand(4, HIDDEN), x.as_strided((4, HIDDEN), (8, 1))):
            assert not router_sqrtsoftplus_topk_supported(bad, w, bias, topk)
            try:
                flydsl_router_sqrtsoftplus_topk(bad, w, bias, SCALE)
            except ValueError:
                continue
            raise AssertionError(f"row stride {bad.stride(0)} accepted")


def case_non_finite():
    m = 96
    for experts in SHAPES:
        x, w, bias = router_inputs(m, 8, experts=experts)
        x[3, 7] = float("nan")
        x[4, 7] = float("inf")
        (tw, ti), _ = route(x, w, bias)
        assert ((ti >= 0) & (ti < experts)).all(), "ids out of range"
        assert torch.isfinite(tw[[0, 1, 2]]).all()


def case_graph_after_warmup():
    """Warm up (every expert count), capture with caller buffers and a padding
    mask, and replay into buffers poisoned before every replay."""
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_warmup,
        shuffle_router_gate,
    )

    router_sqrtsoftplus_topk_warmup(weight_shuffled=True, with_padding=True)
    for (experts, topk), m in itertools.product(SHAPES.items(), (5, 100, 700)):
        x, w, bias = router_inputs(m, 5, experts=experts)
        ws = shuffle_router_gate(w)
        tw = torch.empty((m, topk))
        ti = torch.empty((m, topk), dtype=dtypes.i32)
        pad = torch.zeros(m, dtype=torch.bool)
        kw = {"weight_shuffled": True, "is_padding": pad}
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = flydsl_router_sqrtsoftplus_topk(
                x, ws, bias, SCALE, topk_weights=tw, topk_ids=ti, **kw
            )
        assert out[0] is tw and out[1] is ti
        for i in range(3):
            x.mul_(0.75).add_(0.0625)
            pad[i] = True
            ew, ei = flydsl_router_sqrtsoftplus_topk(x, ws, bias, SCALE, **kw)
            tw.fill_(float("nan"))
            ti.fill_(-7)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(tw, ew) and torch.equal(ti, ei), f"replay {m} {i}"


def case_eligibility():
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_supported,
    )

    x, w, bias = router_inputs(8, 0)
    _, w128, bias128 = router_inputs(8, 0, experts=128)
    assert router_sqrtsoftplus_topk_supported(x, w, bias)
    assert router_sqrtsoftplus_topk_supported(x, w128, bias128, 3)
    # The top-k must be the one of the gate's expert count.
    assert not router_sqrtsoftplus_topk_supported(x, w, bias, top_k=8)
    assert not router_sqrtsoftplus_topk_supported(x, w, bias, top_k=3)
    assert not router_sqrtsoftplus_topk_supported(x, w128, bias128, top_k=6)
    # Above the tuned rows the helper declines; the op stays correct there
    # (test_router_logits covers 1024 and 2049 rows).
    big = torch.zeros((769, HIDDEN), dtype=dtypes.bf16)
    for gate, b, topk in ((w, bias, 6), (w128, bias128, 3)):
        assert router_sqrtsoftplus_topk_supported(big[:768], gate, b, topk)
        assert not router_sqrtsoftplus_topk_supported(big, gate, b, topk)
        out = flydsl_router_sqrtsoftplus_topk(x[:0], gate, b, SCALE)
        assert out[1].shape == (0, topk)
    bias2 = torch.zeros(385)
    bad = [
        (x.float(), w, bias),
        (x, w[:256], bias[:256]),  # 256 experts
        (x, w, bias.to(dtypes.bf16)),
        (x, w, bias128),  # bias of another expert count
        (x[:, 1:], w[:, 1:], bias),  # hidden / alignment
        (x, w, bias2[1:]),  # 4 B aligned bias
    ]
    for args in bad:
        assert not router_sqrtsoftplus_topk_supported(*args)
        try:
            flydsl_router_sqrtsoftplus_topk(*args, SCALE)
        except ValueError:
            continue
        raise AssertionError("unsupported input accepted")


EDGE_CASES = {
    "shuffle_router_gate is a permutation": case_shuffle_is_permutation,
    "exact ties pick the lowest ids": case_ties_lowest_id,
    "logits > 20 and padding rows": case_large_logits_and_padding,
    "is_padding at an odd byte offset": case_offset_padding_view,
    "row strides (zero / overlapping declined)": case_row_strides,
    "NaN / inf inputs": case_non_finite,
    "graph replay after warmup, poisoned buffers": case_graph_after_warmup,
    "eligibility helper and ValueError": case_eligibility,
}


def summarize(title, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL router unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MoE router: bf16 GEMM + sqrt-softplus top-k",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        nargs="*",
        default=list(SHAPES),
        choices=list(SHAPES),
        help="routed experts (384: top-6, 128: top-3)",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128, 192, 256, 384, 512, 768, 1024],
        help="rows of the perf sweep (decode tokens), e.g. -m 16 192",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        nargs="*",
        default=[1, 7, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320]
        + [384, 512, 768, 1024, 2049],
        help="rows of the logits / ids sweep (every bucket bound and beyond)",
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="*",
        default=list(LAYOUTS),
        choices=LAYOUTS,
        help="gate layouts of the logits / ids sweep",
    )
    parser.add_argument(
        "-v",
        "--vllm-rows",
        type=int,
        nargs="*",
        default=[1, 16, 192, 768],
        help="rows of the comparison with vLLM's router (needs vLLM)",
    )
    args = parser.parse_args()

    for experts in args.experts:
        summarize(
            f"router {experts} experts, sqrt-softplus top-{SHAPES[experts]} summary",
            [test_router_sqrtsoftplus_topk(m, experts) for m in args.m],
        )
        summarize(
            f"router {experts} experts, logits / ids sweep",
            [
                test_router_logits(m, layout, experts)
                for layout, m in itertools.product(args.layout, args.rows)
            ],
        )
    try:
        import vllm  # noqa: F401
    except ImportError:
        aiter.logger.warning("vLLM not importable; skipping the vLLM comparison")
    else:
        summarize(
            "router vs vLLM",
            [
                test_router_vllm(m, experts)
                for experts, m in itertools.product(args.experts, args.vllm_rows)
            ],
        )
    edge = []
    for name, fn in EDGE_CASES.items():
        fn()
        edge.append({"case": name, "passed": True})
    summarize("router edge cases", edge)


if __name__ == "__main__":
    main()
