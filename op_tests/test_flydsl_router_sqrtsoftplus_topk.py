# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance of ``flydsl_router_sqrtsoftplus_topk`` (gfx950).

The DeepSeek-V4.1-Flash MoE router for decode rows:

    logits = x [M, 5120] @ gate [384, 5120]^T      (bf16, fp32 accumulate)
    score  = sqrt(softplus(logits))                (logits above 20 pass through)
    ids    = top-6 of score + bias                 (bias fp32 [384])
    w      = score[ids] * 1.5 / sum(score[ids])

Four tables:

1. ``test_router_sqrtsoftplus_topk`` (perf + correctness, ``-m``): the FlyDSL op
   with the row-major gate, with the load-time shuffled gate
   (``shuffle_router_gate``), and as the model calls it (shuffled gate, an
   ``is_padding`` mask, caller-allocated outputs filled with NaN / -7), plus
   AITER's ``topk_gating`` on a ``torch.mm`` (hipBLASLt) fp32 GEMM, each against
   an fp32 torch reference (rows whose 6th and 7th candidates tie within fp32
   noise excluded).
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
HIDDEN, EXPERTS, TOPK = 5120, 384, 6
# 6th / 7th ``score + bias`` closer than this (relative) may pick either.
MARGIN = 1e-5
LAYOUTS = ("row_major", "shuffled")


def score_ref(logits):
    return torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits))).sqrt()


def run_torch_router(logits, bias, is_padding=None):
    """fp32 reference: weights, ids, and the rows without a 6th / 7th tie."""
    score = score_ref(logits)
    top = torch.topk(score + bias, 7, dim=-1)
    ids = top.indices[:, :6]
    w = score.gather(1, ids)
    w = w * (SCALE / w.sum(-1, keepdim=True))
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    gap = top.values[:, 5] - top.values[:, 6]
    return w, ids, gap > MARGIN * top.values[:, 5].abs().clamp_min(1.0)


def topk_ref(logits, bias, is_padding=None):
    """The op's top-k decoding of given logits: weight ``(score + bias) - bias``
    summed in slot order (vLLM's ``topk_softplus_sqrt``)."""
    score = score_ref(logits.float())
    ids = torch.topk(score + bias, 6, dim=-1)[1]
    w = (score + bias).gather(1, ids) - bias[ids]
    total = w[:, 0]
    for k in range(1, 6):
        total = total + w[:, k]
    w = w * (SCALE / total[:, None])
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    return w, ids


def router_inputs(m, seed, wscale=0.02):
    """x rows at per-row scales 2^-2..2^2, so a row read with another row's
    data or scale shows up."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, HIDDEN), generator=g) * torch.exp2(
        torch.randint(-2, 3, (m, 1), generator=g).float()
    )
    w = torch.randn((EXPERTS, HIDDEN), generator=g) * wscale
    bias = torch.randn(EXPERTS, generator=g) * 0.1
    return x.to(dtypes.bf16), w.to(dtypes.bf16), bias


def route(x, w, bias, **kw):
    """Run the op with a caller scratch; also return its own logits."""
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_workspace_shape,
    )

    shape = router_sqrtsoftplus_topk_workspace_shape(
        x.shape[0], kw.get("weight_shuffled", False)
    )
    partials = torch.empty(shape, dtype=dtypes.fp32)
    out = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, partials=partials, **kw)
    torch.cuda.synchronize()
    logits = partials[0].clone()
    for p in partials[1:]:
        logits = logits + p
    return out, logits


def close_rows(logits, bias):
    """Rows whose 6th and 7th ``score + bias`` tie within MARGIN."""
    top = torch.topk(score_ref(logits) + bias, 7, dim=-1)[0]
    return (top[:, 5] - top[:, 6]).abs() < MARGIN * top[:, 5].abs().clamp_min(1.0)


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
def test_router_sqrtsoftplus_topk(m):
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk, shuffle_router_gate
    from aiter.ops.topk import topk_gating

    x, w, bias = router_inputs(m, m)
    ws = shuffle_router_gate(w)
    logits = torch.mm(x.float(), w.float().t())
    ref = run_torch_router(logits, bias)
    # The model passes a padding mask and its own output buffers.
    pad = torch.zeros(m, dtype=torch.bool)
    pad[1::5] = True
    ref_pad = run_torch_router(logits, bias, pad)
    out_w = torch.full((m, TOPK), float("nan"))
    out_ids = torch.full((m, TOPK), -7, dtype=dtypes.i32)

    def mm_topk_gating():
        tw = torch.empty(m, TOPK)
        ti = torch.empty(m, TOPK, dtype=dtypes.i32)
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
    flops = 2 * m * HIDDEN * EXPERTS
    nbytes = (m + EXPERTS) * HIDDEN * 2 + m * TOPK * 8
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
def test_router_logits(m, layout):
    """Logits, ids / weights and repeatability at one row count and layout."""
    from aiter.ops.flydsl import shuffle_router_gate

    shuffled = layout == "shuffled"
    x, w, bias = router_inputs(m, m + 11 * shuffled)
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
def test_router_vllm(m):
    """Bitwise against vLLM's top-k on the op's logits (normal and above the
    softplus switch), and vLLM's GEMM + top-k off near-ties."""
    import importlib

    from vllm import _custom_ops as ops

    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    ret = {}
    for seed, wscale in ((m, 0.02), (m + 1, 0.05), (m + 2, 0.2)):
        x, w, bias = router_inputs(m, seed, wscale)
        out, logits = route(x, w, bias)
        bw = torch.empty((m, TOPK), dtype=dtypes.fp32)
        bi = torch.empty((m, TOPK), dtype=dtypes.i32)
        ops.topk_hash_softplus_sqrt(
            bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, *[None] * 4, 0
        )
        assert torch.equal(out[1], bi), f"{m} wscale {wscale}: ids vs vLLM"
        assert torch.equal(out[0], bw), f"{m} wscale {wscale}: weights vs vLLM"
        ret[f"bitwise wscale {wscale}"] = True
    # Registers torch.ops.vllm.fp32_router_gemm_dispatch.
    importlib.import_module("vllm.model_executor.layers.fused_moe.router.gate_linear")
    x, w, bias = router_inputs(m, m + 3)
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    logits = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
    bw = torch.empty((m, TOPK), dtype=dtypes.fp32)
    bi = torch.empty((m, TOPK), dtype=dtypes.i32)
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

    _, w, _ = router_inputs(1, 0)
    back = shuffle_router_gate(w).reshape(EXPERTS // 16, HIDDEN // 32, 4, 16, 8)
    back = back.permute(0, 3, 1, 2, 4).reshape(EXPERTS, HIDDEN)
    assert torch.equal(back, w)


def case_ties_lowest_id():
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    x = torch.zeros((16, HIDDEN), dtype=dtypes.bf16)
    w = torch.zeros((EXPERTS, HIDDEN), dtype=dtypes.bf16)
    bias = torch.zeros(EXPERTS)
    for i in (383, 5, 100, 7, 200, 9, 300, 11):
        bias[i] = 1.0
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    want = torch.tensor([5, 7, 9, 11, 100, 200], dtype=dtypes.i32)
    assert (ti == want).all(), "biased ties: lowest ids"
    err = checkAllclose(
        torch.full_like(tw, SCALE / 6), tw, rtol=1e-6, atol=0, tol_err_ratio=0,
        msg="ties: weights",
    )  # fmt: skip
    assert err == 0
    bias.zero_()
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    assert (ti == torch.arange(6, dtype=dtypes.i32)).all(), "all tie: ids 0..5"


def case_large_logits_and_padding():
    for m in (16, 192):
        x, w, bias = router_inputs(m, 9, wscale=0.2)
        pad = torch.zeros(m, dtype=torch.bool)
        pad[::3] = True
        out, logits = route(x, w, bias, is_padding=pad)
        assert (logits > 20).any(), "no logit above the softplus switch"
        check_route(out, logits, bias, f"large logits {m}", pad)
        assert (out[1][pad] == -1).all() and (out[0][pad] == 0).all()
        assert (out[1][~pad] >= 0).all()


def case_offset_padding_view():
    """is_padding at an odd byte offset (a view into a larger mask)."""
    m = 37
    x, w, bias = router_inputs(m, 4)
    big = torch.zeros(m + 3, dtype=torch.bool)
    big[1::4] = True
    pad = big[1 : m + 1]
    assert pad.data_ptr() % 4 == 1
    out, logits = route(x, w, bias, is_padding=pad)
    check_route(out, logits, bias, "offset is_padding", pad)


def case_row_strides():
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_supported,
    )

    m = 96
    x, w, bias = router_inputs(m, 8)
    wide = torch.zeros((m, HIDDEN + 64), dtype=dtypes.bf16)
    wide[:, :HIDDEN] = x
    a, _ = route(wide[:, :HIDDEN], w, bias)
    b, _ = route(x, w, bias)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "row stride"
    # One row: its stride is irrelevant, even 0 (an expanded row).
    one = x[3].as_strided((1, HIDDEN), (0, 1))
    assert one.stride(0) == 0 and router_sqrtsoftplus_topk_supported(one, w, bias)
    c, _ = route(one, w, bias)
    d, _ = route(x[3:4].contiguous(), w, bias)
    assert torch.equal(c[0], d[0]) and torch.equal(c[1], d[1]), "one expanded row"
    # Several rows with a zero or overlapping row stride are declined.
    for bad in (x[3:4].expand(4, HIDDEN), x.as_strided((4, HIDDEN), (8, 1))):
        assert not router_sqrtsoftplus_topk_supported(bad, w, bias)
        try:
            flydsl_router_sqrtsoftplus_topk(bad, w, bias, SCALE)
        except ValueError:
            continue
        raise AssertionError(f"row stride {bad.stride(0)} accepted")


def case_non_finite():
    m = 96
    x, w, bias = router_inputs(m, 8)
    x[3, 7] = float("nan")
    x[4, 7] = float("inf")
    (tw, ti), _ = route(x, w, bias)
    assert ((ti >= 0) & (ti < EXPERTS)).all(), "ids out of range"
    assert torch.isfinite(tw[[0, 1, 2]]).all()


def case_graph_after_warmup():
    """Warm up, capture with caller buffers and a padding mask, and replay
    into buffers poisoned before every replay."""
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_warmup,
        shuffle_router_gate,
    )

    router_sqrtsoftplus_topk_warmup(weight_shuffled=True, with_padding=True)
    for m in (5, 100, 700):
        x, w, bias = router_inputs(m, 5)
        ws = shuffle_router_gate(w)
        tw = torch.empty((m, TOPK))
        ti = torch.empty((m, TOPK), dtype=dtypes.i32)
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
    assert router_sqrtsoftplus_topk_supported(x, w, bias)
    assert not router_sqrtsoftplus_topk_supported(x, w, bias, top_k=8)
    # Above the tuned rows the helper declines; the op stays correct there
    # (test_router_logits covers 1024 and 2049 rows).
    big = torch.zeros((769, HIDDEN), dtype=dtypes.bf16)
    assert router_sqrtsoftplus_topk_supported(big[:768], w, bias)
    assert not router_sqrtsoftplus_topk_supported(big, w, bias)
    out = flydsl_router_sqrtsoftplus_topk(x[:0], w, bias, SCALE)
    assert out[1].shape == (0, TOPK)
    bias2 = torch.zeros(EXPERTS + 1)
    bad = [
        (x.float(), w, bias),
        (x, w[:256], bias),  # 256 experts
        (x, w, bias.to(dtypes.bf16)),
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
        description="FlyDSL MoE router: bf16 GEMM + sqrt-softplus top-6",
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

    summarize(
        "router sqrt-softplus top-6 summary",
        [test_router_sqrtsoftplus_topk(m) for m in args.m],
    )
    summarize(
        "router logits / ids sweep",
        [
            test_router_logits(m, layout)
            for layout, m in itertools.product(args.layout, args.rows)
        ],
    )
    try:
        import vllm  # noqa: F401
    except ImportError:
        aiter.logger.warning("vLLM not importable; skipping the vLLM comparison")
    else:
        summarize("router vs vLLM", [test_router_vllm(m) for m in args.vllm_rows])
    edge = []
    for name, fn in EDGE_CASES.items():
        fn()
        edge.append({"case": name, "passed": True})
    summarize("router edge cases", edge)


if __name__ == "__main__":
    main()
