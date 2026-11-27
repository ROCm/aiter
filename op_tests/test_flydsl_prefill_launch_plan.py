# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch plans of the MiniMax-M3 NVFP4 prefill scorer and sparse attention (gfx950).

`make_score_prefill_plan` and `make_sparse_pa_prefill_plan` validate a forward pass'
batch once; a layer's call then only swaps pointers. The plan must launch the same
kernel on the same arguments as the plain entry point, so every check here compares
the plan against `score_prefill_flydsl_nvfp4` / `sparse_pa_nvfp4_prefill` bit for bit
(two "layers" per plan: different caches, same batch). The plan also has to refuse an
operand whose layout is not the one it was built for. The benchmarks time the host
side alone (the compiled launcher is replaced by a no-op, so no kernel runs and the
time is the Python in front of it) and the plan against the plain call on the GPU.

Run (exits nonzero if any check fails):
    python op_tests/test_flydsl_prefill_launch_plan.py [--case SUBSTR ...]
"""

import argparse
import itertools
import os
import sys
import time
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels import minimax_m3_index_score_prefill as score_mod
from aiter.ops.flydsl.kernels import sparse_pa_nvfp4_prefill as attn_mod
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    LOG2E,
    alloc_score,
    make_score_prefill_plan,
    score_prefill_flydsl,
    score_prefill_flydsl_fp4,
    score_prefill_flydsl_nvfp4,
)
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_prefill import (
    make_sparse_pa_prefill_plan,
    sparse_pa_nvfp4_prefill,
)
from aiter.test_common import benchmark, run_perftest
from op_tests.test_flydsl_minimax_m3_index_score_prefill_fp4 import (
    fp4_prefill_case,
    make_prefill_case,
    nv_case,
)
from op_tests.test_flydsl_sparse_pa_nvfp4 import SM_SCALE, make_case

SUPPORTED_GFX = ["gfx950"]  # the FlyDSL kernels are built for gfx950
NEG_INF = -float("inf")


# ---------------------------------------------------------------------------
# scorer
# ---------------------------------------------------------------------------
def score_layers(c):
    """Two layers' caches over one batch: the case's pages, and the same pages in the
    reverse order (other scores, still valid e2m1 / e4m3 bytes)."""
    return [
        (c["kd"], c["ks"]),
        (c["kd"].flip(0).contiguous(), c["ks"].flip(0).contiguous()),
    ]


def score_args(c):
    return (
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
    )


def score_direct(c, q, kd, ks, k_round):
    rows, heads = q.shape[:2]
    out = alloc_score(rows, heads, c["mb"], "cuda")
    score_prefill_flydsl_nvfp4(q, kd, ks, *score_args(c), out=out, k_round=k_round)
    return out


def check_score_plan_matches_direct(qls, seqs, heads, qdt, k_round):
    c = nv_case(qls, seqs, heads)
    q = c["q_bf16"] if qdt == "bf16" else c["q8"]
    layers = score_layers(c)
    out = alloc_score(q.shape[0], heads, c["mb"], "cuda")
    plan = make_score_prefill_plan(
        q,
        layers[0][0],
        *score_args(c),
        out=out,
        k_scale=layers[0][1],
        nvfp4=k_round,
    )
    for kd, ks in layers * 2:  # a layer revisited: the plan keeps no per-layer state
        out.fill_(NEG_INF)
        got = plan(q, kd, ks)
        assert got is out, "the plan returns its own output"
        ref = score_direct(c, q, kd, ks, k_round)
        assert torch.equal(got, ref), f"plan != direct ({qls} {seqs} {qdt} {k_round})"
    other = score_direct(c, q, *layers[0], k_round)
    assert bool(torch.isfinite(ref).any()), "the case wrote nothing"
    assert not torch.equal(other, ref), "the two layers must score differently"


def check_lazy_exports():
    """Every lazy export is a (module, attribute) pair, and the new ones resolve."""
    from aiter.ops import flydsl

    for name, target in flydsl._LAZY_IMPORTS.items():
        assert len(target) == 2, f"{name}: {target}"
    assert flydsl.make_minimax_m3_index_score_prefill_plan is make_score_prefill_plan
    assert flydsl.make_sparse_pa_nvfp4_prefill_plan is make_sparse_pa_prefill_plan
    assert flydsl.make_fp4_mqa_plan.__name__ == "make_fp4_mqa_plan"
    for name in flydsl.__all__:
        assert name in flydsl._LAZY_IMPORTS or hasattr(flydsl, name), name


def check_score_plan_fp8_cache():
    """The fp8 cache has no scale view: the plan takes (idx_q, cache) only."""
    c = make_prefill_case(2, 300, 1500)
    q, cache = c["q_bf16"], c["cache"]
    plan = make_score_prefill_plan(q, cache, *score_args(c))
    for layer in (cache, cache.flip(0).contiguous()):
        plan.out.fill_(NEG_INF)
        ref = alloc_score(q.shape[0], 1, c["mb"], "cuda")
        score_prefill_flydsl(q, layer, *score_args(c), out=ref)
        assert torch.equal(plan(q, layer), ref)
    expect_value_error("scale view", plan, q, cache, cache)


def check_score_plan_mxfp4_typed_views():
    """per_1x32_f4_quant's own dtypes (fp4x2 data, e8m0 scale) are viewed per layer."""
    c, kd, ks = fp4_prefill_case(1, 256, 1000)
    q = c["q_bf16"]
    tkd, tks = kd.view(dtypes.fp4x2), ks.view(dtypes.fp8_e8m0)
    plan = make_score_prefill_plan(q, tkd, *score_args(c), k_scale=tks)
    other = kd.flip(0).contiguous(), ks.flip(0).contiguous()
    for lkd, lks in (
        (tkd, tks),
        (other[0].view(dtypes.fp4x2), other[1].view(dtypes.fp8_e8m0)),
    ):
        plan.out.fill_(NEG_INF)
        ref = alloc_score(q.shape[0], 1, c["mb"], "cuda")
        score_prefill_flydsl_fp4(q, lkd, lks, *score_args(c), out=ref)
        assert torch.equal(plan(q, lkd, lks), ref)
    expect_value_error("uint8 views", plan, q, kd, ks)


def check_score_plan_owns_its_output():
    c = nv_case([300, 200], [2000, 700], 1)
    kd, ks = c["kd"], c["ks"]
    plan = make_score_prefill_plan(
        c["q8"], kd, *score_args(c), k_scale=ks, nvfp4="e4m3"
    )
    assert plan.out.shape == (1, 500, c["mb"]) and plan.out is plan(c["q8"], kd, ks)
    assert torch.equal(plan.out, score_direct(c, c["q8"], kd, ks, "e4m3"))


def expect_value_error(what, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return
    raise AssertionError(f"{what}: expected a ValueError")


def check_score_plan_rejections():
    c = nv_case([256, 128], [1024, 640], 1)
    q, kd, ks = c["q8"], c["kd"], c["ks"]
    plan = make_score_prefill_plan(q, kd, *score_args(c), k_scale=ks, nvfp4="e4m3")
    plan(q, kd, ks)  # the layout it was built from is accepted
    expect_value_error("no scale", plan, q, kd)
    expect_value_error("bf16 q", plan, c["q_bf16"], kd, ks)
    expect_value_error("fewer rows", plan, q[:-1], kd, ks)
    expect_value_error("other pool", plan, q, kd[:-1], ks[:-1])
    expect_value_error("cache dtype", plan, q, kd.view(torch.int8), ks)
    shifted = torch.empty(q.numel() + 16, dtype=q.dtype, device=q.device)[1:-15]
    expect_value_error("misaligned q", plan, shifted.view_as(q), kd, ks)
    expect_value_error("cpu", plan, q.cpu(), kd, ks)
    expect_value_error(
        "bad k_round",
        make_score_prefill_plan,
        q,
        kd,
        *score_args(c),
        k_scale=ks,
        nvfp4="e5m2",
    )
    expect_value_error(
        "plan validates like the plain call",
        make_score_prefill_plan,
        q,
        kd,
        c["bt"],
        c["cu"],
        c["seq"].long(),
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        k_scale=ks,
        nvfp4="e4m3",
    )


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------
def attn_layers(c):
    """Two layers: the case's cache, its page-reversed copy, each with its own top-k
    (the second one is the first row-shifted: valid ids, other blocks first)."""
    cache2 = c["cache"].flip(0).contiguous()
    topk2 = c["topk"].flip(-1).contiguous()
    ks2 = c["k_scale"] * 2
    return [
        (c["q"], c["cache"], c["topk"], c["k_scale"], c["v_scale"]),
        (c["q"].flip(-1).contiguous(), cache2, topk2, ks2, c["v_scale"] * 0.5),
    ]


def attn_direct(c, q, cache, topk, ks, vs, **kw):
    out = torch.full_like(q, float("nan"))
    return sparse_pa_nvfp4_prefill(
        q, cache, topk, c["bt"], c["row_req"], c["row_kv"], SM_SCALE, out, ks, vs, **kw
    )


def check_attn_plan_matches_direct(seq_lens, q_lens, hkv, group, kw):
    c = make_case(seq_lens, q_lens, hkv, group)
    layers = attn_layers(c)
    q, cache, topk, ks, vs = layers[0]
    plan = make_sparse_pa_prefill_plan(
        q,
        cache,
        topk,
        c["bt"],
        c["row_req"],
        c["row_kv"],
        SM_SCALE,
        k_scale=ks,
        v_scale=vs,
        **kw,
    )
    for q, cache, topk, ks, vs in layers * 2:
        ref = attn_direct(c, q, cache, topk, ks, vs, **kw)
        got = plan(q, cache, topk, k_scale=ks, v_scale=vs)  # the plan's own output
        assert got is plan.out and torch.equal(got, ref), "plan != direct"
        own = torch.full_like(q, float("nan"))
        assert plan(q, cache, topk, own, ks, vs) is own and torch.equal(own, ref)
    assert bool(torch.isfinite(ref).all()), "the case left rows unwritten"
    assert not torch.equal(ref, attn_direct(c, *layers[0], **kw)), "layers must differ"


def check_attn_plan_default_scales():
    c = make_case([1500, 600], [300, 200], 1, 16)
    plan = make_sparse_pa_prefill_plan(
        c["q"], c["cache"], c["topk"], c["bt"], c["row_req"], c["row_kv"], SM_SCALE
    )
    ref = sparse_pa_nvfp4_prefill(
        c["q"], c["cache"], c["topk"], c["bt"], c["row_req"], c["row_kv"], SM_SCALE
    )
    assert torch.equal(plan(c["q"], c["cache"], c["topk"]), ref)
    # Built with the case's scales: a call that leaves them out still means one.
    scaled = make_sparse_pa_prefill_plan(
        c["q"],
        c["cache"],
        c["topk"],
        c["bt"],
        c["row_req"],
        c["row_kv"],
        SM_SCALE,
        k_scale=c["k_scale"],
        v_scale=c["v_scale"],
    )
    got = scaled(c["q"], c["cache"], c["topk"])
    assert torch.equal(got, ref), "a scale left out must be one, not the build-time one"
    assert not torch.equal(
        scaled(c["q"], c["cache"], c["topk"], None, c["k_scale"], c["v_scale"]), ref
    ), "the case's scales must matter"


def check_attn_plan_empty_batch():
    c = make_case([300], [10], 1, 16)
    plan = make_sparse_pa_prefill_plan(
        c["q"][:0],
        c["cache"],
        c["topk"][:, :0],
        c["bt"],
        c["row_req"][:0],
        c["row_kv"][:0],
        SM_SCALE,
    )
    assert plan(c["q"][:0], c["cache"], c["topk"][:, :0]).shape[0] == 0


def check_attn_plan_rejections():
    c = make_case([1200, 500], [200, 100], 1, 16)
    q, cache, topk = c["q"], c["cache"], c["topk"]
    plan = make_sparse_pa_prefill_plan(
        q, cache, topk, c["bt"], c["row_req"], c["row_kv"], SM_SCALE
    )
    plan(q, cache, topk)
    expect_value_error("fewer rows", plan, q[:-1], cache, topk)
    expect_value_error("fp16 q", plan, q.half(), cache, topk)
    expect_value_error("other page size", plan, q, cache[:, :-4], topk)
    expect_value_error("topk dtype", plan, q, cache, topk.long())
    expect_value_error("out layout", plan, q, cache, topk, torch.empty_like(q).half())
    expect_value_error("scale dtype", plan, q, cache, topk, None, c["k_scale"].half())
    expect_value_error(
        "scale numel", plan, q, cache, topk, None, None, c["v_scale"].repeat(2)
    )
    expect_value_error("scale device", plan, q, cache, topk, None, c["k_scale"].cpu())
    expect_value_error(
        "plan validates like the plain call",
        make_sparse_pa_prefill_plan,
        q,
        cache,
        topk,
        c["bt"],
        c["row_req"].long(),
        c["row_kv"],
        SM_SCALE,
    )


# ---------------------------------------------------------------------------
# benchmarks
# ---------------------------------------------------------------------------
def host_us(fn, n=300):
    """Python time of `fn` with the compiled launcher replaced by a no-op."""
    noop = mock.Mock()
    with mock.patch.object(score_mod, "_run_compiled", noop), mock.patch.object(
        attn_mod, "_run_compiled", noop
    ):
        for _ in range(20):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1e6


@benchmark()
def test_score_plan(batch, q_len, ctx_len, k_round):
    c = nv_case([q_len] * batch, [ctx_len] * batch, exact_sum=True)
    q, kd, ks = c["q8"], c["kd"], c["ks"]
    out = alloc_score(q.shape[0], 1, c["mb"], "cuda")
    plan = make_score_prefill_plan(
        q, kd, *score_args(c), out=out, k_scale=ks, nvfp4=k_round
    )

    def direct():
        return score_prefill_flydsl_nvfp4(
            q, kd, ks, *score_args(c), out=out, k_round=k_round
        )

    ref = score_direct(c, q, kd, ks, k_round)
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in (("direct", direct), ("plan", lambda: plan(q, kd, ks))):
        out.fill_(NEG_INF)
        fn()
        assert torch.equal(out, ref), f"{name}: scores differ from the direct call"
        _, us = run_perftest(fn, num_iters=10, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} host us"] = host_us(fn)
    return ret


@benchmark()
def test_attn_plan(q_len, ctx, hkv, group):
    c = make_case([ctx], [q_len], hkv, group)
    layers = attn_layers(c)
    q, cache, topk, ks, vs = layers[0]
    args = (c["bt"], c["row_req"], c["row_kv"], SM_SCALE)
    out = torch.empty_like(q)
    plan = make_sparse_pa_prefill_plan(q, cache, topk, *args, out, ks, vs)

    def direct():
        return sparse_pa_nvfp4_prefill(q, cache, topk, *args, out, ks, vs)

    ref = attn_direct(c, q, cache, topk, ks, vs)
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in (
        ("direct", direct),
        ("plan", lambda: plan(q, cache, topk, out, ks, vs)),
    ):
        out.fill_(float("nan"))
        fn()
        assert torch.equal(out, ref), f"{name}: output differs from the direct call"
        _, us = run_perftest(fn, num_iters=10, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} host us"] = host_us(fn)
    return ret


def build_cases():
    cases = []
    for qdt, k_round in itertools.product(("bf16", "e4m3"), ("e4m3", "exact")):
        for qls, seqs, heads in (
            ([256], [256], 1),
            ([300, 129, 777], [5000, 129, 1300], 1),
            ([128, 512], [2048, 2048], 2),
        ):
            cases.append(
                (
                    f"score_plan_{qdt}_{k_round}_q{qls}_ctx{seqs}_h{heads}",
                    check_score_plan_matches_direct,
                    (qls, seqs, heads, qdt, k_round),
                )
            )
    cases.append(("lazy_exports", check_lazy_exports, ()))
    cases.append(("score_plan_fp8_cache", check_score_plan_fp8_cache, ()))
    cases.append(
        ("score_plan_mxfp4_typed_views", check_score_plan_mxfp4_typed_views, ())
    )
    cases.append(("score_plan_owns_its_output", check_score_plan_owns_its_output, ()))
    cases.append(("score_plan_rejections", check_score_plan_rejections, ()))
    for name, seqs, qls, hkv, group, kw in (
        ("ragged_h1", [3000, 1500, 400], [700, 300, 400], 1, 16, {}),
        ("ragged_h4", [3000, 1500], [700, 300], 4, 16, {}),
        ("group8", [2100, 900], [500, 100], 1, 8, {}),
        ("tile1", [2100, 900], [500, 100], 1, 16, {"tq": 1, "rounds": 1}),
        ("tile4", [4100, 2500], [1000, 900], 1, 16, {"tq": 4, "rounds": 2}),
    ):
        cases.append(
            (
                f"attn_plan_{name}",
                check_attn_plan_matches_direct,
                (seqs, qls, hkv, group, kw),
            )
        )
    cases.append(("attn_plan_default_scales", check_attn_plan_default_scales, ()))
    cases.append(("attn_plan_empty_batch", check_attn_plan_empty_batch, ()))
    cases.append(("attn_plan_rejections", check_attn_plan_rejections, ()))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", nargs="*", default=None, help="only cases containing")
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--q-len", type=int, nargs="*", default=[512, 2048])
    ap.add_argument("--ctx-len", type=int, nargs="*", default=[8192, 32768])
    ap.add_argument("--k-round", nargs="*", default=["e4m3", "exact"])
    args = ap.parse_args(argv)
    if not torch.cuda.is_available() or get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("requires %s; skipping", SUPPORTED_GFX)
        return 0
    rows = []
    for name, check, cargs in build_cases():
        if args.case is not None and not any(k in name for k in args.case):
            continue
        try:
            check(*cargs)
            rows.append({"case": name, "status": "PASS", "detail": ""})
        except AssertionError as e:
            rows.append({"case": name, "status": "FAIL", "detail": repr(e)[:200]})
    if rows:
        aiter.logger.info(
            "prefill launch plan checks (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    failed = [r["case"] for r in rows if r["status"] != "PASS"]
    if failed:
        aiter.logger.error("%d of %d checks failed: %s", len(failed), len(rows), failed)
        return 1
    if args.case is not None:
        return 0
    perf = [
        test_score_plan(b, q, c, k)
        for b, q, c, k in itertools.product(
            args.batch, args.q_len, args.ctx_len, args.k_round
        )
        if q <= c
    ]
    aiter.logger.info(
        "score plan summary (markdown):\n%s",
        pd.DataFrame(perf).to_markdown(index=False),
    )
    perf = [
        test_attn_plan(q, c, hkv, 16)
        for q, c, hkv in itertools.product(args.q_len, args.ctx_len, (1, 4))
        if q <= c
    ]
    aiter.logger.info(
        "attention plan summary (markdown):\n%s",
        pd.DataFrame(perf).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
