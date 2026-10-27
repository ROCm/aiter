# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index score reading an MXFP4 cache natively (no staging).

The native kernel is the staged path's fp8 scorer with only its K side replaced, so
it must give the staged path's scores bit for bit -- and, as the staged path is
itself judged bit for bit against aiter's `pa_sparse_block_score_prefill` on the
dequantized fp8 values, so is this one. Both judges run here, for both query dtypes.
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels import minimax_m3_index_score_prefill_fp4_native as native
from aiter.test_common import benchmark, run_perftest

try:
    from op_tests import test_flydsl_minimax_m3_index_score_fp4 as fp4_test
except ImportError:  # run as a script from op_tests/
    import test_flydsl_minimax_m3_index_score_fp4 as fp4_test

fp8_test = fp4_test.fp8_test
requires_gfx950 = fp4_test.requires_gfx950
SENTINEL, LOG2E, P, D = fp4_test.SENTINEL, fp4_test.LOG2E, fp4_test.P, fp4_test.D
qdt = fp4_test.qdt


def run_native(c, kd, ks, qdt="bf16", **kw):
    heads = c["q8"].shape[1]
    out = fp8_test.prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    native.score_prefill_flydsl_fp4_native(
        c["q_bf16"] if qdt == "bf16" else c["q8"],
        kd,
        ks,
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
        **kw,
    )
    return out.reshape(heads, c["q8"].shape[0], c["mb"])


# The staged path's shapes, plus short-context / long-query ones and a multi-request
# chunk with a long prefix, which is where the native kernel is meant to matter.
SHAPES = fp8_test.PREFILL_SHAPES + [
    (1, 8192, 8192),
    (1, 7000, 7000),
    (2, 4096, 20000),
    (4, 512, 8192),
    (3, 300, 33333),
]


@requires_gfx950
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}")
def test_native_matches_aiter_and_the_staged_path(shape, qdt):
    c, kd, ks = fp4_test.fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_native(c, kd, ks, qdt)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], fp4_test.run_prefill_fp4(c, kd, ks, qdt)[live])


@requires_gfx950
@pytest.mark.parametrize(
    "shape", fp8_test.GEOM_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_native_every_geometry(shape, qdt):
    c, kd, ks = fp4_test.fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_native(c, kd, ks, qdt), ref, live, f"{shape}")


@requires_gfx950
def test_native_two_heads(qdt):
    c, kd, ks = fp4_test.fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_native(c, kd, ks, qdt), ref, live, "heads=2")


@requires_gfx950
@pytest.mark.parametrize("ctx", [1, 127, 128, 129, 1000, 4097])
def test_native_ragged_tails(ctx):
    """Lengths off a page boundary: the tail of the last page holds whatever the
    cache holds, and must not reach a live score."""
    c, kd, ks = fp4_test.fp4_prefill_case(3, min(ctx, 128), ctx)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_native(c, kd, ks), ref, live, f"ctx={ctx}")


@requires_gfx950
def test_native_scale_extremes(qdt):
    """Per-token magnitudes spread over 2^-3 .. 2^3, so the e8m0 bytes walk across
    the window e4m3 can still hold exactly: the scale really is applied per lane
    group and per token, not once."""
    c = fp8_test.make_prefill_case(2, 256, 3000)
    span = torch.randint(-3, 4, (c["cache"].shape[0], P, 1), device="cuda")
    k = c["cache"].to(torch.bfloat16) * (2.0 ** span.to(torch.bfloat16))
    kd, ks, deq = fp4_test.quantize_mxfp4(k)
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    assert ks.min() < ks.max() - 6
    c = dict(c, cache=as_fp8)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_native(c, kd, ks, qdt), ref, live, "scales")


@requires_gfx950
def test_native_ragged_batch_and_garbage_tails(qdt):
    """Unequal lengths, and the bytes past each request's end overwritten with 0xFF
    (e8m0 255 and NaN-ish nibbles): live scores do not move."""
    c, kd, ks = fp4_test.fp4_prefill_case(3, 128, 1500)
    c["seq"] = torch.tensor([1500, 300, 700], dtype=torch.int32, device="cuda")
    c["prefix"] = torch.tensor([1372, 172, 572], dtype=torch.int32, device="cuda")
    base = run_native(c, kd, ks, qdt)
    kd2, ks2 = kd.clone(), ks.clone()
    for b, n in enumerate([1500, 300, 700]):
        for p in range(-(-n // P)):
            page = int(c["bt"][b, p])
            first_dead = n - p * P
            if first_dead < P:
                kd2[page, first_dead:] = 0xFF
                ks2[page, first_dead:] = 0xFF
    poisoned = run_native(c, kd2, ks2, qdt)
    live = base != SENTINEL
    assert torch.equal(base[live], poisoned[live])
    ref, ref_live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(base, ref, ref_live, "ragged lengths")


@requires_gfx950
def test_native_mixed_query_lengths_and_trailing_empty_request(qdt):
    dev = "cuda"
    torch.manual_seed(0)
    q_a, ctx_a, ctx_b = 1024, 1024, 16384
    cu = torch.tensor([0, q_a, q_a + 1, q_a + 1], dtype=torch.int32, device=dev)
    seq = torch.tensor([ctx_a, ctx_b, ctx_b], dtype=torch.int32, device=dev)
    prefix = torch.tensor(
        [ctx_a - q_a, ctx_b - 1, ctx_b], dtype=torch.int32, device=dev
    )
    mb = -(-ctx_b // P)
    q8 = (torch.randn(q_a + 1, 1, D, device=dev) / 4).to(dtypes.fp8)
    kd, ks, deq = fp4_test.fp4_cache_of(
        (torch.randn(3 * mb, P, D, device=dev) / 4).to(dtypes.fp8)
    )
    as_fp8 = deq.to(dtypes.fp8)
    bt = torch.randperm(3 * mb, device=dev, dtype=torch.int32).view(3, mb).contiguous()
    ref = torch.full((1, q_a + 1, mb), -float("inf"), device=dev)
    from aiter.ops.msa_block_select import pa_sparse_block_score_prefill

    pa_sparse_block_score_prefill(
        q8, as_fp8, ref, bt[:2], cu[:3], seq[:2], max_query_len=q_a, max_seq_len=ctx_b
    )
    out = fp8_test.prefill_alloc_score(q_a + 1, 1, mb, dev)
    out.fill_(SENTINEL)
    native.score_prefill_flydsl_fp4_native(
        q8.float().bfloat16() if qdt == "bf16" else q8,
        kd,
        ks,
        bt,
        cu,
        seq,
        prefix,
        q_a,
        ctx_b,
        1.0 / LOG2E,
        out=out,
    )
    fp8_test.assert_matches_aiter(
        out.reshape(1, q_a + 1, mb), ref, torch.isfinite(ref), "mixed + trailing empty"
    )


@requires_gfx950
def test_native_strided_views_and_q_scale():
    """Single-region pages (data and scale views of one [pages, 8704] buffer), a
    strided e4m3 query, and `q_scale`: all bit-identical to the plain inputs."""
    c, kd, ks = fp4_test.fp4_prefill_case(2, 256, 3000, heads=2)
    plain = run_native(c, kd, ks, "e4m3")
    _buf, kd1, ks1 = fp4_test.single_region_views(kd, ks)
    assert torch.equal(plain, run_native(c, kd1, ks1, "e4m3"))
    rows, heads = c["q8"].shape[:2]
    wide = torch.zeros(rows, heads + 1, D + 64, dtype=dtypes.fp8, device="cuda")
    view = wide[:, :heads, :D]
    view.copy_(c["q8"])
    assert torch.equal(plain, run_native(dict(c, q8=view), kd1, ks1, "e4m3"))
    out = fp8_test.prefill_alloc_score(rows, heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    native.score_prefill_flydsl_fp4_native(
        c["q8"], kd, ks, c["bt"], *args, out=out, q_scale=0.5
    )
    got, live = out.reshape(plain.shape), plain != SENTINEL
    assert torch.equal(got[live], plain[live] * 0.5)


@requires_gfx950
def test_default_entry_is_native_and_staged_is_opt_in(qdt):
    """`score_prefill_flydsl_fp4` reads the cache natively unless `staged=True`; the
    two routes agree, and `workspace` belongs to the staged one only."""
    c, kd, ks = fp4_test.fp4_prefill_case(2, 300, 5000)
    default = fp4_test.run_prefill_fp4(c, kd, ks, qdt, staged=False)
    live = default != SENTINEL
    assert torch.equal(default[live], run_native(c, kd, ks, qdt)[live])
    assert torch.equal(default[live], fp4_test.run_prefill_fp4(c, kd, ks, qdt)[live])
    with pytest.raises(ValueError, match="workspace"):
        fp4_test.run_prefill_fp4(
            c, kd, ks, qdt, staged=False, workspace=torch.empty(1, P, D, device="cuda")
        )


@requires_gfx950
def test_native_rejections():
    c, kd, ks = fp4_test.fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    f = native.score_prefill_flydsl_fp4_native
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd.view(torch.int8), ks, c["bt"], *args)
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd, ks[:, :, :2], c["bt"], *args)
    with pytest.raises(ValueError, match="block_table"):
        f(c["q_bf16"], kd, ks, c["bt"][:, :2], *args)
    with pytest.raises(ValueError):
        f(c["q_bf16"].half(), kd, ks, c["bt"], *args)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q_scale"):
            f(c["q8"], kd, ks, c["bt"], *args, q_scale=bad)
    bad_rows = torch.zeros(128, 1, D + 8, dtype=dtypes.fp8, device="cuda")[:, :, :D]
    with pytest.raises(ValueError, match="16-byte aligned"):
        f(bad_rows, kd, ks, c["bt"], *args)


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4_native.py"
)
@benchmark()
def test_native_perf(batch, q_len, ctx_len):
    c, kd, ks = fp4_test.fp4_prefill_case(batch, q_len, ctx_len)
    ref, live = fp8_test.aiter_scores(c)
    ret = {"gfx": get_gfx()}
    cands = {
        "fp8": lambda: fp8_test.run_prefill(c),
        "staged": lambda: fp4_test.run_prefill_fp4(c, kd, ks),
        "native": lambda: run_native(c, kd, ks),
        "native_q8": lambda: run_native(c, kd, ks, "e4m3"),
    }
    for name, run in cands.items():
        got, us = run_perftest(run, num_iters=10, num_rotate_args=1)
        fp8_test.assert_matches_aiter(got, ref, live, name)
        ret[f"{name} us"] = us
    return ret


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MXFP4 native prefill index score sweep")
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--q-len", type=int, nargs="*", default=[512, 4096])
    ap.add_argument("--ctx-len", type=int, nargs="*", default=[8192, 32768])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        aiter.logger.warning("requires gfx950; skipping sweep")
        return 0
    rows = [
        test_native_perf(b, q, c)
        for b, q, c in itertools.product(args.batch, args.q_len, args.ctx_len)
        if q <= c
    ]
    aiter.logger.info(
        "native mxfp4 prefill summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
