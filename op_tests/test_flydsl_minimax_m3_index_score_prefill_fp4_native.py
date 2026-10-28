# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index score reading an MXFP4 cache natively (no staging).

The native kernel is the staged path's fp8 scorer with only its K side replaced, so
it must give the staged path's scores bit for bit -- and, as the staged path is
itself judged bit for bit against aiter's `pa_sparse_block_score_prefill` on the
dequantized fp8 values, so is this one. Both judges run here, for both query dtypes.
The route-independent checks (geometries, heads, mixed lengths, strided views,
rejections) are parametrized over both routes in
test_flydsl_minimax_m3_index_score_prefill_fp4.py.

Run:
    python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4_native.py
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

try:
    from op_tests import test_flydsl_minimax_m3_index_score_prefill_fp4 as fp4_test
except ImportError:  # run as a script from op_tests/
    import test_flydsl_minimax_m3_index_score_prefill_fp4 as fp4_test

fp8_test = fp4_test.fp8_test
requires_gfx950 = fp4_test.requires_gfx950
SENTINEL, LOG2E, P, D = fp4_test.SENTINEL, fp4_test.LOG2E, fp4_test.P, fp4_test.D
qdt = fp4_test.qdt  # fixture, collected here


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
    assert torch.equal(
        got[live], fp4_test.run_prefill_fp4(c, kd, ks, qdt, staged=True)[live]
    )


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
def test_default_entry_is_native_and_staged_is_opt_in(qdt):
    """`score_prefill_flydsl_fp4` reads the cache natively unless `staged=True`; the
    two routes agree."""
    c, kd, ks = fp4_test.fp4_prefill_case(2, 300, 5000)
    default = fp4_test.run_prefill_fp4(c, kd, ks, qdt)
    live = default != SENTINEL
    assert torch.equal(default[live], run_native(c, kd, ks, qdt)[live])
    staged = fp4_test.run_prefill_fp4(c, kd, ks, qdt, staged=True)
    assert torch.equal(default[live], staged[live])


def _prefill_bytes(batch, q_len, ctx_len, k_row_bytes, q_elem_bytes):
    pairs, pages = fp8_test._prefill_work(batch, q_len, ctx_len)
    return (
        pages * P * k_row_bytes  # K, read once at best
        + batch * q_len * D * q_elem_bytes  # Q
        + pairs * 4  # the fp32 scores written
    )


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4_native.py"
)
@benchmark()
def test_native_perf(batch, q_len, ctx_len):
    c, kd, ks = fp4_test.fp4_prefill_case(batch, q_len, ctx_len)
    ref, live = fp8_test.aiter_scores(c)
    ret = {"gfx": get_gfx()}
    fp4_row = D // 2 + D // 32  # e2m1 data + e8m0 scales per token
    # name -> (run, K bytes per token, Q bytes per element). TB/s counts the traffic
    # each route needs at best; the staged route's extra fp8 round trip is not in it.
    cands = {
        "fp8": (lambda: fp8_test.run_prefill(c), D, 2),
        "staged": (
            lambda: fp4_test.run_prefill_fp4(c, kd, ks, staged=True),
            fp4_row,
            2,
        ),
        "native": (lambda: run_native(c, kd, ks), fp4_row, 2),
        "native_q8": (lambda: run_native(c, kd, ks, "e4m3"), fp4_row, 1),
    }
    pairs, _ = fp8_test._prefill_work(batch, q_len, ctx_len)
    flops = 2 * pairs * P * D
    for name, (run, k_row, q_elem) in cands.items():
        got, us = run_perftest(run, num_iters=10, num_rotate_args=1)
        err = checkAllclose(
            ref[live].float(), got[live].float(), rtol=0, atol=0, msg=f"{name}: prefill"
        )
        assert err == 0, f"{name}: mismatch fraction {err} against aiter"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = (
            _prefill_bytes(batch, q_len, ctx_len, k_row, q_elem) / us / 1e6
        )
        ret[f"{name} err"] = err
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
        aiter.logger.warning("requires gfx950; skipping")
        return 0
    aiter.logger.info("mxfp4 native prefill: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "mxfp4 native prefill: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1
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
