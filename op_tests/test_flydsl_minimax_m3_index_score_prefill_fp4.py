# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index score over an MXFP4 index cache.

`score_prefill_flydsl_fp4` is the fp8 prefill kernel built with `k_fp4`: the fp4
pages stay MXFP4 through LDS and feed the scaled MFMA. It must match aiter's
`pa_sparse_block_score_prefill` and the fp8 FlyDSL kernel bit for bit on the
dequantized values.

Reference: aiter's `per_1x32_f4_quant` (RoundUp) quantizes, `fp4_utils.mxfp4_to_f32`
x `e8m0_to_f32` dequantizes, and aiter's `pa_sparse_block_score_prefill` (plus the fp8
FlyDSL kernel) scores the dequantized values as fp8 -- nothing from the kernel under
test. e8m0 bytes 0 and 255 are outside its contract and are not generated.

Run (exits nonzero if any check fails):
    python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4.py [--case SUBSTR ...] [--no-perf]
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    score_prefill_flydsl,
    score_prefill_flydsl_fp4,
)
from aiter.ops.msa_block_select import pa_sparse_block_score_prefill
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils
from op_tests.test_flydsl_minimax_m3_index_score import (
    GEOM_SHAPES,
    LOG2E,
    PREFILL_SHAPES,
    SENTINEL,
    D,
    P,
    _prefill_work,
    aiter_scores,
    assert_matches_aiter,
    make_prefill_case,
    prefill_alloc_score,
    run_prefill,
)

SUPPORTED_GFX = ["gfx950"]  # the FlyDSL scorers are built for gfx950
QCHUNK = 512  # pages quantized per torch call; the reference makes fp32 temporaries


# ---------------------------------------------------------------------------
# Reference quantizer
# ---------------------------------------------------------------------------
def quantize_mxfp4(k):
    """bf16 [pages,128,128] -> (data u8 [pages,128,64], scale u8 [pages,128,4],
    dequantized bf16 [pages,128,128]).

    Low nibble is the even element; one e8m0 byte per 32 elements; the
    dequantized tensor is what the kernel is contractually scoring.
    """
    pages = k.shape[0]
    data = torch.empty((pages, P, D // 2), dtype=torch.uint8, device=k.device)
    scale = torch.empty((pages, P, D // 32), dtype=torch.uint8, device=k.device)
    deq = torch.empty_like(k)
    for i in range(0, pages, QCHUNK):
        x = k[i : i + QCHUNK].reshape(-1, D)
        n = x.shape[0] // P
        q, s = per_1x32_f4_quant(x)
        d = q.view(torch.uint8).reshape(n, P, D // 2)
        sc = s.view(torch.uint8).reshape(n, P, D // 32)
        f = fp4_utils.mxfp4_to_f32(d.reshape(-1, D // 2).clone())
        e = fp4_utils.e8m0_to_f32(sc.reshape(-1, D // 32)).repeat_interleave(32, dim=-1)
        data[i : i + n] = d
        scale[i : i + n] = sc
        deq[i : i + n] = (f * e).to(torch.bfloat16).reshape(n, P, D)
    assert int(scale.min()) >= 1 and int(scale.max()) <= 254
    return data, scale, deq


def fp4_cache_of(cache8):
    """An MXFP4 version of an fp8 test cache: the same values, re-quantized."""
    return quantize_mxfp4(cache8.to(torch.bfloat16))


def single_region_views(data, scale):
    """Both views of ONE `[pages, 8704]` allocation: 8192 B of data followed by
    512 B of scale per page, the physical layout the cache is meant to have."""
    pages = data.shape[0]
    buf = torch.empty((pages, P * 64 + P * 4), dtype=torch.uint8, device=data.device)
    buf[:, : P * 64] = data.reshape(pages, -1)
    buf[:, P * 64 :] = scale.reshape(pages, -1)
    kd = torch.as_strided(buf, (pages, P, 64), (buf.stride(0), 64, 1))
    ks = torch.as_strided(buf, (pages, P, 4), (buf.stride(0), 4, 1), P * 64)
    return kd, ks


# ---------------------------------------------------------------------------
# Prefill. Two independent judges on the SAME dequantized values: aiter's
# `pa_sparse_block_score_prefill` and the fp8 FlyDSL prefill scorer, both fed
# `dequantize(K).to(fp8)` (exact for scales in e4m3's range, which the cases keep
# to). The MXFP4 path must match them bit for bit.
# ---------------------------------------------------------------------------


def fp4_prefill_case(batch, q_len, ctx_len, heads=1, seed=0):
    """A fp8 prefill case whose cache is re-quantized to MXFP4. Returns the
    case (cache = the dequantized values as fp8, for the judges) and the two
    MXFP4 views."""
    c = make_prefill_case(batch, q_len, ctx_len, heads, seed)
    kd, ks, deq = fp4_cache_of(c["cache"])
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    return dict(c, cache=as_fp8), kd, ks


def run_fp4(c, kd, ks, qdt="bf16", out=None):
    """`qdt` picks the query handed to the scorer: the bf16 upcast or the e4m3 itself
    (the case's query came from e4m3, so both are the same numbers)."""
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    if out is None:
        out = prefill_alloc_score(rows, heads, c["mb"], "cuda")
        out.fill_(SENTINEL)
    score_prefill_flydsl_fp4(
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
    )
    return out.reshape(heads, rows, c["mb"])


def mixed_lengths_case():
    """A 1024-row request, a 1-row request on a 16K context and a TRAILING empty
    one. aiter judges the first two. Returns the case, the views, aiter's scores
    and the live mask."""
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
    kd, ks, deq = fp4_cache_of(
        (torch.randn(3 * mb, P, D, device=dev) / 4).to(dtypes.fp8)
    )
    bt = torch.randperm(3 * mb, device=dev, dtype=torch.int32).view(3, mb).contiguous()
    ref = torch.full((1, q_a + 1, mb), -float("inf"), device=dev)
    pa_sparse_block_score_prefill(
        q8,
        deq.to(dtypes.fp8),
        ref,
        bt[:2],
        cu[:3],
        seq[:2],
        max_query_len=q_a,
        max_seq_len=ctx_b,
    )
    c = {
        "q8": q8,
        "q_bf16": q8.float().bfloat16(),
        "bt": bt,
        "cu": cu,
        "seq": seq,
        "prefix": prefix,
        "max_q": q_a,
        "max_seq": ctx_b,
        "mb": mb,
    }
    return c, kd, ks, ref, torch.isfinite(ref)


# The fp8 cases' shapes, plus short-context / long-query ones and a multi-request
# chunk with a long prefix.
SHAPES = PREFILL_SHAPES + [
    (1, 8192, 8192),
    (1, 7000, 7000),
    (2, 4096, 20000),
    (4, 512, 8192),
    (3, 300, 33333),
]


def check_prefill_matches_aiter_and_fp8_kernel(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = aiter_scores(c)
    got = run_fp4(c, kd, ks, qdt)
    assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], run_prefill(c)[live])


def check_prefill_every_geometry(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_fp4(c, kd, ks, qdt), ref, live, f"{shape}")


def check_prefill_two_heads(qdt):
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_fp4(c, kd, ks, qdt), ref, live, "heads=2")


def check_prefill_ragged_tails(ctx):
    """Lengths off a page boundary: the tail of the last page holds whatever the
    cache holds, and must not reach a live score."""
    c, kd, ks = fp4_prefill_case(3, min(ctx, 128), ctx)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_fp4(c, kd, ks), ref, live, f"ctx={ctx}")


def check_prefill_scale_extremes(qdt):
    """Per-token magnitudes spread over 2^-3 .. 2^3, so the e8m0 bytes walk across
    the window e4m3 can still hold exactly: the scale really is applied per lane
    group and per token, not once."""
    c = make_prefill_case(2, 256, 3000)
    span = torch.randint(-3, 4, (c["cache"].shape[0], P, 1), device="cuda")
    k = c["cache"].to(torch.bfloat16) * (2.0 ** span.to(torch.bfloat16))
    kd, ks, deq = quantize_mxfp4(k)
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    assert ks.min() < ks.max() - 6
    c = dict(c, cache=as_fp8)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_fp4(c, kd, ks, qdt), ref, live, "scales")


def check_prefill_ragged_batch_and_garbage_tails(qdt):
    """Unequal lengths, and the bytes past each request's end overwritten with 0xFF
    (e8m0 255 and NaN-ish nibbles): live scores do not move."""
    c, kd, ks = fp4_prefill_case(3, 128, 1500)
    c["seq"] = torch.tensor([1500, 300, 700], dtype=torch.int32, device="cuda")
    c["prefix"] = torch.tensor([1372, 172, 572], dtype=torch.int32, device="cuda")
    base = run_fp4(c, kd, ks, qdt)
    kd2, ks2 = kd.clone(), ks.clone()
    for b, n in enumerate([1500, 300, 700]):
        for p in range(-(-n // P)):
            page = int(c["bt"][b, p])
            first_dead = n - p * P
            if first_dead < P:
                kd2[page, first_dead:] = 0xFF
                ks2[page, first_dead:] = 0xFF
    poisoned = run_fp4(c, kd2, ks2, qdt)
    live = base != SENTINEL
    assert torch.equal(base[live], poisoned[live])
    ref, ref_live = aiter_scores(c)
    assert_matches_aiter(base, ref, ref_live, "ragged lengths")


def check_prefill_mixed_query_lengths_and_trailing_empty_request(qdt):
    c, kd, ks, ref, live = mixed_lengths_case()
    got = run_fp4(c, kd, ks, qdt)
    assert_matches_aiter(got, ref, live, "mixed lengths + trailing empty")


def check_prefill_e4m3_q_strided_and_single_region():
    """A strided e4m3 query (rows and heads padded, as a packed qkv output would hand
    it over) and single-region pages (data and scale views of one [pages, 8704]
    buffer) score like the plain inputs, bit for bit."""
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    rows, heads = c["q8"].shape[:2]
    plain = run_fp4(c, kd, ks, "e4m3")
    wide = torch.zeros(rows, heads + 1, D + 64, dtype=dtypes.fp8, device="cuda")
    view = wide[:, :heads, :D]  # head stride 192, row stride 576 (both 16 B multiples)
    view.copy_(c["q8"])
    assert view.stride() == ((heads + 1) * (D + 64), D + 64, 1)
    assert torch.equal(plain, run_fp4(dict(c, q8=view), kd, ks, "e4m3"))
    kd1, ks1 = single_region_views(kd, ks)
    assert torch.equal(plain, run_fp4(c, kd1, ks1, "e4m3"))


def expect_value_error(what, fn, *args, match=None):
    """`fn(*args)` must raise ValueError (whose message contains `match`)."""
    try:
        fn(*args)
    except ValueError as e:
        assert match is None or match in str(e), f"{what}: wrong message: {e}"
        return
    raise AssertionError(f"{what}: no ValueError")


def check_prefill_rejections():
    c, kd, ks = fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    f = score_prefill_flydsl_fp4
    bad_rows = torch.zeros(128, 1, D + 8, dtype=dtypes.fp8, device="cuda")[:, :, :D]
    expect_value_error(
        "int8 k_data", f, c["q_bf16"], kd.view(torch.int8), ks, c["bt"], *args
    )
    expect_value_error(
        "narrow k_scale", f, c["q_bf16"], kd, ks[:, :, :2], c["bt"], *args
    )
    expect_value_error(
        "narrow block_table",
        f,
        c["q_bf16"],
        kd,
        ks,
        c["bt"][:, :2],
        *args,
        match="block_table",
    )
    # fp16 is neither bf16 nor e4m3
    expect_value_error("fp16 q", f, c["q_bf16"].half(), kd, ks, c["bt"], *args)
    expect_value_error(
        "misaligned q rows",
        f,
        bad_rows,
        kd,
        ks,
        c["bt"],
        *args,
        match="16-byte aligned",
    )
    seq_strided = torch.stack([c["seq"], c["seq"]], dim=1)[:, 0]
    assert seq_strided.stride() == (2,)
    expect_value_error(
        "strided seq_lens",
        f,
        c["q_bf16"],
        kd,
        ks,
        c["bt"],
        c["cu"],
        seq_strided,
        *args[2:],
        match="seq_lens",
    )


def _prefill_bytes(batch, q_len, ctx_len, k_row_bytes, q_elem_bytes):
    pairs, pages = _prefill_work(batch, q_len, ctx_len)
    return (
        pages * P * k_row_bytes  # K, read once at best
        + batch * q_len * D * q_elem_bytes  # Q
        + pairs * 4  # the fp32 scores written
    )


@benchmark()
def bench_prefill_fp4(batch, q_len, ctx_len):
    c, kd, ks = fp4_prefill_case(batch, q_len, ctx_len)
    ref, live = aiter_scores(c)
    ret = {"gfx": get_gfx()}
    fp4_row = D // 2 + D // 32  # e2m1 data + e8m0 scales per token
    # Every candidate scores into its own preallocated, sentinel-filled buffer, so the
    # timed region is the scorer alone (no allocation or fill launch).
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    outs = {
        n: prefill_alloc_score(rows, heads, c["mb"], "cuda")
        for n in ("fp8", "fp4", "fp4_q8")
    }
    for o in outs.values():
        o.fill_(SENTINEL)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    # name -> (run, K bytes per token, Q bytes per element)
    cands = {
        "fp8": (
            lambda: score_prefill_flydsl(
                c["q_bf16"], c["cache"], c["bt"], *args, out=outs["fp8"]
            ),
            D,
            2,
        ),
        "fp4": (
            lambda: score_prefill_flydsl_fp4(
                c["q_bf16"], kd, ks, c["bt"], *args, out=outs["fp4"]
            ),
            fp4_row,
            2,
        ),
        "fp4_q8": (
            lambda: score_prefill_flydsl_fp4(
                c["q8"], kd, ks, c["bt"], *args, out=outs["fp4_q8"]
            ),
            fp4_row,
            1,
        ),
    }
    pairs, _ = _prefill_work(batch, q_len, ctx_len)
    flops = 2 * pairs * P * D
    for name, (run, k_row, q_elem) in cands.items():
        got, us = run_perftest(run, num_iters=10, num_rotate_args=1)
        assert got.data_ptr() == outs[name].data_ptr()
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


QDTS = ["bf16", "e4m3"]  # the dtype the query reaches the scorer in


def build_cases():
    """(name, check, args): one row per case; the same set the pytest matrix had."""
    cases = []
    for shape, qdt in itertools.product(SHAPES, QDTS):
        name = f"b{shape[0]}_q{shape[1]}_ctx{shape[2]}_{qdt}"
        cases.append(
            (
                f"matches_aiter_and_fp8_kernel_{name}",
                check_prefill_matches_aiter_and_fp8_kernel,
                (shape, qdt),
            )
        )
    for shape, qdt in itertools.product(GEOM_SHAPES, QDTS):
        name = f"b{shape[0]}_q{shape[1]}_ctx{shape[2]}_{qdt}"
        cases.append(
            (f"every_geometry_{name}", check_prefill_every_geometry, (shape, qdt))
        )
    for qdt in QDTS:
        cases.append((f"two_heads_{qdt}", check_prefill_two_heads, (qdt,)))
    for ctx in [1, 127, 128, 129, 1000, 4097]:
        cases.append((f"ragged_tails_ctx{ctx}", check_prefill_ragged_tails, (ctx,)))
    for qdt in QDTS:
        cases.append((f"scale_extremes_{qdt}", check_prefill_scale_extremes, (qdt,)))
        cases.append(
            (
                f"ragged_batch_and_garbage_tails_{qdt}",
                check_prefill_ragged_batch_and_garbage_tails,
                (qdt,),
            )
        )
        cases.append(
            (
                f"mixed_query_lengths_trailing_empty_{qdt}",
                check_prefill_mixed_query_lengths_and_trailing_empty_request,
                (qdt,),
            )
        )
    cases.append(
        (
            "e4m3_q_strided_and_single_region",
            check_prefill_e4m3_q_strided_and_single_region,
            (),
        )
    )
    cases.append(("rejections", check_prefill_rejections, ()))
    return cases


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MXFP4 prefill index score checks + sweep")
    ap.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="run only the correctness cases whose name contains one of these",
    )
    ap.add_argument(
        "--no-perf", action="store_true", help="correctness cases only, no perf table"
    )
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--q-len", type=int, nargs="*", default=[512, 4096])
    ap.add_argument("--ctx-len", type=int, nargs="*", default=[8192, 32768])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("requires %s; skipping", SUPPORTED_GFX)
        return 0
    aiter.logger.info("mxfp4 prefill: running correctness checks...")
    rows = []
    for name, check, cargs in build_cases():
        if args.case is not None and not any(k in name for k in args.case):
            continue
        try:
            check(*cargs)
            rows.append({"case": name, "status": "PASS", "detail": ""})
        except (AssertionError, RuntimeError, ValueError) as e:  # list every failure
            rows.append({"case": name, "status": "FAIL", "detail": repr(e)[:200]})
    if rows:
        aiter.logger.info(
            "mxfp4 prefill checks (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    else:
        aiter.logger.warning("Empty selection; no checks executed")
    failed = [r["case"] for r in rows if r["status"] != "PASS"]
    if failed:
        aiter.logger.error(
            "mxfp4 prefill: %d of %d checks failed: %s", len(failed), len(rows), failed
        )
        return 1
    if args.no_perf:
        return 0
    perf = [
        bench_prefill_fp4(b, q, c)
        for b, q, c in itertools.product(args.batch, args.q_len, args.ctx_len)
        if q <= c
    ]
    aiter.logger.info(
        "mxfp4 prefill summary (markdown):\n%s",
        pd.DataFrame(perf).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
