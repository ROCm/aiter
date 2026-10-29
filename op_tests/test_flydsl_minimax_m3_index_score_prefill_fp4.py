# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 index score with an MXFP4 index cache: prefill.

`score_prefill_flydsl_fp4` has two routes: the default reads the fp4 pages natively
(the fp8 prefill kernel built with `k_fp4`), `staged=True` dequantizes the referenced
pages to fp8 with a Triton kernel and runs the fp8 scorer on them. Both must match
aiter's `pa_sparse_block_score_prefill` and the fp8 FlyDSL kernel bit for bit on the
dequantized values, and each other; route-independent checks run on both.

Reference: aiter's `per_1x32_f4_quant` (RoundUp) quantizes, `fp4_utils.mxfp4_to_f32`
x `e8m0_to_f32` dequantizes, and aiter's `pa_sparse_block_score_prefill` (plus the fp8
FlyDSL kernel) scores the dequantized values as fp8 -- nothing from the kernels under
test. e8m0 bytes 0 and 255 are outside their contract and are not generated.

Run:
    python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4.py
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
from aiter.ops.flydsl.kernels import (
    minimax_m3_index_score_prefill_fp4 as prefill_kernel,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    score_prefill_flydsl,
)
from aiter.ops.msa_block_select import pa_sparse_block_score_prefill
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

try:
    from op_tests import test_flydsl_minimax_m3_index_score as fp8_test
except ImportError:  # run as a script from op_tests/
    import test_flydsl_minimax_m3_index_score as fp8_test

SENTINEL = fp8_test.SENTINEL
LOG2E = fp8_test.LOG2E
P, D = fp8_test.P, fp8_test.D
SUPPORTED_GFX = ["gfx950"]  # the FlyDSL scorers and the dequant are built for gfx950
requires_gfx950 = fp8_test.requires_gfx950
both_routes = pytest.mark.parametrize("staged", [True, False], ids=["staged", "native"])
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


@pytest.fixture(params=["bf16", "e4m3"], ids=["qbf16", "qe4m3"])
def qdt(request):
    """The dtype the query reaches the scorer in."""
    return request.param


# ---------------------------------------------------------------------------
# Prefill. Two independent judges on the SAME dequantized values: aiter's
# `pa_sparse_block_score_prefill` and the fp8 FlyDSL prefill scorer, both fed
# `dequantize(K).to(fp8)` (exact for scales in e4m3's range, which the cases keep
# to). Every route must match them bit for bit.
# ---------------------------------------------------------------------------


def fp4_prefill_case(batch, q_len, ctx_len, heads=1, seed=0):
    """A fp8 prefill case whose cache is re-quantized to MXFP4. Returns the
    case (cache = the dequantized values as fp8, for the judges) and the two
    MXFP4 views."""
    c = fp8_test.make_prefill_case(batch, q_len, ctx_len, heads, seed)
    kd, ks, deq = fp4_cache_of(c["cache"])
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    return dict(c, cache=as_fp8), kd, ks


def run_prefill_fp4(c, kd, ks, qdt="bf16", staged=None, **kw):
    """`qdt` picks the query handed to the scorer: the bf16 upcast or the e4m3 itself
    (the case's query came from e4m3, so both are the same numbers). `staged=None`
    leaves the route to the entry's default."""
    heads = c["q8"].shape[1]
    out = fp8_test.prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    if staged is not None:
        kw["staged"] = staged
    prefill_kernel.score_prefill_flydsl_fp4(
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


@requires_gfx950
@pytest.mark.parametrize(
    "shape", fp8_test.PREFILL_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_staged_matches_aiter_and_fp8_kernel(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_prefill_fp4(c, kd, ks, qdt, staged=True)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], fp8_test.run_prefill(c)[live])


@requires_gfx950
@both_routes
@pytest.mark.parametrize(
    "shape", fp8_test.GEOM_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_prefill_every_geometry(shape, qdt, staged):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_prefill_fp4(c, kd, ks, qdt, staged=staged)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")


@requires_gfx950
@both_routes
def test_prefill_two_heads(qdt, staged):
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = fp8_test.aiter_scores(c)
    got = run_prefill_fp4(c, kd, ks, qdt, staged=staged)
    fp8_test.assert_matches_aiter(got, ref, live, "heads=2")


@requires_gfx950
@pytest.mark.parametrize("ctx", [1, 127, 128, 129, 1000, 4097])
def test_staging_is_the_fp8_cast_bitwise(ctx):
    """Every staged page of every request equals `dequantize(K).to(fp8)` of the
    page its block table names; lengths off a page boundary included."""
    c, kd, ks = fp4_prefill_case(3, min(ctx, 128), ctx)
    staged = prefill_kernel.dequant_pages_fp8(kd, ks, c["bt"], c["seq"], c["mb"])
    want = c["cache"][c["bt"].long().reshape(-1)]  # [batch*mb, 128, 128]
    assert torch.equal(staged.view(torch.uint8), want.view(torch.uint8))


@requires_gfx950
def test_staging_leaves_dead_pages_alone_and_scores_ignore_them():
    """Pages past a request's length are not staged; poisoning the workspace
    with NaN bytes must not change a live score."""
    c, kd, ks = fp4_prefill_case(3, 128, 1500)
    c["seq"] = torch.tensor([1500, 300, 700], dtype=torch.int32, device="cuda")
    c["prefix"] = torch.tensor([1372, 172, 572], dtype=torch.int32, device="cuda")
    base = run_prefill_fp4(c, kd, ks, staged=True)
    ws = torch.full((3 * c["mb"], P, D), 0xFF, dtype=torch.uint8, device="cuda").view(
        dtypes.fp8
    )
    poisoned = run_prefill_fp4(c, kd, ks, staged=True, workspace=ws)
    live = base != SENTINEL
    assert torch.equal(base[live], poisoned[live])
    ref, ref_live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(base, ref, ref_live, "ragged lengths")


@requires_gfx950
@both_routes
def test_prefill_mixed_query_lengths_and_trailing_empty_request(qdt, staged):
    c, kd, ks, ref, live = mixed_lengths_case()
    got = run_prefill_fp4(c, kd, ks, qdt, staged=staged)
    fp8_test.assert_matches_aiter(got, ref, live, "mixed lengths + trailing empty")


@requires_gfx950
@both_routes
def test_prefill_e4m3_q_strided_single_region_and_scaled(staged):
    """A strided e4m3 query (rows and heads padded, as a packed qkv output would hand
    it over) and single-region pages (data and scale views of one [pages, 8704]
    buffer) score like the plain inputs, bit for bit, and `q_scale` multiplies the
    scores."""
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    rows, heads = c["q8"].shape[:2]
    plain = run_prefill_fp4(c, kd, ks, "e4m3", staged=staged)
    wide = torch.zeros(rows, heads + 1, D + 64, dtype=dtypes.fp8, device="cuda")
    view = wide[:, :heads, :D]  # head stride 192, row stride 576 (both 16 B multiples)
    view.copy_(c["q8"])
    assert view.stride() == ((heads + 1) * (D + 64), D + 64, 1)
    assert torch.equal(
        plain, run_prefill_fp4(dict(c, q8=view), kd, ks, "e4m3", staged=staged)
    )
    kd1, ks1 = single_region_views(kd, ks)
    assert torch.equal(plain, run_prefill_fp4(c, kd1, ks1, "e4m3", staged=staged))
    scaled = run_prefill_fp4(c, kd, ks, "e4m3", staged=staged, q_scale=0.5)
    live = plain != SENTINEL
    assert torch.equal(scaled[live], plain[live] * 0.5)


@requires_gfx950
@both_routes
def test_prefill_rejections(staged):
    c, kd, ks = fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)

    def f(q, kd, ks, bt, **kw):
        return prefill_kernel.score_prefill_flydsl_fp4(
            q, kd, ks, bt, *args, staged=staged, **kw
        )

    bad_rows = torch.zeros(128, 1, D + 8, dtype=dtypes.fp8, device="cuda")[:, :, :D]
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd.view(torch.int8), ks, c["bt"])
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd, ks[:, :, :2], c["bt"])
    with pytest.raises(ValueError, match="block_table"):
        f(c["q_bf16"], kd, ks, c["bt"][:, :2])
    with pytest.raises(ValueError):  # fp16 is neither bf16 nor e4m3
        f(c["q_bf16"].half(), kd, ks, c["bt"])
    with pytest.raises(ValueError, match="16-byte aligned"):
        f(bad_rows, kd, ks, c["bt"])
    # A strided `seq_lens` view must be refused before any launch reads it.
    seq_strided = torch.stack([c["seq"], c["seq"]], dim=1)[:, 0]
    assert seq_strided.stride() == (2,)
    with pytest.raises(ValueError, match="seq_lens"):
        prefill_kernel.score_prefill_flydsl_fp4(
            c["q_bf16"],
            kd,
            ks,
            c["bt"],
            c["cu"],
            seq_strided,
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            1.0 / LOG2E,
            staged=staged,
        )
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q_scale"):
            f(c["q8"], kd, ks, c["bt"], q_scale=bad)
    # `workspace` is the staged route's buffer: it must fit there, and is refused elsewhere.
    for ws in (
        torch.empty(1, P, D, dtype=dtypes.fp8, device="cuda"),
        torch.empty(c["mb"], P, D, dtype=torch.uint8, device="cuda"),
    ):
        with pytest.raises(ValueError, match="workspace"):
            f(c["q_bf16"], kd, ks, c["bt"], workspace=ws)


def run_native(c, kd, ks, qdt="bf16", **kw):
    """The fp8 prefill kernel's MXFP4 path, called directly (no route selection)."""
    heads = c["q8"].shape[1]
    out = fp8_test.prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        c["q_bf16"] if qdt == "bf16" else c["q8"],
        kd,
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
        k_scale=ks,
        **kw,
    )
    return out.reshape(heads, c["q8"].shape[0], c["mb"])


# The staged path's shapes, plus short-context / long-query ones and a multi-request
# chunk with a long prefix, which is where the native route is meant to matter.
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
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_native(c, kd, ks, qdt)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], run_prefill_fp4(c, kd, ks, qdt, staged=True)[live])


@requires_gfx950
@pytest.mark.parametrize("ctx", [1, 127, 128, 129, 1000, 4097])
def test_native_ragged_tails(ctx):
    """Lengths off a page boundary: the tail of the last page holds whatever the
    cache holds, and must not reach a live score."""
    c, kd, ks = fp4_prefill_case(3, min(ctx, 128), ctx)
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
    kd, ks, deq = quantize_mxfp4(k)
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
    c, kd, ks = fp4_prefill_case(3, 128, 1500)
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
    c, kd, ks = fp4_prefill_case(2, 300, 5000)
    default = run_prefill_fp4(c, kd, ks, qdt)
    live = default != SENTINEL
    assert torch.equal(default[live], run_native(c, kd, ks, qdt)[live])
    staged = run_prefill_fp4(c, kd, ks, qdt, staged=True)
    assert torch.equal(default[live], staged[live])


@requires_gfx950
@pytest.mark.parametrize("q_scale", [0.5, 3.0])
def test_native_q_scale(qdt, q_scale):
    """`q_scale` scales the scores (up to fp32 rounding of the factor folded into `sm_scale`)."""
    c, kd, ks = fp4_prefill_case(2, 256, 3000)
    plain = run_native(c, kd, ks, qdt)
    got = run_native(c, kd, ks, qdt, q_scale=q_scale)
    live = plain != SENTINEL
    err = checkAllclose(
        (plain[live] * q_scale).float(),
        got[live].float(),
        rtol=1e-6,
        atol=0,
        msg="q_scale",
    )
    assert err == 0, f"q_scale mismatch fraction {err}"


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
    c, kd, ks = fp4_prefill_case(batch, q_len, ctx_len)
    ref, live = fp8_test.aiter_scores(c)
    ret = {"gfx": get_gfx()}
    fp4_row = D // 2 + D // 32  # e2m1 data + e8m0 scales per token
    # name -> (run, K bytes per token, Q bytes per element). TB/s counts the traffic
    # each route needs at best; the staged route's extra fp8 round trip is not in it.
    # Every candidate scores into its own preallocated, sentinel-filled buffer, so the
    # timed region is the scorer alone (no allocation or fill launch).
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    outs = {
        n: fp8_test.prefill_alloc_score(rows, heads, c["mb"], "cuda")
        for n in ("fp8", "staged", "native", "native_q8")
    }
    for o in outs.values():
        o.fill_(SENTINEL)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    cands = {
        "fp8": (
            lambda: score_prefill_flydsl(
                c["q_bf16"], c["cache"], c["bt"], *args, out=outs["fp8"]
            ),
            D,
            2,
        ),
        "staged": (
            lambda: prefill_kernel.score_prefill_flydsl_fp4(
                c["q_bf16"], kd, ks, c["bt"], *args, out=outs["staged"], staged=True
            ),
            fp4_row,
            2,
        ),
        "native": (
            lambda: score_prefill_flydsl(
                c["q_bf16"], kd, c["bt"], *args, out=outs["native"], k_scale=ks
            ),
            fp4_row,
            2,
        ),
        "native_q8": (
            lambda: score_prefill_flydsl(
                c["q8"], kd, c["bt"], *args, out=outs["native_q8"], k_scale=ks
            ),
            fp4_row,
            1,
        ),
    }
    pairs, _ = fp8_test._prefill_work(batch, q_len, ctx_len)
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


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4.py"
)
@benchmark()
def test_dequant_staging_perf(batch, ctx_len):
    """The staged route's dequant launch alone (the native route has no such step;
    the end-to-end comparison is the native test file's sweep)."""
    c, kd, ks = fp4_prefill_case(batch, min(ctx_len, 128), ctx_len)
    want = c["cache"][c["bt"].long().reshape(-1)]
    staged, us = run_perftest(
        lambda: prefill_kernel.dequant_pages_fp8(kd, ks, c["bt"], c["seq"], c["mb"]),
        num_iters=10,
        num_rotate_args=1,
    )
    err = checkAllclose(
        want.float(), staged.float(), rtol=0, atol=0, msg="dequant staging"
    )
    assert err == 0, f"staging mismatch fraction {err}"
    # e2m1 + e8m0 read and fp8 written per token; pure data movement, no FLOPs.
    nbytes = batch * c["mb"] * P * (D // 2 + D // 32 + D)
    return {
        "gfx": get_gfx(),
        "dequant us": us,
        "dequant TB/s": nbytes / us / 1e6,
        "dequant err": err,
    }


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MXFP4 prefill index score sweep")
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--q-len", type=int, nargs="*", default=[512, 4096])
    ap.add_argument("--ctx-len", type=int, nargs="*", default=[8192, 32768])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("requires %s; skipping", SUPPORTED_GFX)
        return 0
    aiter.logger.info("mxfp4 staged prefill: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "mxfp4 staged prefill: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1
    rows = [
        test_native_perf(b, q, c)
        for b, q, c in itertools.product(args.batch, args.q_len, args.ctx_len)
        if q <= c
    ]
    aiter.logger.info(
        "mxfp4 prefill summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    # A second table: the two @benchmark signatures differ.
    rows = [
        test_dequant_staging_perf(b, c)
        for b, c in itertools.product(args.batch, args.ctx_len)
    ]
    aiter.logger.info(
        "mxfp4 dequant staging summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
