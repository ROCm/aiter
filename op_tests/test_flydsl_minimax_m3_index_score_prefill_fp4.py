# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 index score with an MXFP4 index cache: staged prefill route.

The default (native) route has its own file,
test_flydsl_minimax_m3_index_score_prefill_fp4_native.py, which reuses the cache
builders and fixtures defined here. The tests here pin `staged=True`: a Triton
kernel dequantizes the referenced pages to fp8, then the fp8 prefill scorer runs.

The quantizer is aiter's own `per_1x32_f4_quant` (RoundUp), the dequantizer
`fp4_utils.mxfp4_to_f32` x `e8m0_to_f32`. e8m0 bytes 0 and 255 are outside the
kernels' contract and are not generated (`quantize_mxfp4` asserts it).
"""

import functools

import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.kernels import (
    minimax_m3_index_score_prefill_fp4 as prefill_kernel,
)
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility import fp4_utils

try:
    from op_tests import test_flydsl_minimax_m3_index_score as fp8_test
except ImportError:  # run as a script from op_tests/
    import test_flydsl_minimax_m3_index_score as fp8_test

# These tests pin the STAGED prefill path (dequantize, then the fp8 kernel); the
# default native path has its own file, test_..._prefill_fp4_native.py.
staged_prefill = functools.partial(prefill_kernel.score_prefill_flydsl_fp4, staged=True)
P, D = fp8_test.P, fp8_test.D
requires_gfx950 = fp8_test.requires_gfx950

# ---------------------------------------------------------------------------
# Reference quantizer
# ---------------------------------------------------------------------------
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


def fp4_cache_of(cache8, transform=None):
    """An MXFP4 version of an fp8 test cache: the same values, re-quantized."""
    pages = cache8.shape[0]
    k = torch.empty((pages, P, D), dtype=torch.bfloat16, device=cache8.device)
    for i in range(0, pages, 2048):
        blk = cache8[i : i + 2048].to(torch.bfloat16)
        k[i : i + 2048] = transform(blk) if transform else blk
    return quantize_mxfp4(k)


def single_region_views(data, scale):
    """Both views of ONE `[pages, 8704]` allocation: 8192 B of data followed by
    512 B of scale per page, the physical layout the cache is meant to have."""
    pages = data.shape[0]
    buf = torch.empty((pages, P * 64 + P * 4), dtype=torch.uint8, device=data.device)
    buf[:, : P * 64] = data.reshape(pages, -1)
    buf[:, P * 64 :] = scale.reshape(pages, -1)
    kd = torch.as_strided(buf, (pages, P, 64), (buf.stride(0), 64, 1))
    ks = torch.as_strided(buf, (pages, P, 4), (buf.stride(0), 4, 1), P * 64)
    return buf, kd, ks


@pytest.fixture(params=["bf16", "e4m3"], ids=["qbf16", "qe4m3"])
def qdt(request):
    """The dtype the query reaches the scorer in."""
    return request.param


# ---------------------------------------------------------------------------
# Prefill: dequantize to fp8 staging, then the unchanged fp8 prefill scorer
#
# Two independent judges on the SAME dequantized values: aiter's own
# `pa_sparse_block_score_prefill` and the fp8 FlyDSL prefill scorer, both fed
# `dequantize(K).to(fp8)` (exact for scales in e4m3's range, which the cases
# keep to). The fp4 route must match both bit for bit.
# ---------------------------------------------------------------------------
SENTINEL = fp8_test.SENTINEL
LOG2E = fp8_test.LOG2E


def fp4_prefill_case(batch, q_len, ctx_len, heads=1, seed=0):
    """A fp8 prefill case whose cache is re-quantized to MXFP4. Returns the
    case (cache = the dequantized values as fp8, for the judges) and the two
    MXFP4 views."""
    c = fp8_test.make_prefill_case(batch, q_len, ctx_len, heads, seed)
    kd, ks, deq = fp4_cache_of(c["cache"])
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    return dict(c, cache=as_fp8), kd, ks


def run_prefill_fp4(c, kd, ks, qdt="bf16", **kw):
    """`qdt` picks the query handed to the scorer: the bf16 upcast or the e4m3 itself
    (the case's query came from e4m3, so both are the same numbers)."""
    heads = c["q8"].shape[1]
    out = fp8_test.prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    kw.setdefault("staged", True)
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


@requires_gfx950
@pytest.mark.parametrize(
    "shape", fp8_test.PREFILL_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_prefill_matches_aiter_and_fp8_kernel(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_prefill_fp4(c, kd, ks, qdt)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], fp8_test.run_prefill(c)[live])


@requires_gfx950
@pytest.mark.parametrize(
    "shape", fp8_test.GEOM_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_prefill_every_geometry(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(
        run_prefill_fp4(c, kd, ks, qdt), ref, live, f"{shape}"
    )


@requires_gfx950
def test_prefill_two_heads(qdt):
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_prefill_fp4(c, kd, ks, qdt), ref, live, "heads=2")


@requires_gfx950
@pytest.mark.parametrize("ctx", [1, 127, 128, 129, 1000, 4097])
def test_staging_is_the_fp8_cast_bitwise(ctx):
    """Every staged page of every request equals `dequantize(K).to(fp8)` of the
    page its block table names; lengths off a page boundary included."""
    batch = 3
    c, kd, ks = fp4_prefill_case(batch, min(ctx, 128), ctx)
    mb = c["mb"]
    staged = prefill_kernel.dequant_pages_fp8(kd, ks, c["bt"], c["seq"], mb)
    want = c["cache"][c["bt"].long().reshape(-1)]  # [batch*mb, 128, 128]
    assert torch.equal(staged.view(torch.uint8), want.view(torch.uint8))


@requires_gfx950
def test_staging_leaves_dead_pages_alone_and_scores_ignore_them():
    """Pages past a request's length are not staged; poisoning the workspace
    with NaN bytes must not change a live score."""
    c, kd, ks = fp4_prefill_case(3, 128, 1500)
    c["seq"] = torch.tensor([1500, 300, 700], dtype=torch.int32, device="cuda")
    c["prefix"] = torch.tensor([1372, 172, 572], dtype=torch.int32, device="cuda")
    base = run_prefill_fp4(c, kd, ks)
    ws = torch.full((3 * c["mb"], P, D), 0xFF, dtype=torch.uint8, device="cuda").view(
        dtypes.fp8
    )
    poisoned = run_prefill_fp4(c, kd, ks, workspace=ws)
    live = base != SENTINEL
    assert torch.equal(base[live], poisoned[live])
    ref, ref_live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(base, ref, ref_live, "ragged lengths")


@requires_gfx950
def test_prefill_mixed_query_lengths_and_trailing_empty_request(qdt):
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
    cache8 = (torch.randn(3 * mb, P, D, device=dev) / 4).to(dtypes.fp8)
    kd, ks, deq = fp4_cache_of(cache8)
    as_fp8 = deq.to(dtypes.fp8)
    bt = torch.randperm(3 * mb, device=dev, dtype=torch.int32).view(3, mb).contiguous()
    ref = torch.full((1, q_a + 1, mb), -float("inf"), device=dev)
    # aiter sees the first two requests; the third is the trailing empty one.
    from aiter.ops.msa_block_select import pa_sparse_block_score_prefill

    pa_sparse_block_score_prefill(
        q8, as_fp8, ref, bt[:2], cu[:3], seq[:2], max_query_len=q_a, max_seq_len=ctx_b
    )
    live = torch.isfinite(ref)
    out = fp8_test.prefill_alloc_score(q_a + 1, 1, mb, dev)
    out.fill_(SENTINEL)
    staged_prefill(
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
        out.reshape(1, q_a + 1, mb), ref, live, "mixed lengths + trailing empty"
    )


@requires_gfx950
def test_prefill_e4m3_q_strided_and_scaled():
    """An e4m3 query that is a strided view (rows and heads padded, as a packed qkv
    output would hand it over) scores like its contiguous copy, bit for bit, and
    `q_scale` multiplies the scores."""
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    rows, heads = c["q8"].shape[:2]
    wide = torch.zeros(rows, heads + 1, D + 64, dtype=dtypes.fp8, device="cuda")
    view = wide[:, :heads, :D]  # head stride 192, row stride 576 (both 16 B multiples)
    view.copy_(c["q8"])
    assert view.stride() == ((heads + 1) * (D + 64), D + 64, 1)
    plain = run_prefill_fp4(c, kd, ks, "e4m3")
    strided = run_prefill_fp4(dict(c, q8=view), kd, ks, "e4m3")
    assert torch.equal(plain, strided)
    out = fp8_test.prefill_alloc_score(rows, heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    staged_prefill(c["q8"], kd, ks, c["bt"], *args, out=out, q_scale=0.5)
    got = out.reshape(plain.shape)
    live = plain != SENTINEL
    assert torch.equal(got[live], plain[live] * 0.5)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q_scale"):
            staged_prefill(c["q8"], kd, ks, c["bt"], *args, q_scale=bad)


@requires_gfx950
def test_prefill_e4m3_q_rejections():
    c, kd, ks = fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    f = staged_prefill
    bad_rows = torch.zeros(128, 1, D + 8, dtype=dtypes.fp8, device="cuda")[:, :, :D]
    with pytest.raises(ValueError, match="16-byte aligned"):
        f(bad_rows, kd, ks, c["bt"], *args)
    with pytest.raises(ValueError):  # fp16 is neither bf16 nor e4m3
        f(c["q_bf16"].half(), kd, ks, c["bt"], *args)


@requires_gfx950
def test_prefill_rejections():
    c, kd, ks = fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    f = staged_prefill
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd.view(torch.int8), ks, c["bt"], *args)
    with pytest.raises(ValueError):
        f(c["q_bf16"], kd, ks[:, :, :2], c["bt"], *args)
    with pytest.raises(ValueError, match="block_table"):
        f(c["q_bf16"], kd, ks, c["bt"][:, :2], *args)
    with pytest.raises(ValueError, match="workspace"):
        f(
            c["q_bf16"],
            kd,
            ks,
            c["bt"],
            *args,
            workspace=torch.empty(1, P, D, dtype=dtypes.fp8, device="cuda"),
        )
    with pytest.raises(ValueError, match="workspace"):
        f(
            c["q_bf16"],
            kd,
            ks,
            c["bt"],
            *args,
            workspace=torch.empty(c["mb"], P, D, dtype=torch.uint8, device="cuda"),
        )


def test_export_resolves():
    import aiter.ops.flydsl as F

    assert "minimax_m3_index_score_prefill_fp4_flydsl" in F.__all__
    assert (
        F.minimax_m3_index_score_prefill_fp4_flydsl
        is prefill_kernel.score_prefill_flydsl_fp4
    )
