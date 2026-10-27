# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 index score with an MXFP4 index cache: decode kernel.

The judge is the fp8 scorer's own oracle (`score_oracle` in the fp8 test): it
lifts K to the query dtype and accumulates in fp32. The MXFP4 cache is
dequantized to bf16 first, which is exact (e2m1 x 2^e fits bf16), so the
tolerance is the fp8 test's and a mismatch is a bug, not a rounding difference.

The quantizer is aiter's own `per_1x32_f4_quant` (RoundUp), the dequantizer
`fp4_utils.mxfp4_to_f32` x `e8m0_to_f32`; neither is reimplemented here beyond
chunking. e8m0 bytes 0 and 255 are outside the kernel's contract and are not
generated (`quantize_mxfp4` clamps amax, as `indexer_fp4_scale_e8m0` does).
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
from aiter.ops.flydsl.kernels import minimax_m3_index_score as fp8_kernel
from aiter.ops.flydsl.kernels import minimax_m3_index_score_fp4 as kernel
from aiter.ops.flydsl.kernels import (
    minimax_m3_index_score_prefill_fp4 as prefill_kernel,
)
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

try:
    from op_tests import test_flydsl_minimax_m3_index_score as fp8_test
except ImportError:  # run as a script from op_tests/
    import test_flydsl_minimax_m3_index_score as fp8_test

P, D = fp8_test.P, fp8_test.D
ATOL, RTOL = fp8_test.ATOL, fp8_test.RTOL
requires_gfx950 = fp8_test.requires_gfx950
DECODE_SUPPORTED_GFX = fp8_test.DECODE_SUPPORTED_GFX
score_oracle, compare, check_scores = (
    fp8_test.score_oracle,
    fp8_test.compare,
    fp8_test.check_scores,
)
make_case, GEOM_POINTS, geom_id = (
    fp8_test.make_case,
    fp8_test.GEOM_POINTS,
    fp8_test.geom_id,
)
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


def topk_values(scores, k=16):
    """Top-k score values per row, ties irrelevant: compare values, not ids."""
    s = scores.masked_fill(torch.isnan(scores), float("-inf"))
    return s.topk(min(k, s.shape[-1]), dim=-1).values


def assert_topk_agrees(got, ref):
    a, b = topk_values(got), topk_values(ref)
    fin = torch.isfinite(b)
    assert torch.equal(torch.isfinite(a), fin)
    err = (a[fin] - b[fin]).abs()
    assert bool((err <= ATOL + RTOL * b[fin].abs()).all())


@pytest.fixture(params=["bf16", "e4m3"], ids=["qbf16", "qe4m3"])
def qdt(request):
    """The dtype the query reaches the scorer in."""
    return request.param


def q_for(q, qdt):
    """(the Q the kernel is given, the bf16 Q the oracle scores). For e4m3 that is
    the index-Q writer's rule: the value is already rounded to bf16, then RNE and
    saturating to e4m3 (torch's own cast would turn an overflow into NaN). e4m3 is a
    subset of bf16, so the oracle's bf16 view of it is exact."""
    if qdt == "bf16":
        return q, q
    q8 = q.float().clamp(-448, 448).to(dtypes.fp8)
    return q8, q8.to(torch.bfloat16)


def run_fp4(q, kd, ks, bt, seq, S, H, mb, **kw):
    return kernel.score_flydsl_fp4(q, kd, ks, bt, seq, S, H, D**-0.5, mb, **kw)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
@requires_gfx950
@pytest.mark.parametrize(
    "case",
    fp8_test.build_matrix(),
    ids=[c[0].replace("fp8", "fp4") for c in fp8_test.build_matrix()],
)
def test_matches_oracle(case, qdt):
    _, B, S, H, lens, kw = case
    q, cache8, bt, seq, mb = make_case(B, S, H, lens, **kw)
    q, q_ref = q_for(q, qdt)
    kd, ks, deq = fp4_cache_of(cache8)
    ref = score_oracle(q_ref, deq, bt, seq, S, H, D**-0.5, mb)
    got = run_fp4(q, kd, ks, bt, seq, S, H, mb)
    ok, why = compare(got, ref)
    assert ok, why
    assert_topk_agrees(got, ref)


@requires_gfx950
@pytest.mark.parametrize("point", GEOM_POINTS, ids=geom_id)
def test_every_geometry_matches_oracle(point, qdt):
    """Every launch geometry the selector can reach (work-chunk depth, token
    split, spread, CP shard) against the oracle. Negative scores expose
    accidental zero-initialised maxima, as in the fp8 test."""
    S, H, batch, max_block, world, rank = point
    glob = max_block * world
    lens = ([0, 1, 130] + [P] * batch)[: batch - 1] + [glob * P]
    lens = [min(x, glob * P) for x in lens]
    q, cache8, bt, seq, mb = make_case(batch, S, H, lens)
    assert mb == glob
    local = -(-(mb - rank) // world)
    q.copy_(q.abs() + 0.25)
    q, q_ref = q_for(q, qdt)
    kd, ks, deq = fp4_cache_of(cache8, lambda b: -(b.abs() + 0.25))
    del cache8
    ref = score_oracle(q_ref, deq, bt, seq, S, H, D**-0.5, local, world, rank)
    out = fp8_kernel.alloc_score(batch, S, H, local, q.device)
    out.fill_(float("nan"))
    got = run_fp4(
        q, kd, ks, bt, seq, S, H, local, out=out, cp_world=world, cp_rank=rank
    )
    assert got.data_ptr() == out.data_ptr()
    ok, why = compare(got, ref)
    assert ok, why
    assert bool((got[torch.isfinite(ref)] < 0).all())


@requires_gfx950
@pytest.mark.parametrize("world,rank", [(1, 0), (4, 3)])
@pytest.mark.parametrize("S,H", [(1, 4), (4, 2), (8, 4)])
@pytest.mark.parametrize("layout", ["contiguous", "feature"])
@pytest.mark.parametrize("map_in_graph", [False, True])
def test_graph_buffers(world, rank, S, H, layout, map_in_graph, qdt):
    """Captured once, replayed with different lengths and block tables."""
    B, mb = 3, 9
    q, cache8, _, seq, _ = make_case(B, S, H, [0, 513, 1])
    q, q_ref = q_for(q, qdt)
    kd, ks, deq = fp4_cache_of(cache8)
    bt = torch.zeros((B, mb * world), dtype=torch.int32, device="cuda")
    bt[:, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    cp = {"cp_world": world, "cp_rank": rank}
    buf = torch.empty(
        (kernel.work_map_capacity(B, mb, S, H, **cp), 2),
        device="cuda",
        dtype=torch.int32,
    )
    wm = kernel.build_work_map(seq, mb, S, H, out=buf, **cp)
    out = (
        torch.empty((H, B * S, mb), device="cuda", dtype=torch.float32)
        if layout == "contiguous"
        else torch.empty((mb, B * S, H), device="cuda", dtype=torch.float32).permute(
            2, 1, 0
        )
    )

    def run():
        current = (
            kernel.build_work_map(seq, mb, S, H, out=buf, **cp) if map_in_graph else wm
        )
        return run_fp4(q, kd, ks, bt, seq, S, H, mb, out=out, work_map=current, **cp)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths, shift in [([0, 513, 1], 0), ([130, 0, 257], 3), ([1, 129, 0], 1)]:
        seq.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        bt.copy_(
            (
                torch.arange(mb * world, device="cuda", dtype=torch.int32)[None, :]
                + shift
            )
            .remainder(kd.shape[0])
            .expand(B, -1)
        )
        if not map_in_graph:
            kernel.build_work_map(seq, mb, S, H, out=buf, **cp)
        graph.replay()
        check_scores(
            out, score_oracle(q_ref, deq, bt, seq, S, H, D**-0.5, mb, world, rank)
        )


@requires_gfx950
def test_split_and_single_region_views_agree_bitwise(qdt):
    """Two separate tensors and two views of one `[pages, 8704]` page are the
    same input to the kernel, down to the bit."""
    q, cache8, bt, seq, mb = make_case(3, 4, 4, [130, 4096, 513])
    q, _ = q_for(q, qdt)
    kd, ks, _ = fp4_cache_of(cache8)
    _buf, kd1, ks1 = single_region_views(kd, ks)
    assert kd1.stride(0) == ks1.stride(0) == 8704
    a = run_fp4(q, kd, ks, bt, seq, 4, 4, mb)
    b = run_fp4(q, kd1, ks1, bt, seq, 4, 4, mb)
    assert torch.equal(a, b)


@requires_gfx950
def test_agrees_with_the_fp8_kernel_on_the_same_values(qdt):
    """The dequantized K is exactly representable in fp8 (for scales inside
    e4m3's range), so the fp8 scorer and this one score the SAME numbers. Not
    bitwise: the two read different k permutations, so the fp32 sums associate
    differently."""
    B, S, H = 2, 4, 4
    q, cache8, bt, seq, mb = make_case(B, S, H, [4096, 1000])
    q, q_ref = q_for(q, qdt)
    kd, ks, deq = fp4_cache_of(cache8)
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "dequantized K left fp8's range"
    a = run_fp4(q, kd, ks, bt, seq, S, H, mb)
    b = fp8_kernel.score_flydsl(q_ref, as_fp8, bt, seq, S, H, D**-0.5, mb)
    check_scores(a, b)


@requires_gfx950
def test_wide_scale_range(qdt):
    """Per-token magnitudes over 2^+-20 give e8m0 bytes spread over the range a
    real cache uses; every lane group must pick its own byte."""
    B, S, H = 2, 4, 4
    q, cache8, bt, seq, mb = make_case(B, S, H, [2048, 700])
    pages = cache8.shape[0]
    gen = torch.Generator(device="cuda").manual_seed(7)
    mag = torch.exp2(
        torch.randint(
            -20, 21, (pages, P, D // 32), device="cuda", generator=gen
        ).float()
    )
    mag = mag.repeat_interleave(32, dim=-1)
    k = torch.randn((pages, P, D), device="cuda", generator=gen) * mag
    kd, ks, deq = quantize_mxfp4(k.to(torch.bfloat16))
    assert int(ks.min()) < 110 and int(ks.max()) > 140
    q2 = (q.float() * 2**-8).to(torch.bfloat16)  # keep scores inside fp32 comfortably
    q2, q2_ref = q_for(q2, qdt)
    ref = score_oracle(q2_ref, deq, bt, seq, S, H, D**-0.5, mb)
    got = run_fp4(q2, kd, ks, bt, seq, S, H, mb)
    ok, why = compare(got, ref)
    assert ok, why


def test_reference_scale_rule_is_the_hip_rule():
    """The quantizer under test picks e8m0 = ceil(log2(amax/6)) + 127 with the
    bit rule of `indexer_fp4_scale_e8m0` (cache_kernels.cu), i.e. the rule a
    writer in ATOM applies."""
    torch.manual_seed(0)
    x = (
        torch.randn(4096, D) * torch.exp2(torch.randint(-12, 12, (4096, 1)).float())
    ).to(torch.bfloat16)
    _, s = per_1x32_f4_quant(x)
    amax = x.float().reshape(-1, 32).abs().amax(-1)
    eps = 6.0 * torch.tensor(2.0**-126)
    bits = (
        torch.maximum(amax, eps) * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    ).view(torch.int32)
    e = (bits >> 23) & 0xFF
    e = e + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    assert torch.equal(e.to(torch.uint8), s.view(torch.uint8).reshape(-1))


@requires_gfx950
def test_default_score_buffer_reads_as_minus_inf():
    lens = [512, 256, 128, 512]
    q, cache8, bt, seq, mb = make_case(len(lens), 1, 1, lens)
    kd, ks, _ = fp4_cache_of(cache8)
    out = run_fp4(q, kd, ks, bt, seq, 1, 1, mb)
    assert (out == float("-inf")).any()
    assert torch.isfinite(out[out != float("-inf")]).all()


@requires_gfx950
def test_e4m3_q_scale_and_dtype_dispatch():
    """`q_scale` multiplies the scores of an e4m3 query (default 1.0 = the plain cast),
    a non-positive one is refused, and the dtype alone picks the variant."""
    B, S, H = 2, 4, 4
    q, cache8, bt, seq, mb = make_case(B, S, H, [513, 4096])
    q8, q_ref = q_for(q, "e4m3")
    kd, ks, deq = fp4_cache_of(cache8)
    ref = score_oracle(q_ref, deq, bt, seq, S, H, D**-0.5, mb)
    base = run_fp4(q8, kd, ks, bt, seq, S, H, mb)
    check_scores(base, ref)
    half = run_fp4(q8, kd, ks, bt, seq, S, H, mb, q_scale=0.5)
    fin = torch.isfinite(base)
    assert torch.equal(torch.isneginf(half), torch.isneginf(base))
    assert torch.allclose(half[fin], 0.5 * base[fin], rtol=1e-6, atol=1e-6)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="q_scale"):
            run_fp4(q8, kd, ks, bt, seq, S, H, mb, q_scale=bad)
        assert not kernel.index_score_supported_fp4(
            q8, kd, ks, S, H, mb, bt, seq_lens=seq, q_scale=bad
        )
    assert kernel.index_score_supported_fp4(q8, kd, ks, S, H, mb, bt, seq_lens=seq)


@requires_gfx950
def test_e4m3_q_against_bf16_q_is_only_the_q_rounding():
    """What taking the query as e4m3 costs, isolated: the same cache scored with the
    bf16 query and with that query rounded to e4m3. The gap is bounded by e4m3's
    2^-4 relative step per element, and the top-16 block set moves only through
    near ties."""
    B, S, H = 2, 4, 4
    q, cache8, bt, seq, mb = make_case(B, S, H, [8192, 6000])
    q8, _ = q_for(q, "e4m3")
    kd, ks, _ = fp4_cache_of(cache8)
    a = run_fp4(q, kd, ks, bt, seq, S, H, mb)
    b = run_fp4(q8, kd, ks, bt, seq, S, H, mb)
    fin = torch.isfinite(a)
    assert torch.equal(fin, torch.isfinite(b))
    gap = (a[fin] - b[fin]).abs().max().item()
    assert gap <= 0.0625 * a[fin].abs().max().item(), gap
    ta = a.masked_fill(~fin, float("-inf")).topk(16, dim=-1).indices
    tb = b.masked_fill(~fin, float("-inf")).topk(16, dim=-1).indices
    same = torch.tensor(
        [
            len(set(x.tolist()) & set(y.tolist())) / 16
            for x, y in zip(ta.reshape(-1, 16), tb.reshape(-1, 16))
        ]
    )
    assert same.mean() > 0.8, same.mean()


@requires_gfx950
def test_e4m3_q_rows_must_be_16_byte_aligned():
    _, cache8, bt, seq, mb = make_case(2, 4, 2, [513, 513])
    kd, ks, _ = fp4_cache_of(cache8)
    flat = torch.empty(8 * 2 * D + 64, dtype=dtypes.fp8, device="cuda")
    bad = torch.as_strided(flat, (8, 2, D), (2 * D + 8, D, 1), 0)  # row stride 264 B
    assert not kernel.index_score_supported_fp4(bad, kd, ks, 4, 2, mb, bt, seq_lens=seq)
    with pytest.raises(ValueError, match="16-byte aligned"):
        run_fp4(bad, kd, ks, bt, seq, 4, 2, mb)


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
def _valid_inputs():
    q, cache8, bt, seq, mb = make_case(2, 4, 4, [130, 513])
    kd, ks, _ = fp4_cache_of(cache8)
    return q, kd, ks, bt, seq, mb


def _bad(name):
    q, kd, ks, bt, seq, mb = _valid_inputs()
    pages = kd.shape[0]
    if name == "data_dtype":
        kd = kd.view(torch.int8)
    elif name == "scale_dtype":
        ks = ks.view(torch.int8)
    elif name == "data_shape":
        kd = kd[:, :, :32]
    elif name == "scale_shape":
        ks = ks[:, :, :2]
    elif name == "pages_differ":
        ks = ks[:-1]
    elif name == "data_token_stride":
        kd = torch.empty((pages, P, 128), dtype=torch.uint8, device="cuda")[:, :, :64]
    elif name == "scale_token_stride":
        ks = torch.empty((pages, P, 8), dtype=torch.uint8, device="cuda")[:, :, :4]
    elif name == "data_page_stride_unaligned":
        buf = torch.empty(pages * (P * 64 + 8) + 16, dtype=torch.uint8, device="cuda")
        kd = torch.as_strided(buf, (pages, P, 64), (P * 64 + 8, 64, 1))
    elif name == "data_pages_overlap":
        kd = torch.as_strided(kd, (pages, P, 64), (P * 64 - 16, 64, 1))
    elif name == "scale_misaligned":
        buf = torch.empty(pages * P * 4 + 8, dtype=torch.uint8, device="cuda")
        ks = torch.as_strided(buf, (pages, P, 4), (P * 4, 4, 1), 2)
    elif name == "cpu":
        kd = kd.cpu()
    elif name == "q_dtype":
        q = q.float()
    return q, kd, ks, bt, seq, mb


BAD = [
    "data_dtype",
    "scale_dtype",
    "data_shape",
    "scale_shape",
    "pages_differ",
    "data_token_stride",
    "scale_token_stride",
    "data_page_stride_unaligned",
    "data_pages_overlap",
    "scale_misaligned",
    "q_dtype",
]


@requires_gfx950
@pytest.mark.parametrize("name", BAD + ["cpu"])
def test_invalid_cache_rejected(name):
    q, kd, ks, bt, seq, mb = _bad(name)
    assert not kernel.index_score_supported_fp4(q, kd, ks, 4, 4, mb, bt, seq_lens=seq)
    with pytest.raises(ValueError):
        run_fp4(q, kd, ks, bt, seq, 4, 4, mb)


def test_metadata_rejects_cpu():
    q = torch.empty(1, 1, D, dtype=torch.bfloat16)
    kd = torch.empty(1, P, 64, dtype=torch.uint8)
    ks = torch.empty(1, P, 4, dtype=torch.uint8)
    bt = torch.zeros(1, 1, dtype=torch.int32)
    seq = torch.zeros(1, dtype=torch.int32)
    assert not kernel.index_score_supported_fp4(q, kd, ks, 1, 1, 1, bt, seq_lens=seq)


@requires_gfx950
def test_supported_accepts_valid_inputs():
    q, kd, ks, bt, seq, mb = _valid_inputs()
    assert kernel.index_score_supported_fp4(q, kd, ks, 4, 4, mb, bt, seq_lens=seq)


@requires_gfx950
def test_exports_resolve():
    import aiter.ops.flydsl as F

    for name in (
        "minimax_m3_index_score_fp4_flydsl",
        "minimax_m3_index_score_fp4_flydsl_supported",
        "minimax_m3_index_score_prefill_fp4_flydsl",
    ):
        assert name in F.__all__
        getattr(F, name)


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


def run_prefill_fp4(c, kd, ks, **kw):
    heads = c["q8"].shape[1]
    out = fp8_test.prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    prefill_kernel.score_prefill_flydsl_fp4(
        c["q_bf16"],
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
def test_prefill_matches_aiter_and_fp8_kernel(shape):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    got = run_prefill_fp4(c, kd, ks)
    fp8_test.assert_matches_aiter(got, ref, live, f"{shape}")
    assert torch.equal(got[live], fp8_test.run_prefill(c)[live])


@requires_gfx950
@pytest.mark.parametrize(
    "shape", fp8_test.GEOM_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_prefill_every_geometry(shape):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_prefill_fp4(c, kd, ks), ref, live, f"{shape}")


@requires_gfx950
def test_prefill_two_heads():
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = fp8_test.aiter_scores(c)
    fp8_test.assert_matches_aiter(run_prefill_fp4(c, kd, ks), ref, live, "heads=2")


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
def test_prefill_mixed_query_lengths_and_trailing_empty_request():
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
    prefill_kernel.score_prefill_flydsl_fp4(
        q8.float().bfloat16(),
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
def test_prefill_rejections():
    c, kd, ks = fp4_prefill_case(1, 128, 1000)
    args = (c["cu"], c["seq"], c["prefix"], c["max_q"], c["max_seq"], 1.0 / LOG2E)
    f = prefill_kernel.score_prefill_flydsl_fp4
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


# ---------------------------------------------------------------------------
# Perf sweep (python op_tests/test_flydsl_minimax_m3_index_score_fp4.py)
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score_fp4.py"
)
@benchmark()
def test_score_fp4(case, B, S, H, lens):
    q, cache8, bt, seq, mb = make_case(B, S, H, lens)
    kd, ks, deq = fp4_cache_of(cache8)
    ref = score_oracle(q, deq, bt, seq, S, H, D**-0.5, mb)
    ret = {"gfx": get_gfx(), "timing": "repeated-buffer (warm effective TB/s)"}
    capacity = fp8_kernel.work_map_capacity(B, mb, S, H)
    buf = torch.empty((capacity, 2), device=q.device, dtype=torch.int32)
    wm = fp8_kernel.build_work_map(seq, mb, S, H, out=buf)
    pages = sum((L + P - 1) // P for L in lens)
    flops = 2 * pages * P * D * S * H
    cands = {
        "fp4": (
            lambda: run_fp4(q, kd, ks, bt, seq, S, H, mb, work_map=wm),
            pages * P * 68,
        ),
        "fp8": (
            lambda: fp8_kernel.score_flydsl(
                q, deq.to(dtypes.fp8), bt, seq, S, H, D**-0.5, mb, work_map=wm
            ),
            pages * P * D,
        ),
    }
    for name, (run, kbytes) in cands.items():
        got, us = run_perftest(run, num_iters=10, num_rotate_args=1)
        err = checkAllclose(
            ref[torch.isfinite(ref)].float(),
            got[torch.isfinite(ref)].float(),
            atol=ATOL,
            rtol=RTOL,
            tol_err_ratio=0,
        )
        nbytes = kbytes + q.numel() * 2 + pages * S * H * 4
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
            }
        )
    return ret


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="MiniMax-M3 MXFP4 index score sweep")
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[8, 25])
    ap.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(4, 1, 32768), (4, 4, 32768), (4, 1, 131072)],
        metavar="S,H,L",
        help="query tokens/request, index heads, sequence length (page size and head dim stay 128)",
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in DECODE_SUPPORTED_GFX:
        aiter.logger.warning("index score requires gfx950; skipping sweep")
        return 0
    rows = [
        test_score_fp4(f"B{B}_S{S}_H{H}_L{L}", B, S, H, [L] * B)
        for B, (S, H, L) in itertools.product(args.batch, args.mnk)
    ]
    if rows:
        aiter.logger.info(
            "mxfp4 decode index score summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
