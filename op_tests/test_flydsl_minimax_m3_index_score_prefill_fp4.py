# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index score over an MXFP4 and an NVFP4 index cache.

`score_prefill_flydsl_fp4` is the fp8 prefill kernel built with `k_fp4`: the fp4
pages stay MXFP4 through LDS and feed the scaled MFMA. It must match aiter's
`pa_sparse_block_score_prefill` and the fp8 FlyDSL kernel bit for bit on the
dequantized values.

Reference: aiter's `per_1x32_f4_quant` (RoundUp) quantizes, `fp4_utils.mxfp4_to_f32`
x `e8m0_to_f32` dequantizes, and aiter's `pa_sparse_block_score_prefill` (plus the fp8
FlyDSL kernel) scores the dequantized values as fp8 -- nothing from the kernel under
test. e8m0 bytes 0 and 255 are outside its contract and are not generated.

The NVFP4 cache (`score_prefill_flydsl_nvfp4`: e2m1 data, one e4m3 scale per 16
elements, 72 B per token, no per-tensor scale) is checked against a torch oracle that
dequantizes the cache itself (K = e2m1 x scale, either exact or rounded to e4m3 like
vLLM's NVFP4 scorer), on data whose fp32 partial sums are exact so the comparison is
bit for bit whatever the accumulation order. There is no NVFP4 write kernel in this
change; the pages are written by a torch reference quantizer in this file.

Run (exits nonzero if any check fails):
    python op_tests/test_flydsl_minimax_m3_index_score_prefill_fp4.py [--case SUBSTR ...]
"""

import argparse
import itertools
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    LOG2E,
    score_prefill_flydsl,
    score_prefill_flydsl_fp4,
    score_prefill_flydsl_nvfp4,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    alloc_score as prefill_alloc_score,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    geometry as prefill_geometry,
)
from aiter.ops.msa_block_select import pa_sparse_block_score_prefill
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

P = 128  # page size and head dim: the public operator contract
D = 128
SENTINEL = -12345.678  # a value no score can take: "did the kernel write here"

# The fp8 op test's shapes (op_tests/test_flydsl_minimax_m3_index_score.py), same lists.
PREFILL_SHAPES = [
    (1, 256, 256),
    (1, 130, 1000),
    (3, 777, 5000),
    (1, 2048, 2048),
    (4, 512, 8192),
    (8, 512, 16384),
    (2, 2048, 32768),
    (1, 1024, 65536),
    (1, 4096, 131072),
    (1, 4096, 524288),
]
# One (batch, q_len, ctx) per geometry the selector can reach.
GEOM_SHAPES = [
    (1, 128, 256),  # tile_q 64, 4 waves, L1, 16-wide
    (8, 512, 1000),  # tile_q 128, 4 waves, L1
    (8, 512, 2048),  # tile_q 128, 4 waves, L2
    (2, 128, 16384),  # tile_q 256, 8 waves, L1
    (4, 128, 16384),  # tile_q 256, 8 waves, L2
    (8, 2048, 16384),  # tile_q 256, 8 waves, L4 -- the production point
]


def _prefill_work(batch, q_len, ctx_len):
    """(live (query, block) pairs, unique pages) for a uniform ragged batch."""
    prefix = ctx_len - q_len
    pairs = sum(-(-(prefix + i + 1) // P) for i in range(q_len)) * batch
    return pairs, batch * (-(-ctx_len // P))


def make_prefill_case(batch, q_len, ctx_len, heads=1, seed=0):
    """A uniform ragged batch: every request has q_len new tokens on ctx_len."""
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=dev)
    prefix = torch.full((batch,), ctx_len - q_len, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=dev)
    max_block = -(-ctx_len // P)
    npages = batch * max_block
    q8 = (torch.randn(batch * q_len, heads, D, device=dev) / 4).to(dtypes.fp8)
    cache = (torch.randn(npages, P, D, device=dev) / 4).to(dtypes.fp8)
    # A permutation on purpose: an identity block table would hide a kernel that
    # assumed pages were visited in storage order.
    bt = (
        torch.randperm(npages, device=dev, dtype=torch.int32)
        .view(batch, max_block)
        .contiguous()
    )
    return {
        "q8": q8,
        # aiter takes Q in fp8; the bf16 arm gets its exact upcast.
        "q_bf16": q8.float().bfloat16(),
        "cache": cache,
        "bt": bt,
        "cu": cu,
        "seq": seq_lens,
        "prefix": prefix,
        "max_q": q_len,
        "max_seq": ctx_len,
        "batch": batch,
        "mb": max_block,
    }


def aiter_scores(c):
    """aiter's scores, and the mask of slots it considers live."""
    heads = c["q8"].shape[1]
    ref = torch.full((heads, c["q8"].shape[0], c["mb"]), -float("inf"), device="cuda")
    pa_sparse_block_score_prefill(
        c["q8"],
        c["cache"],
        ref,
        c["bt"],
        c["cu"],
        c["seq"],
        max_query_len=c["max_q"],
        max_seq_len=c["max_seq"],
    )
    return ref, torch.isfinite(ref)


def run_prefill(c):
    """The fp8 FlyDSL scorer on the case's fp8 cache, into a sentinel-filled buffer."""
    heads = c["q8"].shape[1]
    out = prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        c["q_bf16"],
        c["cache"],
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
    )
    return out.reshape(heads, c["q8"].shape[0], c["mb"])


SUPPORTED_GFX = ["gfx950"]  # the FlyDSL scorers are built for gfx950
QCHUNK = 512  # pages quantized per torch call; the reference makes fp32 temporaries


def check_equal(got, ref, what):
    """Bit-exact comparison, reported through aiter's `checkAllclose`."""
    err = checkAllclose(ref.float(), got.float(), rtol=0, atol=0, msg=f"{what}: ")
    assert err == 0, f"{what}: mismatch fraction {err}"


def check_scores(got, ref, live, what):
    """Every live slot was written and equals `ref`, bit for bit."""
    unwritten = int((live & (got == SENTINEL)).sum())
    assert unwritten == 0, f"{what}: {unwritten} live slots left unwritten"
    check_equal(got[live], ref[live], what)


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
    (3, 300, 33333),
]


def check_prefill_matches_aiter_and_fp8_kernel(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = aiter_scores(c)
    got = run_fp4(c, kd, ks, qdt)
    check_scores(got, ref, live, f"{shape}")
    check_equal(got[live], run_prefill(c)[live], f"{shape}: fp8 kernel")


def check_prefill_every_geometry(shape, qdt):
    c, kd, ks = fp4_prefill_case(*shape)
    ref, live = aiter_scores(c)
    check_scores(run_fp4(c, kd, ks, qdt), ref, live, f"{shape}")


def check_prefill_geometry_arms_are_exhaustive():
    """Every geometry the selector can reach has an every_geometry case: without
    this, narrowing `geometry` leaves the arms passing on the surviving shape."""
    dev = torch.device("cuda")

    def key(g):
        return (g["tile_q"], g["waves"], g["pages_per_wave"], g["m32"])

    reachable = set()
    for ctx in (256, 1000, 2048, 5000, 8192, 16384, 32768, 131072, 524288):
        for q_len in (128, 130, 256, 512, 777, 1024, 2048, 4096):
            for batch in (1, 2, 3, 4, 8):
                if q_len <= ctx:
                    reachable.add(
                        key(prefill_geometry(q_len, batch, 1, -(-ctx // P), dev))
                    )
    covered = {
        key(prefill_geometry(q_len, batch, 1, -(-ctx // P), dev))
        for batch, q_len, ctx in GEOM_SHAPES
    }
    assert covered == reachable, f"uncovered: {sorted(reachable - covered)}"


def check_prefill_typed_views(qdt):
    """per_1x32_f4_quant's own dtypes (fp4x2 data, e8m0 scale) are accepted and
    score bit for bit like the uint8 views."""
    c, kd, ks = fp4_prefill_case(1, 256, 1000)
    plain = run_fp4(c, kd, ks, qdt)
    typed = run_fp4(c, kd.view(dtypes.fp4x2), ks.view(dtypes.fp8_e8m0), qdt)
    check_equal(typed, plain, "typed views")


def check_prefill_two_heads(qdt):
    c, kd, ks = fp4_prefill_case(2, 256, 3000, heads=2)
    ref, live = aiter_scores(c)
    check_scores(run_fp4(c, kd, ks, qdt), ref, live, "heads=2")


def check_prefill_ragged_tails(ctx):
    """Lengths off a page boundary: the tail of the last page holds whatever the
    cache holds, and must not reach a live score."""
    c, kd, ks = fp4_prefill_case(3, min(ctx, 128), ctx)
    ref, live = aiter_scores(c)
    check_scores(run_fp4(c, kd, ks), ref, live, f"ctx={ctx}")


def check_prefill_scale_extremes(qdt):
    """Per-token magnitudes spread over 2^-3 .. 2^3, so the e8m0 bytes walk across
    the window e4m3 can still hold exactly: the scale really is applied per lane
    group and per token, not once. Bytes outside that window have no fp8 judge:
    see `check_prefill_scale_bytes_beyond_e4m3`."""
    c = make_prefill_case(2, 256, 3000)
    span = torch.randint(-3, 4, (c["cache"].shape[0], P, 1), device="cuda")
    k = c["cache"].to(torch.bfloat16) * (2.0 ** span.to(torch.bfloat16))
    kd, ks, deq = quantize_mxfp4(k)
    as_fp8 = deq.to(dtypes.fp8)
    assert torch.equal(as_fp8.to(torch.bfloat16), deq), "left e4m3's exact range"
    assert ks.min() < ks.max() - 6
    c = dict(c, cache=as_fp8)
    ref, live = aiter_scores(c)
    check_scores(run_fp4(c, kd, ks, qdt), ref, live, "scales")


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]  # magnitudes of the 3 low code bits
# e8m0 bytes below 119 and above 133 leave e4m3's exact range for K, so aiter's fp8
# scorer cannot judge them; 119 and 133 are the edge, 127 is 2^0.
WIDE_BYTES = [100, 110, 118, 119, 127, 133, 134, 140, 150]


def check_prefill_scale_bytes_beyond_e4m3(qdt):
    """e8m0 bytes the e4m3 oracle cannot hold (< 119 and > 133). The cache is built
    straight from e2m1 codes and one scale byte per token, and the expected scores
    are an fp64 dequantisation computed here, value = e2m1 * 2^(byte - 127), then
    `max over live tokens of dot(K, Q)` -- nothing from the kernel or the quantizer.
    Q holds +-2^-2..2^1, so every dot is a sum of multiples of 2^-3 and exact in fp32
    in any order: the comparison is bit for bit."""
    c = make_prefill_case(1, 128, 1000)
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    g = torch.Generator(device="cuda").manual_seed(1)
    q = 2.0 ** torch.randint(-2, 2, (rows, heads, D), device="cuda", generator=g)
    q8 = (q * (1 - 2 * torch.randint(0, 2, q.shape, device="cuda", generator=g))).to(
        dtypes.fp8
    )
    npages = c["bt"].numel()
    kd = torch.randint(0, 256, (npages, P, D // 2), device="cuda", generator=g).to(
        torch.uint8
    )
    byte = torch.tensor(WIDE_BYTES, dtype=torch.uint8, device="cuda")
    ks = byte[torch.arange(npages * P, device="cuda") % len(WIDE_BYTES)]
    ks = ks.view(npages, P, 1).expand(npages, P, D // 32).contiguous()
    assert ks.min() < 119 and ks.max() > 133
    # fp64 dequantisation: low nibble is the even element, bit 3 is the sign
    code = torch.stack([kd & 0xF, kd >> 4], dim=-1).reshape(npages, P, D).long()
    mag = torch.tensor(E2M1, dtype=torch.float64, device="cuda")[code & 7]
    val = torch.where(code & 8 != 0, -mag, mag)
    val = val * 2.0 ** (ks.double() - 127).repeat_interleave(32, dim=-1)
    # expected[h, r, p] = max over t with p*P + t <= prefix + r of dot(K, Q) * 1
    pos = (c["prefix"][0] + torch.arange(rows, device="cuda")).view(1, rows, 1, 1)
    gpos = (torch.arange(c["mb"] * P, device="cuda")).view(1, 1, c["mb"], P)
    dots = torch.einsum("rhd,pld->hrpl", q8.double(), val[c["bt"][0].long()])
    ref = dots.masked_fill(gpos > pos, -float("inf")).amax(dim=-1).float()
    c = dict(c, q8=q8, q_bf16=q8.float().bfloat16())
    check_scores(run_fp4(c, kd, ks, qdt), ref, torch.isfinite(ref), "wide scale bytes")


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
    check_equal(poisoned[live], base[live], "garbage tails")
    ref, ref_live = aiter_scores(c)
    check_scores(base, ref, ref_live, "ragged lengths")


def check_prefill_mixed_query_lengths_and_trailing_empty_request(qdt):
    c, kd, ks, ref, live = mixed_lengths_case()
    got = run_fp4(c, kd, ks, qdt)
    check_scores(got, ref, live, "mixed lengths + trailing empty")


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
    check_equal(run_fp4(dict(c, q8=view), kd, ks, "e4m3"), plain, "strided q")
    kd1, ks1 = single_region_views(kd, ks)
    check_equal(run_fp4(c, kd1, ks1, "e4m3"), plain, "single-region pages")


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


# ==========================================================================
# NVFP4 PREFILL (score_prefill_flydsl_nvfp4)
#
# A token is 64 B of e2m1 plus 8 e4m3 scale bytes (one per 16 elements, no
# per-tensor scale). K is e4m3_rn_satfinite(e2m1 * scale), which fits an fp8 cache
# exactly, so aiter's own fp8 judges (`aiter_scores`) score the same values; and an
# independent torch oracle scores them in fp32. The quantizer is test-side and has
# vLLM's `_quant4` numerics.
#
# Two kinds of comparison, because the fp8 MFMA's internal accumulation depends on
# which k elements share an instruction: on exact-sum data (every partial sum exact
# in fp32) the oracle must be matched to the BIT whatever the partition; on general
# data the judge is matched to a few fp32 ulps (the kernel's partition is vLLM's,
# not the fp8 scorer's -- see `_k32_chunk`).
# ==========================================================================
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0
RCP6_F32 = 0x3E2AAAAB
NV_ROUNDS = ["e4m3"]


def e2m1_encode(a):
    """fp32 -> 4-bit e2m1 codes, round to nearest even (ties to the even code)."""
    m = a.abs()
    code = (
        (m > 0.25).int()
        + (m >= 0.75).int()
        + (m > 1.25).int()
        + (m >= 1.75).int()
        + (m > 2.5).int()
        + (m >= 3.5).int()
        + (m > 5.0).int()
    )
    return code + 8 * (a < 0).int()


def nv_product(data, scale):
    """e2m1 data u8 [..., 64] x e4m3 scale u8 [..., 8] -> exact fp32 [..., 128]."""
    lut = torch.tensor(E2M1, device=data.device)
    codes = torch.stack([(data & 0xF).long(), (data >> 4).long()], -1)
    codes = codes.reshape(*data.shape[:-1], D)
    val = lut[codes & 7] * (1 - 2 * (codes >> 3).float())
    return val * scale.view(dtypes.fp8).float().repeat_interleave(16, dim=-1)


def nv_pack(codes, scale_bytes):
    """codes int [pages,128,128], scale u8 [pages,128,8] -> (data, scale, exact, e4m3)."""
    pair = codes.reshape(*codes.shape[:-1], D // 2, 2)
    data = (pair[..., 0] | (pair[..., 1] << 4)).to(torch.uint8)
    prod = nv_product(data, scale_bytes)
    e4m3 = prod.clamp(-E4M3_MAX, E4M3_MAX).to(dtypes.fp8).float()
    return data, scale_bytes, prod, e4m3


def nv_quantize(k):
    """bf16 [pages,128,128] -> (data, scale, exact fp32, e4m3 fp32), vLLM `_quant4`
    numerics: scale = RNE e4m3 of min(amax * RN(1/6), 448) per 16 elements, elements
    * (1 / decoded scale) -> RNE e2m1."""
    x = k.float().reshape(*k.shape[:-1], D // 16, 16)
    rcp6 = torch.tensor(RCP6_F32, dtype=torch.int32).view(torch.float32).item()
    sf = (x.abs().amax(-1) * rcp6).clamp(max=E4M3_MAX).to(dtypes.fp8)
    sv = sf.float().unsqueeze(-1)
    quant = torch.where(sv != 0, 1.0 / sv, torch.zeros_like(sv))
    return nv_pack(
        e2m1_encode(x * quant).reshape(*k.shape[:-1], D), sf.view(torch.uint8)
    )


def nv_cache_of(cache8):
    outs = [[], [], [], []]
    for i in range(0, cache8.shape[0], QCHUNK):
        for o, t in zip(outs, nv_quantize(cache8[i : i + QCHUNK].to(torch.bfloat16))):
            o.append(t)
    return tuple(torch.cat(o) for o in outs)


def nv_oracle(c, deq):
    """[H, total_q, mb] block scores of `c` over the dequantized cache `deq` (fp32
    [pages,128,D]), in fp32: max over a page's tokens visible to the row, -inf where
    the page is not visible at all."""
    H, total_q = c["q8"].shape[1], c["q8"].shape[0]
    out = torch.full((H, total_q, c["mb"]), -float("inf"), device="cuda")
    cu, seq, pre = c["cu"].tolist(), c["seq"].tolist(), c["prefix"].tolist()
    for b in range(len(cu) - 1):
        ql = cu[b + 1] - cu[b]
        if ql == 0:
            continue
        nblk = -(-seq[b] // P)
        k = deq[c["bt"][b, :nblk].long()].reshape(nblk * P, D)
        q = c["q8"][cu[b] : cu[b + 1]].float()  # [ql, H, D]
        dot = torch.einsum("rhd,td->rht", q, k)  # [ql, H, nblk*128]
        vis = pre[b] + torch.arange(ql, device="cuda") + 1
        tok = torch.arange(nblk * P, device="cuda")
        dot = dot.masked_fill(tok[None, None, :] >= vis[:, None, None], -float("inf"))
        page = dot.reshape(ql, H, nblk, P).amax(-1)  # [ql, H, nblk]
        out[:, cu[b] : cu[b + 1], :nblk] = page.permute(1, 0, 2)
    return out


def nv_case(qls, seqs, heads=1, seed=0, exact_sum=False):
    """A ragged prefill case over an NVFP4 cache. Request b has qls[b] new rows on
    seqs[b] tokens (prefix = the rest). `exact_sum` makes every partial sum exact in
    fp32 (|Q| and every scale in [0.5, 2), see the decode test) so a kernel must match
    the oracle to the bit whatever the accumulation order."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    B, total_q = len(qls), sum(qls)
    mb = max(1, max(-(-L // P) for L in seqs))
    pages = B * mb
    if exact_sum:
        qb = torch.randint(0x30, 0x48, (total_q, heads, D), device=dev, generator=g)
        qb = qb | (torch.randint(0, 2, qb.shape, device=dev, generator=g) << 7)
        q8 = qb.to(torch.uint8).view(dtypes.fp8)
        codes = torch.randint(0, 16, (pages, P, D), device=dev, generator=g)
        sb = torch.randint(0x30, 0x48, (pages, P, 8), device=dev, generator=g)
        kd, ks, exact, e4m3 = nv_pack(codes, sb.to(torch.uint8))
    else:
        q8 = (torch.randn(total_q, heads, D, device=dev, generator=g) / 4).to(
            dtypes.fp8
        )
        cache8 = (torch.randn(pages, P, D, device=dev, generator=g) / 4).to(dtypes.fp8)
        kd, ks, exact, e4m3 = nv_cache_of(cache8)
    bt = torch.randperm(pages, device=dev, dtype=torch.int32, generator=g).view(B, mb)
    seq = torch.tensor(seqs, dtype=torch.int32, device=dev)
    ql = torch.tensor(qls, dtype=torch.int32, device=dev)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32, device=dev), ql.cumsum(0).int()])
    return {
        "q8": q8,
        "q_bf16": q8.float().bfloat16(),
        "kd": kd,
        "ks": ks,
        "exact": exact,
        "e4m3": e4m3,
        "bt": bt.contiguous(),
        "cu": cu.contiguous(),
        "seq": seq,
        "prefix": (seq - ql).contiguous(),
        "max_q": max(max(qls), 1),
        "max_seq": max(seqs),
        "mb": mb,
    }


def run_nv(c, qdt="e4m3", k_round="e4m3", views=None, out=None):
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    kd, ks = views or (c["kd"], c["ks"])
    if out is None:
        out = prefill_alloc_score(rows, heads, c["mb"], "cuda")
        out.fill_(SENTINEL)
    score_prefill_flydsl_nvfp4(
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
        k_round=k_round,
    )
    return out.reshape(heads, rows, c["mb"])


def _with_m32(m32):
    """Force the 32x32x64 (1) or 16x16x128 (0) instruction shape; None keeps the selector."""
    import aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill as mod

    class _Patch:
        def __enter__(self):
            self.orig = mod.geometry
            if m32 is not None:
                # the 32-wide shape has no DMA staging (4 waves, 16-wide only)
                mod.geometry = lambda *a, **k: {
                    **self.orig(*a, **k),
                    "m32": m32,
                    **({"dma": 0} if m32 else {}),
                }

        def __exit__(self, *exc):
            mod.geometry = self.orig

    return _Patch()


def _with_xwide(ppw=None):
    """Force the 512-row tile (and optionally the pages-per-wave) for e4m3, whatever the
    shape: its selection thresholds keep it off the small shapes the tests use."""
    import aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill as mod

    class _Patch:
        def __enter__(self):
            self.saved = (
                mod._XWIDE_MIN_BLOCKS,
                mod._XWIDE_MIN_TILES,
                mod._XWIDE_PPW4_MIN_CHUNK,
            )
            mod._XWIDE_MIN_BLOCKS = mod._XWIDE_MIN_TILES = mod._XWIDE_PPW4_MIN_CHUNK = 0
            mod._use_xwide.cache_clear()
            self.orig = mod.geometry
            if ppw is not None:
                mod.geometry = lambda *a, **k: {
                    **self.orig(*a, **k),
                    "pages_per_wave": ppw,
                }

        def __exit__(self, *exc):
            mod.geometry = self.orig
            mod._XWIDE_MIN_BLOCKS, mod._XWIDE_MIN_TILES, mod._XWIDE_PPW4_MIN_CHUNK = (
                self.saved
            )
            mod._use_xwide.cache_clear()

    return _Patch()


def nv_ref_of(c, k_round):
    return c["e4m3"] if k_round == "e4m3" else c["exact"]


def check_prefill_nvfp4_bit_exact(qls, seqs, heads, qdt, k_round, m32=None):
    """Exact-sum data against the fp32 oracle, bit for bit, on either instruction shape.
    Dead slots (no visible token) may stay unwritten or be -inf, never a score; live ones must all be written.
    """
    c = nv_case(qls, seqs, heads, exact_sum=True)
    ref = nv_oracle(c, nv_ref_of(c, k_round))
    with _with_m32(m32):
        try:
            got = run_nv(c, qdt, k_round)
        except ValueError as e:
            if m32 == 1 and "m32" in str(e):
                return  # this geometry has no 32-wide shape
            raise
    live = torch.isfinite(ref)
    check_scores(got, ref, live, f"q{qls[:3]} seq{seqs[:3]} H{heads}")
    dead = got[~live]
    assert bool(
        ((dead == SENTINEL) | torch.isneginf(dead)).all()
    ), "a dead slot got a score"


def check_prefill_nvfp4_matches_aiter(shape, qdt):
    """Random data against aiter's HIP scorer on the same e4m3 values, fp8 cache. Equal
    to a few fp32 ulps, not the bit: the fp8 MFMA sums differently depending on which
    k elements one instruction holds, and this kernel uses vLLM's grouping."""
    c = make_prefill_case(*shape)
    kd, ks, _, e4m3 = nv_cache_of(c["cache"])
    c = dict(c, cache=e4m3.to(dtypes.fp8))
    ref, live = aiter_scores(c)
    cn = dict(c, kd=kd, ks=ks)
    got = run_nv(cn, qdt)
    unwritten = int((live & (got == SENTINEL)).sum())
    assert unwritten == 0, f"{shape}: {unwritten} live slots left unwritten"
    err = checkAllclose(
        ref[live].float(), got[live].float(), rtol=1e-6, atol=0, msg=f"{shape}: "
    )
    assert err == 0, f"{shape}: mismatch fraction {err} against aiter"


def prefill_tail_cases():
    """(q_lens, seq_lens): cutoffs on and around page / tile boundaries, one-row and
    one-token requests, a prefix-only continuation, a trailing empty request, chunked
    prefill on a long context."""
    return [
        ([1], [1]),
        ([1], [127]),
        ([1], [128]),
        ([1], [129]),
        ([2, 1, 3], [128, 129, 257]),
        ([127, 128, 129], [256, 257, 255]),
        ([64, 64, 64, 64], [127, 128, 129, 255]),
        ([100, 1, 300, 17], [100, 2000, 301, 4096]),
        ([128], [128]),
        ([129], [129]),
        ([300], [300]),
        ([1000], [1001]),
        ([256, 256], [384, 640]),
        ([16, 16, 16], [16 + 112, 16 + 128, 16 + 129]),
        ([512], [20000]),
        ([5, 0, 7], [900, 900, 1500]),
        ([4, 0], [300, 300]),
    ]


def check_prefill_nvfp4_tails(qls, seqs, heads, qdt, m32=None):
    check_prefill_nvfp4_bit_exact(qls, seqs, heads, qdt, "e4m3", m32)


def check_prefill_nvfp4_xwide(qls, seqs, heads, qdt, ppw):
    """e4m3 arm on the 512-row tile, exact-sum data, bit for bit against the oracle."""
    with _with_xwide(ppw):
        check_prefill_nvfp4_bit_exact(qls, seqs, heads, qdt, "e4m3")


def check_prefill_nvfp4_exact_wide(qls, seqs, heads, qdt, ppw):
    """The exact arm on the 256-row, 4-wave tile (the one big shapes get: lane-spread
    score stores, staged 64 B pieces when ppw is 4), forced on whatever the shape."""
    import aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill as mod

    orig = mod.exact_geometry
    mod.exact_geometry = lambda *a, **k: {
        **orig(*a, **k),
        "tile_q": 256,
        "pages_per_wave": ppw,
    }
    try:
        check_prefill_nvfp4_bit_exact(qls, seqs, heads, qdt, "exact")
    finally:
        mod.exact_geometry = orig


def check_prefill_nvfp4_exact_tails(qls, seqs, heads, qdt):
    """The exact arm (bf16 MFMA, K = e2m1 * scale unrounded) on the same lengths."""
    check_prefill_nvfp4_bit_exact(qls, seqs, heads, qdt, "exact")


def check_prefill_nvfp4_exact_every_shape(shape, qdt):
    batch, q_len, ctx_len = shape
    check_prefill_nvfp4_bit_exact([q_len] * batch, [ctx_len] * batch, 1, qdt, "exact")


def check_prefill_nvfp4_exact_random(shape, qdt):
    """Random data against the fp32 oracle on the UNROUNDED K. The accumulation order
    is the MFMA's, so allow a few fp32 ulps -- and require the e4m3 arm to differ (it
    scores rounded K), so a kernel that silently fell back to it fails here."""
    c = make_prefill_case(*shape)
    kd, ks, exact, e4m3 = nv_cache_of(c["cache"])
    cn = dict(c, kd=kd, ks=ks, exact=exact, e4m3=e4m3)
    ref = nv_oracle(cn, exact)
    live = torch.isfinite(ref)
    got = run_nv(cn, qdt, "exact")
    assert int((live & (got == SENTINEL)).sum()) == 0, "live slots left unwritten"
    rel = (got[live] - ref[live]).abs() / ref[live].abs().clamp(min=1e-2)
    assert float(rel.max()) <= 4e-6, f"{shape}: max rel error {float(rel.max())}"
    rounded = run_nv(cn, qdt, "e4m3")
    assert not torch.equal(got[live], rounded[live]), "exact scored like e4m3"


def check_prefill_nvfp4_every_geometry(shape, qdt, m32):
    batch, q_len, ctx = shape
    check_prefill_nvfp4_bit_exact([q_len] * batch, [ctx] * batch, 1, qdt, "e4m3", m32)


def check_prefill_nvfp4_single_region_views_and_typed_scale():
    """Both views of one [pages, 9216] allocation (8192 B of e2m1, then 1024 B of
    scale per page) score like two separate tensors."""
    c = nv_case([300, 200], [1000, 700], 2)
    pages = c["kd"].shape[0]
    buf = torch.empty((pages, P * 64 + P * 8), dtype=torch.uint8, device="cuda")
    buf[:, : P * 64] = c["kd"].reshape(pages, -1)
    buf[:, P * 64 :] = c["ks"].reshape(pages, -1)
    kd = torch.as_strided(buf, (pages, P, 64), (buf.stride(0), 64, 1))
    ks = torch.as_strided(buf, (pages, P, 8), (buf.stride(0), 8, 1), P * 64)
    check_equal(run_nv(c, views=(kd, ks)), run_nv(c), "single region")


def check_prefill_nvfp4_default_k_round_is_e4m3():
    """Leaving `k_round` out scores like `k_round="e4m3"` (the vLLM-identical numerics)."""
    c = nv_case([300, 200], [1000, 700], 2, exact_sum=True)
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    out = prefill_alloc_score(rows, heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl_nvfp4(
        c["q8"],
        c["kd"],
        c["ks"],
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
    )
    expect = run_nv(c, "e4m3", "e4m3")
    assert torch.equal(
        out.reshape(expect.shape), expect
    ), "the default k_round is not e4m3"


def check_prefill_nvfp4_modes_do_not_share_a_compiled_kernel():
    """MXFP4 and NVFP4 prefill in one process, interleaved, each reproducing its
    own first answer."""
    c, kd4, ks4 = fp4_prefill_case(1, 300, 1000)
    cn = nv_case([300], [1000])
    first = (run_fp4(c, kd4, ks4, "e4m3"), run_nv(cn))
    again = (run_nv(cn), run_fp4(c, kd4, ks4, "e4m3"))
    check_equal(again[1], first[0], "mxfp4 after nvfp4")
    check_equal(again[0], first[1], "nvfp4 after mxfp4")


def check_prefill_nvfp4_rejections():
    c = nv_case([64], [300])

    def bad(what, fn, match):
        try:
            fn()
        except ValueError as e:
            assert match in str(e), f"{what}: wrong message: {e}"
            return
        raise AssertionError(f"{what}: no ValueError")

    kd4, ks4 = c["kd"], c["ks"][:, :, :4]  # an MXFP4-shaped scale view
    bad("mx scale", lambda: run_nv(c, views=(kd4, ks4)), "[pages, 128, 8]")
    bad("k_round", lambda: run_nv(c, k_round="bf16"), "k_round")
    bad(
        "scale dtype",
        lambda: run_nv(c, views=(c["kd"], c["ks"].view(torch.int8))),
        "k_scale",
    )
    bad(
        "sm_scale",
        lambda: score_prefill_flydsl_nvfp4(
            c["q8"],
            c["kd"],
            c["ks"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            64,
            300,
            -1.0,
        ),
        "sm_scale",
    )


def _prefill_bytes(batch, q_len, ctx_len, k_row_bytes, q_elem_bytes):
    pairs, pages = _prefill_work(batch, q_len, ctx_len)
    return (
        pages * P * k_row_bytes  # K, read once at best
        + batch * q_len * D * q_elem_bytes  # Q
        + pairs * 4  # the fp32 scores written
    )


@benchmark()
def test_prefill_fp4(batch, q_len, ctx_len):
    c, kd, ks = fp4_prefill_case(batch, q_len, ctx_len)
    ref, live = aiter_scores(c)
    ret = {"gfx": get_gfx_runtime()}
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


@benchmark()
def test_prefill_nvfp4(batch, q_len, ctx_len):
    # exact_sum data: every partial sum is exact in fp32, so the torch oracle and the
    # kernel agree to the bit whatever the accumulation order, and err must be 0.
    c = nv_case([q_len] * batch, [ctx_len] * batch, exact_sum=True)
    ret = {"gfx": get_gfx_runtime()}
    heads, rows = c["q8"].shape[1], c["q8"].shape[0]
    pairs, _ = _prefill_work(batch, q_len, ctx_len)
    flops = 2 * pairs * P * D
    k_row = D // 2 + D // 16  # e2m1 data + one e4m3 scale per 16 elements per token
    nbytes = _prefill_bytes(batch, q_len, ctx_len, k_row, 1)  # Q is e4m3
    for k_round in ("e4m3", "exact"):
        ref = nv_oracle(c, nv_ref_of(c, k_round))  # reference only: not timed
        live = torch.isfinite(ref)
        # Preallocated, sentinel-filled score buffer: the timed region is the scorer
        # alone, and an unwritten live slot fails the comparison.
        out = prefill_alloc_score(rows, heads, c["mb"], "cuda")
        out.fill_(SENTINEL)
        got, us = run_perftest(
            lambda out=out, k_round=k_round: run_nv(c, "e4m3", k_round, out=out),
            num_iters=10,
            num_rotate_args=1,
        )
        assert not bool((live & (got == SENTINEL)).any()), f"{k_round}: unwritten slot"
        err = checkAllclose(
            ref[live], got[live], rtol=0, atol=0, msg=f"nvfp4_{k_round}: prefill"
        )
        assert err == 0, f"nvfp4_{k_round}: mismatch fraction {err} against the oracle"
        ret[f"nvfp4_{k_round} us"] = us
        ret[f"nvfp4_{k_round} TFLOPS"] = flops / us / 1e6
        ret[f"nvfp4_{k_round} TB/s"] = nbytes / us / 1e6
        ret[f"nvfp4_{k_round} err"] = err
    return ret


QDTS = ["bf16", "e4m3"]  # the dtype the query reaches the scorer in


def build_cases():
    """(name, check, args): one row per case."""
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
    cases.append(
        ("geometry_arms_are_exhaustive", check_prefill_geometry_arms_are_exhaustive, ())
    )
    for qdt in QDTS:
        cases.append((f"typed_views_{qdt}", check_prefill_typed_views, (qdt,)))
        cases.append((f"two_heads_{qdt}", check_prefill_two_heads, (qdt,)))
    for ctx in [1, 127, 128, 129, 1000, 4097]:
        cases.append((f"ragged_tails_ctx{ctx}", check_prefill_ragged_tails, (ctx,)))
    for qdt in QDTS:
        cases.append((f"scale_extremes_{qdt}", check_prefill_scale_extremes, (qdt,)))
        cases.append(
            (
                f"scale_bytes_beyond_e4m3_{qdt}",
                check_prefill_scale_bytes_beyond_e4m3,
                (qdt,),
            )
        )
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
    # NVFP4, a minimal matrix (the full regression matrix lives outside this file):
    # exact-sum data to the bit against the torch oracle for both k_round values, on
    # tile and page boundaries, ragged and empty requests, H = 1/4, both Q dtypes, the
    # 16- and 32-wide MFMA, the 512-row DMA tile and the 8-wave tile; random data
    # against aiter's fp8 scorer.
    tails = prefill_tail_cases()
    short = [
        tails[3],
        tails[5],
        tails[15],
    ]  # a page tail, a tile tail, an empty request
    wide = [([513], [513]), ([1100, 1, 600], [5000, 4099, 2500])]  # 512-row tail tiles
    for (qls, seqs), heads, qdt, m32 in itertools.product(short, (1, 4), QDTS, (0, 1)):
        name = f"q{'_'.join(map(str, qls))}_seq{'_'.join(map(str, seqs))}_H{heads}_{qdt}_m32{m32}"
        cases.append(
            (
                f"nvfp4_tails_{name}",
                check_prefill_nvfp4_tails,
                (qls, seqs, heads, qdt, m32),
            )
        )
    for (qls, seqs), heads, qdt, ppw in itertools.product(
        wide, (1, 4), QDTS, (None, 4)
    ):
        name = f"q{'_'.join(map(str, qls))}_seq{'_'.join(map(str, seqs))}_H{heads}_{qdt}_ppw{ppw}"
        cases.append(
            (
                f"nvfp4_xwide_{name}",
                check_prefill_nvfp4_xwide,
                (qls, seqs, heads, qdt, ppw),
            )
        )
        cases.append(
            (
                f"nvfp4_exact_wide_{name}",
                check_prefill_nvfp4_exact_wide,
                (qls, seqs, heads, qdt, ppw or 2),
            )
        )
    for (qls, seqs), heads, qdt in itertools.product(short, (1, 4), QDTS):
        name = (
            f"q{'_'.join(map(str, qls))}_seq{'_'.join(map(str, seqs))}_H{heads}_{qdt}"
        )
        cases.append(
            (
                f"nvfp4_exact_tails_{name}",
                check_prefill_nvfp4_exact_tails,
                (qls, seqs, heads, qdt),
            )
        )
    for shape in (GEOM_SHAPES[2], GEOM_SHAPES[5]):  # a 128-row and the 8-wave tile
        name = f"b{shape[0]}_q{shape[1]}_ctx{shape[2]}"
        cases.append(
            (
                f"nvfp4_every_geometry_{name}",
                check_prefill_nvfp4_every_geometry,
                (shape, "e4m3", None),
            )
        )
        cases.append(
            (
                f"nvfp4_exact_every_shape_{name}",
                check_prefill_nvfp4_exact_every_shape,
                (shape, "e4m3"),
            )
        )
    for shape, qdt in itertools.product([SHAPES[2], SHAPES[4], SHAPES[6]], QDTS):
        name = f"b{shape[0]}_q{shape[1]}_ctx{shape[2]}_{qdt}"
        cases.append(
            (
                f"nvfp4_matches_aiter_{name}",
                check_prefill_nvfp4_matches_aiter,
                (shape, qdt),
            )
        )
        cases.append(
            (
                f"nvfp4_exact_random_{name}",
                check_prefill_nvfp4_exact_random,
                (shape, qdt),
            )
        )
    for fn in (
        check_prefill_nvfp4_single_region_views_and_typed_scale,
        check_prefill_nvfp4_default_k_round_is_e4m3,
        check_prefill_nvfp4_modes_do_not_share_a_compiled_kernel,
        check_prefill_nvfp4_rejections,
    ):
        cases.append((fn.__name__.removeprefix("check_prefill_"), fn, ()))
    return cases


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MXFP4 and NVFP4 prefill index score checks + sweep",
    )
    ap.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="run only the correctness cases whose name contains one of these\n"
        "e.g.: --case rejections scale_\n",
    )
    ap.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--q-len", type=int, nargs="*", default=[512, 4096])
    ap.add_argument("--ctx-len", type=int, nargs="*", default=[8192, 32768])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("requires %s; skipping", SUPPORTED_GFX)
        return 0
    aiter.logger.info("mxfp4 and nvfp4 prefill: running correctness checks...")
    rows = []
    for name, check, cargs in build_cases():
        if args.case is not None and not any(k in name for k in args.case):
            continue
        try:
            check(*cargs)
            rows.append({"case": name, "status": "PASS", "detail": ""})
        except AssertionError as e:  # list every failed check; a HIP RuntimeError or
            # an unexpected ValueError propagates, so the run stops on a bad context
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
    perf = [
        test_prefill_fp4(b, q, c)
        for b, q, c in itertools.product(args.batch, args.q_len, args.ctx_len)
        if q <= c
    ]
    aiter.logger.info(
        "mxfp4 prefill summary (markdown):\n%s",
        pd.DataFrame(perf).to_markdown(index=False),
    )
    # A second table, not a merged one: the benchmark signatures differ in columns.
    nv = [
        test_prefill_nvfp4(b, q, c)
        for b, q, c in itertools.product(args.batch, args.q_len, args.ctx_len)
        if q <= c
    ]
    aiter.logger.info(
        "nvfp4 prefill summary (markdown):\n%s",
        pd.DataFrame(nv).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
