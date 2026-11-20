# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MXFP8 x MXFP4 routed experts (hidden 5120, inter 2304): correctness and perf.

Against a torch reference of the contract (``moe_a8w4_experts`` module docstring):
``checkAllclose`` per element, cosine >= 0.99999 over the output and >= 0.9999
per row (the kernels accumulate in another order, so a bf16 slot or an e4m3 code
of the intermediate may differ near a rounding tie), two runs and graph replays
bit for bit (also with new inputs copied into the captured buffers), the fused
launch equal to the unfused GEMM1 / GEMM2 / combine launches bit for bit.

    pytest -q op_tests/flydsl_tests/test_flydsl_moe_a8w4_experts.py   # correctness (CI)
    python3 op_tests/flydsl_tests/test_flydsl_moe_a8w4_experts.py     # perf sweep table
"""

import argparse
import itertools

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import moe_a8w4_experts as mod
from aiter.ops.flydsl.moe_a8w4_experts import (
    A8W4ExpertWeights,
    flydsl_moe_a8w4_experts,
    flydsl_moe_a8w4_experts_rows,
    flydsl_moe_a8w4_experts_warmup,
    flydsl_moe_a8w4_experts_workspace,
    moe_a8w4_experts_supported,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() not in SUPPORTED_GFX,
    reason="gfx950 only",
)

H, I = 5120, 2304
LIMIT = 10.0
COS_MIN = 0.99999
ROW_COS_MIN = 0.9999
ERR_MAX = 1e-3  # fraction of elements outside RTOL / ATOL_ROW
ROW_ERR_MAX = 0.05  # the same, in any one row
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------- reference
def dequant_mxfp4(codes, scales):
    """Row-major MXFP4 (``[.., K/2]`` u8, low nibble first; ``[.., K/32]`` e8m0) -> f32."""
    grid = torch.tensor(_GRID, device=codes.device)
    lut = torch.cat([grid, -grid])
    v = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1)
    v = v.flatten(-2)
    return v * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def dequant_mxfp8(codes, scales):
    m, k = codes.shape
    v = codes.view(torch.float8_e4m3fn).float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scales.float() - 127.0).unsqueeze(-1)).reshape(m, k)


def quantize_mxfp8(x):
    """e4m3 codes and e8m0 = exponent of ceil_pow2(amax * fp32(1/448)), at least 1."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    u = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp_min(1)
    q = blocks * torch.exp2(127.0 - e.float()).unsqueeze(-1)
    q = q.clamp(-448.0, 448.0).reshape(m, k)
    return q.to(torch.float8_e4m3fn), e.to(torch.uint8)


def _expert(x, w13, s13, w2, s2):
    gu = x @ dequant_mxfp4(w13, s13).t()
    act = torch.nn.functional.silu(gu[:, :I].clamp(max=LIMIT))
    act = act * gu[:, I:].clamp(-LIMIT, LIMIT)
    q = dequant_mxfp8(*quantize_mxfp8(act))
    return q @ dequant_mxfp4(w2, s2).t()


def ordered_sum(slots):
    acc = slots[:, 0].float()
    for s in range(1, slots.shape[1]):
        acc = acc + slots[:, s].float()
    return acc.bfloat16()


def experts_reference(xc, xs, raw, wt, ids, return_slots=False):
    w13, s13, w2, s2 = raw
    x = dequant_mxfp8(xc, xs)
    slots = torch.zeros((x.shape[0], ids.shape[1], H), dtype=torch.bfloat16)
    slots = slots.to(x.device)
    for e in ids.unique().tolist():
        if not 0 <= e < w13.shape[0]:
            continue
        t, s = torch.nonzero(ids == e, as_tuple=True)
        y = _expert(x[t], w13[e], s13[e], w2[e], s2[e])
        slots[t, s] = (y * wt[t, s].float().unsqueeze(1)).bfloat16()
    return slots if return_slots else ordered_sum(slots)


def experts_rows_reference(xc, xs, raw, row_weights, offsets):
    w13, s13, w2, s2 = raw
    x = dequant_mxfp8(xc, xs)
    out = torch.zeros((x.shape[0], H), dtype=torch.bfloat16, device=x.device)
    offs = offsets.tolist()
    for e in range(len(offs) - 1):
        if offs[e + 1] > offs[e]:
            r = slice(offs[e], offs[e + 1])
            y = _expert(x[r], w13[e], s13[e], w2[e], s2[e])
            out[r] = (y * row_weights[r].float().unsqueeze(1)).bfloat16()
    return out


# ---------------------------------------------------------------------- inputs
def make_experts(e, seed=0):
    """Random MXFP4 experts (checkpoint layouts), scales over 6 binades."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(
            lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    return (
        rnd(0, 256, (e, 2 * I, H // 2)),
        rnd(118, 124, (e, 2 * I, H // 32)),
        rnd(0, 256, (e, H, I // 2)),
        rnd(118, 124, (e, H, I // 32)),
    )


def make_activation(n, seed=0, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, H), device="cuda", generator=g) * scale
    return quantize_mxfp8(x.bfloat16())


def make_routes(n, topk, e, dist="uniform", seed=0):
    """``topk`` distinct experts per token. ``skew``: odds ~ 1 / (rank + 1)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.zeros(e, device="cuda")
    if dist == "skew":
        order = torch.randperm(e, device="cuda", generator=g)
        ranks = torch.arange(1, e + 1, device="cuda").float()
        logits[order] = -torch.log(ranks)
    u = torch.rand((n, e), device="cuda", generator=g).clamp_min(1e-20)
    ids = (logits - torch.log(-torch.log(u))).topk(topk, dim=1).indices.int()
    wt = torch.softmax(torch.randn((n, topk), device="cuda", generator=g), -1)
    return ids, wt


def make_sorted_rows(n_rows, topk, e, dist="uniform", seed=0):
    ids, _ = make_routes(n_rows // topk, topk, e, dist, seed)
    counts = torch.bincount(ids.flatten().long(), minlength=e)
    zero = torch.zeros(1, dtype=torch.long, device="cuda")
    offsets = torch.cat([zero, counts.cumsum(0)]).int()
    g = torch.Generator(device="cuda").manual_seed(seed + 77)
    return offsets, torch.rand(n_rows, device="cuda", generator=g) + 0.1


def cosine(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


# Per element: bf16 rounding (rtol) plus one e4m3 step of the intermediate near a
# rounding tie, which moves an output by a small fraction of its row's scale (atol).
RTOL, ATOL_ROW = 2e-2, 2e-2


def _per_row_rms(expected):
    exp = expected.float()
    return exp, exp.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)


def allclose_err(got, expected, msg=""):
    """``checkAllclose`` mismatch fraction, atol scaled per row by the row's RMS."""
    exp, rms = _per_row_rms(expected)
    return checkAllclose(
        exp / rms,
        got.float() / rms,
        rtol=RTOL,
        atol=ATOL_ROW,
        msg=msg,
        printLog=False,
    )


def worst_row_err(got, expected):
    """The largest fraction of one row's elements outside RTOL / ATOL_ROW."""
    exp, rms = _per_row_rms(expected)
    close = torch.isclose(got.float() / rms, exp / rms, rtol=RTOL, atol=ATOL_ROW)
    return (~close).float().mean(-1).max().item() if close.numel() else 0.0


def check(got, expected):
    assert torch.isfinite(got.float()).all()
    err = allclose_err(got, expected)
    assert err <= ERR_MAX, f"{err:.3%} of elements outside rtol/atol"
    # an e4m3 code of the intermediate flipped near a tie moves < 1 % of a row;
    # a row read from the wrong token, slot or expert moves nearly all of it
    assert worst_row_err(got, expected) <= ROW_ERR_MAX
    assert cosine(got, expected) >= COS_MIN
    rows = expected.double().norm(dim=-1) > 0
    if rows.any():
        row_cos = torch.nn.functional.cosine_similarity(
            got.double()[rows], expected.double()[rows], dim=-1
        )
        assert row_cos.min() >= ROW_COS_MIN


# (local experts, top-k): an EP8 rank of 384 experts top-6, and of 128 top-3
SHAPES = ((48, 6), (16, 3))
_cache = {}


def experts_of(e, seed=0):
    key = (e, seed)
    if key not in _cache:
        _cache.clear()
        torch.cuda.empty_cache()
        raw = make_experts(e, seed)
        _cache[key] = raw, A8W4ExpertWeights.from_row_major(*raw)
    return _cache[key]


# ---------------------------------------------------------------------- tests
def test_supported_gate():
    assert moe_a8w4_experts_supported(5120, 2304, 48, 6, 1024)
    assert moe_a8w4_experts_supported(5120, 2304, 16, 3, 1)
    assert moe_a8w4_experts_supported(5120, 2304, 63, 6)
    assert not moe_a8w4_experts_supported(5120, 2304, 48, 6, 1025)
    assert not moe_a8w4_experts_supported(5120, 2304, 48, 8)
    assert not moe_a8w4_experts_supported(5120, 2304, 64, 6)
    assert not moe_a8w4_experts_supported(7168, 2304, 48, 6)
    assert not moe_a8w4_experts_supported(5120, 2048, 48, 6)
    assert not moe_a8w4_experts_supported(5120, 2304, 48, 6, swiglu_limit=7.0)
    assert not moe_a8w4_experts_supported(5120, 2304, 48, 6, swiglu_limit=None)


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", [16, 64, 192, 384, 768])
@torch.inference_mode()
def test_matches_reference_repeats_and_replays(e, topk, dist, n):
    raw, w = experts_of(e)
    xc, xs = make_activation(n, seed=n)
    ids, wt = make_routes(n, topk, e, dist, seed=n)
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))
    assert torch.equal(got, flydsl_moe_a8w4_experts(xc, xs, w, wt, ids))
    # fused launch == unfused GEMM1 / GEMM2 / combine; counters back to zero
    slots = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, return_slots=True)
    assert torch.equal(got, mod.combine_slots(slots, ids, e))
    assert not mod._ready_counters(xc.device).any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, got)


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize(
    "n", [1, 2, 7, 17, 33, 49, 65, 101, 129, 161, 193, 250, 401, 449, 500, 1024]
)
@torch.inference_mode()
def test_every_dispatch_tier(e, topk, n):
    raw, w = experts_of(e)
    xc, xs = make_activation(n, seed=100 + n)
    ids, wt = make_routes(n, topk, e, "uniform", seed=100 + n)
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))


@pytest.mark.parametrize("e,topk", SHAPES)
@torch.inference_mode()
def test_slots_are_bf16_weighted_per_route(e, topk):
    raw, w = experts_of(e)
    n = 40
    xc, xs = make_activation(n, seed=3)
    ids, wt = make_routes(n, topk, e, "uniform", seed=3)
    slots = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, return_slots=True)
    assert slots.shape == (n, topk, H) and slots.dtype == torch.bfloat16
    check(slots, experts_reference(xc, xs, raw, wt, ids, return_slots=True))
    assert torch.equal(mod.combine_slots(slots, ids, e), ordered_sum(slots))


@torch.inference_mode()
def test_swiglu_clamp_and_zero_rows():
    """~40x activations push gate and up past the limit; zero rows give zero outputs."""
    raw, w = experts_of(48)
    n = 192
    xc, xs = make_activation(n, seed=9, scale=40.0)
    xc.view(torch.uint8)[::5] = 0
    xs[::5] = 0
    ids, wt = make_routes(n, 6, 48, "uniform", seed=9)
    x = dequant_mxfp8(xc, xs)
    gate = x[:8] @ dequant_mxfp4(raw[0][ids[0, 0]], raw[1][ids[0, 0]])[:I].t()
    assert gate.abs().max() > 4 * LIMIT  # the clamp is exercised
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))
    assert got[::5].count_nonzero() == 0


@pytest.mark.parametrize("n", [16, 128, 384])
@torch.inference_mode()
def test_all_tokens_on_the_same_experts(n):
    """One hot expert set: 6 experts with n rows each, 42 idle."""
    raw, w = experts_of(48)
    xc, xs = make_activation(n, seed=n + 1)
    _, wt = make_routes(n, 6, 48, "uniform", seed=n + 1)
    ids = torch.arange(6, device="cuda", dtype=torch.int32).repeat(n, 1)
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [16, 192])
@torch.inference_mode()
def test_ids_outside_the_local_experts_are_not_routed(e, topk, n):
    """Padding (-1) and other ranks' experts (>= e) contribute zero."""
    raw, w = experts_of(e)
    xc, xs = make_activation(n, seed=n + 7)
    ids, wt = make_routes(n, topk, e, "uniform", seed=n + 7)
    ids[::4] = -1
    ids[1::4, 1:] += e
    ids[2::4, 0] = 8 * e - 1
    wt[::4] = 0
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))
    assert got[::4].count_nonzero() == 0


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [1, 100, 384])
@torch.inference_mode()
def test_bf16_input_is_quantized_bit_exactly(e, topk, n):
    """The routing launch's MXFP8 quant equals ``quantize_mxfp8``: same outputs."""
    _, w = experts_of(e)
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn((n, H), device="cuda", generator=g).bfloat16()
    x[0] = 0
    xc, xs = quantize_mxfp8(x)
    ids, wt = make_routes(n, topk, e, "uniform", seed=n)
    assert torch.equal(
        flydsl_moe_a8w4_experts(x, None, w, wt, ids),
        flydsl_moe_a8w4_experts(xc, xs, w, wt, ids),
    )


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", [16, 192, 768])
@torch.inference_mode()
def test_rows_match_reference_and_replay(e, topk, dist, n):
    raw, w = experts_of(e)
    rows = n * topk
    offsets, weights = make_sorted_rows(rows, topk, e, dist, seed=n)
    xc, xs = make_activation(rows, seed=n + 5)
    got = flydsl_moe_a8w4_experts_rows(xc, xs, w, weights, offsets, topk)
    check(got, experts_rows_reference(xc, xs, raw, weights, offsets))
    assert torch.equal(
        got, flydsl_moe_a8w4_experts_rows(xc, xs, w, weights, offsets, topk)
    )
    assert not mod._ready_counters(xc.device).any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = flydsl_moe_a8w4_experts_rows(xc, xs, w, weights, offsets, topk)
    replayed.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, got)


@torch.inference_mode()
def test_rows_are_the_slots_of_the_token_entry():
    """A token's routes sorted by expert give the token entry's slots bit for bit."""
    _, w = experts_of(48)
    n = 96
    xc, xs = make_activation(n, seed=1)
    ids, wt = make_routes(n, 6, 48, "skew", seed=1)
    slots = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, return_slots=True)
    order = torch.argsort(ids.flatten().long(), stable=True)
    token = order // 6
    counts = torch.bincount(ids.flatten().long(), minlength=48)
    zero = torch.zeros(1, dtype=torch.long, device="cuda")
    offsets = torch.cat([zero, counts.cumsum(0)]).int()
    got = flydsl_moe_a8w4_experts_rows(
        xc[token], xs[token], w, wt.flatten()[order], offsets
    )
    assert torch.equal(got, slots.view(-1, H)[order])


@torch.inference_mode()
def test_rows_up_to_the_row_count():
    """A buffer larger than the received rows, 47 empty experts: rows below
    ``expert_offsets[-1]`` are checked; the rows after it are undefined."""
    raw, w = experts_of(48)
    valid, cap = 100, 150
    offsets = torch.zeros(49, dtype=torch.int32, device="cuda")
    offsets[8:] = valid
    xc, xs = make_activation(cap, seed=4)
    weights = torch.rand(cap, device="cuda") + 0.1
    got = flydsl_moe_a8w4_experts_rows(xc, xs, w, weights, offsets)
    expected = experts_rows_reference(xc, xs, raw, weights, offsets)
    check(got[:valid], expected[:valid])


@pytest.mark.parametrize("e", [52, 56])
@pytest.mark.parametrize("n", [16, 192, 449])
@torch.inference_mode()
def test_redundant_experts(e, n):
    """EPLB: more local experts than the even split (the count is the weights' dim 0)."""
    raw, w = experts_of(e, seed=3)
    assert w.num_experts == e
    xc, xs = make_activation(n, seed=n)
    ids, wt = make_routes(n, 6, e, "skew", seed=n)
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    check(got, experts_reference(xc, xs, raw, wt, ids))
    slots = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, return_slots=True)
    assert torch.equal(got, mod.combine_slots(slots, ids, e))


@torch.inference_mode()
def test_warmup_compiles_every_tier_before_capture():
    """After the warm-up a capture at any count compiles nothing new."""
    _, w = experts_of(16)
    flydsl_moe_a8w4_experts_warmup(w, 3)
    sizes = (
        mod.compile_experts.cache_info().currsize,
        mod.compile_sort.cache_info().currsize,
    )
    for n in (1, 5, 80, 333, 1000):
        xc, xs = make_activation(n, seed=n)
        ids, wt = make_routes(n, 3, 16, "uniform", seed=n)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
        graph.replay()
    torch.cuda.synchronize()
    assert sizes == (
        mod.compile_experts.cache_info().currsize,
        mod.compile_sort.cache_info().currsize,
    )


# ---------------------------------------------------------------- graph replay with new inputs
def _replay_tokens(e, topk, n, bf16_input):
    raw, w = experts_of(e)
    flydsl_moe_a8w4_experts_warmup(w, topk, bf16_input=bf16_input)

    def inputs(seed, dist):
        ids, wt = make_routes(n, topk, e, dist, seed=seed)
        if bf16_input:
            g = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn((n, H), device="cuda", generator=g).bfloat16()
            return [x, None, wt, ids]
        return [*make_activation(n, seed=seed), wt, ids]

    buf = inputs(1, "uniform")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flydsl_moe_a8w4_experts(buf[0], buf[1], w, buf[2], buf[3])
    for seed, dist in ((2, "skew"), (3, "uniform"), (4, "skew")):
        new = inputs(seed, dist)
        for b, v in zip(buf, new):
            if b is not None:
                b.copy_(v)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        eager = flydsl_moe_a8w4_experts(new[0], new[1], w, new[2], new[3])
        assert torch.equal(out, eager)
        xc, xs = quantize_mxfp8(new[0]) if bf16_input else (new[0], new[1])
        check(out, experts_reference(xc, xs, raw, new[2], new[3]))
        assert not mod._ready_counters(out.device).any()


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [16, 100, 192, 449, 1024])
@torch.inference_mode()
def test_graph_replay_new_inputs(e, topk, n):
    """New activations, ids and weights copied into the captured buffers before each
    replay: equal to eager on the same inputs bit for bit, and to the reference."""
    _replay_tokens(e, topk, n, bf16_input=False)


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [1, 77, 384])
@torch.inference_mode()
def test_graph_replay_new_inputs_bf16(e, topk, n):
    _replay_tokens(e, topk, n, bf16_input=True)


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [16, 60, 192])
@torch.inference_mode()
def test_rows_graph_replay_new_offsets(e, topk, n):
    """Rows entry: new rows, weights and offsets (fewer valid rows than the buffer)."""
    raw, w = experts_of(e)
    flydsl_moe_a8w4_experts_warmup(w, topk, rows=True)
    rows = n * topk
    offsets, weights = make_sorted_rows(rows, topk, e, "uniform", seed=1)
    xc, xs = make_activation(rows, seed=8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flydsl_moe_a8w4_experts_rows(xc, xs, w, weights, offsets, topk)
    for seed, drop in ((9, 7), (10, 0), (11, rows // 3)):
        o2, w2 = make_sorted_rows(rows, topk, e, "skew", seed=seed)
        o2 = torch.minimum(o2, o2[-1] - drop)
        c2, s2 = make_activation(rows, seed=seed + 1)
        for b, v in ((offsets, o2), (weights, w2), (xc, c2), (xs, s2)):
            b.copy_(v)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        valid = int(o2[-1])
        eager = flydsl_moe_a8w4_experts_rows(c2, s2, w, w2, o2, topk)
        assert torch.equal(out[:valid], eager[:valid])
        ref = experts_rows_reference(c2, s2, raw, w2, o2)
        check(out[:valid], ref[:valid])
        assert not mod._ready_counters(out.device).any()


@torch.inference_mode()
def test_one_graph_alternates_layer_shapes():
    """One graph alternating E48 / top-6 and E16 / top-3 calls (the counters are shared)."""
    _, w48 = experts_of(48)
    w16 = A8W4ExpertWeights.from_row_major(*make_experts(16, seed=7))
    flydsl_moe_a8w4_experts_warmup(w48, 6)
    flydsl_moe_a8w4_experts_warmup(w16, 3)
    xa, sa = make_activation(200, seed=20)
    ia, wa = make_routes(200, 6, 48, "skew", seed=20)
    xb, sb = make_activation(40, seed=21)
    ib, wb = make_routes(40, 3, 16, "uniform", seed=21)
    ea = flydsl_moe_a8w4_experts(xa, sa, w48, wa, ia)
    eb = flydsl_moe_a8w4_experts(xb, sb, w16, wb, ib)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = []
        for _ in range(4):
            outs.append(flydsl_moe_a8w4_experts(xa, sa, w48, wa, ia))
            outs.append(flydsl_moe_a8w4_experts(xb, sb, w16, wb, ib))
    for _ in range(5):
        for o in outs:
            o.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(o, ea if i % 2 == 0 else eb) for i, o in enumerate(outs))
    assert not mod._ready_counters(xa.device).any()


@pytest.mark.parametrize("e,topk", SHAPES)
@pytest.mark.parametrize("n", [1, 16, 300])
@torch.inference_mode()
def test_all_ids_invalid(e, topk, n):
    """No route on this rank: zero output, the counters stay zero."""
    _, w = experts_of(e)
    xc, xs = make_activation(n, seed=11)
    ids = torch.full((n, topk), -1, dtype=torch.int32, device="cuda")
    wt = torch.rand((n, topk), device="cuda")
    out = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    torch.cuda.synchronize()
    assert out.count_nonzero() == 0
    assert not mod._ready_counters(xc.device).any()


@pytest.mark.parametrize("e,topk", SHAPES)
@torch.inference_mode()
def test_duplicate_ids_int64_ids_bf16_weights(e, topk):
    """A token naming one expert twice, int64 ids and bf16 route weights."""
    raw, w = experts_of(e)
    n = 50
    xc, xs = make_activation(n, seed=12)
    ids, wt = make_routes(n, topk, e, "uniform", seed=12)
    ids[:, 1] = ids[:, 0]
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt.bfloat16(), ids.long())
    check(got, experts_reference(xc, xs, raw, wt.bfloat16().float(), ids))


@torch.inference_mode()
def test_empty_inputs_and_rejected_dtypes():
    _, w = experts_of(16)
    xc, xs = make_activation(4, seed=1)
    ids, wt = make_routes(4, 3, 16, seed=1)
    out = flydsl_moe_a8w4_experts(xc[:0], xs[:0], w, wt[:0], ids[:0])
    assert out.shape == (0, H) and out.dtype == torch.bfloat16
    offsets = torch.zeros(17, dtype=torch.int32, device="cuda")
    out = flydsl_moe_a8w4_experts_rows(xc[:0], xs[:0], w, wt.flatten()[:0], offsets, 3)
    assert out.shape == (0, H)
    with pytest.raises(AssertionError):
        flydsl_moe_a8w4_experts(xc.view(torch.int8), xs, w, wt, ids)
    with pytest.raises(AssertionError):
        flydsl_moe_a8w4_experts(xc.view(torch.float8_e4m3fnuz), xs, w, wt, ids)


# ---------------------------------------------------------------- ready-counter workspaces
@torch.inference_mode()
def test_concurrent_streams_with_own_workspaces():
    """Two streams, each with its own workspace, run at the same time."""
    _, w = experts_of(48)
    n = 192
    xa, sa = make_activation(n, seed=30)
    ia, wa = make_routes(n, 6, 48, "skew", seed=30)
    xb, sb = make_activation(n, seed=31)
    ib, wb = make_routes(n, 6, 48, "uniform", seed=31)
    ea = flydsl_moe_a8w4_experts(xa, sa, w, wa, ia)
    eb = flydsl_moe_a8w4_experts(xb, sb, w, wb, ib)
    work = [flydsl_moe_a8w4_experts_workspace(xa.device) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = [[], []]
    for _ in range(20):
        for i, (s, x, xs, wt, ids) in enumerate(
            zip(streams, (xa, xb), (sa, sb), (wa, wb), (ia, ib))
        ):
            with torch.cuda.stream(s):
                outs[i].append(
                    flydsl_moe_a8w4_experts(x, xs, w, wt, ids, ready_counters=work[i])
                )
    torch.cuda.synchronize()
    assert all(torch.equal(o, ea) for o in outs[0])
    assert all(torch.equal(o, eb) for o in outs[1])
    assert not any(t.any() for t in work)


@torch.inference_mode()
def test_counter_check_reports_a_dirty_workspace(monkeypatch):
    """``AITER_FLYDSL_MOE_A8W4_CHECK=1`` raises on a workspace a concurrent call left non-zero."""
    _, w = experts_of(16)
    xc, xs = make_activation(16, seed=2)
    ids, wt = make_routes(16, 3, 16, seed=2)
    work = flydsl_moe_a8w4_experts_workspace(xc.device)
    monkeypatch.setenv("AITER_FLYDSL_MOE_A8W4_CHECK", "1")
    flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, ready_counters=work)
    work[0] = 1  # what an overlapping call on the same workspace can leave behind
    with pytest.raises(RuntimeError, match="not zero before"):
        flydsl_moe_a8w4_experts(xc, xs, w, wt, ids, ready_counters=work)


# ---------------------------------------------------------------- sort and GEMM1 alone
@pytest.mark.parametrize("bm", [16, 32, 64, 128])
@pytest.mark.parametrize("n", [1, 16, 100, 768])
@torch.inference_mode()
def test_sort_places_every_pair_once(n, bm):
    e, topk = 48, 6
    ids, wt = make_routes(n, topk, e, "skew", seed=n)
    sid, wts, eids, nv = mod._sort(ids, wt, bm, e)
    torch.cuda.synchronize()
    rows = int(nv)
    counts = torch.bincount(ids.flatten().long(), minlength=e)
    assert rows == int(((counts + bm - 1) // bm * bm).sum())
    token, slot = (sid[:rows] & 0xFFFFFF).long(), (sid[:rows] >> 24).long()
    real = token < n
    assert int(real.sum()) == n * topk
    block_expert = eids[: rows // bm].long().repeat_interleave(bm)
    assert torch.equal(ids[token[real], slot[real]].long(), block_expert[real])
    assert torch.equal(wts[:rows][real], wt[token[real], slot[real]])
    assert (wts[:rows][~real] == 0).all()
    pairs = {(int(t), int(s)) for t, s in zip(token[real], slot[real])}
    assert len(pairs) == n * topk


def unpack_rows(q, s, rows, k=I):
    """GEMM1's output layout (GEMM2's operand layout) -> row-major codes ``[rows, k]``
    and e8m0 ``[rows, k / 32]``. Per 16-row block and 128-K step:
    ``[half 2][lane 64][16 B]`` with lane = 16 * (K % 64 // 16) + row and
    half = K // 64 % 2; the e8m0 words (4 bytes = 4 K groups) as ``[step][row]``."""
    steps = k // 128
    c = q.view(rows // 16, steps, 2, 4, 16, 16).permute(0, 4, 1, 2, 3, 5)
    e = s.view(rows // 16, steps, 16, 4).permute(0, 2, 1, 3)
    return c.reshape(rows, k), e.reshape(rows, k // 32)


# (bm, kw, pf, ni, nbuf, nwv): the dispatch's GEMM1 configs and other legal ones
GEMM1_CASES = sorted(
    {
        (cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], cfg[9])
        for tiers in mod._DISPATCH.values()
        for _, cfg in tiers
    }
    | {
        (16, 2, 3, 2, 0, 4),
        (32, 4, 3, 2, 0, 4),
        (32, 1, 3, 4, 0, 4),
        (64, 2, 2, 4, 0, 4),
        (128, 1, 3, 2, 3, 4),
        (64, 1, 2, 2, 4, 8),
        (128, 1, 2, 2, 3, 8),
    }
)


@pytest.mark.parametrize("case", GEMM1_CASES, ids=str)
@torch.inference_mode()
def test_gemm1_rows(case):
    """GEMM1 + clamp-SwiGLU + MXFP8 requant alone, per sorted row, against the reference."""
    raw, w = experts_of(48)
    bm, kw, pf, ni, nbuf, nwv = case
    n, e, topk = 192, 48, 6
    xc, xs = make_activation(n, seed=11)
    ids, wt = make_routes(n, topk, e, "uniform", seed=11)
    sid, _, eids, nv = mod._sort(ids, wt, bm, e)
    rows = sid.numel()
    exe = mod.compile_gemm1(ne=e, kw=kw, pf=pf, bm=bm, ni=ni, nbuf=nbuf, nwv=nwv)

    def run():
        q = torch.empty(rows * I, dtype=torch.uint8, device="cuda")
        s = torch.empty(rows * (I // 32), dtype=torch.uint8, device="cuda")
        args = (xc.view(torch.uint8).view(-1), xs.view(-1), w.w13.view(-1))
        args += (w.w13_scale.view(-1), q, s, sid, eids, nv, n, rows)
        nmb = mod.max_blocks(n, bm, e, topk)
        mod._launch(exe, xc.device, *args, nmb, torch.cuda.current_stream())
        return q, s

    (q, s), (q2, s2) = run(), run()
    nvalid = int(nv)
    codes, scales = unpack_rows(q, s, rows)
    codes2, scales2 = unpack_rows(q2, s2, rows)
    token, slot = (sid[:nvalid] & 0xFFFFFF).long(), (sid[:nvalid] >> 24).long()
    real = (token < n).nonzero().flatten()
    assert torch.equal(codes[real], codes2[real])
    assert torch.equal(scales[real], scales2[real])
    got = dequant_mxfp8(codes[real].view(torch.float8_e4m3fn), scales[real])
    x = dequant_mxfp8(xc, xs)
    cos = []
    for i, r in enumerate(real.tolist()):
        if i % 7:
            continue
        ex = int(ids[token[r], slot[r]])
        gu = x[token[r]] @ dequant_mxfp4(raw[0][ex], raw[1][ex]).t()
        act = torch.nn.functional.silu(gu[:I].clamp(max=LIMIT))
        act = act * gu[I:].clamp(-LIMIT, LIMIT)
        ref = dequant_mxfp8(*quantize_mxfp8(act[None]))[0]
        cos.append(torch.nn.functional.cosine_similarity(ref, got[i], dim=0).item())
    # a row of 2304 codes can hold one code that the summation order flips (cosine -3e-5)
    assert min(cos) >= 0.9999 and sum(cos) / len(cos) >= 0.999999


# ---------------------------------------------------------------- weight layouts
def w13_layout(w13):
    """``[E, 2I, H/2]`` (gate rows, then up rows) -> the kernel's GEMM1 layout
    ``[E, I/16, 2 (gate, up), H/128, klane 4, nlane 16, 16 B]``."""
    e, n2, half_k = w13.shape
    v = w13.view(e, 2, n2 // 32, 16, half_k // 64, 4, 16)
    return v.permute(0, 2, 1, 4, 5, 3, 6).contiguous()


def w2_layout(w2):
    """``[E, H, I/2]`` -> the kernel's GEMM2 layout ``[E, H/16, I/128, klane 4, nlane 16, 16 B]``."""
    e, n, half_k = w2.shape
    v = w2.view(e, n // 16, 16, half_k // 64, 4, 16)
    return v.permute(0, 1, 3, 4, 2, 5).contiguous()


@torch.inference_mode()
def test_layouts_match_aiter_shuffle():
    """The layouts the kernels read are aiter's ``shuffle_weight_a16w4`` layouts."""
    from aiter.ops.shuffle import shuffle_weight_a16w4

    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randint(
        0, 256, (2, 2 * I, H // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    w2 = torch.randint(
        0, 256, (2, H, I // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    shuffled13 = shuffle_weight_a16w4(w13.clone(), 16, True).view(torch.uint8)
    shuffled2 = shuffle_weight_a16w4(w2.clone(), 16, False).view(torch.uint8)
    assert torch.equal(shuffled13.flatten(), w13_layout(w13).flatten())
    assert torch.equal(shuffled2.flatten(), w2_layout(w2).flatten())


def fused_moe_tensors(raw):
    """The four tensors ``aiter.fused_moe``'s a8w4 path holds for these experts."""
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w13, s13, w2, s2 = raw
    e = w13.shape[0]
    w1 = shuffle_weight_a16w4(w13, 16, True).view(e, 2 * I, H // 2).view(dtypes.fp4x2)
    w2s = shuffle_weight_a16w4(w2, 16, False).view(e, H, I // 2).view(dtypes.fp4x2)
    w1.is_shuffled = w2s.is_shuffled = True
    w1.gate_up_interleave, w2s.gate_up_interleave = True, False
    sc1 = shuffle_scale_a16w4(s13.reshape(-1, H // 32), e, True)
    sc2 = shuffle_scale_a16w4(s2.reshape(-1, I // 32), e, False)
    sc1 = sc1.view(-1, H // 32).view(dtypes.fp8_e8m0)
    sc2 = sc2.view(-1, I // 32).view(dtypes.fp8_e8m0)
    return w1, sc1, w2s, sc2


@pytest.mark.parametrize("n", [16, 128])
@torch.inference_mode()
def test_from_shuffled_wraps_the_fused_moe_tensors(n):
    """``from_shuffled`` on the tensors ``fused_moe`` holds: no copy, same output as
    ``fused_moe`` (a8w4, swiglu_limit 10) up to the bf16 summation order."""
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe

    raw, w_ref = experts_of(48)
    w1, sc1, w2, sc2 = fused_moe_tensors(raw)
    w = A8W4ExpertWeights.from_shuffled(w1, sc1, w2, sc2)
    for ours, theirs in zip((w.w13, w.w13_scale, w.w2, w.w2_scale), (w1, sc1, w2, sc2)):
        assert ours.data_ptr() == theirs.data_ptr()
    for a, b in zip((w.w13, w.w13_scale, w.w2, w.w2_scale), vars(w_ref).values()):
        assert torch.equal(a.flatten(), b.flatten())
    xc, xs = make_activation(n, seed=40)
    ids, wt = make_routes(n, 6, 48, "uniform", seed=40)
    got = flydsl_moe_a8w4_experts(xc, xs, w, wt, ids)
    base = fused_moe(
        xc,
        w1,
        w2,
        wt,
        ids,
        w1_scale=sc1,
        w2_scale=sc2,
        a1_scale=xs.view(dtypes.fp8_e8m0),
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        swiglu_limit=10.0,
        quant_dtype_a=dtypes.fp8,
        dtype=torch.bfloat16,
    )
    assert cosine(got, base) >= 0.9999
    check(got, experts_reference(xc, xs, raw, wt, ids))


# ---------------------------------------------------------------- perf sweep (python3 <file>)
def _bytes_moved(ids, e, n):
    """Weights of the experts this call touches, activations in, output out."""
    per_expert = 2 * I * H // 2 + 2 * I * H // 32 + H * I // 2 + H * I // 32
    valid = ids[(ids >= 0) & (ids < e)]
    touched = int(valid.unique().numel())
    return touched * per_expert + n * (H + H // 32) + n * H * 2


@benchmark()
def bench_moe_a8w4_experts(tokens, experts, topk, dist, x_dtype):
    """One call of the op (MXFP8 or bf16 input) vs the torch reference."""
    raw, w = experts_of(experts)
    xc, xs = make_activation(tokens, seed=tokens)
    if x_dtype == "bf16":
        g = torch.Generator(device="cuda").manual_seed(tokens)
        x = torch.randn((tokens, H), device="cuda", generator=g).bfloat16()
        xc, xs = quantize_mxfp8(x)
        args = (x, None)
    else:
        args = (xc, xs)
    ids, wt = make_routes(tokens, topk, experts, dist, seed=tokens)
    ref = experts_reference(xc, xs, raw, wt, ids)
    candidates = {"flydsl": lambda: flydsl_moe_a8w4_experts(*args, w, wt, ids)}
    flops = 2 * tokens * topk * 3 * I * H
    nbytes = _bytes_moved(ids, experts, tokens)
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = allclose_err(out, ref, msg=f"{name}: moe a8w4 experts ")
        assert err <= ERR_MAX, f"{name}: {err:.3%} of elements outside rtol/atol"
        assert cosine(out, ref) >= COS_MIN
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    import pandas as pd

    import aiter

    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("moe a8w4 experts unsupported on %s", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL a8w4 routed experts: correctness + perf sweep",
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[16, 64, 128, 384, 1024]
    )
    parser.add_argument(
        "-e",
        "--experts_topk",
        type=lambda s: tuple(int(v) for v in s.split(",")),
        nargs="*",
        default=list(SHAPES),
        help="local experts,topk pairs, e.g. 48,6 16,3",
    )
    parser.add_argument("--dist", nargs="*", default=["uniform", "skew"])
    parser.add_argument("--x_dtype", nargs="*", default=["mxfp8", "bf16"])
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    rows = [
        bench_moe_a8w4_experts(t, e, k, d, xd)
        for (e, k), xd, d, t in itertools.product(
            args.experts_topk, args.x_dtype, args.dist, args.tokens
        )
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "moe a8w4 experts summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
