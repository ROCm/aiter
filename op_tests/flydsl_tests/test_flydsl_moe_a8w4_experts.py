# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL MXFP8 x MXFP4 routed experts (hidden 5120, inter 2304).

Against a torch reference of the contract (``moe_a8w4_experts`` module docstring):
cosine >= 0.99999 over the output and >= 0.9999 per row (the kernels accumulate
in another order, so a bf16 slot or an e4m3 code of the intermediate may differ
near a rounding tie), two runs and a graph replay bit for bit, the fused launch
equal to the unfused GEMM1 / GEMM2 / combine launches bit for bit.

    pytest -q op_tests/flydsl_tests/test_flydsl_moe_a8w4_experts.py
"""

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
    moe_a8w4_experts_supported,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="gfx950 only",
)

H, I = 5120, 2304
LIMIT = 10.0
COS_MIN = 0.99999
ROW_COS_MIN = 0.9999
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


def check(got, expected):
    assert torch.isfinite(got.float()).all()
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
def test_rows_beyond_the_row_count_are_untouched():
    """A buffer larger than the received rows, 47 empty experts."""
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
