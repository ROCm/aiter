# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL router GEMM + sqrt-softplus top-3 (128 experts, hidden 5120).

The kernel's logits (its fp32 partials summed in slice order) must be within
fp32 accumulation error of the torch GEMM, and its ids and weights must equal
the torch top-k of those logits, except where the 3rd and 4th ``score + bias``
are within 1e-5 (the order there depends on the summation order). Also: ties
to the lowest expert id, padding rows, logits above the softplus switch, row
strides, the load-time gate layout, caller buffers, graph replay with poisoned
outputs and repeat runs bit for bit.
"""

import pytest
import torch

pytest.importorskip("flydsl")

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.router_sqrtsoftplus_topk import (
    flydsl_router_sqrtsoftplus_topk,
    router_sqrtsoftplus_topk_partials_shape,
    router_sqrtsoftplus_topk_supported,
    shuffle_router_gate,
    warmup_router_sqrtsoftplus_topk,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="gfx950 and FlyDSL are required",
)

H, E, TOPK = 5120, 128, 3
SCALE = 1.5
ROWS = (1, 7, 16, 17, 40, 48, 80, 96, 160, 192, 320, 384, 640, 768, 1024, 2049)
MARGIN = 1e-5


def score_ref(logits):
    return torch.where(logits > 20.0, logits, torch.log(1.0 + torch.exp(logits))).sqrt()


def topk_ref(logits, bias, scale=SCALE, is_padding=None):
    score = score_ref(logits.float())
    ids = torch.topk(score + bias, TOPK, dim=-1)[1]
    w = (score + bias).gather(1, ids) - bias[ids]
    total = w[:, 0]
    for k in range(1, TOPK):
        total = total + w[:, k]
    w = w * (scale / total[:, None])
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    return w, ids


def make(m, seed, wscale=0.02):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, H), generator=g, device="cuda") * torch.exp2(
        torch.randint(-2, 3, (m, 1), generator=g, device="cuda").float()
    )
    w = torch.randn((E, H), generator=g, device="cuda") * wscale
    bias = torch.randn(E, generator=g, device="cuda") * 0.1
    return x.bfloat16(), w.bfloat16(), bias


def summed(partials):
    logits = partials[0].clone()
    for p in partials[1:]:
        logits = logits + p
    return logits


def close_rows(logits, bias):
    """Rows whose 3rd and 4th ``score + bias`` are within MARGIN."""
    top = torch.topk(score_ref(logits) + bias, TOPK + 1, dim=-1)[0]
    gap = (top[:, TOPK - 1] - top[:, TOPK]).abs()
    return gap < MARGIN * top[:, TOPK - 1].abs().clamp_min(1.0)


def check(out, logits, bias, is_padding=None):
    tw, ti = out
    rw, ri = topk_ref(logits, bias, SCALE, is_padding)
    ok = ~close_rows(logits, bias)
    assert ok.float().mean().item() > 0.9
    assert torch.equal(ti[ok], ri[ok]), "ids"
    assert bool(
        ((tw[ok] - rw[ok]).abs() <= 2e-6 * rw[ok].abs().clamp_min(1e-3)).all()
    ), "weights"
    live = ok if is_padding is None else ok & ~is_padding
    assert torch.allclose(
        tw[live].sum(-1), torch.full_like(tw[live].sum(-1), SCALE), rtol=1e-5
    )


def run_logits(x, w, bias, **kw):
    m = x.shape[0]
    shape = router_sqrtsoftplus_topk_partials_shape(
        m, E, TOPK, kw.get("weight_shuffled", False)
    )
    partials = torch.full(shape, float("nan"), dtype=torch.float32, device="cuda")
    out = flydsl_router_sqrtsoftplus_topk(
        x, w, bias, TOPK, SCALE, partials=partials, **kw
    )
    torch.cuda.synchronize()
    return out, summed(partials)


@pytest.mark.parametrize("weight_shuffled", [False, True])
@pytest.mark.parametrize("m", ROWS)
def test_contract(m, weight_shuffled):
    x, w, bias = make(m, m + 11 * weight_shuffled)
    gate = shuffle_router_gate(w) if weight_shuffled else w
    out, logits = run_logits(x, gate, bias, weight_shuffled=weight_shuffled)
    ref = torch.mm(x, w.T, out_dtype=torch.float32)
    assert (logits - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()
    check(out, logits, bias)
    again, _ = run_logits(x, gate, bias, weight_shuffled=weight_shuffled)
    assert torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])


def test_shuffle_is_a_permutation():
    _, w, _ = make(1, 3)
    ws = shuffle_router_gate(w)
    back = ws.reshape(E // 16, H // 32, 4, 16, 8).permute(0, 3, 1, 2, 4).reshape(E, H)
    assert torch.equal(back, w)


def test_ties_lowest_id():
    m = 16
    x = torch.zeros((m, H), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((E, H), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(E, device="cuda")
    for i in (127, 5, 100, 7, 64, 9, 3, 11):
        bias[i] = 1.0
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, TOPK, SCALE)
    assert (ti == torch.tensor([3, 5, 7], device="cuda", dtype=torch.int32)).all()
    assert torch.allclose(tw, torch.full_like(tw, SCALE / 3), rtol=1e-6)
    bias.zero_()
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, TOPK, SCALE)
    assert (ti == torch.arange(TOPK, device="cuda", dtype=torch.int32)).all()


@pytest.mark.parametrize("m", (16, 192))
def test_large_logits_and_padding(m):
    x, w, bias = make(m, 9, wscale=0.2)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    pad[::3] = True
    out, logits = run_logits(x, w, bias, is_padding=pad)
    assert (logits > 20).any()
    check(out, logits, bias, pad)
    assert (out[1][pad] == -1).all() and (out[0][pad] == 0).all()
    assert (out[1][~pad] >= 0).all()


def test_row_stride():
    m = 96
    x, w, bias = make(m, 8)
    wide = torch.zeros((m, H + 64), device="cuda", dtype=torch.bfloat16)
    wide[:, :H] = x
    a = flydsl_router_sqrtsoftplus_topk(wide[:, :H], w, bias, TOPK, SCALE)
    b = flydsl_router_sqrtsoftplus_topk(x, w, bias, TOPK, SCALE)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_non_finite_inputs_give_in_range_ids():
    m = 16
    x, w, bias = make(m, 4)
    x[3, 7] = float("nan")
    x[4, 7] = float("inf")
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, TOPK, SCALE)
    torch.cuda.synchronize()
    assert ((ti >= 0) & (ti < E)).all()
    assert torch.isfinite(tw[[0, 1, 2]]).all()


def test_caller_buffers_and_graph_replay():
    m = 192
    x, w, bias = make(m, 5)
    tw = torch.empty((m, TOPK), device="cuda")
    ti = torch.empty((m, TOPK), device="cuda", dtype=torch.int32)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    warmup_router_sqrtsoftplus_topk(x.device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flydsl_router_sqrtsoftplus_topk(
            x, w, bias, TOPK, SCALE, is_padding=pad, topk_weights=tw, topk_ids=ti
        )
    assert out[0] is tw and out[1] is ti
    for i in range(3):
        x.mul_(0.75).add_(0.0625)
        pad[i] = True
        ew, ei = flydsl_router_sqrtsoftplus_topk(
            x, w, bias, TOPK, SCALE, is_padding=pad
        )
        tw.fill_(float("nan"))
        ti.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(tw, ew) and torch.equal(ti, ei)


def test_unsupported_shapes_decline():
    x, w, bias = make(4, 1)
    assert router_sqrtsoftplus_topk_supported(x, w, bias, TOPK)
    assert not router_sqrtsoftplus_topk_supported(x, w, bias, 6)
    w384 = torch.zeros((384, H), dtype=torch.bfloat16, device="cuda")
    b384 = torch.zeros(384, device="cuda")
    assert not router_sqrtsoftplus_topk_supported(x, w384, b384, TOPK)
    assert not router_sqrtsoftplus_topk_supported(x.float(), w, bias, TOPK)
    assert not router_sqrtsoftplus_topk_supported(x, w, bias.bfloat16(), TOPK)
    with pytest.raises(ValueError):
        flydsl_router_sqrtsoftplus_topk(x, w, bias, 6, SCALE)


@pytest.mark.parametrize("m", (1, 16, 192, 768))
def test_topk_matches_vllm_bitwise(m):
    """Given the kernel's own logits, ids and weights equal vLLM's
    ``topk_softplus_sqrt`` bit for bit (optional: needs vLLM)."""
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops

    for seed, wscale in ((m, 0.02), (m + 1, 0.05), (m + 2, 0.2)):
        x, w, bias = make(m, seed, wscale)
        out, logits = run_logits(x, w, bias)
        bw = torch.empty((m, TOPK), dtype=torch.float32, device="cuda")
        bi = torch.empty((m, TOPK), dtype=torch.int32, device="cuda")
        ops.topk_hash_softplus_sqrt(
            bw, bi, torch.empty_like(bi), logits, True, SCALE, bias,
            None, None, None, None, 0,
        )  # fmt: skip
        torch.cuda.synchronize()
        assert torch.equal(out[1], bi)
        assert torch.equal(out[0], bw)
