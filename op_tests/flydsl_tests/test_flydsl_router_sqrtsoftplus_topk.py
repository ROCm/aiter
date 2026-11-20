# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Edge cases of ``flydsl_router_sqrtsoftplus_topk`` (gfx950).

The DeepSeek-V4.1-Flash MoE router (hidden 5120, 384 experts, top-6) against an
fp32 torch reference at every decode row bucket and beyond, for the row-major
and the load-time shuffled gate: the op's logits (its own fp32 partials, summed
in slice order) within fp32 accumulation error of torch, ids and weights equal
to the reference top-k of those logits, exact ties (lowest expert id), logits
above the softplus switch, padding rows, row strides, non-finite inputs,
caller buffers poisoned before a CUDA-graph replay, determinism and the
eligibility helper. When vLLM is importable, the ids and weights are also
compared bit for bit with vLLM's ``topk_hash_softplus_sqrt`` on the same logits.

Run: ``python3 -m pytest -q op_tests/flydsl_tests/test_flydsl_router_sqrtsoftplus_topk.py``
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="the FlyDSL router kernels are gfx950 kernels",
)

ROWS = (1, 7, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768)
ROWS += (1024, 2049)
SCALE = 1.5
HIDDEN, EXPERTS = 5120, 384
MARGIN = 1e-5


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def score_ref(logits):
    return torch.where(logits > 20.0, logits, torch.log1p(torch.exp(logits))).sqrt()


def topk_ref(logits, bias, scale, is_padding=None):
    score = score_ref(logits.float())
    ids = torch.topk(score + bias, 6, dim=-1)[1]
    w = (score + bias).gather(1, ids) - bias[ids]
    total = w[:, 0]
    for k in range(1, 6):
        total = total + w[:, k]
    w = w * (scale / total[:, None])
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    return w, ids


def _router_inputs(m, seed, wscale=0.02):
    g = _gen(seed)
    x = torch.randn((m, HIDDEN), generator=g, device="cuda") * torch.exp2(
        torch.randint(-2, 3, (m, 1), generator=g, device="cuda").float()
    )
    w = torch.randn((EXPERTS, HIDDEN), generator=g, device="cuda") * wscale
    bias = torch.randn(EXPERTS, generator=g, device="cuda") * 0.1
    return x.bfloat16(), w.bfloat16(), bias


def _route(x, w, bias, **kw):
    """Run the op with a caller scratch and return its own logits too."""
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_workspace_shape,
    )

    shape = router_sqrtsoftplus_topk_workspace_shape(
        x.shape[0], kw.get("weight_shuffled", False)
    )
    partials = torch.empty(shape, dtype=torch.float32, device=x.device)
    out = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, partials=partials, **kw)
    torch.cuda.synchronize()
    logits = partials[0].clone()
    for p in partials[1:]:
        logits = logits + p
    return out, logits


def _close_rows(logits, bias):
    """Rows whose 6th and 7th ``score + bias`` tie within MARGIN."""
    top = torch.topk(score_ref(logits) + bias, 7, dim=-1)[0]
    return (top[:, 5] - top[:, 6]).abs() < MARGIN * top[:, 5].abs().clamp_min(1.0)


def _check_route(out, logits, bias, is_padding=None):
    tw, ti = out
    rw, ri = topk_ref(logits, bias, SCALE, is_padding)
    ok = ~_close_rows(logits, bias)
    assert torch.equal(ti[ok], ri[ok]), "ids"
    assert ((tw[ok] - rw[ok]).abs() <= 2e-6 * rw[ok].abs().clamp_min(1e-3)).all()
    live = ok if is_padding is None else ok & ~is_padding
    total = tw[live].sum(-1)
    assert torch.allclose(total, torch.full_like(total, SCALE), rtol=1e-5)


@pytest.mark.parametrize("weight_shuffled", (False, True))
@pytest.mark.parametrize("m", ROWS)
def test_router_sqrtsoftplus_topk(m, weight_shuffled):
    from aiter.ops.flydsl import shuffle_router_gate

    x, w, bias = _router_inputs(m, m + 11 * weight_shuffled)
    wk = shuffle_router_gate(w) if weight_shuffled else w
    out, logits = _route(x, wk, bias, weight_shuffled=weight_shuffled)
    ref = torch.mm(x.float(), w.float().t())
    assert (logits - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()
    _check_route(out, logits, bias)
    again, _ = _route(x, wk, bias, weight_shuffled=weight_shuffled)
    assert torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])


def test_shuffle_router_gate_is_permutation():
    from aiter.ops.flydsl import shuffle_router_gate

    _, w, _ = _router_inputs(1, 0)
    ws = shuffle_router_gate(w)
    back = ws.reshape(EXPERTS // 16, HIDDEN // 32, 4, 16, 8).permute(0, 3, 1, 2, 4)
    assert torch.equal(back.reshape(EXPERTS, HIDDEN), w)


@pytest.mark.parametrize("m", (1, 16, 192, 768))
def test_router_vllm_topk_bitwise(m):
    """vLLM's top-k on the op's own logits: ids and weights bit for bit, at
    normal and above-switch (> 20) logits."""
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops

    for seed, wscale in ((m, 0.02), (m + 1, 0.05), (m + 2, 0.2)):
        x, w, bias = _router_inputs(m, seed, wscale)
        out, logits = _route(x, w, bias)
        bw = torch.empty((m, 6), dtype=torch.float32, device="cuda")
        bi = torch.empty((m, 6), dtype=torch.int32, device="cuda")
        ops.topk_hash_softplus_sqrt(
            bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, *[None] * 4, 0
        )
        assert torch.equal(out[1], bi) and torch.equal(out[0], bw)


@pytest.mark.parametrize("m", (16, 192))
def test_router_vllm_gemm_and_topk(m):
    """vLLM's whole router (its GEMM and top-k): equal ids wherever the 6th
    and 7th ``score + bias`` differ by more than the margin, weights within
    2e-6 relative (the GEMMs sum in different orders)."""
    pytest.importorskip("vllm")
    import importlib

    from vllm import _custom_ops as ops

    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    # Registers torch.ops.vllm.fp32_router_gemm_dispatch.
    importlib.import_module("vllm.model_executor.layers.fused_moe.router.gate_linear")
    x, w, bias = _router_inputs(m, m + 3)
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    logits = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
    bw = torch.empty((m, 6), dtype=torch.float32, device="cuda")
    bi = torch.empty((m, 6), dtype=torch.int32, device="cuda")
    ops.topk_hash_softplus_sqrt(
        bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, *[None] * 4, 0
    )
    ok = ~_close_rows(logits, bias)
    assert torch.equal(ti[ok], bi[ok])
    assert ((tw[ok] - bw[ok]).abs() <= 2e-6 * bw[ok].abs()).all()


def test_router_ties_lowest_id():
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    x = torch.zeros((16, HIDDEN), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((EXPERTS, HIDDEN), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(EXPERTS, device="cuda")
    for i in (383, 5, 100, 7, 200, 9, 300, 11):
        bias[i] = 1.0
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    want = torch.tensor([5, 7, 9, 11, 100, 200], device="cuda", dtype=torch.int32)
    assert (ti == want).all()
    assert torch.allclose(tw, torch.full_like(tw, SCALE / 6), rtol=1e-6)
    bias.zero_()
    tw, ti = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE)
    assert (ti == torch.arange(6, device="cuda", dtype=torch.int32)).all()


@pytest.mark.parametrize("m", (16, 192))
def test_router_large_logits_and_padding(m):
    x, w, bias = _router_inputs(m, 9, wscale=0.2)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    pad[::3] = True
    out, logits = _route(x, w, bias, is_padding=pad)
    assert (logits > 20).any()
    _check_route(out, logits, bias, pad)
    assert (out[1][pad] == -1).all() and (out[0][pad] == 0).all()
    assert (out[1][~pad] >= 0).all()


def test_router_row_stride_and_non_finite():
    m = 96
    x, w, bias = _router_inputs(m, 8)
    wide = torch.zeros((m, HIDDEN + 64), device="cuda", dtype=torch.bfloat16)
    wide[:, :HIDDEN] = x
    a, _ = _route(wide[:, :HIDDEN], w, bias)
    b, _ = _route(x, w, bias)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    x[3, 7] = float("nan")
    x[4, 7] = float("inf")
    (tw, ti), _ = _route(x, w, bias)
    assert ((ti >= 0) & (ti < EXPERTS)).all()
    assert torch.isfinite(tw[[0, 1, 2]]).all()


def test_router_graph_and_buffers():
    from aiter.ops.flydsl import flydsl_router_sqrtsoftplus_topk

    m = 192
    x, w, bias = _router_inputs(m, 5)
    tw = torch.empty((m, 6), device="cuda")
    ti = torch.empty((m, 6), device="cuda", dtype=torch.int32)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    kw = {"is_padding": pad, "topk_weights": tw, "topk_ids": ti}
    flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, **kw)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, **kw)
    assert out[0] is tw and out[1] is ti
    for i in range(3):
        x.mul_(0.75).add_(0.0625)
        pad[i] = True
        ew, ei = flydsl_router_sqrtsoftplus_topk(x, w, bias, SCALE, is_padding=pad)
        tw.fill_(float("nan"))
        ti.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(tw, ew) and torch.equal(ti, ei)


def test_router_eligibility():
    from aiter.ops.flydsl import (
        flydsl_router_sqrtsoftplus_topk,
        router_sqrtsoftplus_topk_supported,
    )

    x, w, bias = _router_inputs(8, 0)
    assert router_sqrtsoftplus_topk_supported(x, w, bias)
    assert not router_sqrtsoftplus_topk_supported(x, w, bias, top_k=8)
    # Above the tuned rows the helper declines; the op itself stays correct
    # there (test_router_sqrtsoftplus_topk covers 1024 and 2049 rows).
    big = torch.zeros((769, HIDDEN), device="cuda", dtype=torch.bfloat16)
    assert router_sqrtsoftplus_topk_supported(big[:768], w, bias)
    assert not router_sqrtsoftplus_topk_supported(big, w, bias)
    out = flydsl_router_sqrtsoftplus_topk(x[:0], w, bias, SCALE)
    assert out[1].shape == (0, 6)
    bad = [
        (x.float(), w, bias),
        (x, w[:256], bias),  # 256 experts
        (x, w, bias.bfloat16()),
        (x[:, 1:], w[:, 1:], bias),  # hidden / alignment
    ]
    for args in bad:
        assert not router_sqrtsoftplus_topk_supported(*args)
        with pytest.raises(ValueError):
            flydsl_router_sqrtsoftplus_topk(*args, SCALE)
