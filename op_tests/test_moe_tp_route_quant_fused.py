# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP route folded into the MoE activation quant vs the separate route launch.

The separate path counts routes with LDS atomics, so its row order inside an
expert is not fixed; the fused launch uses route order. Route outputs are
therefore checked exactly against a CPU reference, and the quantized rows are
compared after mapping each route's row from one layout onto the other. The
public MoE output must match bit for bit.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

W = Path(__file__).resolve().parents[1]
fm = importlib.import_module("aiter.fused_moe")
gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
mk = importlib.import_module("aiter.ops.flydsl.moe_kernels")

FUSE_ENV = "AITER_TDM_FUSE_TP_ROUTE_QUANT"


@pytest.fixture(autouse=True)
def _fresh_env_switches():
    # AITER_TDM_FUSE_TP_ROUTE_QUANT is read once per process: tests that set it
    # clear the cached value, and none leaks into the next test.
    gm._fuse_tp_route_quant_enabled.cache_clear()
    yield
    gm._fuse_tp_route_quant_enabled.cache_clear()


@pytest.fixture
def gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the route-indexed pk8 quant is gfx1250 only")


def m3_routes(tokens, experts, seed, real=None, drop=False):
    """Top-4 routed experts plus the last expert as a shared one on every row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, experts - 1, generator=g, device="cuda")
    ids = torch.topk(logits, 4, dim=-1).indices
    ids = torch.cat([ids, torch.full((tokens, 1), experts - 1, device="cuda")], 1)
    ids = ids.to(torch.int32).contiguous()
    if drop:
        r = torch.rand(tokens, 4, generator=g, device="cuda")
        ids[:, :4][r < 0.2] = -1
        ids[:, :4][(r >= 0.2) & (r < 0.25)] = experts
    if real is not None:
        # TP graph padding: pad rows keep only the shared expert.
        ids[real:, :4] = -1
    return ids


def reference(ids, experts, align):
    flat = ids.reshape(-1).cpu().long()
    valid = (flat >= 0) & (flat < experts)
    hist = torch.bincount(flat[valid], minlength=experts)
    padded = (hist + align - 1) // align * align
    starts = padded.cumsum(0) - padded
    rows = torch.full_like(flat, -1)
    seen = torch.zeros(experts, dtype=torch.long)
    for i, e in enumerate(flat.tolist()):
        if 0 <= e < experts:
            rows[i] = starts[e] + seen[e]
            seen[e] += 1
    return hist, starts + hist, rows


def geometry(tokens, topk, experts, align):
    numel = tokens * topk
    cm = max(align, gm._tdm_align_up(numel + experts * align - topk, align))
    max_m = max(align, gm._tdm_align_up(numel, align))
    return cm, max_m


def quant(hidden, ids, cm, rep, outs, rows, tp=None):
    tokens, topk = ids.shape
    mk.flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, tokens, -1),
        1,
        cm,
        wmma_rep=rep,
        quant_mode="fp4",
        topids_to_rows=rows,
        source_topk=topk,
        out_payload=outs[0],
        out_scale=outs[1],
        tp_route_params=tp,
    )


def scale_rows(s, cm, rep, ws):
    return (
        s.reshape(cm // (16 * rep), ws // 4, rep, 16, 4)
        .permute(0, 2, 3, 1, 4)
        .reshape(cm, ws)
    )


@pytest.mark.parametrize("tokens", [1, 4, 16, 17, 63, 409])
@pytest.mark.parametrize("pattern", ["m3", "pad", "drop"])
@pytest.mark.parametrize("align,rep", [(16, 1), (32, 2)])
def test_fused_route_quant_matches_split(tokens, pattern, align, rep, gfx1250):
    experts, topk, dim = 129, 5, 1024
    ids = m3_routes(
        tokens,
        experts,
        seed=tokens,
        real=max(1, tokens // 2) if pattern == "pad" else None,
        drop=pattern == "drop",
    )
    hidden = torch.randn(tokens, dim, device="cuda").to(torch.bfloat16)
    cm, max_m = geometry(tokens, topk, experts, align)
    pb, ws = dim // 2, dim // 32

    def buffers():
        return (
            torch.full((1, cm, pb), 173, device="cuda", dtype=torch.uint8),
            torch.full((1, cm // rep, ws * rep), 197, device="cuda", dtype=torch.uint8),
        )

    split_out = buffers()
    m_split, rows_split, psum_split = gm.fused_route_psum_remap(
        ids, experts, max_m, align
    )
    quant(hidden, ids, cm, rep, split_out, rows_split)

    fused_out = buffers()
    m_f, rows_f, psum_f = (
        torch.full((experts,), -3, device="cuda", dtype=torch.int32),
        torch.full((tokens, topk), -3, device="cuda", dtype=torch.int32),
        torch.full((experts,), -3, device="cuda", dtype=torch.int32),
    )
    tp = {
        "topk_ids": ids,
        "masked_m": m_f,
        "psum": psum_f,
        "experts": experts,
        "tile_m": align,
    }
    quant(hidden, ids, cm, rep, fused_out, rows_f, tp)

    hist, ref_psum, ref_rows = reference(ids, experts, align)
    assert torch.equal(m_f.cpu().long(), hist)
    assert torch.equal(m_split.cpu(), m_f.cpu())
    assert torch.equal(psum_f.cpu().long(), ref_psum)
    assert torch.equal(psum_split.cpu(), psum_f.cpu())
    assert torch.equal(rows_f.reshape(-1).cpu().long(), ref_rows)

    # Map the split rows onto the fused order; untouched rows keep the fill.
    src = rows_split.reshape(-1).long()
    dst = rows_f.reshape(-1).long()
    live = src >= 0
    assert torch.equal(live, dst >= 0)
    p = split_out[0].reshape(cm, pb).clone()
    p[dst[live]] = split_out[0].reshape(cm, pb)[src[live]]
    assert torch.equal(p, fused_out[0].reshape(cm, pb))
    s = scale_rows(split_out[1], cm, rep, ws)
    s2 = s.clone()
    s2[dst[live]] = s[src[live]]
    assert torch.equal(s2, scale_rows(fused_out[1], cm, rep, ws))

    # Graph replay with new routes and poisoned outputs leaves nothing behind.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        quant(hidden, ids, cm, rep, fused_out, rows_f, tp)
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (101, 102):
        ids.copy_(m3_routes(tokens, experts, seed, real=1 if seed == 102 else None))
        for t in (m_f, rows_f, psum_f):
            t.fill_(-9)
        graph.replay()
        hist, ref_psum, ref_rows = reference(ids, experts, align)
        assert torch.equal(m_f.cpu().long(), hist)
        assert torch.equal(psum_f.cpu().long(), ref_psum)
        assert torch.equal(rows_f.reshape(-1).cpu().long(), ref_rows)


@pytest.mark.parametrize("tokens", [3, 16, 40])
def test_public_moe_fused_route_bitwise(tokens, monkeypatch, gfx1250):
    spec = importlib.util.spec_from_file_location(
        "_tp_route_quant_moe_reference",
        W / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py",
    )
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        captured = []
        public = ref.fused_moe

        def capture(*a, **kw):
            captured.append((a, kw))
            return public(*a, **kw)

        ref.fused_moe = capture
        experts = 17
        monkeypatch.setenv(FUSE_ENV, "0")
        gm._fuse_tp_route_quant_enabled.cache_clear()
        ref._run_grouped_via_fused_moe(
            experts=experts,
            tokens=tokens,
            topk=5,
            model_dim=256,
            inter_dim=256,
            data_format="a4w4",
            activation=fm.ActivationType.Swiglu,
            use_bias=True,
            seed=11,
        )
        a, kw = captured[-1]
        a = list(a)
        a[4] = m3_routes(tokens, experts, seed=5, real=max(1, tokens - 2))
        a[3] = torch.where(a[4] >= 0, a[3], torch.zeros_like(a[3]))
        route_calls = []
        route_fn = gm.fused_route_psum_remap

        def counted(*args, **kwargs):
            route_calls.append(True)
            return route_fn(*args, **kwargs)

        monkeypatch.setattr(gm, "fused_route_psum_remap", counted)
        split = fm.fused_moe(*a, **kw)
        assert route_calls, "the split path must run the route launch"
        route_calls.clear()
        monkeypatch.setenv(FUSE_ENV, "1")
        gm._fuse_tp_route_quant_enabled.cache_clear()
        fused = fm.fused_moe(*a, **kw)
        assert not route_calls, "the fused path must not run the route launch"
        assert torch.equal(fused, split)
        assert torch.isfinite(fused.float()).all()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = fm.fused_moe(*a, **kw)
        torch.cuda.current_stream().wait_stream(s)
        for seed in (6, 7):
            a[4].copy_(m3_routes(tokens, experts, seed=seed))
            a[0].neg_()
            graph.replay()
            monkeypatch.setenv(FUSE_ENV, "0")
            gm._fuse_tp_route_quant_enabled.cache_clear()
            torch.testing.assert_close(out, fm.fused_moe(*a, **kw), rtol=0, atol=0)
            monkeypatch.setenv(FUSE_ENV, "1")
            gm._fuse_tp_route_quant_enabled.cache_clear()
    finally:
        torch.set_default_device(default)
