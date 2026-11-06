# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 TP prefill MoE route: one route launch, rows placed by the quant."""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
fm = importlib.import_module("aiter.fused_moe")
mk = importlib.import_module("aiter.ops.flydsl.moe_kernels")

EXPERTS = 129  # MiniMax-M3 TP4: 128 routed + the shared expert
TOPK = 5
MODEL_DIM = 6144
INTER_DIM = 768


def require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a ROCm GPU")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 prefill route")


def make_ids(tokens, seed, pad_from=None):
    """Top-4 of 128 routed experts plus the shared expert 128 on every row.

    Rows from ``pad_from`` on are graph padding: -1 in every column.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, EXPERTS - 1, generator=g, device="cuda")
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK - 1, dim=-1)
    ids = torch.cat([ids, torch.full((tokens, 1), EXPERTS - 1, device="cuda")], 1)
    w = torch.cat([w, torch.ones(tokens, 1, device="cuda")], 1)
    ids, w = ids.to(torch.int32).contiguous(), w.float().contiguous()
    if pad_from is not None:
        ids[pad_from:] = -1
        w[pad_from:] = 0
    return ids, w


def route_reference(ids, tile_m):
    flat = ids.cpu().long().flatten()
    kept = flat >= 0
    counts = torch.bincount(flat[kept], minlength=EXPERTS)
    aligned = (counts + tile_m - 1) // tile_m * tile_m
    starts = aligned.cumsum(0) - aligned
    return counts, starts


def check_rows(ids, rows, tile_m):
    counts, starts = route_reference(ids, tile_m)
    flat = ids.cpu().long().flatten()
    rows = rows.cpu().long().flatten()
    assert (rows[flat < 0] == -1).all()
    for e in range(EXPERTS):
        got = rows[flat == e].sort().values
        want = torch.arange(int(starts[e]), int(starts[e] + counts[e]))
        assert torch.equal(got, want), f"rows of expert {e}"


def state_is_zero():
    return all(not bool(b.any()) for b in gm._TP_ROUTE_STATE_CACHE.values())


@pytest.mark.parametrize("tokens,tile_m,wmma_rep", [(410, 32, 2), (4536, 128, 8)])
def test_route_and_quant_place_every_route(tokens, tile_m, wmma_rep):
    require_gfx1250()
    numel = tokens * TOPK
    contiguous_m = gm._tdm_align_up(numel + EXPERTS * tile_m - TOPK, tile_m)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    for seed, pad_from in ((1, None), (2, tokens // 3)):
        ids, _ = make_ids(tokens, seed, pad_from)
        counts, starts = route_reference(ids, tile_m)
        for _ in range(3):
            masked_m, enc, psum, got_starts = gm.tp_prefill_route_psum(
                ids, EXPERTS, tile_m
            )
            rows = torch.full_like(enc, -7)
            mk.flydsl_moe_fused_quant_preshuffle(
                hidden.reshape(1, tokens, MODEL_DIM),
                1,
                contiguous_m,
                wmma_rep=wmma_rep,
                quant_mode="fp4",
                topids_to_rows=enc,
                source_topk=TOPK,
                route_starts=got_starts,
                route_expert_bits=gm.tp_route_expert_bits(EXPERTS),
                route_rows_out=rows,
            )
            torch.cuda.synchronize()
            assert torch.equal(masked_m.cpu().long(), counts)
            assert torch.equal(got_starts.cpu().long(), starts)
            assert torch.equal(psum.cpu().long(), starts + counts)
            check_rows(ids, rows, tile_m)
            assert state_is_zero()


def _moe_args(tokens):
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_route_in_quant_ref", path)
    ref = importlib.util.module_from_spec(spec)
    # The reference module sets the default device to cuda when it loads: take
    # the caller's default first, so it is the one restored below.
    default = torch.get_default_device()
    spec.loader.exec_module(ref)
    captured = []
    ref.fused_moe = lambda *a, **k: captured.append((a, k)) or torch.zeros(
        a[0].shape, dtype=torch.bfloat16, device="cuda"
    )
    ref._torch_moe_ref = lambda hidden, *a, **k: torch.zeros(
        hidden.shape, dtype=torch.bfloat16, device="cuda"
    )
    torch.set_default_device("cuda")
    try:
        ref._run_grouped_via_fused_moe(
            experts=EXPERTS,
            tokens=tokens,
            topk=TOPK,
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
            data_format="a4w4",
            activation=fm.ActivationType.Swiglu,
            use_bias=False,
            seed=393,
        )
    finally:
        torch.set_default_device(default)
    args, kwargs = captured[-1]
    return list(args), kwargs


def _moe(args, kwargs, hidden, ids, w):
    a = list(args)
    a[0], a[3], a[4] = hidden, w, ids
    return fm.fused_moe(*a, **kwargs)


def _kernels(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name
        for e in prof.events()
        if e.device_type == torch.autograd.DeviceType.CUDA
    ]


@pytest.mark.parametrize("tokens", [410, 2048, 4536])
def test_moe_matches_split_route(tokens, monkeypatch):
    require_gfx1250()
    args, kwargs = _moe_args(tokens)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    for seed, pad_from in ((3, None), (4, tokens - 7)):
        ids, w = make_ids(tokens, seed, pad_from)
        with monkeypatch.context() as split:
            split.setattr(gm, "_tp_route_quant_takes_route", lambda *a, **k: False)
            base = _moe(args, kwargs, hidden, ids, w)
            base_names = _kernels(lambda: _moe(args, kwargs, hidden, ids, w))
        out = [_moe(args, kwargs, hidden, ids, w) for _ in range(3)]
        names = _kernels(lambda: _moe(args, kwargs, hidden, ids, w))
        torch.cuda.synchronize()
        assert all(torch.equal(o, base) for o in out)
        assert state_is_zero()
        assert "moe_route_lds_psum_sc" in names
        assert not any(n.startswith("moe_contiguous_psum_remap") for n in names)
        assert not any("FillFunctor" in n for n in names)
        assert len(names) == len(base_names) - 2

    # Two calls per graph (consecutive layers); inputs change between replays.
    ids_s, w_s = make_ids(tokens, 5)
    hid_s = hidden.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        g1 = _moe(args, kwargs, hid_s, ids_s, w_s)
        g2 = _moe(args, kwargs, g1, ids_s, w_s)
    torch.cuda.current_stream().wait_stream(stream)
    for seed, pad_from in ((6, None), (7, tokens // 2), (8, None)):
        ids, w = make_ids(tokens, seed, pad_from)
        ids_s.copy_(ids)
        w_s.copy_(w)
        graph.replay()
        with monkeypatch.context() as split:
            split.setattr(gm, "_tp_route_quant_takes_route", lambda *a, **k: False)
            e1 = _moe(args, kwargs, hid_s, ids_s, w_s)
            e2 = _moe(args, kwargs, e1, ids_s, w_s)
        torch.cuda.synchronize()
        assert torch.equal(g1, e1) and torch.equal(g2, e2)
        assert state_is_zero()


def test_first_use_under_capture_fills(monkeypatch):
    """No state allocated yet: the capture keeps a filled counter."""
    require_gfx1250()
    tokens = 2048
    args, kwargs = _moe_args(tokens)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    ids, w = make_ids(tokens, 9)
    expect = _moe(args, kwargs, hidden, ids, w)
    torch.cuda.synchronize()
    monkeypatch.setattr(gm, "_TP_ROUTE_STATE_CACHE", {})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _moe(args, kwargs, hidden, ids, w)
    torch.cuda.current_stream().wait_stream(stream)
    assert gm._TP_ROUTE_STATE_CACHE == {}
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expect)


def test_capture_after_small_eager_call_self_cleans(monkeypatch):
    """A token graph recorded before any eager prefill: the counter that an
    eager decode-size call allocated keeps the capture on the self-cleaning
    route, without the fill."""
    require_gfx1250()
    tokens, small = 2048, 16
    args, kwargs = _moe_args(tokens)
    s_args, s_kwargs = _moe_args(small)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    ids, w = make_ids(tokens, 11)
    expect = _moe(args, kwargs, hidden, ids, w)  # compiles; allocates the counter
    torch.cuda.synchronize()
    monkeypatch.setattr(gm, "_TP_ROUTE_STATE_CACHE", {})
    # An eager decode-size call takes the small-batch route, not this one.
    s_hidden = (torch.randn(small, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    s_ids, s_w = make_ids(small, 12)
    _moe(s_args, s_kwargs, s_hidden, s_ids, s_w)
    torch.cuda.synchronize()
    assert len(gm._TP_ROUTE_STATE_CACHE) == 1 and state_is_zero()
    builds = []
    getter = gm._get_compiled_route_lds_psum

    def spy(experts, self_clean):
        builds.append(self_clean)
        return getter(experts, self_clean)

    monkeypatch.setattr(gm, "_get_compiled_route_lds_psum", spy)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _moe(args, kwargs, hidden, ids, w)
    torch.cuda.current_stream().wait_stream(stream)
    assert builds == [True]
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expect)
        assert state_is_zero()
