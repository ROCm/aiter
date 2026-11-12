# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 TP prefill MoE route: one route launch, rows placed by the quant."""

import argparse
import importlib
import importlib.util
import itertools
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
fm = importlib.import_module("aiter.fused_moe")
mk = importlib.import_module("aiter.ops.flydsl.moe_kernels")
SPLIT_ROUTE = mock.patch.object(gm, "_tp_route_quant_takes_route", return_value=False)

EXPERTS = 129  # MiniMax-M3 TP4: 128 routed + the shared expert
TOPK = 5
MODEL_DIM = 6144
CASES = [(410, 32, 2), (4536, 128, 8)]  # tokens, tile_m, wmma_rep of the CSV rows
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="gfx1250 prefill route",
)


def make_ids(tokens, seed, pad_from=None):
    """Top-4 of 128 routed experts plus shared expert 128; pad rows are -1."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, EXPERTS - 1, generator=g, device="cuda")
    w, ids = torch.topk(torch.softmax(logits, -1), TOPK - 1, dim=-1)
    ids = torch.cat([ids, torch.full((tokens, 1), EXPERTS - 1, device="cuda")], 1)
    w = torch.cat([w, torch.ones(tokens, 1, device="cuda")], 1)
    ids, w = ids.to(torch.int32).contiguous(), w.float().contiguous()
    if pad_from is not None:
        ids[pad_from:], w[pad_from:] = -1, 0
    return ids, w


def state_is_zero():
    return all(not bool(b.any()) for b in gm._TP_ROUTE_STATE_CACHE.values())


def route_quant(ids, hidden, tile_m, wmma_rep, one_launch, payload, scale):
    """one_launch: this PR's route, rows resolved by the quant. Otherwise the
    launches it replaces: fill + moe_route_lds, psum remap, quant."""
    tokens, kw = ids.shape[0], {"out_payload": payload, "out_scale": scale}
    if one_launch:
        masked_m, enc, psum, starts = gm.tp_prefill_route_psum(ids, EXPERTS, tile_m)
        rows = torch.empty_like(enc)
        bits = gm.tp_route_expert_bits(EXPERTS)
        kw.update(route_starts=starts, route_rows_out=rows, route_expert_bits=bits)
    else:
        max_m = gm._tdm_align_up(tokens * TOPK, tile_m)
        masked_m, rows = mk.flydsl_moe_topids_to_rows(ids, EXPERTS, max_m)
        remap = (masked_m, rows, EXPERTS, max_m, tile_m)  # rows remapped in place
        starts, psum, _ = gm.contiguous_psum_remap(*remap, topk=TOPK)
        enc = rows
    kw.update(wmma_rep=wmma_rep, quant_mode="fp4", source_topk=TOPK, topids_to_rows=enc)
    x = hidden.reshape(1, tokens, MODEL_DIM)
    mk.flydsl_moe_fused_quant_preshuffle(x, 1, payload.shape[1], **kw)
    return masked_m[:EXPERTS], starts, psum, rows


def run_stage(tokens, tile_m, wmma_rep, pad_from=None, timed=False):
    """Both routes vs a CPU reference: exact counts/starts/psum, each expert
    owning rows [start, start + count). Slot order follows atomics, so compare
    payload bytes per route and scale bytes per expert segment (whole tiles)."""
    m = gm._tdm_align_up(tokens * TOPK + EXPERTS * tile_m - TOPK, tile_m)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    ids, _ = make_ids(tokens, tokens, pad_from)
    flat = ids.cpu().long().flatten()
    counts = torch.bincount(flat[flat >= 0], minlength=EXPERTS)
    aligned = (counts + tile_m - 1) // tile_m * tile_m
    starts = aligned.cumsum(0) - aligned
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    ret, outs = {}, {}
    for name in ("split", "one_launch"):
        pay = torch.full((1, m, MODEL_DIM // 2), 0xA5, **u8)
        sc = torch.full((1, m // wmma_rep, MODEL_DIM // 32 * wmma_rep), 0x5A, **u8)
        args = (ids, hidden, tile_m, wmma_rep, name == "one_launch", pay, sc)
        if timed:
            res, us = run_perftest(route_quant, *args)
            nbytes = hidden.nbytes + ids.numel() * (4 + MODEL_DIM * 17 // 32)
            ret.update({f"{name} us": us, f"{name} TB/s": nbytes / us / 1e6})
        else:
            res = [route_quant(*args) for _ in range(2)][-1]  # repeated calls
        torch.cuda.synchronize()
        masked_m, got_starts, psum, rows = (t.cpu().long().flatten() for t in res)
        assert torch.equal(masked_m, counts) and torch.equal(got_starts, starts)
        assert torch.equal(psum, starts + counts) and (rows[flat < 0] == -1).all()
        for e in range(EXPERTS):
            want = torch.arange(int(starts[e]), int(starts[e] + counts[e]))
            assert torch.equal(rows[flat == e].sort().values, want), f"expert {e}"
        outs[name] = rows, pay.cpu().view(m, -1), sc.cpu().view(m // wmma_rep, -1)
    assert state_is_zero()
    (rs, ps, ss), (ro, po, so) = outs["split"], outs["one_launch"]
    bad = int((ps[rs[flat >= 0]] != po[ro[flat >= 0]]).sum())
    for lo, n in zip((starts // wmma_rep).tolist(), (aligned // wmma_rep).tolist()):
        a, b = ss[lo : lo + n].flatten().sort(), so[lo : lo + n].flatten().sort()
        bad += int((a.values != b.values).sum())
    assert bad == 0, f"{bad} payload/scale bytes differ from the split route"
    return {**ret, "one_launch err": bad}


@pytest.mark.parametrize("tokens,tile_m,wmma_rep", CASES)
@pytest.mark.parametrize("padded", [False, True])
def test_route_and_quant_place_every_route(tokens, tile_m, wmma_rep, padded):
    run_stage(tokens, tile_m, wmma_rep, tokens // 3 if padded else None)


@benchmark()
def bench(tokens, tile_m, wmma_rep):
    return {"gfx": get_gfx_runtime(), **run_stage(tokens, tile_m, wmma_rep, timed=True)}


@pytest.fixture(autouse=True)
def _tuned_row_without_a_preshuffle(monkeypatch):
    """The route needs the A1 quant without A-preshuffle, which a tuned row may
    ask for (a MiniMax-M3 table does at 4097-8192 tokens): turn it off."""
    find = gm._find_grouped_config
    no_apre = lambda **k: (r := find(**k)) and {**r, "a_preshuffle": "0"}
    monkeypatch.setattr(gm, "_find_grouped_config", no_apre)


def _moe_fn(tokens):
    """fused_moe(x, ids, w) with the grouped-GEMM reference test's weights."""
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_route_in_quant_ref", path)
    ref = importlib.util.module_from_spec(spec)
    default = torch.get_default_device()  # the module sets cuda when it loads
    spec.loader.exec_module(ref)
    captured = []
    ref.fused_moe = lambda *a, **k: captured.append((a, k)) or torch.zeros_like(a[0])
    ref._torch_moe_ref = lambda hidden, *a, **k: torch.zeros_like(hidden)
    torch.set_default_device("cuda")
    try:
        ref._run_grouped_via_fused_moe(
            experts=EXPERTS, tokens=tokens, topk=TOPK, model_dim=MODEL_DIM,
            inter_dim=768, data_format="a4w4", activation=fm.ActivationType.Swiglu,
            use_bias=False, seed=393,
        )  # fmt: skip
    finally:
        torch.set_default_device(default)
    args, kwargs = captured[-1]
    return lambda x, ids, w: fm.fused_moe(x, *args[1:3], w, ids, *args[5:], **kwargs)


def _capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _kernels(fn, *args):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(*args)
        torch.cuda.synchronize()
    return [e.name for e in p.events() if e.device_type.name == "CUDA"]


@pytest.mark.parametrize("tokens", [410, 2048, 4536])
def test_moe_matches_split_route(tokens):
    moe = _moe_fn(tokens)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    for seed, pad_from in ((3, None), (4, tokens - 7)):
        ids, w = make_ids(tokens, seed, pad_from)
        with SPLIT_ROUTE:
            base, base_names = moe(hidden, ids, w), _kernels(moe, hidden, ids, w)
        out = [moe(hidden, ids, w) for _ in range(3)]
        names = _kernels(moe, hidden, ids, w)
        assert all(torch.equal(o, base) for o in out) and state_is_zero()
        assert "moe_route_lds_psum_sc" in names and len(names) == len(base_names) - 2
        assert not any("psum_remap" in n or "FillFunctor" in n for n in names)

    ids_s, w_s = make_ids(tokens, 5)  # two layers per graph, inputs change

    def two_layers():
        return (g1 := moe(hidden, ids_s, w_s)), moe(g1, ids_s, w_s)

    graph, graph_out = _capture(two_layers)
    for seed, pad_from in ((6, None), (7, tokens // 2), (8, None)):
        for dst, src in zip((ids_s, w_s), make_ids(tokens, seed, pad_from)):
            dst.copy_(src)
        graph.replay()
        with SPLIT_ROUTE:
            eager = two_layers()
        assert all(map(torch.equal, graph_out, eager)) and state_is_zero()


@pytest.mark.parametrize("eager_tokens", [None, 16])
def test_capture_route_state(eager_tokens, monkeypatch):
    """No counter: the capture uses a filled scratch one and allocates nothing.
    A counter from an eager decode-size call keeps it on the self-cleaning one."""
    tokens, moe = 2048, _moe_fn(2048)
    hidden = (torch.randn(tokens, MODEL_DIM, device="cuda") * 0.5).bfloat16()
    ids, w = make_ids(tokens, 11)
    expect = moe(hidden, ids, w)  # compiles; allocates the counter
    monkeypatch.setattr(gm, "_TP_ROUTE_STATE_CACHE", {})
    if eager_tokens:  # takes the small-batch route, not this one
        x = torch.randn(eager_tokens, MODEL_DIM, device="cuda").bfloat16()
        _moe_fn(eager_tokens)(x, *make_ids(eager_tokens, 12))
    builds, getter = [], gm._get_compiled_route_lds_psum
    spy = lambda *a: builds.append(a[1]) or getter(*a)
    monkeypatch.setattr(gm, "_get_compiled_route_lds_psum", spy)
    graph, out = _capture(lambda: moe(hidden, ids, w))
    assert builds == [bool(eager_tokens)]
    assert len(gm._TP_ROUTE_STATE_CACHE) == int(bool(eager_tokens))
    for _ in range(2):
        graph.replay()
        assert torch.equal(out, expect) and state_is_zero()


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("prefill route: unsupported on %s", get_gfx_runtime())
        return 0
    parser = argparse.ArgumentParser(description="gfx1250 TP prefill route + quant")
    parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[410, 4536])
    tokens = parser.parse_args().tokens
    if pytest.main([__file__, "-q", "-p", "no:cacheprovider"]) != pytest.ExitCode.OK:
        return 1
    rows = [bench(t, *c[1:]) for t, c in itertools.product(tokens, CASES)]
    df = pd.DataFrame(rows).to_markdown(index=False)
    aiter.logger.info("prefill route + quant stage summary (markdown):\n%s", df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
