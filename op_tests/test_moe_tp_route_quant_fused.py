# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP route folded into the MoE activation quant vs the separate route launch.

The separate route orders rows within an expert by LDS atomics, the fused one by
route order: route outputs are checked against a CPU reference, payload and
scales after moving each route's row, and the whole MoE output bit for bit.
"""

import argparse
import importlib
import importlib.util
import os
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

W = Path(__file__).resolve().parents[1]
fm = importlib.import_module("aiter.fused_moe")
gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
mk = importlib.import_module("aiter.ops.flydsl.moe_kernels")
FUSE_ENV = "AITER_TDM_FUSE_TP_ROUTE_QUANT"
MOE = {"experts": 17, "topk": 5, "model_dim": 256, "inter_dim": 256, "seed": 11}
MOE.update(data_format="a4w4", activation=fm.ActivationType.Swiglu, use_bias=True)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="the route-indexed pk8 quant is gfx1250 only",
)


def set_fuse(value):  # the switch is read once per process
    os.environ[FUSE_ENV] = value
    gm._fuse_tp_route_quant_enabled.cache_clear()


@pytest.fixture(autouse=True)
def _restore_switch(monkeypatch):
    monkeypatch.setenv(FUSE_ENV, os.environ.get(FUSE_ENV, "1"))
    gm._fuse_tp_route_quant_enabled.cache_clear()
    yield
    gm._fuse_tp_route_quant_enabled.cache_clear()


def m3_routes(tokens, experts, seed, real=None, drop=False):
    """Top-4 routed experts plus the last expert as a shared one on every row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(tokens, experts - 1, generator=g, device="cuda")
    ids = torch.topk(logits, 4, dim=-1).indices
    ids = torch.cat([ids, torch.full((tokens, 1), experts - 1, device="cuda")], 1)
    ids = ids.to(torch.int32).contiguous()
    if drop:  # dropped (-1) and out-of-range (== experts) ids
        r = torch.rand(tokens, 4, generator=g, device="cuda")
        ids[:, :4][r < 0.2] = -1
        ids[:, :4][(r >= 0.2) & (r < 0.25)] = experts
    if real is not None:  # TP pad rows keep only the shared expert
        ids[real:, :4] = -1
    return ids


def check_route(ids, tp, align):
    flat = ids.reshape(-1).cpu().long()
    hist = torch.bincount(
        flat[(flat >= 0) & (flat < tp["experts"])], minlength=tp["experts"]
    )
    padded = (hist + align - 1) // align * align
    starts = padded.cumsum(0) - padded
    rows, seen = torch.full_like(flat, -1), torch.zeros_like(hist)
    for i, e in enumerate(flat.tolist()):
        if 0 <= e < tp["experts"]:
            rows[i], seen[e] = starts[e] + seen[e], seen[e] + 1
    assert torch.equal(tp["masked_m"].cpu().long(), hist)
    assert torch.equal(tp["psum"].cpu().long(), starts + hist)
    assert torch.equal(tp["rows"].reshape(-1).cpu().long(), rows)


def quant(c, outs, rows, tp=None):
    mk.flydsl_moe_fused_quant_preshuffle(
        *(c["hidden"][None], 1, c["cm"]),
        wmma_rep=c["rep"],
        quant_mode="fp4",
        topids_to_rows=rows,
        source_topk=5,
        out_payload=outs[0],
        out_scale=outs[1],
        tp_route_params=tp,
    )


def run_case(tokens, pattern, align, rep, experts=129, dim=1024):
    """Split and fused route + quant once each, asserted bitwise equal."""
    real = max(1, tokens // 2) if pattern == "pad" else None
    ids = m3_routes(tokens, experts, seed=tokens, real=real, drop=pattern == "drop")
    numel, pb, ws = tokens * 5, dim // 2, dim // 32
    cm = max(align, gm._tdm_align_up(numel + experts * align - 5, align))
    max_m = max(align, gm._tdm_align_up(numel, align))
    hidden = torch.randn(tokens, dim, device="cuda").to(torch.bfloat16)
    # Sentinel fills, so an unwritten row or scale shows up.
    fills = (((1, cm, pb), 173), ((1, cm // rep, ws * rep), 197))
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    outs = [[torch.full(sh, v, **u8) for sh, v in fills] for _ in "sf"]
    c = {"ids": ids, "hidden": hidden, "cm": cm, "rep": rep, "outs": outs}
    c["split"] = lambda: quant(
        c, outs[0], gm.fused_route_psum_remap(ids, experts, max_m, align)[1]
    )
    tp = c["tp"] = {"topk_ids": ids, "experts": experts, "tile_m": align}
    for k, n in (("masked_m", experts), ("psum", experts), ("rows", numel)):
        tp[k] = torch.full((n,), -3, dtype=torch.int32, device="cuda")
    c["fused"] = lambda: quant(c, outs[1], tp["rows"], tp)
    m_s, rows_s, psum_s = gm.fused_route_psum_remap(ids, experts, max_m, align)
    quant(c, outs[0], rows_s)
    c["fused"]()
    check_route(ids, tp, align)
    assert torch.equal(m_s, tp["masked_m"]) and torch.equal(psum_s, tp["psum"])
    src, dst = rows_s.reshape(-1).long(), tp["rows"].long()
    live = src >= 0
    assert torch.equal(live, dst >= 0)

    def as_rows(o):  # payload and scales, one row per grouped row
        sc = o[1].reshape(cm // (16 * rep), ws // 4, rep, 16, 4).permute(0, 2, 3, 1, 4)
        return o[0].reshape(cm, pb), sc.reshape(cm, ws)

    for split, fused in zip(as_rows(outs[0]), as_rows(outs[1])):
        moved = split.clone()
        moved[dst[live]] = split[src[live]]
        assert torch.equal(moved, fused)
    return c


@benchmark()
def bench_route_quant(tokens, pattern, align, rep, dim=1024):
    c = run_case(tokens, pattern, align, rep, dim=dim)
    numel = tokens * 5  # bf16 rows in; payload, scale, id and row per route
    nbytes = tokens * dim * 2 + numel * (dim // 2 + dim // 32 + 8)
    ret = {"gfx": get_gfx_runtime()}
    for name in ("split", "fused"):
        _, us = run_perftest(c[name])
        ret[f"{name} us"], ret[f"{name} err"] = us, 0  # run_case asserted
        ret[f"{name} TFLOPS"] = numel * dim / us / 1e6  # one scale mul/element
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


@pytest.mark.parametrize("tokens", [1, 4, 16, 17, 63, 409])
@pytest.mark.parametrize("pattern", ["m3", "pad", "drop"])
@pytest.mark.parametrize("align,rep", [(16, 1), (32, 2)])
def test_fused_route_quant_matches_split(tokens, pattern, align, rep):
    c = run_case(tokens, pattern, align, rep)
    # Graph replay with new routes and poisoned outputs leaves nothing behind.
    stream, graph = torch.cuda.Stream(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        c["fused"]()
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (101, 102):
        c["ids"].copy_(m3_routes(tokens, 129, seed, real=1 if seed > 101 else None))
        for k in ("masked_m", "psum", "rows"):
            c["tp"][k].fill_(-9)
        graph.replay()
        check_route(c["ids"], c["tp"], align)


def test_tp_route_params_contract():
    """Buffers the fused route cannot write in place are refused."""
    c = run_case(4, "m3", 16, 1)
    tp, rows, out = c["tp"], c["tp"]["rows"], c["outs"][1]
    with pytest.raises(ValueError, match="route-indexed"):
        quant(c, out, None, tp)
    strided = torch.empty(258, device="cuda", dtype=torch.int32)[::2]
    with pytest.raises(ValueError, match="TP route fusion"):
        quant(c, out, rows, {**tp, "masked_m": strided})
    # 20 routes over 20 experts padded to 16 need 320 rows; give 304.
    with pytest.raises(ValueError, match="TP route fusion"):
        quant({**c, "cm": 304}, out, rows, tp)


@pytest.mark.parametrize("tokens", [3, 16, 40])
def test_public_moe_fused_route_bitwise(tokens, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_tp_route_ref", W / "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py"
    )
    ref, captured = importlib.util.module_from_spec(spec), []
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        public = ref.fused_moe
        ref.fused_moe = lambda *a, **kw: captured.append((a, kw)) or public(*a, **kw)
        ref._run_grouped_via_fused_moe(tokens=tokens, **MOE)
    finally:
        torch.set_default_device(default)
    a, kw = list(captured[-1][0]), captured[-1][1]
    a[4] = m3_routes(tokens, 17, seed=5, real=max(1, tokens - 2))
    a[3] = torch.where(a[4] >= 0, a[3], torch.zeros_like(a[3]))
    calls, route_fn = [], gm.fused_route_psum_remap
    monkeypatch.setattr(
        gm, "fused_route_psum_remap", lambda *x: calls.append(1) or route_fn(*x)
    )
    outs = []
    for fuse in ("0", "1"):
        set_fuse(fuse)
        calls.clear()
        outs.append(fm.fused_moe(*a, **kw))
        assert bool(calls) == (fuse == "0"), "route launch only on the split path"
    assert torch.equal(outs[0], outs[1]) and outs[1].isfinite().all()


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("TP route-in-quant is gfx1250 only; skipping")
        return
    parser = argparse.ArgumentParser(description="TP MoE route fused into quant")
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[1, 4, 16, 32, 64, 128, 256, 409]
    )
    tokens = parser.parse_args().tokens
    rows = [bench_route_quant(t, p, 16, 1) for p in ("m3", "drop") for t in tokens]
    df = pd.DataFrame(rows)
    aiter.logger.info("TP route + quant (us):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
