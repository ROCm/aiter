# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_topk_route_quant (gfx1250): one launch for topk_softmax(renormalize) +
fused_route_psum_remap + flydsl_moe_fused_quant_preshuffle at decode sizes.

Correctness is bitwise against the split path: top-k ids and weights, masked_m,
psum, every route's MXFP4 payload and preshuffled e8m0 scale bytes (grouped rows
inside an expert may be permuted: the split path orders them by LDS atomics),
and the whole grouped MoE output through MoE1, MoE2 and the combine.

``python op_tests/test_flydsl_moe_topk_route_quant.py`` also prints the CUDA
graph replay time of the split sequence against the fused launch.
"""

import argparse

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx1250", reason="gfx1250 only"
)

E, TOPK, D, ALIGN = 256, 8, 2048, 16
ROWS = [1, 5, 13, 16, 17, 33, 64, 65, 80, 100, 128, 160, 340, 512]


def _align_up(x, a):
    return (x + a - 1) // a * a


def _contiguous_m(m):
    return max(ALIGN, _align_up(m * TOPK + E * ALIGN - TOPK, ALIGN))


def _inputs(m, kind, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + m)
    if kind == "ties":
        # Few distinct values, a +0 row and a -0-heavy row: the tie-breaks.
        full = (
            torch.randint(-4, 4, (m, E + 1), device="cuda", generator=g).float() * 0.5
        ).to(torch.bfloat16)
        full[0] = 0.0
        if m > 1:
            full[1, ::2] = -0.0
    else:
        full = (torch.randn(m, E + 1, device="cuda", generator=g) * 3).to(
            torch.bfloat16
        )
    # The router GEMM's [routed | shared-expert gate] output: odd row stride.
    logits = full[:, :E]
    hidden = torch.randn(m, D, device="cuda", generator=g).to(torch.bfloat16)
    return logits, hidden


def split_path(logits, hidden, nvr):
    from aiter.ops.flydsl.grouped_moe_gfx1250 import fused_route_psum_remap
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

    m = hidden.shape[0]
    w = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
    ids = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
    aiter.topk_softmax(w, ids, torch.empty_like(ids), logits, True, 0, "", nvr)
    max_m = max(ALIGN, _align_up(m * TOPK, ALIGN))
    masked_m, rows, psum = fused_route_psum_remap(ids, E, max_m, ALIGN)
    payload, scale = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, m, D),
        1,
        _contiguous_m(m),
        wmma_rep=1,
        quant_mode="fp4",
        masked_m=None,
        topids_to_rows=rows,
        source_topk=TOPK,
        num_valid_routes=None if nvr is None else (nvr * TOPK),
    )
    return w, ids, masked_m, rows, psum, payload, scale


def fused_path(logits, hidden, nvr):
    from aiter.ops.flydsl.grouped_moe_gfx1250 import fused_topk_route_quant

    return fused_topk_route_quant(
        logits,
        hidden,
        TOPK,
        align_m=ALIGN,
        contiguous_m=_contiguous_m(hidden.shape[0]),
        wmma_rep=1,
        quant_mode="fp4",
        num_valid_rows=nvr,
    )


def _scale_row(scale, row):
    """A grouped row's 64 e8m0 bytes out of the wmma_rep=1 preshuffled layout."""
    flat = scale.reshape(-1)
    dw = D // 128
    out = []
    for sd in range(dw):
        d = (row // 16) * 16 * dw + sd * 16 + row % 16
        out.append(flat[d * 4 : d * 4 + 4])
    return torch.cat(out)


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("kind", ["randn", "ties"])
@pytest.mark.parametrize("pad", [None, 0, 3])
def test_matches_split_path(m, kind, pad):
    logits, hidden = _inputs(m, kind)
    nv = m if pad is None else max(m - pad, 0)
    nvr = None if pad is None else torch.tensor([nv], dtype=torch.int32, device="cuda")
    w, ids, mm, rows, psum, pay, sc = split_path(logits, hidden, nvr)
    w2, ids2, mm2, rows2, psum2, pay2, sc2 = fused_path(logits, hidden, nvr)
    torch.cuda.synchronize()
    assert torch.equal(ids, ids2)
    assert torch.equal(w.view(torch.int32), w2.view(torch.int32))
    assert torch.equal(mm, mm2) and torch.equal(psum, psum2)
    assert torch.equal(rows[nv:], rows2[nv:])  # dropped routes: -1
    ids_c, rows_c, rows2_c = ids[:nv].cpu(), rows[:nv].cpu(), rows2[:nv].cpu()
    for e in range(E):
        sel = ids_c == e
        assert sorted(rows_c[sel].tolist()) == sorted(rows2_c[sel].tolist())
    for t in range(nv):
        for k in range(TOPK):
            r1, r2 = int(rows_c[t, k]), int(rows2_c[t, k])
            assert torch.equal(pay[0, r1], pay2[0, r2])
            assert torch.equal(_scale_row(sc, r1), _scale_row(sc2, r2))


def _moe_weights(inter):
    g = torch.Generator(device="cuda").manual_seed(1)
    u8 = torch.uint8
    w1 = torch.randint(
        0, 256, (E, 2 * inter, D // 2), dtype=u8, device="cuda", generator=g
    )
    w2 = torch.randint(0, 256, (E, D, inter // 2), dtype=u8, device="cuda", generator=g)
    s1 = torch.randint(
        118, 126, (E, 2 * inter, D // 32), dtype=u8, device="cuda", generator=g
    )
    s2 = torch.randint(
        118, 126, (E, D, inter // 32), dtype=u8, device="cuda", generator=g
    )
    return w1.view(dtypes.fp4x2), w2.view(dtypes.fp4x2), s1, s2


def _moe(logits, hidden, nvr, weights, deferred):
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.grouped_moe_gfx1250 import defer_topk_softmax

    w1, w2, s1, s2 = weights
    m = hidden.shape[0]
    w = torch.empty(m, TOPK, dtype=torch.float32, device="cuda")
    ids = torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
    if deferred:
        defer_topk_softmax(w, ids, logits, True, nvr)
    else:
        aiter.topk_softmax(w, ids, torch.empty_like(ids), logits, True, 0, "", nvr)
    out = fused_moe(
        hidden,
        w1,
        w2,
        w,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
        num_local_tokens=nvr,
        gate_mode="interleave",
    )
    return w, ids, out


@pytest.mark.parametrize("inter", [128, 256])
@pytest.mark.parametrize("m", ROWS)
def test_grouped_moe_output_bitwise(m, inter):
    weights = _moe_weights(inter)
    logits, hidden = _inputs(m, "randn")
    nv = max(m - 3, 0)
    nvr = torch.tensor([nv], dtype=torch.int32, device="cuda")
    wa, ia, oa = _moe(logits, hidden, nvr, weights, deferred=False)
    wb, ib, ob = _moe(logits, hidden, nvr, weights, deferred=True)
    torch.cuda.synchronize()
    assert torch.equal(ia, ib)
    assert torch.equal(wa.view(torch.int32), wb.view(torch.int32))
    assert torch.equal(oa[:nv].view(torch.int16), ob[:nv].view(torch.int16))


def test_deferred_falls_back_when_not_fusable():
    """Rows past the fused kernel's limit: the grouped path runs the plain
    top-k first, so the buffers and the output are still right."""
    weights = _moe_weights(256)
    logits, hidden = _inputs(600, "randn")
    _, ia, oa = _moe(logits, hidden, None, weights, deferred=False)
    _, ib, ob = _moe(logits, hidden, None, weights, deferred=True)
    torch.cuda.synchronize()
    assert torch.equal(ia, ib) and torch.equal(
        oa.view(torch.int16), ob.view(torch.int16)
    )


def _graph_us(fn, reps=20, rounds=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    for _ in range(5):
        graph.replay()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    times = []
    for _ in range(rounds):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (10 * reps))
    return sorted(times)[len(times) // 2]


def benchmark(rows, repeats=3):
    print(
        "| rows | split: topk + route_psum + quant (us) | fused (us) | saved (us) |\n"
        "|---:|---:|---:|---:|"
    )
    for m in rows:
        logits, hidden = _inputs(m, "randn")
        nvr = torch.tensor([m], dtype=torch.int32, device="cuda")
        ts, tf = [], []
        for _ in range(repeats):  # interleaved, best of
            ts.append(_graph_us(lambda a=(logits, hidden, nvr): split_path(*a)))
            tf.append(_graph_us(lambda a=(logits, hidden, nvr): fused_path(*a)))
        print(f"| {m} | {min(ts):.2f} | {min(tf):.2f} | {min(ts) - min(tf):.2f} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows", type=int, nargs="+", default=[13, 16, 64, 80, 128, 340, 512]
    )
    benchmark(ap.parse_args().rows)
