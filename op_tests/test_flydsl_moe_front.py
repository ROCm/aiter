# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MoE front: ``flydsl_moe_router_topk`` (router GEMM + biased sigmoid
top-k in one launch) against ``tgemm.mm`` + ``topk_gating``, byte for byte.

Usage:
  python op_tests/test_flydsl_moe_front.py
  python op_tests/test_flydsl_moe_front.py --num-tokens 64,192,256 --no-bench
"""

import argparse
import statistics

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_front_gfx1250 import (
    flydsl_moe_router_topk,
    router_group_counts_shape,
    router_split_k,
)
from aiter.ops.topk import topk_gating
from aiter.tuned_gemm import tgemm

D, E, TOPK = 6144, 128, 4


def _inputs(T, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(7919 * T + seed)
    x = torch.randn(T, D, generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn(E, D, generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, generator=g, device="cuda") * 0.05
    nvr = None
    if kind == "ties":  # equal experts across lanes and inside one lane
        for a, b in ((5, 77), (3, 35), (64, 96), (0, 127)):
            w[b] = w[a]
            bias[b] = bias[a]
    elif kind == "saturate":  # sigmoid == 1.0 on many experts
        x = (x.float() * 60).to(torch.bfloat16)
        bias.zero_()
    elif kind == "pad_rows" and T > 1:
        nvr = torch.tensor([T - 2], dtype=torch.int32, device="cuda")
    return x, w, bias, nvr


def _reference(x, w, bias, nvr):
    logits = tgemm.mm(x, w, None, otype=torch.bfloat16)
    tw = torch.empty(x.shape[0], TOPK, dtype=torch.float32, device="cuda")
    ti = torch.empty(x.shape[0], TOPK, dtype=torch.int32, device="cuda")
    topk_gating(
        tw, ti, logits, bias, True, 1.0, score_func="sigmoid", num_valid_rows=nvr
    )
    return tw, ti


def _graph_us(fn, reps=10, replays=40, rounds=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) * 1000 / (replays * reps))
    return statistics.median(out)


def run_case(T, kind, bench):
    sk = router_split_k(T, E, D, str(torch.bfloat16), str(torch.bfloat16))
    if sk is None:
        return None
    x, w, bias, nvr = _inputs(T, kind, 0)
    rw, ri = _reference(x, w, bias, nvr)
    cnt = torch.empty(router_group_counts_shape(T), dtype=torch.int32, device="cuda")
    nw, ni = flydsl_moe_router_topk(
        x, w, bias, TOPK, sk, num_valid_rows=nvr, group_counts=cnt
    )
    torch.cuda.synchronize()
    exact = torch.equal(rw.view(torch.int32), nw.view(torch.int32)) and torch.equal(
        ri, ni
    )
    ids = torch.full((cnt.shape[0] * 8, TOPK), -1, dtype=torch.int32, device="cuda")
    ids[:T] = ri
    grp = ids.view(cnt.shape[0], 8 * TOPK).long()
    hist = torch.zeros(cnt.shape[0], E + 1, dtype=torch.int32, device="cuda")
    hist.scatter_add_(
        1, torch.where(grp < 0, E, grp), torch.ones_like(grp, dtype=torch.int32)
    )
    row = {
        "T": T,
        "case": kind,
        "split_k": sk,
        "bit_exact": exact,
        "counts_exact": torch.equal(hist[:, :E], cnt),
    }
    if bench:

        def ref():
            _reference(x, w, bias, nvr)

        def fused():
            flydsl_moe_router_topk(
                x,
                w,
                bias,
                TOPK,
                sk,
                num_valid_rows=nvr,
                topk_weights=nw,
                topk_ids=ni,
                group_counts=cnt,
            )

        row["gemm+topk_gating us"] = round(_graph_us(ref), 2)
        row["fused us"] = round(_graph_us(fused), 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tokens", default="1,16,33,64,128,192,256,320,512")
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()
    if not get_gfx_runtime().startswith("gfx1250"):
        print("flydsl_moe_router_topk is gfx1250-only; skipped")
        return
    rows = []
    for T in [int(t) for t in args.num_tokens.split(",")]:
        for kind in ("rand", "ties", "saturate", "pad_rows"):
            row = run_case(T, kind, not args.no_bench and kind == "rand")
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    assert df["bit_exact"].all() and df["counts_exact"].all(), "fused router differs"


if __name__ == "__main__":
    main()
