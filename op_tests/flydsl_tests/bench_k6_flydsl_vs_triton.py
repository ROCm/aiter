# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone K6: FlyDSL vs Triton, across the batch*head range.

Values are irrelevant to timing, so h / v_new are random rather than produced by
a real K5 run. Only the shapes and layouts have to be faithful.

BV=128 is the point of interest on gfx950: it puts the whole V=128 axis in one
CTA, so q and k are re-read only across the GQA ratio instead of across the GQA
ratio times the V-tile count.
"""

from __future__ import annotations

import argparse
import json

import torch

from aiter.ops.flydsl.gdn_o_kernels import chunk_fwd_o_flydsl
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
    chunk_fwd_o_opt_vk,
)

BT = 64


def _bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us


def build(*, B, H, Hg, K, V, seqlen, device="cuda"):
    dt = torch.bfloat16
    T = B * seqlen
    cu = torch.tensor(
        [i * seqlen for i in range(B + 1)], dtype=torch.int32, device=device
    )
    nt_flat = B * ((seqlen + BT - 1) // BT)
    return {
        "q": torch.randn(1, T, Hg, K, dtype=dt, device=device) * 0.1,
        "k": torch.randn(1, T, Hg, K, dtype=dt, device=device) * 0.1,
        "v": torch.randn(1, H, T, V, dtype=dt, device=device) * 0.1,
        "h": torch.randn(nt_flat, H, V, K, dtype=dt, device=device) * 0.01,
        "g": -torch.rand(1, H, T, dtype=torch.float32, device=device).cumsum(dim=2),
        "o": torch.empty(1, T, H, V, dtype=dt, device=device),
        "cu": cu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--bv", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    K = V = 128
    scale = K**-0.5
    # (TP, B): TP sets H = 64//TP and Hg = 16//TP, matching the K5 sweep shapes.
    points = [(8, 1), (8, 2), (8, 4), (8, 8), (4, 8), (2, 8), (1, 8)]

    hdr = f"{'B*H':>5} {'H':>3} {'B':>2} {'triton':>9}"
    for bv in args.bv:
        hdr += f" {'fly bv' + str(bv):>10} {'speedup':>8}"
    print(hdr)

    rows = []
    for tp, B in points:
        H, Hg = 64 // tp, 16 // tp
        t = build(B=B, H=H, Hg=Hg, K=K, V=V, seqlen=args.seqlen)
        common = dict(
            q=t["q"],
            k=t["k"],
            v=t["v"],
            h=t["h"],
            g=t["g"],
            o=t["o"],
            scale=scale,
            cu_seqlens=t["cu"],
            use_exp2=False,
        )
        us_t = _bench(lambda: chunk_fwd_o_opt_vk(**common))
        rec = {"BH": B * H, "H": H, "B": B, "triton": us_t}
        line = f"{B * H:5d} {H:3d} {B:2d} {us_t:9.1f}"
        for bv in args.bv:
            try:
                us_f = _bench(lambda: chunk_fwd_o_flydsl(BV=bv, **common))
                rec[f"flydsl_bv{bv}"] = us_f
                line += f" {us_f:10.1f} {us_t / us_f:7.2f}x"
            except Exception as exc:  # noqa: BLE001
                rec[f"flydsl_bv{bv}"] = None
                rec[f"flydsl_bv{bv}_error"] = f"{type(exc).__name__}: {exc}"
                line += f" {'n/a':>10} {'--':>8}"
        print(line)
        rows.append(rec)
        del t
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"seqlen": args.seqlen, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
