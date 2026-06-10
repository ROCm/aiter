# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end test of the jagged_dense_bmm_dispatch_v2 layer (gfx950).

Verifies:
  1. The public v2 wrapper runs on each headline shape with cos>0.999 vs eager.
  2. For an in-table shape, resolve_config picks the JSON table winner.
  3. An OFF-table shape exercises the D-bucketed heuristic fallback (cos>0.999).

Run (inside container, GPU 7):
    HIP_VISIBLE_DEVICES=7 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python op_tests/flydsl_tests/test_jdbba_dispatch_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch_v2 import (
    _dispatch_json_paths,
    jagged_dense_bmm_dispatched,
    resolve_config,
    shape_id,
)

MI = 7680


def _build_inputs(B, D, Kout, Mi):
    N, K = Kout, D  # output N=Kout, reduction K=D
    so = torch.zeros(B + 1, dtype=torch.int32)
    for i in range(B):
        so[i + 1] = so[i] + Mi
    L = B * Mi
    jag = torch.randn(L, K, dtype=torch.bfloat16).cuda()
    dense = torch.randn(B, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(B, N, dtype=torch.bfloat16).cuda()
    sod = so.cuda()
    dt = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bf = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + 128, N, dtype=torch.bfloat16).cuda()
    tA = flyc.from_dlpack(jag).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    return dict(tA=tA, tC=tC, dt=dt, bf=bf, sod=sod, jag=jag, dense=dense,
                bias=bias, so=so, out=out, L=L, B=B, N=N, msl=Mi)


def _cos(d):
    L, B, N = d["L"], d["B"], d["N"]
    ref = torch.zeros(L, N, dtype=torch.float32, device="cuda")
    for b in range(B):
        s, e = int(d["so"][b]), int(d["so"][b + 1])
        if e > s:
            ref[s:e] = d["jag"][s:e].float() @ d["dense"][b].float() + d["bias"][b].float()[None, :]
    got = d["out"][:L].float()
    return torch.nn.functional.cosine_similarity(got.reshape(-1), ref.reshape(-1), dim=0).item()


def main() -> int:
    table = json.loads(_dispatch_json_paths()[0].read_text())
    winners = table.get("winners") or {}
    assert table.get("gfx") == "gfx950", "JSON must be tagged gfx950"

    st = torch.cuda.current_stream()
    ok = True

    # 1 + 2: headline (in-table) shapes -> correctness + table winner selected.
    headline = [(120, 256, 256), (120, 512, 512), (1024, 256, 256), (1024, 512, 512)]
    for B, D, Kout in headline:
        sid = shape_id(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=MI)
        cfg = resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=MI)
        d = _build_inputs(B, D, Kout, MI)
        jagged_dense_bmm_dispatched(
            d["tC"], d["tA"], d["dt"], d["bf"], d["sod"], B, d["msl"], stream=st
        )
        torch.cuda.synchronize()
        cos = _cos(d)
        in_table = sid in winners
        picks_winner = True
        if in_table:
            w = winners[sid]
            picks_winner = cfg["xcd_c"] == w["xcd_c"] and cfg["xcd_w"] == w["xcd_w"]
        status = "OK" if (cos > 0.999 and (not in_table or picks_winner)) else "FAIL"
        if status == "FAIL":
            ok = False
        print(
            f"[{status}] {sid:22} in_table={in_table} "
            f"resolved(xcd_c={cfg['xcd_c']},xcd_w={cfg['xcd_w']}) "
            f"picks_winner={picks_winner} cos={cos:.6f}"
        )

    # 3: OFF-table shape -> heuristic fallback path (not in winners) + correctness.
    B, D, Kout = 120, 384, 384  # D=384 -> d_le_512 bucket, not a headline cell
    sid = shape_id(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=MI)
    assert sid not in winners, "off-table shape unexpectedly in winners"
    cfg = resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=MI)
    d = _build_inputs(B, D, Kout, MI)
    jagged_dense_bmm_dispatched(
        d["tC"], d["tA"], d["dt"], d["bf"], d["sod"], B, d["msl"], stream=st
    )
    torch.cuda.synchronize()
    cos = _cos(d)
    status = "OK" if cos > 0.999 else "FAIL"
    if status == "FAIL":
        ok = False
    print(
        f"[{status}] {sid:22} FALLBACK(heuristic) "
        f"resolved(xcd_c={cfg['xcd_c']},xcd_w={cfg['xcd_w']}) cos={cos:.6f}"
    )

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
