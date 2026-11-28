# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.mla_reduce_v1 on persistent decode reduce tables (with a final map) against a torch reference, every tile.

Covers what kn_mla_reduce_v1_adaptive (gfx1250, 512-wide heads) splits between its wave-head grid and its helpers:
  * synthetic tables: per-tile split counts from 1 to 100 (more than 32 walked in chunks), 1 / 2 / 4-row tiles,
    16 / 32 / 64 / 96 / 128 heads, a padded table tail (empty tiles, as a graph-batch-sized table), LSE on / off;
  * planner tables (get_mla_metadata_v1, persistent fast mode, page 1) for speculative-decode verify (96 heads, 5 rows, causal,
    round-robin) and draft blocks (96 / 64 heads, 4 rows, non-causal) of skewed and uniform batches.
Run once per routing: default (adaptive on gfx1250), AITER_MLA_REDUCE_ADAPTIVE=0 (wave-head), and
AITER_MLA_REDUCE_WAVE_HEAD=0 (kn_mla_reduce_v1); the routing is read once per process.

  python3 op_tests/test_mla_reduce_adaptive.py
"""

import argparse
import os
import sys

os.environ.setdefault(
    "AITER_MLA_DECODE_PS1_FLYDSL", "1"
)  # the gfx1250 persistent (PS1) decode planner route

import torch

import aiter
from aiter import dtypes

DV = 512


def reference(indptr, fmap, pmap, po, plse, heads):
    """For every tile with splits: out = sum(w o) / sum(w), lse = log(sum w) + m, w = exp(lse - m)."""
    ip, fm, pm = indptr.cpu().tolist(), fmap.view(-1, 2).cpu().tolist(), pmap.cpu()
    rows, outs, lses = [], [], []
    for t in range(len(ip) - 1):
        s, e = ip[t], ip[t + 1]
        if e <= s:
            continue
        sl = pm[s:e].to(po.device).long()
        for r in range(fm[t][1] - fm[t][0]):
            lse = plse[sl + r].view(len(sl), heads)
            out = po[sl + r].view(len(sl), heads, DV)
            m = lse.max(0).values
            m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
            w = torch.exp(lse - m)
            sw = w.sum(0)
            outs.append((w.unsqueeze(-1) * out).sum(0) / sw.unsqueeze(-1))
            lses.append(torch.log(sw) + m)
            rows.append(fm[t][0] + r)
    return torch.tensor(rows, device=po.device), torch.stack(outs), torch.stack(lses)


def check(name, indptr, fmap, pmap, heads, qlen, nrows, with_lse=True, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    live = int(
        indptr[-1].item()
    )  # the map's tail past the last tile's splits is not written by the planner
    nslot = int(pmap[:live].max().item()) + qlen + 1 if live else qlen + 1
    po = torch.randn(nslot, 1, heads, DV, dtype=dtypes.fp32, device="cuda", generator=g)
    plse = (
        torch.randn(nslot, 1, heads, 1, dtype=dtypes.fp32, device="cuda", generator=g)
        * 2
        + 6
    )
    out = torch.full((nrows, heads, DV), float("nan"), dtype=dtypes.bf16, device="cuda")
    lse = (
        torch.full((nrows, heads), float("nan"), dtype=dtypes.fp32, device="cuda")
        if with_lse
        else None
    )
    aiter.mla_reduce_v1(po, plse, indptr, fmap, pmap, qlen, 16, out, lse)
    torch.cuda.synchronize()
    rows, ro, rl = reference(
        indptr, fmap, pmap, po.view(nslot, -1), plse.view(nslot, -1), heads
    )
    o = out[rows].float()
    err = ((o - ro).abs().max() / ro.abs().max().clamp_min(1e-6)).item()
    ok = err < 2e-2 and bool(torch.isfinite(o).all())
    lerr = 0.0
    if with_lse:
        lerr = (lse[rows] - rl).abs().max().item()
        ok = ok and lerr < 1e-3
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: rows {len(rows)} out err {err:.2e} lse err {lerr:.2e}",
        flush=True,
    )
    return ok


def synthetic(heads, rows_per_tile, splits, pad_tiles, seed):
    """Tiles of rows_per_tile rows with the given split counts (contiguous slots, rows interleaved per split as the
    planner's partial map lays them out), then pad_tiles empty tiles at the table's end.
    """
    ip, fm, pm = [0], [], []
    slot = 0
    for t, n in enumerate(splits):
        for _ in range(n):
            pm.append(slot)
            slot += rows_per_tile
        ip.append(ip[-1] + n)
        fm.append([t * rows_per_tile, (t + 1) * rows_per_tile])
    for _ in range(pad_tiles):
        ip.append(ip[-1])
        fm.append([len(splits) * rows_per_tile, len(splits) * rows_per_tile])
    dev = "cuda"
    return (
        torch.tensor(ip, dtype=torch.int32, device=dev),
        torch.tensor(fm, dtype=torch.int32, device=dev),
        torch.tensor(pm, dtype=torch.int32, device=dev),
        len(splits) * rows_per_tile,
    )


def planner(heads, qlen, lens, causal, cprr, table_bs):
    dev = torch.device("cuda")
    B, dcp = len(lens), 8
    local = [len(range(0, L, dcp)) for L in lens]
    qo = torch.arange(B + 1, dtype=torch.int32, device=dev) * qlen
    kv = torch.zeros(B + 1, dtype=torch.int32, device=dev)
    kv[1:] = (
        torch.tensor(local, dtype=torch.int64, device=dev).cumsum(0).to(torch.int32)
    )
    last = torch.ones(B, dtype=torch.int32, device=dev)
    info = aiter.get_mla_metadata_info_v1(
        max(B, table_bs),
        qlen,
        heads,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
    )
    wmd, wi, wis, ri, rfm, rpm = [torch.empty(s, dtype=d, device=dev) for s, d in info]
    aiter.get_mla_metadata_v1(
        qo,
        kv,
        last,
        heads,
        1,
        causal,
        wmd,
        wis,
        wi,
        ri,
        rfm,
        rpm,
        page_size=1,
        dtype_q=dtypes.fp8,
        dtype_kv=dtypes.fp8,
        is_cp_round_robin=cprr,
        kv_granularity=16,
        max_seqlen_qo=qlen,
        uni_seqlen_qo=qlen,
        fast_mode=1,
        max_split_per_batch=-1,
    )
    return ri, rfm, rpm, B * qlen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ok = True
    g = torch.Generator().manual_seed(7)
    # synthetic: skewed / uniform split counts, chunked (> 32) splits, multi-row tiles, padded tails
    for heads, rpt in ((96, 1), (128, 1), (64, 2), (32, 4), (16, 4), (64, 1)):
        for kind in ("skewed", "uniform", "long", "single"):
            if kind == "skewed":
                splits = [int(x) for x in torch.randint(1, 7, (24,), generator=g)] + [
                    47,
                    30,
                    18,
                    13,
                ]
            elif kind == "uniform":
                splits = [4] * 40
            elif kind == "long":
                splits = [100, 65, 33, 32, 25, 2, 1]
            else:
                splits = [1] * 9
            ip, fm, pm, nr = synthetic(heads, rpt, splits, pad_tiles=37, seed=1)
            for with_lse in ((True, False) if kind == "long" else (True,)):
                ok &= check(
                    f"synthetic H={heads} rows={rpt} {kind} lse={with_lse}",
                    ip,
                    fm,
                    pm,
                    heads,
                    rpt,
                    nr,
                    with_lse,
                )
    # planner tables: speculative verify / draft-block shapes on 8-way decode context parallel
    k = 1024
    cases = [
        (
            "verify B5 skewed",
            96,
            5,
            [190 * k, 2 * k, 2 * k, 2 * k, 2 * k],
            True,
            True,
            32,
        ),
        (
            "verify B6 mix",
            96,
            5,
            [190 * k, 40 * k, 35 * k, 30 * k, 25 * k, 10 * k],
            True,
            True,
            32,
        ),
        ("verify uniform", 96, 5, [32 * k] * 16, True, True, 32),
        (
            "verify B128",
            96,
            5,
            [int(x) for x in torch.randint(2 * k, 120 * k, (128,), generator=g)],
            True,
            True,
            128,
        ),
        (
            "draft96 B5 skewed",
            96,
            4,
            [150 * k, 120 * k, 9 * k, 3 * k, 1 * k],
            False,
            False,
            32,
        ),
        (
            "draft96 B128",
            96,
            4,
            [int(x) for x in torch.randint(2 * k, 120 * k, (128,), generator=g)],
            False,
            False,
            128,
        ),
        (
            "draft64 B5 skewed",
            64,
            4,
            [150 * k, 120 * k, 9 * k, 3 * k, 1 * k],
            False,
            False,
            32,
        ),
        ("draft64 uniform", 64, 4, [64 * k] * 8, False, False, 8),
    ]
    if a.quick:
        cases = cases[:2]
    for name, heads, q, lens, causal, cprr, tbs in cases:
        ri, rfm, rpm, nr = planner(heads, q, lens, causal, cprr, tbs)
        n_live = int((ri[1:] > ri[:-1]).sum().item())
        last_row = (
            int(rfm.view(-1, 2)[: ri.numel() - 1][ri[1:] > ri[:-1]][:, 1].max().item())
            if n_live
            else 0
        )
        if (
            last_row > nr
        ):  # an aiter without the 32/64-head multi-token decode route folds the heads instead
            print(
                f"SKIP planner {name}: the plan folds {heads} heads (needs the multi-token PS1 route)",
                flush=True,
            )
            continue
        ok &= check(
            f"planner {name} (H={heads} q={q} table {tbs})", ri, rfm, rpm, heads, q, nr
        )
    print("ALL PASSED" if ok else "FAILURES", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
