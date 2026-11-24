# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bytes and timing of the MiniMax-M3.1 TP1 one-launch MoE sort
(aiter.ops.m3_tp1_prefill.moe_sort) and moe_g2_w8 on its tile list (moe_g2_w8_tiles).

The reference is aiter's moe_sorting as fused_moe calls it for the MXMOE pipeline
(block 128, the Opus aux sort, no accumulation) on the same routing: num_valid_ids, every valid sorted row (token |
slot << 24, weight, m_index), the valid blocks' experts and every reverse entry
(the rows past num_local_tokens read -1) must be equal byte for byte. The tile list
must pair each expert's blocks as moe_g2_w8's own prepass does, and moe_g2_w8_tiles
must write moe_g2_w8's route rows. A routing that names an expert twice in a row
(only NaN logits do) must still give consistent arrays: every valid row written,
each token once per expert. Registered as a presort, fused_moe must take it.

    python3 op_tests/test_m3_tp1_moe_sort.py
"""

import argparse
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.m3_tp1_prefill import (
    moe_g2_w8,
    moe_g2_w8_tiles,
    moe_sort,
    relayout_w2,
)
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")
E, H, I, TOPK, BM = 128, 6144, 3072, 4, 128


def routing(m, seed, kind):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + m)
    logits = torch.randn(m, E, generator=g)
    if kind == "skew":
        logits += 1.5 * torch.randn(E, generator=g)
    elif kind == "same4":
        logits[:, :TOPK] += 100.0
    elif kind == "half":
        logits[:, E // 2 :] = -1e9
    elif kind == "one":
        logits[:, 5] += 10.0
    tw, tid = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    tid = tid.to(torch.int32)
    if kind == "dup":  # a repeated expert id in a third of the rows
        rows = torch.arange(0, m, 3)
        tid[rows, 2] = tid[rows, 0]
    return tid.contiguous(), tw.float().contiguous()


def reference(tid, tw, nlt):
    # as fused_moe sorts for the MXMOE pipeline with its reduce GEMM2
    s = moe_sorting(
        tid,
        tw,
        E,
        H,
        dtypes.bf16,
        BM,
        None,
        nlt,
        accumulate=False,
        output_aux=AUX_SORT_OPUS,
    )
    return {
        "sorted_token_ids": s[0],
        "sorted_weights": s[1],
        "sorted_expert_ids": s[2],
        "num_valid_ids": s[3],
        "m_indices": s[5],
        "reverse_sorted": s[6],
    }


def tiles_of(eids, nvalid, ids, m):
    """moe_g2_w8's prepass in Python: valid rows per block, an expert's blocks two by two."""
    nb = nvalid // BM
    e = eids[:nb].tolist()
    t = ids[:nvalid].view(nb, BM)
    cnt = (((t & 0xFFFFFF) < m) & ((t >> 24) < TOPK)).sum(1).tolist()
    out, b = [], 0
    while b < nb:
        pair = b + 1 < nb and e[b + 1] == e[b]
        out.append((b * BM, cnt[b] + (cnt[b + 1] if pair else 0), e[b], 0))
        b += 2 if pair else 1
    return out


def ndiff(a, b):
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


@benchmark()
def test_moe_sort(m, seed=0, kind="random", local=None):
    tid, tw = routing(m, seed, kind)
    nlt = None if local is None else torch.tensor([local], dtype=torch.int32)
    got = moe_sort(tid, tw, nlt)
    ref = reference(tid, tw, nlt)
    nv = int(got["num_valid_ids"][0])
    d = {"num_valid diff": ndiff(got["num_valid_ids"], ref["num_valid_ids"])}
    for k in ("sorted_token_ids", "sorted_weights", "m_indices"):
        d[f"{k} diff"] = ndiff(got[k][:nv], ref[k][:nv])
    d["experts diff"] = ndiff(
        got["sorted_expert_ids"][: nv // BM], ref["sorted_expert_ids"][: nv // BM]
    )
    d["reverse diff"] = ndiff(got["reverse_sorted"], ref["reverse_sorted"])
    want = tiles_of(got["sorted_expert_ids"], nv, got["sorted_token_ids"], m)
    nt = int(got["g2_ctl"][0])
    d["tiles diff"] = int(
        nt != len(want) or got["g2_tiles"][:nt].tolist() != [list(t) for t in want]
    )
    d["queues set"] = int(got["g2_ctl"][1:9].tolist() == [0] * 8)
    assert (
        all(v == 0 for k, v in d.items() if k != "queues set") and d["queues set"]
    ), f"m={m} seed={seed} {kind} local={local}: {d}"
    return {**d, "rows": nv}


@benchmark()
def test_duplicates_stay_consistent(m):
    """Every valid row holds a real (token, slot) or a pad, each token once per expert."""
    tid, tw = routing(m, 1, "dup")
    got = moe_sort(tid, tw)
    nv = int(got["num_valid_ids"][0])
    ids = got["sorted_token_ids"][:nv]
    tok, slot = ids & 0xFFFFFF, ids >> 24
    real = (tok < m) & (slot < TOPK)
    pad = (tok == m) & (slot == TOPK)
    expert = got["sorted_expert_ids"][: nv // BM].repeat_interleave(BM)
    key = tok[real].long() * E + expert[real].long()
    rev = got["reverse_sorted"].view(m, TOPK)
    first = torch.ones_like(rev, dtype=torch.bool)
    for s in range(1, TOPK):
        first[:, s] = (tid[:, s : s + 1] != tid[:, :s]).all(1)
    d = {
        "garbage rows": int((~(real | pad)).sum()),
        "repeated (token, expert)": int(key.numel() - torch.unique(key).numel()),
        "routes": int(real.sum()) - int(first.sum()),
        "dropped slots not -1": int((rev[~first] != -1).sum()),
        "reverse misses": int((rev[first] < 0).sum()),
    }
    assert all(v == 0 for v in d.values()), f"m={m}: {d}"
    return d


@benchmark()
def test_w8_tiles(m, kind="random", local=None):
    tid, tw = routing(m, 2, kind)
    nlt = None if local is None else torch.tensor([local], dtype=torch.int32)
    got = moe_sort(tid, tw, nlt)
    rows = got["sorted_token_ids"].numel()
    g = torch.Generator(device="cuda").manual_seed(m)
    a2 = torch.randint(0, 256, (rows, I // 2), dtype=torch.uint8, generator=g)
    a2s = torch.randint(118, 131, (rows, I // 32), dtype=torch.uint8, generator=g)
    out = {}
    for name in ("w8", "tiles"):
        out[name] = torch.full((m, TOPK, H), float("nan"), dtype=torch.bfloat16)
        args = (a2, a2s, W2B, W2BS, got["sorted_token_ids"], got["sorted_weights"])
        if name == "w8":
            moe_g2_w8(
                *args,
                got["sorted_expert_ids"],
                got["num_valid_ids"],
                out[name],
                m,
                TOPK,
            )
        else:
            moe_g2_w8_tiles(*args, got["g2_tiles"], got["g2_ctl"], out[name], m, TOPK)
    n = local if local is not None else m
    d = {
        "route diff": int(
            (
                out["w8"][:n].view(torch.uint16) != out["tiles"][:n].view(torch.uint16)
            ).sum()
        ),
        "unwritten": int(torch.isnan(out["tiles"][:n]).sum()),
    }
    assert all(v == 0 for v in d.values()), f"m={m} {kind} local={local}: {d}"
    return d


def test_presort_is_taken():
    """fused_moe_prepare_stage2 on a 128-block MXMOE row takes the registered sort."""
    import aiter.fused_moe as fm

    m = 8192
    tid, tw = routing(m, 3, "random")
    nlt = torch.tensor([m - 7], dtype=torch.int32)
    seen = []
    orig = fm.match_presorted_moe
    fm.match_presorted_moe = (
        lambda aux, *a, **k: seen.append(orig(aux, *a, **k)) or seen[-1]
    )
    try:
        aux = moe_sort(tid, tw, nlt)
        fm.register_presorted_moe(aux)
        assert (
            fm.match_presorted_moe(
                aux, tid, tw, block_m=BM, num_experts=E, topk=TOPK, num_local_tokens=nlt
            )
            is aux
        )
        assert (
            fm.match_presorted_moe(aux, tid, tw, block_m=BM, num_experts=E, topk=TOPK)
            is None
        )
        assert (
            fm.match_presorted_moe(
                aux, tid, tw, block_m=64, num_experts=E, topk=TOPK, num_local_tokens=nlt
            )
            is None
        )
    finally:
        fm.match_presorted_moe = orig
        fm.register_presorted_moe(None)
    aiter.logger.info("moe_sort presort accepted for its own num_local_tokens only")


@benchmark()
def perf_moe_sort(m):
    tid, tw = routing(m, 0, "random")
    nlt = torch.tensor([m], dtype=torch.int32)
    _, us = run_perftest(moe_sort, tid, tw, nlt, num_iters=50, num_warmup=5)
    _, us_ref = run_perftest(reference, tid, tw, nlt, num_iters=50, num_warmup=5)
    return {"moe_sort us": us, "moe_sorting us": us_ref}


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--quick", action="store_true", help="fewer cases")
args = parser.parse_args()

if get_gfx_runtime() != "gfx950":
    aiter.logger.info("m3_tp1 moe_sort targets gfx950; skipping")
    sys.exit(0)

wq = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
ws = torch.randint(118, 127, (E, H, I // 32), dtype=torch.uint8)
W2B, W2BS = relayout_w2(wq, ws)
sizes = (
    [4096, 8192] if args.quick else [1, 37, 2049, 4096, 5000, 8191, 8192, 16384, 32768]
)
rows = [test_moe_sort(m) for m in sizes]
for kind in ("skew", "same4", "half", "one"):
    rows.append(test_moe_sort(8192, 1, kind))
for m, local in ((8192, 8192), (8192, 8185), (8192, 4097), (4096, 1), (5000, 0)):
    rows.append(test_moe_sort(m, 2, "random", local))
aiter.logger.info(
    f"moe_sort vs moe_sorting:\n{pd.DataFrame(rows).to_markdown(index=False)}"
)
dups = [test_duplicates_stay_consistent(m) for m in (4096, 8192)]
aiter.logger.info(
    f"repeated expert ids:\n{pd.DataFrame(dups).to_markdown(index=False)}"
)
w8 = [test_w8_tiles(m) for m in (4096, 8192, 16384)]
w8 += [test_w8_tiles(8192, "skew"), test_w8_tiles(8192, "random", 6001)]
aiter.logger.info(
    f"moe_g2_w8_tiles vs moe_g2_w8:\n{pd.DataFrame(w8).to_markdown(index=False)}"
)
test_presort_is_taken()
perf = [perf_moe_sort(m) for m in (4096, 8192, 16384, 32768)]
aiter.logger.info(f"timing:\n{pd.DataFrame(perf).to_markdown(index=False)}")
