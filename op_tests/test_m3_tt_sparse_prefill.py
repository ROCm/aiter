# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""tt_sparse_prefill against the GQA-16 ASM paged attention it replaces and an fp32
reference, on MiniMax-M3 TP1 shapes: 4 KV x 16 query heads (8 KV heads in one case),
head_dim 128, top-16 blocks.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.m3_tt_sparse_prefill import (
    TT_MAX_SEQ_LEN,
    TT_TQ,
    is_supported,
    tt_sparse_prefill,
)
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

HKV, G, D, TOPK, BLK = 4, 16, 128, 16, 128
SCALE = D**-0.5
ASM_MAX_ROWS = 1 << 16  # rows one pa_fwd_asm launch takes on gfx950

# name: (requests as (cached prefix, new tokens), block selection[, KV heads if not 4])
CASES = {
    "ctx8k": ([(0, 8192)], "correlated"),
    "ctx16k": ([(8192, 8192)], "correlated"),
    "ctx24k": ([(16384, 8192)], "correlated"),
    "ctx32k": ([(24576, 8192)], "correlated"),
    "ctx32k_random": ([(24576, 8192)], "random"),
    "chunk3791": ([(8401, 3791)], "correlated"),
    "chunk4096": ([(28672, 4096)], "random"),
    "chunk16k": ([(16384, 16384)], "correlated"),
    "chunk32k": ([(0, 32768)], "correlated"),
    "ragged5": (
        [(16383, 1), (8000, 13), (20000, 2500), (4000, 4001), (30000, 1677)],
        "correlated",
    ),
    "ragged3": ([(24499, 77), (12000, 4023), (0, 4092)], "random"),
    "ctx512k": ([(TT_MAX_SEQ_LEN - 1024, 1024)], "random"),
    "hkv8": ([(8192, 8192)], "correlated", 8),
}


def make_pool(blocks):
    """FP8 page-16 SHUFFLE K / V (8 pages a 128-token block) with fp32 per-key scales."""
    pages = blocks * 8
    k = (torch.randn(pages, HKV, 8, 16, 16) * 2).to(dtypes.fp8)
    v = (torch.randn(pages, HKV, 1, D, 16) * 2).to(dtypes.fp8)
    ks = torch.empty(pages, HKV, 16).uniform_(0.25, 0.75)
    vs = torch.empty(pages, HKV, 16).uniform_(0.25, 0.75)
    return k, v, ks, vs


def as_hkv(pool, hkv):
    """The pool's bytes with hkv KV heads (4 / hkv times the pages)."""
    return tuple(t.view(t.shape[0] * HKV // hkv, hkv, *t.shape[2:]) for t in pool)


def make_step(reqs, pool_blocks, mode, gen, hkv=HKV):
    """Each token selects its own block and 15 earlier ones; "correlated" shares one pick
    per run of 64 tokens up to noise, as the indexer's neighbouring tokens do."""
    nblk = [(p + n + BLK - 1) // BLK for p, n in reqs]
    perm = torch.randperm(pool_blocks, generator=gen, device="cuda")
    bt = torch.zeros(len(reqs), max(nblk), dtype=torch.int32)
    topk, req, kvl, used = [], [], [], 0
    for b, ((p, n), nb) in enumerate(zip(reqs, nblk)):
        bt[b, :nb] = perm[used : used + nb]
        used += nb
        pos = torch.arange(p, p + n)
        own = pos // BLK
        r = torch.rand(hkv, n, nb, generator=gen)
        if mode == "correlated":
            base = torch.rand(hkv, (n + 63) // 64, nb, generator=gen)
            r = base.repeat_interleave(64, 1)[:, :n] + 0.25 * r
        r = torch.where(torch.arange(nb) < own[:, None], r, -1.0)
        val, idx = r.topk(min(TOPK - 1, nb), dim=-1)
        sel = torch.full((hkv, n, TOPK), -1, dtype=torch.int64)
        sel[..., 0] = own
        sel[..., 1 : 1 + idx.shape[-1]] = torch.where(val >= 0, idx, -1)
        topk.append(sel)
        req.append(torch.full((n,), b))
        kvl.append(pos + 1)
    i32 = torch.int32
    return (
        torch.cat(topk, 1).to(i32),
        bt,
        torch.cat(req).to(i32),
        torch.cat(kvl).to(i32),
    )


def tile_map(reqs):
    """TT_TQ-token tiles that never straddle a request, plus two empty padding tiles."""
    t0, n, start = [], [], 0
    for _, length in reqs:
        for s in range(0, length, TT_TQ):
            t0.append(start + s)
            n.append(min(TT_TQ, length - s))
        start += length
    t0, n = t0 + [0, 0], n + [0, 0]
    return torch.tensor(t0, dtype=torch.int32), torch.tensor(n, dtype=torch.int32)


def asm_tables(topk, bt, req, kvl, gen=None):
    """pa_fwd_asm's view of the step: a (token, KV head) row per page-16 list, the full
    blocks first and the token's own (partial) block last; `gen` reorders the full ones.
    """
    hkv, T = topk.shape[:2]
    sel = topk.transpose(0, 1).reshape(T * hkv, TOPK).long()
    kv = kvl.long().repeat_interleave(hkv)
    own = ((kv - 1) // BLK)[:, None]
    full = (sel >= 0) & (sel < own)
    key = torch.where(full, 0.0, torch.where(sel == own, 2.0, 3.0))
    if gen is not None:
        key = torch.where(full, torch.rand(key.shape, generator=gen), key)
    sel = sel.gather(1, key.argsort(dim=1, stable=True))
    valid = (sel >= 0) & (sel <= own)
    phys = bt[req.long().repeat_interleave(hkv)[:, None], sel.clamp(min=0)].long()
    head = (torch.arange(T * hkv) % hkv)[:, None, None]
    pages = (phys[:, :, None] * 8 + torch.arange(8)) * hkv + head
    pages = torch.where(valid[:, :, None], pages, 0).reshape(T * hkv, TOPK * 8)
    ctx = full.sum(1) * BLK + kv - own[:, 0] * BLK
    return pages.to(torch.int32), ctx.to(torch.int32)


def run_asm(q, pool, pages, ctx, out):
    k, v, ks, vs = pool
    P = k.shape[0] * k.shape[1]
    qv, ov = q.view(-1, G, D), out.view(-1, G, D)
    for s in range(0, qv.shape[0], ASM_MAX_ROWS):
        e = min(qv.shape[0], s + ASM_MAX_ROWS)
        aiter.pa_fwd_asm(
            qv[s:e],
            k.view(P, 1, 8, 16, 16),
            v.view(P, 1, 1, D, 16),
            pages[s:e],
            ctx[s:e],
            pages.stride(0),
            K_QScale=ks.view(P, 1, 16),
            V_QScale=vs.view(P, 1, 16),
            out_=ov[s:e],
            high_precision=0,
        )


@torch.no_grad()
def reference(q, pool, topk, bt, req, kvl, chunk=128):
    """fp32 softmax(q k^T * scale) v over each row's selected blocks, keys < its kv_len,
    gathered from topk + block_table (not from the ASM tables)."""
    k, v, ks, vs = pool
    hkv = topk.shape[0]
    k8, v8 = k.view(-1, hkv, 8, 16, 16), v.view(-1, hkv, D, 16)
    ks, vs = ks.view(-1, hkv, 16), vs.view(-1, hkv, 16)
    out = torch.empty(q.shape, dtype=torch.float32)
    pg, tok = torch.arange(8), torch.arange(16)
    for h in range(hkv):
        for s in range(0, q.shape[0], chunk):
            e = min(q.shape[0], s + chunk)
            blk = topk[h, s:e].long()
            pages = bt[req[s:e, None].long(), blk.clamp(min=0)].long()[..., None] * 8
            pages = pages + pg
            pos = blk[..., None, None] * BLK + pg[:, None] * 16 + tok
            kk = k8[pages, h].float().permute(0, 1, 2, 4, 3, 5).reshape(e - s, -1, D)
            vv = v8[pages, h].float().permute(0, 1, 2, 4, 3).reshape(e - s, -1, D)
            kk = kk * ks[pages, h].reshape(e - s, -1, 1)
            vv = vv * vs[pages, h].reshape(e - s, -1, 1)
            mask = (blk[..., None, None] < 0) | (pos >= kvl[s:e, None, None, None])
            qh = q[s:e, h * G : (h + 1) * G].float()
            sc = torch.einsum("ngd,nkd->ngk", qh, kk) * SCALE
            sc = sc.masked_fill(mask.reshape(e - s, 1, -1), float("-inf"))
            out[s:e, h * G : (h + 1) * G] = torch.einsum(
                "ngk,nkd->ngd", sc.softmax(-1), vv
            )
    return out


def err(o, ref):
    d = (o.float() - ref).abs()
    worst = ref.flatten()[d.argmax()].abs()  # one bf16 step there: output rounding
    return {
        "rel_l2": (d.norm() / ref.norm()).item(),
        "mean_abs": d.mean().item(),
        "max_abs": d.max().item(),
        "ulp": (2.0 ** (torch.floor(torch.log2(worst)) - 7)).item(),
        "nan": int(torch.isnan(o).sum()),
    }


@benchmark()
def test_tt_sparse_prefill(case, seed, pool):
    reqs, mode, hkv = (*CASES[case], HKV)[:3]
    pool = as_hkv(pool, hkv)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    topk, bt, req, kvl = make_step(reqs, pool[0].shape[0] // 8, mode, gen, hkv)
    t0, n = tile_map(reqs)
    T, max_seq_len = topk.shape[1], max(p + m for p, m in reqs)
    q = torch.randn(T, hkv * G, D, generator=gen).to(dtypes.bf16)

    def tt(out):
        tt_sparse_prefill(
            q, *pool, topk, bt, req, kvl, t0, n, t0.numel(), SCALE, max_seq_len, out
        )

    o_tt, o_tt2, o_asm = (torch.full_like(q, float("nan")) for _ in range(3))
    tt(out=o_tt)
    tt(out=o_tt2)
    pages, ctx = asm_tables(topk, bt, req, kvl)
    run_asm(q, pool, pages, ctx, o_asm)
    ref = reference(q, pool, topk, bt, req, kvl)
    e_tt, e_asm = err(o_tt, ref), err(o_asm, ref)
    # ASM's own max-abs moves with the order it adds the blocks in: its envelope
    envelope = e_asm["max_abs"]
    for _ in range(3):
        o_perm = torch.full_like(q, float("nan"))
        run_asm(q, pool, *asm_tables(topk, bt, req, kvl, gen), o_perm)
        envelope = max(envelope, err(o_perm, ref)["max_abs"])

    assert torch.equal(o_tt, o_tt2), f"{case}: not deterministic"
    assert e_tt["nan"] == 0 and e_asm["nan"] == 0, f"{case}: NaN"
    for key in ("rel_l2", "mean_abs"):
        assert e_tt[key] <= e_asm[key], f"{case}: {key} {e_tt} vs ASM {e_asm}"
    bound = envelope + e_tt["ulp"]
    assert e_tt["max_abs"] <= bound, f"{case}: max_abs {e_tt} vs ASM {envelope}"

    _, us_tt = run_perftest(tt, out=o_tt)
    _, us_asm = run_perftest(run_asm, q, pool, pages, ctx, o_asm)
    return {
        "rows": T,
        "ctx": max_seq_len,
        "hkv": hkv,
        **{f"tt {k}": e_tt[k] for k in ("rel_l2", "mean_abs", "max_abs")},
        **{f"asm {k}": e_asm[k] for k in ("rel_l2", "mean_abs", "max_abs")},
        "asm max_abs envelope": envelope,
        "tt ulp at max": e_tt["ulp"],
        "tt us": us_tt,
        "asm us": us_asm,
        "speedup": us_asm / us_tt,
    }


def test_refuses_past_the_bitmap(pool):
    """Past TT_MAX_SEQ_LEN tokens block ids no longer fit the per-tile bitmap: refused."""
    reqs = [(599040 - 1024, 1024)]
    gen = torch.Generator(device="cuda").manual_seed(0)
    topk, bt, req, kvl = make_step(reqs, pool[0].shape[0] // 8, "random", gen)
    t0, n = tile_map(reqs)
    out = torch.full((1024, HKV * G, D), float("nan"), dtype=dtypes.bf16)
    q = torch.randn(out.shape).to(dtypes.bf16)
    try:
        tt_sparse_prefill(
            q, *pool, topk, bt, req, kvl, t0, n, t0.numel(), SCALE, 599040, out=out
        )
    except ValueError:
        assert torch.isnan(out).all(), "a refused call must not launch"
        return
    raise AssertionError("tt_sparse_prefill accepted max_seq_len 599040")


def main():
    if not is_supported():
        aiter.logger.warning("tt_sparse_prefill: gfx950 only, skipping")
        return
    parser = argparse.ArgumentParser(description="tt_sparse_prefill vs GQA-16 ASM")
    parser.add_argument("-c", "--case", nargs="*", default=list(CASES), choices=CASES)
    parser.add_argument("-s", "--seed", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()
    torch.manual_seed(0)
    pool = make_pool(4800)  # past the 4680 blocks of a 599,040-token request
    test_refuses_past_the_bitmap(pool)
    df = pd.DataFrame(
        [test_tt_sparse_prefill(c, s, pool) for c in args.case for s in args.seed]
    )
    df = df.drop(columns=["pool"])
    aiter.logger.info("tt_sparse_prefill summary (markdown):\n%s", df.to_markdown())


if __name__ == "__main__":
    main()
