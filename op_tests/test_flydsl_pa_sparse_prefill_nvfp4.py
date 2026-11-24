# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 sparse-prefill attention with K/V reuse (gfx1250) against ``pa_decode``
and an FP32 reference.

MiniMax-M3.1 TP4 prefill: requests of T new tokens after C cached ones; every
token attends its 16 selected 128-token blocks (its own block forced, then 15
by indexer score), through the compacted page table the selector emits (full
pages in selection order, the partial tail page last). Today's path runs
``pa_decode``'s one-partition NVFP4 wave tile per token; the reuse kernel
(``aiter.ops.flydsl.pa_sparse_prefill``) walks the union of a tile of tokens'
pages. Both are checked against an FP32 attention over the dequantized pages,
and the reuse kernel's error must not exceed today's by more than rounding.

Picks: "shared" / "swapNN" (the indexer's pattern used by Inferact/amd-kernels'
tests: a per-request base of 15 earlier blocks, each slot swapped for a random
earlier block with probability 0.1 / NN% per token), "random" (15 random earlier
blocks, the union's worst case).

Run:  python -m pytest -q op_tests/test_flydsl_pa_sparse_prefill_nvfp4.py
Bench: python op_tests/test_flydsl_pa_sparse_prefill_nvfp4.py --tokens 14336 --prefix 99968
"""

import argparse
import importlib
import importlib.util
import statistics

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

PAGE = 128
D = 128
GROUP = 16
TOPK = 16
SCALE = D**-0.5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("NVFP4 sparse prefill runs on gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    return torch.device("cuda")


def nvfp4_quantize(x):
    """As ATOM's NVFP4 store: scale = e4m3(amax / 6), codes RNE, saturated at 6."""
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    scale = (blocks.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    inv = torch.where(scale_f > 0, 1.0 / scale_f, torch.zeros_like(scale_f))
    y = blocks * inv[..., None]
    a = y.abs()
    mag = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    codes = (mag | torch.where(y < 0, 8, 0)).reshape(*x.shape[:-1], -1, 2)
    payload = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return payload, scale


def nvfp4_dequantize(payload, scales):
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scales.float()[..., None]).flatten(-2)


def _swap_probability(kind):
    """``shared`` = ``swap10``; ``swapNN``: each slot of the request's base set is
    replaced by a random earlier block with probability NN% per token."""
    if kind == "shared":
        return 0.1
    if kind.startswith("swap"):
        return int(kind[4:]) / 100.0
    if kind == "random":
        return 1.0
    raise ValueError(f"unknown picks {kind!r}")


def _picks(pos, first_blk, gen, kind):
    """[n, 16] logical blocks in selection order: own block first (the forced
    local block), then up to 15 earlier ones; -1 padded."""
    n = pos.numel()
    cur = (pos // PAGE).long()
    swap = _swap_probability(kind)
    if kind == "random":
        width = max(int(cur.max()) + 1, TOPK)
        keys = torch.rand(n, width, generator=gen)
        keys = keys.masked_fill(torch.arange(width) >= cur[:, None], 2.0)
        cand = keys.argsort(-1)[:, : TOPK - 1]
        keep = cand < cur[:, None]
    else:
        base = torch.randperm(max(first_blk, TOPK), generator=gen)[: TOPK - 1]
        cand = base[None].expand(n, TOPK - 1).clone()
        swapped = torch.rand(n, TOPK - 1, generator=gen) < swap
        alt = (torch.rand(n, TOPK - 1, generator=gen) * cur.clamp(min=1)[:, None]).long()
        cand = torch.where(swapped, alt, cand)
        keep = cand < cur[:, None]
        same = cand[:, :, None] == cand[:, None, :]
        earlier = torch.ones(TOPK - 1, TOPK - 1, dtype=torch.bool).tril(-1)
        keep &= ~(same & earlier & keep[:, None, :]).any(-1)
    order = torch.sort((~keep).int(), dim=-1, stable=True).indices
    cand = cand.gather(-1, order)
    kept = keep.gather(-1, order)
    picks = torch.where(kept, cand, torch.full_like(cand, -1))
    return torch.cat([cur[:, None], picks], -1)


def load_dumped_picks(path, kv_head=0):
    """([(cached, new)], [logical picks [new, 16]]) of an ATOM top-k dump
    (``ATOM_M3_DUMP_PREFILL_TOPK``: ``chunk<NNN>_layer<LL>.pt``)."""
    d = torch.load(path)
    topk = d["topk_idx"][kv_head].long()
    cu = d["cu_seqlens_q"].long()
    prefix = d["prefix_lens"].long()
    requests, picks = [], []
    for b in range(cu.numel() - 1):
        new = int(cu[b + 1] - cu[b])
        requests.append((int(prefix[b]), new))
        picks.append(topk[cu[b] : cu[b + 1]])
    return requests, picks


def make_case(requests, device, picks="shared", seed=0, copies=1):
    """``requests``: [(cached, new)] tokens. Returns the flattened rows' query,
    ``copies`` NVFP4 caches (pages shuffled, tails poisoned) and the compacted
    table / lengths ATOM's selector emits. ``picks``: a synthetic pattern name,
    or a list (one a request) of dumped logical picks ``[new, 16]``."""
    gen = torch.Generator().manual_seed(seed)
    ggen = torch.Generator(device=device).manual_seed(seed + 1)
    blocks = [(c + n + PAGE - 1) // PAGE for c, n in requests]
    num_pages = sum(blocks) + 1
    perm = torch.randperm(num_pages, generator=gen)
    tables, rows_pick, rows_pos, start = [], [], [], 0
    for r, ((cached, new), nb) in enumerate(zip(requests, blocks)):
        bt = perm[start : start + nb]
        start += nb
        pos = torch.arange(cached, cached + new)
        if isinstance(picks, str):
            logical = _picks(pos, cached // PAGE, gen, picks)
        else:
            logical = picks[r].cpu().long()
        tables.append(bt)
        rows_pick.append((bt, logical, pos))
    # Compacted rows: full pages in selection order, the own (tail) block last.
    table_rows, ctx_rows = [], []
    for bt, logical, pos in rows_pick:
        cur = pos // PAGE
        valid = logical >= 0
        full = valid & (logical != cur[:, None])
        slot = torch.where(full, full.int().cumsum(-1) - 1, full.int().sum(-1, keepdim=True))
        slot = torch.where(valid, slot, TOPK)  # unused
        page = torch.where(valid, bt[logical.clamp(min=0)], 0)
        out = torch.zeros(pos.numel(), TOPK + 1, dtype=torch.long)
        out.scatter_(1, slot, page)
        table_rows.append(out[:, :TOPK])
        ctx_rows.append(full.sum(-1) * PAGE + (pos - cur * PAGE + 1))
    table = torch.cat(table_rows).to(torch.int32).to(device)
    ctx = torch.cat(ctx_rows).to(torch.int32).to(device)
    rows = table.shape[0]
    q = (torch.randn(rows, GROUP, D, device=device, generator=ggen) * 2).to(torch.bfloat16)
    caches = []
    for c in range(copies):
        k = torch.randn(num_pages, 1, PAGE, D, device=device, generator=ggen)
        v = torch.randn(num_pages, 1, PAGE, D, device=device, generator=ggen)
        k = k * torch.exp2(torch.randint(-6, 4, (num_pages, 1, PAGE, 1), device=device, generator=ggen).float())
        kp, ks = nvfp4_quantize(k)
        vp, vs = nvfp4_quantize(v)
        # Never-written slots past each request's end: NaN scales, random codes.
        start = 0
        for (cached, new), nb in zip(requests, blocks):
            tail = (cached + new) % PAGE
            if tail:
                last = int(perm[start + nb - 1])
                for t in (kp, vp):
                    t[last, :, tail:] = torch.randint_like(t[last, :, tail:], 0, 256)
                for t in (ks, vs):
                    t[last, :, tail:] = torch.tensor(float("nan")).to(torch.float8_e4m3fn)
            start += nb
        caches.append((kp, ks, vp, vs))
    return dict(q=q, caches=caches, table=table, ctx=ctx, requests=requests, num_pages=num_pages)


def union_stats(table, ctx, tq):
    """Mean distinct pages a tile of ``tq`` consecutive rows selects."""
    rows = table.shape[0]
    n_pages = (ctx.long() + PAGE - 1) // PAGE
    valid = torch.arange(TOPK, device=table.device)[None] < n_pages[:, None]
    keyed = torch.where(valid, table.long(), -1)
    pad = (-rows) % tq
    keyed = torch.cat([keyed, torch.full((pad, TOPK), -1, device=table.device)])
    tiles = keyed.view(-1, tq * TOPK).sort(-1).values
    distinct = (tiles[:, 1:] != tiles[:, :-1]) & (tiles[:, 1:] >= 0)
    count = distinct.sum(-1) + (tiles[:, 0] >= 0).long()
    return count.float().mean().item()


def reference(case, cache, rows=None, chunk=256):
    """FP32 attention of each row over its dequantized compacted pages."""
    kp, ks, vp, vs = cache
    kd = nvfp4_dequantize(kp[:, 0], ks[:, 0])  # [P, 128, D]
    vd = nvfp4_dequantize(vp[:, 0], vs[:, 0])
    kd = torch.nan_to_num(kd)  # poisoned slots are never visible
    vd = torch.nan_to_num(vd)
    table, ctx, q = case["table"], case["ctx"], case["q"]
    rows = torch.arange(table.shape[0], device=q.device) if rows is None else rows
    out = torch.zeros(rows.numel(), GROUP, D, device=q.device)
    tok = torch.arange(TOPK * PAGE, device=q.device)
    for s in range(0, rows.numel(), chunk):
        r = rows[s : s + chunk]
        pages = table[r].long()
        k = kd[pages].reshape(r.numel(), TOPK * PAGE, D)
        v = vd[pages].reshape(r.numel(), TOPK * PAGE, D)
        visible = tok[None] < ctx[r, None]
        sc = torch.einsum("rhd,rtd->rht", q[r].float(), k) * SCALE
        sc = sc.masked_fill(~visible[:, None], float("-inf"))
        p = torch.softmax(sc, -1).nan_to_num()
        out[s : s + chunk] = torch.einsum("rht,rtd->rhd", p, v)
    return out


def run_current(case, cache, out):
    """Today's path: pa_decode's one-partition NVFP4 wave tile, one row a CTA."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    kp, ks, vp, vs = cache
    plan = case.get("_plan")
    if plan is None:
        rows = case["ctx"].shape[0]
        # ATOM's prefill plan: batches past aiter's default bound get their own
        # (``_flydsl_prepare_explicit_plan``).
        extra = {"max_batch_size": rows} if rows > 4096 else {}
        plan = pa.plan_pa_decode(case["ctx"], 1, max_partitions=1, **extra)
        case["_plan"] = plan

    def call():
        pa.pa_decode(
            out,
            case["q"],
            kp,
            vp,
            case["ctx"],
            case["table"],
            SCALE,
            1,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            max_context_length=TOPK * PAGE,
        )

    return call


def run_new(case, cache, out, tile_rows, **variant):
    sp = importlib.import_module("aiter.ops.flydsl.pa_sparse_prefill")
    kp, ks, vp, vs = cache

    def call():
        sp.pa_sparse_prefill_nvfp4(
            out, case["q"], kp, vp, ks, vs, case["table"], case["ctx"], SCALE,
            tile_rows=tile_rows, **variant,
        )

    return call


def parse_variant(spec):
    """``tq[:q_regs[:row_group[:ring]]][+sr][+fe]`` (P2's kernel) or
    ``v2:tq[:rebase_log2[:waves_per_eu[:ring]]][+fK][+vh|+nvh]`` (pair-batched)
    -> (tile_rows, kwargs)."""
    flags = spec.split("+")
    spec = flags.pop(0)
    if spec.startswith("v2:"):
        parts = [int(x) for x in spec[3:].split(":")]
        v2 = {}
        if len(parts) > 1:
            v2["rebase_log2"] = parts[1]
        if len(parts) > 2:
            v2["waves_per_eu"] = parts[2]
        kw = {"impl": "v2", "v2": v2}
        if len(parts) > 3:
            kw["nvfp4_buffers"] = parts[3]
        for f in flags:
            if f.startswith("f") and f[1:].isdigit():
                v2["fence"] = int(f[1:])
            elif f in ("vh", "nvh"):
                v2["v_hoist"] = f == "vh"
            elif f.startswith("vv") and f[2:].isdigit():
                v2["v_valu"] = int(f[2:])
            elif f.startswith("dbg") and f[3:].isdigit():
                v2["dbg"] = int(f[3:])
            elif f == "cd":
                v2["cta_dma"] = True
            elif f == "ad":
                v2["async_dma"] = True
            elif f == "qr":
                v2["q_regs"] = True
            elif f == "vw":
                v2["v_wide"] = True
            elif f == "all":
                v2["all_rows"] = True
        return parts[0], kw
    parts = [int(x) for x in spec.split(":")]
    kw = {"skip_rescale": "sr" in flags, "fma_exp": "fe" in flags}
    if len(parts) > 1:
        kw["q_regs"] = bool(parts[1])
    if len(parts) > 2:
        kw["row_group"] = parts[2]
    if len(parts) > 3:
        kw["nvfp4_buffers"] = parts[3]
    return parts[0], kw


def errors(out, ref):
    o = out.float()
    diff = (o - ref).abs()
    return dict(
        max_abs=diff.max().item(),
        rel_l2=(torch.linalg.vector_norm(o - ref) / torch.linalg.vector_norm(ref)).item(),
        finite=bool(torch.isfinite(o).all()),
    )


def check(case, tile_rows, sample=None, **variant):
    cache = case["caches"][0]
    rows = case["q"].shape[0]
    cur = torch.full_like(case["q"], float("nan"))
    new = torch.full_like(case["q"], float("nan"))
    run_current(case, cache, cur)()
    run_new(case, cache, new, tile_rows, **variant)()
    torch.cuda.synchronize()
    idx = None
    if sample and rows > sample:
        g = torch.Generator().manual_seed(7)
        idx = torch.randperm(rows, generator=g)[:sample].sort().values.to(case["q"].device)
        last = torch.arange(max(0, rows - 8), rows, device=idx.device)
        idx = torch.unique(torch.cat([idx, last, torch.arange(min(8, rows), device=idx.device)]))
    ref = reference(case, cache, idx)
    pick = (lambda t: t) if idx is None else (lambda t: t[idx])
    e_cur = errors(pick(cur), ref)
    e_new = errors(pick(new), ref)
    same = (cur.view(torch.int16) == new.view(torch.int16)).float().mean().item()
    d = (new.float() - cur.float()).abs().max().item()
    return e_cur, e_new, same, d


CASES = {
    # (requests [(cached, new)], picks)
    "one-tile": ([(4096, 4)], "shared"),
    "short-prefix": ([(0, 300)], "shared"),
    "ragged-two-requests": ([(1000, 77), (5000, 259)], "shared"),
    "random-picks": ([(20000, 512)], "random"),
    "shared-picks": ([(30000, 1024)], "shared"),
    "unaligned-prefix": ([(12345, 333)], "random"),
}


VARIANTS = ["2+sr+fe", "3+sr+fe", "4", "4:0", "4+sr+fe", "4:1:4+fe", "6+sr+fe", "8+sr+fe",
            "v2:2", "v2:2:8", "v2:4", "v2:2:0:0:2", "v2:4:0:0:3+nvh", "v2:4+all", "v2:2+all", "v2:2+vv8"]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("case", list(CASES))
def test_pa_sparse_prefill_nvfp4(case, variant, device):
    requests, picks = CASES[case]
    c = make_case(requests, device, picks)
    tile_rows, kw = parse_variant(variant)
    e_cur, e_new, same, d = check(c, tile_rows, **kw)
    assert e_new["finite"], "non-finite output"
    # The reuse kernel accumulates each row's pages in another order: rounding.
    assert e_new["max_abs"] <= max(2 * e_cur["max_abs"], 2e-2), (e_cur, e_new)
    assert e_new["rel_l2"] <= 1.5 * e_cur["rel_l2"] + 1e-4, (e_cur, e_new)


@pytest.mark.parametrize("variant", ["4", "v2:2", "v2:4"])
def test_pa_sparse_prefill_nvfp4_padding_rows(variant, device):
    """Rows of length 0 (a padded token bucket) store zeros."""
    c = make_case([(2048, 37)], device)
    c["ctx"][30:] = 0
    out = torch.full_like(c["q"], float("nan"))
    tile_rows, kw = parse_variant(variant)
    run_new(c, c["caches"][0], out, tile_rows, **kw)()
    torch.cuda.synchronize()
    assert (out[30:] == 0).all()
    ref = reference(c, c["caches"][0], torch.arange(30, device=device))
    assert errors(out[:30], ref)["max_abs"] < 2e-2


def mxfp8_reference(x):
    """``op_mxfp8``'s group-32 quant of BF16 rows ``[rows, 16, 128]``: (E4M3 codes
    [rows, 2048] uint8, E8M0 scales [rows, 64] uint8)."""
    g = x.float().reshape(x.shape[0], -1, 32)
    amax = g.abs().amax(-1).clamp(min=1e-10)
    v = amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)
    m, e = torch.frexp(v)
    exp = torch.where(m == 0.5, e - 1, e)  # ceil to a power of two: 2**exp
    code = (exp + 127).clamp(0, 254)
    scaled = g * torch.exp2((127 - code).float())[..., None]
    q = scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(x.shape[0], -1), code.to(torch.uint8)


@pytest.mark.parametrize("kv_heads", [1, 4])
@pytest.mark.parametrize("variant", ["4+sr+fe", "3+sr+fe", "v2:2", "v2:4"])
def test_pa_sparse_prefill_nvfp4_mxfp8(variant, kv_heads, device):
    """The MXFP8 output is the group-32 quant of the kernel's own BF16 rows, bit
    for bit; padded bucket rows are FP8 zeros with unit scales. With ``kv_heads``
    KV heads the flattened rows are (token, KV head) pairs and the padding is
    whole tokens (two here)."""
    c = make_case([(3000, 203 if kv_heads == 1 else 204)], device, "shared")
    tile_rows, kw = parse_variant(variant)
    kw["kv_heads"] = kv_heads
    sp = importlib.import_module("aiter.ops.flydsl.pa_sparse_prefill")
    kp, ks, vp, vs = c["caches"][0]
    rows, pad = c["q"].shape[0], 5 if kv_heads == 1 else 2 * kv_heads
    out = torch.empty_like(c["q"])
    sp.pa_sparse_prefill_nvfp4(out, c["q"], kp, vp, ks, vs, c["table"], c["ctx"], SCALE,
                               tile_rows=tile_rows, **kw)
    codes = torch.full((rows + pad, GROUP, D), 0x55, dtype=torch.uint8, device=device)
    scales = torch.full((rows + pad, GROUP * D // 32), 0x33, dtype=torch.uint8, device=device)
    sp.pa_sparse_prefill_nvfp4(torch.empty_like(c["q"]), c["q"], kp, vp, ks, vs, c["table"], c["ctx"],
                               SCALE, tile_rows=tile_rows, quantized_output=codes,
                               output_scale=scales, quantized_pad_rows=pad, **kw)
    torch.cuda.synchronize()
    ref_codes, ref_scales = mxfp8_reference(out)
    assert torch.equal(codes[:rows].reshape(rows, -1), ref_codes)
    assert torch.equal(scales[:rows], ref_scales)
    assert (codes[rows:] == 0).all() and (scales[rows:] == 127).all()


def time_graph(calls, replays=10, rounds=5):
    """Median over rounds of the mean us per call of one CUDA graph of ``calls``."""
    for f in calls:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1e3 / (replays * len(calls)))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=14336)
    ap.add_argument("--prefix", type=int, default=99968)
    ap.add_argument("--picks", nargs="+", default=["shared", "swap50", "random"],
                    help="synthetic patterns, or ATOM top-k dump files (*.pt)")
    ap.add_argument("--variants", nargs="+", default=["4", "8"],
                    help="tq[:q_regs[:row_group[:ring]]] (see compile_pa_sparse_prefill_nvfp4)")
    ap.add_argument("--copies", type=int, default=4, help="distinct caches/queries rotated per graph")
    ap.add_argument("--check-rows", type=int, default=2048, help="rows checked against FP32 (0 = all)")
    ap.add_argument("--no-current", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one CTA, no graph: first-launch safety run")
    args = ap.parse_args()
    dev = torch.device("cuda")
    variants = [(v, *parse_variant(v)) for v in args.variants]
    if args.smoke:
        # One workgroup a launch: tile_rows rows (BF16 output, then the MXFP8 one).
        sp = importlib.import_module("aiter.ops.flydsl.pa_sparse_prefill")
        for name, tq, kw in variants:
            c = make_case([(args.prefix, tq)], dev, args.picks[0])
            out = torch.full_like(c["q"], float("nan"))
            run_new(c, c["caches"][0], out, tq, **kw)()
            torch.cuda.synchronize()
            ref = reference(c, c["caches"][0])
            print(f"smoke {name} rows={c['q'].shape[0]} (1 workgroup): {errors(out, ref)}", flush=True)
            kp, ks, vp, vs = c["caches"][0]
            codes = torch.zeros((tq, GROUP, D), dtype=torch.uint8, device=dev)
            scales = torch.zeros((tq, GROUP * D // 32), dtype=torch.uint8, device=dev)
            sp.pa_sparse_prefill_nvfp4(torch.empty_like(c["q"]), c["q"], kp, vp, ks, vs, c["table"], c["ctx"],
                                       SCALE, tile_rows=tq, quantized_output=codes, output_scale=scales, **kw)
            torch.cuda.synchronize()
            rc, rs = mxfp8_reference(out)
            print(f"smoke {name} MXFP8 (1 workgroup): bitwise {torch.equal(codes.view(tq, -1), rc) and torch.equal(scales, rs)}",
                  flush=True)
        return
    for picks in args.picks:
        if picks.endswith(".pt"):
            requests, dumped = load_dumped_picks(picks)
            c = make_case(requests, dev, dumped, copies=args.copies)
        else:
            c = make_case([(args.prefix, args.tokens)], dev, picks, copies=args.copies)
        rows = c["q"].shape[0]
        tqs = sorted({tq for _, tq, _ in variants})
        unions = {tq: union_stats(c["table"], c["ctx"], tq) for tq in tqs}
        print(f"== T={args.tokens} C={args.prefix} picks={picks}: rows {rows}, "
              + ", ".join(f"tq={tq}: {u:.1f} distinct pages a tile (reuse {16 * tq / u:.2f}x)"
                          for tq, u in unions.items()),
              flush=True)
        for name, tq, kw in variants:
            e_cur, e_new, same, d = check(c, tq, args.check_rows or None, **kw)
            print(f"  check {name}: cur vs fp32 max {e_cur['max_abs']:.3e} relL2 {e_cur['rel_l2']:.3e} | "
                  f"new vs fp32 max {e_new['max_abs']:.3e} relL2 {e_new['rel_l2']:.3e} finite {e_new['finite']} | "
                  f"new vs cur: max {d:.3e}, bitwise-equal {100 * same:.1f}%", flush=True)
        qs = [c["q"]] + [torch.randn_like(c["q"]) for _ in range(args.copies - 1)]
        outs = [torch.empty_like(c["q"]) for _ in range(args.copies)]

        def calls(make, *extra, **kw):
            out = []
            for i in range(args.copies):
                ci = dict(c, q=qs[i])
                out.append(make(ci, c["caches"][i], outs[i], *extra, **kw))
            return out

        if not args.no_current:
            print(f"  current pa_decode (1 CTA a token): {time_graph(calls(run_current)):8.1f} us", flush=True)
        for name, tq, kw in variants:
            print(f"  reuse kernel {name:10s}:             {time_graph(calls(run_new, tq, **kw)):8.1f} us", flush=True)


if __name__ == "__main__":
    main()
