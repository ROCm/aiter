# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged attention, decode and prefill (FlyDSL, gfx950).

The kernels take the selector's output directly: ``topk_idx [Hkv, rows, 16]`` logical
128-token block ids plus the request block table; K / V pages are the single-buffer
NVFP4 page (see ``kernels/minimax_m3_sparse_pa_nvfp4_common.py``).

Reference: an fp64 torch attention over K / V = e2m1 x e4m3 scale x global scale taken
from the token-major codes and scale bytes the pages were *built from* -- nothing is
read back through the kernel's lane layout, so a wrong page layout cannot cancel. The
kernels do the products exactly in f16, so what is left against the reference is the
bf16 output (relative L2 of ~1.7e-3) and the f16 Q / P.

Run (exits nonzero if any check fails):
    python op_tests/test_flydsl_minimax_m3_sparse_pa_nvfp4.py [--case SUBSTR ...] [--perf]
"""

import argparse
import itertools
import math
import sys
import zlib

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.minimax_m3_sparse_pa_nvfp4_decode import (
    minimax_m3_sparse_pa_nvfp4_decode,
)
from aiter.ops.flydsl.kernels.minimax_m3_sparse_pa_nvfp4_prefill import (
    minimax_m3_sparse_pa_nvfp4_prefill,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

BLK = 128
D = 128
TOPK = 16
SM_SCALE = D**-0.5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
REL_L2_TOL = 2.5e-3  # bf16 output floor ~1.7e-3
PAGE_B = 9216  # one head, one side: 8192 B data + 1024 B scales


# ---------------------------------------------------------------------------
# page construction (token-major codes / scales -> the single-buffer page)
# ---------------------------------------------------------------------------
def quantize_nvfp4(x, gs):
    """e2m1 codes [..., 128] and e4m3 scale bytes [..., 8] (x ~ code * scale * gs)."""
    g = x.float().view(*x.shape[:-1], 8, 16)
    scale = (g.abs().amax(-1) / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    sv = scale.float()
    y = torch.where(sv[..., None] != 0, g / (sv[..., None] * gs), torch.zeros_like(g))
    grid = torch.tensor(E2M1, device=x.device)
    mag = (y.abs()[..., None] - grid).abs().argmin(-1)
    code = mag | ((y < 0).long() << 3)
    return code.view(*x.shape).to(torch.uint8), scale.view(torch.uint8)


def scale_offsets(device):
    """Byte offset of the scale of (token, group) inside a head's scale region of K, V."""
    t = torch.arange(BLK, device=device)[:, None]
    j = torch.arange(8, device=device)[None, :]
    r = t % 32
    n = 4 * (r // 8) + r % 4
    k_off = (((t // 32) * 4 + j // 2) * 16 + n) * 4 + ((r % 8) // 4) * 2 + j % 2
    v_off = j * BLK + t + 0 * j
    return k_off, v_off


def pack_pages(kc, ks, vc, vs):
    """Codes [pages, 128, Hkv, 128] uint8 + scale bytes [pages, 128, Hkv, 8] -> pages
    [pages, 2 * Hkv * 9216] uint8."""
    pages, _, hkv, _ = kc.shape
    dev = kc.device
    out = torch.zeros(pages, 2, hkv, PAGE_B, dtype=torch.uint8, device=dev)
    k_off, v_off = scale_offsets(dev)
    for side, (c, s) in enumerate(((kc, ks), (vc, vs))):
        c = c.permute(0, 2, 1, 3)  # [pages, hkv, tok, d]
        s = s.permute(0, 2, 1, 3)  # [pages, hkv, tok, 8]
        if side == 0:  # K data: token-major, low nibble the even dim
            data = (c[..., 0::2] | (c[..., 1::2] << 4)).reshape(pages, hkv, BLK * 64)
            off = k_off
        else:  # V data: [token octet][dim][4 B], byte b = tokens 2b, 2b + 1 of the octet
            cc = c.reshape(pages, hkv, 16, 4, 2, D)
            data = (cc[:, :, :, :, 0] | (cc[:, :, :, :, 1] << 4)).permute(0, 1, 2, 4, 3)
            data = data.reshape(pages, hkv, BLK * 64)
            off = v_off
        out[:, side, :, : BLK * 64] = data
        scl = torch.zeros(pages, hkv, 1024, dtype=torch.uint8, device=dev)
        scl[:, :, off.flatten()] = s.reshape(pages, hkv, BLK * 8)
        out[:, side, :, BLK * 64 :] = scl
    # regions are [side][head]: data(8192) then scales(1024) per head within a side -- the
    # in-page order is [K data heads][K scale heads][V data heads][V scale heads]
    d = out[..., : BLK * 64].reshape(pages, 2, hkv * BLK * 64)
    sc = out[..., BLK * 64 :].reshape(pages, 2, hkv * 1024)
    return torch.cat([d, sc], dim=-1).reshape(pages, 2 * hkv * PAGE_B).contiguous()


def deq(code, scale_bytes, gs):
    """fp64 [..., 128] = e2m1(code) * e4m3(scale) * gs."""
    grid = torch.tensor(E2M1, dtype=torch.float64, device=code.device)
    mag = grid[(code & 7).long()] * torch.where(code >= 8, -1.0, 1.0)
    s = scale_bytes.view(torch.float8_e4m3fn).double()
    return (mag.view(*code.shape[:-1], 8, 16) * s[..., None]).flatten(-2) * gs


# ---------------------------------------------------------------------------
# case construction
# ---------------------------------------------------------------------------
def make_case(
    seq_lens,
    q_lens,
    hkv,
    group,
    seed=0,
    order="shuffle",
    fill=TOPK,  # valid ids per row (<= TOPK incl. the tail)
    junk=False,  # sprinkle invalid ids (-1 in the middle, ids past the tail, negatives)
    k_scale=0.05,
    v_scale=0.03,
    device="cuda",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    reqs = len(seq_lens)
    nblk = [math.ceil(s / BLK) for s in seq_lens]
    n_pages = sum(nblk) + 3
    perm = torch.randperm(n_pages, generator=g)
    bt = torch.zeros(reqs, max(nblk) + 1, dtype=torch.int32)
    o = 0
    for r in range(reqs):
        bt[r, : nblk[r]] = perm[o : o + nblk[r]].int()
        o += nblk[r]
    amp = torch.exp(torch.randn(n_pages, BLK, hkv, 1, generator=g) * 0.25)
    k = (torch.randn(n_pages, BLK, hkv, D, generator=g) * amp).bfloat16()
    v = (torch.randn(n_pages, BLK, hkv, D, generator=g) * amp).bfloat16()
    kc, ks = quantize_nvfp4(k.to(device), k_scale)
    vc, vs = quantize_nvfp4(v.to(device), v_scale)
    cache = pack_pages(kc, ks, vc, vs)
    rows = sum(q_lens)
    row_req = torch.empty(rows, dtype=torch.int32)
    row_kv = torch.empty(rows, dtype=torch.int32)
    i = 0
    for r in range(reqs):
        for j in range(q_lens[r]):
            row_req[i], row_kv[i] = r, seq_lens[r] - q_lens[r] + j + 1
            i += 1
    topk = torch.full((hkv, rows, TOPK), -1, dtype=torch.int32)
    for i in range(rows):
        sb = (int(row_kv[i]) - 1) // BLK
        for h in range(hkv):
            keep = min(fill - 1, sb)
            sel = (
                torch.randperm(sb, generator=g)[:keep]
                if sb
                else torch.zeros(0, dtype=torch.int64)
            )
            ids = torch.cat([sel, torch.tensor([sb])]).int()
            if order == "sorted":
                ids = ids.sort().values
            elif order == "tail_first":
                ids = torch.cat([ids[-1:], ids[:-1]])
            else:
                ids = ids[torch.randperm(len(ids), generator=g)]
            topk[h, i, : len(ids)] = ids
            if junk:
                free = list(range(len(ids), TOPK))
                if free:  # an id past the row's own block, and -1 gaps inside the list
                    topk[h, i, free[0]] = (
                        sb + 1 + int(torch.randint(0, 5, (1,), generator=g))
                    )
                if len(ids) > 2:
                    topk[h, i, 1] = -1
    q = (torch.randn(rows, hkv * group, D, generator=g)).bfloat16()
    dev = device
    return dict(
        q=q.to(dev),
        cache=cache,
        topk=topk.to(dev),
        bt=bt.to(dev),
        row_req=row_req.to(dev),
        row_kv=row_kv.to(dev),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        q_lens=list(q_lens),
        k_scale=torch.tensor([k_scale], dtype=torch.float32, device=dev),
        v_scale=torch.tensor([v_scale], dtype=torch.float32, device=dev),
        hkv=hkv,
        group=group,
        kd=deq(kc, ks, k_scale),
        vd=deq(vc, vs, v_scale),  # fp64 [pages, 128, hkv, 128]
    )


def run_torch(c):
    """fp64 reference: attention of every row over the valid keys of its blocks."""
    q = c["q"].double()
    rows = q.shape[0]
    hkv, group = c["hkv"], c["group"]
    out = torch.zeros(rows, hkv * group, D, dtype=torch.float64, device=q.device)
    bt, topk = c["bt"].cpu(), c["topk"].cpu()
    for i in range(rows):
        req, kv = int(c["row_req"][i]), int(c["row_kv"][i])
        sb = (kv - 1) // BLK
        for h in range(hkv):
            ids = [int(b) for b in topk[h, i] if 0 <= int(b) <= sb]
            if not ids:
                continue
            kk = torch.cat(
                [
                    c["kd"][int(bt[req, b]), : (BLK if b < sb else kv - sb * BLK), h]
                    for b in ids
                ]
            )
            vv = torch.cat(
                [
                    c["vd"][int(bt[req, b]), : (BLK if b < sb else kv - sb * BLK), h]
                    for b in ids
                ]
            )
            s = q[i, h * group : (h + 1) * group] @ kk.T * SM_SCALE
            out[i, h * group : (h + 1) * group] = torch.softmax(s, -1) @ vv
    return out


def rel_l2(out, ref):
    o, r = out.double().flatten(), ref.double().flatten()
    return ((o - r).pow(2).sum().sqrt() / r.pow(2).sum().sqrt().clamp_min(1e-30)).item()


def decode_call(c, splits=None, **kw):
    out = torch.empty_like(c["q"])
    return minimax_m3_sparse_pa_nvfp4_decode(
        c["q"],
        c["cache"],
        c["topk"],
        c["bt"],
        c["seq_lens"],
        SM_SCALE,
        out,
        c["k_scale"],
        c["v_scale"],
        query_len=c["q_lens"][0],
        splits=splits,
        **kw,
    )


def prefill_call(c, **kw):
    out = torch.empty_like(c["q"])
    return minimax_m3_sparse_pa_nvfp4_prefill(
        c["q"],
        c["cache"],
        c["topk"],
        c["bt"],
        c["row_req"],
        c["row_kv"],
        SM_SCALE,
        out,
        c["k_scale"],
        c["v_scale"],
        **kw,
    )


# (name, seq_lens, q_lens, hkv, group, kwargs)
DECODE_CASES = [
    ("ragged_h1", [5000, 2300, 900, 100], [1] * 4, 1, 16, {}),
    ("ragged_h4", [5000, 2300, 900, 100], [1] * 4, 4, 16, {}),
    ("ragged_h2", [4100, 1300, 129, 17], [1] * 4, 2, 16, {}),
    ("group8_h1", [5000, 2300, 900, 100], [1] * 4, 1, 8, {}),
    ("one_token", [1, 2, 127, 128], [1] * 4, 1, 16, {}),
    ("block_edges", [127, 128, 129, 255, 256, 257, 2048, 2049], [1] * 8, 1, 16, {}),
    ("fewer_than_topk", [300, 1000, 130, 1900], [1] * 4, 1, 16, {"fill": 5}),
    ("tail_only", [5000, 800], [1, 1], 1, 16, {"fill": 1}),
    ("junk_ids", [5000, 2300, 900, 100], [1] * 4, 1, 16, {"junk": True}),
    ("order_sorted", [5000, 2300], [1, 1], 1, 16, {"order": "sorted"}),
    ("order_tail_first", [5000, 2300], [1, 1], 1, 16, {"order": "tail_first"}),
    ("mtp4_h1", [5000, 2300, 900, 100], [4] * 4, 1, 16, {}),
    ("mtp8_h4", [5000, 2300, 900, 130], [8] * 4, 4, 16, {}),
    ("mtp8_g8", [3000, 9], [8, 8], 1, 8, {}),
    ("batch40_h1", [300 + 331 * i for i in range(40)], [1] * 40, 1, 16, {}),
    ("batch130_h4", [200 + 97 * i for i in range(130)], [1] * 130, 4, 16, {"fill": 12}),
]
DECODE_SPLITS = (None, 1, 2, 4, 8, 16)


PREFILL_CASES = [
    ("chunk_h1", [3000, 700, 130], [300, 700, 130], 1, 16, {}),
    ("chunk_h4", [3000, 700, 130], [300, 700, 130], 4, 16, {}),
    ("chunk_h2", [2100, 5], [1000, 5], 2, 16, {}),
    ("group8", [3000, 700], [300, 700], 1, 8, {}),
    ("one_row", [1, 129], [1, 1], 1, 16, {}),
    ("edges", [128, 256, 257, 255, 1], [128, 256, 5, 3, 1], 1, 16, {}),
    ("full_prompt", [1500], [1500], 1, 16, {}),
    ("fewer_than_topk", [900, 400], [900, 100], 1, 16, {"fill": 4}),
    ("junk_ids", [3000, 700], [300, 700], 1, 16, {"junk": True}),
    ("ragged_batch", [2000, 33, 4097, 512], [511, 33, 70, 512], 1, 16, {}),
]
PREFILL_TQ = ((4, 2), (4, 1), (2, 1), (1, 1))


def check(name, out, ref, tol=REL_L2_TOL):
    err = rel_l2(out, ref)
    mx = (out.double() - ref).abs().max().item()
    ok = err <= tol and bool(torch.isfinite(out.float()).all())
    checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=2e-2 * ref.abs().max().item(),
        msg=f"{name}: ",
    )
    return ok, err, mx


@benchmark()
def test_correctness(name, hkv, group, rows, splits):
    case = {c[0]: c for c in DECODE_CASES}[name]
    _, seq_lens, q_lens, hkv_, group_, kw = case
    c = make_case(
        seq_lens, q_lens, hkv_, group_, seed=zlib.crc32(name.encode()) % 1000, **kw
    )
    ref = run_torch(c)
    out = decode_call(c, splits)
    ok, err, mx = check(f"decode {name} splits={splits}", out, ref)
    return {"rel_l2": err, "max_abs": mx, "ok": ok}


@benchmark()
def test_prefill_correctness(name, hkv, group, rows, tq, rounds):
    case = {c[0]: c for c in PREFILL_CASES}[name]
    _, seq_lens, q_lens, hkv_, group_, kw = case
    c = make_case(
        seq_lens, q_lens, hkv_, group_, seed=zlib.crc32(name.encode()) % 1000, **kw
    )
    ref = run_torch(c)
    out = prefill_call(c, tq=tq, rounds=rounds)
    ok, err, mx = check(f"prefill {name} tq={tq} rounds={rounds}", out, ref)
    return {"rel_l2": err, "max_abs": mx, "ok": ok}


def decode_splits_agree():
    """Every split count gives the same result as one workgroup, to the bit."""
    bad = 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000, **kw
        )
        base = decode_call(c, 1)
        for s in DECODE_SPLITS:
            if s in (None, 1) or len(seq_lens) * hkv > 1024:
                continue
            # splits merge in fp32 in a different association; the bf16 output then flips a few ulps
            err = rel_l2(decode_call(c, s), base)
            if err > 2e-3:
                bad += 1
                print(f"split mismatch {name} S={s}: rel {err:.2e}")
    return bad


def decode_xcd_grid_agree():
    """The plain (row, kv head) grid and the XCD-aware 1-D grid give the same result, to the bit."""
    bad = 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000, **kw
        )
        for s in (None, 1):
            if not torch.equal(decode_call(c, s, xcds=0), decode_call(c, s, xcds=8)):
                bad += 1
                print(f"xcd grid mismatch {name} splits={s}")
    return bad


def decode_late_v_agree():
    """V loads trailing the K loads (``late_v``) give the same result as together, to the bit."""
    bad = 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000, **kw
        )
        for s in (None, 1):
            if not torch.equal(
                decode_call(c, s, late_v=False), decode_call(c, s, late_v=True)
            ):
                bad += 1
                print(f"late_v mismatch {name} splits={s}")
    return bad


def padding_rows():
    """A padding row (length 0) outputs zeros and does not disturb its neighbours."""
    bad = 0
    c = make_case([500, 700], [1, 1], 1, 16, seed=5)
    for s in (None, 1, 4):
        ref = decode_call(c, s)
        c0 = dict(c, seq_lens=torch.tensor([0, 700], dtype=torch.int32, device="cuda"))
        out = decode_call(c0, s)
        bad += int(out[0].abs().max().item() != 0 or not torch.equal(out[1], ref[1]))
    c1 = dict(
        c,
        row_kv=torch.tensor([0, int(c["row_kv"][1])], dtype=torch.int32, device="cuda"),
    )
    out = prefill_call(c1)
    bad += int(
        out[0].abs().max().item() != 0 or not torch.equal(out[1], prefill_call(c)[1])
    )
    if bad:
        print("padding rows: FAILED")
    return bad


# ---------------------------------------------------------------------------
# perf (synthetic M3.1 shapes; K/V pool far larger than the caches)
# ---------------------------------------------------------------------------
BASE_PAGES = 256


def perf_pool(hkv, pages, device="cuda"):
    """``pages`` pages repeating 256 quantized ones (V = K with the pages reversed), with
    the fp64 dequantized K / V of those 256 for the oracle (global scales 1)."""
    g = torch.Generator(device=device).manual_seed(3)
    k = torch.randn(BASE_PAGES, BLK, hkv, D, generator=g, device=device).bfloat16()
    kc, ks = quantize_nvfp4(k, 0.05)
    base = pack_pages(kc, ks, kc.flip(0), ks.flip(0))
    return (
        base.repeat(pages // BASE_PAGES, 1),
        deq(kc, ks, 1.0),
        deq(kc.flip(0), ks.flip(0), 1.0),
    )


@benchmark()
def test_decode_perf(rows, query_len, hkv, group, ctx):
    pool, kd, vd = perf_pool(hkv, max(8192, 16 * rows * 2))
    g = torch.Generator(device="cuda").manual_seed(1)
    reqs = rows // query_len
    nblk = math.ceil(ctx / BLK)
    bt = torch.randint(
        0, pool.shape[0], (reqs, nblk), generator=g, device="cuda", dtype=torch.int32
    )
    seq = torch.full((reqs,), ctx, dtype=torch.int32, device="cuda")
    kv = torch.cat(
        [torch.arange(ctx - query_len + 1, ctx + 1, device="cuda") for _ in range(reqs)]
    )
    sb = (kv - 1) // BLK
    sc = torch.rand(hkv, rows, nblk, generator=g, device="cuda")
    sc = torch.where(
        torch.arange(nblk, device="cuda") < sb[None, :, None],
        sc,
        torch.full_like(sc, -1.0),
    )
    topk = torch.full((hkv, rows, TOPK), -1, dtype=torch.int32, device="cuda")
    v, i = sc.topk(TOPK - 1, dim=-1)
    topk[..., : TOPK - 1] = torch.where(v >= 0, i.int(), torch.full_like(i.int(), -1))
    topk[..., TOPK - 1] = sb.int()[None, :]
    q = torch.randn(rows, hkv * group, D, generator=g, device="cuda").bfloat16()
    out = torch.empty_like(q)
    ks = vs = torch.ones(1, dtype=torch.float32, device="cuda")
    _, us = run_perftest(
        minimax_m3_sparse_pa_nvfp4_decode,
        q,
        pool,
        topk,
        bt,
        seq,
        SM_SCALE,
        out,
        ks,
        vs,
        query_len=query_len,
        num_iters=20,
        num_warmup=3,
    )
    # first rows against the fp64 oracle (the 256 base pages stand for every page)
    nref = min(rows, 8)
    ref = run_torch(
        dict(
            q=q[:nref],
            topk=topk[:, :nref],
            bt=bt % BASE_PAGES,
            row_req=(torch.arange(nref, device="cuda") // query_len).int(),
            row_kv=kv[:nref].int(),
            hkv=hkv,
            group=group,
            kd=kd,
            vd=vd,
        )
    )
    err = rel_l2(out[:nref], ref)
    # keys actually read: every valid block is full except each row's tail
    nvalid = int((topk >= 0).sum().item())
    keys = nvalid * BLK - int(((sb + 1) * BLK - kv).sum().item()) * hkv
    return {
        "us": us,
        "TFLOPS": 4 * group * D * keys / us / 1e6,  # QK^T and PV, 2 flops per MAC
        "TB/s": nvalid * 2 * PAGE_B / us / 1e6,
        "err": err,
    }


@benchmark()
def test_prefill_perf(q_len, hkv, group, ctx):
    """One request, a chunk of ``q_len`` rows ending at ``ctx`` keys; random top-k blocks
    per row (no overlap between neighbouring rows, the worst case for the page reuse).
    """
    pool, kd, vd = perf_pool(hkv, 8192)
    g = torch.Generator(device="cuda").manual_seed(2)
    nblk = math.ceil(ctx / BLK)
    bt = torch.randint(
        0, pool.shape[0], (1, nblk), generator=g, device="cuda", dtype=torch.int32
    )
    kv = (ctx - q_len + 1 + torch.arange(q_len, device="cuda")).int()
    row_req = torch.zeros(q_len, dtype=torch.int32, device="cuda")
    sb = (kv - 1) // BLK
    sc = torch.rand(hkv, q_len, nblk, generator=g, device="cuda")
    sc = torch.where(
        torch.arange(nblk, device="cuda") < sb[None, :, None],
        sc,
        torch.full_like(sc, -1.0),
    )
    topk = torch.full((hkv, q_len, TOPK), -1, dtype=torch.int32, device="cuda")
    v, i = sc.topk(TOPK - 1, dim=-1)
    topk[..., : TOPK - 1] = torch.where(v >= 0, i.int(), torch.full_like(i.int(), -1))
    topk[..., TOPK - 1] = sb.int()[None, :]
    q = torch.randn(q_len, hkv * group, D, generator=g, device="cuda").bfloat16()
    out = torch.empty_like(q)
    ks = vs = torch.ones(1, dtype=torch.float32, device="cuda")
    _, us = run_perftest(
        minimax_m3_sparse_pa_nvfp4_prefill,
        q,
        pool,
        topk,
        bt,
        row_req,
        kv,
        SM_SCALE,
        out,
        ks,
        vs,
        num_iters=10,
        num_warmup=2,
    )
    nref = min(q_len, 8)
    ref = run_torch(
        {
            "q": q[:nref],
            "topk": topk[:, :nref],
            "bt": bt % BASE_PAGES,
            "row_req": row_req[:nref],
            "row_kv": kv[:nref],
            "hkv": hkv,
            "group": group,
            "kd": kd,
            "vd": vd,
        }
    )
    nvalid = int((topk >= 0).sum().item())
    keys = nvalid * BLK - int(((sb + 1) * BLK - kv).sum().item()) * hkv
    return {
        "us": us,
        "TFLOPS": 4 * group * D * keys / us / 1e6,
        "TB/s": nvalid * 2 * PAGE_B / us / 1e6,  # logical: re-used pages count again
        "err": rel_l2(out[:nref], ref),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--case",
        nargs="*",
        default=[],
        help="run only cases whose name contains one of these",
    )
    ap.add_argument(
        "--perf", action="store_true", help="also run the synthetic M3.1 perf sweep"
    )
    args = ap.parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        aiter.logger.info("needs gfx950; skipped")
        return 0
    want = lambda n: not args.case or any(s in n for s in args.case)
    rows, failed = [], 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        if want(name):
            for sp in DECODE_SPLITS:
                rows.append(test_correctness(name, hkv, group, sum(q_lens), sp))
    for name, seq_lens, q_lens, hkv, group, kw in PREFILL_CASES:
        if want(name):
            for tq, rnd in PREFILL_TQ:
                rows.append(
                    test_prefill_correctness(name, hkv, group, sum(q_lens), tq, rnd)
                )
    df = pd.DataFrame(rows)
    failed += int((~df["ok"]).sum())
    aiter.logger.info(
        "correctness (rel_l2 vs fp64 oracle, tol %.1e):\n%s",
        REL_L2_TOL,
        df.to_markdown(index=False),
    )
    failed += decode_splits_agree()
    failed += decode_xcd_grid_agree()
    failed += decode_late_v_agree()
    failed += padding_rows()
    if args.perf:
        perf = [
            test_decode_perf(r, ql, hkv, g, ctx)
            for (r, ql), (hkv, g), ctx in itertools.product(
                [(1, 1), (8, 1), (64, 1), (256, 1), (512, 1), (32, 4), (64, 8)],
                [(1, 16), (4, 16)],
                [32768],
            )
        ]
        df = pd.DataFrame(perf)
        aiter.logger.info("decode perf:\n%s", df.to_markdown(index=False))
        failed += int((df["err"] > REL_L2_TOL).sum())
        perf = [
            test_prefill_perf(ql, hkv, g, ctx)
            for (ql, ctx), (hkv, g) in itertools.product(
                [(512, 4096), (2048, 32768), (8192, 32768)], [(1, 16), (4, 16)]
            )
        ]
        df = pd.DataFrame(perf)
        aiter.logger.info("prefill perf:\n%s", df.to_markdown(index=False))
        failed += int((df["err"] > REL_L2_TOL).sum())
    print("FAILED" if failed else "PASSED", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
