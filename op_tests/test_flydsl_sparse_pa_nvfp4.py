# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged attention, decode and prefill (FlyDSL, gfx950).

The kernels take the selector's output directly: ``topk_idx [Hkv, rows, topk]`` logical
128-token block ids plus the request block table; K / V pages are the single-buffer
NVFP4 page (see ``kernels/sparse_pa_nvfp4_common.py``).

Reference: an fp64 torch attention over K / V = e2m1 x e4m3 scale x global scale taken
from the token-major codes and scale bytes the pages were *built from* -- nothing is
read back through the kernel's lane layout, so a wrong page layout cannot cancel. The
kernels do the products exactly in f16, so what is left against the reference is the
bf16 output (relative L2 of ~1.7e-3) and the f16 Q / P.

Run (exits nonzero if any check fails):
    python op_tests/test_flydsl_sparse_pa_nvfp4.py [--case SUBSTR ...] [--perf]
"""

import argparse
import itertools
import math
import sys
import zlib

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_common import (
    unit_scale as _unit_scale,
)
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import (
    MAX_SPLIT_ROWS,
    _split_buffers,
    sparse_pa_nvfp4_decode,
    sparse_pa_nvfp4_decode_alloc,
)
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_prefill import (
    sparse_pa_nvfp4_prefill,
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
    v_off = j * BLK + t
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
    # Above, each head of a side is [data 8192 B | scales 1024 B]. A page groups them
    # differently: [K data of all heads | K scales of all heads | V data ... | V scales ...].
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
    topk_n=TOPK,  # slots per row
    fill=TOPK,  # valid ids per row (<= topk_n incl. the tail)
    junk=False,  # sprinkle invalid ids (-1 in the middle, ids past the tail, negatives)
    dup=0,  # repeat the first `dup` ids of each row in the free slots (out of contract)
    alias=False,  # every logical block of a request maps to the request's first page
    poison=False,  # NaN (e4m3 0x7f) scale bytes in the unwritten tail of every request's last page
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
        if alias:
            bt[r, : nblk[r]] = bt[r, 0].clone()
        o += nblk[r]
    amp = torch.exp(torch.randn(n_pages, BLK, hkv, 1, generator=g) * 0.25)
    k = (torch.randn(n_pages, BLK, hkv, D, generator=g) * amp).bfloat16()
    v = (torch.randn(n_pages, BLK, hkv, D, generator=g) * amp).bfloat16()
    kc, ks = quantize_nvfp4(k.to(device), k_scale)
    vc, vs = quantize_nvfp4(v.to(device), v_scale)
    if poison:
        for r in range(reqs):
            page = int(bt[r, nblk[r] - 1])
            used = seq_lens[r] - (nblk[r] - 1) * BLK
            ks[page, used:] = vs[page, used:] = 0x7F
    cache = pack_pages(kc, ks, vc, vs)
    rows = sum(q_lens)
    row_req = torch.empty(rows, dtype=torch.int32)
    row_kv = torch.empty(rows, dtype=torch.int32)
    i = 0
    for r in range(reqs):
        for j in range(q_lens[r]):
            row_req[i], row_kv[i] = r, seq_lens[r] - q_lens[r] + j + 1
            i += 1
    topk = torch.full((hkv, rows, topk_n), -1, dtype=torch.int32)
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
            if dup:
                ids = torch.cat([ids, ids[:dup]])[:TOPK]
            topk[h, i, : len(ids)] = ids
            if junk:
                free = list(range(len(ids), topk_n))
                if free:  # an id past the row's own block, and -1 gaps inside the list
                    topk[h, i, free[0]] = (
                        sb + 1 + int(torch.randint(0, 5, (1,), generator=g))
                    )
                if len(ids) > 2:
                    topk[h, i, 1] = -1
    q = (torch.randn(rows, hkv * group, D, generator=g)).bfloat16()
    dev = device
    return {
        "q": q.to(dev),
        "cache": cache,
        "topk": topk.to(dev),
        "bt": bt.to(dev),
        "row_req": row_req.to(dev),
        "row_kv": row_kv.to(dev),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=dev),
        "q_lens": list(q_lens),
        "k_scale": torch.tensor([k_scale], dtype=torch.float32, device=dev),
        "v_scale": torch.tensor([v_scale], dtype=torch.float32, device=dev),
        "hkv": hkv,
        "group": group,
        "kd": deq(kc, ks, k_scale),
        "vd": deq(vc, vs, v_scale),  # fp64 [pages, 128, hkv, 128]
    }


def run_torch(c):
    """fp64 reference: attention of every row over the valid keys of its blocks. A block id
    repeated in a row counts once per slot (both kernels do the same)."""
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
    return sparse_pa_nvfp4_decode(
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
    return sparse_pa_nvfp4_prefill(
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
    ("alias_pages", [129, 3000, 700], [1] * 3, 1, 16, {"alias": True}),
    ("topk8_h1", [5000, 2300, 900, 100], [1] * 4, 1, 16, {"topk_n": 8, "fill": 8}),
    ("topk8_h2_mtp4", [5000, 2300, 900, 130], [4] * 4, 2, 8, {"topk_n": 8, "fill": 8}),
    ("topk1_h1", [5000, 2300, 900, 100], [1] * 4, 1, 16, {"topk_n": 1, "fill": 1}),
    ("topk64_h1", [9000, 8300, 4000, 300], [1] * 4, 1, 16, {"topk_n": 64, "fill": 64}),
    ("topk64_h4", [9000, 8300, 4000, 300], [1] * 4, 4, 16, {"topk_n": 64, "fill": 40}),
]
DECODE_SPLITS = (None, 1, 2, 4, 8, 16)


def splits_of(kw):
    """The split counts a case can take (splits <= topk)."""
    return tuple(s for s in DECODE_SPLITS if s is None or s <= kw.get("topk_n", TOPK))


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
    # Out of contract (ids within a row should be distinct): a repeated id counts once per
    # slot, whatever the tile, so every tile matches the oracle (and the decode kernel).
    ("dup_ids", [3000, 700, 130], [300, 700, 130], 1, 16, {"fill": 12, "dup": 3}),
    ("dup_ids_h2", [900, 2100], [200, 1000], 2, 16, {"fill": 13, "dup": 2}),
    ("ragged_batch", [2000, 33, 4097, 512], [511, 33, 70, 512], 1, 16, {}),
    # A table that maps several logical blocks of a request to one page: each block keeps
    # its own causal limit (the first is the block_table [[p, p]] / row_kv 129 case).
    ("alias_two_blocks", [129], [1], 1, 16, {"alias": True}),
    ("alias_chunk", [3000, 700, 130], [300, 700, 130], 1, 16, {"alias": True}),
]
PREFILL_TQ = ((4, 2), (4, 1), (2, 1), (1, 1))


def check(name, out, ref, tol=REL_L2_TOL):
    err = rel_l2(out, ref)
    mx = (out.double() - ref).abs().max().item()
    ok = err <= tol and bool(torch.isfinite(out.float()).all())
    mismatch = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=2e-2 * ref.abs().max().item(),
        msg=f"{name}: ",
    )
    return ok and mismatch == 0, err, mx


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
    """Every split count agrees with one workgroup to rel-L2 <= 2e-3 (the fp32 merge order
    differs, so the bf16 output is not bitwise equal)."""
    bad = 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000, **kw
        )
        base = decode_call(c, 1)
        for s in splits_of(kw):
            if s in (None, 1) or len(seq_lens) * hkv > 1024:
                continue
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
    # Every tile: the auto tile for 2 rows is tq 1, where a padding row never shares a
    # workgroup with its neighbour.
    ref = run_torch(c)
    for tq, rnd in PREFILL_TQ:
        out = prefill_call(c1, tq=tq, rounds=rnd)
        same = torch.equal(out[1], prefill_call(c, tq=tq, rounds=rnd)[1])
        bad += int(
            out[0].abs().max().item() != 0
            or not same
            or rel_l2(out[1], ref[1]) > REL_L2_TOL
        )
    if bad:
        print("padding rows: FAILED")
    return bad


def padding_empty_table():
    """A padding-only call (length 0, every slot -1) touches neither the block table nor
    the pages: it must work with an empty table and with an empty KV cache ([0, page]).
    """
    bad = 0
    c = make_case([500], [1], 1, 16, seed=5)
    topk = torch.full_like(c["topk"], -1)
    zero_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    caches = {
        "cache": c["cache"],
        "empty cache": torch.empty(
            0, c["cache"].shape[1], dtype=torch.uint8, device="cuda"
        ),
    }
    tables = {
        "empty table": torch.zeros(1, 0, dtype=torch.int32, device="cuda"),
        "table": torch.zeros(1, 4, dtype=torch.int32, device="cuda"),
    }
    for (cn, cache), (bn, bt) in itertools.product(caches.items(), tables.items()):
        for s, x in itertools.product((None, 1, 4), (0, 8)):
            out = torch.full_like(c["q"], 1.0)
            sparse_pa_nvfp4_decode(
                c["q"], cache, topk, bt, zero_len, SM_SCALE, out,
                c["k_scale"], c["v_scale"], splits=s, xcds=x,
            )  # fmt: skip
            if out.abs().max().item() != 0:
                bad += 1
                print(f"{cn} / {bn}: decode padding row not zero (splits={s} xcds={x})")
        for tq, rnd in PREFILL_TQ:
            out = torch.full_like(c["q"], 1.0)
            sparse_pa_nvfp4_prefill(
                c["q"], cache, topk, bt, zero_len, zero_len, SM_SCALE, out,
                c["k_scale"], c["v_scale"], tq=tq, rounds=rnd,
            )  # fmt: skip
            if out.abs().max().item() != 0:
                bad += 1
                print(f"{cn} / {bn}: prefill padding row not zero (tq={tq})")
    return bad


# (name, seq_lens, q_lens, hkv, group, kwargs); the unwritten tail of the last page holds NaN scales
POISON_CASES = [
    ("poison_one_token", [1, 2, 5, 127], [1] * 4, 1, 16, {}),
    ("poison_tail_h4", [129, 300, 1000, 5000], [1] * 4, 4, 16, {}),
    ("poison_mtp4", [5, 130, 900, 4097], [4] * 4, 1, 16, {}),
    ("poison_g8_tail_only", [1, 700], [1, 1], 1, 8, {"fill": 1}),
]


def poisoned_tail():
    """Keys past a request's length are masked out of P, but 0 * NaN is NaN in the PV MFMA:
    a recycled page may hold a NaN V scale there (e4m3 0x7f), and the kernel must zero it.
    """
    bad = 0
    for name, seq_lens, q_lens, hkv, group, kw in POISON_CASES:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000,
            poison=True, **kw,
        )  # fmt: skip
        ref = run_torch(c)
        for s, lv in ((None, None), (1, False), (1, True), (4, None)):
            out = decode_call(c, s, late_v=lv)
            bad += int(not check(f"decode {name} splits={s} late_v={lv}", out, ref)[0])
    # Prefill: tiles of rows share the tail page; the keys past the last row of a tile are
    # unwritten, the ones between its rows are not.
    for name, seq_lens, q_lens, hkv, group, kw in POISON_CASES + [
        ("poison_chunk", [5, 200, 700], [5, 70, 300], 1, 16, {}),
    ]:
        c = make_case(
            seq_lens, q_lens, hkv, group, seed=zlib.crc32(name.encode()) % 1000,
            poison=True, **kw,
        )  # fmt: skip
        ref = run_torch(c)
        for tq, rnd in PREFILL_TQ:
            out = prefill_call(c, tq=tq, rounds=rnd)
            bad += int(not check(f"prefill {name} tq={tq}", out, ref)[0])
    return bad


def refusals():
    """Arguments the kernels cannot read correctly are refused before launch (ValueError,
    which ``python -O`` does not strip), each with its own message."""
    bad = 0
    c = make_case([500, 700], [1, 1], 1, 16, seed=5)
    cache, ks = c["cache"], c["k_scale"]
    pages, page_b = cache.shape
    # The kernel loads Q, the output and the pages in dwords: their addresses and the page
    # stride must be 4-byte aligned (a padded page stays fine).
    padded = torch.zeros(pages, page_b + 1, dtype=torch.uint8, device="cuda")
    padded[:, :page_b] = cache
    shifted = torch.zeros(pages, page_b + 4, dtype=torch.uint8, device="cuda")
    shifted[:, 1 : page_b + 1] = cache
    odd_q = torch.empty(c["q"].numel() + 1, dtype=torch.bfloat16, device="cuda")[1:]
    odd_q = odd_q.view_as(c["q"])
    odd_q.copy_(c["q"])
    # Element offsets (rows x stride) are 32-bit: views reaching past 2**31 elements are
    # refused (the 8 GiB buffer below is never read; the refusal comes first).
    c2 = make_case([500, 700], [1, 1], 2, 8, seed=5)
    wide = torch.empty(2**31 + 64, dtype=torch.int32, device="cuda")
    # 513 rows x 2 kv heads = 1026 > MAX_SPLIT_ROWS units: only splits = 1 can hold them.
    c3 = make_case([128] * 513, [1] * 513, 2, 8, seed=5)
    over = {k: c3[k] for k in ("q", "cache", "bt", "seq_lens", "topk")}
    ws = sparse_pa_nvfp4_decode_alloc(c["q"].device)
    aligned = torch.empty(ws.partials.numel() + 4, dtype=torch.float32, device="cuda")
    shared = [
        ("strided cache view", {"cache": cache[:, ::2]}, "unit last stride"),
        ("cache too narrow", {"cache": cache[:, :-1]}, "smaller than its payload"),
        ("cpu k_scale", {"k_scale": ks.cpu()}, "k_scale and v_scale"),
        ("fp16 k_scale", {"k_scale": ks.half()}, "k_scale and v_scale"),
        ("empty k_scale", {"k_scale": ks[:0]}, "k_scale and v_scale"),
        ("fp16 q", {"q": c["q"].half()}, "bfloat16"),
        ("group 4", {"q": c["q"][:, :4]}, "8 or 16 heads"),
        ("2-D topk_idx", {"topk": c["topk"][0]}, "must be 3-D"),
        ("topk_idx rows < q rows", {"topk": c["topk"][:, :1]}, "[Hkv, >= rows, topk]"),
        ("int8 kv_cache", {"cache": cache.view(torch.int8)}, "kv_cache must be uint8"),
        (
            "strided block_table last dim",
            {"bt": torch.stack([c["bt"], c["bt"]], -1)[..., 0]},
            "block_table must be int32 with a unit last stride",
        ),
        (
            "strided topk_idx last dim",
            {"topk": torch.stack([c["topk"], c["topk"]], -1)[..., 0]},
            "with a unit last stride",
        ),
        (
            "non-contiguous q",
            {
                "q": torch.empty(2, 16, 2 * D, dtype=torch.bfloat16, device="cuda")[
                    ..., :D
                ]
            },
            "q and out must be contiguous",
        ),
        ("int64 topk_idx", {"topk": c["topk"].long()}, "topk_idx must be int32"),
        ("flattened block_table", {"bt": c["bt"].flatten()}, "block_table 2-D"),
        ("sm_scale 0", {"sm_scale": 0.0}, "sm_scale"),
        ("sm_scale nan", {"sm_scale": float("nan")}, "sm_scale"),
        ("cpu q", {"q": c["q"].cpu()}, "CUDA"),
        ("page stride not 4-aligned", {"cache": padded[:, :page_b]}, "4-byte"),
        ("cache base not 4-aligned", {"cache": shifted[:, 1 : page_b + 1]}, "4-byte"),
        ("cache base +2 bytes", {"cache": shifted[:, 2 : page_b + 2]}, "4-byte"),
        ("q base not 4-aligned", {"q": odd_q}, "4-byte"),
        ("out base not 4-aligned", {"out": odd_q}, "4-byte"),
        (
            "topk_idx offsets past 2**31",
            {
                "q": c2["q"],
                "cache": c2["cache"],
                "bt": c2["bt"],
                "topk": wide.as_strided((2, 2, 16), (2**31, 16, 1)),
            },
            "topk_idx element offsets",
        ),
        (
            "block_table offsets past 2**31",
            {
                "q": c2["q"],
                "cache": c2["cache"],
                "topk": c2["topk"],
                "bt": wide.as_strided((2, 4), (2**31, 1)),
            },
            "block_table element offsets",
        ),
        (
            "q over 2 GiB",
            {"q": torch.empty(2**19 + 1, 16, D, dtype=torch.bfloat16, device="cuda")},
            "2 GiB",
        ),
        # Exactly 2 GiB is refused too (`<`, not `<=`): the masked lanes store to a sentinel
        # dword that is only dropped when it lies past the end of `out`, and at exactly 2 GiB
        # it is the last dword of `out`.
        (
            "q exactly 2 GiB",
            {"q": torch.empty(2**19, 16, D, dtype=torch.bfloat16, device="cuda")},
            "2 GiB",
        ),
        (
            "fp16 out",
            {"out": torch.empty_like(c["q"], dtype=torch.float16)},
            "bfloat16",
        ),
        (
            "out of another shape",
            {
                "out": torch.empty(
                    3, *c["q"].shape[1:], dtype=torch.bfloat16, device="cuda"
                )
            },
            "same shape",
        ),
    ]
    decode_only = [
        ("splits > MAX_SPLITS", {"splits": 32}, "splits must be"),
        ("splits 0", {"splits": 0}, "splits must be"),
        ("short block_table", {"bt": c["bt"][:1]}, "row per request"),
        ("2-D seq_lens", {"seq_lens": c["seq_lens"][None]}, "seq_lens 1-D"),
        ("xcds 1", {"xcds": 1}, "xcds must be"),
        ("rows != reqs * query_len", {"query_len": 2}, "rows must be reqs * query_len"),
        (
            "splits 2 with rows * Hkv > MAX_SPLIT_ROWS",
            {**over, "splits": 2},
            f"splits > 1 needs rows * Hkv <= {MAX_SPLIT_ROWS}",
        ),
        (
            "topk 65",
            {"topk": torch.full((1, 2, 65), -1, dtype=torch.int32, device="cuda")},
            "topk must be in [1, 64]",
        ),
        (
            "splits > topk",
            {
                "topk": torch.full((1, 2, 4), -1, dtype=torch.int32, device="cuda"),
                "splits": 8,
            },
            "splits must be in [1, topk = 4]",
        ),
        (
            "strided seq_lens",
            {"seq_lens": torch.stack([c["seq_lens"], c["seq_lens"]], -1)[..., 0]},
            "seq_lens must be int32 with a unit last stride",
        ),
        ("workspace not a pair", {"workspace": ws.partials}, "workspace must be the"),
        (
            "workspace partials fp16",
            {"workspace": (ws.partials.half(), ws.tickets)},
            "partials must be float32",
        ),
        (
            "workspace tickets int64",
            {"workspace": (ws.partials, ws.tickets.long())},
            "tickets int32",
        ),
        (
            "workspace on cpu",
            {"workspace": (ws.partials[:1].cpu(), ws.tickets)},
            "q's device",
        ),
        (
            "workspace strided",
            {"workspace": (ws.partials[::2], ws.tickets)},
            "contiguous",
        ),
        (
            "workspace partials too small",
            {"workspace": (ws.partials[:-1], ws.tickets)},
            "workspace needs",
        ),
        (
            "workspace tickets too small",
            {"workspace": (ws.partials, ws.tickets[:-1])},
            "workspace needs",
        ),
        (
            "workspace partials misaligned",
            {"workspace": (aligned[1 : 1 + ws.partials.numel()], ws.tickets)},
            "aligned",
        ),
    ]
    prefill_only = [
        ("empty block_table", {"bt": c["bt"][:0]}, "at least one request row"),
        ("tq 3", {"tq": 3}, "tq must be"),
        ("rounds without tq", {"rounds": 1}, "rounds needs tq"),
    ]

    def decode(a):
        sparse_pa_nvfp4_decode(
            a["q"], a["cache"], a["topk"], a["bt"], a["seq_lens"], a["sm_scale"],
            a.get("out"), a["k_scale"], a["v_scale"], query_len=a.get("query_len", 1),
            splits=a.get("splits"), xcds=a.get("xcds"), workspace=a.get("workspace"),
        )  # fmt: skip

    def prefill(a):
        sparse_pa_nvfp4_prefill(
            a["q"], a["cache"], a["topk"], a["bt"], c["row_req"], c["row_kv"],
            a["sm_scale"], a.get("out"), a["k_scale"], a["v_scale"], tq=a.get("tq"),
            rounds=a.get("rounds"),
        )  # fmt: skip

    for kind, call, cases in (
        ("decode", decode, shared + decode_only),
        ("prefill", prefill, shared + prefill_only),
    ):
        for name, kw, msg in cases:
            a = {
                "q": c["q"], "cache": cache, "sm_scale": SM_SCALE, "k_scale": ks,
                "v_scale": c["v_scale"], "bt": c["bt"], "topk": c["topk"],
                "seq_lens": c["seq_lens"], **kw,
            }  # fmt: skip
            try:
                call(a)
            except ValueError as e:
                if msg in str(e):
                    continue
                print(f"refusal: {kind} {name}: wrong message: {e!r}")
            else:
                print(f"refusal: {kind} {name} was accepted")
            bad += 1
    # Just below the limit a call still runs (prefill; decode shares the check): q and out
    # are 2**31 - 4096 bytes, and the masked lanes of the last, partial tile store to the
    # sentinel dword, which has to fall past the end of `out`. The rows are copies of one row.
    n = 2**19 - 1
    one = dict(c, q=c["q"][:1], topk=c["topk"][:, :1], row_req=c["row_req"][:1])
    one["row_kv"] = c["row_kv"][:1]
    ref = run_torch(one)
    big = dict(
        c,
        q=c["q"][:1].repeat(n, 1, 1),
        topk=c["topk"][:, :1].expand(-1, n, -1).contiguous(),
        row_req=c["row_req"][:1].expand(n).contiguous(),
        row_kv=c["row_kv"][:1].expand(n).contiguous(),
    )
    out = prefill_call(big)
    for r in (0, n - 1):
        if not check(f"q just below 2 GiB, row {r}", out[r : r + 1], ref)[0]:
            bad += 1
    del big, out
    decode_call(c, 16)  # the largest split count still launches
    # Past the limit the default falls back to one workgroup per unit; only an explicit
    # split count is refused.
    if not torch.equal(decode_call(c3), decode_call(c3, 1)):
        print("rows * Hkv > MAX_SPLIT_ROWS: the default is not splits = 1")
        bad += 1
    # An aligned padded page is accepted and reads the same data.
    pad4 = torch.zeros(pages, page_b + 4, dtype=torch.uint8, device="cuda")
    pad4[:, :page_b] = cache
    if not torch.equal(decode_call(dict(c, cache=pad4[:, :page_b])), decode_call(c)):
        print("padded page (stride +4) gives a different result")
        bad += 1
    return bad


def capture_scratch():
    """The scale placeholder and the default split scratch are allocated (and zero-filled) on
    the first call that needs them; inside a capture the fill would be recorded, not run, so
    that call is refused. After an eager call the capture replays correctly (and poisoned
    tickets, what a skipped fill leaves, fail the oracle check); an explicit workspace lets
    the capture skip the default scratch altogether."""
    bad = 0
    dev = torch.cuda.current_device()
    c = make_case([3000, 2300, 900, 130], [1] * 4, 1, 16, seed=7)
    ref = run_torch(c)

    # (name, splits, caches the warm-up leaves out, text the refusal must contain)
    for name, splits, cold, msg in (
        ("splits 4, cold", 4, (_unit_scale, _split_buffers), "scale placeholder"),
        ("splits 4, warm for splits 1", 4, (_split_buffers,), "split scratch"),
        ("splits 1, cold", 1, (_unit_scale,), "scale placeholder"),
    ):
        # the compile happens here, so the capture below only meets the allocation
        decode_call(c, splits)
        for f in cold:
            f.cache_clear()
        g = torch.cuda.CUDAGraph()
        err = None
        try:
            with torch.cuda.graph(g):
                decode_call(c, splits)
        except ValueError as e:
            err = str(e)
        if err is None or msg not in err:
            print(f"capture {name}: expected a refusal naming {msg!r}, got {err!r}")
            bad += 1
        if any(f.cache_info().currsize for f in cold):
            print(f"capture {name}: a refused call left a cached allocation")
            bad += 1
    # The explicit alloc is refused inside a capture, and outside one it replaces the
    # eager warm-up of the scratch.
    for f in (_unit_scale, _split_buffers):
        f.cache_clear()
    try:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            sparse_pa_nvfp4_decode_alloc(dev)
        print("capture: sparse_pa_nvfp4_decode_alloc was accepted inside a capture")
        bad += 1
    except ValueError:
        pass
    ws = sparse_pa_nvfp4_decode_alloc(dev)
    ws.partials.fill_(float("nan"))  # every split stores its partial before it is read
    out = torch.empty_like(c["q"])

    def run(**kw):
        sparse_pa_nvfp4_decode(
            c["q"], c["cache"], c["topk"], c["bt"], c["seq_lens"], SM_SCALE, out,
            c["k_scale"], c["v_scale"], splits=4, **kw,
        )  # fmt: skip

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(workspace=ws)
    if _split_buffers.cache_info().currsize:
        print("capture: a call with a workspace allocated the default scratch")
        bad += 1
    for i in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        bad += int(not check(f"capture with workspace, replay {i}", out, ref)[0])
    # The default scratch: eager warm-up, capture, replays.
    decode_call(c, 4)  # allocates and fills
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for i in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        bad += int(not check(f"capture replay {i}", out, ref)[0])
    # Eager calls after the replays still see clean tickets.
    bad += int(not check("eager after replay", decode_call(c, 4), ref)[0])
    # A skipped fill would leave the tickets nonzero (no merge): the oracle check must see it.
    cnt = _split_buffers(dev).tickets
    cnt.fill_(-1)
    out.fill_(float("nan"))
    run()
    if check("tickets poisoned (expected to fail)", out, ref)[0]:
        print("capture: poisoned tickets went unnoticed")
        bad += 1
    cnt.zero_()
    return bad


def two_streams(n=64, rounds=3):
    """Split launches on two streams of one device. Each stream replays a graph of ``n``
    launches (back to back, so they overlap), unsynchronised, every launch with its own
    output. With a workspace per stream (partials poisoned with NaN) every output matches the
    oracle; the same launches on the shared default scratch are the caller's error: their
    wrong outputs are only reported (a race, so the count varies), then the tickets are reset.
    """
    dev = torch.cuda.current_device()
    c = make_case([3000, 2300, 900, 130, 2000, 1500, 700, 5000], [1] * 8, 1, 16, seed=3)
    ref = run_torch(c)
    decode_call(c, 4)  # loads the variant and the default scratch

    def wrong_outputs(workspaces):
        streams = [torch.cuda.Stream() for _ in range(2)]
        outs = [[torch.empty_like(c["q"]) for _ in range(n)] for _ in streams]
        graphs = []
        for st, ws, os in zip(streams, workspaces, outs):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=st):
                for o in os:
                    sparse_pa_nvfp4_decode(
                        c["q"], c["cache"], c["topk"], c["bt"], c["seq_lens"], SM_SCALE,
                        o, c["k_scale"], c["v_scale"], splits=4, workspace=ws,
                    )  # fmt: skip
            graphs.append(g)
        wrong = 0
        for _ in range(rounds):
            for os in outs:
                for o in os:
                    o.fill_(float("nan"))
            torch.cuda.synchronize()
            for st, g in zip(streams, graphs):
                with torch.cuda.stream(st):
                    g.replay()
            torch.cuda.synchronize()
            wrong += sum(
                not (
                    rel_l2(o, ref) <= REL_L2_TOL
                    and bool(torch.isfinite(o.float()).all())
                )
                for os in outs
                for o in os
            )
        return wrong

    wss = [sparse_pa_nvfp4_decode_alloc(dev) for _ in range(2)]
    for ws in wss:
        ws.partials.fill_(float("nan"))
    bad = wrong_outputs(wss)
    if bad:
        print(f"two streams, a workspace each: {bad} of {2 * n * rounds} outputs wrong")
    shared = wrong_outputs([None, None])
    _split_buffers(dev).tickets.zero_()
    print(
        f"two streams, shared default scratch (caller error): {shared} of {2 * n * rounds} outputs wrong"
    )
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
        sparse_pa_nvfp4_decode,
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
        {
            "q": q[:nref],
            "topk": topk[:, :nref],
            "bt": bt % BASE_PAGES,
            "row_req": (torch.arange(nref, device="cuda") // query_len).int(),
            "row_kv": kv[:nref].int(),
            "hkv": hkv,
            "group": group,
            "kd": kd,
            "vd": vd,
        }
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
    per row (neighbouring rows overlap only by chance: a low case for the page reuse).
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
        sparse_pa_nvfp4_prefill,
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
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
        aiter.logger.info("needs gfx950; skipped")
        return 0
    want = lambda n: not args.case or any(s in n for s in args.case)
    rows, failed = [], 0
    for name, seq_lens, q_lens, hkv, group, kw in DECODE_CASES:
        if want(name):
            for sp in splits_of(kw):
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
    failed += padding_empty_table()
    failed += poisoned_tail()
    failed += refusals()
    failed += capture_scratch()
    failed += two_streams()
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
        failed += int((~(df["err"] <= REL_L2_TOL)).sum())  # a NaN err fails too
        perf = [
            test_prefill_perf(ql, hkv, g, ctx)
            for (ql, ctx), (hkv, g) in itertools.product(
                [(512, 4096), (2048, 32768), (8192, 32768)], [(1, 16), (4, 16)]
            )
        ]
        df = pd.DataFrame(perf)
        aiter.logger.info("prefill perf:\n%s", df.to_markdown(index=False))
        failed += int((~(df["err"] <= REL_L2_TOL)).sum())
    print("FAILED" if failed else "PASSED", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
