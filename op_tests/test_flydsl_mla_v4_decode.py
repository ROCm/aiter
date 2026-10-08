# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL v4 nm MLA decode (gfx950) vs an fp64 reference, with the asm
aiter.mla.mla_decode_fwd_v4_nm in the perf tables (run_perftest, warm caches):
flydsl_mla_decode_fwd_v4_nm (bf16), flydsl_mla_v4_decode_fused (inverse RoPE + wo_a
mxfp8 quant) and grouped DSpark verify (q_kv_bounds).
Rows are packed like the live KV pool, 0xFF padding at bytes 462..511."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
import aiter.mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_mla_decode_fwd_v4_nm,
    flydsl_mla_v4_decode_fused,
    flydsl_mla_v4_decode_supported,
    flydsl_mla_v4_grouped_verify_meta,
)
from aiter.ops.flydsl import mla_v4_decode as fly_v4
from aiter.ops.inverse_rope_group_quant import inverse_rope_group_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
NOPE, ROPE, DV, ROW, QB = 448, 64, 512, 512, 128
PAD = NOPE + 14
WIN, RING, CRATIO = 128, 256, 128  # SWA window, ring slots per request, HCA ratio


def pack_rows(n, gen, pad=0xFF):
    """fp8 rows [n, 512] u8 (448 e4m3 + 7 e8m0 scales, each twice) + bf16 rope."""
    nope = torch.randn(n, NOPE // 64, 64, generator=gen)
    rope = torch.randn(n, ROPE, generator=gen).to(dtypes.bf16)
    scale = torch.exp2(torch.ceil(torch.log2((nope.abs().amax(-1) / 448).clamp(1e-4))))
    row = torch.full((n, ROW), pad, dtype=torch.uint8)
    row[:, :NOPE] = (
        (nope / scale[..., None]).to(dtypes.fp8).view(n, NOPE).view(torch.uint8)
    )
    e8 = (torch.log2(scale).to(torch.int32) + 127).to(torch.uint8)
    row[:, NOPE:PAD] = e8.repeat_interleave(2, dim=-1)
    return row, rope


def dequant_rows(row, rope):
    vals = row[..., :NOPE].view(dtypes.fp8).double()
    sc = torch.exp2(row[..., NOPE:PAD:2].double() - 127)
    return torch.cat(
        [(vals.unflatten(-1, (7, 64)) * sc[..., None]).flatten(-2), rope.double()], -1
    )


def make_inputs(H, msq, batch, kv_len, ragged, seed=0):
    """batch sequences of msq q rows; ragged: kv lengths in [0, 2 kv_len], every 5th empty
    (2: the first one 16 kv_len)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gcpu = torch.Generator().manual_seed(seed)
    kv_lens = [kv_len] * batch
    if ragged:
        kv_lens = torch.randint(
            0, 2 * kv_len + 1, (batch,), generator=gcpu, device="cpu"
        )
        kv_lens = [0 if b % 5 == 3 else n for b, n in enumerate(kv_lens.tolist())]
        kv_lens[0] = 16 * kv_len if ragged == 2 else kv_lens[0]
    rows = max(1024, sum(kv_lens) + 64)
    kv, kv_rope = pack_rows(rows, gen)
    kv_indptr = torch.tensor([0] + kv_lens).cumsum(0).to(dtypes.i32)
    kv_indices = torch.randperm(rows, generator=gen)[: max(1, sum(kv_lens))].to(
        dtypes.i32
    )
    q, q_rope = pack_rows(batch * msq * H, gen)
    return {
        "q": q.view(-1, H, ROW),
        "q_rope": q_rope.view(-1, H, ROPE),
        "kv": kv,
        "kv_rope": kv_rope,
        "qo_indptr": torch.arange(batch + 1, dtype=dtypes.i32) * msq,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "sink": torch.randn(H, generator=gen),
    }, kv_lens


def run_torch(q, q_rope, kv, kv_rope, qo_indptr, kv_indptr, kv_indices, sink, **_):
    """fp64 softmax([q k^T / sqrt(512), sink]) @ kv per sequence; rows without keys NaN."""
    qd = dequant_rows(q, q_rope)
    out = torch.full(qd.shape, float("nan"), dtype=torch.float64)
    qo, ki = qo_indptr.tolist(), kv_indptr.tolist()
    for b in range(len(qo) - 1):
        rows = kv_indices[ki[b] : ki[b + 1]].long()
        if rows.numel() == 0 or qo[b + 1] == qo[b]:
            continue
        kd = dequant_rows(kv[rows], kv_rope[rows])
        s = torch.einsum("qhd,kd->qhk", qd[qo[b] : qo[b + 1]], kd) * DV**-0.5
        sk = sink.double()[None, :, None].expand(s.shape[0], -1, 1)
        p = torch.softmax(torch.cat([s, sk], -1), -1)[..., :-1]
        out[qo[b] : qo[b + 1]] = torch.einsum("qhk,kd->qhd", p, kd)
    return out


def inv_rope(ref, positions, freqs):
    T, H = ref.shape[:2]
    fc = torch.view_as_complex(freqs[positions].double().view(T, 1, ROPE // 2, 2))
    tail = torch.view_as_complex(ref[..., NOPE:].contiguous().view(T, H, ROPE // 2, 2))
    out = ref.clone()
    out[..., NOPE:] = torch.view_as_real(tail * fc.conj()).view(T, H, ROPE)
    return out


def rope_freqs(max_pos=1 << 18):
    inv = 10000.0 ** (-torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)
    ang = torch.arange(max_pos, dtype=torch.float64)[:, None] * inv
    return torch.stack([ang.cos(), ang.sin()], -1).flatten(1).float()


def dequant_mxfp8(xq, xs):
    T = xq.shape[0]
    x = xq.view(dtypes.fp8).view(T, -1, QB).double()
    return x * torch.exp2(xs.view(torch.uint8).view(T, -1, 1).double() - 127)


def check_mxfp8(ref, xq, xs, msg):
    """Error in units of each 128-wide block's amax (one e4m3 step <= 1/14 of it)."""
    T = ref.shape[0]
    blk = ref.reshape(T, -1, QB)
    amax = blk.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    out = dequant_mxfp8(xq, xs)
    err = checkAllclose(
        (blk / amax).float(), (out / amax).float(), rtol=0, atol=0.08, msg=msg
    )
    return err, ((out - blk).norm() / blk.norm()).item()


def rel_l2(o, ref):
    return ((o.double() - ref).norm() / ref.norm()).item()


def decode(inp, out, msq, backend="flydsl", **kw):
    rows = inp["kv"].shape[0]
    args = (
        inp["q"].view(dtypes.fp8),
        inp["q_rope"],
        inp["kv"].view(dtypes.fp8).view(rows, 1, 1, ROW),
        inp["kv_rope"].view(rows, 1, 1, ROPE),
        out,
        inp["qo_indptr"],
        inp["kv_indptr"],
        inp["kv_indices"],
        msq,
    )
    if backend == "flydsl":
        return flydsl_mla_decode_fwd_v4_nm(*args, sink=inp["sink"], **kw)
    aiter.mla.mla_decode_fwd_v4_nm(*args, sink=inp["sink"], **kw)
    return out


def graph_replay(fn, n=3, st=None):
    """Warm up eagerly on a side stream (as sglang does), capture, replay n times."""
    st = st or torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=st):
        fn()
    for _ in range(n):
        g.replay()
    torch.cuda.synchronize()
    return g


def bits(x):
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.uint8)


@benchmark()
def test_mla_v4_decode(H, msq, batch, kv_len, ragged, hint=None):
    inp, kv_lens = make_inputs(H, msq, batch, kv_len, ragged)
    ref = run_torch(**inp)
    ok = torch.tensor(
        [b * msq + r for b, n in enumerate(kv_lens) if n for r in range(msq)],
        dtype=torch.long,
    )
    empty = torch.tensor(
        [b * msq + r for b, n in enumerate(kv_lens) if not n for r in range(msq)],
        dtype=torch.long,
    )
    out = torch.empty((batch * msq, H, DV), dtype=dtypes.bf16)
    candidates = {"flydsl": lambda: decode(inp, out, msq, hint=hint)}
    if not ragged:
        out_asm = torch.empty_like(out)
        candidates["asm"] = lambda: decode(inp, out_asm, msq, "asm")
    cfg = fly_v4._plan(H, msq, batch, hint=hint)
    nkv = sum(kv_lens)
    nbytes = nkv * (ROW + ROPE * 2) + batch * msq * H * (ROW + ROPE * 2 + DV * 2)
    ret = {
        "gfx": get_gfx(),
        "layout": cfg.LAYOUT + ("q16" if cfg.Q16 else ""),
        "S": cfg.S,
    }
    for name, fn in candidates.items():
        o, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 4 * H * msq * nkv * DV / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref[ok].float(), o[ok].float(), msg=f"{name}: vs fp64"
        )
        ret[f"{name} rel_l2"] = rel_l2(o[ok], ref[ok])
    base = out.clone()
    assert ret["flydsl rel_l2"] < 3e-3, ret
    assert torch.isnan(base[empty].float()).all(), "kv_len == 0 rows must be NaN"
    p = dict(inp, q=inp["q"].clone(), kv=inp["kv"].clone())
    for k in ("q", "kv"):  # padding bytes are never read
        p[k][..., PAD:] = torch.randint(
            0, 256, p[k][..., PAD:].shape, dtype=torch.uint8
        )
    decode(p, out, msq, hint=hint)
    assert torch.equal(bits(out[ok]), bits(base[ok]))
    out.zero_()
    graph_replay(lambda: decode(inp, out, msq, hint=hint))
    assert torch.equal(bits(out), bits(base)), "graph replay"
    return ret


@benchmark()
def test_mla_v4_decode_fused(T, H, compress_ratio, kv_len, ragged):
    inp, kv_lens = make_inputs(H, 1, T, kv_len, ragged)
    del inp["qo_indptr"]
    inp["positions"] = (
        65536
        + torch.randint(
            0, 4096, (T,), generator=torch.Generator().manual_seed(2), device="cpu"
        ).cuda()
    )
    inp["freqs"] = rope_freqs()
    ref = inv_rope(
        run_torch(**inp, qo_indptr=torch.arange(T + 1, dtype=dtypes.i32)),
        inp["positions"],
        inp["freqs"],
    )
    ok = torch.tensor([t for t, n in enumerate(kv_lens) if n], dtype=torch.long)
    empty = torch.tensor([t for t, n in enumerate(kv_lens) if not n], dtype=torch.long)
    res = {}

    def fused():
        res["x"] = flydsl_mla_v4_decode_fused(**inp, compress_ratio=compress_ratio)
        return res["x"]

    candidates = {"flydsl_fused": fused}
    if not ragged:
        cos, sin = inp["freqs"][:, 0::2].to(dtypes.bf16), inp["freqs"][:, 1::2].to(
            dtypes.bf16
        )
        o = torch.empty((T, H, DV), dtype=dtypes.bf16)
        a = dict(inp, qo_indptr=torch.arange(T + 1, dtype=dtypes.i32))

        def asm_chain():
            decode(a, o, 1, "asm")
            return inverse_rope_group_quant(o, inp["positions"], cos, sin, H // 8, QB)

        candidates["asm+inverse_rope_group_quant"] = asm_chain
    hint = {4: "csa", 128: "hca"}.get(compress_ratio, "swa")
    cfg = fly_v4._plan(H, 1, T, "invrope_mxfp8", hint)
    ret = {"gfx": get_gfx(), "layout": cfg.LAYOUT, "S": cfg.S}
    for name, fn in candidates.items():
        (xq, xs), us = run_perftest(fn)
        err, rel = check_mxfp8(ref[ok], xq[ok], xs[ok], msg=f"{name}: vs fp64")
        ret[f"{name} us"] = us
        ret[f"{name} err"] = err
        ret[f"{name} rel_l2"] = rel
    q8, s8 = [x.view(torch.uint8).clone() for x in fused()]
    assert ret["flydsl_fused rel_l2"] < 0.05, ret
    assert not (q8[ok] & 0x7F).eq(0x7F).any(), "NaN codes in xq"
    assert ((q8[empty] & 0x7F) == 0x7F).all() and (s8[empty] == 0xFF).all()
    graph_replay(fused)
    assert torch.equal(res["x"][0].view(torch.uint8), q8) and torch.equal(
        res["x"][1], s8
    )
    return ret


def make_dspark(num_draft, kind, requests, seed=0, H=16):
    """One DSpark verify step on a pool [SWA ring: R x RING rows][compressed rows]: the
    drafts of request r end at positions around the SWA window / block commits; per
    token (sglang's streams): its window + (hca) its committed rows (p + 1) // 128."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    base = [6, 10, 126, 127, 128, 131, 133, 135, 255, 256, 260, 4000, 70000, 131000]
    last = [base[r % len(base)] + 7 * (r // len(base)) for r in range(requests)]
    pos = [p - num_draft + 1 + d for p in last for d in range(num_draft)]
    ncomp = [(p + 1) // CRATIO if kind == "hca" else 0 for p in last]
    swa_pages = requests * RING
    kv, kv_rope = pack_rows(swa_pages + max(1, sum(ncomp)), gen)
    comp = torch.randperm(max(1, sum(ncomp)), generator=gen, device="cuda").tolist()
    lists = [comp[sum(ncomp[:r]) : sum(ncomp[: r + 1])] for r in range(requests)]
    tail_pages = torch.full((len(pos), max(1, max(ncomp))), -1, dtype=dtypes.i32)
    tail_len, toks = [], []
    for t, p in enumerate(pos):
        r = t // num_draft
        c = (p + 1) // CRATIO if kind == "hca" else 0
        tail_len.append(c)
        tail_pages[t, :c] = torch.tensor(lists[r][:c], dtype=dtypes.i32)
        swa = [r * RING + x % RING for x in range(max(0, p - WIN + 1), p + 1)]
        toks.append(swa + [x + swa_pages for x in lists[r][:c]])
    q, q_rope = pack_rows(len(pos) * H, gen)
    hca = kind == "hca"
    return {
        "q": q.view(-1, H, ROW),
        "q_rope": q_rope.view(-1, H, ROPE),
        "kv": kv,
        "kv_rope": kv_rope,
        "sink": torch.randn(H, generator=gen, device="cuda"),
        "meta": (
            torch.tensor([t // num_draft for t in range(len(pos))], dtype=dtypes.i32),
            torch.tensor(pos),
            torch.tensor(tail_len, dtype=dtypes.i32) if hca else None,
            tail_pages if hca else None,
        ),
        "swa_pages": swa_pages,
        "qo_indptr": torch.arange(len(pos) + 1, dtype=dtypes.i32),
        "kv_indptr": torch.tensor([0] + [len(x) for x in toks])
        .cumsum(0)
        .to(dtypes.i32),
        "kv_indices": torch.tensor(list(itertools.chain(*toks)), dtype=dtypes.i32),
    }


def grouped_meta_ref(slot, pos, tail_len, tail_pages, win, ring_stride, swa_pages, nd):
    """Torch reference of flydsl_mla_v4_grouped_verify_meta."""
    slot, pos = slot.tolist(), pos.tolist()
    tl_ = tail_len.tolist() if tail_len is not None else [0] * len(pos)
    idx, kvp, bnd = [], [0], []
    for g in range(len(pos) // nd):
        last = g * nd + nd - 1
        p, c = pos[last], tl_[last]
        n = min(p + 1, win + nd - 1)
        if c:
            idx += [
                x + swa_pages if x >= 0 else -1 for x in tail_pages[last, :c].tolist()
            ]
        idx += [
            slot[last] * ring_stride + (p - n + 1 + i) % ring_stride for i in range(n)
        ]
        s0 = p - n + 1
        for t in range(g * nd, last + 1):
            bnd.append(
                [0, tl_[t], c + max(0, pos[t] - win + 1 - s0), c + pos[t] - s0 + 1]
            )
        kvp.append(len(idx))
    i32 = dtypes.i32
    return (
        torch.tensor(idx, dtype=i32),
        torch.tensor(kvp, dtype=i32),
        torch.tensor(bnd, dtype=i32),
    )


@benchmark()
def test_mla_v4_decode_grouped(kind, num_draft, requests, epi):
    d = make_dspark(num_draft, kind, requests, seed=num_draft + requests)
    N, H = d["q"].shape[:2]
    ref = run_torch(**d)
    idx, kvp, qop, bnd = flydsl_mla_v4_grouped_verify_meta(
        *d["meta"],
        win=WIN,
        ring_stride=RING,
        swa_pages=d["swa_pages"],
        num_draft=num_draft,
    )
    r_idx, r_kvp, r_bnd = grouped_meta_ref(
        *d["meta"], WIN, RING, d["swa_pages"], num_draft
    )
    L = int(r_kvp[-1])
    assert (
        torch.equal(kvp, r_kvp)
        and torch.equal(bnd, r_bnd)
        and torch.equal(idx[:L], r_idx)
    )
    assert torch.equal(qop, torch.arange(requests + 1, dtype=dtypes.i32) * num_draft)
    grp = dict(d, qo_indptr=qop, kv_indptr=kvp, kv_indices=idx)
    cfg = fly_v4._plan(
        H, num_draft, requests, epi, "hca" if kind == "hca" else "swa", grouped=True
    )
    ret = {
        "gfx": get_gfx(),
        "layout": cfg.LAYOUT + ("q16" if cfg.Q16 else ""),
        "S": cfg.S,
    }
    # a reversed interval (x1 < x0) is empty
    b_rev, b_emp = bnd.clone(), bnd.clone()
    b_rev[3] = torch.tensor([L + 7, 1, 97, 2], dtype=dtypes.i32)
    b_emp[3] = 0
    if epi == "bf16":
        out_g, out_t = [torch.empty((N, H, DV), dtype=dtypes.bf16) for _ in range(2)]
        hint = "hca" if kind == "hca" else "swa"

        def grouped(b=bnd):
            return decode(grp, out_g, num_draft, hint=hint, q_kv_bounds=b)

        candidates = {
            "flydsl_grouped": grouped,
            "flydsl_per_token": lambda: decode(d, out_t, 1, hint=hint),
        }
        for name, fn in candidates.items():
            o, us = run_perftest(fn)
            ret[f"{name} us"] = us
            ret[f"{name} err"] = checkAllclose(
                ref.float(), o.float(), msg=f"{name}: vs fp64"
            )
            ret[f"{name} rel_l2"] = rel_l2(o, ref)
            assert ret[f"{name} rel_l2"] < 3e-3, ret
        g0 = out_g.clone()
        e0 = grouped(b_emp).clone()
        assert torch.isnan(e0[3].float()).all() and torch.equal(
            bits(grouped(b_rev)), bits(e0)
        )
        out_g.zero_()
        graph_replay(grouped)
        assert torch.equal(bits(out_g), bits(g0))
        return ret
    freqs = rope_freqs()
    positions = d["meta"][1]
    ref_f = inv_rope(ref, positions, freqs)
    fa = (d["q"], d["q_rope"], d["kv"], d["kv_rope"])
    cr = 128 if kind == "hca" else 0
    res = {}

    def fused_grouped(b=bnd):
        res["x"] = flydsl_mla_v4_decode_fused(
            *fa,
            kvp,
            idx,
            d["sink"],
            positions,
            freqs,
            cr,
            qo_indptr=qop,
            max_seqlen_q=num_draft,
            q_kv_bounds=b,
        )
        return res["x"]

    def fused_tok():
        return flydsl_mla_v4_decode_fused(
            *fa,
            d["kv_indptr"],
            d["kv_indices"],
            d["sink"],
            positions,
            freqs,
            cr,
        )

    for name, fn in (
        ("flydsl_fused_grouped", fused_grouped),
        ("flydsl_fused_per_token", fused_tok),
    ):
        (xq, xs), us = run_perftest(fn)
        err, rel = check_mxfp8(ref_f, xq, xs, msg=f"{name}: vs fp64")
        ret[f"{name} us"] = us
        ret[f"{name} err"] = err
        ret[f"{name} rel_l2"] = rel
        assert rel < 0.05, ret
    q8, s8 = [x.view(torch.uint8).clone() for x in fused_grouped()]
    e0 = [x.view(torch.uint8).clone() for x in fused_grouped(b_emp)]
    assert (e0[1][3] == 0xFF).all()
    assert all(
        torch.equal(x.view(torch.uint8), y) for x, y in zip(fused_grouped(b_rev), e0)
    )
    graph_replay(fused_grouped)
    assert torch.equal(res["x"][0].view(torch.uint8), q8) and torch.equal(
        res["x"][1], s8
    )
    return ret


def test_contracts():
    """Unsupported pairs rejected; B = 0 is a no-op; > 4096 merge groups run unsplit;
    graphs sharing one stream's workspace replay interleaved; a graph captured on a
    fresh stream takes pre-zeroed protocol slots (no memset in the graph)."""
    for H, msq in ((32, 2), (64, 4), (16, 3), (8, 1)):
        assert not flydsl_mla_v4_decode_supported(H, msq)
    assert flydsl_mla_v4_decode_supported(16, 7, grouped=True)
    inp, _ = make_inputs(16, 1, 1, 128, False)
    empty = dict(
        inp,
        q=inp["q"][:0],
        q_rope=inp["q_rope"][:0],
        qo_indptr=inp["qo_indptr"][:1],
        kv_indptr=inp["kv_indptr"][:1],
    )
    decode(empty, torch.empty((0, 16, DV), dtype=dtypes.bf16), 1)
    B = 4097
    assert fly_v4._plan_for(16, 1, B, "bf16", None, 256, False).S > 1
    assert fly_v4._plan(16, 1, B).S == 1
    inp, kv_lens = make_inputs(16, 2, B, 300, True)
    inp = dict(
        inp, q=inp["q"][::2].contiguous(), q_rope=inp["q_rope"][::2].contiguous()
    )
    inp["qo_indptr"] = torch.arange(B + 1, dtype=dtypes.i32)
    out = torch.empty((B, 16, DV), dtype=dtypes.bf16)
    decode(inp, out, 1)
    seqs = [b for b in range(0, B, 97) if kv_lens[b]]
    sub = dict(inp)
    sub["q"], sub["q_rope"] = [inp[k][seqs] for k in ("q", "q_rope")]
    sub["qo_indptr"] = torch.arange(len(seqs) + 1, dtype=dtypes.i32)
    kvi = inp["kv_indptr"].tolist()
    sub["kv_indptr"] = (
        torch.tensor([0] + [kv_lens[b] for b in seqs]).cumsum(0).to(dtypes.i32)
    )
    sub["kv_indices"] = torch.cat(
        [inp["kv_indices"][kvi[b] : kvi[b + 1]] for b in seqs]
    )
    assert rel_l2(out[seqs], run_torch(**sub)) < 3e-3
    cases = []
    for batch in (3, 37, 200):
        inp, kv_lens = make_inputs(16, 1, batch, 659, True, seed=batch)
        out = torch.empty((batch, 16, DV), dtype=dtypes.bf16)
        decode(inp, out, 1)
        cases.append((inp, out, out.clone()))
    st = torch.cuda.Stream()
    graphs = [
        graph_replay(lambda i=i, o=o: decode(i, o, 1), 1, st) for i, o, _ in cases
    ]
    for n in range(3000):
        graphs[n % 3].replay()
    torch.cuda.synchronize()
    assert all(torch.equal(bits(o), bits(r)) for _, o, r in cases)
    inp, out, ref = cases[1]
    spare = fly_v4._HDR_SPARE[out.device.index].data_ptr()
    # high-priority pool: a default-priority Stream() may reuse a handle that already owns a workspace
    s3, g = torch.cuda.Stream(priority=-1), torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s3):
        decode(inp, out, 1)
    with torch.cuda.stream(s3):
        assert fly_v4._workspace(out.device).hdr.data_ptr() == spare
    out.zero_()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), bits(ref))


def summarize(name, rows):
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl v4 MLA decode unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--hmsq",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(16, 1), (16, 2), (16, 4), (32, 1), (64, 1), (128, 1)],
        help="(num_heads, max_seqlen_q)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 5, 9, 17, 37, 100, 129, 200, 300],
        help="sequences / decode tokens",
    )
    parser.add_argument(
        "--kv_len",
        type=int,
        nargs="*",
        default=[128, 1152, 4096],
        help="kv rows per sequence",
    )
    parser.add_argument(
        "--ragged",
        type=int,
        nargs="*",
        default=[0, 1],
        help="ragged kv lengths (0 / 1)",
    )
    parser.add_argument(
        "--hint",
        type=str,
        nargs="*",
        default=["none", "csa", "hca", "swa"],
        help="stream hint (H16 msq1)",
    )
    parser.add_argument(
        "--fused",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (16, 4, 1152),
            (16, 128, 659),
            (16, 0, 128),
            (64, 128, 659),
            (64, 0, 128),
            (128, 128, 659),
        ],
        help="fused: (H, compress_ratio, kv_len)",
    )
    parser.add_argument(
        "--fused_T",
        type=int,
        nargs="*",
        default=[3, 8, 40, 42, 100, 200, 512],
        help="fused: decode tokens",
    )
    parser.add_argument(
        "--num_draft",
        type=int,
        nargs="*",
        default=[7, 3],
        help="DSpark drafts per request",
    )
    parser.add_argument(
        "--requests", type=int, nargs="*", default=[14, 32, 72], help="DSpark requests"
    )
    args = parser.parse_args()

    rows = []
    for (H, msq), batch, kv_len, ragged in itertools.product(
        args.hmsq, args.batch, args.kv_len, args.ragged
    ):
        hints = (
            [None if h == "none" else h for h in args.hint]
            if (H, msq) == (16, 1)
            else [None]
        )
        for hint in hints if kv_len != 4096 else [None]:
            rows.append(test_mla_v4_decode(H, msq, batch, kv_len, ragged, hint))
    if (16, 1) in args.hmsq:
        # one long stream among short ones: the key-balanced grid at short runs
        rows.append(test_mla_v4_decode(16, 1, 96, 256, 2))
    summarize("mla_decode_fwd_v4_nm", rows)
    rows = []
    for (H, cr, kv_len), T, ragged in itertools.product(
        args.fused, args.fused_T, args.ragged
    ):
        rows.append(test_mla_v4_decode_fused(T, H, cr, kv_len, ragged))
    summarize("flydsl_mla_v4_decode_fused", rows)
    for epi in ("bf16", "invrope_mxfp8"):
        rows = []
        for kind, nd, req in itertools.product(
            ("hca", "swa"), args.num_draft, args.requests
        ):
            rows.append(test_mla_v4_decode_grouped(kind, nd, req, epi))
        summarize(f"grouped verify (DSpark, {epi})", rows)
    test_contracts()


if __name__ == "__main__":
    main()
