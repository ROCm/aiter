# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL MXFP4 indexer top-512 (``aiter.ops.flydsl.mxfp4_indexer_topk``)
against an fp64 torch reference of ``paged_mxfp4_mqa_logits``' logits and the
exact top-512 selection.

Tables:

- ``topk``: dense and candidate-source layers at 4k-262k contexts and 18-768
  decode rows, 1-8 rows a request, 64- and 128-entry pages: every row a valid
  top-512 of the fp64 logits (up to fp32 rounding at the 512th rank), most
  rows equal to the exact selection, the source's candidate lists; timed
  against ``paged_mxfp4_mqa_logits`` + aiter's ``top_k_per_row_decode``.
- ``consumer``: the candidate consumer (positions inside the candidate pool,
  the pool's top-512).
- ``paths``: the token, block, long-row and split-consumer paths bit for bit
  against each other (tier edges, ties, ragged lists, candidate lists).
- ``cases``: zero rows, short and empty rows, ties, repeat runs, block
  maxima, the ``paged_mxfp4_mqa_logits`` contract on the same cache, vLLM's
  call form, CUDA-graph replay with new inputs (and after the shared scratch
  grew), caller workspaces, 0.5M-1M entry rows, unsupported shapes.

Outputs are pre-filled with -9, so an unwritten element fails the checks.

    python3 op_tests/test_flydsl_mxfp4_indexer_topk.py
    python3 op_tests/test_flydsl_mxfp4_indexer_topk.py -c 131072 -b 32 --layer dense
"""

import argparse
import contextlib
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]  # FP4 scaled MFMA
DEV = "cuda"
H, D, ENTRY = 32, 128, 68
TOPK, CANDIDATES, BLOCK = 512, 2048, 8
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-x for x in E2M1]


def _ix():
    from aiter.ops.flydsl import mxfp4_indexer_topk

    return mxfp4_indexer_topk


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


# ----------------------------------------------------------------------------
# Torch reference (fp64)
# ----------------------------------------------------------------------------


def _codes(packed):
    lut = torch.tensor(E2M1, dtype=torch.float64, device=packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    return torch.stack([lut[lo], lut[hi]], -1).flatten(-2)


def dequant_q(q, q_scale):
    """[rows, 32, 128] fp64."""
    if q_scale.dtype == torch.int32:
        q_scale = q_scale.contiguous().view(torch.uint8).view(*q_scale.shape, 4)
    s = torch.exp2(q_scale.double() - 127)[..., None]
    s = s.expand(*q_scale.shape, 32).flatten(-2)
    return _codes(q) * s


def dequant_keys(pages, P):
    """[n * P, 128] fp64 of pages [n, P, 68]: a 32-entry group stores its
    values as [chunk][entry][16 B] and its scales as
    [scale % 2][entry][scale // 2]."""
    n = pages.shape[0]
    flat = pages.reshape(n, P * ENTRY)
    codes = flat[:, : P * 64].reshape(n, P // 32, 4, 32, 16)
    codes = codes.permute(0, 1, 3, 2, 4).reshape(n, P, 64)
    scales = flat[:, P * 64 :].reshape(n, P // 32, 2, 32, 2)
    scales = scales.permute(0, 1, 3, 4, 2).reshape(n, P, 4)
    s = torch.exp2(scales.double() - 127)[..., None].expand(n, P, 4, 32)
    return (_codes(codes) * s.reshape(n, P, D)).reshape(n * P, D)


def logits_reference(q, q_scale, w, cache, bt, ends, nn, P):
    """Per row the fp64 logits [row_end]: sum_h w_h relu(q_h . k_t)."""
    Q = dequant_q(q, q_scale)
    pool = cache.reshape(cache.shape[0], P, ENTRY)
    out = []
    for req in range(bt.shape[0]):
        rows = range(req * nn, (req + 1) * nn)
        length = max(max(int(ends[r]) for r in rows), 0)
        npg = -(-length // P)
        keys = dequant_keys(pool[bt[req, :npg].long()], P)[:length]
        for r in rows:
            e = max(int(ends[r]), 0)
            x = torch.einsum("hd,td->ht", Q[r], keys[:e])
            out.append((torch.relu(x) * w[r].double()[:, None]).sum(0))
    return out


def select_reference(logits, k=TOPK):
    """Positions of the k best (logit descending, position ascending), sorted
    ascending, -1 padded."""
    order = torch.sort(logits, descending=True, stable=True).indices
    order = order[: min(k, logits.numel())]
    out = torch.full((k,), -1, dtype=torch.int32, device=logits.device)
    out[: order.numel()] = order.sort().values.int()
    return out


def block_scores_reference(logits):
    n = logits.numel()
    pad = torch.full(
        (-(-n // BLOCK) * BLOCK,), float("-inf"), dtype=logits.dtype, device=DEV
    )
    pad[:n] = logits
    return pad.view(-1, BLOCK).amax(1)


def candidates_reference(logits, k=CANDIDATES):
    """The newest key's block and the k - 1 best others by block maximum (ties
    to the lower block), ids ascending, -1 padded."""
    scores = block_scores_reference(logits)
    n = scores.numel()
    out = torch.full((k,), -1, dtype=torch.int32, device=DEV)
    if n == 0:
        return out
    pin = n - 1
    rest = scores[:pin]
    order = torch.sort(rest, descending=True, stable=True).indices[: k - 1]
    ids = torch.cat([order, torch.tensor([pin], device=DEV)]).sort().values.int()
    out[: ids.numel()] = ids
    return out


def ragged_reference(topk, nn, slot_bt, bs, valid):
    rows = topk.shape[0]
    lens = torch.where(valid.bool(), (topk >= 0).sum(1), torch.zeros_like(valid))
    lens = lens.int()
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.cumsum(lens, 0)
    slots = torch.zeros(rows * TOPK, dtype=torch.int32, device=DEV)
    for r in range(rows):
        p = topk[r][topk[r] >= 0].long()
        if int(lens[r]) == 0:
            continue
        pg = slot_bt[r // nn].long()[p // bs]
        a = int(indptr[r])
        slots[a : a + p.numel()] = (pg * bs + p % bs).int()
    return slots, indptr, lens


def topk_matches(got, logits, k=TOPK, rtol=1e-5):
    """None if ``got`` is a valid top-k of ``logits`` up to fp32 rounding:
    right count, ascending, inside the row, every chosen logit at least the
    k-th best minus the tolerance and every other one at most it plus the
    tolerance; else a message."""
    n = logits.numel()
    want = min(k, n)
    g = got[got >= 0].long()
    if g.numel() != want or bool((got[want:] != -1).any()):
        return f"count {g.numel()} vs {want}, or the -1 tail is wrong"
    if bool((g[1:] <= g[:-1]).any()):
        return "positions not ascending"
    if want == 0:
        return None
    if int(g.max()) >= n:
        return "position outside the row"
    srt = torch.sort(logits, descending=True).values
    thr = srt[want - 1]
    tol = rtol * float(srt.abs().max()) + 1e-300
    mask = torch.zeros(n, dtype=torch.bool, device=DEV)
    mask[g] = True
    if bool((logits[mask] < thr - tol).any()):
        return "a chosen logit is below the k-th"
    if bool((logits[~mask] > thr + tol).any()):
        return "a better logit was left out"
    return None


def selected_values(out, logits):
    """[rows, 512] fp32: per row the logits of the chosen positions and of the
    exact top-512, each sorted descending and divided by the row's largest
    |logit| (0 padded), for ``checkAllclose``."""
    rows = len(logits)
    got = torch.zeros(rows, TOPK, dtype=torch.float64, device=DEV)
    ref = torch.zeros_like(got)
    for r, x in enumerate(logits):
        if x.numel() == 0:
            continue
        scale = float(x.abs().max()) or 1.0
        g = out[r][out[r] >= 0].long().clamp(max=x.numel() - 1)
        gv = torch.sort(x[g], descending=True).values / scale
        rv = torch.sort(x, descending=True).values[:TOPK] / scale
        got[r, : gv.numel()] = gv
        ref[r, : rv.numel()] = rv
    return ref.float(), got.float()


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------


def random_cache(num_pages, P, seed=0, scale_lo=120, scale_hi=127):
    g = torch.Generator(device=DEV).manual_seed(seed)
    cache = torch.randint(
        0, 256, (num_pages, P * ENTRY), dtype=torch.uint8, device=DEV, generator=g
    )
    cache[:, P * 64 :] = torch.randint(
        scale_lo,
        scale_hi,
        (num_pages, P * 4),
        dtype=torch.uint8,
        device=DEV,
        generator=g,
    )
    return cache.view(num_pages, P, ENTRY)


def random_inputs(lengths, nn, P, seed=0, extra_pages=0):
    """Requests of ``nn`` rows (row n of a request ends at length - nn + n +
    1), pages scattered over the cache: (q, q_scale, w, cache, bt, ends)."""
    B = len(lengths)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    need = [-(-max(c, 0) // P) for c in lengths]
    total = max(sum(need) + extra_pages, 1)
    cache = random_cache(total, P, seed)
    perm = torch.randperm(total, device=DEV, generator=g).int()
    bt = torch.zeros(B, max(need + [1]), dtype=torch.int32, device=DEV)
    o = 0
    for i, n in enumerate(need):
        bt[i, :n] = perm[o : o + n]
        o += n
    T = B * nn
    q = torch.randint(0, 256, (T, H, 64), dtype=torch.uint8, device=DEV, generator=g)
    qs = torch.randint(120, 127, (T, H, 4), dtype=torch.uint8, device=DEV, generator=g)
    w = torch.randn(T, H, device=DEV, generator=g) * 0.1
    cl = torch.tensor(lengths, dtype=torch.int32, device=DEV).view(-1)
    ends = cl[:, None] - nn + torch.arange(nn, device=DEV)[None] + 1
    ends = ends.clamp(min=0).reshape(-1).int().contiguous()
    return q, qs, w, cache, bt, ends


def lengths_of(ctx, reqs, lo=0.5):
    return [int(ctx * (lo + (1 - lo) * (i + 0.5) / reqs)) for i in range(reqs)]


def spread(ctx, reqs, lo=0.55):
    out = [int(ctx * (lo + (1 - lo) * (i + 1) / reqs)) for i in range(reqs)]
    out[-1] = ctx
    return out


def ragged_of(ix, rows, nreq, ctx, bs, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    slot_bt = torch.randint(
        0, 5000, (nreq, -(-ctx // bs) + 3), dtype=torch.int32, device=DEV, generator=g
    )
    valid = (torch.arange(rows, device=DEV) % 5 != 2).int()
    return ix.Mxfp4IndexerRagged(
        torch.full((rows * TOPK,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows + 1,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows,), -9, dtype=torch.int32, device=DEV),
        valid,
        slot_bt,
        bs,
    )


def run(ix, inputs, nn, ctx, *, cand=False, ragged=None, workspace=None):
    q, qs, w, cache, bt, ends = inputs
    out = torch.full((q.shape[0], TOPK), -9, dtype=torch.int32, device=DEV)
    c = None
    if cand:
        c = torch.full((q.shape[0], CANDIDATES), -9, dtype=torch.int32, device=DEV)
    ix.flydsl_mxfp4_indexer_topk(
        q,
        qs,
        w,
        cache,
        bt,
        ends,
        out,
        next_n=nn,
        max_ctx=ctx,
        candidates_out=c,
        ragged=ragged,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    return out, c


def consume(ix, inputs, cand, nn, ctx, *, ragged=None, workspace=None):
    q, qs, w, cache, bt, ends = inputs
    out = torch.full((q.shape[0], TOPK), -9, dtype=torch.int32, device=DEV)
    ix.flydsl_mxfp4_indexer_candidate_topk(
        q,
        qs,
        w,
        cache,
        bt,
        ends,
        cand,
        out,
        next_n=nn,
        max_ctx=ctx,
        ragged=ragged,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    return out


def check_rows(out, inputs, nn, P, equal_frac=0.9, logits=None):
    """Every row a valid top-512 of the fp64 logits; most equal to the exact
    selection (rows whose 512th and 513th logits are within fp32 rounding may
    differ). Returns the logits and the mismatch fraction of the chosen
    logits against the exact top-512's."""
    if logits is None:
        logits = logits_reference(*inputs, nn, P)
    exact = 0
    for r, x in enumerate(logits):
        msg = topk_matches(out[r], x)
        assert msg is None, (r, int(inputs[5][r]), msg)
        exact += int(bool((out[r] == select_reference(x)).all()))
    assert exact >= equal_frac * len(logits), (exact, len(logits))
    ref, got = selected_values(out, logits)
    err = checkAllclose(ref, got, rtol=0, atol=1e-5, msg="chosen logits")
    assert err == 0, err
    return logits, err


def check_candidates(cand_row, x):
    scores = block_scores_reference(x)
    n = scores.numel()
    got = cand_row[cand_row >= 0].long()
    assert got.numel() == min(CANDIDATES, n), (got.numel(), n)
    assert bool((cand_row[got.numel() :] == -1).all())
    assert bool((got[1:] > got[:-1]).all())
    if n == 0:
        return
    assert bool((got == n - 1).any()), "the newest key's block"
    rest = got[got != n - 1]
    compact = torch.full((CANDIDATES,), -1, dtype=torch.int32, device=DEV)
    compact[: rest.numel()] = rest.int()
    if n > 1:
        assert topk_matches(compact, scores[: n - 1], CANDIDATES - 1) is None


def check_consumer(out_row, cand_row, x):
    """The row's 512 best among its candidate blocks' tokens."""
    ids = cand_row[cand_row >= 0].long()
    pos = (ids[:, None] * BLOCK + torch.arange(BLOCK, device=DEV)).flatten()
    pos = pos[pos < x.numel()]
    sel = out_row[out_row >= 0].long()
    at = torch.searchsorted(pos, sel)
    assert bool((pos[at.clamp(max=pos.numel() - 1)] == sel).all()), "outside pool"
    compact = torch.full((TOPK,), -1, dtype=torch.int32, device=DEV)
    compact[: at.numel()] = at.int()
    assert topk_matches(compact, x[pos]) is None


def check_ragged(rg, out, nn):
    slots, indptr, lens = ragged_reference(
        out, nn, rg.slot_block_table, rg.slot_block_size, rg.is_valid
    )
    assert torch.equal(rg.lens, lens) and torch.equal(rg.indptr, indptr)
    n = int(indptr[-1])
    assert torch.equal(rg.slots[:n], slots[:n])


def bit_mismatch(a, b, msg):
    """Fraction of differing elements of two int32 tensors (0 when equal)."""
    err = checkAllclose(a.float(), b.float(), rtol=0, atol=0, msg=msg)
    assert torch.equal(a, b), (msg, err)
    return err


def roofline(lengths, nn, us):
    """TFLOPS (2 * heads * dims per (row, key)) and TB/s (keys read once a
    request, q / scales / weights, the 512 positions) of one call."""
    keys = sum(max(c, 0) for c in lengths)
    flops = 2 * H * D * nn * keys
    nbytes = keys * ENTRY + len(lengths) * nn * (H * 64 + H * 8 + TOPK * 4)
    return flops / us / 1e6, nbytes / us / 1e6


# ----------------------------------------------------------------------------
# Dense and candidate-source layers
# ----------------------------------------------------------------------------


def baseline_topk(inputs, nn, P, ctx, out):
    """``paged_mxfp4_mqa_logits`` (Gluon) + aiter's ``top_k_per_row_decode``:
    the chain this op replaces (positions in top-k order, not sorted)."""
    from aiter.ops.topk import top_k_per_row_decode
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa

    q, qs, w, cache, bt, ends = inputs
    reqs, rows = bt.shape[0], q.shape[0]
    width = -(-ctx // 128) * 128
    logits = torch.empty(rows, width, dtype=torch.float32, device=DEV)
    cl = ends.view(reqs, nn)[:, -1].contiguous()

    def call():
        pa.paged_mxfp4_mqa_logits(
            q.view(reqs, nn, H, 64),
            qs.view(reqs, nn, H, 4),
            cache.view(-1, P, 1, ENTRY),
            w,
            cl,
            bt,
            width,
            out_logits=logits,
            clean_logits=False,
            row_ends=ends,
        )
        top_k_per_row_decode(
            logits, 1, ends, out, rows, logits.stride(0), logits.stride(1), k=TOPK
        )
        return out

    return call


@benchmark()
def test_mxfp4_indexer_topk(ctx, reqs, next_n, page, layer):
    """``layer`` dense or source (with ``candidates_out``); requests of
    0.5-1x ``ctx`` entries, ``max_ctx`` = ``ctx``."""
    ix = _ix()
    lens = lengths_of(ctx, reqs)
    inputs = random_inputs(lens, next_n, page, seed=reqs + ctx + next_n + page)
    rows = inputs[0].shape[0]
    source = layer == "source"
    out = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
    cand = torch.full((rows, CANDIDATES), -9, dtype=torch.int32, device=DEV)

    def flydsl():
        ix.flydsl_mxfp4_indexer_topk(
            *inputs,
            out,
            next_n=next_n,
            max_ctx=ctx,
            candidates_out=cand if source else None,
        )
        return out

    candidates = {"flydsl": flydsl}
    if not source:
        theirs = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
        candidates["gluon_topk"] = baseline_topk(inputs, next_n, page, ctx, theirs)

    logits = logits_reference(*inputs, next_n, page)
    ret = {"gfx": get_gfx_runtime(), "rows": rows}
    for name, fn in candidates.items():
        got, us = run_perftest(fn, num_iters=20, num_warmup=2)
        torch.cuda.synchronize()
        if name == "flydsl":
            _, err = check_rows(got, inputs, next_n, page, logits=logits)
            if source:
                same = 0
                for r, x in enumerate(logits):
                    check_candidates(cand[r], x)
                    same += int(torch.equal(cand[r], candidates_reference(x)))
                assert same >= 0.9 * rows, (same, rows)
        else:
            srt = torch.where(got >= 0, got, torch.iinfo(torch.int32).max)
            srt = srt.sort(1).values
            srt[srt == torch.iinfo(torch.int32).max] = -1
            for r, x in enumerate(logits):
                assert topk_matches(srt[r], x) is None, (name, r)
            ref, sel = selected_values(srt, logits)
            err = checkAllclose(ref, sel, rtol=0, atol=1e-5, msg=f"{name} chosen")
            assert err == 0, err
        tflops, tbs = roofline(lens, next_n, us)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = tflops
        ret[f"{name} TB/s"] = tbs
        ret[f"{name} err"] = err
    return ret


# ----------------------------------------------------------------------------
# Candidate consumer
# ----------------------------------------------------------------------------


@benchmark()
def test_mxfp4_indexer_candidate_topk(ctx, reqs, next_n, page):
    """The top-512 among the reference candidate blocks (every third row's
    pool cut to 100 blocks)."""
    ix = _ix()
    lens = lengths_of(ctx, reqs)
    inputs = random_inputs(lens, next_n, page, seed=reqs * 7 + ctx)
    rows = inputs[0].shape[0]
    logits = logits_reference(*inputs, next_n, page)
    cand = torch.stack([candidates_reference(x) for x in logits]).contiguous()
    cand[::3, 100:] = -1
    out = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)

    def flydsl():
        ix.flydsl_mxfp4_indexer_candidate_topk(
            *inputs, cand, out, next_n=next_n, max_ctx=ctx
        )
        return out

    got, us = run_perftest(flydsl, num_iters=20, num_warmup=2)
    torch.cuda.synchronize()
    for r, x in enumerate(logits):
        check_consumer(got[r], cand[r], x)
    # the chosen logits against the exact top-512 of each row's pool (indices
    # into the pool's positions)
    pools = []
    at = torch.full_like(got, -1)
    for r, x in enumerate(logits):
        ids = cand[r][cand[r] >= 0].long()
        pos = (ids[:, None] * BLOCK + torch.arange(BLOCK, device=DEV)).flatten()
        pos = pos[pos < x.numel()]
        sel = got[r][got[r] >= 0].long()
        at[r, : sel.numel()] = torch.searchsorted(pos, sel).int()
        pools.append(x[pos])
    ref, sel = selected_values(at, pools)
    err = checkAllclose(ref, sel, rtol=0, atol=1e-5, msg="consumer chosen")
    assert err == 0, err
    tflops, tbs = roofline([min(c, CANDIDATES * BLOCK) for c in lens], next_n, us)
    return {
        "gfx": get_gfx_runtime(),
        "rows": rows,
        "split": ix._candidate_split(rows, ctx),
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbs,
        "flydsl err": err,
    }


# ----------------------------------------------------------------------------
# Path equivalence (bit for bit)
# ----------------------------------------------------------------------------

# (path, max_ctx, lengths, next_n, page, ties)
PATH_CASES = [
    ("token", 4096, [4096, 3000, 513, 100, 1, 0, 4000], 6, 64, ""),
    ("token", 4097, [4097, 600, 4000], 6, 64, "equal_pages"),  # the 8192 tier
    ("token", 8192, [8192, 100, 4300, 7000, 512, 511], 3, 128, ""),
    ("token", 8193, [8193, 3000, 5555], 6, 128, "zero_q"),  # the 16384 tier
    ("token", 16384, [16384, 9000, 5000, 8193, 4097, 700, 12345], 6, 64, ""),
    ("token", 16384, [16000, 16384, 12000, 3000, 20, 7777, 9999, 15000], 8, 128, ""),
    ("token", 4096, [4096, 2048, 2], 1, 64, "equal_pages"),
    ("split", 131072, [131072, 90000, 40000, 20000], 6, 128, ""),
    ("split", 65536, [65536, 30000, 17000, 9000, 2000, 7, 0], 6, 64, ""),
    ("split", 20000, [20000, 19000, 100], 3, 128, ""),
    ("split", 131072, [131000, 120000], 8, 128, ""),
    (
        "long",
        90000,
        [90000, 70000, 40000, 33000, 32768, 20000, 5000, 100, 0, 1],
        6,
        128,
        "",
    ),
    ("long", 100000, [100000, 64000, 31000, 12000], 3, 64, ""),
    ("long", 100000, [100000, 48000, 9000, 40], 6, 128, "equal_pages"),
    ("long", 100000, [100000, 48000, 9000, 40], 6, 64, "zero_q"),
    ("long", 140000, [140000, 100000, 66000, 3000, 12], 6, 128, ""),  # two segments
    ("long", 66000, [66000, 50000], 8, 128, ""),
    ("long", 70000, [70000, 4000], 1, 64, ""),
]


def _path_outputs(ix, inputs, nn, ctx, nreq, consumer_cand=None):
    """(dense positions, source positions, candidate lists, consumer
    positions, ragged lists of the dense call)."""
    rows = inputs[0].shape[0]
    rg = ragged_of(ix, rows, nreq, ctx, 128, 3)
    out, _ = run(ix, inputs, nn, ctx, ragged=rg)
    src, cand = run(ix, inputs, nn, ctx, cand=True)
    pool = cand if consumer_cand is None else consumer_cand
    rg_c = ragged_of(ix, rows, nreq, ctx, 128, 4)
    cons = consume(ix, inputs, pool, nn, ctx, ragged=rg_c)
    return out, src, cand, cons, rg, rg_c


@benchmark()
def test_mxfp4_indexer_paths(case):
    """``token``: the token path (max_ctx <= 16384) against the block path;
    ``split``: the split candidate consumer against the one-workgroup one;
    ``long``: the long-row path forced at contexts the block path serves."""
    ix = _ix()
    path, ctx, lens, nn, P, ties = PATH_CASES[case]
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=ctx + nn + len(lens))
    if ties == "equal_pages":
        cache[:] = cache[0:1]
    elif ties == "zero_q":
        q[::2] = 0
    inputs = (q, qs, w, cache, bt, ends)
    cut = None
    if path == "split":
        _, cut = run(ix, inputs, nn, ctx, cand=True)
        cut[::3, 300:] = -1  # a short pool on some rows
    a = _path_outputs(ix, inputs, nn, ctx, len(lens), cut)
    if path == "token":
        assert ix._tokens_tier(ctx) is not None
        ctxm = patched(ix, "_tokens_tier", lambda max_ctx: None)
    elif path == "split":
        assert ix._candidate_split(inputs[0].shape[0], ctx)
        ctxm = patched(ix, "CAND_SPLIT_ROWS", 0)
    else:
        ctxm = patched(ix, "BLOCK_CTX_MAX", 16384)
    with ctxm:
        if path == "long":
            assert ix._blocks_tier(ix.BLOCK_CTX_MAX) * BLOCK < ctx
        b = _path_outputs(ix, inputs, nn, ctx, len(lens), cut)
    err = 0.0
    names = ("dense", "source", "candidates", "consumer")
    for name, x, y in zip(names, a[:4], b[:4]):
        err = max(err, bit_mismatch(x, y, f"{path} {name}"))
    for rx, ry in ((a[4], b[4]), (a[5], b[5])):
        for f in ("slots", "indptr", "lens"):
            err = max(err, bit_mismatch(getattr(rx, f), getattr(ry, f), f))
    check_ragged(a[4], a[0], nn)
    check_ragged(a[5], a[3], nn)
    check_rows(a[0], inputs, nn, P, equal_frac=0.0)
    return {"path": path, "max_ctx": ctx, "rows": q.shape[0], "err": err}


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


def case_zero_rows(ix):
    """A step with no decode rows (a DP rank without requests) launches
    nothing and leaves the device usable."""
    q, qs, w, cache, bt, ends = random_inputs([100], 6, 64)
    z = (q[:0], qs[:0], w[:0], cache, bt[:0], ends[:0])
    out = torch.empty(0, TOPK, dtype=torch.int32, device=DEV)
    cand = torch.empty(0, CANDIDATES, dtype=torch.int32, device=DEV)
    for ctx in (4096, 65536, 1 << 20):
        ix.flydsl_mxfp4_indexer_topk(*z, out, next_n=6, max_ctx=ctx)
        ix.flydsl_mxfp4_indexer_topk(
            *z, out, next_n=6, max_ctx=ctx, candidates_out=cand
        )
        ix.flydsl_mxfp4_indexer_candidate_topk(*z, cand, out, next_n=6, max_ctx=ctx)
        s = ix.flydsl_mxfp4_indexer_block_scores(*z, next_n=6, max_ctx=ctx)
        assert s.shape[0] == 0
    x = torch.ones(4, device=DEV) * 2  # the next op still runs
    torch.cuda.synchronize()
    assert float(x.sum()) == 8.0


def case_short_and_empty_rows(ix):
    lens = [0, 1, 5, 6, 7, 100, 511, 512, 513, 1023, 4097]
    inputs = random_inputs(lens, 6, 128, seed=5)
    out, _ = run(ix, inputs, 6, 4100)
    check_rows(out, inputs, 6, 128)
    out, _ = run(ix, inputs, 6, 40000)  # the block path
    check_rows(out, inputs, 6, 128)


def case_ties(ix):
    for ctx in (3000, 30000):
        q, qs, w, cache, bt, ends = random_inputs([ctx, ctx // 10], 6, 128, seed=9)
        cache[:] = cache[0:1]  # equal logits at equal in-page offsets
        out, _ = run(ix, (q, qs, w, cache, bt, ends), 6, ctx)
        for r, x in enumerate(logits_reference(q, qs, w, cache, bt, ends, 6, 128)):
            assert torch.equal(out[r], select_reference(x)), r
        q.zero_()  # every logit 0: the first 512 positions
        out0, _ = run(ix, (q, qs, w, cache, bt, ends), 6, ctx)
        for r in range(out0.shape[0]):
            n = min(TOPK, int(ends[r]))
            assert torch.equal(out0[r, :n], torch.arange(n, device=DEV).int()), r
            assert bool((out0[r, n:] == -1).all()), r


def case_repeat_runs(ix):
    inputs = random_inputs(lengths_of(65536, 16), 6, 64, seed=11)
    a, _ = run(ix, inputs, 6, 65536)
    b, _ = run(ix, inputs, 6, 65536)
    assert torch.equal(a, b)


def case_block_scores(ix):
    q, qs, w, cache, bt, ends = random_inputs([2000, 40000, 777], 6, 128, seed=3)
    scores = ix.flydsl_mxfp4_indexer_block_scores(
        q, qs, w, cache, bt, ends, next_n=6, max_ctx=40000
    )
    torch.cuda.synchronize()
    for r, x in enumerate(logits_reference(q, qs, w, cache, bt, ends, 6, 128)):
        ref = block_scores_reference(x)
        got = scores[r, : ref.numel()].double()
        fin = torch.isfinite(ref)
        assert torch.equal(torch.isfinite(got), fin), r
        err = (got[fin] - ref[fin]).abs().max() if bool(fin.any()) else 0.0
        assert float(err) <= 1e-5 * float(ref[fin].abs().max()), r


def case_paged_mxfp4_mqa_logits_contract(ix):
    """The contract is aiter's paged_mxfp4_mqa_logits on the same cache: its
    logits agree with the reference, and its top-512 and this op's both are
    valid top-512 of the reference logits."""
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa

    for reqs, ctx, P in ((4, 20000, 128), (7, 60000, 64)):
        nn = 6
        inputs = random_inputs(lengths_of(ctx, reqs), nn, P, seed=17)
        q, qs, w, cache, bt, ends = inputs
        rows = q.shape[0]
        cl = ends.view(reqs, nn)[:, -1].contiguous()
        width = -(-ctx // 128) * 128
        logits = torch.empty(rows, width, dtype=torch.float32, device=DEV)
        pa.paged_mxfp4_mqa_logits(
            q.view(reqs, nn, H, 64),
            qs.view(reqs, nn, H, 4),
            cache.view(-1, P, 1, ENTRY),
            w,
            cl,
            bt,
            width,
            out_logits=logits,
            clean_logits=False,
            row_ends=ends,
        )
        mine, _ = run(ix, inputs, nn, ctx)
        for r, x in enumerate(logits_reference(*inputs, nn, P)):
            e = int(ends[r])
            got = logits[r, :e].double()
            assert float((got - x).abs().max()) <= 1e-5 * float(x.abs().max()), r
            assert topk_matches(mine[r], x) is None, r


def case_vllm_call_form(ix):
    """The cache as vLLM passes it to paged_mxfp4_mqa_logits: a
    [pages, P, 1, 68] view of a block-major pool whose pages are not adjacent,
    and the Q scales as one int32 per head."""
    P, nn = 128, 6
    q, qs, w, cache, bt, ends = random_inputs(lengths_of(30000, 5), nn, P, seed=61)
    pages = cache.shape[0]
    pool = torch.full(
        (pages, 2 * P * ENTRY + 256), 0xFF, dtype=torch.uint8, device=DEV
    )  # each page in a wider block, garbage around it
    pool[:, : P * ENTRY] = cache.view(pages, -1)
    view = torch.as_strided(
        pool, (pages, P, 1, ENTRY), (pool.stride(0), ENTRY, ENTRY, 1)
    )
    qs32 = qs.view(torch.int32).view(q.shape[0], H)
    out = torch.full((q.shape[0], TOPK), -9, dtype=torch.int32, device=DEV)
    ix.flydsl_mxfp4_indexer_topk(
        q, qs32, w, view, bt, ends, out, next_n=nn, max_ctx=30000
    )
    torch.cuda.synchronize()
    ref, _ = run(ix, (q, qs, w, cache, bt, ends), nn, 30000)
    assert torch.equal(out, ref)
    check_rows(out, (q, qs, w, cache, bt, ends), nn, P)


def _graph_replay(ix, ctx, P, lens, cand, grow):
    """Capture a call, replay it on new inputs; ``grow``: a larger eager call
    between capture and replay replaces the shared scratch first."""
    nn = 6
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=21)
    rows = q.shape[0]
    out = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
    c = torch.full((rows, CANDIDATES), -9, dtype=torch.int32, device=DEV)

    def call():
        ix.flydsl_mxfp4_indexer_topk(
            q,
            qs,
            w,
            cache,
            bt,
            ends,
            out,
            next_n=nn,
            max_ctx=ctx,
            candidates_out=c if cand else None,
        )

    if grow:  # the call sizes the scratch, the graph captures it
        ix._SCRATCH.clear()
        ix._SCRATCH_CAPTURED.clear()
    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    if grow:
        retired = len(ix._RETIRED)
        big = random_inputs([ctx] * 40, nn, P, seed=5)
        run(ix, big, nn, ctx, cand=cand)
        assert len(ix._RETIRED) == retired + 1  # kept for the graph
    used = sum(-(-x // P) for x in lens)
    for seed in (22, 23):
        b = random_inputs(lens, nn, P, seed=seed, extra_pages=cache.shape[0] - used)
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        out.fill_(-9)
        c.fill_(-9)
        g.replay()
        torch.cuda.synchronize()
        eager, eager_c = run(ix, (q, qs, w, cache, bt, ends), nn, ctx, cand=cand)
        assert torch.equal(out, eager)
        if cand:
            assert torch.equal(c, eager_c)
        else:
            check_rows(out, (q, qs, w, cache, bt, ends), nn, P)


def case_graph_replay(ix):
    _graph_replay(ix, 20000, 128, lengths_of(20000, 12), False, False)  # block
    _graph_replay(ix, 7000, 64, lengths_of(7000, 12), False, True)  # token
    lens = [300000, 40000, 700000, 120000, 20000, 1 << 20]
    _graph_replay(ix, 1 << 20, 128, lens, True, False)  # long, source


def case_workspace(ix):
    """A caller workspace full of garbage gives the same bits as the shared
    scratch; a short one is refused."""
    cases = (
        (4096, [4096, 3000, 100], "dense"),
        (40000, [40000, 30000, 7], "source"),
        (300000, [300000, 200000, 5000], "dense"),
        (300000, [300000, 200000, 5000], "source"),
    )
    for ctx, lens, layer in cases:
        inputs = random_inputs(lens, 6, 128, seed=ctx + len(layer))
        rows = inputs[0].shape[0]
        n = ix.mxfp4_indexer_workspace_bytes(rows, ctx, layer)
        assert n > 0
        ws = torch.full((n + 4,), 0xFF, dtype=torch.uint8, device=DEV)
        a = run(ix, inputs, 6, ctx, cand=layer == "source")
        b = run(ix, inputs, 6, ctx, cand=layer == "source", workspace=ws)
        assert torch.equal(a[0], b[0]) and (a[1] is None or torch.equal(a[1], b[1]))
        nc = ix.mxfp4_indexer_workspace_bytes(rows, ctx, "consumer")
        cand = a[1] if a[1] is not None else run(ix, inputs, 6, ctx, cand=True)[1]
        wsc = torch.full((max(nc, 1),), 0xFF, dtype=torch.uint8, device=DEV)
        assert torch.equal(
            consume(ix, inputs, cand, 6, ctx),
            consume(ix, inputs, cand, 6, ctx, workspace=wsc),
        )
        try:
            run(ix, inputs, 6, ctx, cand=layer == "source", workspace=ws[: n - 1])
        except ValueError:
            pass
        else:
            raise AssertionError("a short workspace was accepted")


def case_long_rows(ix):
    for ctx, reqs, P in ((524288, 3, 128), (524288, 4, 64), (1048576, 1, 128)):
        inputs = random_inputs(spread(ctx, reqs), 6, P, seed=ctx + reqs)
        out, cand = run(ix, inputs, 6, ctx, cand=True)
        plain, _ = run(ix, inputs, 6, ctx)
        assert torch.equal(out, plain)
        again, _ = run(ix, inputs, 6, ctx)
        assert torch.equal(again, out)
        logits, _ = check_rows(out, inputs, 6, P)
        for r, x in enumerate(logits):
            check_candidates(cand[r], x)


def case_rows_of_every_length(ix):
    """Tier 1M: empty, short, both sides of 262144 entries and the longest row
    in one call, every row exact, one ragged list for rows of both paths; then
    the consumer layers over the source's list."""
    ctx = 1048576
    lens = [0, 1, 5, 100, 4097, 100000, 262143, 262150, 600000, ctx]
    inputs = random_inputs(lens, 6, 128, seed=77)
    rows = inputs[0].shape[0]
    for bs in (64, 256):
        rg = ragged_of(ix, rows, len(lens), ctx, bs, 5)
        out, cand = run(ix, inputs, 6, ctx, cand=True, ragged=rg)
        logits, _ = check_rows(out, inputs, 6, 128)
        for r, x in enumerate(logits):
            check_candidates(cand[r], x)
        check_ragged(rg, out, 6)
    q, qs, w, cache, bt, ends = inputs
    g = torch.Generator(device=DEV).manual_seed(13)
    q2 = torch.randint(0, 256, q.shape, dtype=torch.uint8, device=DEV, generator=g)
    w2 = torch.randn_like(w) * 0.1
    cand[::4, 300:] = -1
    inputs2 = (q2, qs, w2, cache, bt, ends)
    cons = consume(ix, inputs2, cand, 6, ctx)
    for r, x in enumerate(logits_reference(*inputs2, 6, 128)):
        check_consumer(cons[r], cand[r], x)


def case_long_ties_and_outliers(ix):
    """Equal block maxima cut by block id, and keys whose scales put their
    logits ~2^40 above the rest."""
    ctx = 524288
    for ties in ("equal_pages", "zero_q", "outliers"):
        q, qs, w, cache, bt, ends = random_inputs(spread(ctx, 2), 6, 128, seed=91)
        if ties == "equal_pages":
            cache[:] = cache[0:1]
        elif ties == "zero_q":
            q.zero_()
        else:
            cache.view(cache.shape[0], -1)[:40:3, 128 * 64 :] = 167
        out, cand = run(ix, (q, qs, w, cache, bt, ends), 6, ctx, cand=True)
        for r, x in enumerate(logits_reference(q, qs, w, cache, bt, ends, 6, 128)):
            if ties == "outliers":
                assert topk_matches(out[r], x) is None, r
                check_candidates(cand[r], x)
            else:
                assert torch.equal(out[r], select_reference(x)), r
                assert torch.equal(cand[r], candidates_reference(x)), r


def case_ragged_lists(ix):
    """The select's ragged slot lists are the reference packing of its own
    top-512 positions (rows not valid get none)."""
    nn = 6
    for ctx, consumer, bs in itertools.product(
        (12000, 30000), (False, True), (64, 128, 256)
    ):
        lens = lengths_of(ctx, 9) + [200, 0]
        inputs = random_inputs(lens, nn, 128, seed=31)
        rows = inputs[0].shape[0]
        rg = ragged_of(ix, rows, len(lens), ctx, bs, 5)
        if consumer:
            _, cand = run(ix, inputs, nn, ctx, cand=True)
            out = consume(ix, inputs, cand, nn, ctx, ragged=rg)
        else:
            out, _ = run(ix, inputs, nn, ctx, ragged=rg)
        check_ragged(rg, out, nn)


def case_unsupported(ix):
    reason = ix.mxfp4_indexer_unsupported_reason
    assert ix.MXFP4_INDEXER_MAX_ENTRIES == 1 << 20
    assert reason(128, ix.MXFP4_INDEXER_MAX_ENTRIES, 6) is None
    assert reason(64, 4096, 1) is None and reason(128, 1, 8) is None
    assert "1048577" in reason(128, ix.MXFP4_INDEXER_MAX_ENTRIES + 1, 6)
    assert "entries per page" in reason(256, 4096, 6)
    assert "rows per request" in reason(64, 4096, 9)
    assert "heads" in reason(64, 4096, 6, num_heads=64)
    assert "top-" in reason(64, 4096, 6, topk=2048)
    preferred = ix.mxfp4_indexer_dense_preferred
    assert not preferred(16384, 768) and preferred(65536, 18)
    assert not preferred(16384, 768, source=True) and preferred(32768, 18, True)
    assert not preferred(262144, 66, True) and preferred(262144, 192, True)
    assert not preferred(524288, 66) and preferred(524288, 126)
    assert preferred(1 << 20, 192) and not preferred((1 << 20) + 1, 192)
    q, qs, w, cache, bt, ends = random_inputs([1000], 6, 64)
    out = torch.empty(6, TOPK, dtype=torch.int32, device=DEV)
    bad = (
        {"block_table": bt.long()},
        {"next_n": 9},
        {"q_scale": qs.float()},  # same bytes per row, wrong dtype
        {"q_scale": qs.view(torch.int32)},  # int32 but [rows, 32, 1]
    )
    for kw in bad:
        args = {
            "q": q,
            "q_scale": qs,
            "weights": w,
            "kv_cache": cache,
            "block_table": bt,
            "row_ends": ends,
            "out": out,
            "next_n": 6,
            "max_ctx": 1000,
        }
        args.update(kw)
        try:
            ix.flydsl_mxfp4_indexer_topk(**args)
        except ValueError:
            continue
        raise AssertionError(f"accepted {list(kw)}")


CASES = {
    f.__name__[len("case_") :]: f
    for f in (
        case_zero_rows,
        case_short_and_empty_rows,
        case_ties,
        case_repeat_runs,
        case_block_scores,
        case_paged_mxfp4_mqa_logits_contract,
        case_vllm_call_form,
        case_graph_replay,
        case_workspace,
        case_ragged_lists,
        case_long_rows,
        case_rows_of_every_length,
        case_long_ties_and_outliers,
        case_unsupported,
    )
}


@benchmark()
def test_mxfp4_indexer_case(case):
    CASES[case](_ix())
    return {"gfx": get_gfx_runtime(), "passed": True}


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "mxfp4 indexer %s summary (markdown):\n%s", name, df.to_markdown(index=False)
    )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mxfp4 indexer top-512 unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MXFP4 indexer top-512",
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        nargs="*",
        default=[4096, 16384, 65536, 131072, 262144],
        help="max_ctx of the dense / source / consumer sweeps",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[3, 32, 128],
        help="requests (next_n rows each)",
    )
    parser.add_argument("--next_n", type=int, nargs="*", default=[6])
    parser.add_argument("--page", type=int, nargs="*", default=[128])
    parser.add_argument("--layer", nargs="*", default=["dense", "source"])
    parser.add_argument(
        "--rows_per_request",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5, 7, 8],
        help="next_n values of the extra 17000-entry sweep",
    )
    parser.add_argument(
        "--path",
        type=int,
        nargs="*",
        default=list(range(len(PATH_CASES))),
        help="indices into PATH_CASES",
    )
    parser.add_argument("--case", nargs="*", default=list(CASES), choices=list(CASES))
    args = parser.parse_args()

    rows = []
    for ctx, reqs, nn, page, layer in itertools.product(
        args.ctx, args.batch, args.next_n, args.page, args.layer
    ):
        rows.append(test_mxfp4_indexer_topk(ctx, reqs, nn, page, layer))
    for nn, page in itertools.product(args.rows_per_request, (64, 128)):
        rows.append(test_mxfp4_indexer_topk(17000, 5, nn, page, "dense"))
    summarize("topk", rows)
    summarize(
        "consumer",
        [
            test_mxfp4_indexer_candidate_topk(ctx, reqs, nn, page)
            for ctx, reqs, nn, page in itertools.product(
                args.ctx, args.batch, args.next_n, args.page
            )
        ],
    )
    summarize("paths", [test_mxfp4_indexer_paths(i) for i in args.path])
    summarize("cases", [test_mxfp4_indexer_case(c) for c in args.case])


if __name__ == "__main__":
    main()
