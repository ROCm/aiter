# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL MXFP4 indexer top-512 (aiter.ops.flydsl.mxfp4_indexer_topk)
against an fp64 torch reference of ``paged_mxfp4_mqa_logits``' logits and the
exact top-512 selection.

Covered: the top-512 positions at 4k-130k contexts and 18-768 decode rows (the
exact fp64 selection up to fp32 rounding at the 512th rank, and equal to it on
most rows), 1-8 rows per request, 64- and 128-entry pages, short and empty
rows, tied logits (one page repeated, zero queries), the candidate source list
and the candidate consumer, the ragged slot lists, the token / block / long
paths bit for bit against each other (tier edges included), the split and the
one-workgroup candidate consumer bit for bit, 524288- and 1048576-entry rows,
repeat runs, CUDA-graph replay with new inputs, the call form vLLM uses
(``[pages, P, 1, 68]`` view of a block-major pool, int32 Q scales), the
agreement of aiter's ``paged_mxfp4_mqa_logits`` with the reference on the same
cache, and the unsupported-shape reasons.

Run: ``python3 -m pytest op_tests/flydsl_tests/test_flydsl_mxfp4_indexer_topk.py``
(``-k "not long"`` skips the 0.5M-1M entry rows).
"""

import pytest
import torch

DEV = "cuda"
H, D, ENTRY = 32, 128, 68
TOPK, CANDIDATES, BLOCK = 512, 2048, 8
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 = E2M1 + [-x for x in E2M1]


def _gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs gfx950")


@pytest.fixture(scope="module")
def ix():
    from aiter.ops.flydsl import mxfp4_indexer_topk

    return mxfp4_indexer_topk


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
    bt = torch.zeros(B, max(max(need), 1), dtype=torch.int32, device=DEV)
    o = 0
    for i, n in enumerate(need):
        bt[i, :n] = perm[o : o + n]
        o += n
    T = B * nn
    q = torch.randint(0, 256, (T, H, 64), dtype=torch.uint8, device=DEV, generator=g)
    qs = torch.randint(120, 127, (T, H, 4), dtype=torch.uint8, device=DEV, generator=g)
    w = torch.randn(T, H, device=DEV, generator=g) * 0.1
    cl = torch.tensor(lengths, dtype=torch.int32, device=DEV)
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


def run(ix, inputs, nn, ctx, *, cand=False, ragged=None):
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
    )
    torch.cuda.synchronize()
    return out, c


def check_rows(out, inputs, nn, P, equal_frac=0.9):
    """Every row a valid top-512 of the fp64 logits; most equal to the exact
    selection (rows whose 512th and 513th logits are within fp32 rounding may
    differ)."""
    logits = logits_reference(*inputs, nn, P)
    exact = 0
    for r, x in enumerate(logits):
        msg = topk_matches(out[r], x)
        assert msg is None, (r, int(inputs[5][r]), msg)
        exact += int(bool((out[r] == select_reference(x)).all()))
    assert exact >= equal_frac * len(logits), (exact, len(logits))
    return logits


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


# ----------------------------------------------------------------------------
# Dense layer
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("reqs", [3, 11, 32, 64, 128])
@pytest.mark.parametrize("ctx", [4096, 16384, 65536, 131072])
def test_dense_matches_reference(ix, reqs, ctx):
    P = 64 if ctx <= 65536 else 128
    inputs = random_inputs(lengths_of(ctx, reqs), 6, P, seed=reqs + ctx)
    out, _ = run(ix, inputs, 6, ctx)
    check_rows(out, inputs, 6, P)


@pytest.mark.parametrize("nn", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("P", [64, 128])
def test_rows_per_request_and_pages(ix, nn, P):
    inputs = random_inputs([9000, 513, 4100, 17000, 300], nn, P, seed=nn * 7 + P)
    out, _ = run(ix, inputs, nn, 17000)
    check_rows(out, inputs, nn, P)


def test_short_and_empty_rows(ix):
    lens = [0, 1, 5, 6, 7, 100, 511, 512, 513, 1023, 4097]
    inputs = random_inputs(lens, 6, 128, seed=5)
    out, _ = run(ix, inputs, 6, 4100)
    check_rows(out, inputs, 6, 128)


@pytest.mark.parametrize("ctx", [3000, 30000])
def test_ties_are_broken_by_position(ix, ctx):
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


def test_repeat_runs_are_bit_identical(ix):
    inputs = random_inputs(lengths_of(65536, 16), 6, 64, seed=11)
    a, _ = run(ix, inputs, 6, 65536)
    b, _ = run(ix, inputs, 6, 65536)
    assert torch.equal(a, b)


def test_block_scores(ix):
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


@pytest.mark.parametrize("reqs,ctx,P", [(4, 20000, 128), (7, 60000, 64)])
def test_reference_is_paged_mxfp4_mqa_logits(ix, reqs, ctx, P):
    """The contract is aiter's paged_mxfp4_mqa_logits on the same cache: its
    logits agree with the reference, and its top-512 and this op's both are
    valid top-512 of the reference logits."""
    from aiter.ops.topk import top_k_per_row_decode
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa

    nn = 6
    q, qs, w, cache, bt, ends = random_inputs(lengths_of(ctx, reqs), nn, P, seed=17)
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
    theirs = torch.empty(rows, TOPK, dtype=torch.int32, device=DEV)
    top_k_per_row_decode(
        logits, 1, ends, theirs, rows, logits.stride(0), logits.stride(1), k=TOPK
    )
    mine, _ = run(ix, (q, qs, w, cache, bt, ends), nn, ctx)
    for r, x in enumerate(logits_reference(q, qs, w, cache, bt, ends, nn, P)):
        e = int(ends[r])
        got = logits[r, :e].double()
        assert float((got - x).abs().max()) <= 1e-5 * float(x.abs().max()), r
        other = torch.full((TOPK,), -1, dtype=torch.int32, device=DEV)
        srt = torch.sort(theirs[r][theirs[r] >= 0]).values.int()
        other[: srt.numel()] = srt
        assert topk_matches(other, x) is None, r
        assert topk_matches(mine[r], x) is None, r


def test_vllm_call_form(ix):
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


@pytest.mark.parametrize("ctx,P", [(20000, 128), (7000, 64)])
def test_graph_replay_with_new_inputs(ix, ctx, P):
    nn = 6
    lens = lengths_of(ctx, 12)
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=21)
    out = torch.full((q.shape[0], TOPK), -9, dtype=torch.int32, device=DEV)

    def call():
        ix.flydsl_mxfp4_indexer_topk(
            q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx
        )

    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    used = sum(-(-c // P) for c in lens)
    for seed in (22, 23):
        b = random_inputs(lens, nn, P, seed=seed, extra_pages=cache.shape[0] - used)
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        out.fill_(-9)
        g.replay()
        torch.cuda.synchronize()
        check_rows(out, (q, qs, w, cache, bt, ends), nn, P)


# ----------------------------------------------------------------------------
# Two-level indexer: candidate source and consumer
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reqs,ctx,P", [(5, 131072, 128), (3, 9000, 128), (4, 1500, 64), (6, 40000, 64)]
)
def test_candidate_source(ix, reqs, ctx, P):
    nn = 6
    inputs = random_inputs(lengths_of(ctx, reqs) + [3, 0], nn, P, seed=ctx)
    out, cand = run(ix, inputs, nn, ctx, cand=True)
    plain, _ = run(ix, inputs, nn, ctx)
    assert torch.equal(out, plain)  # the source's top-512 is the dense one's
    same = 0
    for r, x in enumerate(logits_reference(*inputs, nn, P)):
        assert topk_matches(out[r], x) is None, r
        check_candidates(cand[r], x)
        same += int(torch.equal(cand[r], candidates_reference(x)))
    assert same >= 0.9 * out.shape[0]


@pytest.mark.parametrize(
    "reqs,ctx,P", [(11, 131072, 128), (32, 16384, 128), (64, 65536, 64)]
)
def test_candidate_consumer(ix, reqs, ctx, P):
    nn = 6
    q, qs, w, cache, bt, ends = random_inputs(lengths_of(ctx, reqs), nn, P, seed=reqs)
    logits = logits_reference(q, qs, w, cache, bt, ends, nn, P)
    cand = torch.stack([candidates_reference(x) for x in logits]).contiguous()
    cand[::3, 100:] = -1  # a short pool
    out = torch.full((q.shape[0], TOPK), -9, dtype=torch.int32, device=DEV)
    ix.flydsl_mxfp4_indexer_candidate_topk(
        q, qs, w, cache, bt, ends, cand, out, next_n=nn, max_ctx=ctx
    )
    torch.cuda.synchronize()
    for r, x in enumerate(logits):
        check_consumer(out[r], cand[r], x)


@pytest.mark.parametrize(
    "ctx,lens,nn,P",
    [
        (131072, [131072, 90000, 40000, 20000], 6, 128),
        (65536, [65536, 30000, 17000, 9000, 2000, 7, 0], 6, 64),
        (20000, [20000, 19000, 100], 3, 128),
        (131072, [131000, 120000], 8, 128),
    ],
)
def test_candidate_consumer_split_equals_gather_select(
    ix, monkeypatch, ctx, lens, nn, P
):
    """Few rows: (row, segment) workgroups score the candidates and the token
    select picks; bit for bit the one-workgroup-a-row gather select."""
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=ctx + nn)
    rows = q.shape[0]
    _, cand = run(ix, (q, qs, w, cache, bt, ends), nn, ctx, cand=True)
    cand[::3, 300:] = -1  # a short pool on some rows
    rg_s = ragged_of(ix, rows, len(lens), ctx, 128, 3)
    rg_f = ragged_of(ix, rows, len(lens), ctx, 128, 3)
    out_s = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
    out_f = out_s.clone()
    ix.flydsl_mxfp4_indexer_candidate_topk(
        q, qs, w, cache, bt, ends, cand, out_s, next_n=nn, max_ctx=ctx, ragged=rg_s
    )
    monkeypatch.setattr(ix, "CAND_SPLIT_ROWS", 0)
    ix.flydsl_mxfp4_indexer_candidate_topk(
        q, qs, w, cache, bt, ends, cand, out_f, next_n=nn, max_ctx=ctx, ragged=rg_f
    )
    torch.cuda.synchronize()
    assert torch.equal(out_s, out_f)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(rg_s, f), getattr(rg_f, f)), f
    check_ragged(rg_s, out_s, nn)


# ----------------------------------------------------------------------------
# Ragged lists and path equivalence
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("ctx", [12000, 30000])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("bs", [64, 128, 256])
def test_ragged_lists(ix, ctx, candidate, bs):
    """The select's ragged slot lists are the reference packing of its own
    top-512 positions (rows not valid get none)."""
    nn = 6
    lens = lengths_of(ctx, 9) + [200, 0]
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, 128, seed=31)
    rows = q.shape[0]
    rg = ragged_of(ix, rows, len(lens), ctx, bs, 5)
    if candidate:
        _, cand = run(ix, (q, qs, w, cache, bt, ends), nn, ctx, cand=True)
        out = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
        ix.flydsl_mxfp4_indexer_candidate_topk(
            q, qs, w, cache, bt, ends, cand, out, next_n=nn, max_ctx=ctx, ragged=rg
        )
        torch.cuda.synchronize()
    else:
        out, _ = run(ix, (q, qs, w, cache, bt, ends), nn, ctx, ragged=rg)
    check_ragged(rg, out, nn)


@pytest.mark.parametrize(
    "ctx,lens,nn,P",
    [
        (4096, [4096, 3000, 513, 100, 1, 0, 4000], 6, 64),
        (4097, [4097, 600, 4000], 6, 64),  # the 8192 tier
        (8192, [8192, 100, 4300, 7000, 512, 511], 3, 128),
        (8193, [8193, 3000, 5555], 6, 128),  # the 16384 tier
        (16384, [16384, 9000, 5000, 8193, 4097, 700, 12345], 6, 64),
        (16384, [16000, 16384, 12000, 3000, 20, 7777, 9999, 15000], 8, 128),
        (4096, [4096, 2048, 2], 1, 64),
    ],
)
@pytest.mark.parametrize("ties", [False, True])
def test_token_path_equals_block_path(ix, monkeypatch, ctx, lens, nn, P, ties):
    """Contexts up to 16384 select from the walk's token logits; positions,
    ragged lists and candidate lists equal the block path's bit for bit."""
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=ctx + nn + len(lens))
    if ties:
        cache[:] = cache[0:1]
        q[::2] = 0
    inputs = (q, qs, w, cache, bt, ends)
    rows = q.shape[0]
    rg_t = ragged_of(ix, rows, len(lens), ctx, 128, 3)
    rg_b = ragged_of(ix, rows, len(lens), ctx, 128, 3)
    out_t, _ = run(ix, inputs, nn, ctx, ragged=rg_t)
    src_t, cand_t = run(ix, inputs, nn, ctx, cand=True)
    monkeypatch.setattr(ix, "_tokens_tier", lambda max_ctx: None)
    out_b, _ = run(ix, inputs, nn, ctx, ragged=rg_b)
    src_b, cand_b = run(ix, inputs, nn, ctx, cand=True)
    assert torch.equal(out_t, out_b)
    assert torch.equal(src_t, src_b) and torch.equal(cand_t, cand_b)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(rg_t, f), getattr(rg_b, f)), f
    check_rows(out_t, inputs, nn, P, equal_frac=0.0)


@pytest.mark.parametrize(
    "lens,nn,P,ties",
    [
        ([90000, 70000, 40000, 33000, 32768, 20000, 5000, 100, 0, 1], 6, 128, ""),
        ([100000, 64000, 31000, 12000], 3, 64, ""),
        ([100000, 48000, 9000, 40], 6, 128, "equal_pages"),
        ([100000, 48000, 9000, 40], 6, 64, "zero_q"),
        ([140000, 100000, 66000, 3000, 12], 6, 128, ""),  # two segments
        ([66000, 50000], 8, 128, ""),
        ([70000, 4000], 1, 64, ""),
    ],
)
def test_long_path_equals_block_path(ix, monkeypatch, lens, nn, P, ties):
    """The long path forced at contexts the block path serves: positions,
    candidate lists, consumer outputs and ragged lists bit for bit the block
    path's, rows below the forced limit (block select, same call) included."""
    ctx = max(lens)
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=ctx + nn)
    if ties == "equal_pages":
        cache[:] = cache[0:1]
    elif ties == "zero_q":
        q[::2] = 0
    rows = q.shape[0]
    inputs = (q, qs, w, cache, bt, ends)

    def all_outputs():
        rg = ragged_of(ix, rows, len(lens), ctx, 128, 3)
        out, cand = run(ix, inputs, nn, ctx, cand=True)
        out2, _ = run(ix, inputs, nn, ctx, ragged=rg)
        cons = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
        ix.flydsl_mxfp4_indexer_candidate_topk(
            q, qs, w, cache, bt, ends, cand, cons, next_n=nn, max_ctx=ctx
        )
        torch.cuda.synchronize()
        return out, cand, out2, cons, rg

    block = all_outputs()
    monkeypatch.setattr(ix, "BLOCK_CTX_MAX", 16384)
    assert ix._blocks_tier(ix.BLOCK_CTX_MAX) * BLOCK < ctx
    long = all_outputs()
    for a, b in zip(block[:4], long[:4]):
        assert torch.equal(a, b)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(block[4], f), getattr(long[4], f)), f


# ----------------------------------------------------------------------------
# Long rows (above 262144 entries)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ctx,reqs,P", [(524288, 3, 128), (524288, 4, 64), (1048576, 1, 128)]
)
def test_long_rows_are_exact(ix, ctx, reqs, P):
    inputs = random_inputs(spread(ctx, reqs), 6, P, seed=ctx + reqs)
    out, cand = run(ix, inputs, 6, ctx, cand=True)
    plain, _ = run(ix, inputs, 6, ctx)
    assert torch.equal(out, plain)
    again, _ = run(ix, inputs, 6, ctx)
    assert torch.equal(again, out)
    for r, x in enumerate(check_rows(out, inputs, 6, P)):
        check_candidates(cand[r], x)


@pytest.mark.parametrize("bs", [64, 256])
def test_rows_of_every_length_in_the_long_tier(ix, bs):
    """Tier 1M: empty, short, both sides of 262144 entries and the longest row
    in one call, every row exact, one ragged list for rows of both paths; then
    the consumer layers over the source's list."""
    ctx = 1048576
    lens = [0, 1, 5, 100, 4097, 100000, 262143, 262150, 600000, ctx]
    inputs = random_inputs(lens, 6, 128, seed=77)
    rows = inputs[0].shape[0]
    rg = ragged_of(ix, rows, len(lens), ctx, bs, 5)
    out, cand = run(ix, inputs, 6, ctx, cand=True, ragged=rg)
    logits = check_rows(out, inputs, 6, 128)
    for r, x in enumerate(logits):
        check_candidates(cand[r], x)
    check_ragged(rg, out, 6)
    q, qs, w, cache, bt, ends = inputs
    g = torch.Generator(device=DEV).manual_seed(13)
    q2 = torch.randint(0, 256, q.shape, dtype=torch.uint8, device=DEV, generator=g)
    w2 = torch.randn_like(w) * 0.1
    cand[::4, 300:] = -1
    cons = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
    ix.flydsl_mxfp4_indexer_candidate_topk(
        q2, qs, w2, cache, bt, ends, cand, cons, next_n=6, max_ctx=ctx
    )
    torch.cuda.synchronize()
    for r, x in enumerate(logits_reference(q2, qs, w2, cache, bt, ends, 6, 128)):
        check_consumer(cons[r], cand[r], x)


@pytest.mark.parametrize("ties", ["equal_pages", "zero_q", "outliers"])
def test_long_ties_and_outliers(ix, ties):
    """Equal block maxima cut by block id, and keys whose scales put their
    logits ~2^40 above the rest."""
    ctx = 524288
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


def test_graph_replay_long_with_new_inputs(ix):
    ctx, nn, P = 1048576, 6, 128
    lens = [300000, 40000, 700000, 120000, 20000, ctx]
    q, qs, w, cache, bt, ends = random_inputs(lens, nn, P, seed=21)
    rows = q.shape[0]
    out = torch.full((rows, TOPK), -9, dtype=torch.int32, device=DEV)
    cand = torch.full((rows, CANDIDATES), -9, dtype=torch.int32, device=DEV)

    def call():
        ix.flydsl_mxfp4_indexer_topk(
            q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=cand
        )

    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    used = sum(-(-c // P) for c in lens)
    for seed in (22, 23):
        b = random_inputs(lens, nn, P, seed=seed, extra_pages=cache.shape[0] - used)
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        out.fill_(-9)
        g.replay()
        torch.cuda.synchronize()
        eager, eager_c = run(ix, (q, qs, w, cache, bt, ends), nn, ctx, cand=True)
        assert torch.equal(out, eager) and torch.equal(cand, eager_c)


# ----------------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------------


def test_unsupported_reasons(ix):
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
    assert not preferred(16384) and preferred(65536) and preferred(262144)
    assert not preferred(262145)
    q, qs, w, cache, bt, ends = random_inputs([1000], 6, 64)
    out = torch.empty(6, TOPK, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError):
        ix.flydsl_mxfp4_indexer_topk(
            q, qs, w, cache, bt.long(), ends, out, next_n=6, max_ctx=1000
        )
    with pytest.raises(ValueError):
        ix.flydsl_mxfp4_indexer_topk(
            q, qs, w, cache, bt, ends, out, next_n=9, max_ctx=1000
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
