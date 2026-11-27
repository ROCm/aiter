# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 decode index block scores over the NVFP4 index cache (gfx1250).

``aiter.ops.flydsl.index_score_decode`` against an FP32 torch reference:
score[h, b * q + t, blk] = max over the block's keys before the row's cutoff
(seq_len - q + t + 1) of (hi + lo / 16) . dequant(K), times sm_scale * log2(e),
hi / lo the query's two E4M3 terms. The kernel's dots are the hardware's
128-deep scaled WMMA, the reference's an FP32 matmul, so they agree to a few
1e-3 relative, not bit for bit: a score is graded against the WMMA's error
envelope (``SCALED_DOT_REL`` of the sum of the products' magnitudes). Bitwise
equality with a Triton scorer that issues the same WMMA is checked on the
caller's side.

Shapes are the model's: 4 index heads x 8 query rows a request at TP1 (DSpark
K = 7), 1 x 8 on a TP4 rank, plain decode (1 row), contexts of 50K-300K tokens,
the 9216 B pages of ATOM's NVFP4 index pool (payload and scales two byte views
of one buffer).

``index_select_decode`` (the decode top-k and its sparse-table emit) against a
torch reference of ATOM's Triton selector: the same ids in the same order, the
same table rows and context lengths, exactly (scores with ties, NaN, -inf, -0.0,
graph pad requests); and ``index_topk_decode_nvfp4`` (the scorer's flat plan +
the selector) end to end against the reference selection of its own scores.

Run with: python op_tests/test_flydsl_index_score_decode_nvfp4.py
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx1250"]
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PAGE, D = 128, 128
PAGE_BYTES = PAGE * (D // 2 + D // 16)


def dequant(payload, scale):
    """[..., 128, 64] e2m1 pairs and [..., 128, 8] E4M3 scales -> fp32 keys."""
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scale.view(torch.float8_e4m3fn).float()[..., None]).flatten(-2)


def make_inputs(lens, query_len, heads, seed=0):
    """The pool as ATOM lays it out: one [pages, 9216] byte buffer, the payload
    its first 8192 bytes a page and the scales the last 1024; pages of a
    request scattered by the block table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nbs = [-(-n // PAGE) for n in lens]
    pages = sum(nbs) + 2
    pool = torch.randint(
        0, 256, (pages, PAGE_BYTES), dtype=torch.uint8, device="cuda", generator=g
    )
    payload = pool[:, : PAGE * D // 2].unflatten(1, (PAGE, D // 2))
    scale = pool[:, PAGE * D // 2 :].unflatten(1, (PAGE, D // 16))
    # Finite scales of mixed magnitude: a constant scale would hide a scale-row
    # mix-up, a NaN one the comparison.
    scale.copy_(
        (torch.rand(pages, PAGE, D // 16, device="cuda", generator=g) * 2 + 0.01)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    # A few queries outside E4M3's range, so the clamp is exercised.
    q = torch.randn(len(lens) * query_len, heads, D, device="cuda", generator=g) * 3
    q[::7, :, ::31] = 600.0
    q = q.to(torch.bfloat16)
    perm = torch.randperm(pages, device="cuda", generator=g).int()
    bt = torch.zeros(len(lens), max(nbs), dtype=torch.int32, device="cuda")
    s = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[s : s + nb]
        s += nb
    return q, payload, scale, bt, torch.tensor(lens, dtype=torch.int32, device="cuda")


# The scaled WMMA does not sum its 128 products exactly in fp32: on MI455X a dot
# lands ~2^-12 of the sum of the products' magnitudes from the exact one. A
# score is graded against this fraction of that sum.
SCALED_DOT_REL = 2.0**-9


def run_torch(q, payload, scale, bt, seq_lens, sm_scale, query_len, max_block):
    """Reference only: FP32 math. Returns (score, live, tol) -- the value of
    each block the kernel must write (the rest of ``score`` is NaN), which
    those are, and the error a WMMA score may have there."""
    from aiter.ops.flydsl.kernels.index_score_decode_nvfp4_gfx1250 import (
        sm_scale_log2e,
    )

    rows, heads, _ = q.shape
    q32 = q.float().clamp(-448, 448)
    hi = q32.to(torch.float8_e4m3fn).float()
    lo = ((q32 - hi) * 16).to(torch.float8_e4m3fn).float()
    q_eff = hi + lo / 16
    q_mag = hi.abs() + lo.abs() / 16
    c = sm_scale_log2e(sm_scale)
    score = torch.full((heads, rows, max_block), float("nan"), device=q.device)
    tol = torch.zeros_like(score)
    live = torch.zeros_like(score, dtype=torch.bool)
    for b, n in enumerate(seq_lens.tolist()):
        nb = min(-(-n // PAGE), max_block)
        if nb == 0:
            continue
        pages = bt[b, :nb].long()
        k = dequant(payload[pages], scale[pages]).reshape(-1, D)
        pos = torch.arange(nb * PAGE, device=q.device)
        for t in range(query_len):
            r = b * query_len + t
            hidden = pos[None] >= n - query_len + t + 1
            dots = (q_eff[r] @ k.T).masked_fill(hidden, float("-inf"))
            mags = (q_mag[r] @ k.abs().T).masked_fill(hidden, 0.0)
            score[:, r, :nb] = dots.view(heads, nb, PAGE).amax(-1) * c
            tol[:, r, :nb] = mags.view(heads, nb, PAGE).amax(-1) * c * SCALED_DOT_REL
            live[:, r, :nb] = True
    return score, live, tol + 1e-6


@benchmark()
def test_index_score_decode_nvfp4(
    batch, context, query_len, heads, workgroups, ring, lens
):
    from aiter.ops.flydsl.index_score_decode import (
        index_score_decode_nvfp4,
        index_score_decode_nvfp4_supported,
    )

    sm_scale = D**-0.5
    q, payload, scale, bt, seq_lens = make_inputs(lens, query_len, heads)
    max_block = bt.shape[1]
    assert index_score_decode_nvfp4_supported(q, payload, scale, query_len)
    ref, live, tol = run_torch(
        q, payload, scale, bt, seq_lens, sm_scale, query_len, max_block
    )
    # A row whose cutoff is not positive sees no key: -inf, exactly.
    hidden = live & (ref == float("-inf"))
    graded = live & ~hidden
    # NaN-poisoned: a block the kernel skips, or writes past the live ones,
    # shows.
    score = torch.full_like(ref, float("nan"))

    def flydsl():
        index_score_decode_nvfp4(
            q,
            payload,
            scale,
            score,
            bt,
            seq_lens,
            sm_scale,
            query_len,
            workgroups=workgroups or None,
            ring=ring,
        )
        return score

    flat_score = torch.full_like(ref, float("nan"))

    def flydsl_flat():
        index_score_decode_nvfp4(
            q,
            payload,
            scale,
            flat_score,
            bt,
            seq_lens,
            sm_scale,
            query_len,
            workgroups=workgroups or None,
            ring=ring,
            flat=True,
        )
        return flat_score

    candidates = {"flydsl": flydsl, "flat": flydsl_flat}
    blocks = sum(min(-(-n // PAGE), max_block) for n in lens)
    nbytes = blocks * PAGE_BYTES + q.numel() * 2 + int(live.sum()) * 4
    # Two scaled dots (hi, lo) a (row, head, key).
    flops = 2 * 2 * heads * query_len * blocks * PAGE * D
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        assert torch.isnan(
            out[~live]
        ).all(), f"{name}: wrote a block past the live ones"
        assert not torch.isnan(out[live]).any(), f"{name}: a live block not written"
        assert (out[hidden] == float("-inf")).all(), f"{name}: a hidden block scored"
        # The error in units of the WMMA's envelope at each block: within 1.
        err = checkAllclose(
            torch.zeros_like(ref),
            ((out - ref) / tol).nan_to_num(0.0, posinf=2.0, neginf=-2.0),
            rtol=0,
            atol=1.0,
            mask=graded,
            msg=f"{name}: index_score_decode_nvfp4 b{batch} ctx{context} "
            f"q{query_len} h{heads}",
        )
        assert err == 0, f"{name}: {err:.2%} of the scores outside the WMMA bound"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def ref_select(score, seq_lens, query_len, topk, init_blocks, local_blocks):
    """ATOM's decode selection (``_score_tile_keys`` + ``_pack_score_key`` +
    ``tl.topk``) in torch: [heads, rows, topk] ids, -1 past the row's blocks."""
    heads, _, max_block = score.shape
    rows = seq_lens.numel() * query_len
    s = score[:, :rows].clone()
    r = torch.arange(rows, device=score.device)
    t = r % query_len
    valid = (seq_lens.long()[r // query_len] - query_len + t + PAGE) // PAGE
    idx = torch.arange(max_block, device=score.device)
    live = idx[None] < valid[:, None]
    local_start = (valid - local_blocks).clamp(min=0)
    s = torch.where(live & (idx[None] < init_blocks), 1e30, s)
    s = torch.where(live & (idx[None] >= local_start[:, None]), 1e29, s)
    bits = s.view(torch.int32).long() & 0xFFFFFFFF
    bits = torch.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ torch.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    ordered = torch.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = (1 << 48) | (ordered << 16) | (idx + 1)
    key = torch.where(live, key, 0)
    k = min(topk, max_block)
    top = key.topk(k, dim=-1).values
    ids = ((top & 0xFFFF) - 1).int()
    ids = torch.where(
        torch.arange(k, device=score.device) < valid.clamp(max=topk)[:, None], ids, -1
    )
    out = torch.full((heads, rows, topk), -1, dtype=torch.int32, device=score.device)
    out[..., :k] = ids
    return out


def ref_emit(topk_idx, seq_lens, bt, query_len, pages_per_block, block_page_stride):
    """``_emit_sparse_block_table_row`` in torch: (table rows, context lengths)
    of row ``r * heads + h``."""
    heads, rows, topk = topk_idx.shape
    sbt = torch.zeros(rows * heads, topk * pages_per_block, dtype=torch.int32)
    sctx = torch.zeros(rows * heads, dtype=torch.int32)
    ids, lens, btc = topk_idx.cpu(), seq_lens.cpu().tolist(), bt.cpu()
    for r in range(rows):
        b, t = divmod(r, query_len)
        causal = lens[b] - query_len + t + 1
        self_blk = int((causal - 1) / PAGE)  # Triton's division truncates
        for h in range(heads):
            blk = ids[h, r].tolist()
            ok = [x >= 0 and x <= self_blk for x in blk]
            if pages_per_block == 1:
                ok = [o and causal > 0 for o in ok]
            full = [o and x < self_blk for o, x in zip(ok, blk)]
            n_full, n_ok = sum(full), sum(ok)
            e = r * heads + h
            seen = 0
            for o, f, x in zip(ok, full, blk):
                if not o:
                    continue
                slot = seen if f else n_full
                seen += f
                base = int(btc[b, x]) * block_page_stride * heads + h
                for j in range(pages_per_block):
                    sbt[e, slot * pages_per_block + j] = base + j * heads
            has_tail = any(o and x == self_blk for o, x in zip(ok, blk))
            ctx = (
                n_full * PAGE + causal - self_blk * PAGE
                if has_tail
                else min(n_ok * PAGE, causal)
            )
            sctx[e] = max(ctx, 0) if pages_per_block == 1 else ctx
    return sbt, sctx


def select_scores(heads, rows, max_block, kind, seed):
    """Score planes the selector must rank exactly as ATOM does."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    s = torch.randn(heads, rows, max_block, device="cuda", generator=g)
    if kind == "ties":  # few distinct values: ties at the cut everywhere
        s = (s * 2).round()
    elif kind == "specials":
        m = torch.rand(s.shape, device="cuda", generator=g)
        s = torch.where(m < 0.02, float("nan"), s)
        s = torch.where((m >= 0.02) & (m < 0.05), float("-inf"), s)
        s = torch.where((m >= 0.05) & (m < 0.08), -0.0, s)
        s = torch.where((m >= 0.08) & (m < 0.1), 0.0, s)
        s = torch.where((m >= 0.1) & (m < 0.11), float("inf"), s)
    elif kind == "ascending":  # the newest blocks score highest
        s = s * 0.01 + torch.arange(max_block, device="cuda") / max_block
    elif kind == "constant":
        s = torch.zeros_like(s)
    return s


@benchmark()
def test_index_select_decode(
    batch,
    context,
    query_len,
    heads,
    topk,
    init_blocks,
    local_blocks,
    kind,
    emit,
    waves,
    lens,
):
    from aiter.ops.flydsl.index_score_decode import (
        index_select_decode,
        index_topk_decode_score_plane,
    )

    max_block = -(-context // PAGE)
    rows = batch * query_len
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    # the plane's padding columns hold garbage too: never read as a block
    score = index_topk_decode_score_plane(heads, rows + 3, max_block, "cuda")
    padded = (heads, rows + 3, score.stride(1))
    torch.as_strided(score, padded, score.stride()).copy_(
        select_scores(*padded, "specials", 7)
    )
    score.copy_(select_scores(heads, rows + 3, max_block, kind, batch + context))
    bt = torch.randint(
        0, 1 << 20, (batch, max_block + 5), dtype=torch.int32, device="cuda"
    )
    ref = ref_select(score, seq_lens, query_len, topk, init_blocks, local_blocks)
    # -7 poison: a slot the kernel misses shows
    ids = torch.full((heads, rows + 3, topk), -7, dtype=torch.int32, device="cuda")
    kw = {}
    if emit:
        ppb, bps = emit
        sbt = torch.full(
            (rows * heads + 2, topk * ppb), -7, dtype=torch.int32, device="cuda"
        )
        sctx = torch.full((rows * heads + 2,), -7, dtype=torch.int32, device="cuda")
        kw = {
            "block_table": bt,
            "sparse_bt": sbt,
            "sparse_ctx": sctx,
            "pages_per_block": ppb,
            "block_page_stride": bps,
        }

    def flydsl():
        index_select_decode(
            score,
            seq_lens,
            ids,
            query_len,
            init_blocks,
            local_blocks,
            waves=waves,
            **kw,
        )
        return ids

    _, us = run_perftest(flydsl)
    assert torch.equal(ids[:, :rows], ref), "top-k ids differ from ATOM's selection"
    assert (ids[:, rows:] == -7).all(), "wrote a row past the batch"
    if emit:
        rsbt, rsctx = ref_emit(ref, seq_lens, bt, query_len, *emit)
        assert torch.equal(sbt[: rows * heads].cpu(), rsbt), "sparse table differs"
        assert torch.equal(sctx[: rows * heads].cpu(), rsctx), "context differs"
        assert (sbt[rows * heads :] == -7).all() and (sctx[rows * heads :] == -7).all()
    return {"gfx": get_gfx_runtime(), "select us": us}


@benchmark()
def test_index_topk_decode_nvfp4(batch, context, query_len, heads, topk, ring, lens):
    """The fused op end to end: its scores within the WMMA bound of the
    reference, and its selection exactly the reference selection of those
    scores (scores are not bitwise the reference's, so the ids are graded on
    the op's own)."""
    from aiter.ops.flydsl.index_score_decode import (
        index_topk_decode_nvfp4,
        index_topk_decode_score_plane,
    )

    sm_scale = D**-0.5
    q, payload, scale, bt, seq_lens = make_inputs(lens, query_len, heads)
    max_block = bt.shape[1]
    rows = batch * query_len
    score = index_topk_decode_score_plane(heads, rows, max_block, "cuda")
    score.fill_(float("nan"))
    ids = torch.full((heads, rows, topk), -7, dtype=torch.int32, device="cuda")
    sbt = torch.full((rows * heads, topk), -7, dtype=torch.int32, device="cuda")
    sctx = torch.full((rows * heads,), -7, dtype=torch.int32, device="cuda")

    def flydsl():
        index_topk_decode_nvfp4(
            q, payload, scale, score, bt, seq_lens, sm_scale, query_len, ids, 1, 1,
            sparse_bt=sbt, sparse_ctx=sctx, ring=ring,
        )  # fmt: skip
        return ids

    _, us = run_perftest(flydsl)
    ref, live, tol = run_torch(
        q, payload, scale, bt, seq_lens, sm_scale, query_len, max_block
    )
    graded = live & (ref != float("-inf"))
    assert (((score - ref).abs() <= tol) | ~graded).all(), "scores off the bound"
    assert torch.equal(ids, ref_select(score, seq_lens, query_len, topk, 1, 1))
    rsbt, rsctx = ref_emit(ids, seq_lens, bt, query_len, 1, 1)
    assert torch.equal(sbt.cpu(), rsbt) and torch.equal(sctx.cpu(), rsctx)
    blocks = sum(min(-(-n // PAGE), max_block) for n in lens)
    return {"gfx": get_gfx_runtime(), "us": us, "TB/s": blocks * PAGE_BYTES / us / 1e6}


def request_lens(batch, context, ragged):
    """``batch`` request lengths: all ``context``, or ragged below it (down to
    one token, block boundaries included)."""
    if not ragged:
        return [context] * batch
    edge = [context, context - 1, 129, 1, context // 2 + 1, 128, 7, 127]
    g = torch.Generator().manual_seed(batch + context)
    rest = torch.randint(1, context + 1, (max(0, batch - len(edge)),), generator=g)
    return (edge + rest.tolist())[:batch]


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "index_score_decode_nvfp4 unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[4, 8, 16])
    parser.add_argument(
        "-c",
        "--context",
        type=int,
        nargs="*",
        default=[50000, 150000, 300000],
        help="context tokens a request",
    )
    parser.add_argument(
        "-q",
        "--rows",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8, 4), (8, 1), (1, 4), (1, 1), (3, 5)],
        help="(query rows a request, index heads), e.g. 8,4",
    )
    parser.add_argument(
        "-w",
        "--workgroups",
        type=int,
        nargs="*",
        default=[0],
        help="workgroup budget of the launch (0: the op's default)",
    )
    parser.add_argument("-r", "--ring", type=int, nargs="*", default=[4])
    parser.add_argument(
        "--ragged", type=int, nargs="*", default=[0, 1], help="1: mixed lengths"
    )
    parser.add_argument(
        "--only",
        choices=["score", "select", "fused"],
        nargs="*",
        default=["score", "select", "fused"],
        help="which of the three tests to run",
    )
    args = parser.parse_args()

    if "score" in args.only:
        cells = list(
            itertools.product(
                args.ragged,
                args.batch,
                args.context,
                args.rows,
                args.workgroups,
                args.ring,
            )
        )
        # Launch knobs off their defaults: one workgroup a 64-block window (the
        # widest chunk), a large budget, the shortest and a long ring.
        cells += [
            (1, 8, 64 * 18 * PAGE, (8, 4), workgroups, ring)
            for workgroups, ring in ((1, 4), (1, 2), (96, 6), (4096, 3))
        ]
        # The flat plan at its widest split (even lengths, budget 1: S = 128)
        # and past one wave of requests.
        cells += [
            (0, 4, 256 * PAGE, (8, 4), 1, 4),
            (0, 4, 256 * PAGE, (8, 4), 1, 8),
            (1, 44, 40000, (8, 4), 0, 4),
            (0, 44, 20000, (8, 4), 1, 4),
        ]
        df = []
        for (
            ragged,
            batch,
            context,
            (query_len, heads),
            workgroups,
            ring,
        ) in cells:
            df.append(
                test_index_score_decode_nvfp4(
                    batch,
                    context,
                    query_len,
                    heads,
                    workgroups,
                    ring,
                    request_lens(batch, context, ragged),
                )
            )
        df = pd.DataFrame(df).drop(columns=["lens"], errors="ignore")
        aiter.logger.info(
            "index_score_decode_nvfp4 summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    if "select" in args.only:
        # The selector: D's shapes, graph pads (a request shorter than its rows),
        # rows of 0-34 blocks (around a lane's and a wave's width), the widest
        # plane it serves at top-16 (8192 blocks, the winners in the last
        # columns when ascending), every score kind, top-k 16 and an odd 13, the
        # emit off / page-128 / page-16.
        edges = [n * PAGE + 7 for n in (0, 1, 15, 16, 17, 31, 32, 33)]
        df = []
        shapes = (
            (16, 88000),
            (16, 600000),
            (3, 1000),
            (5, 300),
            (8, 34 * PAGE),
            (2, 8192 * PAGE),
        )
        # every one-wave cell first: a four-wave fault cannot hide them
        for waves, (batch, context) in itertools.product((1, 4), shapes):
            lens = request_lens(batch, context, 1)
            if batch == 8:
                lens = edges
            lens[-1] = 0  # a graph pad request
            for kind, (topk, init, local), emit in itertools.product(
                ("random", "ties", "specials", "ascending", "constant"),
                ((16, 1, 1), (16, 0, 0), (13, 2, 3)),
                (None, (1, 1), (8, 16)),
            ):
                df.append(
                    test_index_select_decode(
                        batch, context, 8, 4, topk, init, local, kind, emit, waves, lens
                    )
                )
        df = pd.DataFrame(df).drop(columns=["lens"], errors="ignore")
        aiter.logger.info(
            "index_select_decode summary (markdown):\n%s", df.to_markdown(index=False)
        )

    if "fused" in args.only:
        df = []
        for batch, context, ragged in ((15, 88000, 0), (16, 150000, 1), (4, 50000, 1)):
            for ring in (4, 6):
                df.append(
                    test_index_topk_decode_nvfp4(
                        batch,
                        context,
                        8,
                        4,
                        16,
                        ring,
                        request_lens(batch, context, ragged),
                    )
                )
        df = pd.DataFrame(df).drop(columns=["lens"], errors="ignore")
        aiter.logger.info(
            "index_topk_decode_nvfp4 summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
