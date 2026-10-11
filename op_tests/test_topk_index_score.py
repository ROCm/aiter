# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness, refusals and perf for the MiniMax-M3 decode index block-score.

The reference here is torch, to a tolerance: this does NOT establish bit-pattern
equality with the framework's Triton kernel, which is not in this repository.
The refusal cases, by contrast, are exact.
"""

import argparse
import sys

import pandas as pd
import torch

from aiter.ops.topk_index_score import (
    OPUS_BUILT_CELLS,
    OPUS_CERTIFIED_CELLS,
    topk_index_score_decode,
    topk_index_score_decode_supported,
)
from aiter.test_common import benchmark, checkAllclose, perftest

BLOCK_SIZE = 128
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn
DEV = "cuda"
LOG2E = 1.4426950408889634
SM_SCALE = HEAD_DIM**-0.5

# A block a row cannot see keeps the -inf the score buffer was filled with.
# Comparisons run on a finite stand-in so a matching pair differences to 0
# rather than nan -- the same device the sibling test uses, for the same reason.
NEG = -1e4


def _setup(batch, ctx, heads, max_q, seed=0):
    """bf16 index query, fp8 index cache, block table, seq_lens, score buffer."""
    total_q = batch * max_q
    max_blk = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_pages = batch * max_blk + 4

    torch.manual_seed(seed)
    q = (torch.randn(total_q, heads, HEAD_DIM, device=DEV) / 4).to(torch.bfloat16)
    # Filled a slice at a time: drawing the whole cache in fp32 first needs four
    # times the cache itself, which is what runs out at the top of the sweep.
    k = torch.empty(num_pages, BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=FP8)
    for i in range(0, num_pages, 4096):
        n = min(4096, num_pages - i)
        k[i : i + n] = (torch.randn(n, BLOCK_SIZE, HEAD_DIM, device=DEV) / 4).to(FP8)
    block_table = torch.arange(batch * max_blk, device=DEV, dtype=torch.int32).view(
        batch, max_blk
    )
    seq = torch.full((batch,), ctx, device=DEV, dtype=torch.int32)
    score = torch.full(
        (heads, total_q, max_blk), -float("inf"), device=DEV, dtype=torch.float32
    )
    return q, k, block_table, seq, score, max_blk


def ref_index_scores(q, k, block_table, seq_lens, max_q, max_blk):
    """score[h, b*Q + t, blk] = max_p( fp32dot(bf16(K), Q) * sm_scale * log2e ).

    One einsum per request: a per-token loop over blocks is far too slow to be a
    useful check at these context lengths.
    """
    total_q, heads, _ = q.shape
    batch = total_q // max_q
    out = torch.full(
        (heads, total_q, max_blk), -float("inf"), device=q.device, dtype=torch.float32
    )
    for r in range(batch):
        seq = int(seq_lens[r])
        live = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[r, :live].long()
        kk = k[pages].to(torch.float32).reshape(live * BLOCK_SIZE, HEAD_DIM)
        for t in range(max_q):
            row = r * max_q + t
            # Token t of a request sees seq - max_q + t + 1 keys (the causal cut
            # the kernel hoists out of its inner loop).
            causal = seq - max_q + t + 1
            if causal <= 0:
                continue
            qq = q[row].to(torch.float32)
            dots = (kk[:causal] @ qq.t()) * (SM_SCALE * LOG2E)
            nblk = (causal + BLOCK_SIZE - 1) // BLOCK_SIZE
            pad = nblk * BLOCK_SIZE - causal
            if pad:
                dots = torch.cat(
                    [dots, torch.full((pad, heads), -float("inf"), device=q.device)]
                )
            out[:, row, :nblk] = (
                dots.view(nblk, BLOCK_SIZE, heads).max(dim=1).values.t()
            )
    return out


@perftest()
def run_port(q, k, score, block_table, seq_lens, max_q, ctx):
    topk_index_score_decode(
        q,
        k,
        score,
        block_table,
        seq_lens,
        SM_SCALE,
        query_len=max_q,
        max_seq_len=ctx,
    )
    return score


# Named bench_*, not test_*: pytest collects test_* names and would fail this
# one on its required arguments. The collectable entry points take none.
@benchmark()
def bench_index_score(batch, ctx, heads, max_q):
    q, k, bt, seq, score, max_blk = _setup(batch, ctx, heads, max_q)
    ref = ref_index_scores(q, k, bt, seq, max_q, max_blk)
    out, us = run_port(q, k, score, bt, seq, max_q, ctx)

    a = torch.nan_to_num(out, neginf=NEG)
    b = torch.nan_to_num(ref, neginf=NEG)
    # bf16 x fp8 through an fp32 accumulator against an fp32 einsum: the
    # tolerance is the accumulation order, not the kernel.
    err = checkAllclose(b, a, rtol=1e-2, atol=1e-2, msg="index block scores")
    return {"us": us, "err": err}


# --------------------------------------------------------------------------- #
# The refusal half of the contract. Exact: a refusal either happens or it does
# not. Each case names what a caller would actually get wrong.
# --------------------------------------------------------------------------- #
def _tiny(heads=1, max_q=1, ctx=1024):
    return _setup(1, ctx, heads, max_q)


def _must_refuse(fn):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        return True, f"{type(e).__name__}: {str(e)[:110]}"
    return False, "NO REFUSAL -- accepted something it documents as unsupported"


def test_aux_k_legs():
    """D08 / rule A3 (review): both compiled cache-policy legs, for every cell."""
    if not torch.cuda.is_available():
        return
    for heads, max_q in OPUS_CERTIFIED_CELLS:
        q, k, bt, seq, score, max_blk = _setup(2, 2048, heads, max_q)
        ref = ref_index_scores(q, k, bt, seq, max_q, max_blk)
        out = {}
        for aux in (0, 3):
            buf = torch.full_like(score, -float("inf"))
            topk_index_score_decode(
                q,
                k,
                buf,
                bt,
                seq,
                SM_SCALE,
                query_len=max_q,
                max_seq_len=2048,
                aux_k=aux,
            )
            torch.cuda.synchronize()
            out[aux] = buf.clone()
            a = torch.nan_to_num(buf, neginf=NEG)
            b = torch.nan_to_num(ref, neginf=NEG)
            assert torch.allclose(
                a, b, rtol=1e-2, atol=1e-2
            ), f"cell ({heads},{max_q}) aux_k={aux} does not match the reference"
        # The two legs differ only in a cache hint, so they must agree BIT for
        # bit with each other even though neither is compared bitwise to torch.
        assert torch.equal(out[0].view(torch.int32), out[3].view(torch.int32)), (
            f"cell ({heads},{max_q}): aux_k 0 and 3 disagree bitwise, so the cache "
            "policy is changing the result"
        )


def test_refusals():
    """Every one of these is a shape or dtype somebody will pass by accident."""
    cases = []

    def case(name, fn):
        ok, why = _must_refuse(fn)
        cases.append({"case": name, "refused": ok, "detail": why})

    q, k, bt, seq, score, _ = _tiny()

    case(
        "fp8 query (the incumbent contract, not this one)",
        lambda: topk_index_score_decode(
            q.to(FP8), k, score, bt, seq, SM_SCALE, query_len=1, max_seq_len=1024
        ),
    )
    case(
        "bf16 key cache (ATOM default for MiniMax-M3 -- must route, not crash)",
        lambda: topk_index_score_decode(
            q,
            k.to(torch.bfloat16),
            score,
            bt,
            seq,
            SM_SCALE,
            query_len=1,
            max_seq_len=1024,
        ),
    )
    case(
        "fp16 score buffer",
        lambda: topk_index_score_decode(
            q,
            k,
            score.to(torch.float16),
            bt,
            seq,
            SM_SCALE,
            query_len=1,
            max_seq_len=1024,
        ),
    )
    case(
        "max_seq_len omitted (the grid must not come from seq_lens)",
        lambda: topk_index_score_decode(q, k, score, bt, seq, SM_SCALE, query_len=1),
    )
    case(
        "int64 block table",
        lambda: topk_index_score_decode(
            q, k, score, bt.long(), seq, SM_SCALE, query_len=1, max_seq_len=1024
        ),
    )
    case(
        "non-finite sm_scale",
        lambda: topk_index_score_decode(
            q, k, score, bt, seq, float("nan"), query_len=1, max_seq_len=1024
        ),
    )
    case(
        "score strided along the block axis",
        lambda: topk_index_score_decode(
            q,
            k,
            torch.full((1, 1, 16, 2), -float("inf"), device=DEV)[..., 0],
            bt,
            seq,
            SM_SCALE,
            query_len=1,
            max_seq_len=1024,
        ),
    )

    # An unbuilt cell: (2, 1) is TP2, deliberately absent. H*Q = 2 is inside the
    # MFMA column budget, so this is a build-table refusal and not an
    # architectural one -- which is exactly why it must be tested: the two
    # failure modes have different fixes.
    if (2, 1) not in OPUS_BUILT_CELLS:
        q2, k2, bt2, seq2, sc2, _ = _tiny(heads=2)
        case(
            "unbuilt cell (2,1) -- TP2",
            lambda: topk_index_score_decode(
                q2, k2, sc2, bt2, seq2, SM_SCALE, query_len=1, max_seq_len=1024
            ),
        )

    # The output-aliasing and descriptor-extent gates added on review findings
    # D03 and D01. Both were reachable before and silently wrong.
    case(
        "expanded score view (heads alias one output element)",
        lambda: topk_index_score_decode(
            _setup(1, 1024, 4, 1)[0],
            k,
            torch.empty(1, 1, 8, dtype=torch.float32, device=DEV).expand(4, 1, 8),
            bt,
            seq,
            SM_SCALE,
            query_len=1,
            max_seq_len=1024,
        ),
    )
    case(
        "CPU score tensor (every tensor is forwarded as a raw pointer)",
        lambda: topk_index_score_decode(
            q, k, score.cpu(), bt, seq, SM_SCALE, query_len=1, max_seq_len=1024
        ),
    )

    # H*Q over the 16 MFMA columns: an architectural wall, not a table gap.
    q3, k3, bt3, seq3, sc3, _ = _setup(1, 1024, 4, 8)
    case(
        "H*Q = 32 exceeds the 16 MFMA columns",
        lambda: topk_index_score_decode(
            q3, k3, sc3, bt3, seq3, SM_SCALE, query_len=8, max_seq_len=1024
        ),
    )

    df = pd.DataFrame(cases)
    print(df)
    bad = [c for c in cases if not c["refused"]]
    assert not bad, f"these were ACCEPTED and must not be: {[c['case'] for c in bad]}"
    return cases


def test_supported_predicate():
    """The routable predicate must agree with what the entry actually does.

    Two copies of a rule need a check that executes both, or they drift: this is
    the check.
    """
    q, k, _, _, score, _ = _tiny()
    ok, reason = topk_index_score_decode_supported(
        q, k, score, query_len=1, max_seq_len=1024
    )
    assert ok, f"predicate refuses a case the entry accepts: {reason}"
    ok_bf16, reason_bf16 = topk_index_score_decode_supported(
        q, k.to(torch.bfloat16), score, query_len=1, max_seq_len=1024
    )
    assert not ok_bf16, "predicate accepts bf16 K while the entry refuses it"
    print(f"supported(): fp8 K -> True; bf16 K -> False ({reason_bf16})")
    return True


def test_index_score():
    """The collectable correctness entry point: no arguments, so pytest runs it."""
    if not torch.cuda.is_available():
        return
    for heads, max_q in OPUS_CERTIFIED_CELLS:
        q, k, bt, seq, score, max_blk = _setup(2, 2048, heads, max_q)
        ref = ref_index_scores(q, k, bt, seq, max_q, max_blk)
        topk_index_score_decode(
            q, k, score, bt, seq, SM_SCALE, query_len=max_q, max_seq_len=2048
        )
        torch.cuda.synchronize()
        a = torch.nan_to_num(score, neginf=NEG)
        b = torch.nan_to_num(ref, neginf=NEG)
        assert torch.allclose(
            a, b, rtol=1e-2, atol=1e-2
        ), f"cell ({heads},{max_q}) does not match the torch reference"


l_batch = [1, 8, 40]
l_ctx = [2048, 32768, 102400]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="decode index block-score: correctness, refusals, perf"
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=None)
    parser.add_argument("-c", "--ctx", type=int, nargs="*", default=None)
    parser.add_argument(
        "--cells",
        type=str,
        default=None,
        help="comma-separated HxQ pairs, e.g. 1x1,1x4; default = the certified set",
    )
    args = parser.parse_args()
    if args.batch is not None:
        l_batch = args.batch
    if args.ctx is not None:
        l_ctx = args.ctx
    if args.cells:
        cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")]
    else:
        cells = list(OPUS_CERTIFIED_CELLS)

    if not torch.cuda.is_available():
        print("no GPU; nothing to test")
        sys.exit(0)

    print(f"certified cells: {OPUS_CERTIFIED_CELLS}")
    print(f"built cells    : {OPUS_BUILT_CELLS}")
    test_supported_predicate()
    test_refusals()
    test_aux_k_legs()

    df = []
    for heads, max_q in cells:
        for ctx in l_ctx:
            for batch in l_batch:
                ret = bench_index_score(batch, ctx, heads, max_q)
                df.append(ret)
    df = pd.DataFrame(df)
    print(df)
