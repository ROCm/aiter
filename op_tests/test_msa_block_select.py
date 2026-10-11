# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the MSA sparse block-select passes.

Covers the two scoring kernels (uniform-length decode and ragged prefill) and
the top-k pass that turns their scores into a sparse block table. Every case is
checked against a torch reference built with one einsum per request; a per-token
loop over blocks is far too slow to be a useful check at these context lengths.

A last sweep scores the same decode shapes with the block axis partitioned, to
put the per-rank cost of context parallelism beside the unpartitioned one. It
needs no second process: the pass is rank-local, so a shard costs the same
whether its peers are running or not.
"""

import argparse
import math
import sys

import pandas as pd
import torch

import aiter
from aiter.ops.msa_block_select import (
    pa_sparse_block_score_decode,
    pa_sparse_block_score_prefill,
    pa_sparse_block_topk,
)
from aiter.test_common import benchmark, checkAllclose, perftest

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
WAVE = 64
FP8 = torch.float8_e4m3fn
DEV = "cuda"

# A block a row cannot see keeps the -inf the score buffer was filled with.
# Comparisons run on a finite stand-in so a matching pair differences to 0
# rather than nan.
NEG = -1e4


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def _slots(max_blk: int) -> int:
    """Score-row width the top-k pass requires: whole 64-wide strips, pow2 many."""
    return _pow2_ceil(math.ceil(max_blk / WAVE))


def _setup(seq_lens, qlens, num_idx_heads):
    """Index-cache query/key pair plus the block table and score buffer."""
    num_reqs = len(seq_lens)
    total_q = sum(qlens)
    max_blk = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    num_pages = num_reqs * max_blk + 4

    torch.manual_seed(0)
    q = (torch.randn(total_q, num_idx_heads, HEAD_DIM, device=DEV) / 4).to(FP8)
    # Filled a slice at a time: drawing the whole cache in fp32 first needs four
    # times the cache itself, which is what runs out at the top of the sweep.
    k = torch.empty(num_pages, BLOCK_SIZE, HEAD_DIM, device=DEV, dtype=FP8)
    for i in range(0, num_pages, 4096):
        n = min(4096, num_pages - i)
        k[i : i + n] = (torch.randn(n, BLOCK_SIZE, HEAD_DIM, device=DEV) / 4).to(FP8)
    block_table = torch.arange(num_reqs * max_blk, device=DEV, dtype=torch.int32).view(
        num_reqs, max_blk
    )
    seq = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    score = torch.full(
        (num_idx_heads, total_q, _slots(max_blk) * WAVE),
        -float("inf"),
        device=DEV,
    )
    return q, k, block_table, seq, score, max_blk


def ref_block_scores(q, k, block_table, seq_lens, qlens, num_slots):
    """score[h, n, b] = max over the causal tokens of block b of q[n,h,:] . k[b,t,:]."""
    total_q, heads, _ = q.shape
    ref = torch.full((heads, total_q, num_slots), -float("inf"), device=DEV)
    qf = q.float()
    row = 0
    for r, qlen in enumerate(qlens):
        seq_len = int(seq_lens[r])
        nblk = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = block_table[r, :nblk].long()
        # Gathered before the upcast: one request's pages are a slice of the
        # cache, and promoting the whole cache to fp32 is what will not fit.
        # [nblk, BLOCK, D] x [qlen, H, D] -> [nblk, qlen, H, BLOCK]
        prod = torch.einsum("btd,qhd->bqht", k[pages].float(), qf[row : row + qlen])

        # Token t of block b is visible to query token j only while it precedes
        # that row's causal length; everything past it must not enter the max.
        kv_len = seq_len - qlen + torch.arange(qlen, device=DEV) + 1
        tok = (
            torch.arange(nblk, device=DEV)[:, None] * BLOCK_SIZE
            + torch.arange(BLOCK_SIZE, device=DEV)[None, :]
        )
        visible = tok[:, None, :] < kv_len[None, :, None]
        prod = prod.masked_fill(~visible[:, :, None, :], -float("inf"))

        ref[:, row : row + qlen, :nblk] = prod.max(dim=-1).values.permute(2, 1, 0)
        row += qlen
    return ref


def _finite(t):
    return torch.nan_to_num(t, neginf=NEG)


def _ragged_qlens(batch, max_query_len):
    """Deterministic mix of query lengths, at least one row hitting the max."""
    return [1 + (i * 7) % max_query_len for i in range(batch - 1)] + [max_query_len]


@perftest()
def run_score_decode(q, k, score, block_table, seq, query_len, max_seq_len):
    pa_sparse_block_score_decode(
        q, k, score, block_table, seq, query_len=query_len, max_seq_len=max_seq_len
    )
    return score


# Rotation left on, which matters more here than anywhere else in this file.
# It cycles the arguments across enough copies to stop an iteration finding the
# last one's reads still cached, and the whole-context pass does not notice --
# 537MB at batch 32 runs past the 256MB of MALL either way, 89.4us rotated
# against 89.8 not. A shard reads a quarter of that, which does fit, so
# re-reading one buffer measures 21.6us where a cold one measures 33.2. Cold is
# the one a model runs: every indexer layer owns its own index cache, so no
# layer's shard is still resident when the next layer's score pass starts.
@perftest()
def run_score_decode_shard(q, k, score, block_table, seq, max_seq_len, rank, world):
    pa_sparse_block_score_decode(
        q,
        k,
        score,
        block_table,
        seq,
        query_len=1,
        max_seq_len=max_seq_len,
        rank=rank,
        world=world,
    )
    return score


@perftest()
def run_score_prefill(q, k, score, block_table, cu, seq, max_query_len, max_seq_len):
    pa_sparse_block_score_prefill(
        q,
        k,
        score,
        block_table,
        cu,
        seq,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
    )
    return score


@perftest()
def run_topk(score, topk_idx, block_table, seq, max_seq_len, query_len, num_kv_heads):
    return pa_sparse_block_topk(
        score,
        topk_idx,
        block_table,
        seq,
        max_seq_len=max_seq_len,
        block_size=BLOCK_SIZE,
        query_len=query_len,
        num_kv_heads=num_kv_heads,
    )


@benchmark()
def test_score_decode(num_idx_heads: int, batch: int, ctx: int, query_len: int):
    """Uniform query length, the shape decode and speculative decode produce."""
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    qlens = [query_len] * batch
    q, k, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_idx_heads)

    score, avg_us = run_score_decode(
        q, k, score, bt, seq, query_len, max_seq_len=max(seq_lens)
    )

    ref = ref_block_scores(q, k, bt, seq, qlens, score.size(2))
    info = f"H:{num_idx_heads}, batch:{batch}, ctx:{ctx}, qlen:{query_len}"
    checkAllclose(
        _finite(ref[:, :, :max_blk]),
        _finite(score[:, :, :max_blk]),
        msg=f"[perf] === {info} === decode scoring {avg_us:<8.2f} us",
        rtol=1e-2,
        atol=1e-2,
    )
    return {"pass": "decode", "us": avg_us}


@benchmark()
def test_score_decode_cp(num_idx_heads: int, batch: int, ctx: int, world: int):
    """Decode scoring with the block axis cut `world` ways, against scoring it whole.

    Context partitioning does not give this pass a kernel of its own: it is the
    same launch with a shard passed in, which is what makes the comparison a
    fair one to put side by side. Every rank scores every index head -- the
    model's whole count, not the one a tensor-parallel split would leave it --
    over the blocks where ``b % world == rank``, so what a rank reads is
    ``1/world`` of the index cache and the win is expected to track that.

    Timed single-process, one rank at a time, which costs nothing in fidelity:
    the pass is rank-local, so the only thing partitioning changes about it is
    how many blocks it walks. ``shard_us`` is the slowest rank, since that is
    the one a step waits for. Both sides are timed by the same wrapper, so the
    ratio is of like with like, and both are timed cold -- see above for why a
    warm shard flatters this by nearly 2x.

    The speedup trails ``world`` rather than reaching it, and by more at small
    batches: a shard is a quarter of the reads but not a quarter of the launch,
    and a smaller read saturates less of the bus.
    """
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    qlens = [1] * batch
    q, k, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_idx_heads)
    max_seq_len = max(seq_lens)
    info = f"H:{num_idx_heads}, batch:{batch}, ctx:{ctx}, world:{world}"

    full, full_us = run_score_decode_shard(q, k, score, bt, seq, max_seq_len, 0, 1)
    ref = _finite(ref_block_scores(q, k, bt, seq, qlens, score.size(2)))
    checkAllclose(
        ref[:, :, :max_blk],
        _finite(full[:, :, :max_blk]),
        msg=f"[perf] === {info} === whole-context scoring {full_us:<8.2f} us",
        rtol=1e-2,
        atol=1e-2,
    )

    # A shard row is as wide as the longest shard, which is rank 0's.
    total_q = score.size(1)
    slots = _slots(math.ceil(max_blk / world)) * WAVE
    shard_us = 0.0
    for rank in range(world):
        shard = torch.full((num_idx_heads, total_q, slots), -float("inf"), device=DEV)
        shard, us = run_score_decode_shard(
            q, k, shard, bt, seq, max_seq_len, rank, world
        )
        shard_us = max(shard_us, us)
        # global = local * world + rank, so the reference for this shard is the
        # whole-context one strided. Checking every rank is what shows the
        # shards partition the context rather than merely covering part of it.
        nloc = math.ceil((max_blk - rank) / world)
        checkAllclose(
            ref[:, :, rank:max_blk:world],
            _finite(shard[:, :, :nloc]),
            msg=f"[perf] === {info} === rank {rank} shard scoring {us:<8.2f} us",
            rtol=1e-2,
            atol=1e-2,
        )

    return {
        "pass": "cp_score",
        "full_us": full_us,
        "shard_us": shard_us,
        "speedup": full_us / shard_us if shard_us else float("nan"),
    }


@benchmark()
def test_score_prefill(num_idx_heads: int, batch: int, ctx: int, max_query_len: int):
    """Ragged query lengths, which is the only form prefill can use."""
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    qlens = _ragged_qlens(batch, max_query_len)
    q, k, bt, seq, score, max_blk = _setup(seq_lens, qlens, num_idx_heads)
    cu = torch.tensor(
        [0] + list(torch.tensor(qlens).cumsum(0)), device=DEV, dtype=torch.int32
    )

    score, avg_us = run_score_prefill(
        q, k, score, bt, cu, seq, max_query_len, max_seq_len=max(seq_lens)
    )

    ref = ref_block_scores(q, k, bt, seq, qlens, score.size(2))
    info = f"H:{num_idx_heads}, batch:{batch}, ctx:{ctx}, max_qlen:{max_query_len}"
    checkAllclose(
        _finite(ref[:, :, :max_blk]),
        _finite(score[:, :, :max_blk]),
        msg=f"[perf] === {info} === prefill scoring {avg_us:<8.2f} us",
        rtol=1e-2,
        atol=1e-2,
    )
    return {"pass": "prefill", "us": avg_us}


@benchmark()
def test_topk(num_idx_heads: int, batch: int, ctx: int, query_len: int):
    """Selection is checked by score, not by index, so ties are not a failure."""
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(batch)]
    qlens = [query_len] * batch
    q, k, bt, seq, score, _ = _setup(seq_lens, qlens, num_idx_heads)

    pa_sparse_block_score_decode(
        q, k, score, bt, seq, query_len=query_len, max_seq_len=max(seq_lens)
    )
    total_q = score.size(1)
    topk_idx = torch.empty(num_idx_heads, total_q, TOPK, dtype=torch.int32, device=DEV)

    _, avg_us = run_topk(
        score, topk_idx, bt, seq, max(seq_lens), query_len, num_idx_heads
    )

    # A kept block must be one of the row's true top-k. Comparing the scores of
    # what was kept rather than the block ids lets an equal-scoring substitute
    # pass, which fp8 rounding can produce near a tie.
    ref = _finite(ref_block_scores(q, k, bt, seq, qlens, score.size(2)))
    want = ref.topk(TOPK, dim=-1).values.sort(dim=-1, descending=True).values
    got = torch.gather(ref, 2, topk_idx.clamp(min=0).long())
    got = torch.where(topk_idx >= 0, got, torch.full_like(got, NEG))
    got = got.sort(dim=-1, descending=True).values

    info = f"H:{num_idx_heads}, batch:{batch}, ctx:{ctx}, qlen:{query_len}"
    checkAllclose(
        want,
        got,
        msg=f"[perf] === {info} === top-k {avg_us:<8.2f} us",
        rtol=1e-2,
        atol=1e-2,
    )
    return {"pass": "topk", "us": avg_us}


l_num_idx_heads = [1, 2]
l_batch = [4, 8, 16, 32, 64, 128]
l_ctx = [4096, 8192, 16384, 32768, 65536, 128000]
l_query_len = [1, 4]

# The context-partitioned sweep is its own, and narrower: it exists to show
# what sharding the block axis buys at the one shape that pays for it, which is
# a long context. A rank scores every index head, so the head count here is the
# model's and not the per-rank one the other sweeps use.
l_cp_num_idx_heads = [4]
l_cp_batch = [15, 32, 64]
l_cp_ctx = [128 * 1024]
l_cp_world = [4]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the MSA sparse block-select scoring and top-k passes",
)
parser.add_argument(
    "-H",
    "--num_idx_heads",
    type=int,
    nargs="*",
    default=None,
    help="Index heads per rank. Only 1 and 2 are built. e.g. -H 1 2",
)
parser.add_argument(
    "-b", "--batch", type=int, nargs="*", default=None, help="Requests per launch"
)
parser.add_argument(
    "-c", "--ctx", type=int, nargs="*", default=None, help="Context length in tokens"
)
parser.add_argument(
    "-q",
    "--query_len",
    type=int,
    nargs="*",
    default=None,
    help="Query tokens per request. num_idx_heads * query_len must not exceed 16",
)
parser.add_argument(
    "-w",
    "--world",
    type=int,
    nargs="*",
    default=None,
    help="Ranks the block axis is partitioned across. 1 skips the CP sweep",
)
parser.add_argument(
    "--cp-only",
    action="store_true",
    help="Run only the partitioned scoring sweep, not the unpartitioned passes",
)
args = parser.parse_args()

# The scoring passes are tested only on the gfx950.
current_gfx = aiter.get_gfx()
if current_gfx != "gfx950":
    print(f"Skipping test_msa_block_select.py: requires gfx950, got {current_gfx}")
    sys.exit(0)

if args.num_idx_heads is not None:
    l_num_idx_heads = args.num_idx_heads
if args.batch is not None:
    l_batch = args.batch
if args.ctx is not None:
    l_ctx = args.ctx
if args.query_len is not None:
    l_query_len = args.query_len
# A narrowing flag narrows every sweep, so one -b or -c means one thing.
if args.num_idx_heads is not None:
    l_cp_num_idx_heads = args.num_idx_heads
if args.batch is not None:
    l_cp_batch = args.batch
if args.ctx is not None:
    l_cp_ctx = args.ctx
if args.world is not None:
    l_cp_world = args.world

rows = {"decode": [], "prefill": [], "topk": [], "cp_score": []}
for num_idx_heads in [] if args.cp_only else l_num_idx_heads:
    for batch in l_batch:
        for ctx in l_ctx:
            for query_len in l_query_len:
                # One MFMA carries the whole head dim, so the heads and the
                # query tokens share its 16 columns.
                if num_idx_heads * query_len > 16:
                    continue
                rows["decode"].append(
                    test_score_decode(
                        num_idx_heads=num_idx_heads,
                        batch=batch,
                        ctx=ctx,
                        query_len=query_len,
                    )
                )
                rows["topk"].append(
                    test_topk(
                        num_idx_heads=num_idx_heads,
                        batch=batch,
                        ctx=ctx,
                        query_len=query_len,
                    )
                )
            rows["prefill"].append(
                test_score_prefill(
                    num_idx_heads=num_idx_heads,
                    batch=batch,
                    ctx=ctx,
                    max_query_len=max(l_query_len),
                )
            )

for num_idx_heads in l_cp_num_idx_heads:
    for batch in l_cp_batch:
        for ctx in l_cp_ctx:
            for world in l_cp_world:
                if world < 2:
                    continue
                rows["cp_score"].append(
                    test_score_decode_cp(
                        num_idx_heads=num_idx_heads,
                        batch=batch,
                        ctx=ctx,
                        world=world,
                    )
                )


def _table(df):
    """Markdown if tabulate is installed, plain columns if it is not."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


for name, r in rows.items():
    if not r:
        continue
    df = pd.DataFrame(r).drop(columns=["pass"])
    aiter.logger.info("msa_block_select %s summary:\n%s", name, _table(df))
