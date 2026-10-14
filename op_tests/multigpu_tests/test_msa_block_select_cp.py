# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multi-rank correctness for the fused CP block selector.

``test_msa_block_select.py`` covers the scoring half of context parallelism,
but it runs one rank at a time in one process: the score pass is rank-local, so
a shard of it can be checked without peers. The selector cannot. The nominate
writes straight into a peer's buffer over IPC and the merge waits on a
generation tag in the candidate rather than on a barrier, so neither the
exchange nor the merge exists until there really are ``world`` processes on
``world`` devices. This file is that run.

What it asserts is the property the whole design rests on: a rank that scored
``1/world`` of the blocks and merged what its peers nominated selects, for each
head it owns, exactly what a rank scoring every block would have selected.
Every rank computes that single-GPU answer itself -- the inputs are drawn from
a fixed seed, so each rank can build the whole context locally -- and compares
it against what came out of ``pa_sparse_block_topk_cp``, per head and per row,
down to the page table the selection expands into.

A few shapes follow from the build rather than from choice. The fused merge
holds one wave of candidates, so ``pa_sparse_block_topk_cp`` refuses a
``world * topk`` above 64; at the built ``SPARSE_TOPK`` of 16 that caps the
worlds here at 4, and 8 ranks is not a configuration the selector has. The
score kernel is built for 1, 2 and 4 index heads, and CP divides the model's
whole count rather than a per-rank one, so the head counts here are 2 and 4.

Each case sweeps every context against every batch, 28 calls through one
candidate buffer and one generation counter. One call would exercise neither
the double buffer nor the generation tag: both only do anything once a call
has a predecessor whose candidates are still in the buffer. Varying the batch
underneath a fixed ``cand_rows`` is what puts a predecessor there, and varying
the context on top of that moves the slot count without moving the layout,
which is the arrangement the served path has. The sequence is identical on every rank,
which is the kernel's one hard requirement -- generations are counted per block
index, so a rank that made a different number of calls would compare tags
against a different call's and the merge would wait forever. That failure mode
is why the parent enforces a deadline instead of joining indefinitely: a desync
has to come back as a failed test and not as a wedged CI job.
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import aiter
from aiter.dist.utils import get_open_port
from aiter.ops.custom_all_reduce import (
    allocate_meta_buffer,
    free_meta_buffer,
    get_meta_buffer_ipc_handle,
)
from aiter.ops.flydsl.quick_allreduce_int4_ipc import UncachedIpcHeap
from aiter.ops.msa_block_select import (
    CP_MAX_BLOCKS,
    pa_sparse_block_score_decode,
    pa_sparse_block_topk,
    pa_sparse_block_topk_cp,
    topk_cp_candidate_numel,
)

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
PAGES_PER_BLOCK = 8
WAVE = 64
FP8 = torch.float8_e4m3fn

# hipIpcGetMemHandle on an uncached allocation below 2MB exports only if the
# size is a whole number of pages, which the candidate buffer is well under.
_HIP_PAGE = 4096

# Seconds a rank gets before the parent calls it hung. The work itself is
# milliseconds; the slack is for the JIT build of the kernel module on a cold
# cache, which the first rank to get there pays.
_RANK_TIMEOUT = float(os.environ.get("AITER_MSA_CP_TEST_TIMEOUT", "900"))


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def _slots(max_blk: int) -> int:
    """Score-row width the selector requires: whole 64-wide strips, pow2 many."""
    return _pow2_ceil(math.ceil(max_blk / WAVE)) * WAVE


def _ipc_alloc(nbytes: int, rank: int, world: int) -> tuple[torch.Tensor, int, list]:
    """Allocate the candidate buffer here and map every peer's.

    Returns the ``[world]`` int64 CPU tensor of device addresses the kernel
    takes, this rank's own raw pointer, and the mappings to close. Entry
    ``rank`` is local, the rest are IPC mappings.
    """
    nbytes = (nbytes + _HIP_PAGE - 1) // _HIP_PAGE * _HIP_PAGE
    ptr = allocate_meta_buffer(nbytes)

    handle = torch.empty(64, dtype=torch.uint8, device="cpu")
    get_meta_buffer_ipc_handle(ptr, handle.data_ptr())
    handles = [None] * world
    dist.all_gather_object(handles, handle.numpy().tobytes())

    peers, opened = [], []
    for r in range(world):
        if r == rank:
            peers.append(ptr)
            continue
        mapped = UncachedIpcHeap.open_mem_handle(handles[r])
        opened.append(mapped)
        peers.append(mapped)
    return torch.tensor(peers, dtype=torch.int64, device="cpu"), ptr, opened


def _inputs(num_reqs: int, ctx: int, num_idx_heads: int, device):
    """The index-cache query/key pair and block table, identical on every rank.

    Drawn from a fixed seed rather than broadcast: every rank needs the whole
    context anyway to compute the single-GPU answer it is checking against, so
    generating it is cheaper than sending it.
    """
    seq_lens = [ctx - (i % 4) * BLOCK_SIZE for i in range(num_reqs)]
    max_blk = max((s + BLOCK_SIZE - 1) // BLOCK_SIZE for s in seq_lens)
    num_pages = num_reqs * max_blk + 4

    torch.manual_seed(0)
    q = (torch.randn(num_reqs, num_idx_heads, HEAD_DIM, device=device) / 4).to(FP8)
    k = torch.empty(num_pages, BLOCK_SIZE, HEAD_DIM, device=device, dtype=FP8)
    for i in range(0, num_pages, 4096):
        n = min(4096, num_pages - i)
        k[i : i + n] = (torch.randn(n, BLOCK_SIZE, HEAD_DIM, device=device) / 4).to(FP8)
    block_table = torch.arange(
        num_reqs * max_blk, device=device, dtype=torch.int32
    ).view(num_reqs, max_blk)
    seq = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return q, k, block_table, seq, max_blk, max(seq_lens)


def _reference(q, k, block_table, seq, max_blk, max_seq_len, owned_slice, device):
    """The single-GPU selection for the heads one rank owns.

    Scored over the whole context -- ``world`` defaults to 1 -- and selected by
    ``pa_sparse_block_topk``, the pass CP replaces. Sliced to the owned heads
    before selecting so the page table comes out in the same ``num_kv_heads``
    numbering the CP call emits, which makes the two comparable element for
    element rather than through a re-index.
    """
    num_reqs, num_idx_heads, _ = q.shape
    score = torch.full(
        (num_idx_heads, num_reqs, _slots(max_blk)), -float("inf"), device=device
    )
    pa_sparse_block_score_decode(
        q, k, score, block_table, seq, query_len=1, max_seq_len=max_seq_len
    )

    lo, hi = owned_slice
    owned = hi - lo
    topk_idx = torch.empty(owned, num_reqs, TOPK, dtype=torch.int32, device=device)
    rows = num_reqs * owned
    sparse_bt = torch.zeros(
        rows, TOPK * PAGES_PER_BLOCK, dtype=torch.int32, device=device
    )
    sparse_ctx = torch.zeros(rows, dtype=torch.int32, device=device)
    pa_sparse_block_topk(
        score[lo:hi],
        topk_idx,
        block_table,
        seq,
        max_seq_len=max_seq_len,
        block_size=BLOCK_SIZE,
        query_len=1,
        num_kv_heads=owned,
        pages_per_block=PAGES_PER_BLOCK,
        sparse_bt=sparse_bt,
        sparse_ctx=sparse_ctx,
    )
    return score, topk_idx, sparse_bt, sparse_ctx


def _picked_scores(idx, score_owned):
    """Score of each selected block, -inf standing in for the -1 pad."""
    s = torch.gather(score_owned, 2, idx.clamp(min=0).long())
    s = torch.where(idx >= 0, s, torch.full_like(s, -float("inf")))
    return s.sort(dim=-1, descending=True).values


def _compare(got_idx, ref_idx, score_owned, tag):
    """Selected blocks, per head and per row, against the single-GPU selection.

    Compared as sets: the two passes reach the same winners by different
    routes, so nothing fixes the order a tie comes out in. A row that picked a
    different block id is only accepted if the scores it picked are
    bit-identical to the ones it replaced, which is what an fp8 tie looks like
    and is the only difference the merge is allowed to produce.

    Returns the rows whose ids matched exactly, so the caller can hold the page
    table to the stricter standard on them.
    """
    exact = (got_idx.sort(dim=-1).values == ref_idx.sort(dim=-1).values).all(dim=-1)
    if exact.all():
        return exact

    bad = ~exact
    got_s = _picked_scores(got_idx, score_owned)[bad]
    ref_s = _picked_scores(ref_idx, score_owned)[bad]
    if not torch.equal(got_s, ref_s):
        h, n = bad.nonzero()[0].tolist()
        raise AssertionError(
            f"{tag}: head {h} row {n} selected {got_idx[h, n].tolist()} but the "
            f"single-GPU top-k selected {ref_idx[h, n].tolist()}, and the scores "
            f"differ -- this is a wrong selection, not a tie"
        )
    return exact


def _worker(rank, world, port, num_idx_heads, ctxs, batches):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
    )

    owned = num_idx_heads // world
    # Sized for the largest batch and held there for every call, contexts
    # included. The op requires it: the buffer's layout is what makes an
    # address mean one (shard, head, row, nominee), so a stride that followed
    # the batch would leave a candidate where a longer call looks for a
    # different row, and a tag that happened to match would read as fresh.
    cand_rows = max(batches)
    cand_ptrs, own_ptr, opened = _ipc_alloc(
        topk_cp_candidate_numel(world, owned, cand_rows, TOPK) * 8, rank, world
    )
    # Rank-local and never re-zeroed after this: a candidate carries its own
    # generation, so the counter is only ever compared against itself. It
    # spans the whole run, contexts included, which is the point -- the
    # counter has to stay meaningful across every call made through the buffer.
    cp_gen = torch.zeros(CP_MAX_BLOCKS, dtype=torch.int32, device=device)
    lo, hi = rank * owned, (rank + 1) * owned

    try:
        call = 0
        # Every rank reaches the kernel the same number of times with the same
        # shapes, in the same order. That is the invariant the exchange rests
        # on, so these loops are deliberately free of anything rank-dependent.
        for ctx in ctxs:
            q_all, k, bt_all, seq_all, max_blk, max_seq_len = _inputs(
                cand_rows, ctx, num_idx_heads, device
            )
            shard_slots = _slots(math.ceil(max_blk / world))

            for num_reqs in batches:
                q, bt, seq = q_all[:num_reqs], bt_all[:num_reqs], seq_all[:num_reqs]

                full_score, ref_idx, ref_bt, ref_ctx = _reference(
                    q, k, bt, seq, max_blk, max_seq_len, (lo, hi), device
                )

                shard = torch.full(
                    (num_idx_heads, num_reqs, shard_slots), -float("inf"), device=device
                )
                pa_sparse_block_score_decode(
                    q,
                    k,
                    shard,
                    bt,
                    seq,
                    query_len=1,
                    max_seq_len=max_seq_len,
                    rank=rank,
                    world=world,
                )

                got_idx = torch.empty(
                    owned, num_reqs, TOPK, dtype=torch.int32, device=device
                )
                rows = num_reqs * owned
                got_bt = torch.zeros(
                    rows, TOPK * PAGES_PER_BLOCK, dtype=torch.int32, device=device
                )
                got_ctx = torch.zeros(rows, dtype=torch.int32, device=device)
                pa_sparse_block_topk_cp(
                    shard,
                    cand_ptrs,
                    cp_gen,
                    got_idx,
                    seq,
                    bt,
                    got_bt,
                    got_ctx,
                    max_seq_len=max_seq_len,
                    block_size=BLOCK_SIZE,
                    cand_rows=cand_rows,
                    query_len=1,
                    num_kv_heads=owned,
                    pages_per_block=PAGES_PER_BLOCK,
                    rank=rank,
                    world=world,
                )
                torch.cuda.synchronize()

                tag = (
                    f"rank {rank}/{world} H{num_idx_heads} ctx{ctx} "
                    f"call {call} batch {num_reqs}"
                )
                exact = _compare(got_idx, ref_idx, full_score[lo:hi], tag)

                # The page table only has to agree where the selection did: a
                # tie substitute is a different block, so its pages
                # legitimately differ.
                keep = exact.permute(1, 0).reshape(rows)
                if not torch.equal(got_ctx[keep], ref_ctx[keep]):
                    raise AssertionError(f"{tag}: sparse_ctx differs from single-GPU")
                if not torch.equal(got_bt[keep], ref_bt[keep]):
                    raise AssertionError(f"{tag}: sparse_bt differs from single-GPU")

                ties = int((~exact).sum())
                if ties and rank == 0:
                    aiter.logger.info(
                        "%s: %d row(s) matched by score, not id", tag, ties
                    )
                call += 1

            # The longest context's index cache is a couple of GB per rank, so
            # it goes back before the next one is built rather than after.
            del q_all, k, bt_all, seq_all
            torch.cuda.empty_cache()

        # No rank may free a buffer a peer could still be reading.
        dist.barrier()
    finally:
        for mapped in opened:
            UncachedIpcHeap.close_mem_handle(mapped)
        free_meta_buffer(own_ptr)
        dist.destroy_process_group()


def run_case(world: int, num_idx_heads: int, ctxs, batches):
    """Spawn ``world`` ranks and fail, rather than hang, if one does not finish."""
    proc_ctx = mp.spawn(
        _worker,
        args=(world, get_open_port(), num_idx_heads, list(ctxs), list(batches)),
        nprocs=world,
        join=False,
    )
    deadline = time.monotonic() + _RANK_TIMEOUT
    while not proc_ctx.join(timeout=1):
        if time.monotonic() > deadline:
            for proc in proc_ctx.processes:
                if proc.is_alive():
                    proc.terminate()
            raise TimeoutError(
                f"world={world} H={num_idx_heads} did not finish within "
                f"{_RANK_TIMEOUT}s. The merge waits on a generation tag rather "
                f"than a barrier, so ranks that disagree on the call sequence "
                f"hang here instead of returning a wrong answer."
            )


def _skip_reason(world: int) -> str | None:
    gfx = aiter.get_gfx()
    if gfx != "gfx950":
        return f"requires gfx950, got {gfx}"
    if torch.cuda.device_count() < world:
        return f"requires {world} GPUs, got {torch.cuda.device_count()}"
    return None


# world * TOPK must fit the one wave of candidates the fused merge holds, which
# with SPARSE_TOPK=16 stops at 4 ranks. num_idx_heads is the model's whole count
# and the score kernel is built for 1, 2 and 4 of them.
CASES = [(2, 2), (2, 4), (4, 4)]
BATCHES = [16, 32, 64, 128, 64, 16, 128]
CTXS = [16384, 32768, 65536, 131072]


try:
    import pytest

    @pytest.mark.parametrize("world,num_idx_heads", CASES)
    def test_cp_selection_matches_single_gpu(world, num_idx_heads):
        reason = _skip_reason(world)
        if reason:
            pytest.skip(reason)
        run_case(world, num_idx_heads, CTXS, BATCHES)

except ImportError:
    pass


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(
        description="Multi-rank correctness for the CP block selector"
    )
    parser.add_argument("-w", "--world", type=int, nargs="*", default=None)
    parser.add_argument("-H", "--num_idx_heads", type=int, nargs="*", default=None)
    parser.add_argument("-c", "--ctx", type=int, nargs="*", default=CTXS)
    parser.add_argument("-b", "--batches", type=int, nargs="*", default=BATCHES)
    args = parser.parse_args()

    cases = [
        (w, h)
        for (w, h) in CASES
        if (args.world is None or w in args.world)
        and (args.num_idx_heads is None or h in args.num_idx_heads)
    ]
    if not cases:
        sys.exit("no built (world, num_idx_heads) case matches the given filters")

    for world, num_idx_heads in cases:
        reason = _skip_reason(world)
        if reason:
            print(f"skipping world={world} H={num_idx_heads}: {reason}")
            continue
        run_case(world, num_idx_heads, args.ctx, args.batches)
        aiter.logger.info(
            "cp selection matches single-GPU top-k: world=%d, num_idx_heads=%d, "
            "ctx=%s, batches=%s, %d calls",
            world,
            num_idx_heads,
            args.ctx,
            args.batches,
            len(args.ctx) * len(args.batches),
        )
