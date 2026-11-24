# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and timing of aiter.ops.m3_fused_index_topk (gfx950).

Every chunk's 16 packed keys are graded bit for bit. The "int" and "sub" modes
use a torch reference: on integers in [-16, 16] and on e4m3 subnormals the MFMA's
dot is exact. Off that grid the MFMA rounds inside the dot, so the "randn" mode
builds the reference keys from aiter's FlyDSL prefill scorer, which with aiter's
top-16 selector is also the timing baseline; it is skipped without FlyDSL.
"""

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.m3_fused_index_topk import fused_index_topk_partial, is_supported
from aiter.test_common import benchmark, run_perftest

try:
    from aiter.ops.flydsl import minimax_m3_index_score_prefill_flydsl as flydsl_score
    from aiter.ops.flydsl.topk.topk_per_row_small_k import topk_per_row_small_k
except ImportError:
    flydsl_score = None

PAGE = 128
HEADS = 4
HEAD_DIM = 128
TOPK = 16
TQ = 64  # query tokens per CTA
SM_SCALE = HEAD_DIM**-0.5
# fp32(sm_scale) * fp32(log2 e), rounded once in fp32 (what the scorers apply)
SCALE = (torch.tensor(SM_SCALE) * torch.tensor(1.4426950409)).item()


def cdiv(a, b):
    return -(-a // b)


def chunking(nchunks, max_query_len, batch, max_block):
    """(nchunks, chunk_blocks); nchunks 0 picks about 512 CTAs of 8-32 chunks of at
    least 64 blocks, as a caller would."""
    if nchunks == 0:
        want = max(1, cdiv(512, cdiv(max_query_len, TQ) * batch))
        nchunks = max(1, min(32, max(8, 1 << (want - 1).bit_length()), max_block // 64))
    chunk_blocks = cdiv(max_block, nchunks)
    return cdiv(max_block, chunk_blocks), chunk_blocks


def grid_values(shape, mode, g, dtype):
    """Values the MFMA's dot is exact on: integers in [-16, 16] ("int") or e4m3
    subnormals k * 2^-9, |k| <= 7 ("sub")."""
    hi = 16 if mode == "int" else 7
    v = torch.randint(-hi, hi + 1, shape, generator=g, device="cuda").float()
    return (v if mode == "int" else v * 2.0**-9).to(dtype)


def make_case(reqs, mode, seed):
    """One prefill step, reqs = [(cached prefix, new tokens)]: pages at random
    places in a pool twice the step's size; a poison page pads the block table,
    so a stray read wins the selection."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    lens = [p + t for p, t in reqs]
    nblk = [cdiv(n, PAGE) for n in lens]
    pages = 2 * sum(nblk)
    total_q = sum(t for _, t in reqs)
    cshape, qshape = (pages + 1, PAGE, HEAD_DIM), (total_q, HEADS, HEAD_DIM)
    if mode == "randn":
        cache = torch.randn(cshape, generator=g, device="cuda").to(dtypes.fp8)
        idx_q = torch.randn(qshape, generator=g, device="cuda").to(dtypes.bf16)
    else:
        cache = grid_values(cshape, mode, g, dtypes.fp8)
        idx_q = grid_values(qshape, mode, g, dtypes.bf16)
    poison = torch.zeros(PAGE, HEAD_DIM, device="cuda")
    poison[torch.arange(PAGE), torch.arange(PAGE)] = 448.0
    cache[pages] = poison.to(dtypes.fp8)
    perm = torch.randperm(pages, generator=g, device="cuda").to(torch.int32)
    bt = torch.full((len(reqs), max(nblk) + 8), pages, dtype=torch.int32, device="cuda")
    for b, (n, off) in enumerate(zip(nblk, itertools.accumulate([0] + nblk))):
        bt[b, :n] = perm[off : off + n]
    cu = [0] + list(itertools.accumulate(t for _, t in reqs))
    return {
        "idx_q": idx_q,
        "cache": cache,
        "bt": bt,
        "cu": torch.tensor(cu, dtype=torch.int32, device="cuda"),
        "prefix": torch.tensor([p for p, _ in reqs], dtype=torch.int32, device="cuda"),
        "seq_lens": torch.tensor(lens, dtype=torch.int32, device="cuda"),
        "pos": torch.cat([p + torch.arange(t, device="cuda") for p, t in reqs]),
        "max_query_len": max(t for _, t in reqs),
        "max_seq_len": max(lens),
        "max_block": max(nblk),
        "cu_list": cu,
        "nblk": nblk,
    }


def torch_scores(c):
    """[4, total_q, max_block] fp32: max over each block's keys of the e4m3 dot (fp64),
    times SCALE in fp32. Blocks past a token's own are garbage here; the keys mask
    them."""
    score = torch.full(
        (HEADS, c["idx_q"].shape[0], c["max_block"]), float("-inf"), device="cuda"
    )
    for b, n in enumerate(c["nblk"]):
        s, e = c["cu_list"][b], c["cu_list"][b + 1]
        k = c["cache"][c["bt"][b, :n].long()].to(torch.float64).view(n * PAGE, HEAD_DIM)
        q = c["idx_q"][s:e].to(dtypes.fp8).to(torch.float64).view(-1, HEAD_DIM)
        for k0 in range(0, n, 32):
            k1 = min(n, k0 + 32)
            dots = q @ k[k0 * PAGE : k1 * PAGE].T
            bmax = dots.view(-1, k1 - k0, PAGE).amax(-1).float()
            score[:, s:e, k0:k1] = bmax.view(e - s, HEADS, k1 - k0).transpose(0, 1)
    return score * SCALE


def chunk_keys(score, pos, nchunks, chunk_blocks):
    """[4, total_q, nchunks, 16]: each chunk's 16 best packed keys, descending, padded
    with 15, 14, ... like the kernel. The token's own block is pinned at 1e29."""
    heads, total_q, width = score.shape
    blk = torch.arange(width, device=score.device)
    local = (pos // PAGE)[None, :, None]
    s = torch.where(blk == local, torch.tensor(1e29, device=score.device), score)
    bits = s.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    bits = torch.where(bits == 0x80000000, 0, bits)
    ordered = torch.where(bits >> 31 != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    ordered = torch.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = torch.where(blk <= local, (1 << 48) | (ordered << 16) | (blk + 1), 0)
    key = F.pad(key, (0, nchunks * chunk_blocks - width))
    key = key.view(heads, total_q, nchunks, chunk_blocks)
    seeds = torch.arange(15, -1, -1, device=score.device)
    seeds = seeds.expand(heads, total_q, nchunks, TOPK)
    return torch.cat([key, seeds], -1).topk(TOPK, dim=-1).values


def run_fused(c, partial, nchunks, chunk_blocks):
    fused_index_topk_partial(
        c["idx_q"],
        c["cache"],
        partial,
        c["bt"],
        c["cu"],
        c["prefix"],
        c["max_query_len"],
        nchunks,
        chunk_blocks,
        SCALE,
    )
    return partial


def flydsl_plane(c):
    """aiter's FlyDSL scores, row stride padded to 4 blocks (its merged store)."""
    total_q, width = c["idx_q"].shape[0], c["max_block"]
    out = torch.full(
        (HEADS, total_q, cdiv(width, 4) * 4), float("-inf"), device="cuda"
    )[:, :, :width]
    return flydsl_score(
        c["idx_q"],
        c["cache"],
        c["bt"],
        c["cu"],
        c["seq_lens"],
        c["prefix"],
        c["max_query_len"],
        c["max_seq_len"],
        SM_SCALE,
        out=out,
    )


def mismatch_report(got, ref):
    bad = (got != ref).any(-1).nonzero()
    h, t, ch = bad[0].tolist()
    return (
        f"{bad.shape[0]} of {ref.shape[0] * ref.shape[1] * ref.shape[2]} chunk lists "
        f"differ, first (head {h}, token {t}, chunk {ch}):\n"
        f"  got {[hex(x) for x in got[h, t, ch].tolist()]}\n"
        f"  ref {[hex(x) for x in ref[h, t, ch].tolist()]}"
    )


@benchmark()
def test_fused_index_topk(reqs, mode, seed, nchunks_list, perf):
    c = make_case(reqs, mode, seed)
    total_q, batch = c["idx_q"].shape[0], len(reqs)
    if mode == "randn":
        score = flydsl_plane(c)
    else:
        score = torch_scores(c)
    ret = {}
    for nc in nchunks_list:
        nchunks, chunk_blocks = chunking(nc, c["max_query_len"], batch, c["max_block"])
        ref = chunk_keys(score, c["pos"], nchunks, chunk_blocks)
        # pre-filled with -1, so a key the kernel does not write fails the comparison
        partial = torch.full(
            (HEADS, total_q, nchunks * TOPK), -1, dtype=torch.int64, device="cuda"
        )
        got = run_fused(c, partial, nchunks, chunk_blocks).view(ref.shape)
        assert torch.equal(
            got, ref
        ), f"{reqs} {mode} seed {seed} nchunks {nchunks}: " + mismatch_report(got, ref)
    ret["chunks checked"] = ",".join(str(nc) for nc in nchunks_list)
    del score
    if not perf:
        return ret

    nchunks, chunk_blocks = chunking(0, c["max_query_len"], batch, c["max_block"])
    partial = torch.empty(
        (HEADS, total_q, nchunks * TOPK), dtype=torch.int64, device="cuda"
    )

    def fused(idx_q, cache, partial):
        fused_index_topk_partial(
            idx_q,
            cache,
            partial,
            c["bt"],
            c["cu"],
            c["prefix"],
            c["max_query_len"],
            nchunks,
            chunk_blocks,
            SCALE,
        )

    _, ret["fused us"] = run_perftest(
        fused, c["idx_q"], c["cache"], partial, num_rotate_args=8
    )
    ret["nchunks"] = nchunks
    if flydsl_score is not None:
        out = flydsl_plane(c)
        idx = torch.empty((HEADS * total_q, TOPK), dtype=torch.int32, device="cuda")
        row_lens = (c["pos"] // PAGE + 1).to(torch.int32).repeat(HEADS)

        def flydsl(idx_q, cache, out):
            flydsl_score(
                idx_q,
                cache,
                c["bt"],
                c["cu"],
                c["seq_lens"],
                c["prefix"],
                c["max_query_len"],
                c["max_seq_len"],
                SM_SCALE,
                out=out,
            )
            topk_per_row_small_k(
                out.view(HEADS * total_q, -1), row_lens, idx, TOPK, 0, 1
            )

        _, ret["flydsl score+select us"] = run_perftest(
            flydsl, c["idx_q"], c["cache"], out, num_rotate_args=8
        )
        ret["fused speedup"] = ret["flydsl score+select us"] / ret["fused us"]
    return ret


def parse_step(s):
    """'131072:8192' or a ragged step '131149:4096,0:1500' -> [(prefix, new tokens)]"""
    return [tuple(int(x) for x in r.split(":")) for r in s.split(",")]


def main():
    if not is_supported():
        aiter.logger.warning("m3_fused_index_topk is gfx950-only; skipping")
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MiniMax-M3 fused FP8 index scorer + per-chunk top-16",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=parse_step,
        nargs="*",
        default=[
            parse_step(s)
            for s in (
                "65536:4096",
                "131072:8192",
                "262144:4096",
                "262144:16384",
                # ragged: short requests with fewer than 16 blocks, odd lengths
                "131149:4096,0:1500,5000:600,40000:1996",
            )
        ],
        help="prefill steps, 'prefix:new_tokens[,prefix:new_tokens...]'",
    )
    parser.add_argument(
        "-m",
        "--modes",
        nargs="*",
        default=["int", "sub", "randn"],
        choices=["int", "sub", "randn"],
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument(
        "--nchunks",
        type=int,
        nargs="*",
        default=[0, 1, 7],
        help="chunk counts to grade, 0 = the default choice",
    )
    parser.add_argument("--no-perf", action="store_true")
    args = parser.parse_args()

    df = []
    for reqs, mode, seed in itertools.product(args.steps, args.modes, args.seeds):
        if mode == "randn" and flydsl_score is None:
            continue
        perf = (
            not args.no_perf
            and len(reqs) == 1
            and seed == args.seeds[0]
            and mode == args.modes[0]
        )
        df.append(test_fused_index_topk(reqs, mode, seed, args.nchunks, perf))
        torch.cuda.empty_cache()
    if flydsl_score is None:
        aiter.logger.warning("FlyDSL is not installed: randn mode and baseline skipped")
    df = pd.DataFrame(df)
    aiter.logger.info(
        "m3_fused_index_topk summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
