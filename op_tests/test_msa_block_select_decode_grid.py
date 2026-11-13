# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""nvfp4 decode score grids against the bound-sized fp8 grid.

The baseline is the fp8 split (_score_split) with no request XOR and no chunk
widening. Scores, top-k, sparse page table and token counts must match bit for
bit across heads, query lengths, uniform and ragged batches, block edges,
max_seq_len bounds, grid sizings, fill_shift values, request XOR masks and
query tiles per wave.
"""

import argparse
import math
import random
import sys

import torch
from msa_nvfp4_ref import ref_nvfp4_pages

import aiter
from aiter.ops import msa_block_select as mbs
from aiter.ops.msa_block_select import (
    nvfp4_page_bytes,
    pa_sparse_block_score_decode,
    pa_sparse_block_topk,
)

BLOCK_SIZE = 128
HEAD_DIM = 128
TOPK = 16
WAVE = 64
PAGES_PER_BLOCK = 8
FP8 = torch.float8_e4m3fn
DEV = "cuda"

# The bound-sized fp8 grid.
BOUND = {
    "_score_split_decode_nvfp4": mbs._score_split,
    "_decode_req_xor_mask": lambda num_reqs: 0,
    "_decode_fill_shift": lambda num_reqs: mbs._DECODE_FILL_OFF,
}


class _patch:
    """Replace msa_block_select attributes for the duration of a with-block."""

    def __init__(self, **kv):
        self.kv = kv

    def __enter__(self):
        self.old = {k: getattr(mbs, k) for k in self.kv}
        for k, v in self.kv.items():
            setattr(mbs, k, v)

    def __exit__(self, *exc):
        for k, v in self.old.items():
            setattr(mbs, k, v)


def _setup(seq_lens, query_len, num_idx_heads, seed):
    """q, a compact [num_pages, page bytes] nvfp4 cache and a shuffled block
    table."""
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    nblk = [-(-s // BLOCK_SIZE) for s in seq_lens]
    num_pages = sum(nblk) + 2
    total_q = len(seq_lens) * query_len
    q = torch.randn(total_q, num_idx_heads, HEAD_DIM, generator=g, device=DEV)
    q = (q / 4).to(FP8)
    page_bytes = nvfp4_page_bytes(BLOCK_SIZE, HEAD_DIM)
    cache = torch.zeros(num_pages, page_bytes, device=DEV, dtype=torch.uint8)
    # A slice of pages at a time to bound the staging buffers.
    step = 4096
    for i in range(0, num_pages, step):
        n = min(step, num_pages - i)
        keys = torch.randn(n * BLOCK_SIZE, HEAD_DIM, generator=g, device=DEV)
        keys = keys.to(torch.bfloat16)
        keys[::37] *= 64
        cache[i : i + n] = ref_nvfp4_pages(keys, BLOCK_SIZE)
    perm = torch.randperm(num_pages, generator=g, device=DEV).to(torch.int32)
    block_table = torch.zeros(
        len(seq_lens), max(nblk) + 3, device=DEV, dtype=torch.int32
    )
    off = 0
    for r, n in enumerate(nblk):
        block_table[r, :n] = perm[off : off + n]
        off += n
    seq = torch.tensor(seq_lens, device=DEV, dtype=torch.int32)
    return q, cache, block_table, seq


def _run(q, cache, bt, seq, query_len, max_seq_len, fill, overrides):
    """Decode pass and top-k with msa_block_select attributes overridden."""
    heads, total_q = q.size(1), q.size(0)
    slots = 1 << (-(-math.ceil(max_seq_len / BLOCK_SIZE) // WAVE) - 1).bit_length()
    score = torch.full((heads, total_q, slots * WAVE), fill, device=DEV)
    topk_idx = torch.full((heads, total_q, TOPK), -7, device=DEV, dtype=torch.int32)
    rows = total_q * heads
    sbt = torch.full((rows, TOPK * PAGES_PER_BLOCK), -7, device=DEV, dtype=torch.int32)
    sctx = torch.full((rows,), -7, device=DEV, dtype=torch.int32)
    with _patch(**overrides):
        pa_sparse_block_score_decode(
            q,
            cache,
            score,
            bt,
            seq,
            init_blocks=0,
            local_blocks=1,
            query_len=query_len,
            max_seq_len=max_seq_len,
        )
        pa_sparse_block_topk(
            score,
            topk_idx,
            bt,
            seq,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
            query_len=query_len,
            sparse_bt=sbt,
            sparse_ctx=sctx,
            num_kv_heads=heads,
            pages_per_block=PAGES_PER_BLOCK,
        )
    torch.cuda.synchronize()
    # Bit patterns, so NaN slots compare equal.
    return score.view(torch.int32), topk_idx, sbt, sctx


def _bounds(seq_lens):
    """The tight bound, then serving-sized ones."""
    return [max(seq_lens), 131072, 262144, 1048576]


def _nan_to_neginf(out):
    score = out[0].view(torch.float32)
    score = score.masked_fill(score.isnan(), -float("inf"))
    return (score.view(torch.int32), *out[1:])


def _one_tile(*a):
    return 1


def _runs(num_reqs, rows_per_req, caps):
    """(label, overrides) of every grid to check against BOUND."""
    runs = [
        (
            f"cap:{c}",
            {
                "_SCORE_DECODE_MAX_WORKGROUPS": c[0],
                "_SCORE_DECODE_MAX_WG_BLOCKS": c[1],
            },
        )
        for c in caps
    ]
    runs += [
        (f"fill_shift:{w}", {"_decode_fill_shift": lambda n, w=w: w})
        for w in args.fill_shift
    ]
    runs += [
        (f"req_xor_mask:{m}", {"_decode_req_xor_mask": lambda n, m=m: m})
        for m in (1, 3, 7)
        if num_reqs % (m + 1) == 0
    ]
    # Rows past one MFMA tile: one query tile per wave against the default.
    if rows_per_req > 16:
        runs += [
            ("bound qtiles:1", {**BOUND, "_decode_q_tiles": _one_tile}),
            ("qtiles:1", {"_decode_q_tiles": _one_tile}),
        ]
    return runs


def check_case(name, seq_lens, query_len, num_idx_heads, caps, seed):
    q, cache, bt, seq = _setup(seq_lens, query_len, num_idx_heads, seed)
    ok = True
    runs = _runs(len(seq_lens), num_idx_heads * query_len, caps)
    parts = ("score", "topk_idx", "sparse_bt", "sparse_ctx")
    for msl in _bounds(seq_lens):
        # NaN fill exposes stray or missing writes; top-k is checked on -inf.
        for fill in (float("nan"), -float("inf")):
            base = _run(q, cache, bt, seq, query_len, msl, fill, BOUND)
            for label, overrides in runs:
                got = _run(q, cache, bt, seq, query_len, msl, fill, overrides)
                want = base
                if "_decode_q_tiles" in overrides and math.isnan(fill):
                    # One tile per wave leaves NaN in a few slots that two
                    # tiles per wave set to -inf.
                    want, got = _nan_to_neginf(base), _nan_to_neginf(got)
                bad = [p for p, a, b in zip(parts, want, got) if not torch.equal(a, b)]
                if math.isnan(fill):
                    bad = [p for p in bad if p == "score"]
                if bad:
                    ok = False
                    aiter.logger.info(
                        f"{name} H:{num_idx_heads} qlen:{query_len} msl:{msl} "
                        f"{label} fill:{fill}: {bad} differ -> FAIL"
                    )
    aiter.logger.info(
        f"{name} H:{num_idx_heads} qlen:{query_len} B:{len(seq_lens)} "
        f"{len(runs)} grids vs bound: {'bit-exact' if ok else 'FAIL'}"
    )
    return ok


def cases(query_len):
    """(name, seq_lens) pairs; every row holds at least query_len tokens."""
    rnd = random.Random(1234 + query_len)
    lo = query_len
    out = [
        ("uniform B1 1k", [1024]),
        ("uniform B4 32k", [32768] * 4),
        ("uniform B64 8k", [8192] * 64),
        ("uniform B128 4k", [4096] * 128),
        ("uniform B256 1k", [1024] * 256),
        ("uniform B32 32k", [32768] * 32),
        ("sub-block", [max(lo, n) for n in (1, 5, 64, 127)]),
        (
            "block edges",
            [
                n
                for b in (1, 2, 3, 8, 64, 255)
                for n in (128 * b - 1, 128 * b, 128 * b + 1)
            ],
        ),
        ("short rows", [max(lo, 64 + 37 * i) for i in range(16)]),
        (
            "ragged",
            [max(lo, rnd.randint(1, 131072)) for _ in range(24)] + [131072, lo],
        ),
        (
            "90/10",
            [rnd.randint(1000, 4000) for _ in range(36)]
            + [rnd.randint(90000, 131072) for _ in range(4)],
        ),
        ("one long", [max(lo, 300)] * 63 + [131072]),
        ("odd B7", [max(lo, 300)] * 6 + [131072]),
        ("B12 one long", [max(lo, 300)] * 11 + [131072]),
        ("B24 two long", [max(lo, 500)] * 22 + [131072, 98765]),
    ]
    return out


parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("-H", "--num_idx_heads", type=int, nargs="*", default=[1, 2, 4])
parser.add_argument("-q", "--query_len", type=int, nargs="*", default=[1, 2, 4, 8])
parser.add_argument(
    "--caps",
    nargs="*",
    default=None,
    help="grid sizings as WORKGROUPS[:WG_BLOCKS] (default: shipped, 16384, 1024)",
)
parser.add_argument(
    "-k", "--filter", default="", help="only case names containing this"
)
parser.add_argument(
    "--fill-shift",
    type=int,
    nargs="*",
    default=[0, 1],
    help="fill_shift values to force besides the shipped one",
)
args = parser.parse_args()

current_gfx = aiter.get_gfx()
if current_gfx != "gfx950":
    print(f"Skipping {__file__}: requires gfx950, got {current_gfx}")
    sys.exit(0)

_SHIPPED = (mbs._SCORE_DECODE_MAX_WORKGROUPS, mbs._SCORE_DECODE_MAX_WG_BLOCKS)


def _cap(text):
    wg, _, blocks = text.partition(":")
    return int(wg), int(blocks) if blocks else 1 << 30


caps = (
    [_cap(c) for c in args.caps]
    if args.caps
    else [_SHIPPED, (16384, 1 << 30), (1024, 1 << 30)]
)
results = []
seed = 0
for h in args.num_idx_heads:
    for ql in args.query_len:
        for name, seq_lens in cases(ql):
            if args.filter not in name:
                continue
            seed += 1
            results.append(check_case(name, seq_lens, ql, h, caps, seed))
            torch.cuda.empty_cache()
print("ALL PASS" if results and all(results) else "FAILURES")
sys.exit(0 if results and all(results) else 1)
