# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the GLM-5.3-Flash indexer kpool cache writers (HIP).

    python bench_kpool_compress.py                         # default sweep
    python bench_kpool_compress.py --insert-tokens 32768 1048576 --decode-batch 256
    python bench_kpool_compress.py -o results.csv

Timings are triton.testing.do_bench medians (L2 flushed). Bandwidth counts
the compulsory bytes only: 512 B read + 33 B written per prefill token
(APE and slots hit L2), and per decode request 512 B K/gate + 512 B stash
(+ 1536 B ring + 132 B pooled key when a pool closes).
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import torch
import triton

import aiter
from aiter.ops.kpool_compress import kpool_pick_lanes

DEVICE = "cuda"
HEAD_DIM = 128
INSERT_TOKENS = [2048, 8192, 32768, 131072, 1048576]
DECODE_BATCH = [1, 64, 256, 1024, 4096]
DECODE_NEXT_N = [1, 4]


def make_prefill(n_tokens, pool, page, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    n_pools = n_tokens // pool
    num_blocks = math.ceil(n_pools / page) + 2
    slot = torch.full((n_tokens,), -1, dtype=torch.int64, device=DEVICE)
    pools = torch.randperm(num_blocks * page, generator=g, device=DEVICE)[:n_pools]
    slot[torch.arange(n_pools, device=DEVICE) * pool + pool - 1] = pools
    k = torch.randn(n_tokens, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    gate = (torch.randn(n_tokens, HEAD_DIM, generator=g, device=DEVICE) * 2).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    kv = torch.zeros(num_blocks, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return k, gate, ape, slot, kv


def make_decode(B, next_n, pool, page, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # production tail ring (vLLM GLM-5.3 MTP): pool * pow2(ceil((pool + next_n - 1) / pool)), == pool for next_n = 1
    ring = pool * (1 << (-(-(pool + next_n - 1) // pool) - 1).bit_length())
    tail = torch.randn(B + 2, 2, ring, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    ctx = torch.randint(pool, 100000, (B,), generator=g, device=DEVICE)
    pos = (ctx[:, None] + torch.arange(next_n, device=DEVICE)[None]).to(torch.int32)
    blocks = torch.arange(B, dtype=torch.int32, device=DEVICE)[:, None]
    tail_slot = (blocks * ring + pos % ring).to(torch.int32)
    n_pages = math.ceil(B * next_n / page) + 2
    locs = torch.randperm(n_pages * page, generator=g, device=DEVICE)[: B * next_n]
    locs = locs.view(B, next_n).to(torch.int32)
    slot_map = torch.where(pos % pool == pool - 1, locs, torch.full_like(locs, -1))
    key = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    score = torch.randn(B, next_n, HEAD_DIM, generator=g, device=DEVICE).bfloat16()
    ape = torch.randn(pool, HEAD_DIM, generator=g, device=DEVICE)
    kv = torch.zeros(n_pages, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE)
    return kv, tail, tail_slot, key, score, ape, slot_map, pos


def bench(fn, warmup=5, rep=50):
    ms, p20, p80 = triton.testing.do_bench(
        fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.2, 0.8]
    )
    return ms * 1e3, p20 * 1e3, p80 * 1e3


def run_insert(args):
    rows = []
    print(f"\ninsert (pool {args.pool}, page {args.page})")
    print(f"{'tokens':>9} {'lanes':>5} {'us':>9} {'p20':>9} {'p80':>9} {'TB/s':>6}")
    for n in args.insert_tokens:
        k, gate, ape, slot, kv = make_prefill(n, args.pool, args.page)
        lanes = args.lanes or kpool_pick_lanes(n, args.pool)
        us, p20, p80 = bench(
            functools.partial(
                aiter.kpool_compress_insert,
                kv,
                k,
                gate,
                ape,
                slot,
                args.pool,
                True,
                lanes,
            )
        )
        tbs = n * (2 * HEAD_DIM * 2 + 33) / us / 1e6
        print(f"{n:>9} {lanes:>5} {us:9.2f} {p20:9.2f} {p80:9.2f} {tbs:6.2f}")
        rows.append(("insert", n, 0, 0, lanes, us, p20, p80, tbs))
        del k, gate, ape, slot, kv
        torch.cuda.empty_cache()
    return rows


def run_decode(args):
    rows = []
    print(f"\ndecode (pool {args.pool}, page {args.page})")
    print(f"{'B':>6} {'next_n':>6} {'us':>9} {'p20':>9} {'p80':>9} {'TB/s':>6}")
    for B in args.decode_batch:
        for next_n in args.decode_next_n:
            kv, tail, tail_slot, key, score, ape, slot_map, pos = make_decode(
                B, next_n, args.pool, args.page
            )
            us, p20, p80 = bench(
                functools.partial(
                    aiter.kpool_decode_update,
                    kv,
                    tail,
                    tail_slot,
                    key,
                    score,
                    ape,
                    slot_map,
                    pos,
                    args.pool,
                )
            )
            closes = int((slot_map >= 0).sum())
            nbytes = B * next_n * 1024 + closes * (1536 + 132)
            tbs = nbytes / us / 1e6
            print(f"{B:>6} {next_n:>6} {us:9.2f} {p20:9.2f} {p80:9.2f} {tbs:6.2f}")
            rows.append(("decode", 0, B, next_n, 64, us, p20, p80, tbs))
    return rows


def run_mixed(args):
    rows = []
    print(
        f"\nmixed (pool {args.pool}, page {args.page}): prefill tokens + decode requests in one launch"
    )
    print(f"{'tokens':>9} {'B':>6} {'next_n':>6} {'us':>9} {'p20':>9} {'p80':>9}")
    for n in args.mixed_tokens:
        for B in args.mixed_batch:
            next_n = args.decode_next_n[0]
            kv_d, tail, tail_slot, key, score, ape, slot_map, pos = make_decode(
                B, next_n, args.pool, args.page
            )
            k, gate, _, slot, kv_p = make_prefill(n, args.pool, args.page, seed=1)
            slot = torch.where(slot >= 0, slot + kv_d.shape[0] * args.page, slot)
            kv = torch.cat([kv_d, kv_p], 0)
            # the prefill rows seed tail block B; every index tensor of one launch shares the prefill slot dtype
            ring = tail.shape[2]
            tslot = (
                torch.full((n,), B, dtype=slot.dtype, device=DEVICE) * ring
                + torch.arange(n, device=DEVICE, dtype=slot.dtype) % ring
            )
            tail_slot, slot_map, pos = (
                t.to(slot.dtype) for t in (tail_slot, slot_map, pos)
            )
            tail = torch.cat(
                [
                    tail,
                    torch.zeros(
                        1, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
                    ),
                ],
                0,
            )
            us, p20, p80 = bench(
                functools.partial(
                    aiter.kpool_mixed_write,
                    kv,
                    ape,
                    args.pool,
                    True,
                    0,
                    k,
                    gate,
                    slot,
                    tslot,
                    tail,
                    tail_slot,
                    key,
                    score,
                    slot_map,
                    pos,
                )
            )
            print(f"{n:>9} {B:>6} {next_n:>6} {us:9.2f} {p20:9.2f} {p80:9.2f}")
            rows.append(
                (
                    "mixed",
                    n,
                    B,
                    next_n,
                    kpool_pick_lanes(n, args.pool),
                    us,
                    p20,
                    p80,
                    0.0,
                )
            )
            del kv, tail, k, gate
            torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser(description="Benchmark the kpool cache writers.")
    p.add_argument("--only", choices=["insert", "decode", "mixed"], default=None)
    p.add_argument("--pool", type=int, default=4)
    p.add_argument("--page", type=int, default=64)
    p.add_argument(
        "--lanes",
        type=int,
        default=0,
        help="force lanes per pool for insert (16/32/64)",
    )
    p.add_argument("--insert-tokens", type=int, nargs="*", default=INSERT_TOKENS)
    p.add_argument("--decode-batch", type=int, nargs="*", default=DECODE_BATCH)
    p.add_argument("--decode-next-n", type=int, nargs="*", default=DECODE_NEXT_N)
    p.add_argument("--mixed-tokens", type=int, nargs="*", default=[2048, 32768])
    p.add_argument("--mixed-batch", type=int, nargs="*", default=[64, 256])
    p.add_argument("-o", type=str, default=None, help="save results to CSV")
    args = p.parse_args()

    rows = []
    if args.only in (None, "insert"):
        rows += run_insert(args)
    if args.only in (None, "decode"):
        rows += run_decode(args)
    if args.only in (None, "mixed"):
        rows += run_mixed(args)
    if args.o:
        import csv

        with Path(args.o).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "op",
                    "tokens",
                    "batch",
                    "next_n",
                    "lanes",
                    "us",
                    "p20_us",
                    "p80_us",
                    "TB_s",
                ]
            )
            w.writerows(rows)
        print(f"\nsaved {len(rows)} rows to {args.o}")


if __name__ == "__main__":
    main()
