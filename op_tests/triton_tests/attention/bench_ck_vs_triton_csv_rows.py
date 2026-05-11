#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Verify a small set of CK-wins-over-Triton rows from a benchmark CSV.

Reads `pawel-2d-3d.csv` (or any compatible CSV), picks the rows by `idx`, and
for each row reconstructs the same shape, runs CK unified attention vs the
Triton unified-attention wrapper on the SAME tensors, and reports:

  - CK correctness vs. Triton (max abs diff, allclose at atol=1e-2 rtol=1e-2)
  - CK ms / Triton ms / speedup
  - the CSV's reported numbers, for sanity

Usage:
    python bench_ck_vs_triton_csv_rows.py
    python bench_ck_vs_triton_csv_rows.py --idx 843,260,979 --warmup 20 --iters 100
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention import unified_attention as ua_mod


# Default 5 rows: a diverse sample of CK wins (different phases/scales).
DEFAULT_IDS = [843, 260, 220, 979, 268]


def parse_window(s):
    a, b = s.split(",")
    return (int(a), int(b))


def make_tensors(row, num_blocks_cap, device="cuda", dtype=torch.bfloat16):
    """Build a varlen+paged tensor batch matching one CSV row.

    The CSV doesn't give a num_blocks pool size, so we pick one that is
    plenty for correctness (block_table is allowed to alias pages — the
    same aliased tables go to both kernels, so the outputs still must
    match)."""
    b = int(row["num_seqs"])
    hq = int(row["num_q_heads"])
    hk = int(row["num_kv_heads"])
    d = int(row["head_size"])
    blk = int(row["block_size"])
    sq = int(row["max_seqlen_q"])
    sk = int(row["max_seqlen_k"])
    total_q = int(row["total_q_tokens"])
    scale = 1.0 / math.sqrt(d)

    max_blks_per_seq = (sk + blk - 1) // blk
    # Pool large enough to cover one sequence with no aliasing, capped to
    # avoid 200GB+ allocations on the longest-context rows.
    nb = min(num_blocks_cap, max(1024, 2 * max_blks_per_seq))

    q = torch.randn(total_q, hq, d, dtype=dtype, device=device)
    k = torch.randn(nb, blk, hk, d, dtype=dtype, device=device)
    v = torch.randn_like(k)

    # Build cu_seqlens_q with non-uniform query lens summing to total_q.
    if sq * b == total_q:
        cu = torch.arange(0, b + 1, dtype=torch.int32, device=device) * sq
    else:
        # Distribute total_q across b sequences with cap = sq.
        base = total_q // b
        rem = total_q - base * b
        lens = [min(sq, base + (1 if i < rem else 0)) for i in range(b)]
        # If our distribution undershoots, dump the remainder in seq 0.
        miss = total_q - sum(lens)
        if miss > 0 and lens[0] + miss <= sq:
            lens[0] += miss
        elif miss > 0:
            for i in range(b):
                if miss == 0:
                    break
                room = sq - lens[i]
                add = min(room, miss)
                lens[i] += add
                miss -= add
        cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()),
                          dtype=torch.int32, device=device)

    seq_lens_k = torch.full((b,), sk, dtype=torch.int32, device=device)
    block_tables = torch.randint(0, nb, (b, max_blks_per_seq),
                                 dtype=torch.int32, device=device)

    return q, k, v, cu, seq_lens_k, block_tables, scale, b, sq, sk, blk, nb


def run_ck(out, q, k, v, bt, seq_lens_k, cu, scale):
    unified_attention_fwd(
        out, q, k, v, bt, seq_lens_k, cu,
        mask_type=2, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )


def run_triton(out, q, k, v, bt, seq_lens_k, cu, sq, sk, scale, window):
    ua_mod.unified_attention(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
        max_seqlen_q=sq, seqused_k=seq_lens_k, max_seqlen_k=sk,
        softmax_scale=scale, causal=True, window_size=window,
        block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
    )


def time_fn(fn, warmup, iters, use_graph=False):
    """Time `fn` either eager or via CUDA graph replay.

    Graph mode matches the methodology of bench_ck_vs_triton_from_jsonl.py
    (which produced pawel-2d-3d.csv): capture once, replay `iters` times,
    measure pure replay wall-time. This strips Python/torch dispatch and
    Triton autotune-cache-lookup overhead, so it isolates GPU kernel cost.
    Eager mode includes those overheads on every iteration.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[iters // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/workspace/aiter/pawel-2d-3d.csv")
    p.add_argument("--idx", default=",".join(str(x) for x in DEFAULT_IDS),
                   help="comma-separated list of csv idx values to verify")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--num-blocks-cap", type=int, default=16384,
                   help="upper bound on KV-page pool size (memory cap)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--use-graph", action="store_true",
                   help="time via CUDA graph replay (matches the methodology "
                        "used by bench_ck_vs_triton_from_jsonl.py / the CSV)")
    args = p.parse_args()

    want = set(args.idx.split(","))
    rows_by_idx = {}
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if row["idx"] in want:
                rows_by_idx[row["idx"]] = row

    missing = want - set(rows_by_idx)
    if missing:
        raise SystemExit(f"idx not found in csv: {sorted(missing)}")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(42)

    print(f"warmup={args.warmup}  iters={args.iters}  dtype={args.dtype}  "
          f"num_blocks_cap={args.num_blocks_cap}  "
          f"timing={'cuda-graph replay' if args.use_graph else 'eager'}")
    try:
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.gcnArchName}  CUs={gpu.multi_processor_count}  "
              f"L2={gpu.L2_cache_size>>20}MB  HBM={gpu.total_memory>>20}MB")
    except Exception:
        pass
    print()
    hdr = (f"{'idx':>4} {'phase':>7} {'b':>5} {'sq':>5} {'sk':>7} {'blk':>3} "
           f"{'win':>7} | {'CK ms':>8} {'Tri ms':>8} {'speedup':>8} "
           f"{'CSV CK':>7} {'CSV t2':>7} {'CSV t3':>7} {'CSV win':>7} | "
           f"{'match':>5} {'maxdiff':>9}")
    print(hdr)
    print("-" * len(hdr), flush=True)

    summary = []  # collected (idx, phase, ck_ms, tri_ms, sp, csv_sp, match, max_diff)

    for idx_str in args.idx.split(","):
        row = rows_by_idx[idx_str]
        window = parse_window(row["window_size"])
        # CK unified_attention_fwd is causal-only; we pass the CSV's window
        # through to Triton as-is so we faithfully reproduce the CSV's
        # comparison. For sk=1 the window is moot (only one KV slot) and the
        # outputs still match; for sk>1 + SWA they would diverge and 'match'
        # will report False — exactly what we want to surface.

        try:
            q, k, v, cu, seq_lens_k, bt, scale, b, sq, sk, blk, nb = make_tensors(
                row, args.num_blocks_cap, dtype=dtype
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"{idx_str:>4}  [skip: OOM at this shape: {e}]")
            torch.cuda.empty_cache()
            continue

        out_ck     = torch.zeros_like(q)
        out_triton = torch.zeros_like(q)

        try:
            run_ck(out_ck, q, k, v, bt, seq_lens_k, cu, scale)
            torch.cuda.synchronize()
            ck_ok = (not torch.isnan(out_ck).any().item()
                     and not (out_ck == 0).all().item())
        except Exception as e:
            print(f"{idx_str:>4}  CK error: {type(e).__name__}: {e}")
            ck_ok = False

        run_triton(out_triton, q, k, v, bt, seq_lens_k, cu, sq, sk, scale, window)
        torch.cuda.synchronize()

        if ck_ok:
            diff = (out_ck.float() - out_triton.float()).abs()
            max_diff = diff.max().item()
            match = max_diff < 2e-2
        else:
            max_diff = -1.0
            match = False

        if ck_ok:
            ck_ms = time_fn(
                lambda: run_ck(out_ck, q, k, v, bt, seq_lens_k, cu, scale),
                args.warmup, args.iters, use_graph=args.use_graph,
            )
        else:
            ck_ms = float("nan")

        tri_ms = time_fn(
            lambda: run_triton(out_triton, q, k, v, bt, seq_lens_k, cu, sq, sk, scale, window),
            args.warmup, args.iters, use_graph=args.use_graph,
        )

        sp = tri_ms / ck_ms if ck_ms == ck_ms else float("nan")
        csv_ck = float(row["ck_ua_ms"])
        csv_t2 = float(row["triton_2d_ms"])
        csv_t3 = float(row["triton_3d_ms"])
        csv_sp = min(csv_t2, csv_t3) / csv_ck

        print(f"{idx_str:>4} {row['phase']:>7} {b:>5d} {sq:>5d} {sk:>7d} "
              f"{blk:>3d} {row['window_size']:>7s} | "
              f"{ck_ms:>8.4f} {tri_ms:>8.4f} {sp:>7.2f}x "
              f"{csv_ck:>7.4f} {csv_t2:>7.4f} {csv_t3:>7.4f} {csv_sp:>6.2f}x | "
              f"{str(match):>5} {max_diff:>9.6f}", flush=True)
        summary.append((idx_str, row["phase"], ck_ms, tri_ms, sp, csv_sp, match, max_diff))

        # Free between rows so we don't keep huge KV pools resident.
        del q, k, v, cu, seq_lens_k, bt, out_ck, out_triton
        torch.cuda.empty_cache()

    if summary:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        n = len(summary)
        n_match = sum(1 for s in summary if s[6])
        n_ck_faster_or_tied = sum(1 for s in summary if s[4] >= 0.99)
        n_ck_faster = sum(1 for s in summary if s[4] >= 1.05)
        n_regression = sum(1 for s in summary if s[4] < 0.95)
        sps = [s[4] for s in summary]
        csv_sps = [s[5] for s in summary]
        diffs = [s[7] for s in summary if s[7] >= 0]
        sps.sort(); csv_sps.sort()
        def med(xs):
            return xs[len(xs)//2] if xs else float("nan")
        print(f"  rows verified           : {n}")
        print(f"  outputs match Triton    : {n_match}/{n}  "
              f"(max abs diff over all rows: {max(diffs) if diffs else float('nan'):.6f})")
        print(f"  CK >= Triton (sp >=0.99): {n_ck_faster_or_tied}/{n}")
        print(f"  CK > Triton  (sp >=1.05): {n_ck_faster}/{n}")
        print(f"  regressions (sp <0.95)  : {n_regression}/{n}")
        print(f"  measured speedup median : {med(sps):.2f}x   p10={sps[max(0,len(sps)//10)]:.2f}x   p90={sps[min(len(sps)-1, len(sps)*9//10)]:.2f}x")
        print(f"  CSV speedup median      : {med(csv_sps):.2f}x   p10={csv_sps[max(0,len(csv_sps)//10)]:.2f}x   p90={csv_sps[min(len(csv_sps)-1, len(csv_sps)*9//10)]:.2f}x")

        # By phase
        for ph in sorted({s[1] for s in summary}):
            ph_sps = sorted(s[4] for s in summary if s[1] == ph)
            ph_csv = sorted(s[5] for s in summary if s[1] == ph)
            print(f"  {ph:<7s} ({len(ph_sps):>2}): measured median {med(ph_sps):.2f}x   "
                  f"CSV median {med(ph_csv):.2f}x")


if __name__ == "__main__":
    main()
