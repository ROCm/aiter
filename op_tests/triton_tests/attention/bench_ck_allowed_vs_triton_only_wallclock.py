#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
End-to-end wall-clock comparison: production wrapper with vs without CK in the mix.

For each CSV row, we time `aiter.ops.triton.attention.unified_attention(...)`
under TWO configurations:

  - "triton_only": both CK short-circuits inside the wrapper are forcibly
    disabled (`_try_ck_unified_attention`, `_try_ck_splitkv_attention`).
    The wrapper still uses its `use_2d_kernel(...)` heuristic to pick
    Triton 2D vs 3D, but it can never bypass to CK.

  - "ck_allowed": production wrapper as-is. The wrapper is free to
    short-circuit to CK (`_try_ck_unified_attention`) for the GQA decode
    sweet spot it was tuned for, otherwise it falls back to Triton 2D/3D.

Both configurations are timed in two modes:

  - eager: `for _ in range(iters): ua_mod.unified_attention(...)` with a
    final `torch.cuda.synchronize()`. Captures full host+kernel cost.

  - graph: capture one call into `torch.cuda.CUDAGraph()`, then
    `iters * graph.replay()`. Captures fixed-overhead replay path used by
    vLLM/TGI in production.

Output: a CSV with all four wall-clock numbers, the implied speedup of
having CK in the mix, and the production CK-allowed dispatch decision
per row (CK / Triton-2D / Triton-3D).

Usage:
    python bench_ck_allowed_vs_triton_only_wallclock.py \
        --csv pawel-2d-3d.csv \
        --idx 873,851,...,1097 \
        --warmup 10 --iters 100 \
        --out-csv pawel-2d-3d_50rows_ck_in_mix_wallclock.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import torch

from aiter.ops.triton.attention import unified_attention as ua_mod


DEFAULT_IDS = "873,851,885,37,835,839,21,1483,1851,74,949,2101,2419,82,101,1963,1849,90,629,759,1329,170,793,1251,137,723,177,1241,174,1375,387,1113,1165,987,249,407,359,1115,549,971,233,415,529,317,523,547,441,557,561,1097"


def make_tensors(row, num_blocks_cap=16384, dtype=torch.bfloat16):
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
    nb = min(num_blocks_cap, max(1024, 2 * max_blks_per_seq))

    q = torch.randn(total_q, hq, d, dtype=dtype, device="cuda")
    k = torch.randn(nb, blk, hk, d, dtype=dtype, device="cuda")
    v = torch.randn_like(k)

    if sq * b == total_q:
        cu = torch.arange(0, b + 1, dtype=torch.int32, device="cuda") * sq
    else:
        base = total_q // b
        rem = total_q - base * b
        lens = [min(sq, base + (1 if i < rem else 0)) for i in range(b)]
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
                          dtype=torch.int32, device="cuda")

    seq_lens_k = torch.full((b,), sk, dtype=torch.int32, device="cuda")
    block_tables = torch.randint(0, nb, (b, max_blks_per_seq),
                                 dtype=torch.int32, device="cuda")
    return q, k, v, cu, seq_lens_k, block_tables, scale, b, sq, sk, blk, nb


# ---------------------------------------------------------------------------
# Heuristic patching: enable / disable CK short-circuit.
# ---------------------------------------------------------------------------

class _Patcher:
    """Snapshot + replace the CK short-circuit functions in the Triton
    wrapper so we can flip between 'CK allowed' and 'Triton only'.
    Restores on __exit__."""
    SHORT_CIRCUIT_ATTRS = ("_try_ck_unified_attention", "_try_ck_splitkv_attention")

    def __init__(self, allow_ck):
        self.allow_ck = allow_ck
        self._saved = {}

    def __enter__(self):
        if not self.allow_ck:
            for attr in self.SHORT_CIRCUIT_ATTRS:
                if hasattr(ua_mod, attr):
                    self._saved[attr] = getattr(ua_mod, attr)
                    setattr(ua_mod, attr, lambda *a, **kw: False)
        return self

    def __exit__(self, *exc):
        for attr, fn in self._saved.items():
            setattr(ua_mod, attr, fn)


def _classify_dispatch(q, k, v, cu, sq, seq_lens_k, sk, scale, window, bt):
    """Predict which backend the production wrapper would pick for these
    inputs, by replicating the wrapper's branch logic. Avoids monkey-
    patching the JIT kernel objects (which break the `kernel[...]` launch
    syntax in unified_attention.py)."""
    # 1) Would the CK short-circuit fire?
    saved_ck = getattr(ua_mod, "_try_ck_unified_attention", None)
    if saved_ck is not None:
        # Pass a throwaway output so the kernel run inside _try_ck doesn't
        # corrupt anything we care about.
        scratch = torch.empty_like(q)
        if saved_ck(q, k, v, scratch, cu, sq, seq_lens_k, sk, scale,
                    window, bt, 0.0, None, None):
            torch.cuda.synchronize()
            return "ck"

    # 2) Else mirror unified_attention()'s 2D-vs-3D selection.
    import triton
    head_size = q.shape[2]
    num_kv_heads = k.shape[2]
    num_query_heads = q.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_M = (16 if num_queries_per_kv <= 16
               else triton.next_power_of_2(num_queries_per_kv))
    block_Q = block_M // num_queries_per_kv
    cu_count = ua_mod.get_num_sms()
    num_seqs = len(seq_lens_k)
    total_num_q_blocks = q.shape[0] // block_Q + num_seqs
    target_num_prgms = cu_count * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    SLIDING_WINDOW = 1 + window[0]
    ALL_DECODE = sq == 1
    if ua_mod.use_2d_kernel(head_size, SLIDING_WINDOW, ALL_DECODE, sq, sk,
                             target_num_prgms, num_2d_prgms):
        return "triton_2d"
    return "triton_3d"


# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------

def time_eager(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms / call


def time_graph(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/workspace/aiter/pawel-2d-3d.csv")
    ap.add_argument("--idx", default=DEFAULT_IDS,
                    help="comma-separated idx list, or 'all' for every row "
                         "with usable timings in the CSV")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=100)
    ap.add_argument("--num-blocks-cap", type=int, default=16384)
    ap.add_argument("--out-csv",
                    default="/workspace/aiter/pawel-2d-3d_50rows_ck_in_mix_wallclock.csv")
    ap.add_argument("--print-every", type=int, default=1,
                    help="print every Nth row's per-row line (1 = every row)")
    args = ap.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(f"csv not found: {args.csv}")

    src_rows = {}
    if args.idx == "all":
        with open(args.csv) as f:
            for r in csv.DictReader(f):
                if r.get("ck_ua_ms") and r.get("triton_3d_ms"):
                    src_rows[r["idx"]] = r
        idx_list = list(src_rows.keys())
    else:
        idx_list = args.idx.split(",")
        with open(args.csv) as f:
            for r in csv.DictReader(f):
                if r["idx"] in idx_list:
                    src_rows[r["idx"]] = r
        missing = [i for i in idx_list if i not in src_rows]
        if missing:
            raise SystemExit(f"missing idx in csv: {missing}")

    print(f"GPU: {torch.cuda.get_device_name(0)}  arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    print(f"warmup={args.warmup} iters={args.iters} rows={len(idx_list)}")
    print()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    hdr = (f"{'idx':>5} {'phase':>7} {'b':>4} {'sk':>6} | "
           f"{'eager_T':>8} {'eager_CK':>8} {'sp_e':>5} | "
           f"{'graph_T':>8} {'graph_CK':>8} {'sp_g':>5} | {'backend':>9}")
    print(hdr); print("-" * len(hdr))

    n_total = len(idx_list)
    n_failed = 0
    for i_row, idx in enumerate(idx_list):
        row = src_rows[idx]
        try:
            torch.manual_seed(42)
            q, k, v, cu, seq_lens_k, bt, scale, b, sq, sk, blk, nb = make_tensors(
                row, num_blocks_cap=args.num_blocks_cap)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"{idx:>5} {row['phase']:>7}  SKIP make_tensors: {str(e)[:120]}")
            n_failed += 1
            torch.cuda.empty_cache()
            continue
        out = torch.empty_like(q)
        window = tuple(int(x) for x in row["window_size"].split(","))

        def _call():
            ua_mod.unified_attention(
                q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=sq,
                seqused_k=seq_lens_k, max_seqlen_k=sk, softmax_scale=scale,
                causal=True, window_size=window, block_table=bt, softcap=0.0,
                q_descale=None, k_descale=None, v_descale=None,
                alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
            )

        try:
            backend = _classify_dispatch(q, k, v, cu, sq, seq_lens_k, sk,
                                         scale, window, bt)

            with _Patcher(allow_ck=False):
                ms_eager_triton_only = time_eager(_call, args.warmup, args.iters)
            with _Patcher(allow_ck=True):
                ms_eager_ck_allowed  = time_eager(_call, args.warmup, args.iters)

            with _Patcher(allow_ck=False):
                ms_graph_triton_only = time_graph(_call, args.warmup, args.iters)
            with _Patcher(allow_ck=True):
                ms_graph_ck_allowed  = time_graph(_call, args.warmup, args.iters)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"{idx:>5} {row['phase']:>7} {b:>4} {sk:>6}  SKIP timing: {str(e)[:120]}")
            n_failed += 1
            del q, k, v, cu, seq_lens_k, bt, out
            torch.cuda.empty_cache()
            continue

        sp_eager = ms_eager_triton_only / ms_eager_ck_allowed
        sp_graph = ms_graph_triton_only / ms_graph_ck_allowed

        if (i_row % args.print_every) == 0 or i_row == n_total - 1:
            prog = f"[{i_row+1:>4}/{n_total}]" if n_total > 50 else ""
            print(f"{prog} {idx:>5} {row['phase']:>7} {b:>4} {sk:>6} | "
                  f"{ms_eager_triton_only:>8.4f} {ms_eager_ck_allowed:>8.4f} "
                  f"{sp_eager:>4.2f}x | "
                  f"{ms_graph_triton_only:>8.4f} {ms_graph_ck_allowed:>8.4f} "
                  f"{sp_graph:>4.2f}x | {backend:>9}", flush=True)

        rows_out.append({
            "idx": idx, "phase": row["phase"],
            "num_seqs": b, "total_q_tokens": row["total_q_tokens"],
            "max_seqlen_q": sq, "max_seqlen_k": sk,
            "num_q_heads": row["num_q_heads"], "num_kv_heads": row["num_kv_heads"],
            "head_size": row["head_size"], "block_size": blk,
            "window_size": row["window_size"], "q_dtype": "torch.bfloat16",
            "softcap": 0.0, "has_sinks": False, "mask": "causal",
            "csv_ck_ua_ms": row["ck_ua_ms"], "csv_triton_2d_ms": row["triton_2d_ms"],
            "csv_triton_3d_ms": row["triton_3d_ms"], "csv_best": row["best"],
            "ck_allowed_backend": backend,
            "eager_triton_only_ms": ms_eager_triton_only,
            "eager_ck_allowed_ms":  ms_eager_ck_allowed,
            "eager_speedup_ck":      sp_eager,
            "graph_triton_only_ms": ms_graph_triton_only,
            "graph_ck_allowed_ms":  ms_graph_ck_allowed,
            "graph_speedup_ck":      sp_graph,
        })

        # free big tensors
        del q, k, v, cu, seq_lens_k, bt, out
        torch.cuda.empty_cache()

    if rows_out:
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print()
        print(f"wrote: {out_path}  ({len(rows_out)} rows; {n_failed} skipped)")


if __name__ == "__main__":
    main()
