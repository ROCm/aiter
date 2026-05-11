#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Bench wrapper for /workspace/attn_3d_info.json (production decode shapes from
MI355X, GQA-6 hd=128 blk=64, FP8 Q + uint8 KV cache).

Built on top of bench_ck_vs_triton_from_jsonl.py.  Two pragmatic adjustments:

  1. FP8 fallback.  CK unified_attention TORCH_CHECKs for fp16/bf16 only
     (csrc/py_itfs_ck/unified_attention_ck_kernels.cu:26).  Until we wire
     FP8 dispatch through CK + descale tensors through the Triton wrapper,
     we map "torch.float8_e4m3fn" -> bf16 here and run apples-to-apples
     across all 5 backends.  This still produces the correct heuristic
     decision (the gate is dtype-independent) and a faithful relative
     comparison; absolute bandwidth numbers will differ from real FP8
     production by ~2x.

  2. Block-pool cap.  The largest shape (num_seqs=512, sk=196608, blk=64)
     would need ~200 GB of KV at full block pool, which OOMs even on
     MI355X 288 GB.  We cap num_phys at --num-blocks-cap (default 16384)
     and let block_table indices alias into the smaller pool, identical
     to how bench_ck_allowed_vs_triton_only_wallclock.py handles it.

Usage:
    python bench_attn_3d_info.py
    python bench_attn_3d_info.py --jsonl /workspace/attn_3d_info.json \
        --warmup 10 --iters 30 --num-blocks-cap 16384 \
        --out-csv /workspace/attn_3d_info_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import json

import torch

# Reuse helpers from the existing tracked script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_ck_vs_triton_from_jsonl as base


# bf16 fallback for FP8 (used unless --fp8 is passed).
base.DTYPE_MAP["torch.float8_e4m3fn"] = torch.bfloat16
base.DTYPE_MAP["torch.float8_e5m2"]   = torch.bfloat16


# Real-FP8 dtype mapping from the JSONL trace strings.
FP8_DTYPE_MAP = {
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e5m2":   torch.float8_e5m2,
    "torch.bfloat16":      torch.bfloat16,
    "torch.float16":       torch.float16,
}


def load_and_dedup(path: Path, max_shapes: int):
    """Variant of base.load_and_dedup that accepts the attn_3d_info.json
    schema, where the sliding-window field is `sliding_window` (not
    `window_size`) and there is no `block_table_shape[1]` ``cols`` value
    we can rely on for paged-KV layout (we synthesize one).
    """
    counts: dict = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            win = r.get("window_size", r.get("sliding_window", [-1, -1]))
            if "block_table_shape" in r:
                blk_cols = int(r["block_table_shape"][1])
            else:
                # synthesize: ceil(sk / blk)
                sk = int(r["max_seqlen_k"])
                blk = int(r["block_size"])
                blk_cols = max((sk + blk - 1) // blk, 1)
            s = base.Shape(
                num_seqs=int(r["num_seqs"]),
                total_q_tokens=int(r["q_shape"][0]),
                max_seqlen_q=int(r["max_seqlen_q"]),
                max_seqlen_k=int(r["max_seqlen_k"]),
                num_query_heads=int(r["num_query_heads"]),
                num_kv_heads=int(r["num_kv_heads"]),
                head_size=int(r["head_size"]),
                block_size=int(r["block_size"]),
                block_table_cols=blk_cols,
                window_size=tuple(win),
                q_dtype=str(r["q_dtype"]),
                softcap=float(r.get("softcap", 0.0)),
                has_sinks=bool(r.get("has_sinks", False)),
            )
            counts[s] = counts.get(s, 0) + 1
    result = sorted(counts.items(), key=lambda x: -x[1])
    if max_shapes > 0:
        result = result[:max_shapes]
    return result


# Shadow make_tensors with a num_blocks_cap-aware version.
_orig_make_tensors = base.make_tensors


def _capped_make_tensors(s, num_blocks_cap, fp8=False, device="cuda"):
    """num_blocks_cap-aware tensor maker.

    fp8=False: q/k/v all in bf16 (or fp16 if logged) -- the fallback path
        used when sharing the bench with CK backends.
    fp8=True : matches production.  q is generated as bf16, cast to fp8
        (fp8_e4m3fn).  kv_cache is generated as bf16, cast to fp8, then
        viewed as torch.uint8 (this is exactly how vLLM stores fp8 KV
        cache).  Output buffer is bf16, matching out_dtype in the trace.
        q_descale must remain None (Triton wrapper asserts so); the kernel
        does the fp8->fp32 dequant internally.  k_descale / v_descale we
        leave as None -- realistic case in the trace did not log a per-
        head scale tensor; if you want to feed scales, pass them via the
        ua_mod.unified_attention call site instead.
    """
    blk = s.block_size
    needed = max((s.max_seqlen_k + blk - 1) // blk, 1)
    num_phys = min(num_blocks_cap, max(needed * max(s.num_seqs, 1) * 2, 64))

    if fp8:
        q_dtype = FP8_DTYPE_MAP.get(s.q_dtype, torch.float8_e4m3fn)
        # NOTE: the trace logs k_dtype/v_dtype as torch.uint8 because vLLM
        # stores fp8 KV cache as uint8 bytes.  But the Triton kernel needs
        # the tensor's dtype to be fp8 so its `K.to(Q.dtype)` cast works.
        # In production the wrapper hands fp8-dtype tensors to the kernel
        # (the uint8 view only exists in the cache allocator).  Match that
        # here: k/v stay as fp8, no .view(uint8).
        q_bf = torch.randn(s.total_q_tokens, s.num_query_heads, s.head_size,
                           dtype=torch.bfloat16, device=device)
        q = q_bf.to(q_dtype)
        k_bf = torch.randn(num_phys, blk, s.num_kv_heads, s.head_size,
                           dtype=torch.bfloat16, device=device)
        v_bf = torch.randn_like(k_bf)
        k = k_bf.to(q_dtype)
        v = v_bf.to(q_dtype)
    else:
        dtype = base.DTYPE_MAP[s.q_dtype]
        q = torch.randn(s.total_q_tokens, s.num_query_heads, s.head_size,
                        dtype=dtype, device=device)
        k = torch.randn(num_phys, blk, s.num_kv_heads, s.head_size,
                        dtype=dtype, device=device)
        v = torch.randn_like(k)

    q_lens = base._synth_q_lens(s.total_q_tokens, s.num_seqs)
    cu_list = [0]
    for ql in q_lens:
        cu_list.append(cu_list[-1] + ql)
    cu_seqlens_q = torch.tensor(cu_list, dtype=torch.int32, device=device)
    seq_lens_k = torch.full((s.num_seqs,), s.max_seqlen_k,
                            dtype=torch.int32, device=device)
    block_tables = torch.randint(0, num_phys,
                                 (s.num_seqs, s.block_table_cols),
                                 dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(s.head_size)
    return q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale


def _bench_triton_fp8(s, warmup, iters, force_2d, inp, use_graph,
                      out_dtype=torch.bfloat16):
    """FP8-aware analogue of base.bench_triton.

    Differs from base.bench_triton only in:
      - allocates `out` as out_dtype (bf16 in production) instead of
        torch.empty_like(q) which would be fp8.
      - patches `use_2d_kernel` to force the path under test.

    Matches base.bench_triton in: monkey-patching the CK short-circuits to
    False so neither `_try_ck_unified_attention` nor `_try_ck_splitkv_attention`
    can bypass the Triton dispatch we're trying to time.
    """
    from aiter.ops.triton.attention import unified_attention as ua_mod
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = inp
    out = torch.empty(q.shape[0], q.shape[1], q.shape[2],
                       dtype=out_dtype, device=q.device)
    saved_splitkv = getattr(ua_mod, "_try_ck_splitkv_attention", None)
    saved_ua     = getattr(ua_mod, "_try_ck_unified_attention", None)
    if saved_splitkv: ua_mod._try_ck_splitkv_attention = lambda *a, **kw: False
    if saved_ua:      ua_mod._try_ck_unified_attention = lambda *a, **kw: False
    saved_use2d = ua_mod.use_2d_kernel
    ua_mod.use_2d_kernel = ((lambda *a, **kw: True) if force_2d
                            else (lambda *a, **kw: False))
    kw = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q,
              max_seqlen_q=s.max_seqlen_q, seqused_k=seq_lens_k,
              max_seqlen_k=s.max_seqlen_k, softmax_scale=scale,
              causal=True, window_size=s.window_size, block_table=block_tables,
              softcap=s.softcap, q_descale=None, k_descale=None, v_descale=None,
              alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)
    try:
        def fn():
            ua_mod.unified_attention(**kw)
        return base._timed(fn, warmup, iters, use_graph)
    finally:
        ua_mod.use_2d_kernel = saved_use2d
        if saved_splitkv: ua_mod._try_ck_splitkv_attention = saved_splitkv
        if saved_ua:      ua_mod._try_ck_unified_attention = saved_ua


def main() -> int:
    torch.manual_seed(42)
    ap = argparse.ArgumentParser(
        description="Bench CK-UA / CK-PK / CK-SK / Triton 2D / Triton 3D "
                    "on /workspace/attn_3d_info.json shapes.")
    ap.add_argument("--jsonl", type=Path,
                    default=Path("/workspace/attn_3d_info.json"))
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=30)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--num-blocks-cap", type=int, default=16384,
                    help="Upper bound on num_phys to keep KV pool < ~1 GB")
    ap.add_argument("--fp8", action="store_true",
                    help="Use real FP8 dtypes from the trace "
                         "(fp8_e4m3fn Q + uint8 KV + bf16 out).  "
                         "Default: bf16 fallback.")
    ap.add_argument("--with-ck", action="store_true",
                    help="Also bench CK unified_attention (only valid for "
                         "non-FP8: CK kernel rejects fp8).  Adds a ck_ua "
                         "column to the CSV.")
    ap.add_argument("--out-csv", type=Path,
                    default=Path("/workspace/attn_3d_info_bench.csv"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        return 2

    shapes = load_and_dedup(args.jsonl, args.max_shapes)
    if args.decode_only:
        shapes = [(s, c) for s, c in shapes if s.max_seqlen_q == 1]
    if not shapes:
        print("No shapes found.")
        return 1

    total = sum(c for _, c in shapes)
    print(f"JSONL: {total} entries -> {len(shapes)} unique shapes")
    print(f"warmup={args.warmup}  iters={args.iters}  graph={'off' if args.no_graph else 'on'}  "
          f"num_blocks_cap={args.num_blocks_cap}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  arch: "
          f"{torch.cuda.get_device_properties(0).gcnArchName}")

    ok = [(s, c) for s, c in shapes if base.compatible(s)[0]]
    skip = [(s, c, base.compatible(s)[1]) for s, c in shapes
            if not base.compatible(s)[0]]
    print(f"Compatible: {len(ok)} shapes ({sum(c for _, c in ok)} calls)")
    if skip:
        for r, cnt in sorted(
                {r: sum(c for _, c, r2 in skip if r2 == r)
                 for _, _, r in skip}.items(), key=lambda x: -x[1]):
            print(f"  skip: {r}  ({cnt} calls)")
    print()

    fp8_present = any(s.q_dtype.startswith("torch.float8") for s, _ in ok)
    if fp8_present and not args.fp8:
        print("NOTE: FP8 q_dtype detected; benchmarking with bf16 fallback "
              "(pass --fp8 to use real FP8).")
        print()
    elif args.fp8:
        print("NOTE: --fp8 enabled.  Q=fp8_e4m3fn, K/V=fp8_e4m3fn, "
              "out=bf16.  q/k/v_descale=None.")
        print()

    # Set up heuristic predictor (mirrors unified_attention.use_2d_kernel).
    import triton
    from aiter.ops.triton.attention import unified_attention as ua_mod
    cu_count = ua_mod.get_num_sms()
    print(f"  cu_count = {cu_count}  (target_num_prgms = cu_count*4 = {cu_count*4})")
    print()

    def predict_dispatch(s):
        qpkv = s.num_query_heads // s.num_kv_heads
        block_M = 16 if qpkv <= 16 else triton.next_power_of_2(qpkv)
        block_Q = block_M // qpkv
        total_num_q_blocks = s.total_q_tokens // block_Q + s.num_seqs
        num_2d_prgms = total_num_q_blocks * s.num_kv_heads
        SLIDING_WINDOW = 1 + s.window_size[0]
        ALL_DECODE = (s.max_seqlen_q == 1)
        # Mirror the call site (unified_attention.py near line 270): the
        # target_num_prgms is sk-aware (8*CU long-ctx, 4*CU short-ctx).
        target_num_prgms = cu_count * (8 if s.max_seqlen_k > 4096 else 4)
        chose_2d = ua_mod.use_2d_kernel(
            s.head_size, SLIDING_WINDOW, ALL_DECODE,
            s.max_seqlen_q, s.max_seqlen_k, target_num_prgms, num_2d_prgms)
        return ("triton_2d" if chose_2d else "triton_3d", num_2d_prgms)

    if args.with_ck:
        hdr = (f"{'#':>4s} {'phase':>8s} {'seqs':>5s} {'q_tok':>6s} {'max_q':>6s} "
               f"{'max_k':>6s} {'heads':>5s} {'hdim':>4s} {'blk':>3s} {'win':>7s} "
               f"{'cnt':>4s} {'T-2D':>9s} {'T-3D':>9s} {'CK':>9s} {'best':>6s} "
               f"{'heur':>6s} {'agree':>6s}")
    else:
        hdr = (f"{'#':>4s} {'phase':>8s} {'seqs':>5s} {'q_tok':>6s} {'max_q':>6s} "
               f"{'max_k':>6s} {'heads':>5s} {'hdim':>4s} {'blk':>3s} {'win':>7s} "
               f"{'cnt':>4s} {'T-2D':>9s} {'T-3D':>9s} {'best':>6s} {'heur':>6s} "
               f"{'agree':>6s}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    use_g = not args.no_graph
    BACKENDS = ["t_2d", "t_3d"] + (["ck_ua"] if args.with_ck else [])

    for i, (s, count) in enumerate(ok):
        phase = base.phase_label(s)
        win_str = f"{s.window_size[0]},{s.window_size[1]}"

        try:
            inp = _capped_make_tensors(s, args.num_blocks_cap, fp8=args.fp8)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{i:4d} {phase:>8s} {s.num_seqs:5d}  SKIP (OOM)")
            continue

        results = {}
        if args.fp8:
            bench_t = lambda force_2d: _bench_triton_fp8(
                s, args.warmup, args.iters, force_2d, inp, use_g)
        else:
            bench_t = lambda force_2d: base.bench_triton(
                s, args.warmup, args.iters, force_2d, inp, use_g)
        backends = [
            ("t_2d", lambda: bench_t(True)),
            ("t_3d", lambda: bench_t(False)),
        ]
        if args.with_ck and not args.fp8:
            backends.append(
                ("ck_ua",
                 lambda: base.bench_ck_ua(s, args.warmup, args.iters, inp, use_g)))

        for name, fn in backends:
            try:
                results[name] = fn()
            except Exception as e:
                results[name] = None
                msg = str(e).split("\n")[0][:80]
                print(f"{i:4d} {name} err: {msg}", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # `best` is purely between t_2d / t_3d (the heuristic competes
        # against those).  CK is reported as an extra column only.
        triton_only = {k: v for k, v in results.items()
                        if k in ("t_2d", "t_3d") and v is not None}
        best = min(triton_only, key=triton_only.get) if triton_only else ""
        heur, num_2d_prgms = predict_dispatch(s)
        agree = "yes" if (best and heur == ({"t_2d": "triton_2d",
                                              "t_3d": "triton_3d"}.get(best, ""))) else "NO"
        vals = {k: f"{results[k]:9.4f}" if results.get(k) is not None
                else "      err" for k in BACKENDS}

        if args.with_ck:
            print(f"{i:4d} {phase:>8s} {s.num_seqs:5d} {s.total_q_tokens:6d} "
                  f"{s.max_seqlen_q:6d} {s.max_seqlen_k:6d} "
                  f"{s.num_query_heads:3d}/{s.num_kv_heads:<2d} "
                  f"{s.head_size:4d} {s.block_size:3d} "
                  f"{win_str:>7s} {count:4d} "
                  f"{vals['t_2d']:>9s} {vals['t_3d']:>9s} "
                  f"{vals.get('ck_ua', '      n/a'):>9s} "
                  f"{best:>6s} {heur:>9s} {agree:>6s}", flush=True)
        else:
            print(f"{i:4d} {phase:>8s} {s.num_seqs:5d} {s.total_q_tokens:6d} "
                  f"{s.max_seqlen_q:6d} {s.max_seqlen_k:6d} "
                  f"{s.num_query_heads:3d}/{s.num_kv_heads:<2d} "
                  f"{s.head_size:4d} {s.block_size:3d} "
                  f"{win_str:>7s} {count:4d} "
                  f"{vals['t_2d']:>9s} {vals['t_3d']:>9s} {best:>6s} {heur:>9s} "
                  f"{agree:>6s}", flush=True)

        csv_rows.append([
            i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
            s.max_seqlen_k, s.num_query_heads, s.num_kv_heads, s.head_size,
            s.block_size, win_str, s.q_dtype, count,
            f"{results.get('t_2d','')}" if results.get('t_2d') is not None else "",
            f"{results.get('t_3d','')}" if results.get('t_3d') is not None else "",
            best, heur, num_2d_prgms,
            (1 if best and heur == {"t_2d": "triton_2d",
                                     "t_3d": "triton_3d"}.get(best) else 0),
            f"{results.get('ck_ua','')}" if results.get('ck_ua') is not None else "",
        ])

    # Summary.  Column layout (matches CSV writer below):
    #   0:idx 1:phase 2:num_seqs 3:total_q 4:max_q 5:max_k 6:hq 7:hk 8:hd
    #   9:blk 10:win 11:q_dtype 12:count 13:t2_ms 14:t3_ms 15:best 16:heur
    #   17:num_2d_prgms 18:matches 19:ck_ms
    print()
    n = len(csv_rows)
    wins = {"t_2d": 0, "t_3d": 0}
    chose = {"triton_2d": 0, "triton_3d": 0}
    agree_n = 0
    sum_2d = sum_3d = sum_ck = sum_best = sum_heur = 0.0
    sum_best_with_ck = 0.0
    ck_runs = ck_wins_vs_best_triton = 0
    for row in csv_rows:
        best_lbl, heur_lbl = row[15], row[16]
        wins[best_lbl] = wins.get(best_lbl, 0) + 1
        chose[heur_lbl] = chose.get(heur_lbl, 0) + 1
        agree_n += row[18]
        t2 = float(row[13]) if row[13] else None
        t3 = float(row[14]) if row[14] else None
        ck = float(row[19]) if len(row) > 19 and row[19] else None
        if t2 is not None: sum_2d += t2
        if t3 is not None: sum_3d += t3
        if ck is not None:
            sum_ck += ck
            ck_runs += 1
        if t2 is not None and t3 is not None:
            best_t = min(t2, t3)
            sum_best += best_t
            sum_heur += t2 if heur_lbl == "triton_2d" else t3
            if ck is not None:
                sum_best_with_ck += min(best_t, ck)
                if ck < best_t:
                    ck_wins_vs_best_triton += 1
            else:
                sum_best_with_ck += best_t

    print(f"Summary: {n} shapes benchmarked")
    print(f"  fastest backend on each shape:  T2D={wins['t_2d']}/{n}  "
          f"T3D={wins['t_3d']}/{n}")
    print(f"  use_2d_kernel heuristic chose:  T2D={chose['triton_2d']}/{n}  "
          f"T3D={chose['triton_3d']}/{n}")
    print(f"  heuristic matches actual best:  {agree_n}/{n} "
          f"({100*agree_n/n:.1f}%)")
    if args.with_ck:
        print(f"  CK ran on:                     {ck_runs}/{n}  "
              f"(beats best Triton in {ck_wins_vs_best_triton}/{ck_runs} "
              f"= {100*ck_wins_vs_best_triton/max(ck_runs,1):.1f}%)")
    print()
    print(f"Sum of per-call ms across {n} shapes (each shape \u00d7 1 call):")
    print(f"  always Triton 2D       : {sum_2d:>9.3f} ms  ({100*(sum_2d/sum_best-1):+5.2f}% over Triton-only oracle)")
    print(f"  always Triton 3D       : {sum_3d:>9.3f} ms  ({100*(sum_3d/sum_best-1):+5.2f}% over Triton-only oracle)")
    print(f"  use_2d_kernel heuristic: {sum_heur:>9.3f} ms  ({100*(sum_heur/sum_best-1):+5.2f}% over Triton-only oracle)")
    print(f"  Triton-only oracle     : {sum_best:>9.3f} ms  (= 0%)")
    if args.with_ck:
        print(f"  CK unified_attention   : {sum_ck:>9.3f} ms  (across {ck_runs} shapes that ran CK)")
        print(f"  best(T2D,T3D,CK) oracle: {sum_best_with_ck:>9.3f} ms  "
              f"({100*(sum_best_with_ck/sum_best-1):+5.2f}% vs Triton-only oracle)")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "idx", "phase", "num_seqs", "total_q_tokens", "max_seqlen_q",
                "max_seqlen_k", "num_q_heads", "num_kv_heads", "head_size",
                "block_size", "window_size", "q_dtype_logged", "trace_count",
                "triton_2d_ms", "triton_3d_ms",
                "best",                # measured: t_2d or t_3d
                "heuristic_choice",    # use_2d_kernel(): triton_2d or triton_3d
                "num_2d_prgms",
                "heuristic_matches_best",
                "ck_ua_ms",            # blank when CK didn't run for this shape
            ])
            w.writerows(csv_rows)
        print(f"\nCSV: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
