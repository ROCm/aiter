#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Benchmark 4 attention backends on realistic shapes from a JSONL trace:
  1. CK unified attention (42_unified_attention, non-split, paged KV)
  2. CK FMHA split-KV (mha_varlen_fwd with block_table)
  3. Triton 2D (single-pass unified attention)
  4. Triton 3D (split-K unified attention + reduce)

Deduplicates shapes, runs all backends, writes CSV + prints summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from aiter.ops.mha import mha_varlen_fwd
from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention import unified_attention as ua_mod


DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


@dataclass(frozen=True)
class Shape:
    """Unique shape key extracted from JSONL."""
    num_seqs: int
    total_q_tokens: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    block_table_cols: int
    window_size: tuple[int, int]
    q_dtype: str
    softcap: float
    has_sinks: bool


def load_and_dedup(path: Path, max_shapes: int) -> list[tuple[Shape, int]]:
    """Load JSONL, deduplicate by shape, return (shape, count) pairs."""
    counts: dict[Shape, int] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = Shape(
                num_seqs=int(r["num_seqs"]),
                total_q_tokens=int(r["q_shape"][0]),
                max_seqlen_q=int(r["max_seqlen_q"]),
                max_seqlen_k=int(r["max_seqlen_k"]),
                num_query_heads=int(r["num_query_heads"]),
                num_kv_heads=int(r["num_kv_heads"]),
                head_size=int(r["head_size"]),
                block_size=int(r["block_size"]),
                block_table_cols=int(r["block_table_shape"][1]),
                window_size=tuple(r["window_size"]),
                q_dtype=str(r["q_dtype"]),
                softcap=float(r.get("softcap", 0.0)),
                has_sinks=bool(r.get("has_sinks", False)),
            )
            counts[s] = counts.get(s, 0) + 1
    result = sorted(counts.items(), key=lambda x: -x[1])
    if max_shapes > 0:
        result = result[:max_shapes]
    return result


def compatible(s: Shape) -> tuple[bool, str]:
    if s.q_dtype not in DTYPE_MAP:
        return False, f"unsupported dtype {s.q_dtype}"
    if s.block_size < 16 or (s.block_size & (s.block_size - 1)) != 0:
        return False, f"block_size={s.block_size} must be power-of-2 >= 16"
    return True, "ok"


def make_tensors(s: Shape, device: str = "cuda"):
    dtype = DTYPE_MAP[s.q_dtype]
    blk = s.block_size

    q = torch.randn(s.total_q_tokens, s.num_query_heads, s.head_size,
                     dtype=dtype, device=device)
    needed_blocks = max((s.max_seqlen_k + blk - 1) // blk, 1)
    num_phys_blocks = max(needed_blocks * max(s.num_seqs, 1) * 2, 64)
    k = torch.randn(num_phys_blocks, blk, s.num_kv_heads, s.head_size,
                     dtype=dtype, device=device)
    v = torch.randn_like(k)

    q_lens = _synth_q_lens(s.total_q_tokens, s.num_seqs)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    seq_lens_k = torch.full((s.num_seqs,), s.max_seqlen_k, dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0, num_phys_blocks, (s.num_seqs, s.block_table_cols),
        dtype=torch.int32, device=device,
    )
    scale = 1.0 / math.sqrt(s.head_size)
    return q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale


def _synth_q_lens(total: int, num_seqs: int) -> list[int]:
    base = total // num_seqs
    rem = total % num_seqs
    return [base + (1 if i < rem else 0) for i in range(num_seqs)]



def _timed(fn, warmup, iters, use_graph):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_ck_ua(s: Shape, warmup: int, iters: int, created_input=None, use_graph: bool = True) -> float:
    """Benchmark CK unified attention (non-split, paged KV)."""
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)
    mask_type = 2 if s.window_size != (0, 0) else 0

    def fn():
        unified_attention_fwd(
            out, q, k, v, block_tables, seq_lens_k, cu_seqlens_q,
            mask_type=mask_type, scale_s=scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0)

    return _timed(fn, warmup, iters, use_graph)


def bench_ck_fmha(s: Shape, warmup: int, iters: int, created_input=None, use_graph: bool = True) -> float:
    """Benchmark CK FMHA split-KV via mha_varlen_fwd with paged KV cache."""
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)
    cu_k = torch.nn.functional.pad(
        seq_lens_k.cumsum(0, dtype=torch.int32), (1, 0))
    window_left = s.window_size[0] if s.window_size[0] >= 0 else -1
    window_right = s.window_size[1] if s.window_size[1] >= 0 else 0

    kw = dict(
        q=q, k=k, v=v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_k,
        max_seqlen_q=s.max_seqlen_q, max_seqlen_k=s.max_seqlen_k,
        min_seqlen_q=s.max_seqlen_q, dropout_p=0.0, softmax_scale=scale,
        logits_soft_cap=s.softcap, zero_tensors=False, is_causal=True,
        window_size_left=window_left, window_size_right=window_right,
        sink_size=0, return_softmax_lse=True, return_dropout_randval=False,
        out=out, block_table=block_tables,
    )

    def fn():
        mha_varlen_fwd(**kw)

    return _timed(fn, warmup, iters, use_graph)


def bench_triton(s: Shape, warmup: int, iters: int, force_2d: bool | None = None, created_input=None, use_graph: bool = True) -> float:
    """Benchmark Triton unified attention (2D or 3D)."""
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)

    saved_splitkv = getattr(ua_mod, '_try_ck_splitkv_attention', None)
    saved_ua = getattr(ua_mod, '_try_ck_unified_attention', None)
    if saved_splitkv:
        ua_mod._try_ck_splitkv_attention = lambda *a, **kw: False
    if saved_ua:
        ua_mod._try_ck_unified_attention = lambda *a, **kw: False

    kw = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=s.max_seqlen_q, seqused_k=seq_lens_k,
        max_seqlen_k=s.max_seqlen_k, softmax_scale=scale,
        causal=True, window_size=s.window_size, block_table=block_tables,
        softcap=s.softcap, q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None,
    )

    saved_use2d = ua_mod.use_2d_kernel
    if force_2d is not None:
        ua_mod.use_2d_kernel = (lambda *a, **kw: True) if force_2d else (lambda *a, **kw: False)

    try:
        def fn():
            ua_mod.unified_attention(**kw)

        return _timed(fn, warmup, iters, use_graph)
    finally:
        ua_mod.use_2d_kernel = saved_use2d
        if saved_splitkv:
            ua_mod._try_ck_splitkv_attention = saved_splitkv
        if saved_ua:
            ua_mod._try_ck_unified_attention = saved_ua


def phase_label(s: Shape) -> str:
    return "decode" if s.max_seqlen_q == 1 else "prefill"


def main() -> int:
    torch.manual_seed(42)

    ap = argparse.ArgumentParser(
        description="Benchmark CK UA / CK FMHA split-KV / Triton 2D / Triton 3D from JSONL trace."
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--max-shapes", type=int, default=0, help="0 = all unique shapes")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-graph", action="store_true",
                    help="Skip CUDA graph capture")
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    shapes = load_and_dedup(args.jsonl, args.max_shapes)
    if not shapes:
        print("No shapes found in JSONL.")
        return 1

    total_jsonl = sum(c for _, c in shapes)
    print(f"JSONL: {total_jsonl} entries -> {len(shapes)} unique shapes")
    print(f"warmup={args.warmup}  iters={args.iters}  graph={'off' if args.no_graph else 'on'}")

    ok = [(s, c) for s, c in shapes if compatible(s)[0]]
    skip = [(s, c, compatible(s)[1]) for s, c in shapes if not compatible(s)[0]]
    print(f"Compatible: {len(ok)} shapes ({sum(c for _, c in ok)} calls)")
    if skip:
        for reason, cnt in sorted({r: sum(c for _, c, r2 in skip if r2 == r)
                                    for _, _, r in skip}.items(), key=lambda x: -x[1]):
            print(f"  skip: {reason}  ({cnt} calls)")
    print()

    hdr = (f"{'#':>4s} {'phase':>8s} {'seqs':>5s} {'q_tok':>6s} {'max_q':>6s} "
           f"{'max_k':>6s} {'heads':>5s} {'hdim':>4s} {'win':>7s} {'cnt':>4s} "
           f"{'CK-UA':>8s} {'CK-SK':>8s} {'T-2D':>8s} {'T-3D':>8s} "
           f"{'best':>6s}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows: list[list] = []
    use_g = not args.no_graph

    for i, (s, count) in enumerate(ok):
        phase = phase_label(s)
        win_str = f"{s.window_size[0]},{s.window_size[1]}"
        prefix = (f"{i:4d} {phase:>8s} {s.num_seqs:5d} {s.total_q_tokens:6d} "
                  f"{s.max_seqlen_q:6d} {s.max_seqlen_k:6d} "
                  f"{s.num_query_heads:3d}/{s.num_kv_heads:<2d} {s.head_size:4d} "
                  f"{win_str:>7s} {count:4d}")

        inp = make_tensors(s)
        results = {}

        for name, fn in [
            ("ck_ua",  lambda: bench_ck_ua(s, args.warmup, args.iters, inp, use_g)),
            ("ck_sk",  lambda: bench_ck_fmha(s, args.warmup, args.iters, inp, use_g)),
            ("t_2d",   lambda: bench_triton(s, args.warmup, args.iters, True, inp, use_g)),
            ("t_3d",   lambda: bench_triton(s, args.warmup, args.iters, False, inp, use_g)),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                results[name] = None
                err_short = str(e).split('\n')[0][:60]
                print(f"{prefix} {name} err: {err_short}")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        vals = {k: f"{v:8.4f}" if v is not None else "     err" for k, v in results.items()}
        valid = {k: v for k, v in results.items() if v is not None}
        best = min(valid, key=valid.get) if valid else ""

        print(f"{prefix} {vals['ck_ua']:>8s} {vals['ck_sk']:>8s} "
              f"{vals['t_2d']:>8s} {vals['t_3d']:>8s} {best:>6s}")

        csv_rows.append([
            i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
            s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
            s.head_size, s.block_size, win_str, count,
            f"{results.get('ck_ua', '')}" if results.get('ck_ua') is not None else "",
            f"{results.get('ck_sk', '')}" if results.get('ck_sk') is not None else "",
            f"{results.get('t_2d', '')}" if results.get('t_2d') is not None else "",
            f"{results.get('t_3d', '')}" if results.get('t_3d') is not None else "",
            best,
        ])

    # Summary
    print()
    wins = {"ck_ua": 0, "ck_sk": 0, "t_2d": 0, "t_3d": 0}
    total = 0
    for row in csv_rows:
        b = row[-1]
        if b in wins:
            wins[b] += 1
            total += 1
    print(f"Summary: {total} shapes benchmarked")
    for k in ["ck_ua", "ck_sk", "t_2d", "t_3d"]:
        label = {"ck_ua": "CK Unified Attn",
                 "ck_sk": "CK FMHA SplitKV",
                 "t_2d": "Triton 2D", "t_3d": "Triton 3D"}[k]
        pct = 100 * wins[k] / total if total else 0
        print(f"  {label:>20s}: {wins[k]:4d} wins ({pct:.1f}%)")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "phase", "num_seqs", "total_q_tokens", "max_seqlen_q",
                         "max_seqlen_k", "num_q_heads", "num_kv_heads", "head_size",
                         "block_size", "window_size", "trace_count",
                         "ck_ua_ms", "ck_splitkv_ms",
                         "triton_2d_ms", "triton_3d_ms", "best"])
            w.writerows(csv_rows)
        print(f"\nCSV written to: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
