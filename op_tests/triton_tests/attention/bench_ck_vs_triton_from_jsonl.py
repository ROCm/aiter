#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Benchmark CK unified attention (via AITER JIT) vs Triton unified attention (2D & 3D)
on realistic shapes replayed from a JSONL trace file.

Deduplicates shapes, runs all three backends, writes CSV + prints summary.

CK compile-time constraints:
  - head_size=128 NumQPerKV=1, or head_size=64 NumQPerKV=8
  - page block size >= 32 (kPageBlockSize = 32)
  - mask: no-mask or causal (no sliding window)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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


def ck_compatible(s: Shape) -> tuple[bool, str]:
    """Check if shape is compatible with compiled CK instances."""
    if s.head_size == 128:
        nqpkv = s.num_query_heads // s.num_kv_heads
        if nqpkv != 1:
            return False, f"d128 requires MHA (nqpkv=1), got {nqpkv}"
    elif s.head_size == 64:
        nqpkv = s.num_query_heads // s.num_kv_heads
        if nqpkv != 8:
            return False, f"d64 requires GQA-8 (nqpkv=8), got {nqpkv}"
    else:
        return False, f"unsupported head_size={s.head_size}"

    min_blk = 64 if s.head_size <= 64 else 32
    if s.block_size < min_blk:
        pass  # we'll override block_size at runtime to min_blk

    if s.window_size != (-1, -1):
        return False, f"sliding window {s.window_size} not supported"

    if s.q_dtype not in DTYPE_MAP:
        return False, f"unsupported dtype {s.q_dtype}"

    return True, "ok"


def _ck_block_size(s: Shape) -> int:
    """CK compile-time kPageBlockSize: 64 for HeadSize<=64, 32 otherwise."""
    return 64 if s.head_size <= 64 else 32


def make_tensors(s: Shape, device: str = "cuda", block_size_override: int | None = None):
    """Build tensors for a given shape."""
    dtype = DTYPE_MAP[s.q_dtype]
    blk = block_size_override or s.block_size

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


def bench_ck(s: Shape, warmup: int, iters: int, created_input=None) -> float:
    blk = _ck_block_size(s)
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)
    mask_type = 2  # causal

    for _ in range(warmup):
        unified_attention_fwd(
            out, q, k, v, block_tables, seq_lens_k, cu_seqlens_q,
            mask_type=mask_type,
            scale_s=scale, scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )
    torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        unified_attention_fwd(
            out, q, k, v, block_tables, seq_lens_k, cu_seqlens_q,
            mask_type=mask_type,
            scale_s=scale, scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )
    
    # Profile graph execution
    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters, out


def bench_triton(s: Shape, warmup: int, iters: int, force_2d: bool | None = None, created_input=None) -> float:
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)

    kw = dict(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=s.max_seqlen_q,
        seqused_k=seq_lens_k,
        max_seqlen_k=s.max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=s.window_size,
        block_table=block_tables,
        softcap=s.softcap,
        q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None,
        sinks=None,
    )

    saved = ua_mod.use_2d_kernel
    if force_2d is not None:
        ua_mod.use_2d_kernel = (lambda *a, **kw: True) if force_2d else (lambda *a, **kw: False)

    try:
        for _ in range(warmup):
            ua_mod.unified_attention(**kw)
        torch.cuda.synchronize()

        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ua_mod.unified_attention(**kw)
        
        # Profile graph execution
        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters, out
    finally:
        ua_mod.use_2d_kernel = saved


def phase_label(s: Shape) -> str:
    if s.max_seqlen_q == 1:
        return "decode"
    return "prefill"


def main() -> int:
    
    torch.manual_seed(42)
    
    ap = argparse.ArgumentParser(
        description="Benchmark CK unified attention vs Triton 2D/3D from JSONL trace."
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--max-shapes", type=int, default=0, help="0 = all unique shapes")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Write per-shape CSV results")
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
    print(f"warmup={args.warmup}  iters={args.iters}")

    ck_ok = [(s, c) for s, c in shapes if ck_compatible(s)[0]]
    ck_skip = [(s, c, ck_compatible(s)[1]) for s, c in shapes if not ck_compatible(s)[0]]

    print(f"CK compatible: {len(ck_ok)} shapes ({sum(c for _, c in ck_ok)} calls)")
    print(f"CK skipped:    {len(ck_skip)} shapes ({sum(c for _, c, _ in ck_skip)} calls)")
    if ck_skip:
        reasons = {}
        for _, cnt, reason in ck_skip:
            reasons[reason] = reasons.get(reason, 0) + cnt
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  skip reason: {reason}  ({cnt} calls)")
    print()

    hdr = (f"{'#':>4s} {'phase':>8s} {'seqs':>5s} {'q_tok':>6s} {'max_q':>6s} "
           f"{'max_k':>6s} {'heads':>5s} {'hdim':>4s} {'cnt':>4s} "
           f"{'CK ms':>8s} {'T-2D ms':>8s} {'T-3D ms':>8s} {'best':>5s} {'ratio':>7s}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows = []
    stats = {"ck_wins": 0, "triton_wins": 0, "errors": 0}

    for i, (s, count) in enumerate(ck_ok):
        phase = phase_label(s)
        nqpkv = s.num_query_heads // s.num_kv_heads
        prefix = (f"{i:4d} {phase:>8s} {s.num_seqs:5d} {s.total_q_tokens:6d} "
                  f"{s.max_seqlen_q:6d} {s.max_seqlen_k:6d} "
                  f"{s.num_query_heads:3d}/{s.num_kv_heads:<2d} {s.head_size:4d} {count:4d}")

        blk = _ck_block_size(s)
        created_input = make_tensors(s, block_size_override=blk)
        try:
            ck_ms, out_ck = bench_ck(s, warmup=args.warmup, iters=args.iters, created_input=created_input)
        except Exception as e:
            print(f"{prefix} CK err: {e}")
            stats["errors"] += 1
            csv_rows.append([i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
                             s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
                             s.head_size, count, "", "", "", "", f"ck_error:{e}"])
            continue

        torch.cuda.empty_cache()  # try to reduce OOM risk for Triton runs
        torch.cuda.synchronize()
        
        try:
            t2d, out_triton_2d = bench_triton(s, warmup=args.warmup, iters=args.iters, force_2d=True, created_input=created_input)
            torch.cuda.empty_cache()  # try to reduce OOM risk for Triton runs
            torch.cuda.synchronize()
            t3d, out_triton_3d = bench_triton(s, warmup=args.warmup, iters=args.iters, force_2d=False, created_input=created_input)
        except Exception as e:
            print(f"{prefix} {ck_ms:8.4f} Triton err: {e}")
            stats["errors"] += 1
            csv_rows.append([i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
                             s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
                             s.head_size, count, ck_ms, "", "", "", f"triton_error:{e}"])
            continue

        # print("triton outputs: 2D vs 3D")
        # print("  out_triton_2d:", out_triton_2d.flatten()[:5])
        # print("  out_triton_3d:", out_triton_3d.flatten()[:5])
        # print("  out_ck:       ", out_ck.flatten()[:5])
        try:
            torch.testing.assert_close(out_ck, out_triton_2d, rtol=1e-2, atol=1e-2)
        except Exception as e:
            print(f"{prefix} output mismatch CK vs Triton 2D: {e}")
            stats["errors"] += 1
            csv_rows.append([i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
                             s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
                             s.head_size, count, f"{ck_ms:.6f}", f"{t2d:.6f}", f"{t3d:.6f}",
                             "", f"output_mismatch_ck_vs_t2d:{e}"])
            continue
        
        best_triton = min(t2d, t3d)
        best_label = "2D" if t2d <= t3d else "3D"
        ratio = ck_ms / best_triton if best_triton > 0 else float("inf")

        if ratio < 1.0:
            stats["ck_wins"] += 1
        else:
            stats["triton_wins"] += 1

        ratio_str = f"{ratio:.3f}x"
        print(f"{prefix} {ck_ms:8.4f} {t2d:8.4f} {t3d:8.4f} {best_label:>5s} {ratio_str:>7s}")

        csv_rows.append([i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
                         s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
                         s.head_size, count, f"{ck_ms:.6f}", f"{t2d:.6f}", f"{t3d:.6f}",
                         best_label, ratio_str])

    print()
    total_bench = stats["ck_wins"] + stats["triton_wins"]
    print(f"Summary: {total_bench} shapes benchmarked, {stats['errors']} errors")
    if total_bench > 0:
        print(f"  CK faster:     {stats['ck_wins']:4d} ({100*stats['ck_wins']/total_bench:.1f}%)")
        print(f"  Triton faster: {stats['triton_wins']:4d} ({100*stats['triton_wins']/total_bench:.1f}%)")
    print("ratio: <1.0 = CK faster, >1.0 = Triton faster")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "phase", "num_seqs", "total_q_tokens", "max_seqlen_q",
                         "max_seqlen_k", "num_q_heads", "num_kv_heads", "head_size",
                         "trace_count", "ck_ms", "triton_2d_ms", "triton_3d_ms",
                         "triton_best", "ck_vs_best"])
            w.writerows(csv_rows)
        print(f"\nCSV written to: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
