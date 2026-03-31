#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Benchmark CK FMHA split-KV (via mha_varlen_fwd) and CK unified attention
vs Triton unified attention (2D & 3D) on realistic shapes replayed from a
JSONL trace file.

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


def ck_fmha_compatible(s: Shape) -> tuple[bool, str]:
    """Check if shape is compatible with CK FMHA split-KV (via mha_varlen_fwd)."""
    if s.q_dtype not in DTYPE_MAP:
        return False, f"unsupported dtype {s.q_dtype}"
    if s.block_size < 16 or (s.block_size & (s.block_size - 1)) != 0:
        return False, f"block_size={s.block_size} must be power-of-2 >= 16"
    if s.max_seqlen_q != 1:
        return False, "CK FMHA split-KV benchmark targets decode only"
    return True, "ok"


def make_tensors(s: Shape, device: str = "cuda"):
    """Build tensors for a given shape."""
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


def bench_ck_fmha(s: Shape, warmup: int, iters: int, created_input=None) -> tuple[float, torch.Tensor]:
    """Benchmark CK FMHA split-KV via mha_varlen_fwd with paged KV cache."""
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)

    cu_k = torch.nn.functional.pad(
        seq_lens_k.cumsum(0, dtype=torch.int32), (1, 0))

    window_left = s.window_size[0] if s.window_size[0] >= 0 else -1
    window_right = s.window_size[1] if s.window_size[1] >= 0 else -1

    kw = dict(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=s.max_seqlen_q,
        max_seqlen_k=s.max_seqlen_k,
        min_seqlen_q=s.max_seqlen_q,
        dropout_p=0.0,
        softmax_scale=scale,
        logits_soft_cap=s.softcap,
        zero_tensors=False,
        is_causal=True,
        window_size_left=window_left,
        window_size_right=window_right,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
        out=out,
        block_table=block_tables,
    )

    for _ in range(warmup):
        mha_varlen_fwd(**kw)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mha_varlen_fwd(**kw)

    t0 = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters, out


def bench_triton(s: Shape, warmup: int, iters: int, force_2d: bool | None = None, created_input=None) -> tuple[float, torch.Tensor]:
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale = created_input
    out = torch.empty_like(q)

    saved_splitkv = getattr(ua_mod, '_try_ck_splitkv_attention', None)
    saved_ua = getattr(ua_mod, '_try_ck_unified_attention', None)
    if saved_splitkv:
        ua_mod._try_ck_splitkv_attention = lambda *a, **kw: False
    if saved_ua:
        ua_mod._try_ck_unified_attention = lambda *a, **kw: False

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

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ua_mod.unified_attention(**kw)

        t0 = time.perf_counter()
        for _ in range(iters):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters, out
    finally:
        ua_mod.use_2d_kernel = saved
        if saved_splitkv:
            ua_mod._try_ck_splitkv_attention = saved_splitkv
        if saved_ua:
            ua_mod._try_ck_unified_attention = saved_ua


def phase_label(s: Shape) -> str:
    if s.max_seqlen_q == 1:
        return "decode"
    return "prefill"


def main() -> int:

    torch.manual_seed(42)

    ap = argparse.ArgumentParser(
        description="Benchmark CK FMHA split-KV vs Triton 2D/3D from JSONL trace."
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

    fmha_ok = [(s, c) for s, c in shapes if ck_fmha_compatible(s)[0]]
    fmha_skip = [(s, c, ck_fmha_compatible(s)[1]) for s, c in shapes if not ck_fmha_compatible(s)[0]]

    print(f"CK FMHA compatible: {len(fmha_ok)} shapes ({sum(c for _, c in fmha_ok)} calls)")
    print(f"CK FMHA skipped:    {len(fmha_skip)} shapes ({sum(c for _, c, _ in fmha_skip)} calls)")
    if fmha_skip:
        reasons: dict[str, int] = {}
        for _, cnt, reason in fmha_skip:
            reasons[reason] = reasons.get(reason, 0) + cnt
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  skip reason: {reason}  ({cnt} calls)")
    print()

    hdr = (f"{'#':>4s} {'phase':>8s} {'seqs':>5s} {'q_tok':>6s} {'max_q':>6s} "
           f"{'max_k':>6s} {'heads':>5s} {'hdim':>4s} {'win':>7s} {'cnt':>4s} "
           f"{'FMHA ms':>8s} {'T-2D ms':>8s} {'T-3D ms':>8s} "
           f"{'best':>6s} {'ratio':>7s}")
    print(hdr)
    print("-" * len(hdr))

    csv_rows: list[list] = []
    stats = {"fmha_wins": 0, "triton_wins": 0, "errors": 0}

    for i, (s, count) in enumerate(fmha_ok):
        phase = phase_label(s)
        win_str = f"{s.window_size[0]},{s.window_size[1]}"
        prefix = (f"{i:4d} {phase:>8s} {s.num_seqs:5d} {s.total_q_tokens:6d} "
                  f"{s.max_seqlen_q:6d} {s.max_seqlen_k:6d} "
                  f"{s.num_query_heads:3d}/{s.num_kv_heads:<2d} {s.head_size:4d} "
                  f"{win_str:>7s} {count:4d}")

        created_input = make_tensors(s)

        # -- CK FMHA split-KV --
        fmha_ms_str = ""
        fmha_ms = None
        try:
            fmha_ms, out_fmha = bench_ck_fmha(s, warmup=args.warmup, iters=args.iters,
                                               created_input=created_input)
            fmha_ms_str = f"{fmha_ms:8.4f}"
        except Exception as e:
            fmha_ms_str = "err"
            print(f"{prefix} CK-FMHA err: {e}")
            stats["errors"] += 1

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # -- Triton 2D & 3D --
        t2d_str = ""
        t3d_str = ""
        t2d = t3d = None
        try:
            t2d, _ = bench_triton(s, warmup=args.warmup, iters=args.iters,
                                  force_2d=True, created_input=created_input)
            t2d_str = f"{t2d:8.4f}"
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t3d, _ = bench_triton(s, warmup=args.warmup, iters=args.iters,
                                  force_2d=False, created_input=created_input)
            t3d_str = f"{t3d:8.4f}"
        except Exception as e:
            if not t2d_str:
                t2d_str = "err"
            t3d_str = "err"
            print(f"{prefix} Triton err: {e}")
            stats["errors"] += 1

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # -- Compare FMHA vs best Triton --
        best_label = ""
        ratio_str = ""
        if fmha_ms is not None and t2d is not None and t3d is not None:
            best_triton = min(t2d, t3d)
            best_label = "T-2D" if t2d <= t3d else "T-3D"
            ratio = fmha_ms / best_triton if best_triton > 0 else float("inf")
            ratio_str = f"{ratio:.3f}x"
            if ratio < 1.0:
                stats["fmha_wins"] += 1
            else:
                stats["triton_wins"] += 1

        print(f"{prefix} {fmha_ms_str:>8s} "
              f"{t2d_str:>8s} {t3d_str:>8s} {best_label:>6s} {ratio_str:>7s}")

        csv_rows.append([
            i, phase, s.num_seqs, s.total_q_tokens, s.max_seqlen_q,
            s.max_seqlen_k, s.num_query_heads, s.num_kv_heads,
            s.head_size, s.block_size, win_str, count,
            f"{fmha_ms:.6f}" if fmha_ms is not None else "",
            f"{t2d:.6f}" if t2d is not None else "",
            f"{t3d:.6f}" if t3d is not None else "",
            best_label, ratio_str,
        ])

    print()
    total_bench = stats["fmha_wins"] + stats["triton_wins"]
    print(f"Summary: {total_bench} shapes benchmarked, {stats['errors']} errors")
    if total_bench > 0:
        print(f"  CK FMHA faster: {stats['fmha_wins']:4d} ({100*stats['fmha_wins']/total_bench:.1f}%)")
        print(f"  Triton faster:  {stats['triton_wins']:4d} ({100*stats['triton_wins']/total_bench:.1f}%)")
    print("ratio: <1.0 = CK FMHA faster, >1.0 = Triton faster")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "phase", "num_seqs", "total_q_tokens", "max_seqlen_q",
                         "max_seqlen_k", "num_q_heads", "num_kv_heads", "head_size",
                         "block_size", "window_size", "trace_count",
                         "ck_fmha_ms", "triton_2d_ms", "triton_3d_ms",
                         "triton_best", "fmha_vs_best"])
            w.writerows(csv_rows)
        print(f"\nCSV written to: {args.out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
