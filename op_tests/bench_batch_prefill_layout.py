# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark CK mha_batch_prefill across KV-cache layouts (dense, apples-to-apples).

Compares the BF16 prefill kernel on:
  - "linear"      : 3D linear KV (page_size=1) -- the layout the external
                    reference bench uses, so these numbers are directly comparable.
  - "vec_k_col_v" : decode-aligned layout (5D vectorized K + 4D ColumnMajor V
                    [NumBlocks, NumHeads, HeadDim, PageSize]); requires page_size
                    in (64, 1024). This is the layout that lets vLLM skip the
                    prefill<->decode KV-cache reshape.

Semantics mirror the external reference bench for an apples-to-apples comparison:
dense full-length sequences (NO randomized lengths), GPU-event timing,
warmup + timed iters, trimmed mean, and causal/full FLOPs. Defaults reproduce
the reference config: batch=1, num_q_heads=64, num_kv_heads=8, head_dim=128, bf16,
causal=True, seq_lens 1k/16k/32k/64k/128k.

Examples:
    python op_tests/bench_batch_prefill_layout.py
    python op_tests/bench_batch_prefill_layout.py --layout linear --seq-lens 1k 16k 128k
    python op_tests/bench_batch_prefill_layout.py --layout vec_k_col_v --page-size 64
    python op_tests/bench_batch_prefill_layout.py --batch-size 1 --csv out.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
from op_tests.test_batch_prefill import (
    apply_kv_layout,
    build_paged_kv_cache,
    convert_lens_to_indptr,
    extract_kv_caches,
    get_vector_size,
)

DEFAULT_SEQ_LENS = [1024, 16384, 32768, 65536, 131072]
# page_size used per layout when --page-size is left at the auto sentinel (0).
AUTO_PAGE_SIZE = {"linear": 1, "vec_k_col_v": 64}
VEC_K_COL_V_SUPPORTED_PAGE = (64, 1024)


def parse_dtype(dtype: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[dtype]


def parse_seq_len(token: str) -> int:
    token = token.strip().lower()
    if token.endswith("k"):
        return int(float(token[:-1]) * 1024)
    if token.endswith("m"):
        return int(float(token[:-1]) * 1024 * 1024)
    return int(token)


def format_seq_len(seq_len: int) -> str:
    return f"{seq_len // 1024}k" if seq_len % 1024 == 0 else str(seq_len)


def trimmed_mean(values: List[float], trim_ratio: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1 or trim_ratio <= 0:
        return statistics.mean(values)
    sorted_values = sorted(values)
    trim = int(len(sorted_values) * trim_ratio)
    if trim * 2 >= len(sorted_values):
        return statistics.mean(sorted_values)
    return statistics.mean(sorted_values[trim : len(sorted_values) - trim])


def causal_flops_per_seq(q_len, k_len, num_q_heads, head_dim) -> float:
    if q_len > k_len:
        valid_elements = (k_len * k_len + k_len) / 2
    else:
        valid_elements = q_len * k_len - ((q_len * q_len - q_len) / 2)
    return valid_elements * num_q_heads * head_dim * 4.0


def full_flops_per_seq(q_len, k_len, num_q_heads, head_dim) -> float:
    return q_len * k_len * num_q_heads * head_dim * 4.0


def compute_total_flops(q_lens, k_lens, num_q_heads, head_dim, causal) -> float:
    fn = causal_flops_per_seq if causal else full_flops_per_seq
    return sum(fn(q, k, num_q_heads, head_dim) for q, k in zip(q_lens, k_lens))


def summarize_times_ms(times_ms: List[float], trim_ratio: float) -> dict:
    return {
        "avg_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "trimmed_mean_ms": trimmed_mean(times_ms, trim_ratio),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def benchmark_once(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def resolve_page_size(layout: str, page_size: int) -> int:
    if page_size == 0:
        return AUTO_PAGE_SIZE[layout]
    return page_size


def make_inputs(args, seq_len: int, layout: str, page_size: int, dtype):
    """Build DENSE (full-length) inputs for the requested layout.

    Returns a zero-arg `fn` that invokes aiter.mha_batch_prefill_func and the
    per-sequence q/k lengths used for the FLOPs accounting.
    """
    device = args.device
    batch = args.batch_size
    nhq, nhk, hd = args.num_q_heads, args.num_kv_heads, args.head_dim
    # Dense: every sequence is exactly seq_len (no randomized lengths).
    q_lens = [seq_len] * batch
    kv_lens_t = torch.full((batch,), seq_len, dtype=torch.int32)
    qo_lens_t = torch.full((batch,), seq_len, dtype=torch.int32)
    total_q = seq_len * batch
    softmax_scale = args.softmax_scale or (hd ** -0.5)

    q = torch.randn(total_q, nhq, hd, device=device, dtype=dtype)
    cu_seqlens_q = convert_lens_to_indptr(qo_lens_t).to(device)

    if layout == "linear":
        # 3D linear page_size=1 KV (matches the external reference bench exactly).
        assert page_size == 1, "linear bench uses page_size=1 (3D KV)"
        total_kv = seq_len * batch
        k = torch.randn(total_kv, nhk, hd, device=device, dtype=dtype)
        v = torch.randn(total_kv, nhk, hd, device=device, dtype=dtype)
        kv_indptr = convert_lens_to_indptr(kv_lens_t).to(device)
        kv_page_indices = torch.arange(total_kv, dtype=torch.int32, device=device)

        def fn():
            return aiter.mha_batch_prefill_func(
                q, k, v, cu_seqlens_q, kv_indptr, kv_page_indices,
                seq_len, seq_len,
                softmax_scale=softmax_scale,
                logits_soft_cap=args.logits_soft_cap,
                causal=args.causal,
                return_lse=args.return_lse,
            )

    elif layout == "vec_k_col_v":
        if page_size not in VEC_K_COL_V_SUPPORTED_PAGE:
            raise ValueError(
                f"vec_k_col_v supports page_size in {VEC_K_COL_V_SUPPORTED_PAGE}, "
                f"got {page_size}"
            )
        k_vector_size = get_vector_size(dtype)
        kv_cache = build_paged_kv_cache(
            batch, seq_len, page_size, nhk, hd, kv_lens_t, -5, 5, dtype,
            use_uniform=False, contiguous_kv=True,
        )
        k_ref, v_ref = extract_kv_caches(kv_cache, True)
        k_cache, v_cache = apply_kv_layout(
            k_ref, v_ref, nhk, hd, page_size, k_vector_size, layout
        )
        kv_indptr = kv_cache["kv_indptr_cpu"].to(device)
        kv_page_indices = kv_cache["kv_indices_cpu"].to(device)
        kv_last_page_lens = kv_cache["kv_last_page_len_cpu"].to(device)

        def fn():
            return aiter.mha_batch_prefill_func(
                q, k_cache, v_cache, cu_seqlens_q, kv_indptr, kv_page_indices,
                seq_len, seq_len,
                softmax_scale=softmax_scale,
                logits_soft_cap=args.logits_soft_cap,
                causal=args.causal,
                return_lse=args.return_lse,
                kv_last_page_lens=kv_last_page_lens,
            )

    else:
        raise ValueError(f"unknown layout: {layout}")

    return fn, q_lens, [seq_len] * batch


def run_case(args, seq_len: int, layout: str, page_size: int) -> dict:
    dtype = parse_dtype(args.dtype)
    fn, q_lens, k_lens = make_inputs(args, seq_len, layout, page_size, dtype)

    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()

    times_ms = [benchmark_once(fn) for _ in range(args.iters)]

    stats = summarize_times_ms(times_ms, args.trim_ratio)
    total_flops = compute_total_flops(
        q_lens, k_lens, args.num_q_heads, args.head_dim, args.causal
    )
    stats["tflops_avg"] = total_flops / (stats["avg_ms"] * 1.0e9)
    stats["tflops_med"] = total_flops / (stats["median_ms"] * 1.0e9)
    stats["seq_len"] = seq_len
    stats["seq_len_label"] = format_seq_len(seq_len)
    stats["layout"] = layout
    stats["page_size"] = page_size
    stats["samples"] = len(times_ms)

    torch.cuda.empty_cache()
    return stats


def print_results(rows: List[dict]) -> None:
    headers = [
        "layout", "page", "seq_len", "avg(ms)", "median(ms)", "min(ms)",
        "max(ms)", "trimmed(ms)", "std(ms)", "TFLOPS(avg)", "TFLOPS(med)",
    ]
    table = []
    for r in rows:
        table.append([
            r["layout"], str(r["page_size"]), r["seq_len_label"],
            f"{r['avg_ms']:.3f}", f"{r['median_ms']:.3f}", f"{r['min_ms']:.3f}",
            f"{r['max_ms']:.3f}", f"{r['trimmed_mean_ms']:.3f}", f"{r['std_ms']:.3f}",
            f"{r['tflops_avg']:.3f}", f"{r['tflops_med']:.3f}",
        ])
    widths = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        return "  ".join(c.rjust(widths[i]) for i, c in enumerate(cells))

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(headers) - 1)))
    for row in table:
        print(fmt(row))


def print_comparison(rows: List[dict]) -> None:
    """Print median(ms) ratio of vec_k_col_v vs linear at matched seq_len."""
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["seq_len"], {})[r["layout"]] = r
    if not all("linear" in v and "vec_k_col_v" in v for v in by_seq.values()):
        return
    print("\nvec_k_col_v vs linear (median ms; <1.0 = vec_k_col_v faster):")
    print(f"{'seq_len':>8}  {'linear':>12}  {'vec_k_col_v':>12}  {'ratio':>8}")
    for seq_len in sorted(by_seq):
        lin = by_seq[seq_len]["linear"]["median_ms"]
        vkc = by_seq[seq_len]["vec_k_col_v"]["median_ms"]
        print(
            f"{format_seq_len(seq_len):>8}  {lin:>12.3f}  {vkc:>12.3f}  "
            f"{vkc / lin:>7.3f}x"
        )


def write_csv(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "layout", "page_size", "seq_len", "avg_ms", "median_ms", "min_ms",
        "max_ms", "trimmed_mean_ms", "std_ms", "tflops_avg", "tflops_med", "samples",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([
                r["layout"], r["page_size"], r["seq_len_label"], r["avg_ms"],
                r["median_ms"], r["min_ms"], r["max_ms"], r["trimmed_mean_ms"],
                r["std_ms"], r["tflops_avg"], r["tflops_med"], r["samples"],
            ])


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Benchmark aiter.mha_batch_prefill_func across KV layouts (dense).",
    )
    p.add_argument(
        "--seq-lens", nargs="+",
        default=[format_seq_len(x) for x in DEFAULT_SEQ_LENS],
        help="seq_len list, accepts 1k/16k/32k/64k/128k style tokens",
    )
    p.add_argument(
        "--layout", choices=["linear", "vec_k_col_v", "both"], default="both",
        help="KV layout(s) to bench",
    )
    p.add_argument(
        "--page-size", type=int, default=0,
        help="page size; 0=auto (linear->1, vec_k_col_v->64)",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-q-heads", type=int, default=64)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--logits-soft-cap", type=float, default=0.0)
    p.add_argument("--softmax-scale", type=float, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--trim-ratio", type=float, default=0.1)
    p.add_argument("--return-lse", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--csv", type=str, default="")
    args = p.parse_args()
    args.seq_lens = [parse_seq_len(x) for x in args.seq_lens]
    return args


def print_config(args, layouts):
    print("=" * 100)
    print("mha_batch_prefill layout benchmark (dense, apples-to-apples vs reference)")
    print("=" * 100)
    print(
        f"config: batch_size={args.batch_size}, num_q_heads={args.num_q_heads}, "
        f"num_kv_heads={args.num_kv_heads}, head_dim={args.head_dim}, "
        f"dtype={args.dtype}, causal={args.causal}, logits_soft_cap={args.logits_soft_cap}, "
        f"warmup={args.warmup}, iters={args.iters}, device={args.device}"
    )
    pages = {lay: resolve_page_size(lay, args.page_size) for lay in layouts}
    print(f"layouts: {[f'{lay}(page={pages[lay]})' for lay in layouts]}")
    print(f"seq_lens: {[format_seq_len(x) for x in args.seq_lens]}")
    print("note: sequences are DENSE (full seq_len, no randomized lengths).\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU device available")
    if args.batch_size <= 0 or args.num_q_heads <= 0 or args.num_kv_heads <= 0:
        raise ValueError("batch_size / head counts must be > 0")
    if not (0.0 <= args.trim_ratio < 0.5):
        raise ValueError("trim_ratio must satisfy 0 <= trim_ratio < 0.5")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup >= 0 and iters > 0 required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)

    layouts = ["linear", "vec_k_col_v"] if args.layout == "both" else [args.layout]
    print_config(args, layouts)

    rows = []
    total_start = time.time()
    for layout in layouts:
        page_size = resolve_page_size(layout, args.page_size)
        for seq_len in args.seq_lens:
            print(f"[run] layout={layout} page={page_size} seq_len={format_seq_len(seq_len)}", flush=True)
            rows.append(run_case(args, seq_len, layout, page_size))
    total_end = time.time()

    print()
    print_results(rows)
    print_comparison(rows)
    print(f"\ntotal_elapsed_s: {total_end - total_start:.3f}")

    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"csv saved to: {args.csv}")


if __name__ == "__main__":
    main()
