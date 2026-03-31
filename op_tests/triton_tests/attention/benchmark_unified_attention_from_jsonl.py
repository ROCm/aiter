#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Replay and compare unified-attention dispatch choices from a JSONL trace.

The script benchmarks:
1) default unified dispatch policy
2) forced 2D policy
3) forced 3D policy

on representative traced call signatures.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from aiter.ops.triton.attention import unified_attention as ua_mod


DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.int8": torch.int8,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e4m3fnuz": torch.float8_e4m3fnuz,
}


@dataclass(frozen=True)
class TraceSignature:
    max_seqlen_q: int
    max_seqlen_k: int
    window_size: tuple[int, int]
    q_shape: tuple[int, int, int]
    k_shape: tuple[int, int, int, int]
    block_table_shape: tuple[int, int]
    num_seqs: int
    softmax_scale: float
    softcap: float
    has_sinks: bool
    q_dtype: str
    k_dtype: str


@dataclass
class BenchCase:
    signature: TraceSignature
    count: int
    observed_kind: str


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in trace: {name}")
    return DTYPE_MAP[name]


def read_cases(jsonl_path: Path, max_cases: int) -> list[BenchCase]:
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError("JSONL file is empty.")

    sig_counter: Counter[TraceSignature] = Counter()
    sig_kind: dict[TraceSignature, str] = {}

    for r in rows:
        sig = TraceSignature(
            max_seqlen_q=int(r["max_seqlen_q"]),
            max_seqlen_k=int(r["max_seqlen_k"]),
            window_size=tuple(r["window_size"]),
            q_shape=tuple(r["q_shape"]),
            k_shape=tuple(r["k_shape"]),
            block_table_shape=tuple(r["block_table_shape"]),
            num_seqs=int(r["num_seqs"]),
            softmax_scale=float(r["softmax_scale"]),
            softcap=float(r.get("softcap", 0.0)),
            has_sinks=bool(r.get("has_sinks", False)),
            q_dtype=str(r["q_dtype"]),
            k_dtype=str(r["k_dtype"]),
        )
        sig_counter[sig] += 1
        sig_kind.setdefault(sig, str(r["kind"]))

    ranked = sig_counter.most_common(max_cases)
    return [
        BenchCase(signature=sig, count=count, observed_kind=sig_kind[sig])
        for sig, count in ranked
    ]


def build_inputs(sig: TraceSignature, seed: int = 0) -> dict[str, torch.Tensor | int | float | tuple[int, int] | None]:
    random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    q_dtype = parse_dtype(sig.q_dtype)
    k_dtype = parse_dtype(sig.k_dtype)
    device = "cuda"

    q = torch.randn(*sig.q_shape, dtype=q_dtype, device=device)
    k = torch.randn(*sig.k_shape, dtype=k_dtype, device=device)
    v = torch.randn(*sig.k_shape, dtype=k_dtype, device=device)
    out = torch.empty_like(q)

    num_seqs = sig.num_seqs
    if num_seqs <= 0:
        raise ValueError(f"Invalid num_seqs={num_seqs}")

    # This trace has enough information to reconstruct sequence layout when num_seqs=1.
    # For num_seqs>1, we synthesize an even split to preserve total tokens.
    q_total_tokens = sig.q_shape[0]
    if num_seqs == 1:
        q_lens = [q_total_tokens]
    else:
        base = q_total_tokens // num_seqs
        rem = q_total_tokens % num_seqs
        q_lens = [base + (1 if i < rem else 0) for i in range(num_seqs)]

    cu = [0]
    for ln in q_lens:
        cu.append(cu[-1] + ln)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)

    seqused_k = torch.full((num_seqs,), sig.max_seqlen_k, dtype=torch.int32, device=device)

    block_table_rows, block_table_cols = sig.block_table_shape
    if block_table_rows != num_seqs:
        # Be permissive if trace shape and num_seqs disagree.
        block_table_rows = num_seqs
    block_table = torch.randint(
        low=0,
        high=sig.k_shape[0],
        size=(block_table_rows, block_table_cols),
        dtype=torch.int32,
        device=device,
    )

    sinks = None
    if sig.has_sinks:
        sinks = torch.randn(sig.q_shape[1], dtype=torch.bfloat16, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": sig.max_seqlen_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": sig.max_seqlen_k,
        "softmax_scale": sig.softmax_scale,
        "causal": True,
        "window_size": sig.window_size,
        "block_table": block_table,
        "softcap": sig.softcap,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": sinks,
    }


@contextlib.contextmanager
def force_dispatch(mode: str):
    old = ua_mod.use_2d_kernel
    if mode == "default":
        yield
        return
    if mode == "force_2d":
        ua_mod.use_2d_kernel = lambda *args, **kwargs: True
    elif mode == "force_3d":
        ua_mod.use_2d_kernel = lambda *args, **kwargs: False
    else:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        yield
    finally:
        ua_mod.use_2d_kernel = old


def run_one_case(
    case: BenchCase,
    warmup: int,
    iters: int,
    mode: str,
) -> float:
    inp = build_inputs(case.signature, seed=123)

    with force_dispatch(mode):
        for _ in range(warmup):
            ua_mod.unified_attention(**inp)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            ua_mod.unified_attention(**inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark unified-attention from JSONL trace.")
    parser.add_argument("--jsonl", type=Path, required=True, help="Path to trace JSONL file.")
    parser.add_argument("--max-cases", type=int, default=12, help="Benchmark top-N frequent signatures.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per mode/case.")
    parser.add_argument("--iters", type=int, default=30, help="Timed iterations per mode/case.")
    args = parser.parse_args()

    cases = read_cases(args.jsonl, max_cases=args.max_cases)
    print(f"Loaded {len(cases)} representative cases from {args.jsonl}")
    print("mode columns are average milliseconds per call.\n")
    print(
        "idx count observed max_q max_k window q_shape"
        "                         default    force_2d   force_3d   best"
    )
    print("-" * 126)

    for i, case in enumerate(cases):
        sig = case.signature
        row = {
            "default": float("nan"),
            "force_2d": float("nan"),
            "force_3d": float("nan"),
        }
        for mode in row:
            try:
                row[mode] = run_one_case(case, warmup=args.warmup, iters=args.iters, mode=mode)
            except Exception as e:
                print(f"[case {i}] mode={mode} failed: {e}")

        valid = {k: v for k, v in row.items() if v == v}
        best_mode = min(valid, key=valid.get) if valid else "n/a"
        print(
            f"{i:3d} {case.count:5d} {case.observed_kind:7s} "
            f"{sig.max_seqlen_q:5d} {sig.max_seqlen_k:5d} "
            f"{str(sig.window_size):>12s} "
            f"{str(sig.q_shape):>24s} "
            f"{row['default']:9.3f} {row['force_2d']:10.3f} {row['force_3d']:10.3f} {best_mode:>8s}"
        )

    print("\nNotes:")
    print("- Trace JSONL has enough parameters for shape-level replay benchmarking.")
    print("- It does not contain original tensor values; random tensors are used.")
    print("- For num_seqs>1 traces, per-seq q_lens are synthesized if not explicitly logged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
