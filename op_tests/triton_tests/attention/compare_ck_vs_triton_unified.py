#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Compare CK tile unified_attention vs Triton unified_attention on matching shapes.

Note: the CK example binary currently has these compile-time constraints:
  - HEAD_SIZE = 128
  - num_queries_per_kv = 1 (MHA, not GQA)
  - BLOCK_SIZE tile = 32

So we match Triton parameters to this config for a fair comparison.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from aiter.ops.triton.attention import unified_attention as ua_mod


@dataclass
class BenchCase:
    tag: str
    batch: int
    num_kv_heads: int
    num_queries_per_kv: int
    head_size: int
    page_blk_size: int
    num_blocks: int
    query_lens: list[int]
    kv_lens: list[int]
    causal: bool
    dtype: str  # "bf16" or "fp16"


DEFAULT_CASES = [
    # --- Prefill cases (CK's sweet spot: large Q, high occupancy) ---
    # num_blocks sized to just fit max_k with some headroom.
    BenchCase("prefill_b1_q256_k256", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[256], kv_lens=[256], causal=True, dtype="bf16"),
    BenchCase("prefill_b1_q512_k512", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[512], kv_lens=[512], causal=True, dtype="bf16"),
    BenchCase("prefill_b1_q1024_k1024", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1024], kv_lens=[1024], causal=True, dtype="bf16"),
    BenchCase("prefill_b1_q2048_k2048", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[2048], kv_lens=[2048], causal=True, dtype="bf16"),
    BenchCase("prefill_b1_q4096_k4096", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[4096], kv_lens=[4096], causal=True, dtype="bf16"),
    BenchCase("prefill_b3_mixed", batch=3, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[512, 1024, 256], kv_lens=[512, 1024, 256], causal=True, dtype="bf16"),
    # --- Decode cases (CK instance is over-provisioned here) ---
    BenchCase("decode_b1_k1001", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1], kv_lens=[1001], causal=True, dtype="bf16"),
    BenchCase("decode_b1_k4096", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1], kv_lens=[4096], causal=True, dtype="bf16"),
    BenchCase("decode_b8_k2048", batch=8, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1]*8, kv_lens=[2048]*8, causal=True, dtype="bf16"),
]


def run_ck_binary(
    binary: Path,
    case: BenchCase,
    warmup: int,
    repeat: int,
) -> float | None:
    num_q_heads = case.num_kv_heads * case.num_queries_per_kv
    max_q = max(case.query_lens)
    max_k = max(case.kv_lens)

    cmd = [
        str(binary),
        f"-prec={case.dtype}",
        f"-h_k={case.num_kv_heads}",
        f"-nqpkv={case.num_queries_per_kv}",
        f"-b={case.batch}",
        f"-d={case.head_size}",
        f"-nb={case.num_blocks}",
        f"-s={max_q}",
        f"-s_k={max_k}",
        f"-page_blk_size={case.page_blk_size}",
        f"-causal={'1' if case.causal else '0'}",
        "-varlen=0",
        "-verify=0",
        f"-warmup={warmup}",
        f"-repeat={repeat}",
        f"-query_lens={','.join(str(x) for x in case.query_lens)}",
        f"-kv_lens={','.join(str(x) for x in case.kv_lens)}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        # Parse "X.XXXXXXXX ms" from CK output
        match = re.search(r"(\d+\.\d+)\s*ms", output)
        if match:
            return float(match.group(1))
        print(f"  CK output (no ms found): {output.strip()[:200]}")
        return None
    except Exception as e:
        print(f"  CK binary error: {e}")
        return None


def run_triton_unified(
    case: BenchCase,
    warmup: int,
    iters: int,
    force_2d: bool | None = None,
) -> float:
    import math

    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    num_q_heads = case.num_kv_heads * case.num_queries_per_kv

    total_q_tokens = sum(case.query_lens)
    q = torch.randn(total_q_tokens, num_q_heads, case.head_size, dtype=dtype, device="cuda")
    k = torch.randn(
        case.num_blocks, case.page_blk_size, case.num_kv_heads, case.head_size,
        dtype=dtype, device="cuda",
    )
    v = torch.randn_like(k)
    out = torch.empty_like(q)

    cu = [0]
    for x in case.query_lens:
        cu.append(cu[-1] + x)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")

    seq_lens = torch.tensor(case.kv_lens, dtype=torch.int32, device="cuda")

    max_blocks_per_seq = math.ceil(max(case.kv_lens) / case.page_blk_size)
    block_tables = torch.randint(
        0, case.num_blocks, (case.batch, max_blocks_per_seq),
        dtype=torch.int32, device="cuda",
    )

    scale = 1.0 / (case.head_size ** 0.5)

    kw = dict(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(case.query_lens),
        seqused_k=seq_lens,
        max_seqlen_k=max(case.kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None,
        sinks=None,
    )

    old = ua_mod.use_2d_kernel
    if force_2d is not None:
        ua_mod.use_2d_kernel = (lambda *a, **kw: True) if force_2d else (lambda *a, **kw: False)

    try:
        for _ in range(warmup):
            ua_mod.unified_attention(**kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ua_mod.unified_attention(**kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3 / iters
    finally:
        ua_mod.use_2d_kernel = old


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare CK tile vs Triton unified attention.")
    ap.add_argument(
        "--ck-binary",
        type=Path,
        default=Path("/workspaces/workspace/composable_kernel/build/bin/tile_example_unified_attention"),
        help="Path to CK unified attention example binary.",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()

    if not args.ck_binary.exists():
        print(f"CK binary not found: {args.ck_binary}")
        return 2

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    print(f"CK binary: {args.ck_binary}")
    print(f"warmup={args.warmup} iters={args.iters}")
    print(f"CK constraints: HEAD_SIZE=128, num_queries_per_kv=1, causal mask")
    print()
    print(
        f"{'case':30s} {'CK ms':>10s} {'Triton def':>11s} "
        f"{'Triton 2D':>10s} {'Triton 3D':>10s} {'CK vs best':>11s}"
    )
    print("-" * 95)

    for case in DEFAULT_CASES:
        ck_ms = run_ck_binary(args.ck_binary, case, warmup=args.warmup, repeat=args.iters)
        triton_def = run_triton_unified(case, warmup=args.warmup, iters=args.iters, force_2d=None)
        triton_2d = run_triton_unified(case, warmup=args.warmup, iters=args.iters, force_2d=True)
        triton_3d = run_triton_unified(case, warmup=args.warmup, iters=args.iters, force_2d=False)

        best_triton = min(triton_def, triton_2d, triton_3d)
        ratio_str = ""
        if ck_ms is not None and best_triton > 0:
            ratio = ck_ms / best_triton
            ratio_str = f"{ratio:.3f}x"

        ck_str = f"{ck_ms:.4f}" if ck_ms is not None else "n/a"
        print(
            f"{case.tag:30s} {ck_str:>10s} {triton_def:10.4f} "
            f"{triton_2d:10.4f} {triton_3d:10.4f} {ratio_str:>11s}"
        )

    print()
    print("CK vs best: <1.0 means CK is faster, >1.0 means Triton is faster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
