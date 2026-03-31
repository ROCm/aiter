#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Compare CK tile unified_attention (via AITER JIT wrapper) vs Triton unified_attention
on matching shapes.  Both backends receive identical paged-KV tensors.

CK compile-time constraints (from the shipped instances):
  - HEAD_SIZE = 128
  - kPageBlockSize = 32   -> page_blk_size must be >= 32 and a multiple of 32
  - num_queries_per_kv = 1 (MHA only)
  - mask: no-mask (0) or causal (2)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field

import torch

from aiter.ops.unified_attention import unified_attention_fwd
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
    # --- Prefill ---
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
    # --- Decode ---
    BenchCase("decode_b1_k1001", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1], kv_lens=[1001], causal=True, dtype="bf16"),
    BenchCase("decode_b1_k4096", batch=1, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1], kv_lens=[4096], causal=True, dtype="bf16"),
    BenchCase("decode_b8_k2048", batch=8, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[1]*8, kv_lens=[2048]*8, causal=True, dtype="bf16"),
    BenchCase("decode_b32_k2048", batch=32, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=2048,
              query_lens=[1]*32, kv_lens=[2048]*32, causal=True, dtype="bf16"),
    # --- Mixed prefill + decode (chunked prefill) ---
    BenchCase("mixed_b4_pf+dec", batch=4, num_kv_heads=8, num_queries_per_kv=1,
              head_size=128, page_blk_size=128, num_blocks=1024,
              query_lens=[256, 1, 512, 1], kv_lens=[256, 2048, 512, 4096],
              causal=True, dtype="bf16"),
]


def build_tensors(case: BenchCase, device: str = "cuda"):
    """Build identical tensors consumed by both CK and Triton."""
    dtype = torch.bfloat16 if case.dtype == "bf16" else torch.float16
    num_q_heads = case.num_kv_heads * case.num_queries_per_kv

    total_q_tokens = sum(case.query_lens)
    q = torch.randn(total_q_tokens, num_q_heads, case.head_size, dtype=dtype, device=device)
    k = torch.randn(case.num_blocks, case.page_blk_size, case.num_kv_heads, case.head_size,
                     dtype=dtype, device=device)
    v = torch.randn_like(k)

    cu = [0]
    for ql in case.query_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)

    seq_lens_k = torch.tensor(case.kv_lens, dtype=torch.int32, device=device)

    max_blocks_per_seq = max(
        math.ceil(kl / case.page_blk_size) for kl in case.kv_lens
    )
    block_tables = torch.randint(
        0, case.num_blocks, (case.batch, max_blocks_per_seq),
        dtype=torch.int32, device=device,
    )

    scale = 1.0 / math.sqrt(case.head_size)
    return q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale, dtype


def verify_accuracy(case: BenchCase, atol: float = 1e-2, rtol: float = 1e-2) -> tuple[bool, float, float]:
    """Run CK and Triton on identical inputs, compare outputs.
    Returns (passed, max_abs_diff, max_rel_diff).
    """
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale, dtype = build_tensors(case)

    out_ck = torch.zeros_like(q)
    mask_type = 2 if case.causal else 0
    unified_attention_fwd(
        out_ck, q, k, v,
        block_tables, seq_lens_k, cu_seqlens_q,
        mask_type=mask_type,
        scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )
    torch.cuda.synchronize()

    out_triton = torch.zeros_like(q)
    kw = dict(
        q=q, k=k, v=v, out=out_triton,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(case.query_lens),
        seqused_k=seq_lens_k,
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
    ua_mod.unified_attention(**kw)
    torch.cuda.synchronize()

    diff = (out_ck.float() - out_triton.float()).abs()
    max_abs = diff.max().item()
    denom = out_triton.float().abs().clamp(min=1e-6)
    max_rel = (diff / denom).max().item()

    ck_nonzero = out_ck.abs().sum().item()
    triton_nonzero = out_triton.abs().sum().item()

    passed = torch.allclose(out_ck.float(), out_triton.float(), atol=atol, rtol=rtol)
    if ck_nonzero == 0.0:
        print(f"  WARNING: CK output is ALL ZEROS -- kernel likely did not run!")
        passed = False
    if triton_nonzero == 0.0:
        print(f"  WARNING: Triton output is ALL ZEROS")
        passed = False

    return passed, max_abs, max_rel


def run_ck_aiter(
    case: BenchCase,
    warmup: int,
    iters: int,
) -> float:
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale, dtype = build_tensors(case)
    out = torch.empty_like(q)
    mask_type = 2 if case.causal else 0

    for _ in range(warmup):
        unified_attention_fwd(
            out, q, k, v,
            block_tables, seq_lens_k, cu_seqlens_q,
            mask_type=mask_type,
            scale_s=scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        unified_attention_fwd(
            out, q, k, v,
            block_tables, seq_lens_k, cu_seqlens_q,
            mask_type=mask_type,
            scale_s=scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def run_triton(
    case: BenchCase,
    warmup: int,
    iters: int,
    force_2d: bool | None = None,
) -> float:
    q, k, v, cu_seqlens_q, seq_lens_k, block_tables, scale, dtype = build_tensors(case)
    out = torch.empty_like(q)

    kw = dict(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(case.query_lens),
        seqused_k=seq_lens_k,
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

    saved = ua_mod.use_2d_kernel
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
        ua_mod.use_2d_kernel = saved


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare CK tile (AITER JIT) vs Triton unified attention."
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip accuracy verification (default: verify first)")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only run accuracy verification, skip benchmarking")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Run only cases whose tag contains any of these substrings")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    cases = DEFAULT_CASES
    if args.cases:
        cases = [c for c in DEFAULT_CASES
                 if any(sub in c.tag for sub in args.cases)]
        if not cases:
            print(f"No cases matched {args.cases}")
            return 1

    do_verify = not args.no_verify
    do_bench = not args.verify_only

    if do_verify:
        print("=" * 70)
        print("ACCURACY VERIFICATION (CK AITER vs Triton, same inputs)")
        print("=" * 70)
        all_pass = True
        for case in cases:
            try:
                passed, max_abs, max_rel = verify_accuracy(case)
                status = "PASS" if passed else "FAIL"
                if not passed:
                    all_pass = False
                print(f"  {case.tag:30s}  {status}  max_abs={max_abs:.6f}  max_rel={max_rel:.6f}")
            except Exception as e:
                all_pass = False
                print(f"  {case.tag:30s}  ERROR: {e}")
        print()
        if not all_pass:
            print("*** ACCURACY CHECK FAILED -- benchmark numbers may be meaningless ***")
            if not do_bench:
                return 1
        else:
            print("All accuracy checks passed.")
        print()

    if do_bench:
        print(f"warmup={args.warmup}  iters={args.iters}")
        print(f"CK constraints: HEAD_SIZE=128, page_blk_size>=32, num_queries_per_kv=1")
        print()
        hdr = (f"{'case':30s} {'CK AITER':>10s} {'Triton 2D':>10s} "
               f"{'Triton 3D':>10s} {'Triton auto':>11s} {'CK/best':>8s}")
        print(hdr)
        print("-" * len(hdr))

        for case in cases:
            try:
                ck_ms = run_ck_aiter(case, warmup=args.warmup, iters=args.iters)
            except Exception as e:
                print(f"{case.tag:30s}  CK error: {e}")
                ck_ms = None

            triton_2d = run_triton(case, warmup=args.warmup, iters=args.iters, force_2d=True)
            triton_3d = run_triton(case, warmup=args.warmup, iters=args.iters, force_2d=False)
            triton_auto = run_triton(case, warmup=args.warmup, iters=args.iters, force_2d=None)

            best_triton = min(triton_2d, triton_3d, triton_auto)
            if ck_ms is not None and best_triton > 0:
                ratio = ck_ms / best_triton
                ratio_str = f"{ratio:.3f}x"
            else:
                ratio_str = "n/a"

            ck_str = f"{ck_ms:.4f}" if ck_ms is not None else "err"
            print(
                f"{case.tag:30s} {ck_str:>10s} {triton_2d:10.4f} "
                f"{triton_3d:10.4f} {triton_auto:11.4f} {ratio_str:>8s}"
            )

        print()
        print("Times in ms.  CK/best: <1.0 = CK faster, >1.0 = Triton faster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
