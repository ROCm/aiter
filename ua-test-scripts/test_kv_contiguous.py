#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Smoke test for the contiguous (THD) K/V path on UnifiedAttention CK.

Strategy: build K/V as a contiguous tensor [N_kv, num_kv_heads, head_dim],
then exercise the same physical memory under two layouts:

  1. Paged with `page_size=P` (default 32) and `block_tables = arange(N_kv/P)`
     (an identity mapping). The kernel walks block_tables for every tile but
     resolves to the same memory location each time — same offsets as the
     contiguous variant.
  2. Contiguous: same physical tensor, `kv_contiguous=True`. The kernel
     skips block_tables and computes offsets directly as `token *
     row_stride`.

The Tier-2 LDS-cache currently holds 4096 page-table entries, so the paged
reference walks N_kv/P pages and N_kv must be divisible by P. P=32 covers
sk up to 128K tokens comfortably.

Both should produce bit-identical (or numerically-identical-up-to-roundoff)
output for the same Q. If they don't, the contiguous path has a bug.

Examples:
  python test_kv_contiguous.py --d 128 --hq 64 --hk 8 --sk 4096 --dtype bf16
  python test_kv_contiguous.py --d 64  --hq 64 --hk 8 --sk 32768 --dtype fp8 --causal
"""

from __future__ import annotations

import argparse
import math
import sys

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sq", type=int, default=4096, help="Query length (default: 4096)")
    p.add_argument("--sk", type=int, default=4096, help="KV length (default: 4096)")
    p.add_argument("--hq", type=int, default=64, help="Query heads (default: 64)")
    p.add_argument("--hk", type=int, default=8, help="KV heads (default: 8)")
    p.add_argument("--d", type=int, default=128, help="Head dim (default: 128)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp8"], default="bf16")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--page-size", type=int, default=32,
                   help="Page size for the paged reference (default: 32). "
                        "sk must be divisible by this. Lower values stress the "
                        "Tier-2 LDS-cache size limit (4096 entries) earlier.")
    p.add_argument("--bench", action="store_true",
                   help="After correctness check, time each path for "
                        "--iters iterations and print the speedup.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters",  type=int, default=20)
    return p.parse_args()


def to_dtype(d: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp8": torch.float8_e4m3fnuz}[d]


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")
    torch.manual_seed(args.seed)

    sq, sk, hq, hk, d = args.sq, args.sk, args.hq, args.hk, args.d
    dtype = to_dtype(args.dtype)

    # ---------------------------------------------------------------
    # Build q, k, v in contiguous-layout shape.
    # ---------------------------------------------------------------
    if args.dtype == "fp8":
        q_bf = torch.randn(sq, hq, d, dtype=torch.bfloat16, device=device) * 0.5
        k_bf = torch.randn(sk, hk, d, dtype=torch.bfloat16, device=device) * 0.5
        v_bf = torch.randn(sk, hk, d, dtype=torch.bfloat16, device=device) * 0.5
        q = q_bf.to(dtype)
        k = k_bf.to(dtype)
        v = v_bf.to(dtype)
    else:
        q = torch.randn(sq, hq, d, dtype=dtype, device=device)
        k = torch.randn(sk, hk, d, dtype=dtype, device=device)
        v = torch.randn(sk, hk, d, dtype=dtype, device=device)

    scale = 1.0 / math.sqrt(d)

    # ---------------------------------------------------------------
    # Paged view: same memory, shape [sk/P, P, hk, d] with page_size = P.
    # block_tables = arange(sk/P) is the identity mapping.
    # ---------------------------------------------------------------
    page_size = args.page_size
    assert sk % page_size == 0, f"sk={sk} must be divisible by page_size={page_size}"
    n_pages = sk // page_size
    k_paged = k.view(n_pages, page_size, hk, d).contiguous()
    v_paged = v.view(n_pages, page_size, hk, d).contiguous()
    block_tables = torch.arange(n_pages, dtype=torch.int32,
                                device=device).view(1, n_pages)

    # ---------------------------------------------------------------
    # Common metadata: one request, length sk.
    # ---------------------------------------------------------------
    seq_lens = torch.tensor([sk], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, sq], dtype=torch.int32, device=device)
    out_dtype = torch.bfloat16 if args.dtype == "fp8" else dtype
    out_paged = torch.zeros(sq, hq, d, dtype=out_dtype, device=device)
    out_contig = torch.zeros_like(out_paged)

    # For contiguous mode the caller can pass any K/V shape whose strides
    # yield a row stride equal to `hk * d` — we reuse k_paged / v_paged
    # here (which are bit-identical to a [sk, hk, d] flat tensor at the
    # memory level) so the comparison is on the exact same bytes.
    from aiter.ops.unified_attention import unified_attention_fwd

    mask_type = 2 if args.causal else 0

    # Paged run (identity block_tables).
    unified_attention_fwd(
        out_paged, q, k_paged, v_paged,
        block_tables, seq_lens, cu_seqlens_q,
        mask_type=mask_type, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        allow_splitkv=False,
    )

    # Contiguous run — same memory, kv_contiguous=True.
    unified_attention_fwd(
        out_contig, q, k_paged, v_paged,
        block_tables, seq_lens, cu_seqlens_q,
        mask_type=mask_type, scale_s=scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        allow_splitkv=False,
        kv_contiguous=True,
    )

    torch.cuda.synchronize()

    # ---------------------------------------------------------------
    # Compare.
    # ---------------------------------------------------------------
    diff = (out_paged.float() - out_contig.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    n_bad = (diff > 1e-3).sum().item()
    n_total = diff.numel()

    print(f"shape: sq={sq} sk={sk} hq={hq} hk={hk} d={d} dtype={args.dtype} "
          f"causal={args.causal}")
    print(f"  max |paged - contig| = {max_diff:.6e}")
    print(f"  mean |paged - contig| = {mean_diff:.6e}")
    print(f"  elements > 1e-3      = {n_bad}/{n_total} "
          f"({100*n_bad/n_total:.2f}%)")

    tol = 5e-2 if args.dtype == "fp8" else 5e-3
    ok = max_diff < tol
    print(f"  {'PASS' if ok else 'FAIL'} (tol={tol})")

    if args.bench:
        def time_path(kv_contiguous: bool, out: torch.Tensor) -> float:
            for _ in range(args.warmup):
                unified_attention_fwd(
                    out, q, k_paged, v_paged,
                    block_tables, seq_lens, cu_seqlens_q,
                    mask_type=mask_type, scale_s=scale,
                    scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
                    allow_splitkv=False, kv_contiguous=kv_contiguous,
                )
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                unified_attention_fwd(
                    out, q, k_paged, v_paged,
                    block_tables, seq_lens, cu_seqlens_q,
                    mask_type=mask_type, scale_s=scale,
                    scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
                    allow_splitkv=False, kv_contiguous=kv_contiguous,
                )
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / args.iters

        t_paged  = time_path(False, out_paged)
        t_contig = time_path(True,  out_contig)
        speedup  = t_paged / t_contig if t_contig > 0 else float("inf")
        # 4 * sq * sk * hq * d for full attention; halve for causal (only
        # the lower-triangular half is computed).
        flops = 4.0 * sq * sk * hq * d * (0.5 if args.causal else 1.0)
        tflops_paged  = flops / (t_paged  * 1e-3) / 1e12
        tflops_contig = flops / (t_contig * 1e-3) / 1e12
        print(f"  paged:  {t_paged:8.4f} ms  {tflops_paged:7.2f} TFLOPs/s")
        print(f"  contig: {t_contig:8.4f} ms  {tflops_contig:7.2f} TFLOPs/s")
        print(f"  contiguous speedup: {speedup:.3f}x "
              f"({100*(t_paged-t_contig)/t_paged:+.1f}%)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
