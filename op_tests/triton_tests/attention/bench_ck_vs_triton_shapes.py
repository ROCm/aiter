#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Benchmark and verify CK unified attention vs Triton on specific shapes.

Usage:
    python bench_ck_vs_triton_shapes.py
    python bench_ck_vs_triton_shapes.py --warmup 10 --iters 50
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention import unified_attention as ua_mod


CONFIGS = [
    {"tag": "b9_sk1001_nb59896", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 59896, "sq": 1, "sk": 1001},
    {"tag": "b9_sk1001_nb512", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 512, "sq": 1, "sk": 1001},
    {"tag": "b64_sk1001_nb59896", "batch": 64, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 59896, "sq": 1, "sk": 1001},
    {"tag": "b64_sk1001_nb512", "batch": 64, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 512, "sq": 1, "sk": 1001},
    {"tag": "b9_sk4096_nb59896", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 59896, "sq": 1, "sk": 4096},
    {"tag": "b9_sk4096_nb512", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 512, "sq": 1, "sk": 4096},
    {"tag": "b256_sk2048_nb59896", "batch": 256, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 59896, "sq": 1, "sk": 2048},
    {"tag": "b1_sk8192_nb59896", "batch": 1, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 59896, "sq": 1, "sk": 8192},
    # block_size=32 (now supported with bs32 instances)
    {"tag": "b9_sk1001_blk32", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 32, "num_blocks": 512, "sq": 1, "sk": 1001},
    {"tag": "b64_sk1001_blk32", "batch": 64, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 32, "num_blocks": 512, "sq": 1, "sk": 1001},
    # int32 overflow test: num_blocks=70000 exceeds ~66K threshold for d64/GQA-8/bs64
    {"tag": "overflow_b9_sk1001_nb70k", "batch": 9, "hq": 64, "hk": 8, "head_size": 64,
     "block_size": 64, "num_blocks": 70000, "sq": 1, "sk": 1001},
]


def make_tensors(cfg, device="cuda", dtype=torch.bfloat16):
    b = cfg["batch"]
    hq, hk, d = cfg["hq"], cfg["hk"], cfg["head_size"]
    blk, nb = cfg["block_size"], cfg["num_blocks"]
    sq, sk = cfg["sq"], cfg["sk"]
    total_q = b * sq
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(total_q, hq, d, dtype=dtype, device=device)
    k = torch.randn(nb, blk, hk, d, dtype=dtype, device=device)
    v = torch.randn_like(k)

    cu = torch.arange(0, b + 1, dtype=torch.int32, device=device) * sq
    seq_lens_k = torch.full((b,), sk, dtype=torch.int32, device=device)

    max_blks_per_seq = (sk + blk - 1) // blk
    block_tables = torch.randint(0, nb, (b, max_blks_per_seq), dtype=torch.int32, device=device)

    return q, k, v, cu, seq_lens_k, block_tables, scale


def bench_ck(q, k, v, cu, seq_lens_k, block_tables, scale, warmup, iters):
    out = torch.empty_like(q)
    kw = dict(mask_type=2, scale_s=scale, scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0)
    for _ in range(warmup):
        unified_attention_fwd(out, q, k, v, block_tables, seq_lens_k, cu, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        unified_attention_fwd(out, q, k, v, block_tables, seq_lens_k, cu, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters, out


def bench_triton(q, k, v, cu, seq_lens_k, block_tables, scale, sq, sk, warmup, iters):
    out = torch.empty_like(q)
    kw = dict(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=sq,
              seqused_k=seq_lens_k, max_seqlen_k=sk, softmax_scale=scale,
              causal=True, window_size=(-1, -1), block_table=block_tables,
              softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
              alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)
    for _ in range(warmup):
        ua_mod.unified_attention(**kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ua_mod.unified_attention(**kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters, out


def main():
    parser = argparse.ArgumentParser(description="Benchmark CK vs Triton unified attention on specific shapes.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(42)

    hdr = f"{'tag':40s} {'CK ms':>8s} {'Triton ms':>10s} {'ratio':>7s} {'CK ok':>6s} {'match':>6s} {'max_diff':>10s}"
    print(hdr)
    print("-" * len(hdr))

    for cfg in CONFIGS:
        q, k, v, cu, seq_lens_k, bt, scale = make_tensors(cfg, dtype=dtype)

        # Correctness
        out_ck = torch.zeros_like(q)
        try:
            unified_attention_fwd(out_ck, q, k, v, bt, seq_lens_k, cu,
                                  mask_type=2, scale_s=scale, scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0)
            torch.cuda.synchronize()
            ck_ok = not torch.isnan(out_ck).any().item() and not (out_ck == 0).all().item()
        except Exception as e:
            print(f"  CK error on {cfg['tag']}: {e}")
            ck_ok = False

        out_triton = torch.zeros_like(q)
        ua_mod.unified_attention(q=q, k=k, v=v, out=out_triton, cu_seqlens_q=cu,
                                 max_seqlen_q=cfg["sq"], seqused_k=seq_lens_k, max_seqlen_k=cfg["sk"],
                                 softmax_scale=scale, causal=True, window_size=(-1, -1),
                                 block_table=bt, softcap=0.0,
                                 q_descale=None, k_descale=None, v_descale=None,
                                 alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)
        torch.cuda.synchronize()

        if ck_ok:
            match = torch.allclose(out_ck.float(), out_triton.float(), atol=1e-2, rtol=1e-2)
            max_diff = (out_ck.float() - out_triton.float()).abs().max().item()
        else:
            match = False
            max_diff = -1.0

        # Benchmark
        if ck_ok:
            ck_ms, _ = bench_ck(q, k, v, cu, seq_lens_k, bt, scale, args.warmup, args.iters)
        else:
            ck_ms = None

        triton_ms, _ = bench_triton(q, k, v, cu, seq_lens_k, bt, scale,
                                    cfg["sq"], cfg["sk"], args.warmup, args.iters)

        ck_str = f"{ck_ms:.4f}" if ck_ms else "err"
        ratio_str = f"{ck_ms / triton_ms:.2f}x" if ck_ms else "n/a"
        print(f"{cfg['tag']:40s} {ck_str:>8s} {triton_ms:10.4f} {ratio_str:>7s} "
              f"{str(ck_ok):>6s} {str(match):>6s} {max_diff:10.6f}")

    print()
    print("ratio: <1.0 = CK faster, >1.0 = Triton faster")


if __name__ == "__main__":
    main()
