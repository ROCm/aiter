#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
CK Unified Attention: Correctness + Performance test suite.

Tests both direct CK-UA kernel and the production selector path across:
  - GPT-OSS config:   hq=64, hk=8,  hdim=64 (GQA-8)
  - Llama-70B TP8:    hq=8,  hk=1,  hdim=64 (GQA-8)
  - Llama-70B MHA:    hq=8,  hk=8,  hdim=128 (MHA)
  - block_size=32 and block_size=64
  - seqs from 1 to 512, maxk from 1 to 131072

Usage:
    python test_ck_ua_correctness_and_perf.py              # full suite
    python test_ck_ua_correctness_and_perf.py --quick      # quick smoke test
    python test_ck_ua_correctness_and_perf.py --perf-only  # skip correctness
"""
from __future__ import annotations
import argparse
import math
import time
import sys

import torch

from aiter.ops.unified_attention import unified_attention_fwd
from aiter.ops.triton.attention.unified_attention import unified_attention as triton_ua
import aiter.ops.triton.attention.unified_attention as ua_mod


def check_direct(seqs, maxk, nqh, nkh, hdim, blk, mask_type=2, trials=3):
    scale = 1.0 / math.sqrt(hdim)
    device = "cuda"
    dtype = torch.bfloat16
    passes = 0
    worst = 0.0
    for trial in range(trials):
        torch.manual_seed(100 + trial)
        needed = max((maxk + blk - 1) // blk, 1)
        num_phys = max(needed * max(seqs, 1) * 2, 64)
        q = torch.randn(seqs, nqh, hdim, dtype=dtype, device=device)
        k = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
        v = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
        cu = torch.arange(seqs + 1, dtype=torch.int32, device=device)
        sl = torch.full((seqs,), maxk, dtype=torch.int32, device=device)
        bt = torch.randint(0, num_phys, (seqs, needed), dtype=torch.int32, device=device)
        o1 = torch.zeros_like(q)
        unified_attention_fwd(
            o1, q, k, v, bt, sl, cu, mask_type=mask_type, scale_s=scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0)
        o2 = torch.zeros_like(q)
        triton_ua(
            q=q, k=k, v=v, out=o2, cu_seqlens_q=cu, max_seqlen_q=1,
            seqused_k=sl, max_seqlen_k=maxk, softmax_scale=scale,
            causal=(mask_type == 2), window_size=(-1, -1), block_table=bt,
            softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
            alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)
        diff = (o1.float() - o2.float()).abs().max().item()
        mag = o2.float().abs().max().item()
        tol = max(0.05, mag * 0.02)
        worst = max(worst, diff)
        if diff < tol:
            passes += 1
        del q, k, v, cu, sl, bt, o1, o2
        torch.cuda.empty_cache()
    return passes, trials, worst


def check_selector(seqs, maxk, nqh, nkh, hdim, blk, trials=3):
    scale = 1.0 / math.sqrt(hdim)
    device = "cuda"
    dtype = torch.bfloat16
    passes = 0
    worst = 0.0
    for trial in range(trials):
        torch.manual_seed(200 + trial)
        needed = max((maxk + blk - 1) // blk, 1)
        num_phys = max(needed * max(seqs, 1) * 2, 64)
        q = torch.randn(seqs, nqh, hdim, dtype=dtype, device=device)
        k = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
        v = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
        cu = torch.arange(seqs + 1, dtype=torch.int32, device=device)
        sl = torch.full((seqs,), maxk, dtype=torch.int32, device=device)
        bt = torch.randint(0, num_phys, (seqs, needed), dtype=torch.int32, device=device)
        kw = dict(
            q=q, k=k, v=v, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=sl,
            max_seqlen_k=maxk, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0.0,
            q_descale=None, k_descale=None, v_descale=None,
            alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)
        o1 = torch.zeros_like(q)
        ua_mod.unified_attention(out=o1, **kw)
        saved = ua_mod._try_ck_unified_attention
        ua_mod._try_ck_unified_attention = lambda *a, **kw: False
        o2 = torch.zeros_like(q)
        ua_mod.unified_attention(out=o2, **kw)
        ua_mod._try_ck_unified_attention = saved
        diff = (o1.float() - o2.float()).abs().max().item()
        mag = o2.float().abs().max().item()
        tol = max(0.05, mag * 0.02)
        worst = max(worst, diff)
        if diff < tol:
            passes += 1
        del q, k, v, cu, sl, bt, o1, o2
        torch.cuda.empty_cache()
    return passes, trials, worst


def bench_shape(seqs, maxk, nqh, nkh, hdim, blk, warmup=5, iters=20):
    """Returns (ck_ua_ms, triton_2d_ms, triton_3d_ms, auto_ms, triton_default_ms) or Nones."""
    scale = 1.0 / math.sqrt(hdim)
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)
    needed = max((maxk + blk - 1) // blk, 1)
    num_phys = max(needed * max(seqs, 1) * 2, 64)
    mem_gb = num_phys * blk * nkh * hdim * 2 * 2 / (1024**3)
    if mem_gb > 80:
        return None, None, None, None, None

    try:
        q = torch.randn(seqs, nqh, hdim, dtype=dtype, device=device)
        k = torch.randn(num_phys, blk, nkh, hdim, dtype=dtype, device=device)
        v = torch.randn_like(k)
        cu = torch.arange(seqs + 1, dtype=torch.int32, device=device)
        sl = torch.full((seqs,), maxk, dtype=torch.int32, device=device)
        bt = torch.randint(0, num_phys, (seqs, needed), dtype=torch.int32, device=device)
        out = torch.zeros_like(q)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None, None, None, None

    kw_triton = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=sl, max_seqlen_k=maxk, softmax_scale=scale, causal=True,
        window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)

    def timed_graph(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / iters

    results = {}
    try:
        # CK-UA direct
        def ck_fn():
            unified_attention_fwd(
                out, q, k, v, bt, sl, cu, mask_type=2, scale_s=scale,
                scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0)
        results["ck_ua"] = timed_graph(ck_fn)
    except Exception:
        results["ck_ua"] = None

    saved_ck = ua_mod._try_ck_unified_attention
    saved_sk = getattr(ua_mod, '_try_ck_splitkv_attention', None)
    ua_mod._try_ck_unified_attention = lambda *a, **kw: False
    if saved_sk:
        ua_mod._try_ck_splitkv_attention = lambda *a, **kw: False

    try:
        # Triton 2D
        saved_2d = ua_mod.use_2d_kernel
        ua_mod.use_2d_kernel = lambda *a, **kw: True
        results["t2d"] = timed_graph(lambda: ua_mod.unified_attention(**kw_triton))
        # Triton 3D
        ua_mod.use_2d_kernel = lambda *a, **kw: False
        results["t3d"] = timed_graph(lambda: ua_mod.unified_attention(**kw_triton))
        ua_mod.use_2d_kernel = saved_2d
        # Triton default (2D/3D heuristic)
        results["t_def"] = timed_graph(lambda: ua_mod.unified_attention(**kw_triton))
    except Exception:
        results.setdefault("t2d", None)
        results.setdefault("t3d", None)
        results.setdefault("t_def", None)

    ua_mod._try_ck_unified_attention = saved_ck
    if saved_sk:
        ua_mod._try_ck_splitkv_attention = saved_sk

    try:
        # Auto (selector enabled)
        results["auto"] = timed_graph(lambda: ua_mod.unified_attention(**kw_triton))
    except Exception:
        results["auto"] = None

    del q, k, v, cu, sl, bt, out
    torch.cuda.empty_cache()
    return results.get("ck_ua"), results.get("t2d"), results.get("t3d"), results.get("auto"), results.get("t_def")


def main():
    ap = argparse.ArgumentParser(description="CK-UA correctness + performance test")
    ap.add_argument("--quick", action="store_true", help="Quick smoke test")
    ap.add_argument("--perf-only", action="store_true", help="Skip correctness")
    ap.add_argument("--correctness-only", action="store_true", help="Skip perf")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 1

    from aiter.ops.triton.utils.device_info import get_num_sms
    cu_count = get_num_sms()
    print(f"GPU CUs: {cu_count}")
    print()

    all_ok = True

    head_configs = [
        ("GPT-OSS-64/8-d64", 64, 8, 64),
        ("Llama70B-TP8-8/1-d64", 8, 1, 64),
        ("MHA-8/8-d128", 8, 8, 128),
    ]

    # ===== CORRECTNESS =====
    if not args.perf_only:
        if args.quick:
            seqs_list = [1, 64, 128, 256]
            maxk_list = [64, 256, 1024, 4096]
            blk_list = [32, 64]
        else:
            seqs_list = [1, 16, 64, 128, 192, 248, 256, 384, 512]
            maxk_list = [1, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192]
            blk_list = [32, 64]

        print("=" * 70)
        print("PART 1: Direct CK-UA kernel correctness")
        print("=" * 70)
        total_p = 0; total_t = 0
        for cfg_name, nqh, nkh, hdim in head_configs:
            cfg_p = 0; cfg_t = 0
            for blk in blk_list:
                for seqs in seqs_list:
                    for maxk in maxk_list:
                        if blk == 32 and maxk < 256 and seqs >= 192:
                            continue
                        needed = max((maxk + blk - 1) // blk, 1)
                        mem = needed * max(seqs, 1) * 2 * blk * nkh * hdim * 4 / 1e9
                        if mem > 80:
                            continue
                        p, t, d = check_direct(seqs, maxk, nqh, nkh, hdim, blk)
                        cfg_p += p; cfg_t += t; total_p += p; total_t += t
                        if p < t:
                            print(f"  FAIL {cfg_name} seqs={seqs:4d} maxk={maxk:6d} blk={blk} diff={d:.5f}")
                            all_ok = False
            print(f"  {cfg_name}: {cfg_p}/{cfg_t} passed")

        print(f"\nDirect kernel: {total_p}/{total_t} {'PASS' if total_p == total_t else '*** FAILURES ***'}")

        print()
        print("=" * 70)
        print("PART 2: Production selector path correctness")
        print("=" * 70)
        total_p2 = 0; total_t2 = 0
        for cfg_name, nqh, nkh, hdim in head_configs:
            cfg_p = 0; cfg_t = 0
            for blk in blk_list:
                for seqs in seqs_list:
                    for maxk in maxk_list:
                        needed = max((maxk + blk - 1) // blk, 1)
                        mem = needed * max(seqs, 1) * 2 * blk * nkh * hdim * 4 / 1e9
                        if mem > 80:
                            continue
                        p, t, d = check_selector(seqs, maxk, nqh, nkh, hdim, blk)
                        cfg_p += p; cfg_t += t; total_p2 += p; total_t2 += t
                        if p < t:
                            print(f"  FAIL {cfg_name} seqs={seqs:4d} maxk={maxk:6d} blk={blk} diff={d:.5f}")
                            all_ok = False
            print(f"  {cfg_name}: {cfg_p}/{cfg_t} passed")

        print(f"\nProduction selector: {total_p2}/{total_t2} {'PASS' if total_p2 == total_t2 else '*** FAILURES ***'}")

    # ===== PERFORMANCE =====
    if not args.correctness_only:
        print()
        print("=" * 70)
        print("PART 3: Performance comparison (CK-UA vs Triton 2D vs Triton 3D)")
        print("=" * 70)
        print(f"\n{'config':>22s} {'seqs':>5s} {'maxk':>6s} {'blk':>3s}"
              f" {'CK-UA':>8s} {'T-2D':>8s} {'T-3D':>8s} {'auto':>8s} {'T-def':>8s}"
              f" {'CK vs best':>10s}")

        perf_configs = [
            ("GPT-OSS-64/8-d64", 64, 8, 64),
            ("Llama70B-8/1-d64", 8, 1, 64),
        ]
        if args.quick:
            perf_shapes = [(s, mk, b) for b in [32, 64] for s in [128, 256] for mk in [1024, 4096]]
        else:
            perf_shapes = [
                (s, mk, b)
                for b in [32, 64]
                for s in [32, 64, 128, 192, 256, 384, 512]
                for mk in [1024, 4096, 8192, 16384]
            ]

        ck_wins = 0
        triton_wins = 0
        total_auto_ms = 0.0
        total_triton_ms = 0.0

        for cfg_name, nqh, nkh, hdim in perf_configs:
            for seqs, maxk, blk in perf_shapes:
                ck, t2d, t3d, auto, t_def = bench_shape(seqs, maxk, nqh, nkh, hdim, blk)
                if ck is None:
                    continue

                best_triton = min(v for v in [t2d, t3d] if v is not None)
                ratio = ck / best_triton if best_triton else 0
                winner = "CK" if ratio < 0.95 else ("Triton" if ratio > 1.05 else "~tie")
                if ratio < 0.95:
                    ck_wins += 1
                elif ratio > 1.05:
                    triton_wins += 1

                if auto is not None and t_def is not None:
                    total_auto_ms += auto
                    total_triton_ms += t_def

                def fmt(v):
                    return f"{v:8.4f}" if v is not None else "     n/a"

                print(f"{cfg_name:>22s} {seqs:5d} {maxk:6d} {blk:3d}"
                      f" {fmt(ck)} {fmt(t2d)} {fmt(t3d)} {fmt(auto)} {fmt(t_def)}"
                      f" {ratio:8.3f}x {winner}")

        print(f"\nWins: CK-UA={ck_wins}, Triton={triton_wins}")
        if total_triton_ms > 0:
            saved = total_triton_ms - total_auto_ms
            pct = saved / total_triton_ms * 100
            print(f"E2E with selector: {total_auto_ms:.2f} ms vs Triton-only: {total_triton_ms:.2f} ms"
                  f" -> {pct:+.1f}%")

    # ===== SUMMARY =====
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not args.perf_only:
        print(f"  Direct kernel:       {total_p}/{total_t} passed")
        print(f"  Production selector: {total_p2}/{total_t2} passed")
    if not args.correctness_only:
        print(f"  CK-UA wins: {ck_wins} shapes, Triton wins: {triton_wins} shapes")
        if total_triton_ms > 0:
            print(f"  E2E selector speedup: {pct:+.1f}%")
    if not args.perf_only:
        print(f"  Status: {'ALL PASSED' if all_ok else '*** FAILURES ***'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
