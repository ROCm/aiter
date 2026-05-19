#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark: FlyDSL Sage Attention vs Triton Sage v1 vs PyTorch SDPA.

Runs on gfx942 (MI300X) / gfx950 (MI350).

Usage:
    python op_tests/flydsl_tests/bench_flydsl_sage.py
    python op_tests/flydsl_tests/bench_flydsl_sage.py --csv results.csv
    python op_tests/flydsl_tests/bench_flydsl_sage.py --warmup 50 --rep 200
"""

import argparse
import os
import sys

# Ensure the repo root is on the path so `aiter` is importable
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from typing import List, Tuple  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

try:
    import triton
    import triton.testing

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ---- provider imports (graceful) ----

# Inject aiter.dtypes before any aiter sub-module that needs it at import time.
# aiter.__init__ only populates dtypes when jit.core succeeds (requires psutil).
# In environments without psutil we inject it manually so triton/flydsl ops load.
try:
    import aiter as _aiter

    if not hasattr(_aiter, "dtypes"):
        from aiter.utility import dtypes as _dtypes_mod

        _aiter.dtypes = _dtypes_mod
except Exception:
    pass

try:
    from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

    HAS_TRITON_SAGE = True
except Exception as e:
    HAS_TRITON_SAGE = False
    _triton_sage_err = str(e)

try:
    import flydsl  # noqa: F401
    from aiter.ops.flydsl.sage_kernels import flydsl_sage_attn_func

    HAS_FLYDSL_SAGE = True
except Exception as e:
    HAS_FLYDSL_SAGE = False
    _flydsl_sage_err = str(e)


# ---------------------------------------------------------------------------
# Shapes: (batch, seq_len, num_q_heads, num_kv_heads, head_dim, causal)
# ---------------------------------------------------------------------------
SHAPES: List[Tuple] = [
    (1, 1024, 8, 8, 128, False),
    (1, 3000, 8, 8, 128, False),  # unaligned seq (3000 % 256 != 0)
    (1, 4096, 8, 8, 128, False),
    (1, 4096, 8, 8, 128, True),
    (1, 8192, 8, 8, 128, False),
    (1, 8192, 8, 8, 128, True),
    (1, 4096, 16, 4, 128, False),  # GQA 4:1
    (2, 4096, 16, 4, 128, False),  # GQA 4:1 batch=2
    (2, 4096, 8, 8, 128, False),
    (1, 16384, 8, 8, 128, False),
    (1, 16384, 24, 24, 128, False),  # large seq + many heads
]


def _label(B, S, Hq, Hk, D, causal):
    tag = "caus" if causal else "fwd"
    return f"B{B} S{S:>5} Hq{Hq} Hk{Hk} D{D} {tag}"


def _make_qkv(B, S, Hq, Hk, D, device="cuda", seed=42):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((B, S, Hq, D), generator=g, dtype=torch.bfloat16, device=device)
    k = torch.randn((B, S, Hk, D), generator=g, dtype=torch.bfloat16, device=device)
    v = torch.randn((B, S, Hk, D), generator=g, dtype=torch.bfloat16, device=device)
    return q, k, v


def _sdpa_ref(q, k, v, causal):
    """PyTorch SDPA in fp32, BSHD → BSHD. Handles GQA via KV head expansion."""
    # q: [B, S, Hq, D], k/v: [B, S, Hk, D] → BHSD for SDPA
    q32 = q.float().transpose(1, 2)  # [B, Hq, S, D]
    k32 = k.float().transpose(1, 2)  # [B, Hk, S, D]
    v32 = v.float().transpose(1, 2)
    Hq, Hk = q32.shape[1], k32.shape[1]
    if Hq != Hk:
        # Expand KV heads to match Q heads for GQA reference
        groups = Hq // Hk
        k32 = k32.repeat_interleave(groups, dim=1)
        v32 = v32.repeat_interleave(groups, dim=1)
    out = F.scaled_dot_product_attention(q32, k32, v32, is_causal=causal)
    return out.transpose(1, 2).to(torch.bfloat16)


def _cos_stats(a, b, D):
    cos = F.cosine_similarity(
        a.float().reshape(-1, D),
        b.float().reshape(-1, D),
        dim=1,
    )
    return cos.min().item(), cos.mean().item()


def _max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def _attn_flops(B, Hq, S, D, causal):
    # 2 GEMMs (QK + PV), each 2 flops per multiply-add
    flops = 4 * B * Hq * S * S * D
    return flops // 2 if causal else flops


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def run_accuracy(shapes, device):
    print("\n" + "=" * 100)
    print("ACCURACY  (cosine similarity vs fp32 SDPA reference)")
    print("=" * 100)

    hdr = (
        f"{'Shape':<38}"
        f"{'FlyDSL cos_min':>14} {'FlyDSL cos_mean':>15} {'FlyDSL max_err':>14}"
        f"{'Triton cos_min':>15} {'Triton cos_mean':>16} {'Triton max_err':>14}"
    )
    print(hdr)
    print("-" * 100)

    rows = []
    for B, S, Hq, Hk, D, causal in shapes:
        label = _label(B, S, Hq, Hk, D, causal)
        q, k, v = _make_qkv(B, S, Hq, Hk, D, device)
        ref = _sdpa_ref(q, k, v, causal)

        # FlyDSL
        if HAS_FLYDSL_SAGE:
            try:
                flydsl_out = flydsl_sage_attn_func(q, k, v, causal=causal)
                torch.cuda.synchronize()
                f_cmin, f_cmean = _cos_stats(flydsl_out, ref, D)
                f_err = _max_abs(flydsl_out, ref)
                f_str = f"{f_cmin:>14.4f} {f_cmean:>15.4f} {f_err:>14.4f}"
            except Exception:
                f_str = f"{'FAILED':>14} {'':>15} {'':>14}"
                flydsl_out = None
        else:
            f_str = f"{'N/A':>14} {'':>15} {'':>14}"
            flydsl_out = None

        # Triton
        if HAS_TRITON_SAGE:
            try:
                triton_out = fav3_sage_wrapper_func(q, k, v, causal=causal)
                torch.cuda.synchronize()
                t_cmin, t_cmean = _cos_stats(triton_out, ref, D)
                t_err = _max_abs(triton_out, ref)
                t_str = f"{t_cmin:>15.4f} {t_cmean:>16.4f} {t_err:>14.4f}"
            except Exception:
                t_str = f"{'FAILED':>15} {'':>16} {'':>14}"
        else:
            t_str = f"{'N/A':>15} {'':>16} {'':>14}"

        line = f"{label:<38}{f_str}{t_str}"
        print(line)
        rows.append((label, f_str, t_str))

    return rows


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------


def _do_bench(fn, warmup, rep):
    if HAS_TRITON:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    # Fallback: manual timing
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    import time

    t0 = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1000


def run_speed(shapes, device, warmup, rep):
    print("\n" + "=" * 110)
    print(f"SPEED  (warmup={warmup}ms  rep={rep}ms)")
    print("=" * 110)

    hdr = (
        f"{'Shape':<38}"
        f"{'SDPA ms':>9} {'SDPA TFLOPS':>11}"
        f"{'Triton ms':>10} {'Triton TFLOPS':>13} {'vs SDPA':>8}"
        f"{'FlyDSL ms':>10} {'FlyDSL TFLOPS':>13} {'vs SDPA':>8} {'vs Triton':>10}"
    )
    print(hdr)
    print("-" * 110)

    csv_rows = []
    for B, S, Hq, Hk, D, causal in shapes:
        label = _label(B, S, Hq, Hk, D, causal)
        q, k, v = _make_qkv(B, S, Hq, Hk, D, device)
        flops = _attn_flops(B, Hq, S, D, causal)

        # SDPA (fp32 reference) — expand KV heads for GQA
        q_bhsd = q.float().transpose(1, 2).contiguous()
        k_bhsd = k.float().transpose(1, 2).contiguous()
        v_bhsd = v.float().transpose(1, 2).contiguous()
        if Hq != Hk:
            groups = Hq // Hk
            k_bhsd = k_bhsd.repeat_interleave(groups, dim=1)
            v_bhsd = v_bhsd.repeat_interleave(groups, dim=1)

        def sdpa_fn():
            return F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=causal
            )

        sdpa_ms = _do_bench(sdpa_fn, warmup, rep)
        sdpa_tflops = flops / sdpa_ms / 1e9

        # Triton sage
        if HAS_TRITON_SAGE:
            try:
                # Warm up JIT
                fav3_sage_wrapper_func(q, k, v, causal=causal)
                torch.cuda.synchronize()

                def triton_fn():
                    return fav3_sage_wrapper_func(q, k, v, causal=causal)

                triton_ms = _do_bench(triton_fn, warmup, rep)
                triton_tflops = flops / triton_ms / 1e9
                triton_vs_sdpa = sdpa_ms / triton_ms
                t_str = (
                    f"{triton_ms:>10.3f} {triton_tflops:>13.2f} {triton_vs_sdpa:>7.2f}x"
                )
            except Exception:
                t_str = f"{'FAILED':>10} {'':>13} {'':>8}"
                triton_ms = None
        else:
            t_str = f"{'N/A':>10} {'':>13} {'':>8}"
            triton_ms = None

        # FlyDSL sage
        if HAS_FLYDSL_SAGE:
            try:
                # Trigger JIT compilation outside timing
                flydsl_sage_attn_func(q, k, v, causal=causal)
                torch.cuda.synchronize()

                def flydsl_fn():
                    return flydsl_sage_attn_func(q, k, v, causal=causal)

                flydsl_ms = _do_bench(flydsl_fn, warmup, rep)
                flydsl_tflops = flops / flydsl_ms / 1e9
                flydsl_vs_sdpa = sdpa_ms / flydsl_ms
                flydsl_vs_triton = (
                    f"{triton_ms / flydsl_ms:.2f}x" if triton_ms else "  N/A"
                )
                f_str = (
                    f"{flydsl_ms:>10.3f} {flydsl_tflops:>13.2f}"
                    f" {flydsl_vs_sdpa:>7.2f}x {flydsl_vs_triton:>10}"
                )
            except Exception:
                f_str = f"{'FAILED':>10} {'':>13} {'':>8} {'':>10}"
                flydsl_ms = None
        else:
            f_str = f"{'N/A':>10} {'':>13} {'':>8} {'':>10}"
            flydsl_ms = None

        line = f"{label:<38}" f"{sdpa_ms:>9.3f} {sdpa_tflops:>11.2f}" f"{t_str}{f_str}"
        print(line)
        csv_rows.append((label, sdpa_ms, sdpa_tflops, triton_ms, flydsl_ms))

    return csv_rows


def save_csv(path, speed_rows):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shape", "sdpa_ms", "sdpa_tflops", "triton_ms", "flydsl_ms"])
        for row in speed_rows:
            w.writerow(row)
    print(f"\nCSV saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="FlyDSL Sage Attention benchmark")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup ms for do_bench")
    parser.add_argument(
        "--rep", type=int, default=100, help="Repetition ms for do_bench"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Save speed results to CSV"
    )
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--speed-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/HIP device found", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda", 0)
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch = "unknown"
    print(f"Device: {torch.cuda.get_device_name(0)}  arch={arch}")
    print(
        f"Providers: triton_sage={'YES' if HAS_TRITON_SAGE else 'NO'}  "
        f"flydsl_sage={'YES' if HAS_FLYDSL_SAGE else 'NO'}"
    )

    if not HAS_TRITON_SAGE:
        print(f"  [triton sage unavailable: {_triton_sage_err}]")
    if not HAS_FLYDSL_SAGE:
        print(f"  [flydsl sage unavailable: {_flydsl_sage_err}]")

    if not args.speed_only:
        run_accuracy(SHAPES, device)

    speed_rows = None
    if not args.accuracy_only:
        speed_rows = run_speed(SHAPES, device, args.warmup, args.rep)

    if args.csv and speed_rows:
        save_csv(args.csv, speed_rows)

    print()


if __name__ == "__main__":
    main()
