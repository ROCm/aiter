#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark: FlyDSL MXFP4 Sage Attention vs Triton MXFP4 vs PyTorch SDPA.

Runs on gfx950 (MI350) only — FP4 MFMA is gfx950-exclusive.

Usage:
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage_mxfp4.py
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage_mxfp4.py --csv results.csv
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage_mxfp4.py --warmup 50 --rep 200
    HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/bench_flydsl_sage_mxfp4.py --q-smooth
"""

import argparse
import os
import sys

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

try:
    import aiter as _aiter
    if not hasattr(_aiter, "dtypes"):
        from aiter.utility import dtypes as _dtypes_mod
        _aiter.dtypes = _dtypes_mod
except Exception:
    pass

try:
    from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
        fav3_sage_mxfp4_wrapper as _triton_mxfp4_wrapper,
    )
    HAS_TRITON_MXFP4 = True
except Exception as e:
    HAS_TRITON_MXFP4 = False
    _triton_mxfp4_err = str(e)

try:
    import flydsl  # noqa: F401
    from aiter.ops.flydsl.sage_mxfp4_kernels import (
        fav3_sage_mxfp4_flydsl_wrapper,
    )
    HAS_FLYDSL_MXFP4 = True
except Exception as e:
    HAS_FLYDSL_MXFP4 = False
    _flydsl_mxfp4_err = str(e)


# (B, S, Hq, Hk, D, causal). MXFP4 wins over INT8 on data-bandwidth-heavy
# (long-S) shapes, so the sweep biases toward long-S.
SHAPES: List[Tuple] = [
    (1, 1024, 8, 8, 128, False),
    (1, 4096, 8, 8, 128, False),
    (1, 4096, 8, 8, 128, True),
    (1, 8192, 8, 8, 128, False),
    (1, 8192, 8, 8, 128, True),
    (1, 16384, 8, 8, 128, False),
    (1, 16384, 8, 8, 128, True),
    (1, 32768, 8, 8, 128, False),
    (2, 8192, 16, 4, 128, False),       # GQA + long-S
    (1, 4096, 24, 24, 128, False),
    (1, 16384, 24, 24, 128, False),
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
    q32 = q.float().transpose(1, 2)
    k32 = k.float().transpose(1, 2)
    v32 = v.float().transpose(1, 2)
    Hq, Hk = q32.shape[1], k32.shape[1]
    if Hq != Hk:
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
    flops = 4 * B * Hq * S * S * D
    return flops // 2 if causal else flops


def _do_bench(fn, warmup, rep):
    if HAS_TRITON:
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 50 * 1000


def run_accuracy(shapes, device, q_smooth):
    print("\n" + "=" * 110)
    print(f"ACCURACY  (q_smooth={q_smooth})")
    print("=" * 110)
    hdr = (
        f"{'Shape':<38}"
        f"{'FlyDSL cos_min':>14} {'FlyDSL cos_mean':>15} {'FlyDSL max_err':>14}"
        f"{'Triton cos_min':>15} {'Triton cos_mean':>16} {'Triton max_err':>14}"
    )
    print(hdr)
    print("-" * 110)

    for B, S, Hq, Hk, D, causal in shapes:
        label = _label(B, S, Hq, Hk, D, causal)
        q, k, v = _make_qkv(B, S, Hq, Hk, D, device)
        ref = _sdpa_ref(q, k, v, causal)

        if HAS_FLYDSL_MXFP4:
            try:
                f_out = fav3_sage_mxfp4_flydsl_wrapper(
                    q, k, v, causal=causal, layout="bshd", q_smooth=q_smooth,
                )
                torch.cuda.synchronize()
                fc_min, fc_mean = _cos_stats(f_out, ref, D)
                f_err = _max_abs(f_out, ref)
                f_str = f"{fc_min:>14.4f} {fc_mean:>15.4f} {f_err:>14.4f}"
            except Exception as e:
                f_str = f"{'FAILED':>14} {str(e)[:30]:>15} {'':>14}"
        else:
            f_str = f"{'N/A':>14} {'':>15} {'':>14}"

        if HAS_TRITON_MXFP4:
            try:
                t_out = _triton_mxfp4_wrapper(
                    q, k, v, causal=causal, layout="bshd", q_smooth=q_smooth,
                    hadamard_rotation=True,
                )
                torch.cuda.synchronize()
                tc_min, tc_mean = _cos_stats(t_out, ref, D)
                t_err = _max_abs(t_out, ref)
                t_str = f"{tc_min:>15.4f} {tc_mean:>16.4f} {t_err:>14.4f}"
            except Exception as e:
                t_str = f"{'FAILED':>15} {str(e)[:30]:>16} {'':>14}"
        else:
            t_str = f"{'N/A':>15} {'':>16} {'':>14}"

        print(f"{label:<38}{f_str}{t_str}")


def run_speed(shapes, device, warmup, rep, q_smooth):
    print("\n" + "=" * 115)
    print(f"SPEED  (warmup={warmup}ms  rep={rep}ms  q_smooth={q_smooth})")
    print("=" * 115)

    hdr = (
        f"{'Shape':<38}"
        f"{'SDPA ms':>9} {'SDPA TFLOPS':>11}"
        f"{'Triton ms':>10} {'Triton TFLOPS':>13} {'vs SDPA':>8}"
        f"{'FlyDSL ms':>10} {'FlyDSL TFLOPS':>13} {'vs SDPA':>8} {'vs Triton':>10}"
    )
    print(hdr)
    print("-" * 115)

    csv_rows = []
    for B, S, Hq, Hk, D, causal in shapes:
        label = _label(B, S, Hq, Hk, D, causal)
        q, k, v = _make_qkv(B, S, Hq, Hk, D, device)
        flops = _attn_flops(B, Hq, S, D, causal)

        # SDPA fp32 (with KV head expansion for GQA)
        q_b = q.float().transpose(1, 2).contiguous()
        k_b = k.float().transpose(1, 2).contiguous()
        v_b = v.float().transpose(1, 2).contiguous()
        if Hq != Hk:
            groups = Hq // Hk
            k_b = k_b.repeat_interleave(groups, dim=1)
            v_b = v_b.repeat_interleave(groups, dim=1)

        def sdpa_fn():
            return F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=causal)

        sdpa_ms = _do_bench(sdpa_fn, warmup, rep)
        sdpa_tflops = flops / sdpa_ms / 1e9

        # Triton MXFP4 wrapper
        if HAS_TRITON_MXFP4:
            try:
                _triton_mxfp4_wrapper(
                    q, k, v, causal=causal, layout="bshd",
                    q_smooth=q_smooth, hadamard_rotation=True,
                )
                torch.cuda.synchronize()

                def triton_fn():
                    return _triton_mxfp4_wrapper(
                        q, k, v, causal=causal, layout="bshd",
                        q_smooth=q_smooth, hadamard_rotation=True,
                    )

                triton_ms = _do_bench(triton_fn, warmup, rep)
                triton_tflops = flops / triton_ms / 1e9
                triton_vs_sdpa = sdpa_ms / triton_ms
                t_str = (
                    f"{triton_ms:>10.3f} {triton_tflops:>13.2f} "
                    f"{triton_vs_sdpa:>7.2f}x"
                )
            except Exception:
                t_str = f"{'FAILED':>10} {'':>13} {'':>8}"
                triton_ms = None
        else:
            t_str = f"{'N/A':>10} {'':>13} {'':>8}"
            triton_ms = None

        # FlyDSL MXFP4 wrapper
        if HAS_FLYDSL_MXFP4:
            try:
                fav3_sage_mxfp4_flydsl_wrapper(
                    q, k, v, causal=causal, layout="bshd",
                    q_smooth=q_smooth, hadamard_rotation=True,
                )
                torch.cuda.synchronize()

                def flydsl_fn():
                    return fav3_sage_mxfp4_flydsl_wrapper(
                        q, k, v, causal=causal, layout="bshd",
                        q_smooth=q_smooth, hadamard_rotation=True,
                    )

                flydsl_ms = _do_bench(flydsl_fn, warmup, rep)
                flydsl_tflops = flops / flydsl_ms / 1e9
                f_vs_sdpa = sdpa_ms / flydsl_ms
                f_vs_triton = triton_ms / flydsl_ms if triton_ms else 0.0
                f_str = (
                    f"{flydsl_ms:>10.3f} {flydsl_tflops:>13.2f} "
                    f"{f_vs_sdpa:>7.2f}x {f_vs_triton:>9.2f}x"
                )
            except Exception:
                f_str = f"{'FAILED':>10} {'':>13} {'':>8} {'':>10}"
                flydsl_ms = None
        else:
            f_str = f"{'N/A':>10} {'':>13} {'':>8} {'':>10}"
            flydsl_ms = None

        line = (
            f"{label:<38}"
            f"{sdpa_ms:>9.3f} {sdpa_tflops:>11.2f}"
            f"{t_str}"
            f"{f_str}"
        )
        print(line)
        csv_rows.append(
            (label, sdpa_ms, sdpa_tflops, triton_ms, flydsl_ms)
        )

    return csv_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=100, help="warmup ms")
    p.add_argument("--rep", type=int, default=500, help="bench rep ms")
    p.add_argument("--csv", type=str, default=None, help="optional CSV output path")
    p.add_argument("--accuracy-only", action="store_true")
    p.add_argument("--speed-only", action="store_true")
    p.add_argument("--q-smooth", action="store_true",
                   help="enable q-smoothing (exercises bias path)")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available")
        sys.exit(1)
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0]
    if not arch_base.startswith("gfx950"):
        print(f"FP4 MFMA requires gfx950, got {arch!r} — exiting cleanly.")
        sys.exit(0)

    print(f"Device: {torch.cuda.get_device_name(0)} ({arch})")
    print(f"FlyDSL MXFP4 available: {HAS_FLYDSL_MXFP4}")
    if not HAS_FLYDSL_MXFP4:
        print(f"  (reason: {_flydsl_mxfp4_err})")
    print(f"Triton MXFP4 available: {HAS_TRITON_MXFP4}")
    if not HAS_TRITON_MXFP4:
        print(f"  (reason: {_triton_mxfp4_err})")

    device = "cuda"
    if not args.speed_only:
        run_accuracy(SHAPES, device, args.q_smooth)
    if not args.accuracy_only:
        rows = run_speed(SHAPES, device, args.warmup, args.rep, args.q_smooth)
        if args.csv:
            import csv
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["shape", "sdpa_ms", "sdpa_tflops",
                            "triton_ms", "flydsl_ms"])
                for r in rows:
                    w.writerow(r)
            print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
