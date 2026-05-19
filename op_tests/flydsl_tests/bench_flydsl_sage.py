#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark: FlyDSL Sage Attention vs Triton Sage v1 vs PyTorch SDPA.

Runs on gfx942 (MI300X) / gfx950 (MI350).

Speed mode (default):
    Only the attention kernel is timed. Q/K/V quantization (``sage_quant``),
    output allocation prep, and kernel-cache lookup are performed once,
    outside the ``do_bench`` loop. This matches ``bench_sage.py`` without
    ``--e2e`` and lets you compare raw kernel throughput across backends.

End-to-end mode (``--e2e``):
    The full high-precision wrapper is timed, including ``sage_quant`` on
    every call (mirrors ``bench_sage.py --e2e``).

Usage:
    python op_tests/flydsl_tests/bench_flydsl_sage.py
    python op_tests/flydsl_tests/bench_flydsl_sage.py --e2e
    python op_tests/flydsl_tests/bench_flydsl_sage.py --csv results.csv
    python op_tests/flydsl_tests/bench_flydsl_sage.py --warmup 50 --rep 200
"""

import argparse
import math
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
# Kernel-only runner builders
#
# Both functions pre-compute everything the high-precision wrappers do *except*
# the actual attention kernel launch, and return a thunk that runs only the
# kernel. This mirrors ``bench_sage.py``'s default (non-e2e) timing path so
# the Triton and FlyDSL columns are apples-to-apples kernel throughput.
# ---------------------------------------------------------------------------


def _build_triton_kernel_runner(q, k, v, causal, layout="bshd"):
    """Return a thunk that times only the Triton ``fav3_sage_func`` kernel."""
    import aiter as _aiter
    from aiter.ops.triton.attention.fav3_sage import (
        fav3_sage_func,
        get_sage_fwd_configs,
    )
    from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
        sage_quant as _triton_sage_quant,
    )

    cfg = get_sage_fwd_configs()
    fp8_type = _aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_type).max

    head_dim = q.shape[-1]
    softmax_scale = head_dim**-0.5

    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = _triton_sage_quant(
        q,
        k,
        v,
        fp8_type,
        fp8_max,
        BLKQ=cfg["BLOCK_M"],
        BLKK=cfg["BLOCK_N"],
        sm_scale=softmax_scale,
        layout=layout,
    )

    return lambda: fav3_sage_func(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=False,
        layout=layout,
        config=cfg,
    )


def _build_flydsl_kernel_runner(q, k, v, causal, layout="bshd"):
    """Return a thunk that times only the FlyDSL ``sage_attn_cdna`` kernel.

    Replicates the prep work inside :func:`flydsl_sage_attn_func` once, then
    captures everything needed for the kernel ``exe(...)`` call in the closure.
    """
    from aiter.utility.dtypes import fp8 as _fp8_dtype
    from aiter.ops.flydsl.sage_kernels import (
        sage_quant as _flydsl_sage_quant_dispatch,
        _get_kernel as _flydsl_get_kernel,
    )

    if layout == "bshd":
        batch, seq_q, num_q_heads, head_dim = q.shape
        _, seq_k, num_kv_heads, _ = k.shape
    else:
        batch, num_q_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _ = k.shape

    softmax_scale = 1.0 / math.sqrt(head_dim)

    try:
        cu_count = torch.cuda.get_device_properties(
            q.device.index
        ).multi_processor_count
    except Exception:
        cu_count = 256
    grid_at_bm256 = batch * num_q_heads * ((seq_q + 255) // 256)
    block_m = 128 if grid_at_bm256 < cu_count else 256
    block_n = 128
    waves_per_eu = 2

    fp8_dtype = _fp8_dtype
    fp8_max = torch.finfo(fp8_dtype).max

    q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale = _flydsl_sage_quant_dispatch(
        q,
        k,
        v,
        FP8_TYPE=fp8_dtype,
        FP8_MAX=fp8_max,
        BLKQ=block_m,
        BLKK=block_n,
        sm_scale=softmax_scale,
        layout=layout,
    )

    if layout == "bhsd":
        q_int8 = q_int8.permute(0, 2, 1, 3).contiguous()
        k_int8 = k_int8.permute(0, 2, 1, 3).contiguous()
        v_fp8 = v_fp8.permute(0, 2, 1, 3).contiguous()
    else:
        q_int8 = q_int8.contiguous()
        k_int8 = k_int8.contiguous()
        v_fp8 = v_fp8.contiguous()

    seq_q_pad = ((seq_q + block_m - 1) // block_m) * block_m
    n_pad_q = seq_q_pad - seq_q
    if n_pad_q > 0:
        q_int8 = torch.nn.functional.pad(q_int8, (0, 0, 0, 0, 0, n_pad_q))

    num_q_blocks = q_scale.shape[2]

    exe = _flydsl_get_kernel(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        waves_per_eu=waves_per_eu,
        block_m=block_m,
        block_n=block_n,
    )

    q_flat = q_int8.reshape(-1)
    k_flat = k_int8.reshape(-1)
    v_flat = v_fp8.reshape(-1)
    out_shape = (batch, seq_q_pad, num_q_heads, head_dim)
    device_index = q.device.index

    def _runner():
        o = torch.empty(out_shape, dtype=torch.bfloat16, device=q.device)
        launch_stream = torch.cuda.current_stream(q.device)
        with torch.cuda.device(device_index):
            exe(
                q_flat,
                k_flat,
                v_flat,
                o.reshape(-1),
                q_scale,
                k_scale,
                v_scale,
                batch,
                seq_q_pad,
                seq_k,
                num_q_blocks,
                stream=launch_stream,
            )
        return o

    return _runner


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


def run_speed(shapes, device, warmup, rep, e2e):
    mode = "e2e (wrapper, incl. sage_quant)" if e2e else "kernel-only (pre-quantized)"
    print("\n" + "=" * 110)
    print(f"SPEED  (warmup={warmup}ms  rep={rep}ms  mode={mode})")
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
                if e2e:

                    def triton_fn():
                        return fav3_sage_wrapper_func(q, k, v, causal=causal)

                else:
                    triton_fn = _build_triton_kernel_runner(q, k, v, causal)

                triton_fn()
                torch.cuda.synchronize()

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
                if e2e:

                    def flydsl_fn():
                        return flydsl_sage_attn_func(q, k, v, causal=causal)

                else:
                    flydsl_fn = _build_flydsl_kernel_runner(q, k, v, causal)

                flydsl_fn()
                torch.cuda.synchronize()

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
    parser.add_argument(
        "--e2e",
        action="store_true",
        help=(
            "End-to-end timing: include sage_quant in the timed region "
            "(default: time only the attention kernel, with quantization "
            "performed once outside the do_bench loop)."
        ),
    )
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
    print(f"Speed mode: {'e2e (incl. sage_quant)' if args.e2e else 'kernel-only'}")

    if not HAS_TRITON_SAGE:
        print(f"  [triton sage unavailable: {_triton_sage_err}]")
    if not HAS_FLYDSL_SAGE:
        print(f"  [flydsl sage unavailable: {_flydsl_sage_err}]")

    if not args.speed_only:
        run_accuracy(SHAPES, device)

    speed_rows = None
    if not args.accuracy_only:
        speed_rows = run_speed(SHAPES, device, args.warmup, args.rep, args.e2e)

    if args.csv and speed_rows:
        save_csv(args.csv, speed_rows)

    print()


if __name__ == "__main__":
    main()
