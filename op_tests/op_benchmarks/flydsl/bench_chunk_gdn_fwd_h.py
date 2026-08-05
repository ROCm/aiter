#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B benchmark: GDN K5 inter-chunk state scan (SILOTIGER-859).

Compares three implementations of ``chunk_gated_delta_rule_fwd_h``:

* ``triton`` — Triton baseline (``chunk_gated_delta_rule_fwd_h_opt_vk``)
* ``flydsl`` — arch-aware FlyDSL (gfx950: native; gfx942: NOT_IMPL until ported)
* ``hip``    — HIP kernel; available on both gfx942 and gfx950

The baseline for speedup columns is configurable via ``--baseline`` (default: triton).

Usage
-----
    # List preset shapes:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h.py --list

    # Time one shape, all impls, graph mode, verify against reference:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h.py \\
        --shape-index 1 --mode graph --verify

    # Full run with HIP as baseline, write Markdown + PNG:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h.py \\
        --baseline hip --mode all --output /tmp/gdn_bench.md

    # Filter to KDA shapes only:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h.py --gate gk

Environment
-----------
    AITER_TRITON_ONLY=1   — load Triton only (skip FlyDSL / HIP imports)
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Repo root on sys.path for direct script execution
# --------------------------------------------------------------------------- #
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_BENCH_DIR = str(Path(__file__).parent)
sys.path.insert(0, _BENCH_DIR)
from utils._bench_timing import EmptyGraphCaptureError, MeasureConfig, measure as _time_measure
from utils.bench_common import (
    add_output_args,
    add_timing_args,
    add_verification_args,
    collect_env_info,
    make_measure_config,
    print_result_table,
    write_bench_markdown,
)
from utils.plot_perf import make_bar_chart, make_summary_md, parse_bench_md

# --------------------------------------------------------------------------- #
# Preset shapes
# (model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode)
# gate_mode: "g" = scalar (GDN/Qwen3), "gk" = per-channel (KDA/Kimi-K3)
# --------------------------------------------------------------------------- #
PRESET_SHAPES: list[tuple] = [
    # KDA Kimi-K3 TP8  (H=12 is the primary serving shape from the ticket)
    ("kda_tp8",     12, 12,  8192, 1, 128, 128, 64, "gk"),
    ("kda_tp8",     12, 12, 32768, 1, 128, 128, 64, "gk"),
    ("kda_tp8",     12, 12,  8192, 4, 128, 128, 64, "gk"),
    ("kda_tp8",     12, 12, 32768, 4, 128, 128, 64, "gk"),
    ("kda_tp8",     12, 12,  8192, 8, 128, 128, 64, "gk"),
    ("kda_tp8",     12, 12, 32768, 8, 128, 128, 64, "gk"),
    # KDA Kimi-K3 TP4
    ("kda_tp4",     24, 24,  8192, 1, 128, 128, 64, "gk"),
    ("kda_tp4",     24, 24, 32768, 1, 128, 128, 64, "gk"),
    ("kda_tp4",     24, 24,  8192, 8, 128, 128, 64, "gk"),
    ("kda_tp4",     24, 24, 32768, 8, 128, 128, 64, "gk"),
    # GDN Qwen3-Next TP8
    ("gdn_q3n_tp8",  4,  2,  8192, 8, 128, 128, 64, "g"),
    ("gdn_q3n_tp8",  4,  2, 32768, 8, 128, 128, 64, "g"),
    # GDN Qwen3-Next TP4
    ("gdn_q3n_tp4",  8,  4,  8192, 4, 128, 128, 64, "g"),
    ("gdn_q3n_tp4",  8,  4, 32768, 4, 128, 128, 64, "g"),
    # GDN Qwen3.5-MoE TP1
    ("gdn_q35_tp1", 16, 16,  8192, 1, 128, 128, 64, "g"),
    ("gdn_q35_tp1", 32,  8,  8192, 1, 128, 128, 64, "g"),
    ("gdn_q35_tp1", 32,  8, 32768, 1, 128, 128, 64, "g"),
]

_BENCH_TITLE = "GDN K5 inter-chunk state scan"


def _shape_label(idx: int, shape: tuple) -> str:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate = shape
    return f"Shape {idx}: {model_tag} H={H} Hg={Hg} T={T_flat} N={N} gate={gate}"


# --------------------------------------------------------------------------- #
# Input tensor construction
# --------------------------------------------------------------------------- #
def _make_inputs(shape: tuple, device="cuda"):
    """Build all K5 input tensors for the given shape tuple.

    Returns:
        k, w_hm, u_hm, w_tm, g, gk, initial_state
        where w_hm/u_hm are head-major (kernel input) and w_tm is token-major
        (reference input).
    """
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode = shape
    B = 1
    dtype = torch.bfloat16

    k    = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    w_tm = torch.randn(B, T_flat, H,  K, dtype=dtype, device=device) * 0.1  # token-major
    u_tm = torch.randn(B, T_flat, H,  V, dtype=dtype, device=device) * 0.1
    w_hm = w_tm.permute(0, 2, 1, 3).contiguous()  # head-major for kernel
    u_hm = u_tm.permute(0, 2, 1, 3).contiguous()

    g, gk = None, None
    if gate_mode == "g":
        g = (torch.randn(H, T_flat, dtype=torch.float32, device=device).abs() * -0.5
             ).cumsum(dim=1).contiguous()
    elif gate_mode == "gk":
        gk = (torch.randn(T_flat, H, K, dtype=torch.float32, device=device).abs() * -0.1
              ).cumsum(dim=0).contiguous()

    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01
    return k, w_hm, u_hm, w_tm, g, gk, h0


# --------------------------------------------------------------------------- #
# Implementation registry
# --------------------------------------------------------------------------- #
_TRITON_ONLY = os.environ.get("AITER_TRITON_ONLY", "0") == "1"


def _load_impls(which: str) -> dict:
    """Return {impl_name: callable} for the requested set."""
    requested = {s.strip() for s in which.split(",")} if which != "all" else None

    def _want(name: str) -> bool:
        return requested is None or name in requested

    impls: dict = {}

    if _want("triton"):
        try:
            from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h_opt_vk,
            )
            impls["triton"] = chunk_gated_delta_rule_fwd_h_opt_vk
        except ImportError as e:
            warnings.warn(f"Triton K5 not available: {e}")

    if _want("flydsl") and not _TRITON_ONLY:
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl,
            )
            impls["flydsl"] = chunk_gated_delta_rule_fwd_h_flydsl
        except ImportError as e:
            warnings.warn(f"FlyDSL K5 not available: {e}")

    # HIP is available on both gfx942 and gfx950 (disabled only on gfx12).
    if _want("hip") and not _TRITON_ONLY:
        try:
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )
            impls["hip"] = chunk_gated_delta_rule_fwd_h_hip_fn
        except ImportError as e:
            warnings.warn(f"HIP K5 not available: {e}")

    return impls


# --------------------------------------------------------------------------- #
# Reference (same as the unit test suite)
# --------------------------------------------------------------------------- #
_ref_fn = None   # loaded lazily on first use


def _load_ref():
    """Load ref_chunk_gated_delta_rule_fwd_h from the unit test file."""
    global _ref_fn
    if _ref_fn is not None:
        return _ref_fn
    test_path = Path(_REPO_ROOT) / "op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py"
    spec = importlib.util.spec_from_file_location("_test_prefill", test_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ref_fn = mod.ref_chunk_gated_delta_rule_fwd_h
    return _ref_fn


# --------------------------------------------------------------------------- #
# TFLOPS
# --------------------------------------------------------------------------- #
def calculate_tflops(N: int, H: int, T_flat: int, K: int, V: int, time_us: float) -> float:
    """Total FLOPs for K5 = GEMM1 (w @ h^T) + GEMM2 (k^T @ v_new).

    Both GEMMs cost 2·BT·K·V per chunk; summed over NT=T_flat/BT chunks:
    total = 4·N·H·T_flat·K·V.
    """
    if time_us <= 0:
        return float("nan")
    return 4 * N * H * T_flat * K * V / (time_us * 1e-6) / 1e12


# --------------------------------------------------------------------------- #
# Per-shape timing
# --------------------------------------------------------------------------- #
def _make_closure(fn, k, w_hm, u_hm, g, gk, h0):
    def _run():
        fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
           initial_state=h0, output_final_state=True, save_new_value=True, use_exp2=True)
    return _run


def _verify_impl(fn, k, w_hm, u_hm, w_tm, g, gk, h0, verification: str,
                 baseline_fn=None) -> str:
    """Run correctness check. Returns a grade string."""
    if verification == "none":
        return "N/A"
    try:
        h, v_new, fs = fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
                          initial_state=h0, output_final_state=True,
                          save_new_value=True, use_exp2=True)
    except Exception as e:
        return f"ERROR({type(e).__name__})"

    def _rmse_ratio(a, b):
        diff = (a.float() - b.float()).pow(2).mean().sqrt()
        return (diff / (b.float().pow(2).mean().sqrt() + 1e-8)).item()

    if verification == "reference":
        try:
            ref = _load_ref()
            # Reference expects token-major w; accepts exactly one of g/gk.
            h_ref, _, _ = ref(k=k, w=w_tm, u=u_hm.permute(0, 2, 1, 3),
                               g=g, gk=gk, initial_state=h0,
                               output_final_state=True)
        except Exception as e:
            return f"REF-ERROR({e})"
        ratio = _rmse_ratio(h, h_ref)
        return f"{'PASS' if ratio < 0.05 else 'FAIL'}(rmse_ratio={ratio:.2e})"

    if verification == "baseline" and baseline_fn is not None:
        try:
            h_base, _, _ = baseline_fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
                                        initial_state=h0, output_final_state=True,
                                        save_new_value=False, use_exp2=True)
        except Exception as e:
            return f"BASELINE-ERROR({e})"
        ratio = _rmse_ratio(h, h_base)
        return f"{'PASS' if ratio < 0.05 else 'FAIL'}(rmse_ratio={ratio:.2e})"

    return "N/A"


def _run_one(idx: int, impls: dict, shape: tuple, args, cfg: MeasureConfig) -> dict:
    """Time and verify all impls for one shape."""
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode = shape
    label = _shape_label(idx, shape)

    try:
        k, w_hm, u_hm, w_tm, g, gk, h0 = _make_inputs(shape)
    except Exception as e:
        return {"label": label, "error": str(e)}

    modes = ["eager", "graph"] if args.mode == "all" else [args.mode]
    baseline_name = args.baseline
    baseline_fn = impls.get(baseline_name)

    baseline_times: dict[str, float] = {}
    results_by_impl: dict = {}

    for impl_name, fn in impls.items():
        print(f"  {impl_name}...", end=" ", flush=True)
        closure = _make_closure(fn, k, w_hm, u_hm, g, gk, h0)

        try:
            closure()
            torch.cuda.synchronize()
        except NotImplementedError as e:
            print(f"NOT_IMPL")
            results_by_impl[impl_name] = {"error": f"NOT_IMPL: {e}"}
            continue
        except Exception as e:
            print(f"PROBE-FAIL: {e}")
            results_by_impl[impl_name] = {"error": str(e)}
            continue

        timing: dict = {}
        tflops_d: dict = {}
        for mode in modes:
            try:
                stats = _time_measure(closure, mode, cfg)
                timing[mode] = stats
                tflops_d[mode] = calculate_tflops(N, H, T_flat, K, V, stats.median_us)
            except EmptyGraphCaptureError as e:
                timing[mode] = f"GRAPH-FAIL: {e}"
                tflops_d[mode] = None
            except Exception as e:
                timing[mode] = f"ERROR: {e}"
                tflops_d[mode] = None

        verify_str = "N/A"
        if args.verification != "none":
            verify_str = _verify_impl(
                fn, k, w_hm, u_hm, w_tm, g, gk, h0,
                verification=args.verification,
                baseline_fn=baseline_fn if impl_name != baseline_name else None,
            )

        results_by_impl[impl_name] = {
            "timing": timing, "tflops": tflops_d, "verify": verify_str,
        }

        if impl_name == baseline_name:
            for mode in modes:
                t = timing.get(mode)
                if hasattr(t, "median_us"):
                    baseline_times[mode] = t.median_us

        # Console summary line
        for mode in modes:
            t = timing.get(mode)
            tf = tflops_d.get(mode)
            if hasattr(t, "median_us"):
                base = baseline_times.get(mode)
                sp = (f"  ×{base / t.median_us:.2f}"
                      if (base and impl_name != baseline_name and t.median_us > 0) else "")
                tf_s = f"{tf:.3f}" if tf is not None else "—"
                print(f"[{mode}] {t.median_us:.1f} us  {tf_s} TFLOPs{sp}", end="  ")
            else:
                print(f"[{mode}] {t}", end="  ")
        print(f"  verify={verify_str}")

    return {
        "label": label,
        "shape": shape,
        "impls": results_by_impl,
        "baseline_times": baseline_times,
        "baseline_name": baseline_name,
        "modes": modes,
        "N": N, "H": H, "T_flat": T_flat, "K": K, "V": V,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(args):
    impls = _load_impls(args.impl)
    if not impls:
        print("No implementations available.", file=sys.stderr)
        sys.exit(1)

    if args.baseline not in impls:
        warnings.warn(
            f"Baseline '{args.baseline}' not in loaded impls {list(impls)}; "
            "speedup columns will be empty."
        )

    cfg = make_measure_config(args)

    if args.shape_index is not None:
        idx = args.shape_index
        if not (1 <= idx <= len(PRESET_SHAPES)):
            print(f"--shape-index must be 1–{len(PRESET_SHAPES)}", file=sys.stderr)
            sys.exit(1)
        shapes_to_run = [(idx, PRESET_SHAPES[idx - 1])]
    else:
        gate_filter = getattr(args, "gate", "all")
        shapes_to_run = [
            (i + 1, s) for i, s in enumerate(PRESET_SHAPES)
            if gate_filter == "all" or s[-1] == gate_filter
        ]

    all_rows: list[dict] = []
    for idx, shape in shapes_to_run:
        print(f"\n{'='*60}\n{_shape_label(idx, shape)}\n{'='*60}")
        row = _run_one(idx, impls, shape, args, cfg)
        print_result_table(row)
        all_rows.append(row)

    if args.output:
        env = collect_env_info()
        write_bench_markdown(
            args.output, _BENCH_TITLE, all_rows, env,
            baseline_name=args.baseline,
        )
        print(f"\nMarkdown report written to {args.output}")

        try:
            results = parse_bench_md(args.output)
            stem = Path(args.output).stem
            out_dir = Path(args.output).parent
            png_path = str(out_dir / f"{stem}-plot.png")
            modes_plot = ["eager", "graph"] if args.mode == "all" else [args.mode]
            make_bar_chart(results, png_path, title=_BENCH_TITLE,
                           mode=modes_plot[0], baseline_label=args.baseline.capitalize())
            summary_md = str(out_dir / f"{stem}-summary.md")
            make_summary_md(results, summary_md, png_path, args.output,
                            title=_BENCH_TITLE, mode=modes_plot[0],
                            baseline_label=args.baseline.capitalize())
        except Exception as e:
            warnings.warn(f"Plot/summary generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark: GDN K5 inter-chunk state scan (SILOTIGER-859)."
    )

    shape_grp = parser.add_mutually_exclusive_group()
    shape_grp.add_argument(
        "--shape-index", type=int, default=None, metavar="N",
        help="Run one preset shape by 1-based index (see --list).",
    )
    shape_grp.add_argument(
        "--list", action="store_true",
        help="List all preset shapes with indices and exit.",
    )

    parser.add_argument(
        "--impl", default="all", metavar="IMPLS",
        help="Comma-separated implementations: triton, flydsl, hip. Default: all.",
    )
    parser.add_argument(
        "--baseline", default="triton", choices=("triton", "hip"),
        help="Implementation used as speedup baseline (default: triton).",
    )
    parser.add_argument(
        "--gate", default="all", choices=("g", "gk", "all"),
        help="Filter shapes by gate type: g (GDN), gk (KDA), all (default).",
    )

    add_timing_args(parser)
    add_output_args(parser)
    add_verification_args(parser)

    args = parser.parse_args()

    if args.list:
        print(f"{'#':>3}  {'label':<60}  gate")
        print("-" * 75)
        for i, s in enumerate(PRESET_SHAPES, 1):
            model_tag, H, Hg, T_flat, N, K, V, BT, gate = s
            label = f"{model_tag}  H={H} Hg={Hg} T={T_flat} N={N} K={K} V={V}"
            print(f"{i:>3}  {label:<60}  {gate}")
        return

    run(args)


if __name__ == "__main__":
    main()
