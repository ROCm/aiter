#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B benchmark: GDN K5+K6 fused forward (inter-chunk scan + output).

Compares the FlyDSL fused K5+K6 kernel against the current separate-kernel
pipelines. Every implementation produces the same output ``o`` ([B, T, H, V]);
the fused one does it in a single dispatch, eliminating the ``h`` snapshot and
``v_new`` HBM round-trips.

Implementations
---------------
* ``K5_triton+K6_triton`` — Triton K5 (``chunk_gated_delta_rule_fwd_h_opt_vk``)
                            then Triton K6 (``chunk_fwd_o_opt_vk``).
* ``K5_hip+K6_triton``    — HIP K5 then Triton K6.
* ``K5_flydsl+K6_triton`` — FlyDSL K5 then Triton K6 (current best FlyDSL path).
* ``K5K6_flydsl_fused``   — FlyDSL fused K5+K6 (this work).

The baseline for speedup columns is configurable via ``--baseline``
(default: ``K5_flydsl+K6_triton``).

Usage
-----
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h_o.py --list
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h_o.py \\
        --shape-index 1 --mode graph --verify
    # A 1-based inclusive range of preset shapes:
    PYTHONPATH=. python op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd_h_o.py \\
        --shape-range 5-8 --mode graph
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch

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

# Reuse the K5 bench's shape presets, input builders, and HIP graph-capture
# machinery so the two benches stay in lockstep (same shapes, same seqlen
# patterns, same TFLOPs chunk accounting, same capture-safe HIP metadata).
#
# ``_hip_meta_cache`` and ``_CaptureSafeMeta`` are imported by reference: the
# K5 bench's ``_adapt_hip`` closure (used by our HIP runner) reads that same
# module-level dict, so populating it here makes the HIP K5 path capture-safe.
from bench_chunk_gdn_fwd_h import (  # noqa: E402
    PRESET_SHAPES,
    _USE_EXP2,
    _CaptureSafeMeta,
    _hip_meta_cache,
    _make_inputs,
    _make_seqlens,
    _shape_label,
)

_BENCH_TITLE = "GDN K5+K6 fused forward"


# --------------------------------------------------------------------------- #
# TFLOPS — K5 + K6 combined
# --------------------------------------------------------------------------- #
def calculate_tflops(
    N, H, T_flat, K, V, time_us, BT=64, seq_lens=None
) -> float:
    """Total FLOPs = K5 (4·BT·K·V) + K6 (2·BT·K·V + 2·BT·BT·K + 2·BT·BT·V) per
    chunk per head. Chunks counted per sequence with ceil() (padded tail), same
    accounting as the K5 bench.
    """
    if time_us <= 0:
        return float("nan")
    lens = list(seq_lens) if seq_lens else [T_flat]
    n_chunks = sum(-(-length // BT) for length in lens)
    per_chunk = (
        4 * K * V              # K5: GEMM1 (w@h) + GEMM2 (k^T@v_new)
        + 2 * K * V            # K6 GEMM3: q@h
        + 2 * BT * K           # K6 GEMM4a: q@k^T
        + 2 * BT * V           # K6 GEMM4b: A@v_new
    )
    return H * n_chunks * BT * per_chunk / (time_us * 1e-6) / 1e12


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #
_TRITON_ONLY = os.environ.get("AITER_TRITON_ONLY", "0") == "1"

_IMPL_KEYS = (
    "K5_triton+K6_triton",
    "K5_hip+K6_triton",
    "K5_flydsl+K6_triton",
    "K5K6_flydsl_fused",
)


def _make_q(shape, device="cuda"):
    """Query tensor [B, T_flat, Hg, K] matching the k layout in _make_inputs."""
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    return torch.randn(1, T_flat, Hg, K, dtype=torch.bfloat16, device=device) * 0.1


def _separate_runner(k5_fn, k6_fn, is_hip=False):
    """Build a closure that runs a separate K5 then Triton K6, returning o.

    ``k5_fn`` must return ``(h, v_new, final_state)`` with v_new in head-major
    [B, H, T_flat, V] (both the Triton VK and FlyDSL K5 wrappers do).

    Under graph capture with a varlen batch, Triton K6 (``chunk_fwd_o_opt_vk``)
    would otherwise build chunk indices at launch (a device->host read that is
    illegal while capturing). We pass the reusable ``prefill_metadata`` built
    once in ``_run_one`` (looked up by ``cu`` identity, no sync) so the captured
    call only reuses it. The HIP K5 path gets its own copy via ``_adapt_hip``,
    which reads the same ``_hip_meta_cache``.
    """
    def _run(*, q, k, w, u, g, gk, h0, cu, scale, o):
        meta = _hip_meta_cache.get(id(cu)) if cu is not None else None
        h, v_new, _ = k5_fn(
            k=k, w=w, u=u, g=g, gk=gk, initial_state=h0,
            output_final_state=True, save_new_value=True,
            cu_seqlens=cu, use_exp2=_USE_EXP2,
        )
        k6_fn(
            q=q, k=k, v=v_new, o=o, h=h, g=g, scale=scale,
            cu_seqlens=cu, use_exp2=_USE_EXP2, prefill_metadata=meta,
        )
        return o
    return _run


def _load_impls(which: str) -> dict:
    requested = {s.strip() for s in which.split(",")} if which != "all" else None

    def _want(name):
        return requested is None or name in requested

    impls: dict = {}

    # Triton K6 is the shared output kernel for all separate paths.
    k6_triton = None
    try:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
            chunk_fwd_o_opt_vk,
        )
        k6_triton = chunk_fwd_o_opt_vk
    except ImportError as e:
        warnings.warn(f"Triton K6 not available: {e}")

    if _want("K5_triton+K6_triton") and k6_triton is not None:
        try:
            from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
                chunk_gated_delta_rule_fwd_h_opt_vk,
            )
            impls["K5_triton+K6_triton"] = _separate_runner(
                chunk_gated_delta_rule_fwd_h_opt_vk, k6_triton
            )
        except ImportError as e:
            warnings.warn(f"Triton K5 not available: {e}")

    if _want("K5_hip+K6_triton") and not _TRITON_ONLY and k6_triton is not None:
        try:
            from bench_chunk_gdn_fwd_h import _adapt_hip
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )
            impls["K5_hip+K6_triton"] = _separate_runner(
                _adapt_hip(chunk_gated_delta_rule_fwd_h_hip_fn), k6_triton
            )
        except ImportError as e:
            warnings.warn(f"HIP K5 not available: {e}")

    if _want("K5_flydsl+K6_triton") and not _TRITON_ONLY and k6_triton is not None:
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl,
            )
            impls["K5_flydsl+K6_triton"] = _separate_runner(
                chunk_gated_delta_rule_fwd_h_flydsl, k6_triton
            )
        except ImportError as e:
            warnings.warn(f"FlyDSL K5 not available: {e}")

    if _want("K5K6_flydsl_fused") and not _TRITON_ONLY:
        try:
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_o_flydsl,
            )

            def _fused_run(*, q, k, w, u, g, gk, h0, cu, scale, o,
                           _fn=chunk_gated_delta_rule_fwd_h_o_flydsl):
                meta = _hip_meta_cache.get(id(cu)) if cu is not None else None
                _fn(
                    q=q, k=k, w=w, u=u, g=g, gk=gk, scale=scale,
                    initial_state=h0, output_final_state=True,
                    cu_seqlens=cu, use_exp2=_USE_EXP2, o=o,
                    prefill_metadata=meta,
                )
                return o

            impls["K5K6_flydsl_fused"] = _fused_run
        except ImportError as e:
            warnings.warn(f"FlyDSL fused K5+K6 not available: {e}")

    return impls


# --------------------------------------------------------------------------- #
# Per-shape timing
# --------------------------------------------------------------------------- #
def _make_closure(fn, q, k, w_hm, u_hm, g, gk, h0, cu, scale, o):
    def _run():
        fn(q=q, k=k, w=w_hm, u=u_hm, g=g, gk=gk, h0=h0, cu=cu, scale=scale, o=o)
    return _run


def _run_one(idx, impls, shape, args, cfg) -> dict:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate_mode, seq_pattern = shape
    _seq_lens = _make_seqlens(seq_pattern, T_flat, N, BT)
    label = _shape_label(idx, shape)

    try:
        k, w_hm, u_hm, w_tm, g, gk, h0, cu = _make_inputs(shape)
        q = _make_q(shape)
    except Exception as e:
        return {"label": label, "error": str(e)}

    scale = K ** -0.5
    o = u_hm.new_empty(1, T_flat, H, V)  # [B, T, H, V]

    # Build reusable prefill metadata once (before any warmup/capture) for the
    # varlen path, so neither the HIP K5 adapter nor Triton K6 builds chunk
    # metadata / does a device->host read inside the graph-captured closure.
    # Keyed by cu identity in the shared _hip_meta_cache; read back by the HIP
    # adapter and by _separate_runner (for K6). Only needed when N>1.
    if cu is not None:
        try:
            from aiter.ops.prefill_batch_metadata import (
                build_gated_delta_rule_prefill_metadata,
            )
            _bounds = cu.detach().to("cpu", torch.int64)
            _sl = (_bounds[1:] - _bounds[:-1]).tolist()
            _meta = build_gated_delta_rule_prefill_metadata(
                _sl, cu_seqlens=cu, chunk_size=BT,
            )
            _hip_meta_cache[id(cu)] = _CaptureSafeMeta(
                _meta, cu, chunk_size=BT,
                total_prefill_tokens=int(T_flat), num_sequences=len(_sl),
            )
        except Exception as e:
            warnings.warn(f"prefill metadata build failed: {e}")

    modes = ["eager", "graph"] if args.mode == "all" else [args.mode]
    baseline_name = args.baseline
    baseline_times: dict = {}
    results_by_impl: dict = {}
    o_ref = None

    for impl_name, fn in impls.items():
        print(f"  {impl_name}...", end=" ", flush=True)
        closure = _make_closure(fn, q, k, w_hm, u_hm, g, gk, h0, cu, scale, o)
        try:
            closure()
            torch.cuda.synchronize()
        except NotImplementedError as e:
            print("NOT_IMPL")
            results_by_impl[impl_name] = {"error": f"NOT_IMPL: {e}"}
            continue
        except Exception as e:
            print(f"PROBE-FAIL: {e}")
            results_by_impl[impl_name] = {"error": str(e)}
            continue

        # Cross-impl verification: first successful impl is the reference.
        verify_str = "N/A"
        if args.verification != "none":
            o_now = o.detach().clone()
            if o_ref is None:
                o_ref = o_now
                verify_str = "REF"
            else:
                diff = (o_now.float() - o_ref.float()).pow(2).mean().sqrt()
                denom = o_ref.float().pow(2).mean().sqrt() + 1e-8
                ratio = (diff / denom).item()
                verify_str = f"{'PASS' if ratio < 5e-2 else 'FAIL'}(rmse={ratio:.2e})"

        timing: dict = {}
        tflops_d: dict = {}
        for mode in modes:
            try:
                stats = _time_measure(closure, mode, cfg)
                timing[mode] = stats
                tflops_d[mode] = calculate_tflops(
                    N, H, T_flat, K, V, stats.median_us, BT, seq_lens=_seq_lens
                )
            except EmptyGraphCaptureError as e:
                timing[mode] = f"GRAPH-FAIL: {e}"
                tflops_d[mode] = None
            except Exception as e:
                timing[mode] = f"ERROR: {e}"
                tflops_d[mode] = None

        results_by_impl[impl_name] = {
            "timing": timing, "tflops": tflops_d, "verify": verify_str,
        }
        if impl_name == baseline_name:
            for mode in modes:
                t = timing.get(mode)
                if hasattr(t, "median_us"):
                    baseline_times[mode] = t.median_us

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
        "label": label, "shape": shape, "impls": results_by_impl,
        "baseline_times": baseline_times, "baseline_name": baseline_name,
        "modes": modes, "N": N, "H": H, "T_flat": T_flat, "K": K, "V": V,
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

    n_shapes = len(PRESET_SHAPES)
    if args.shape_index is not None:
        idx = args.shape_index
        if not (1 <= idx <= n_shapes):
            print(f"--shape-index must be 1–{n_shapes}", file=sys.stderr)
            sys.exit(1)
        shapes_to_run = [(idx, PRESET_SHAPES[idx - 1])]
    elif getattr(args, "shape_range", None) is not None:
        lo, hi = args.shape_range
        if not (1 <= lo <= hi <= n_shapes):
            print(
                f"--shape-range must satisfy 1 <= START <= END <= {n_shapes}",
                file=sys.stderr,
            )
            sys.exit(1)
        shapes_to_run = [(i, PRESET_SHAPES[i - 1]) for i in range(lo, hi + 1)]
    else:
        gate_filter = getattr(args, "gate", "all")
        shapes_to_run = [
            (i + 1, s) for i, s in enumerate(PRESET_SHAPES)
            if gate_filter == "all" or s[-1] == gate_filter
        ]

    all_rows = []
    for idx, shape in shapes_to_run:
        print(f"\n{'='*60}\n{_shape_label(idx, shape)}\n{'='*60}")
        row = _run_one(idx, impls, shape, args, cfg)
        print_result_table(row)
        all_rows.append(row)

    if args.output:
        env = collect_env_info()
        write_bench_markdown(
            args.output, _BENCH_TITLE, all_rows, env, baseline_name=args.baseline,
        )
        print(f"\nMarkdown report written to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="A/B benchmark: GDN K5+K6 fused forward."
    )
    def _shape_range(s: str) -> tuple[int, int]:
        """Parse ``START-END`` (1-based, inclusive) or a single ``N`` into (lo, hi)."""
        parts = s.split("-")
        try:
            if len(parts) == 1:
                v = int(parts[0]); return (v, v)
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
        raise argparse.ArgumentTypeError(
            f"--shape-range expects 'START-END' or 'N' (1-based); got {s!r}"
        )

    shape_grp = parser.add_mutually_exclusive_group()
    shape_grp.add_argument("--shape-index", type=int, default=None, metavar="N",
                           help="Run one preset shape by 1-based index (see --list).")
    shape_grp.add_argument("--shape-range", type=_shape_range, default=None,
                           metavar="START-END",
                           help="Run a 1-based inclusive range of preset shapes, "
                                "e.g. '5-8' (or a single 'N').")
    shape_grp.add_argument("--list", action="store_true",
                           help="List all preset shapes with indices and exit.")
    parser.add_argument("--impl", default="all", metavar="IMPLS",
                        help=f"Comma-separated impls: {', '.join(_IMPL_KEYS)}. Default: all.")
    parser.add_argument("--baseline", default="K5_flydsl+K6_triton", choices=_IMPL_KEYS,
                        help="Speedup baseline (default: K5_flydsl+K6_triton).")
    parser.add_argument("--gate", default="all", choices=("g", "gk", "all"),
                        help="Filter shapes by gate type.")
    add_timing_args(parser)
    add_output_args(parser)
    add_verification_args(parser)
    args = parser.parse_args()

    if args.list:
        print(f"{'#':>3}  {'label':<60}  gate")
        print("-" * 75)
        for i, s in enumerate(PRESET_SHAPES, 1):
            model_tag, H, Hg, T_flat, N, K, V, BT, gate, seq_pattern = s
            _sp = "" if seq_pattern == "equal" else f"  seqs={seq_pattern}"
            label = f"{model_tag}  H={H} Hg={Hg} T={T_flat} N={N} K={K} V={V}{_sp}"
            print(f"{i:>3}  {label:<60}  {gate}")
        return

    run(args)


if __name__ == "__main__":
    main()
