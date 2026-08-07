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
import functools
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
from utils.plot_perf import (
    category_label,
    make_bar_chart,
    make_summary_md,
    parse_bench_md,
)

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

# The bench builds gates (``g``/``gk``) in the natural-log domain and the fp32
# reference applies ``torch.exp``. The K5 wrapper's ``use_exp2=True`` path expects
# the scalar ``g`` already pre-scaled to log2 by an upstream K1+K2 producer (it does
# NOT rescale ``g`` itself), which would double-apply log2(e) here and corrupt the
# scalar-gate result. ``use_exp2=False`` selects the kernel's ``exp2(x*log2(e)) ==
# exp(x)`` lowering, matching the reference for both the ``g`` and ``gk`` paths.
_USE_EXP2 = False


def _shape_label(idx: int, shape: tuple) -> str:
    model_tag, H, Hg, T_flat, N, K, V, BT, gate = shape
    return f"Shape {idx}: {model_tag} H={H} Hg={Hg} T={T_flat} N={N} gate={gate}"


# --------------------------------------------------------------------------- #
# Input tensor construction
# --------------------------------------------------------------------------- #
def _make_inputs(shape: tuple, device="cuda"):
    """Build all K5 input tensors for the given shape tuple.

    ``N`` is the number of variable-length sequences packed into the flat
    ``T_flat`` token axis. When ``N > 1`` a ``cu_seqlens`` of ``N`` equal-length
    sequences is built and the varlen path is exercised (matching the pytest
    fixture); the SSM state ``h0`` is ``[N, H, V, K]`` (one state per sequence).

    Returns:
        k, w_hm, u_hm, w_tm, g, gk, initial_state, cu_seqlens
        where w_hm/u_hm are head-major (kernel input) and w_tm is token-major
        (reference input). ``cu_seqlens`` is ``None`` when ``N == 1``.
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

    # cu_seqlens: split T_flat into N equal sequences (last absorbs remainder).
    if N > 1:
        per = T_flat // N
        bounds = [0]
        for i in range(N):
            bounds.append(T_flat if i == N - 1 else bounds[-1] + per)
        cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    else:
        cu = None

    return k, w_hm, u_hm, w_tm, g, gk, h0, cu


# --------------------------------------------------------------------------- #
# Implementation registry
# --------------------------------------------------------------------------- #
_TRITON_ONLY = os.environ.get("AITER_TRITON_ONLY", "0") == "1"

# FlyDSL kernel variants get one row each, keyed ``flydsl:<tag>``. The K5 kernel's
# only compile-time tuning axis is BV (the V-tile width) -- BT=64 is fixed by the
# K1-K3 pipeline -- so the tags are ``bv16``/``bv32``/``bv64`` plus the special
# ``auto``, which defers to the kernel's shape-adaptive heuristic per shape.
FLYDSL_PREFIX = "flydsl:"
AUTO_VARIANT = "auto"


def _available_variants():
    """``(tuple_of_tags, default_tag)``, or ``(None, None)`` if FlyDSL is absent."""
    if _TRITON_ONLY:
        return None, None
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            K5_DEFAULT_VARIANT,
            K5_VARIANTS,
        )
    except ImportError:
        return None, None
    return tuple(K5_VARIANTS), K5_DEFAULT_VARIANT


def _auto_variant_for_shape(shape, cu) -> str | None:
    """The tag the heuristic picks for this shape, for display purposes.

    Lets the table show ``flydsl:auto(bv64)`` instead of a bare ``auto``, so a
    sweep records what actually ran. Returns None if it cannot be determined.
    """
    _tag, H, Hg, T_flat, N, _K, V, _BT, _gate = shape
    try:
        from aiter.ops.flydsl.linear_attention_prefill_kernels import _auto_variant

        return _auto_variant(
            H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=cu is not None
        )
    except Exception:
        return None


def _load_impls(which: str, flydsl_variants: list[str] | None = None) -> dict:
    """Return {impl_name: callable} for the requested set.

    ``flydsl_variants`` is a list of variant tags (or the specials ``all`` /
    ``auto``); each becomes its own ``flydsl:<tag>`` entry. Defaults to ``auto``,
    which reproduces the historical single-row behaviour exactly.
    """
    requested = {s.strip() for s in which.split(",")} if which != "all" else None
    if flydsl_variants is None:
        flydsl_variants = [AUTO_VARIANT]

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
            available, _default = _available_variants()
            tags = list(flydsl_variants)
            if tags == ["all"]:
                tags = list(available or ())
            for tag in tags:
                if tag == AUTO_VARIANT:
                    # variant=None -> the kernel's own shape-adaptive selection.
                    impls[FLYDSL_PREFIX + AUTO_VARIANT] = (
                        chunk_gated_delta_rule_fwd_h_flydsl
                    )
                    continue
                if available is not None and tag not in available:
                    raise SystemExit(
                        f"[error] unknown FlyDSL variant {tag!r}; available: "
                        f"{list(available)} (or {AUTO_VARIANT!r})."
                    )
                impls[FLYDSL_PREFIX + tag] = functools.partial(
                    chunk_gated_delta_rule_fwd_h_flydsl, variant=tag
                )
        except ImportError as e:
            warnings.warn(f"FlyDSL K5 not available: {e}")

    # HIP is available on both gfx942 and gfx950 (disabled only on gfx12).
    if _want("hip") and not _TRITON_ONLY:
        try:
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )
            impls["hip"] = _adapt_hip(chunk_gated_delta_rule_fwd_h_hip_fn)
        except ImportError as e:
            warnings.warn(f"HIP K5 not available: {e}")

    return impls


def _adapt_hip(hip_fn):
    """Wrap the HIP K5 wrapper so it accepts the same call shape as flydsl/triton.

    The bench closures call every impl uniformly as
    ``fn(k=, w=, u=, g=, gk=, initial_state=, output_final_state=,
        save_new_value=, cu_seqlens=, use_exp2=)`` with ``g`` in the FlyDSL/Triton
    2-D head-major ``[H, T_flat]`` layout. The production HIP wrapper
    (``chunk_gated_delta_rule_fwd_h_hip_fn``) instead requires a **3-D** ``g``
    (``[B, H, T]`` head-major or ``[B, T, H]`` token-major) and a ``g_head_major``
    flag. This adapter reshapes ``g`` to ``[1, H, T_flat]`` and sets
    ``g_head_major=True`` so the HIP kernel can be benchmarked without editing any
    production code. ``gk`` (``[T_flat, H, K]``) already matches HIP's expected
    layout and is forwarded unchanged.

    Graph capture: the HIP wrapper otherwise (a) does a device-to-host read of the
    chunk schedule during launch and (b) *builds* the prefill metadata inside the
    call -- both illegal under CUDA-graph capture ("operation not permitted when
    stream is capturing" / "GDR metadata cannot be built during ... capture"). The
    fix is to build a reusable ``GatedDeltaRulePrefillMetadata`` ONCE before capture
    (in ``_run_one``) and pass it as ``prefill_metadata=``; the captured call then
    only reuses it. The metadata is looked up here by ``cu_seqlens`` tensor identity
    (no device sync). With no ``cu_seqlens`` (single sequence) there is no schedule
    to build, so nothing extra is needed.
    """
    import functools

    @functools.wraps(hip_fn)
    def _wrapped(*, k, w, u, g=None, gk=None, cu_seqlens=None, **kwargs):
        g_hip = g
        if g is not None:
            # [H, T_flat] -> [1, H, T_flat] (batch=1, head-major).
            g_hip = g.unsqueeze(0) if g.dim() == 2 else g
        prefill_metadata = _hip_meta_cache.get(
            id(cu_seqlens) if cu_seqlens is not None else None
        )
        return hip_fn(
            k=k,
            w=w,
            u=u,
            g=g_hip,
            gk=gk,
            cu_seqlens=cu_seqlens,
            g_head_major=True,
            prefill_metadata=prefill_metadata,
            **kwargs,
        )

    return _wrapped


# Maps id(cu_seqlens tensor) -> reusable GatedDeltaRulePrefillMetadata, built once
# per shape in _run_one BEFORE warmup/capture so the HIP adapter never builds
# metadata (or does a device-to-host read) inside the timed / graph-captured
# closure.
_hip_meta_cache: dict = {}


class _CaptureSafeMeta:
    """Thin proxy over ``GatedDeltaRulePrefillMetadata`` that no-ops ``validate``.

    The production metadata's ``validate()`` raises unconditionally while a HIP/
    CUDA graph is capturing ("Typed prefill metadata cannot be used during ...
    capture"). Everything else the HIP wrapper touches on the metadata
    (``get_chunk_schedule`` -> returns precomputed on-device tensors) is
    capture-safe. The bench validates the metadata ONCE before capture (below),
    so skipping the redundant in-capture ``validate`` is sound and lets the HIP
    kernel be graph-captured. This is a benchmark-only shim; it does not alter any
    production code path (the real wrapper still runs identically outside capture).
    """

    def __init__(self, meta, cu_seqlens, *, chunk_size, total_prefill_tokens,
                 num_sequences):
        self._meta = meta
        # One-time, pre-capture validation using the real implementation
        # (keyword-only signature; see GatedDeltaRulePrefillMetadata.validate).
        meta.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            num_decodes=0,
            num_decode_tokens=0,
            total_prefill_tokens=total_prefill_tokens,
            num_sequences=num_sequences,
        )

    def validate(self, *args, **kwargs):
        return None  # already validated pre-capture

    def __getattr__(self, name):
        return getattr(self._meta, name)


# --------------------------------------------------------------------------- #
# Reference (same as the unit test suite)
# --------------------------------------------------------------------------- #
_ref_fn = None   # loaded lazily on first use


def _load_ref():
    """Load ``ref_chunk_gated_delta_rule_fwd_h`` from the unit test module.

    Import via the package path (``importlib.import_module``) rather than
    ``spec_from_file_location``: the test module has top-level
    ``pytest.skip(..., allow_module_level=True)`` guards that only trip inside a
    pytest session, so a normal import resolves the reference cleanly. Falls back
    to file-loading if the package import is unavailable (e.g. run from a tree
    where ``op_tests`` is not importable).
    """
    global _ref_fn
    if _ref_fn is not None:
        return _ref_fn
    try:
        mod = importlib.import_module(
            "op_tests.flydsl_tests.test_flydsl_linear_attention_prefill"
        )
    except Exception:
        test_path = (
            Path(_REPO_ROOT)
            / "op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py"
        )
        spec = importlib.util.spec_from_file_location("_test_prefill", test_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    _ref_fn = mod.ref_chunk_gated_delta_rule_fwd_h
    return _ref_fn


# --------------------------------------------------------------------------- #
# TFLOPS
# --------------------------------------------------------------------------- #
def calculate_tflops(
    N: int, H: int, T_flat: int, K: int, V: int, time_us: float, BT: int = 64
) -> float:
    """Total FLOPs for K5 = GEMM1 (w @ h^T) + GEMM2 (k^T @ v_new).

    Both GEMMs cost 2·BT·K·V per chunk per head, so 4·BT·K·V for the pair.

    ``T_flat`` is the TOTAL token count across all ``N`` sequences -- they are
    packed into a single flat token axis by ``_make_inputs`` (every tensor is
    ``[B=1, T_flat, ...]`` and ``cu_seqlens`` merely splits that axis).

    Chunks are counted per sequence with ceil(), because the kernel pads each
    sequence's final chunk out to BT and issues the full BT-wide MFMA work for
    it. That matches what the hardware actually executes: this formula
    reproduces SQ_INSTS_MFMA x 8192 exactly on the preset shapes.
    """
    if time_us <= 0:
        return float("nan")
    # Mirrors the cu_seqlens split in _make_inputs: N-1 sequences of T_flat//N,
    # with the last absorbing the remainder.
    per = T_flat // N
    lens = [per] * (N - 1) + [T_flat - per * (N - 1)]
    n_chunks = sum(-(-length // BT) for length in lens)
    return 4 * H * n_chunks * BT * K * V / (time_us * 1e-6) / 1e12


# --------------------------------------------------------------------------- #
# Per-shape timing
# --------------------------------------------------------------------------- #
def _make_closure(fn, k, w_hm, u_hm, g, gk, h0, cu):
    def _run():
        fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
           initial_state=h0, output_final_state=True, save_new_value=True,
           cu_seqlens=cu, use_exp2=_USE_EXP2)
    return _run


def _verify_impl(fn, k, w_hm, u_hm, w_tm, g, gk, h0, cu, verification: str,
                 baseline_fn=None) -> str:
    """Run correctness check. Returns a grade string."""
    if verification == "none":
        return "N/A"
    try:
        h, v_new, fs = fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
                          initial_state=h0, output_final_state=True,
                          save_new_value=True, cu_seqlens=cu, use_exp2=_USE_EXP2)
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
                               output_final_state=True, cu_seqlens=cu)
        except Exception as e:
            return f"REF-ERROR({e})"
        ratio = _rmse_ratio(h, h_ref)
        return f"{'PASS' if ratio < 0.05 else 'FAIL'}(rmse_ratio={ratio:.2e})"

    if verification == "baseline" and baseline_fn is not None:
        try:
            h_base, _, _ = baseline_fn(k=k, w=w_hm, u=u_hm, g=g, gk=gk,
                                        initial_state=h0, output_final_state=True,
                                        save_new_value=False, cu_seqlens=cu,
                                        use_exp2=_USE_EXP2)
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
        k, w_hm, u_hm, w_tm, g, gk, h0, cu = _make_inputs(shape)
    except Exception as e:
        return {"label": label, "error": str(e)}

    # Build reusable HIP prefill metadata ONCE (before any warmup/capture) so the
    # HIP adapter never builds metadata or does a device-to-host read inside the
    # timed / graph-captured closure. Only needed for the varlen (N>1) path and
    # only when HIP is among the impls.
    if cu is not None and "hip" in impls:
        try:
            from aiter.ops.prefill_batch_metadata import (
                build_gated_delta_rule_prefill_metadata,
            )
            _bounds = cu.detach().to("cpu", torch.int64)
            _seq_lens = (_bounds[1:] - _bounds[:-1]).tolist()
            _meta = build_gated_delta_rule_prefill_metadata(
                _seq_lens, cu_seqlens=cu, chunk_size=BT,
            )
            _hip_meta_cache[id(cu)] = _CaptureSafeMeta(
                _meta, cu,
                chunk_size=BT,
                total_prefill_tokens=int(T_flat),
                num_sequences=len(_seq_lens),
            )
        except Exception as e:
            warnings.warn(f"HIP prefill metadata build failed: {e}")

    modes = ["eager", "graph"] if args.mode == "all" else [args.mode]
    baseline_name = args.baseline
    baseline_fn = impls.get(baseline_name)

    baseline_times: dict[str, float] = {}
    results_by_impl: dict = {}

    # Report the auto row under the concrete variant the heuristic picked for
    # THIS shape -- i.e. "flydsl:bv32", exactly the label an explicit
    # --flydsl-variants run produces, so rows are directly comparable across
    # runs. Keys may differ per shape; both output tables iterate row["impls"]
    # per row, so that is fine.
    auto_key = FLYDSL_PREFIX + AUTO_VARIANT
    auto_label = None
    if auto_key in impls:
        resolved = _auto_variant_for_shape(shape, cu)
        if resolved:
            auto_label = FLYDSL_PREFIX + resolved
            if auto_label in impls:
                # The same variant was ALSO requested explicitly; it is the same
                # kernel, so drop the duplicate rather than benchmark it twice
                # (and rather than let one row silently overwrite the other).
                print(f"  [skip] {auto_key} resolves to {auto_label}, already requested")
                impls = {k: v for k, v in impls.items() if k != auto_key}

    for impl_name, fn in impls.items():
        if impl_name == auto_key and auto_label:
            impl_name = auto_label
        print(f"  {impl_name}...", end=" ", flush=True)
        closure = _make_closure(fn, k, w_hm, u_hm, g, gk, h0, cu)

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
                tflops_d[mode] = calculate_tflops(
                    N, H, T_flat, K, V, stats.median_us, BT
                )
            except EmptyGraphCaptureError as e:
                timing[mode] = f"GRAPH-FAIL: {e}"
                tflops_d[mode] = None
            except Exception as e:
                timing[mode] = f"ERROR: {e}"
                tflops_d[mode] = None

        verify_str = "N/A"
        if args.verification != "none":
            verify_str = _verify_impl(
                fn, k, w_hm, u_hm, w_tm, g, gk, h0, cu,
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
    impls = _load_impls(args.impl, getattr(args, "flydsl_variants", None))
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
                           mode=modes_plot[0], baseline_label=category_label(args.baseline))
            summary_md = str(out_dir / f"{stem}-summary.md")
            make_summary_md(results, summary_md, png_path, args.output,
                            title=_BENCH_TITLE, mode=modes_plot[0],
                            baseline_label=category_label(args.baseline))
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
        "--flydsl-variants", type=str, default=AUTO_VARIANT, metavar="V1,V2,...",
        help=(
            "Comma-separated FlyDSL kernel-variant tags to benchmark, each as its "
            "own row (see --list-variants). 'all' runs every registered variant; "
            "'auto' (default) defers to the kernel's shape-adaptive selection, "
            "picking one variant per shape."
        ),
    )
    parser.add_argument(
        "--list-variants", action="store_true",
        help="List the registered FlyDSL kernel variants and exit.",
    )
    parser.add_argument(
        "--gate", default="all", choices=("g", "gk", "all"),
        help="Filter shapes by gate type: g (GDN), gk (KDA), all (default).",
    )

    add_timing_args(parser)
    add_output_args(parser)
    add_verification_args(parser)

    args = parser.parse_args()

    if args.list_variants:
        avail, default = _available_variants()
        if avail is None:
            print("FlyDSL is unavailable; no variants to list.")
        else:
            print("FlyDSL GDN K5 kernel variants (BV = V-tile width):")
            for v in avail:
                print(f"    {v}")
            print(
                f"  {'*' if default == AUTO_VARIANT else ' '} {AUTO_VARIANT}"
                "  (shape-adaptive: picks a variant per shape via _heuristic_bv)"
            )
        return

    # Resolve the requested FlyDSL variants (comma list). Validation against what
    # is actually registered happens in _load_impls, so a --impl triton run still
    # works when FlyDSL is unavailable.
    args.flydsl_variants = [
        v.strip() for v in args.flydsl_variants.split(",") if v.strip()
    ] or [AUTO_VARIANT]

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
