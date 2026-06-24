# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B performance harness: FlyDSL vs Triton FP8 MQA logits (gfx942).

Times Triton plus one or more FlyDSL kernel *variants* (kernel versions) on
identical inputs and prints one table per shape, with each variant as a row
(columns: time_us, TFLOPs, verify, vs_triton ratio).

The FlyDSL kernel ships several versions, registered in
``aiter.ops.flydsl.kernels.fp8_mqa_logits`` (``KERNEL_VARIANTS``): ``"mfma"`` is
the baseline, ``"scalar"`` the correctness-first fallback, and new versions can
be added there. Pick which to benchmark with ``--flydsl-variants`` (default:
all registered variants). Each becomes its own row, e.g. ``fly:mfma``.

Operand-dtype control (``--dtype-combo``):

  * ``fnuz/fnuz`` (default) -- same-type fp8 on both Q and K
  * ``fn/fnuz`` -- the live DeepSeek-V4 indexer combo (Q=e4m3fn, K=e4m3fnuz).
  * ``all`` -- run both combos side by side.

Examples:
    # default DeepSeek-ish shape, baseline FlyDSL variant vs Triton
    python op_tests/op_benchmarks/triton/bench_flydsl_vs_triton_fp8_mqa_logits.py

    # compare two FlyDSL variants against Triton on a custom shape, with parity
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py \
        --flydsl-variants mfma,scalar \
        --seq_q_l 1024 --seq_kv_l 4096 --num_heads_q 64 --head_dim 128 \
        --verification reference

    # verify FlyDSL against Triton output directly (lighter, no torch ref OOM)
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py \
        --verification triton --dtype-combo all

    # DeepSeek-V4 combo
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py --dtype-combo fn/fnuz

    # both combos side by side
    python .../bench_flydsl_vs_triton_fp8_mqa_logits.py --dtype-combo all
"""
import argparse
import functools
import os
import subprocess
import sys
from datetime import datetime, timezone

# Make the script runnable directly (python path/to/bench.py) by putting the
# repo root on sys.path, so `import aiter` / `import op_tests` resolve without
# requiring PYTHONPATH or an installed aiter package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.ops.triton._triton_kernels.attention.fp8_mqa_logits import (
    _fp8_mqa_logits_kernel,
)
from aiter.ops.triton.utils.types import e4m3_dtype
from triton.runtime.errors import OutOfResources as TritonOutOfResources
from aiter.ops.flydsl import is_flydsl_available

# Import the standalone GEAK v4 kernel.
_GEAK_V4_PATH = os.path.join(
    _REPO_ROOT,
    "aiter", "ops", "flydsl", "kernels", "fp8_mqa_logits_flydsl_geak_v4.py",
)
_geak_v4_mod = None
if os.path.exists(_GEAK_V4_PATH):
    import importlib.util as _imputil
    _spec = _imputil.spec_from_file_location("fp8_mqa_logits_flydsl_geak_v4", _GEAK_V4_PATH)
    _geak_v4_mod = _imputil.module_from_spec(_spec)
    try:
        _spec.loader.exec_module(_geak_v4_mod)
    except Exception as _e:
        print(f"[warn] failed to load geak_v4 kernel: {_e}", file=sys.stderr)
        _geak_v4_mod = None

def _triton_logits_fallback(Q, KV, kv_scales, weights, cu_starts, cu_ends,
                            clean_logits=True, block_kv=64):
    """Direct-launch Triton kernel with reduced BLOCK_KV for mixed-dtype combos.

    When Q and KV have different fp8 types, Triton's tl.dot() emulation path
    requires extra LDS for operand upcast buffers.  With the standard
    BLOCK_KV=128 this exceeds the 64 KiB shared-memory limit on MI300X/MI325X.
    Falling back to BLOCK_KV=64 (+ num_stages=1) halves the LDS requirement and
    lets the kernel compile.
    """
    seq_len, num_heads, head_size = Q.shape
    seq_len_kv = KV.shape[0]
    aligned_size = 256
    seq_len_kv_aligned = (seq_len_kv + aligned_size - 1) // aligned_size * aligned_size
    if clean_logits:
        logits = torch.full(
            (seq_len, seq_len_kv_aligned),
            fill_value=-float("inf"),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]
    else:
        logits = torch.empty(
            (seq_len, seq_len_kv_aligned),
            dtype=torch.float32,
            device=Q.device,
        )[:, :seq_len_kv]

    matrix_instr_nonkdim = 16 if seq_len <= 1024 else 32

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=Q,
        KV_ptr=KV,
        kv_scales_ptr=kv_scales,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=Q.stride(0),
        stride_q_h=Q.stride(1),
        stride_q_d=Q.stride(2),
        stride_kv_s=KV.stride(0),
        stride_kv_d=KV.stride(1),
        stride_w_s=weights.stride(0),
        stride_w_h=weights.stride(1),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        BLOCK_KV=block_kv,
        num_warps=4,
        num_stages=1,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )
    return logits


# FP8 dtype combos for benchmarking. On gfx942 the native MFMA format is FNUZ;
# the DeepSeek-V4 indexer uses q=FN, k=FNUZ ("fn/fnuz"). 
DTYPE_COMBOS = {
    "fnuz/fnuz": (e4m3_dtype, e4m3_dtype),
    "fn/fnuz": (torch.float8_e4m3fn, e4m3_dtype),
}
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
)
from op_tests.op_benchmarks.triton._bench_timing import (
    EmptyGraphCaptureError,
    MeasureConfig,
    measure as _measure,
)

# Timing strategies the bench can report. "eager" = per-call latency (device +
# host bubble); "graph" = HIP graph-replay steady state (host overhead stripped).
TIMING_MODES = ("eager", "graph")


def calculate_tflops(start_inds, end_inds, num_heads_q, head_dim, time_ms):
    time_s = time_ms * 1e-3
    start_inds = start_inds.to("cpu").numpy()
    end_inds = end_inds.to("cpu").numpy()
    total_flops = 0.0
    for i in range(len(start_inds)):
        total_flops += 2.0 * num_heads_q * head_dim * max(0, end_inds[i] - start_inds[i])
    return total_flops / (time_s * 1e12)


def _make_inputs(batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim,
                  q_fp8_dtype=None, kv_fp8_dtype=None):
    s_q = batch_size * seq_q_l
    s_k = batch_size * seq_kv_l

    if q_fp8_dtype is None:
        q_fp8_dtype = e4m3_dtype
    if kv_fp8_dtype is None:
        kv_fp8_dtype = e4m3_dtype

    q = torch.randn(s_q, num_heads_q, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, head_dim, device="cuda", dtype=torch.bfloat16)
    # Round-trip kv through fp8 so the bf16 `kv` used by the torch reference
    # matches what the kernels actually consume (mirrors the unit test).
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    kv = (kv_fp8.to(torch.float32) * scales.reshape(-1, 1)).to(torch.bfloat16)
    weights = torch.randn(s_q, num_heads_q, device="cuda", dtype=torch.float32)

    ks = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ke = torch.zeros(s_q, dtype=torch.int, device="cuda")
    arange_q = torch.arange(seq_q_l, dtype=torch.int, device="cuda")
    for b in range(batch_size):
        qs = b * seq_q_l
        kvs = b * seq_kv_l
        ks[qs : qs + seq_q_l] = kvs
        ke[qs : qs + seq_q_l] = kvs + (seq_kv_l - seq_q_l) + arange_q + 1

    q_fp8 = q.to(q_fp8_dtype)
    # KV scaling always uses the arch-native dtype (FNUZ on gfx942); the
    # kv_fp8_dtype parameter is accepted for symmetry but
    # per_custom_dims_cast_to_fp8 already produces the right type.
    kv_fp8, scales = per_custom_dims_cast_to_fp8(kv, (0,), False)
    if kv_fp8.dtype != kv_fp8_dtype:
        kv_fp8 = kv_fp8.view(kv_fp8_dtype)
    # `q`/`kv` (bf16) are returned for the torch reference; the rest feed the
    # actual kernels.
    return q, kv, q_fp8, kv_fp8, scales, weights, ks, ke


def _time_modes(closure, modes, cfg, quiet_probe=False):
    """Time ``closure`` under each requested strategy.

    Returns ``{mode: result}`` where each result is a ``TimingStats``, ``None``
    (implementation not ready / NotImplementedError), or the string
    ``"PROBE-FAIL"`` (graph capture failed -- e.g. a NULL-stream launcher or
    allocation during capture). One eager probe call first surfaces
    NotImplementedError / compile errors before timing.
    """
    try:
        closure()  # probe: surface NotImplementedError / compile errors
    except NotImplementedError:
        return {m: None for m in modes}
    except (RuntimeError, torch.OutOfMemoryError, TritonOutOfResources) as e:
        # Triton's mixed-dtype fp8 codegen can blow shared-memory limits
        # (OutOfResources) or otherwise fail to compile. Report as PROBE-FAIL
        # rather than crashing the whole sweep.
        if not quiet_probe:
            print(f"[warn] probe call failed: {e}")
        return {m: "PROBE-FAIL" for m in modes}
    torch.cuda.synchronize()

    out = {}
    for mode in modes:
        try:
            out[mode] = _measure(closure, mode, cfg)
        except EmptyGraphCaptureError:
            print("[warn] graph capture failed (e.g. NULL-stream launch or allocation during capture)")
            out[mode] = "GRAPH-FAIL"
        except (RuntimeError, torch.OutOfMemoryError) as e:
            # Allocation-during-capture and other capture-time failures surface
            # as RuntimeError; treat as a graph-capture failure for that cell
            # rather than aborting the whole sweep.
            print(f"[warn] {mode} timing failed: {e}")
            if mode == "graph":
                out[mode] = "GRAPH-FAIL"
            else:
                raise e
    return out


def _median_ms(result):
    """Median latency in ms from a _time_modes result, or None if not timed."""
    if result is None or isinstance(result, str):
        return None
    return result.median_us * 1e-3


# Perf-oriented preset sweep (DeepSeek-style square + rectangular shapes).
# Each entry: (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim).
# Covers both head counts the ticket calls out (H in {64, 128}) at D=128,
# square + rectangular windows, plus a non-aligned s_kv "tail" shape.
#
# Caveats for the long-ISL shapes:
#   * The torch reference OOMs (it materializes a dense [s_q, s_kv, H] tensor:
#     256 GiB at 32k), so --verify automatically falls back to grading against
#     the Triton kernel (itself validated vs torch where torch fits). Those
#     grades are suffixed "*" (e.g. PASS*) and Triton is shown as "REF".
#   * The geak variant indexes with int32 and writes a dense logits buffer, so it
#     is limited to outputs with < 2^31 elements (~32k square is the practical
#     ceiling); larger square shapes overflow FlyDSL's argument packing. In real
#     serving the dense logits are never materialized at once.
PRESET_SHAPES = [
    # H = 64
    (1, 1024, 1024, 64, 128),
    # (1, 1024, 1000, 64, 128),
    # (1, 1000, 1024, 64, 128),
    (1, 2048, 2048, 64, 128),
    (1, 4096, 4096, 64, 128),
    (1, 1024, 4096, 64, 128),
    (1, 4096, 1024, 64, 128),
    #(1, 4096, 16384, 64, 128),
    #(1, 4096, 16000, 64, 128),  # non-aligned s_kv (tail)
    # H = 64, long-ISL prefill (square; run without --verify, see note above)
    (1, 4096, 8192, 64, 128),
    (1, 8192, 4096, 64, 128),
    (1, 8192, 8192, 64, 128),
    #(1, 16384, 16384, 64, 128),
    #(1, 32768, 32768, 64, 128),  # ~32k: practical ceiling for the geak variant
    # H = 128
    # (1, 1024, 1024, 128, 128),
    # (1, 2048, 2048, 128, 128),
    # (1, 4096, 4096, 128, 128),
    # (1, 1024, 4096, 128, 128),
    # (1, 4096, 16384, 128, 128),
    # (1, 4096, 16000, 128, 128),  # non-aligned s_kv (tail)
    # H = 128, long-ISL prefill (mirrors the H=64 long-ISL group above)
    # (1, 8192, 8192, 128, 128),
    # (1, 16384, 16384, 128, 128),
    # (1, 32768, 32768, 128, 128),
]


FLYDSL_PREFIX = "flydsl:"
GEAK_IMPL_NAME = "geak_v4"


def _is_flydsl_impl(name):
    return name.startswith(FLYDSL_PREFIX) or name == GEAK_IMPL_NAME


def _flydsl_variant_of(name):
    """Return the variant tag for a 'flydsl:<variant>' impl name."""
    return name[len(FLYDSL_PREFIX):]


def _impl_label(name):
    """Short, column-friendly label for an impl name (e.g. 'triton', 'fly:mfma')."""
    if name == "triton":
        return "triton"
    if name == GEAK_IMPL_NAME:
        return "flydsl:" + GEAK_IMPL_NAME
    if name.startswith(FLYDSL_PREFIX):
        return "flydsl:" + _flydsl_variant_of(name)
    return name


def _available_variants():
    """Return ``(tuple_of_variant_tags, default_tag)`` or ``(None, None)``.

    ``None`` when FlyDSL isn't importable (so the bench can still run Triton-only).
    The standalone "geak_v4" variant is appended if the v4 module loaded.
    """
    if not is_flydsl_available():
        return None, None
    from aiter.ops.flydsl import (
        FP8_MQA_LOGITS_VARIANTS,
        FP8_MQA_LOGITS_DEFAULT_VARIANT,
    )

    variants = list(FP8_MQA_LOGITS_VARIANTS)
    if _geak_v4_mod is not None:
        variants.append("geak_v4")
    return tuple(variants), FP8_MQA_LOGITS_DEFAULT_VARIANT


def _select_impls(which, flydsl_variants):
    """Return an ordered ``{impl_name: fn}`` for the requested selection.

    ``which`` is the impl-family selection ('all'/'triton'/'flydsl').
    ``flydsl_variants`` is the resolved list of FlyDSL kernel-version tags to
    benchmark; each becomes its own impl named ``flydsl:<variant>`` bound to that
    variant via ``functools.partial``. The special ``"geak"`` tag maps to the
    standalone v4 module (imported at the top of this file).
    """
    flydsl_fn = None
    available_variants = ()
    if is_flydsl_available():
        from aiter.ops.flydsl import (
            flydsl_fp8_mqa_logits,
            FP8_MQA_LOGITS_VARIANTS,
        )

        flydsl_fn = flydsl_fp8_mqa_logits
        available_variants = list(FP8_MQA_LOGITS_VARIANTS)
        if _geak_v4_mod is not None:
            available_variants.append("geak_v4")
        available_variants = tuple(available_variants)

    want_triton = which in ("all", "triton")
    want_flydsl = which in ("all", "flydsl")

    if want_flydsl and flydsl_fn is None:
        if which == "flydsl":
            raise SystemExit(
                "[error] --impl flydsl requested but flydsl is unavailable."
            )
        print("[warn] flydsl unavailable -- benchmarking Triton only.")
        want_flydsl = False

    impls = {}
    if want_triton:
        impls["triton"] = triton_logits
    if want_flydsl:
        for v in flydsl_variants:
            if v == "geak_v4":
                if _geak_v4_mod is None:
                    raise SystemExit(
                        f"[error] geak variant requested but {_GEAK_V4_PATH} "
                        f"failed to load."
                    )
                impls[GEAK_IMPL_NAME] = _geak_v4_mod.flydsl_fp8_mqa_logits
                continue
            if v not in available_variants:
                raise SystemExit(
                    f"[error] unknown FlyDSL variant {v!r}; available: "
                    f"{list(available_variants)}."
                )
            impls[FLYDSL_PREFIX + v] = functools.partial(flydsl_fn, variant=v)

    if not impls:
        raise SystemExit("[error] no implementations selected.")
    return impls


def run(args):
    impls = _select_impls(args.impl, args.flydsl_variants)
    shapes = _resolve_shapes(args)

    rows = []
    total = len(shapes) * len(args.dtype_combos)
    n = 0
    for idx, (batch_size, seq_q_l, seq_kv_l, num_heads_q, head_dim) in shapes:
        for combo_tag in args.dtype_combos:
            n += 1
            q_dt, kv_dt = DTYPE_COMBOS[combo_tag]
            shape = argparse.Namespace(
                batch_size=batch_size,
                seq_q_l=seq_q_l,
                seq_kv_l=seq_kv_l,
                num_heads_q=num_heads_q,
                head_dim=head_dim,
                clean_logits=args.clean_logits,
                dtype_combo=combo_tag,
            )
            # Progress to stderr so it doesn't interleave with the final table.
            print(
                f"[{n}/{total}] {_fmt_shape(shape)} [{combo_tag}]",
                file=sys.stderr, flush=True,
            )
            rows.append(_run_one(idx, impls, shape, args, q_dt, kv_dt))

    _print_table(rows, impls, args)

    if args.output:
        _write_markdown(args.output, rows, impls, args)
        print(f"\n[wrote] {args.output}", file=sys.stderr, flush=True)


def _make_closure(name, fn, shape, q_fp8, kv_fp8, scales, weights, ks, ke):
    """Build a timed closure for one impl.

    The geak v4 standalone kernel has a 6-arg ABI (no ``clean_logits``); all
    other impls (triton, flydsl:* variants) take the standard 7 args.
    """
    if name == GEAK_IMPL_NAME:
        # Geak v4 has a 6-arg ABI (no clean_logits); it always prefills with -inf.
        def closure():
            fn(q_fp8, kv_fp8, scales, weights, ks, ke)
        return closure

    # All other impls (triton, flydsl:* variants) take 7 args.
    def closure():
        fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
    return closure


def _run_one(idx, impls, shape, args, q_fp8_dtype=None, kv_fp8_dtype=None):
    """Time + (optionally) verify one case; return a row dict for the table."""
    q, kv, q_fp8, kv_fp8, scales, weights, ks, ke = _make_inputs(
        shape.batch_size, shape.seq_q_l, shape.seq_kv_l, shape.num_heads_q,
        shape.head_dim, q_fp8_dtype=q_fp8_dtype, kv_fp8_dtype=kv_fp8_dtype,
    )

    cfg_base = dict(
        warmup_iters=args.warmup,
        bench_iters=args.bench_iters,
        graph_replay_iters=args.graph_replay_iters,
    )

    # times[name][mode] -> TimingStats | None | "GRAPH-FAIL";
    # tflops[name][mode] -> float | None.
    # triton_label tracks the actual Triton variant used (for the table footer).
    times, tflops = {}, {}
    triton_label = None
    for name, fn in impls.items():
        closure = _make_closure(
            name, fn, shape, q_fp8, kv_fp8, scales, weights, ks, ke
        )
        # For Triton on mixed-dtype combos, suppress the probe warn on first
        # attempt since we'll retry with reduced BLOCK_KV if it fails.
        maybe_retry = (name == "triton"
                       and getattr(shape, "dtype_combo", "fnuz/fnuz") != "fnuz/fnuz")
        res = _time_modes(
            closure, args.modes, MeasureConfig(**cfg_base),
            quiet_probe=maybe_retry,
        )
        # If Triton's standard launcher hit OutOfResources (mixed-dtype LDS
        # overflow), automatically retry with a reduced BLOCK_KV=64.
        if (name == "triton"
                and all(v == "PROBE-FAIL" for v in res.values())
                and getattr(shape, "dtype_combo", "fnuz/fnuz") != "fnuz/fnuz"):
            print(
                "[info] Triton BLOCK_KV=128 failed (LDS overflow on mixed-dtype "
                "combo); retrying with BLOCK_KV=64",
                file=sys.stderr, flush=True,
            )
            triton_label = "triton(bkv64)"
            fb_fn = functools.partial(_triton_logits_fallback, block_kv=64)
            closure = _make_closure(
                name, fb_fn, shape, q_fp8, kv_fp8, scales, weights, ks, ke
            )
            res = _time_modes(
                closure, args.modes, MeasureConfig(**cfg_base),
            )
            # Swap the impl fn so _verify also uses the fallback.
            impls[name] = fb_fn
        times[name] = res
        tflops[name] = {}
        for mode in args.modes:
            t_ms = _median_ms(res.get(mode))
            tflops[name][mode] = (
                calculate_tflops(ks, ke, shape.num_heads_q, shape.head_dim, t_ms)
                if t_ms is not None else None
            )

    if args.verification == "reference":
        verify = _verify_reference(
            impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape
        )
    elif args.verification == "triton":
        verify = _verify_vs_triton(
            impls, q_fp8, kv_fp8, scales, weights, ks, ke, shape
        )
    else:
        verify = {name: "N/A" for name in impls}

    # Per-impl speedup vs the Triton baseline, computed per mode.
    speedups = {}
    for name in impls:
        if name == "triton":
            continue
        speedups[name] = {}
        for mode in args.modes:
            base = _median_ms((times.get("triton") or {}).get(mode))
            t = _median_ms(times[name].get(mode))
            speedups[name][mode] = (
                f"{base / t:.2f}x" if base and t else "-"
            )

    return {
        "idx": idx,
        "shape": shape,
        "times": times,
        "tflops": tflops,
        "verify": verify,
        "speedups": speedups,
        "triton_label": triton_label,
    }


def _verify_reference(impls, q, kv, q_fp8, kv_fp8, scales, weights, ks, ke, shape):
    """Grade every implementation against the torch reference (``--verification reference``).

    The primary ground truth is ``ref_fp8_mqa_logits`` (the DeepGEMM-derived
    torch implementation, also used by the unit test). It materializes a dense
    ``[s_q, s_kv, H]`` tensor, which OOMs at long ISL (256 GiB at 32k). In that
    case we fall back to the **Triton** kernel as the reference.
    Grades against the Triton reference are suffixed ``*`` to flag the
    weaker check, and Triton (when present) is marked ``REF`` since it can't be
    graded against itself.

    Returns {impl_name: "PASS"|"FAIL"|"SKIP"|"PASS*"|"FAIL*"|"REF"|"OOM"|"ERR"}.
    """
    from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
        calc_diff,
        ref_fp8_mqa_logits,
    )

    ref_is_triton = False
    ref = None
    try:
        ref, _ = ref_fp8_mqa_logits(
            q=q, kv=kv, weights=weights, cu_seqlen_ks=ks, cu_seqlen_ke=ke
        )
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            ref = triton_logits(
                q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits
            )
            ref_is_triton = True
        except (torch.OutOfMemoryError, RuntimeError, TritonOutOfResources):
            torch.cuda.empty_cache()
            # Standard Triton launcher may fail on mixed-dtype (LDS overflow);
            # try the reduced BLOCK_KV=64 fallback.
            try:
                ref = _triton_logits_fallback(
                    q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits,
                    block_kv=64,
                )
                ref_is_triton = True
            except (torch.OutOfMemoryError, RuntimeError, TritonOutOfResources):
                torch.cuda.empty_cache()
                return {name: "OOM" for name in impls}

    suffix = "*" if ref_is_triton else ""
    status = {}
    for name, fn in impls.items():
        if ref_is_triton and name == "triton":
            status[name] = "REF"
            continue
        try:
            if name == GEAK_IMPL_NAME:
                out = fn(q_fp8, kv_fp8, scales, weights, ks, ke)
            else:
                out = fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
        except NotImplementedError:
            status[name] = "SKIP"
            continue
        except (RuntimeError, TritonOutOfResources):
            status[name] = "ERR"
            continue

        m = (ref == float("-inf")) | (out == float("-inf"))
        diff = calc_diff(out.masked_fill(m, 0), ref.masked_fill(m, 0))
        status[name] = ("PASS" if diff < 1e-3 else "FAIL") + suffix
    return status


def _verify_vs_triton(impls, q_fp8, kv_fp8, scales, weights, ks, ke, shape):
    """Grade every non-Triton implementation against the Triton kernel output.

    Uses the external script's verification approach: assert all inf/NaN
    positions match, then compute max relative error on finite values only.
    Triton itself is marked ``REF``.

    This mode is useful when the torch reference OOMs or when you want to
    directly compare kernel outputs without worrying about the bf16 reference
    round-trip.

    Returns {impl_name: "PASS"|"FAIL"|"SKIP"|"REF"|"ERR"
             | "PASS(rel=X.Xe-Y)" | "FAIL(rel=X.Xe-Y)"}.
    """
    status = {}

    # Obtain Triton reference output.
    triton_fn = impls.get("triton")
    if triton_fn is None:
        return {name: "NO-REF" for name in impls}

    try:
        ref = triton_fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
    except NotImplementedError:
        return {name: "NO-REF" for name in impls}
    except (RuntimeError, TritonOutOfResources):
        return {name: "NO-REF" for name in impls}

    status["triton"] = "REF"

    for name, fn in impls.items():
        if name == "triton":
            continue
        try:
            if name == GEAK_IMPL_NAME:
                out = fn(q_fp8, kv_fp8, scales, weights, ks, ke)
            else:
                out = fn(q_fp8, kv_fp8, scales, weights, ks, ke, shape.clean_logits)
        except NotImplementedError:
            status[name] = "SKIP"
            continue
        except (RuntimeError, TritonOutOfResources):
            status[name] = "ERR"
            continue

        # Check that inf positions match (mirroring external script's inf_ok).
        inf_ok = (torch.isinf(out) == torch.isinf(ref)).all().item()

        # Max relative error on finite values only.
        fin = torch.isfinite(ref)
        if fin.any():
            rel = (
                (out[fin] - ref[fin]).abs()
                / ref[fin].abs().clamp_min(1e-3)
            ).max().item()
        else:
            # All values are inf/-inf; if inf positions match, it's a pass.
            rel = 0.0

        if not inf_ok:
            status[name] = f"FAIL(inf,rel={rel:.1e})"
        elif rel < 0.05:
            status[name] = f"PASS(rel={rel:.1e})"
        else:
            status[name] = f"FAIL(rel={rel:.1e})"

    return status


def _fmt_shape(shape):
    s = (
        f"bs{shape.batch_size} {shape.seq_q_l}x{shape.seq_kv_l} "
        f"H{shape.num_heads_q} D{shape.head_dim}"
    )
    combo = getattr(shape, "dtype_combo", None)
    if combo:
        s += f" [{combo}]"
    return s


def _git_commit():
    """Return the current short+long git commit hash, or None if unavailable."""
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ) != 0
        return full + (" (dirty)" if dirty else "")
    except Exception:
        return None


def _gpu_info():
    """Return a list of (label, value) describing the GPU / runtime environment."""
    info = []
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.append(("GPU", props.name))
        arch = getattr(props, "gcnArchName", None)
        if arch:
            info.append(("Arch", arch))
        info.append(("GPU count", str(torch.cuda.device_count())))
        info.append(
            ("VRAM", f"{props.total_memory / (1024 ** 3):.1f} GiB")
        )
    else:
        info.append(("GPU", "none (CUDA/HIP not available)"))
    info.append(("torch", torch.__version__))
    info.append(("triton", triton.__version__))
    hip = getattr(getattr(torch, "version", None), "hip", None)
    if hip:
        info.append(("HIP", hip))
    info.append(("flydsl", "available" if is_flydsl_available() else "unavailable"))
    return info


def _time_cell(result):
    """Format a _time_modes result as a 'median(\u00b1std)us' string or status.

    ``None`` -> 'N/A'; the 'GRAPH-FAIL' status string passes through; a
    ``TimingStats`` renders its median (and std when there is spread).
    """
    if result is None:
        return "N/A"
    if isinstance(result, str):
        return result  # e.g. GRAPH-FAIL
    med = result.median_us
    std = result.std_us
    if std >= 0.05 * max(med, 1e-9):
        return f"{med:.1f}\u00b1{std:.1f}"
    return f"{med:.1f}"


def _tflops_cell(val):
    return "N/A" if val is None else f"{val:.1f}"


def _per_shape_rows(row, impls, modes):
    """Return (header_names, data_rows) for a single shape's result table.

    Each impl becomes one data row with columns:
      impl | time_us (per mode) | TFLOPs (per mode) | verify | vs_triton (per mode)
    """
    names = list(impls)
    has_tri = "triton" in names

    # Build column headers.
    headers = ["impl"]
    for m in modes:
        headers.append(f"time_{m}_us")
        headers.append(f"TFLOPs_{m}")
    headers.append("verify")
    if has_tri:
        for m in modes:
            headers.append(f"vs_triton_{m}")

    triton_label = row.get("triton_label")
    data_rows = []
    for name in names:
        lbl = _impl_label(name)
        if name == "triton" and triton_label:
            lbl = triton_label
        cells = [lbl]
        for m in modes:
            cells.append(_time_cell(row["times"].get(name, {}).get(m)))
            cells.append(_tflops_cell(row["tflops"].get(name, {}).get(m)))
        cells.append(row["verify"].get(name, "N/A"))
        if has_tri:
            for m in modes:
                cells.append(row["speedups"].get(name, {}).get(m, "-"))
        data_rows.append(cells)

    return headers, data_rows


def _markdown_table(rows, impls, modes):
    """Return per-shape tables as a GitHub-flavored Markdown string."""
    parts = []
    for row in rows:
        shape_label = f"### Shape {row['idx']}: {_fmt_shape(row['shape'])}"
        headers, data_rows = _per_shape_rows(row, impls, modes)

        lines = [shape_label, ""]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for dr in data_rows:
            lines.append("| " + " | ".join(dr) + " |")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _write_markdown(path, rows, impls, args):
    """Write per-shape tables + environment/git metadata as Markdown."""
    commit = _git_commit() or "unknown"
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    parts = [
        "# FlyDSL vs Triton FP8 MQA Logits benchmark",
        "",
        f"- Generated: {when}",
        f"- Git commit: `{commit}`",
        f"- Verification: {args.verification}",
        f"- Dtype combos: {', '.join(args.dtype_combos)}",
        f"- Timing modes: {', '.join(args.modes)} "
        f"(warmup={args.warmup}, bench_iters={args.bench_iters}, "
        f"graph_replay_iters={args.graph_replay_iters})",
        "- Time cells: `median(\u00b1std) us`. `GRAPH-FAIL` = not graph-capturable "
        "(e.g. Triton allocates its output internally).",
        "",
        "## Environment",
        "",
    ]
    for label, value in _gpu_info():
        parts.append(f"- {label}: {value}")
    parts += ["", "## Results", "", _markdown_table(rows, impls, args.modes), ""]

    base_dir = os.path.dirname(path)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir)
    with open(path, "w") as f:
        f.write("\n".join(parts))


def _print_table(rows, impls, args):
    """Print one table per shape; each impl is one row.

    Columns: impl | time_us (per mode) | TFLOPs (per mode) | verify | vs_triton (per mode)
    Each column is widened to fit its header.
    """
    for row in rows:
        print(f"\nShape {row['idx']}: {_fmt_shape(row['shape'])}")

        headers, data_rows = _per_shape_rows(row, impls, args.modes)
        widths = [max(len(h), max((len(dr[i]) for dr in data_rows), default=0)) + 2
                  for i, h in enumerate(headers)]

        header_line = "".join(f"{h:>{w}}" for h, w in zip(headers, widths))
        print(header_line)
        print("-" * len(header_line))
        for dr in data_rows:
            print("".join(f"{v:>{w}}" for v, w in zip(dr, widths)))


# Names of the per-shape manual override flags (no defaults -- all or nothing).
_MANUAL_SHAPE_FLAGS = ("batch_size", "num_heads_q", "head_dim", "seq_q_l", "seq_kv_l")


def _resolve_shapes(args):
    """Resolve the shape selection into a list of (idx, (bs, s_q, s_kv, H, D)).

    ``idx`` is the 1-based case number shown in the table and accepted by
    ``--shape-index`` (converted to a 0-based offset into PRESET_SHAPES here).

    Three mutually-exclusive modes:
      * default (no shape args)      -> the full PRESET_SHAPES sweep
      * --shape-index N              -> a single preset shape by 1-based index
      * manual flags (all required)  -> a single custom shape
    """
    manual = {f: getattr(args, f) for f in _MANUAL_SHAPE_FLAGS}
    any_manual = any(v is not None for v in manual.values())

    if args.shape_index is not None and any_manual:
        raise SystemExit(
            "[error] --shape-index and manual shape flags are mutually exclusive."
        )

    if args.shape_index is not None:
        if not (1 <= args.shape_index <= len(PRESET_SHAPES)):
            raise SystemExit(
                f"[error] --shape-index must be in [1, {len(PRESET_SHAPES)}]; "
                f"got {args.shape_index}. Use --list to see the preset shapes."
            )
        zero_based = args.shape_index - 1
        return [(args.shape_index, PRESET_SHAPES[zero_based])]

    if any_manual:
        missing = [f for f, v in manual.items() if v is None]
        if missing:
            raise SystemExit(
                "[error] manual shape requires all of "
                f"{list(_MANUAL_SHAPE_FLAGS)}; missing: {missing}."
            )
        # "M" marks a manual (non-preset) shape -- not re-runnable by index.
        return [
            (
                "M",
                (
                    manual["batch_size"],
                    manual["seq_q_l"],
                    manual["seq_kv_l"],
                    manual["num_heads_q"],
                    manual["head_dim"],
                ),
            )
        ]

    return [(i + 1, s) for i, s in enumerate(PRESET_SHAPES)]


def main():
    p = argparse.ArgumentParser(
        description="FlyDSL vs Triton FP8 MQA Logits A/B benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Shape selection (3 mutually-exclusive modes; see _resolve_shapes).
    p.add_argument(
        "--shape-index",
        type=int,
        default=None,
        help="run a single shape from the preset sweep by 1-based index (see --list)",
    )
    # Manual override flags -- NO defaults; supplying any requires all of them.
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_heads_q", type=int, default=None)
    p.add_argument("--head_dim", type=int, default=None)
    p.add_argument("--seq_q_l", type=int, default=None)
    p.add_argument("--seq_kv_l", type=int, default=None)

    p.add_argument(
        "--impl",
        choices=["all", "triton", "flydsl"],
        default="all",
        help="which implementation family(ies) to run",
    )
    p.add_argument(
        "--flydsl-variants",
        type=str,
        default=None,
        metavar="V1,V2,...",
        help=(
            "comma-separated FlyDSL kernel-version tags to benchmark, each as its "
            "own column (default: all registered variants). See --list-variants."
        ),
    )
    p.add_argument(
        "--list-variants",
        action="store_true",
        help="list the available FlyDSL kernel variants and exit",
    )
    p.add_argument("--clean_logits", type=int, default=1)
    p.add_argument(
        "--dtype-combo",
        type=str,
        default="fnuz/fnuz",
        metavar="COMBO",
        help=(
            "comma-separated fp8 operand-dtype combos to benchmark. Each combo "
            "is 'q_type/kv_type'. Available: "
            + ", ".join(DTYPE_COMBOS)
            + ". 'all' = every combo. "
        ),
    )
    p.add_argument(
        "--mode",
        choices=["eager", "graph", "all"],
        default="eager",
        help=(
            "timing strategy: 'eager' = per-call latency (device + host bubble); "
            "'graph' = HIP graph-replay steady state (host overhead stripped); "
            "'all' = both side by side. Triton allocates its output internally so "
            "it is eager-only (GRAPH-FAIL in the graph column)."
        ),
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="warmup iterations before timing (MeasureConfig.warmup_iters)",
    )
    p.add_argument(
        "--bench-iters", type=int, default=20,
        help="number of timed samples per mode (MeasureConfig.bench_iters)",
    )
    p.add_argument(
        "--graph-replay-iters", type=int, default=50,
        help="graph replays bracketed per sample (MeasureConfig.graph_replay_iters)",
    )
    p.add_argument(
        "--replay-iters", type=int, default=50,
        help="graph replays bracketed per sample (MeasureConfig.replay_iters)",
    )
    p.add_argument(
        "--verification",
        choices=["none", "reference", "triton"],
        default="none",
        help=(
            "verification mode. 'none' = skip verification. "
            "'reference' = grade against the torch reference (calc_diff < 1e-3; "
            "falls back to Triton when torch OOMs). "
            "'triton' = grade all non-Triton impls against the Triton kernel "
            "output directly (checks inf positions match, reports max relative "
            "error on finite values; Triton shown as REF)."
        ),
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="shorthand for --verification reference",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="write the comparison table (Markdown) + GPU/git metadata to FILE",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list the preset shapes (with their --shape-index) and exit",
    )
    args = p.parse_args()

    # --verify is shorthand for --verification reference (backward compat).
    if args.verify and args.verification == "none":
        args.verification = "reference"

    # Resolve the timing-mode selection into an ordered tuple of strategies.
    args.modes = TIMING_MODES if args.mode == "all" else (args.mode,)

    # Resolve dtype combos.
    raw_combos = args.dtype_combo
    if raw_combos == "all":
        args.dtype_combos = list(DTYPE_COMBOS)
    else:
        args.dtype_combos = [c.strip() for c in raw_combos.split(",") if c.strip()]
        for c in args.dtype_combos:
            if c not in DTYPE_COMBOS:
                raise SystemExit(
                    f"[error] unknown dtype combo {c!r}; "
                    f"available: {list(DTYPE_COMBOS)}"
                )

    if args.list:
        print("preset shapes (index: bs, s_q, s_kv, H, D):")
        for i, s in enumerate(PRESET_SHAPES):
            print(f"  {i + 1}: {s}")
        return

    avail, default = _available_variants()
    if args.list_variants:
        if avail is None:
            print("FlyDSL is unavailable; no variants to list.")
        else:
            print("FlyDSL kernel variants (default marked *):")
            for v in avail:
                print(f"  {'*' if v == default else ' '} {v}")
        return

    # Resolve the requested FlyDSL variants (comma list), defaulting to all
    # registered variants so new entries (e.g. "geak") appear automatically.
    # Validation against what's actually registered happens in _select_impls so
    # an unavailable-FlyDSL run still works for --impl triton.
    if args.flydsl_variants:
        args.flydsl_variants = [
            v.strip() for v in args.flydsl_variants.split(",") if v.strip()
        ]
    else:
        args.flydsl_variants = list(avail) if avail is not None else []

    run(args)


if __name__ == "__main__":
    main()
