# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Perf sweep for the FlyDSL fused GDN K5+K6 forward (gfx942).

Two kernel modes share one set of preset shapes:

  ``--kernel fused`` (default)
      The fused K5+K6 kernel against the two-kernel pipelines
      (Triton K5 + Triton K6, HIP K5 + Triton K6, FlyDSL "opt" K5 + Triton K6).

  ``--kernel k5``
      The inter-chunk state scan alone: Triton vs HIP vs both FlyDSL K5
      kernels.

Usage::

    # List the preset shapes with their 1-based indices:
    python op_tests/op_benchmarks/flydsl/bench_gdn_fused_k5k6.py --list

    # One shape, every kernel variant, ratios against the HIP pipeline:
    python op_tests/op_benchmarks/flydsl/bench_gdn_fused_k5k6.py \
        --shape 7 --instance all --baseline hip

    # Full sweep, heuristic-selected variant, markdown report to a file:
    python op_tests/op_benchmarks/flydsl/bench_gdn_fused_k5k6.py \
        --output-path /tmp/fused.md
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch

# Importable when run as a script from anywhere in the repo.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.k5_variants import (
    K5_VARIANTS,
    _bv_of_variant,
    _legal_bv_candidates,
)
from aiter.test_common import (
    checkAllclose,
    run_perftest,
)
from op_tests.gdn_common import _make_inputs

# Shapes: Kimi-K3 (KDA, gk gate) and Qwen (GDN, g gate), real TP configs.
# Columns: (model_tag, H, Hg, T_flat, N, gate)
# K=V=128 (fixed for all KDA/GDN production shapes).
_SWEEP_SHAPES: list[tuple] = [
    # KDA Kimi-K3 TP8 (H=12, gk)
    ("kda_tp8", 12, 12, 8192, 1, "gk"),
    ("kda_tp8", 12, 12, 8192, 4, "gk"),
    ("kda_tp8", 12, 12, 8192, 8, "gk"),
    ("kda_tp8", 12, 12, 32768, 1, "gk"),
    ("kda_tp8", 12, 12, 32768, 4, "gk"),
    ("kda_tp8", 12, 12, 32768, 8, "gk"),
    # KDA Kimi-K3 TP4 (H=24, gk)
    ("kda_tp4", 24, 24, 8192, 1, "gk"),
    ("kda_tp4", 24, 24, 8192, 8, "gk"),
    ("kda_tp4", 24, 24, 32768, 1, "gk"),
    ("kda_tp4", 24, 24, 32768, 8, "gk"),
    # GDN Qwen3-Next TP8 (H=4, Hg=2, g)
    ("gdn_q3n_tp8", 4, 2, 8192, 4, "g"),
    ("gdn_q3n_tp8", 4, 2, 8192, 8, "g"),
    ("gdn_q3n_tp8", 4, 2, 32768, 4, "g"),
    ("gdn_q3n_tp8", 4, 2, 32768, 8, "g"),
    # GDN Qwen3-Next TP4 (H=8, Hg=4, g)
    ("gdn_q3n_tp4", 8, 4, 8192, 1, "g"),
    ("gdn_q3n_tp4", 8, 4, 8192, 4, "g"),
    ("gdn_q3n_tp4", 8, 4, 32768, 1, "g"),
    ("gdn_q3n_tp4", 8, 4, 32768, 4, "g"),
    # GDN Qwen3.5-MoE TP1 (H=16, g)
    ("gdn_q35_tp1", 16, 16, 8192, 1, "g"),
    ("gdn_q35_tp1", 16, 16, 8192, 4, "g"),
    ("gdn_q35_tp1", 16, 16, 8192, 8, "g"),
    # GDN Qwen3.5-MoE TP1 (H=32, Hg=8, g)
    ("gdn_q35_tp1", 32, 8, 32768, 1, "g"),
    ("gdn_q35_tp1", 32, 8, 32768, 8, "g"),
    # -- Shapes benchmarked in PR #4732 (the flydsl_opt / mfma16_hip fork) ----
    # "varlen-qwen-ali-tp1" (TP1, H=32, Hg=16)
    ("gdn_qwen_tp1", 32, 16, 8192, 1, "g"),
    ("gdn_qwen_tp1", 32, 16, 16384, 2, "g"),
    ("gdn_qwen_tp1", 32, 16, 32768, 4, "g"),
    ("gdn_qwen_tp1", 32, 16, 65536, 8, "g"),
    # "varlen-qwen3.5-397b-ptpc-ali" (TP8, H=8, Hg=2)
    ("gdn_q35_397b_tp8", 8, 2, 8192, 1, "g"),
    ("gdn_q35_397b_tp8", 8, 2, 16384, 2, "g"),
    ("gdn_q35_397b_tp8", 8, 2, 32768, 4, "g"),
    ("gdn_q35_397b_tp8", 8, 2, 65536, 8, "g"),
]

# Every T_flat present above; --T accepts any of these (or "all").
_SWEEP_T_VALUES = sorted({s[3] for s in _SWEEP_SHAPES})

SUPPORTED_GFX = ["gfx942"]

_USE_EXP2 = False  # gates are natural-log domain; see bench_chunk_gdn_fwd.py

# The reference implementations, in the spelling ``--baseline`` accepts. Each
# maps to the candidate name used in ``fused`` and in ``k5`` mode respectively:
# in fused mode every baseline is a two-kernel pipeline ending in Triton K6.
_BASELINES = {
    "triton": ("triton+triton", "triton"),
    "hip": ("hip+triton", "hip"),
    "flydsl_opt": ("flydsl_opt+triton", "flydsl_opt"),
}

# ``--instance`` sentinels. Anything else must be a tag in K5_VARIANTS.
_INSTANCE_AUTO = "auto"
_INSTANCE_ALL = "all"

# ``--verification ref`` = the pure-PyTorch reference in op_tests/gdn_common.py;
# the other choices are the reference implementations in _BASELINES.
_VERIFY_REF = "ref"
_VERIFY_CHOICES = (_VERIFY_REF, *sorted(_BASELINES))


class _CaptureSafeMeta:
    """Proxy over GatedDeltaRulePrefillMetadata that no-ops validate() during capture.

    Triton K5, Triton K6, and HIP K5 all call validate() on every invocation.
    validate() raises when torch.cuda.is_current_stream_capturing() is True, so
    passing raw metadata breaks graph capture.  This proxy validates once before
    capture (at construction time) and then silently skips subsequent calls, which
    is safe because the tensor identity and version are stable across a replay.
    """

    def __init__(
        self, meta, cu_seqlens, *, chunk_size, total_prefill_tokens, num_sequences
    ):
        self._meta = meta
        meta.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            num_decodes=0,
            num_decode_tokens=0,
            total_prefill_tokens=total_prefill_tokens,
            num_sequences=num_sequences,
        )

    def validate(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        return getattr(self._meta, name)


_NUM_WARMUP = 2
_NUM_ITERS = 101
# ``checkAllclose`` returns the RATIO of elements outside (rtol, atol), so 0 is
# clean. Matches its own ``tol_err_ratio`` default.
_ERR_RATIO_TOL = 0.05


def _graph_time_us(fn) -> float:
    """Per-iteration device time, via HIP graph capture of ``_NUM_ITERS`` calls.

    Capturing the calls (rather than replaying one captured call N times) keeps
    the ROCm profiler out of the loop: its device-time attribution for the
    CUDAGraphExec event is unreliable and returns 0 on many shapes.

    Kernels that allocate their outputs internally -- K5 allocates the h
    snapshots, ~537 MB/call at T=32768/H=32 -- do not blow up the capture: the
    caching allocator reuses blocks freed within the capture, so the graph pool
    holds ~1 iteration's worth, not ``_NUM_ITERS``.
    """
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(_NUM_ITERS):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / _NUM_ITERS  # ms -> us, per iter


# Set by _bench_candidates whenever a kernel we own disagrees with the selected
# baseline; main() turns it into a nonzero exit status.
_STRICT_FAILURES = []


def _bench_candidates(
    candidates,
    *,
    flops,
    nbytes,
    label,
    baseline,
    strict,
    out_of=lambda o: o,
    ref_name=None,
    ref_out=None,
):
    """Time every candidate; return one row per candidate.

    Two independent reference axes, deliberately decoupled:

      * ``baseline`` is the PERF denominator -- every ``vs <baseline>`` ratio.
        It must be a key of ``candidates``, named explicitly rather than taken
        from dict order.
      * ``ref_name`` / ``ref_out`` is the correctness reference. Pass
        ``ref_out`` for an externally computed tensor (the torch reference), or
        ``ref_name`` to compare against another candidate's output. Default
        (both None) is ``baseline``, i.e. what ``--verification`` overrides.

    A candidate that is the correctness reference runs first so its output
    exists before anything is compared to it; with an external ``ref_out``,
    even the baseline is checked against it.

    ``strict`` is the set of candidate names we own. Only those gate the run:

      * ``strict`` candidate outside tolerance -> **FAIL** (logged at error
        level, recorded in ``_STRICT_FAILURES``, nonzero exit from ``main``).
      * any other candidate outside tolerance -> **warn** only. ``hip`` and
        ``flydsl_opt`` are upstream implementations.

    ``out_of`` maps a candidate's return value to the tensor to compare; K5
    returns ``(h, v_new, final_state)`` while the fused path returns ``o``.

    ``flops`` and ``nbytes`` are properties of the *shape*, identical for every
    candidate, so TFLOPS and TB/s are pure functions of the measured time. They
    are still reported per row because the per-case table is read row-wise.
    """
    if baseline not in candidates:
        raise KeyError(f"baseline {baseline!r} is not one of {list(candidates)}")
    unknown = set(strict) - set(candidates)
    if unknown:
        raise KeyError(f"strict names not in candidates: {sorted(unknown)}")
    if ref_out is None and ref_name is None:
        ref_name = baseline
    if ref_name is not None and ref_name not in candidates:
        raise KeyError(f"reference {ref_name!r} is not one of {list(candidates)}")
    ref_label = ref_name if ref_out is None else "ref"

    # The correctness reference first (it defines ref_out), then the baseline,
    # then the rest. Ratios are resolved after the loop, so the run order does
    # not have to put the perf baseline first.
    order = [n for n in (ref_name, baseline) if n is not None]
    order += [n for n in candidates if n not in order]

    rows = []
    for name in dict.fromkeys(order):
        fn = candidates[name]
        # Correctness: minimal eager run just to get the output tensor.
        out, _ = run_perftest(fn, num_iters=2, num_warmup=_NUM_WARMUP)
        out = out_of(out)
        if ref_out is None:
            ref_out = out
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} vs {ref_label}: {label}",
        )

        us = _graph_time_us(fn)
        aiter.logger.info(f"{name}: {us:.3f} us/iter with hipgraph")

        if err <= _ERR_RATIO_TOL:
            check = "ok" if err == 0 else f"ok({err:.1g})"
        elif name in strict:
            aiter.logger.error("FAIL vs %s: %s=%.2g (%s)", ref_label, name, err, label)
            _STRICT_FAILURES.append(f"{label}: {name}={err:.2g} vs {ref_label}")
            check = f"FAIL({err:.2g})"
        else:
            aiter.logger.warning(
                "known upstream mismatch vs %s: %s=%.2g (%s)",
                ref_label,
                name,
                err,
                label,
            )
            check = f"warn({err:.2g})"

        rows.append(
            {
                "instance": name,
                "us": round(us, 3),
                "TFLOPS": round(flops / (us * 1e-6) / 1e12, 1) if us > 0 else 0.0,
                "TB/s": round(nbytes / (us * 1e-6) / 1e12, 3) if us > 0 else 0.0,
                f"check (vs {ref_label})": check,
                "_us": us,
                "_ours": name in strict,
            }
        )

    # Ratios last: the baseline is no longer guaranteed to have run first.
    base_us = next(r["_us"] for r in rows if r["instance"] == baseline)
    for r in rows:
        r[f"vs {baseline}"] = f"{base_us / r['_us']:.2f}x" if r["_us"] > 0 else "n/a"
    # Column order: us, ratio, then the derived rates and the check.
    key_order = ["instance", "us", f"vs {baseline}", "TFLOPS", "TB/s"]
    return [
        {k: r[k] for k in key_order + [k for k in r if k not in key_order]}
        for r in rows
    ]


def _auto_tag(kernel, *, H, Hg, V, T_flat, N, is_varlen) -> str:
    """The tile tag the launcher's own heuristic resolves to for this shape.

    Same chain the launcher uses, so a label built from it names the kernel
    that actually ran rather than the one we asked for.
    """
    if kernel == "fused":
        from aiter.ops.flydsl.gdn_fused_gfx942_kernels import _fused_bv_for_shape

        bv, num_waves = _fused_bv_for_shape(H=H, V=V, N=N, variant=None)
        return f"bv{bv}w{num_waves}" if num_waves > 4 else f"bv{bv}"

    from aiter.ops.flydsl.linear_attention_prefill_kernels import _resolve_variant

    return _resolve_variant(
        None, H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=is_varlen
    )


def _resolve_instances(requested, *, kernel, H, Hg, V, T_flat, N, is_varlen):
    """``--instance`` -> ``[(display_tag, variant_or_None), ...]``.

    ``auto`` passes ``variant=None`` so the launcher runs its own heuristic;
    the tag it resolves to is shown in the label (``auto:bv32``) so a per-case
    table says which tile actually ran. ``all`` expands to every registered tag
    that is legal for this V.
    """
    if requested == _INSTANCE_AUTO:
        auto_tag = _auto_tag(
            kernel, H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=is_varlen
        )
        return [(f"auto:{auto_tag}", None)]
    if requested == _INSTANCE_ALL:
        legal_bv = set(_legal_bv_candidates(V))
        return [(t, t) for t in K5_VARIANTS if _bv_of_variant(t) in legal_bv]
    return [(requested, requested)]


def _prep_case(H, Hg, T_flat, N, gate, *, K, V, BT):
    """Inputs + capture-safe metadata shared by every candidate for one shape."""
    seq_lens = [T_flat // N] * (N - 1) + [T_flat - (T_flat // N) * (N - 1)]
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)

    # Build capture-safe metadata once before any graph capture.  Without this,
    # validate() would raise inside the captured closure for all varlen shapes.
    safe_meta = None
    if inp["cu"] is not None:
        from aiter.ops.prefill_batch_metadata import (
            build_gated_delta_rule_prefill_metadata,
        )

        raw_meta = build_gated_delta_rule_prefill_metadata(
            seq_lens,
            cu_seqlens=inp["cu"],
            chunk_size=BT,
        )
        safe_meta = _CaptureSafeMeta(
            raw_meta,
            inp["cu"],
            chunk_size=BT,
            total_prefill_tokens=T_flat,
            num_sequences=N,
        )
    return inp, seq_lens, safe_meta


def _reference_spec(verification, *, kernel, inp, scale):
    """``--verification`` -> ``(ref_name, ref_out)`` for ``_bench_candidates``.

    ``None`` (the default) leaves both unset, which means "check against the
    perf baseline" -- the historical behaviour. ``ref`` computes the
    pure-PyTorch reference eagerly; the others just name a candidate.
    """
    if verification is None:
        return None, None
    if verification != _VERIFY_REF:
        return _BASELINES[verification][0 if kernel == "fused" else 1], None

    if kernel == "fused":
        from op_tests.gdn_common import _reference_o

        return None, _reference_o(inp, scale=scale, use_exp2=_USE_EXP2)

    from op_tests.gdn_common import ref_chunk_gated_delta_rule_fwd_h

    # Token-major w/u and the 3-D g: the reference's own convention, not the
    # head-major layout the kernels take. ``[0]`` -> h, matching ``out_of``.
    h_ref, _, _ = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )
    return None, h_ref


def bench_fused_k5k6(
    model_tag, H, Hg, T_flat, N, gate, *, baseline, instances, verification=None
):
    """Fused FlyDSL K5+K6 vs the Triton / HIP / FlyDSL-opt two-kernel pipelines.

    All candidates are timed with HIP graph capture, which eliminates Python
    dispatch overhead and measures pure GPU kernel time.

    Besides the requested ``instances`` (explicit BV tiles of the fused kernel),
    the sweep always times ``auto_dispatch`` -- the production wrapper that
    applies the fused-vs-separate routing heuristic -- so the table shows both
    what the fused kernel can do and what production would actually pick.
    """
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )
    from aiter.ops.flydsl.gdn_fused_gfx942_kernels import (
        chunk_gated_delta_rule_fwd_h_o_auto,
        chunk_gated_delta_rule_fwd_h_o_flydsl,
        should_use_fused_k5k6_gfx942,
    )
    from aiter.ops.gated_delta_rule_fusion import K5K6Fusion
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_fwd_o_opt_vk as k6_triton,
    )
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_fwd_h_opt_vk as k5_triton,
    )

    K = V = 128
    BT = 64
    scale = K**-0.5

    inp, seq_lens, safe_meta = _prep_case(H, Hg, T_flat, N, gate, K=K, V=V, BT=BT)
    # ``inp["q"]``, not a fresh tensor: ``--verification ref`` runs the torch
    # reference over the same dict, so a locally generated q would compare the
    # candidates against a reference computed from different data.
    q = inp["q"]
    # The HIP K5 and the FlyDSL "opt" port both want g as 3-D [B, H, T]
    # head-major; ``inp["g"]`` is the 2-D [H, T_flat] form the vk kernel takes.
    g_hm3 = inp["g"].unsqueeze(0) if inp["g"] is not None else None

    # One output buffer per candidate: a shared buffer would let a later
    # candidate overwrite the tensor an earlier one is still compared against.
    outs = {}

    def _out(name):
        return outs.setdefault(name, inp["u_tm"].new_empty(1, T_flat, H, V))

    def _run_triton():
        h, v_new, _ = k5_triton(
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=inp["g"],
            gk=inp["gk"],
            initial_state=inp["h0"],
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        o = _out("triton+triton")
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o

    def _run_hip():
        h, v_new, _ = chunk_gated_delta_rule_fwd_h_hip_fn(
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=g_hm3,
            # HIP K5 wants gk flat [T, H, K]; the batched [B, T, H, K] the
            # fused/Triton paths accept is rejected here.
            gk=inp["gk_flat"],
            initial_state=inp["h0"],
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            g_head_major=True,
            prefill_metadata=safe_meta,
        )
        o = _out("hip+triton")
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o

    def _run_flydsl_opt():
        # Kernel (1) K5 + Triton K6 -- the separate path that production runs
        # today, i.e. the number the fused candidates actually have to beat.
        # (1) has no fused build, so it can only appear here as a two-kernel
        # pipeline.
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_opt,
        )

        h, v_new, _ = chunk_gated_delta_rule_fwd_h_flydsl_opt(
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=g_hm3,
            gk=inp["gk"],
            initial_state=inp["h0"],
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            g_head_major=True,
            prefill_metadata=safe_meta,
        )
        o = _out("flydsl_opt+triton")
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o

    def _make_fused(name, variant):
        def _run():
            o = _out(name)
            chunk_gated_delta_rule_fwd_h_o_flydsl(
                q=q,
                k=inp["k"],
                w=inp["w_hm"],
                u=inp["u_hm"],
                g=inp["g"],
                gk=inp["gk"],
                scale=scale,
                initial_state=inp["h0"],
                # Must match the baselines' output_final_state=True: the
                # separate pipelines pay for a final-state HBM write, and
                # letting the fused candidates skip it inflated their speedup.
                output_final_state=True,
                cu_seqlens=inp["cu"],
                use_exp2=_USE_EXP2,
                o=o,
                variant=variant,
            )
            return o

        return _run

    # What the routing wrapper will actually dispatch to, resolved through the
    # same helpers it uses. Spelling it out in the candidate name matters
    # because the two routes are different kernels: the fused build at one BV
    # tile, or the FlyDSL vk K5 (its own BV rule, not the fused one) followed
    # by Triton K6.
    routes_fused = should_use_fused_k5k6_gfx942(H=H, N=N, V=V)
    tag_kw = {
        "H": H,
        "Hg": Hg,
        "V": V,
        "T_flat": T_flat,
        "N": N,
        "is_varlen": inp["cu"] is not None,
    }
    dispatch_name = (
        f"auto_dispatch[fused {_auto_tag('fused', **tag_kw)}]"
        if routes_fused
        else f"auto_dispatch[k5 {_auto_tag('k5', **tag_kw)} + triton k6]"
    )

    def _run_auto_dispatch():
        o = _out(dispatch_name)
        chunk_gated_delta_rule_fwd_h_o_auto(
            q=q,
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=inp["g"],
            gk=inp["gk"],
            scale=scale,
            initial_state=inp["h0"],
            # Aligned with the baselines (see _make_fused).
            output_final_state=True,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            o=o,
            fusion=K5K6Fusion.AUTO,
        )
        return o

    candidates = {
        "triton+triton": _run_triton,
        "hip+triton": _run_hip,
        "flydsl_opt+triton": _run_flydsl_opt,
    }
    ours = []
    for tag, variant in instances:
        name = f"fused[{tag}]"
        candidates[name] = _make_fused(name, variant)
        ours.append(name)
    candidates[dispatch_name] = _run_auto_dispatch
    ours.append(dispatch_name)

    # FLOPs: K5 (GEMM1+GEMM2) + K6 (GEMM3+GEMM4a+GEMM4b) per chunk per head
    n_chunks = sum(-(-s // BT) for s in seq_lens)
    per_chunk = 4 * K * V + 2 * K * V + 2 * BT * K + 2 * BT * V
    flops = H * n_chunks * BT * per_chunk
    # bytes: k,w,u (read) + v_new,h (K5->K6 handoff) + q (read) + o (written)
    nbytes = (
        T_flat * Hg * K  # k
        + T_flat * H * K  # w
        + T_flat * H * V  # u
        + T_flat * H * V  # v_new (fused keeps in LDS; baseline spills to HBM)
        + N * H * V * K  # h state
        + T_flat * Hg * K  # q
        + T_flat * H * V  # o
    ) * 2  # bf16

    ref_name, ref_out = _reference_spec(
        verification, kernel="fused", inp=inp, scale=scale
    )
    rows = _bench_candidates(
        candidates,
        flops=flops,
        nbytes=nbytes,
        label=f"fused_k5k6 H={H} N={N} T={T_flat}",
        baseline=_BASELINES[baseline][0],
        # Ours: the fused kernel in every requested tile, plus the routing
        # wrapper. hip / flydsl_opt are upstream and only warn.
        strict=tuple(ours),
        ref_name=ref_name,
        ref_out=ref_out,
    )
    return rows, {"auto routes": "fused" if routes_fused else "separate"}


def bench_k5(
    model_tag, H, Hg, T_flat, N, gate, *, baseline, instances, verification=None
):
    """Triton K5 vs HIP K5 vs both FlyDSL K5 kernels -- the state scan alone.

    Candidates: ``triton``, ``hip`` (hand-written HIP/C++), ``flydsl_opt``
    (kernel 1, the HIP-aligned FlyDSL port) and one ``flydsl_vk[...]`` per
    requested instance (kernel 2, the gfx942-tuned build reached through
    ``chunk_gated_delta_rule_fwd_h_flydsl``). ``flydsl_opt`` vs ``flydsl_vk``
    is the comparison that decides whether kernel (2) should be routed into
    production.

    No K6 output here, so the h snapshots and v_new are drained to HBM as the
    separate pipeline requires. That HBM traffic is exactly what fusing K6
    removes, so these numbers are the baseline the fused path is trying to beat
    -- they are NOT comparable to the fused rows (different work, different
    byte counts).

    Kernel (1) takes no tile argument -- it resolves BV internally -- so it
    always appears exactly once regardless of ``instances``.
    """
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl,
        chunk_gated_delta_rule_fwd_h_flydsl_opt,
    )
    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_fwd_h_opt_vk as k5_triton,
    )

    K = V = 128
    BT = 64

    inp, seq_lens, safe_meta = _prep_case(H, Hg, T_flat, N, gate, K=K, V=V, BT=BT)
    # The HIP K5 and the FlyDSL "opt" port both want g as 3-D [B, H, T]
    # head-major; ``inp["g"]`` is the 2-D [H, T_flat] form the vk kernel takes.
    g_hm3 = inp["g"].unsqueeze(0) if inp["g"] is not None else None

    # ``gk`` is NOT in ``common``: HIP K5 rejects the batched [B, T, H, K] form
    # the Triton/FlyDSL paths accept and demands flat [T, H, K]. Sharing one
    # spelling across all four candidates is what made every ``gk`` (KDA) shape
    # raise here while the ``g`` shapes passed.
    common = {
        "k": inp["k"],
        "w": inp["w_hm"],
        "u": inp["u_hm"],
        "initial_state": inp["h0"],
        "output_final_state": True,
        "save_new_value": True,
        "cu_seqlens": inp["cu"],
        "use_exp2": _USE_EXP2,
        "prefill_metadata": safe_meta,
    }

    def _run_triton():
        return k5_triton(g=inp["g"], gk=inp["gk"], **common)

    def _run_hip():
        return chunk_gated_delta_rule_fwd_h_hip_fn(
            g=g_hm3, gk=inp["gk_flat"], g_head_major=True, **common
        )

    def _run_flydsl_opt():
        # Kernel (1): the HIP-aligned FlyDSL port. No ``variant=`` -- it picks
        # BV internally (tuned CSV -> _hipeq_select_bv).
        return chunk_gated_delta_rule_fwd_h_flydsl_opt(
            g=g_hm3, gk=inp["gk"], g_head_major=True, **common
        )

    def _make_flydsl_vk(variant):
        def _run():
            return chunk_gated_delta_rule_fwd_h_flydsl(
                g=inp["g"], gk=inp["gk"], variant=variant, **common
            )

        return _run

    candidates = {
        "triton": _run_triton,
        "hip": _run_hip,
        "flydsl_opt": _run_flydsl_opt,
    }
    ours = []
    for tag, variant in instances:
        name = f"flydsl_vk[{tag}]"
        candidates[name] = _make_flydsl_vk(variant)
        ours.append(name)

    # FLOPs: K5 only = GEMM1 (w@h^T) + GEMM2 (k^T@v_new) = 4*BT*K*V per chunk/head.
    n_chunks = sum(-(-s // BT) for s in seq_lens)
    flops = 4 * H * n_chunks * BT * K * V
    # bytes: k, w, u read; v_new and the h snapshots written out to HBM.
    nbytes = (
        T_flat * Hg * K  # k
        + T_flat * H * K  # w
        + T_flat * H * V  # u
        + T_flat * H * V  # v_new
        + n_chunks * H * V * K  # h snapshots (one per chunk, not per sequence)
    ) * 2  # bf16

    ref_name, ref_out = _reference_spec(
        verification, kernel="k5", inp=inp, scale=K**-0.5
    )
    rows = _bench_candidates(
        candidates,
        flops=flops,
        nbytes=nbytes,
        label=f"k5 H={H} N={N} T={T_flat}",
        baseline=_BASELINES[baseline][1],
        # Ours: the vk kernel in every requested tile. hip and flydsl_opt are
        # upstream and only warn.
        strict=tuple(ours),
        out_of=lambda ret: ret[0],  # (h, v_new, final_state) -> compare h
        ref_name=ref_name,
        ref_out=ref_out,
    )
    return rows, {}


def _collect_env_info() -> dict:
    """GPU / toolchain / repo versions, for the report header.

    Every probe is individually guarded: a missing ``hipcc`` or a checkout with
    no git metadata should degrade one field to "N/A", never abort a sweep that
    has already spent GPU time. The guards name the failure modes they cover
    rather than catching everything, so a genuine bug in here still surfaces.
    """
    info: dict = {}

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_arch"] = get_gfx()
        info["cu_count"] = props.multi_processor_count
    else:
        info["gpu"] = info["gpu_arch"] = info["cu_count"] = "N/A"

    info["torch"] = torch.__version__
    for mod in ("triton", "flydsl"):
        try:
            # Not installed, or installed without a __version__.
            info[mod] = __import__(mod).__version__
        except (ImportError, AttributeError):
            info[mod] = "N/A"

    # OSError covers "hipcc not on PATH"; SubprocessError covers a nonzero
    # exit; IndexError covers the fallback indexing empty output.
    try:
        hip_ver = subprocess.check_output(
            ["hipcc", "--version"], stderr=subprocess.STDOUT, text=True
        )
        info["hip"] = next(
            (
                ln.strip()
                for ln in hip_ver.splitlines()
                if "HIP version" in ln or "ROCm" in ln
            ),
            hip_ver.splitlines()[0].strip(),
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        info["hip"] = "N/A"

    # ``--dirty`` matters more than the hash itself: an A/B run against
    # uncommitted kernel edits is not reproducible from the commit alone.
    try:
        info["git_commit"] = subprocess.check_output(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "describe",
                "--always",
                "--dirty",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = "N/A"

    info["timing"] = f"hipgraph, {_NUM_ITERS} iters/capture, {_NUM_WARMUP} warmup"
    return info


def _format_env_section(env: dict) -> str:
    return "## Environment\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in env.items())


def _shape_label(shape) -> str:
    model_tag, H, Hg, T_flat, N, gate = shape
    return f"{model_tag} H={H} Hg={Hg} T={T_flat} N={N} gate={gate}"


def _list_shapes() -> str:
    lines = [f"{'#':>3}  {'shape':<52}  HxN", "-" * 66]
    for i, s in enumerate(_SWEEP_SHAPES, 1):
        lines.append(f"{i:>3}  {_shape_label(s):<52}  {s[1] * s[4]}")
    return "\n".join(lines)


def _shape_selector(s: str) -> tuple[int, int]:
    """Parse ``N`` or ``START-END`` (1-based, inclusive) into ``(lo, hi)``."""
    parts = s.split("-")
    try:
        if len(parts) == 1:
            v = int(parts[0])
            return (v, v)
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"--shape expects 'N' or 'START-END' (1-based); got {s!r}"
    )


def _select_shapes(parser, args) -> list[tuple[int, tuple]]:
    """``[(1-based index, shape), ...]`` for the requested subset."""
    n = len(_SWEEP_SHAPES)
    if args.shape:
        picked = []
        for lo, hi in args.shape:
            if not (1 <= lo <= hi <= n):
                parser.error(f"--shape must satisfy 1 <= START <= END <= {n}")
            picked.extend(range(lo, hi + 1))
        # dict.fromkeys: de-duplicate overlapping ranges, keep the given order.
        return [(i, _SWEEP_SHAPES[i - 1]) for i in dict.fromkeys(picked)]

    gate_set = set(args.gate)
    if args.T == "all":
        t_set = set(_SWEEP_T_VALUES)
    else:
        try:
            t_set = {int(t) for t in args.T}
        except ValueError:
            parser.error(f"--T takes integers or 'all'; got {args.T}")
        unknown = t_set - set(_SWEEP_T_VALUES)
        if unknown:
            parser.error(
                f"no shapes with T_flat in {sorted(unknown)}; "
                f"available: {_SWEEP_T_VALUES}"
            )
    return [
        (i, s)
        for i, s in enumerate(_SWEEP_SHAPES, 1)
        if s[5] in gate_set and s[3] in t_set
    ]


def _summary_row(idx, shape, rows, baseline_name, extra):
    """One summary row: the fastest instance we own, next to the baseline.

    Every candidate we own competes, ``auto_dispatch`` included -- it is what
    production actually runs, so a summary it cannot appear in would not answer
    "what do we ship on this shape". Note that on the fused route it and
    ``fused[auto:*]`` are the same kernel, so which of the two wins is noise;
    the name each carries says which path was taken.
    """
    ours = [r for r in rows if r["_ours"]]
    base = next(r for r in rows if r["instance"] == baseline_name)
    best = min(ours, key=lambda r: r["_us"]) if ours else base
    # The check column carries the reference in its name (``check (vs ref)``),
    # which varies with --verification.
    check_key = next(k for k in best if k.startswith("check"))
    row = {
        "#": idx,
        "shape": _shape_label(shape),
        "HxN": shape[1] * shape[4],
        "best instance": best["instance"],
        "best us": best["us"],
        f"{baseline_name} us": base["us"],
        f"best vs {baseline_name}": (
            f"{base['_us'] / best['_us']:.2f}x" if best["_us"] > 0 else "n/a"
        ),
        check_key: best[check_key],
    }
    row.update(extra)
    return row


def main(argv=None):
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fused GDN K5+K6 unsupported on %s; skipping", get_gfx())
        return 0

    parser = argparse.ArgumentParser(
        prog="bench_gdn_fused_k5k6.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused GDN K5+K6 perf sweep",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the preset shapes with their 1-based indices and exit.",
    )
    parser.add_argument(
        "--shape",
        type=_shape_selector,
        nargs="*",
        default=None,
        metavar="N|START-END",
        help=(
            "Run only these preset shapes, by 1-based index (see --list).\n"
            "Accepts single indices and inclusive ranges, e.g. '3 7-9'.\n"
            "Takes precedence over --gate / --T."
        ),
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=_INSTANCE_AUTO,
        metavar="TAG",
        help=(
            f"Which build of our kernel to time: '{_INSTANCE_AUTO}' (the tile\n"
            f"the H*N heuristic selects, the default), '{_INSTANCE_ALL}' (every\n"
            f"tile legal for this V), or an explicit tag from {list(K5_VARIANTS)}."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="triton",
        choices=sorted(_BASELINES),
        help=(
            "Reference implementation. Defines both the correctness reference\n"
            "and the denominator of every speedup. In --kernel fused it means\n"
            "that K5 followed by Triton K6. Default: triton."
        ),
    )
    parser.add_argument(
        "--verification",
        type=str,
        default=None,
        choices=_VERIFY_CHOICES,
        help=(
            "What to check correctness against, independently of --baseline\n"
            "(which stays the perf denominator). 'ref' is the pure-PyTorch\n"
            "reference and checks every candidate including the baseline; the\n"
            "others name a reference implementation. Default: the --baseline\n"
            "candidate, which is then trivially correct against itself."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        metavar="PATH",
        help="Also write the per-case tables and the summary to this file, as markdown.",
    )
    parser.add_argument(
        "--gate",
        type=str,
        nargs="*",
        default=["g", "gk"],
        choices=["g", "gk"],
        help="Gate type(s) to include in the sweep. Ignored when --shape is given.",
    )
    parser.add_argument(
        "--T",
        type=str,
        nargs="*",
        default="all",
        help=(
            f"T_flat values to include, or 'all'. Available: {_SWEEP_T_VALUES}.\n"
            f"Ignored when --shape is given. Default: all."
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        nargs="*",
        default=["fused"],
        choices=["fused", "k5"],
        help=(
            "Which kernel(s) to sweep. 'fused' (default) = K5+K6 fused vs the\n"
            "Triton/HIP two-kernel baselines. 'k5' = the state scan alone\n"
            "(Triton vs HIP vs FlyDSL). Pass both to run both sweeps; their\n"
            "rows are NOT comparable (k5 does less work and writes h to HBM)."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        print(_list_shapes())
        return 0

    if args.instance not in (_INSTANCE_AUTO, _INSTANCE_ALL, *K5_VARIANTS):
        parser.error(
            f"unknown instance {args.instance!r}; choices: "
            f"{[_INSTANCE_AUTO, _INSTANCE_ALL, *K5_VARIANTS]}"
        )

    shapes = _select_shapes(parser, args)
    if not shapes:
        parser.error("no shapes match the given filters; the sweep would be empty")

    import pandas as pd

    def _table(rows):
        # The leading-underscore keys are bookkeeping for the summary.
        public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
        return pd.DataFrame(public).to_markdown(index=False)

    env = _collect_env_info()
    print(f"\n{_format_env_section(env)}\n", flush=True)

    # Console and file want opposite orders. On the console the summary can
    # only come last -- it is not known until the last case has run. In the
    # file it goes first, so a reader lands on the conclusion rather than
    # scrolling past N per-case tables to reach it. Hence the per-kernel
    # sections are assembled separately from what is streamed to stdout.
    report = []
    for kernel in args.kernel:
        title = "fused GDN K5+K6" if kernel == "fused" else "GDN K5 (state scan)"
        baseline_name = _BASELINES[args.baseline][0 if kernel == "fused" else 1]

        cases = []
        summary = []
        for idx, shape in shapes:
            model_tag, H, Hg, T_flat, N, gate = shape
            instances = _resolve_instances(
                args.instance,
                kernel=kernel,
                H=H,
                Hg=Hg,
                V=128,
                T_flat=T_flat,
                N=N,
                is_varlen=N > 1,
            )
            bench = bench_fused_k5k6 if kernel == "fused" else bench_k5
            rows, extra = bench(
                model_tag,
                H,
                Hg,
                T_flat,
                N,
                gate,
                baseline=args.baseline,
                instances=instances,
                verification=args.verification,
            )
            head = f"## [{idx}] {_shape_label(shape)}"
            if extra:
                head += "  (" + ", ".join(f"{k}: {v}" for k, v in extra.items()) + ")"
            case_md = f"{head}\n\n{_table(rows)}\n"
            print(f"\n{case_md}", flush=True)
            cases.append(case_md)
            summary.append(_summary_row(idx, shape, rows, baseline_name, extra))

        summary_md = f"## Summary -- best instance per shape\n\n{_table(summary)}\n"
        heading = f"# {title} ({get_gfx()}, baseline: {baseline_name})\n"
        print(f"\n{heading}\n{summary_md}", flush=True)
        report.append(heading)
        report.append(summary_md)
        report.extend(cases)

    if args.output_path:
        # Environment last: it is the provenance of the numbers above, and
        # putting it between the summary and the per-case tables would push
        # the tables below the fold for no benefit.
        with open(args.output_path, "w") as f:
            f.write("\n".join([*report, _format_env_section(env), ""]))
        print(f"markdown report written to {args.output_path}")

    # Correctness gate: only kernels we own can fail the run. Upstream
    # hip / flydsl_opt mismatches are reported as warnings and do not affect
    # the exit status.
    if _STRICT_FAILURES:
        aiter.logger.error(
            "%d case(s) where our kernel disagreed with the baseline:\n  %s",
            len(_STRICT_FAILURES),
            "\n  ".join(_STRICT_FAILURES),
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
