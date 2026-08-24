# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests and perf sweep for the FlyDSL fused GDN K5+K6 forward.

Verifies ``chunk_gated_delta_rule_fwd_h_o_flydsl`` (inter-chunk state scan fused
with the inter/intra-chunk output) against a known-correct reference:

    reference o  =  Triton K6( q, k, v_new_ref, h_ref, g )

where ``h_ref`` / ``v_new_ref`` come from the pure-PyTorch K5 reference
(``ref_chunk_gated_delta_rule_fwd_h``). This decouples the fused output check
from the FlyDSL K5 kernel itself (whose correctness is covered by
``test_flydsl_linear_attention_prefill.py``).

Two levels:
  * ``test_fused_unit``     — call the fused wrapper directly.
  * ``test_fused_pipeline`` — call the end-to-end pipeline with
    ``fusion=K5K6Fusion.ALWAYS`` and compare to the NEVER/Triton baseline.

Run as a script for the perf sweep (not collected by pytest):
    PYTHONPATH=. python op_tests/test_flydsl_gdn_fused_k5k6.py                  # fused
    PYTHONPATH=. python op_tests/test_flydsl_gdn_fused_k5k6.py --kernel k5      # state scan
    PYTHONPATH=. python op_tests/test_flydsl_gdn_fused_k5k6.py \
        --kernel k5 --k5-variants all                                # per-variant K5
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest
import sys

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL fused K5+K6 tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        K5_VARIANTS,
        chunk_gated_delta_rule_fwd_h_o_flydsl,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk,
    )
    from op_tests.test_flydsl_linear_attention_prefill import (
        ref_chunk_gated_delta_rule_fwd_h,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Unable to import FlyDSL fused K5+K6 dependencies: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

_RMSE_TOL = 5e-2  # same tolerance as the K5 suite


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt()
    denom = b.float().pow(2).mean().sqrt() + 1e-8
    return (diff / denom).item()


def _make_cu_seqlens(seq_lens: list[int], device="cuda") -> torch.Tensor:
    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate, *, device="cuda"):
    """Build fused-forward inputs.

    Returns a dict with both token-major (reference) and head-major (kernel)
    layouts of w/u, plus q/k, the chosen gate tensor, h0, and cu_seqlens.
    """
    dtype = torch.bfloat16
    B = 1
    N = len(seq_lens)
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None

    q = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    w_tm = torch.randn(B, T_flat, H, K, dtype=dtype, device=device) * 0.1
    u_tm = torch.randn(B, T_flat, H, V, dtype=dtype, device=device) * 0.1
    w_hm = w_tm.permute(0, 2, 1, 3).contiguous()
    u_hm = u_tm.permute(0, 2, 1, 3).contiguous()

    g = gk = None
    if gate == "g":
        g = (
            (torch.randn(H, T_flat, dtype=torch.float32, device=device).abs() * -0.5)
            .cumsum(dim=1)
            .contiguous()
        )
    else:  # "gk"
        gk = (
            (torch.randn(T_flat, H, K, dtype=torch.float32, device=device).abs() * -0.1)
            .cumsum(dim=0)
            .contiguous()
        )

    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    # ``g`` is built 2-D head-major [H, T_flat] -- B is folded away (B == 1
    # here, and varlen flattens the batch regardless). The kernel accepts that:
    # it only checks ``g.shape[-2:] == (H, T_flat)``. The pure-PyTorch
    # reference, however, indexes ``g[b_idx, i_h]`` and needs the explicit batch
    # dim, so expose a 3-D view for it rather than reshaping at each call site.
    g_ref = g.unsqueeze(0) if g is not None else None

    return {
        "q": q,
        "k": k,
        "w_tm": w_tm,
        "u_tm": u_tm,
        "w_hm": w_hm,
        "u_hm": u_hm,
        "g": g,
        "g_ref": g_ref,
        "gk": gk,
        "h0": h0,
        "cu": cu,
        "N": N,
        "H": H,
        "Hg": Hg,
        "K": K,
        "V": V,
        "T_flat": T_flat,
    }


def _reference_o(inp, *, scale, use_exp2):
    """reference o via pure-PyTorch K5 ref then Triton K6.

    K5 ref produces token-major v_new [B, T, H, V] and h [B, NT, H, V, K];
    Triton K6 expects head-major v [B, H, T, V], so permute v_new. The K6 gate
    is scalar-``g`` only (KDA folds gk into K5), so pass g through and gk=None.
    """
    h_ref, v_new_ref, _ = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )
    # token-major [B, T, H, V] -> head-major [B, H, T, V]
    v_hm = v_new_ref.permute(0, 2, 1, 3).contiguous().to(inp["u_tm"].dtype)
    o = inp["u_tm"].new_empty(inp["u_tm"].shape)  # [B, T, H, V]
    chunk_fwd_o_opt_vk(
        q=inp["q"],
        k=inp["k"],
        v=v_hm,
        o=o,
        h=h_ref.to(inp["u_tm"].dtype),
        g=inp["g"],
        scale=scale,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
    )
    return o


# --------------------------------------------------------------------------- #
# Unit test: fused wrapper vs reference
# --------------------------------------------------------------------------- #
# BV in {16, 32, 64} plus the wave-widened bv64w8 (NR_SPLIT=2, splits b_A across
# the V-split waves) are all supported. ``None`` exercises the auto heuristic.
@pytest.mark.parametrize("gate", ["g", "gk"])
@pytest.mark.parametrize(
    "H,Hg",
    [(12, 12), (24, 24), (4, 2)],  # MHA (KDA), MHA (KDA TP4), GQA (GDN)
)
@pytest.mark.parametrize("seq_lens", [[512], [512, 512], [640, 384, 512]])
@pytest.mark.parametrize("variant", ["bv16", "bv32", "bv64", "bv64w8", None])
def test_fused_unit(gate, H, Hg, seq_lens, variant):
    """Fused wrapper output matches pure-PyTorch K5 ref + Triton K6."""
    K = V = 128
    T_flat = sum(seq_lens)
    scale = K**-0.5
    use_exp2 = False  # inputs are natural-log gates (see K5 bench rationale)

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)

    o_ref = _reference_o(inp, scale=scale, use_exp2=use_exp2)

    o_fused, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        gk=inp["gk"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=False,
        cu_seqlens=inp["cu"],
        use_exp2=use_exp2,
        variant=variant,
    )

    ratio = _rmse_ratio(o_fused, o_ref)
    assert ratio < _RMSE_TOL, (
        f"fused o mismatch: rmse_ratio={ratio:.3e} "
        f"(gate={gate} H={H} Hg={Hg} seqs={seq_lens} variant={variant})"
    )


def test_fused_final_state():
    """output_final_state=True returns a final state matching the K5 ref."""
    H = Hg = 12
    K = V = 128
    seq_lens = [512, 512]
    T_flat = sum(seq_lens)
    scale = K**-0.5
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, "g")

    (
        _,
        fs_ref,
    ) = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"],
        w=inp["w_tm"],
        u=inp["u_tm"],
        g=inp["g_ref"],
        gk=inp["gk"],
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        g_head_major=True,
    )[0::2]

    _, fs_fused = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"],
        k=inp["k"],
        w=inp["w_hm"],
        u=inp["u_hm"],
        g=inp["g"],
        scale=scale,
        initial_state=inp["h0"],
        output_final_state=True,
        cu_seqlens=inp["cu"],
        use_exp2=False,
    )
    assert fs_fused is not None
    ratio = _rmse_ratio(fs_fused, fs_ref)
    assert ratio < _RMSE_TOL, f"final_state mismatch: rmse_ratio={ratio:.3e}"


# --------------------------------------------------------------------------- #
# Pipeline tests: the K5K6Fusion API on chunk_gated_delta_rule_fwd_opt_vk.
# --------------------------------------------------------------------------- #
def _pipeline_inputs(H, Hg, seq_lens, device="cuda"):
    K = V = 128
    T_flat = sum(seq_lens)
    N = len(seq_lens)
    dtype = torch.bfloat16
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None
    q = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(1, T_flat, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(1, T_flat, H, dtype=torch.float32, device=device) * -0.5
    beta = torch.randn(1, T_flat, H, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu,
    )


@pytest.mark.parametrize("H,Hg", [(4, 2), (8, 4)])
@pytest.mark.parametrize("seq_lens", [[512], [512, 512]])
def test_fused_pipeline(H, Hg, seq_lens):
    """fusion=ALWAYS (FlyDSL backend) matches the pure-Triton baseline."""
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import K5K6Fusion

    common = _pipeline_inputs(H, Hg, seq_lens)
    # Baseline: pure Triton (no FlyDSL backend -> fusion ignored regardless).
    _, o_base, fs_base = chunk_gated_delta_rule_fwd_opt_vk(**common)
    # Force the fused FlyDSL kernel. NOTE: fusion is gated behind
    # use_chunk_flydsl, so both flags are required.
    _, o_fused, fs_fused = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl=True, fusion=K5K6Fusion.ALWAYS, **common
    )

    ratio_o = _rmse_ratio(o_fused, o_base)
    assert ratio_o < _RMSE_TOL, f"pipeline o mismatch: rmse_ratio={ratio_o:.3e}"
    ratio_fs = _rmse_ratio(fs_fused, fs_base)
    assert ratio_fs < _RMSE_TOL, f"pipeline final_state mismatch: {ratio_fs:.3e}"


# --------------------------------------------------------------------------- #
# Fused selection rules (both closed-form): H*N variant rule + fill>=0.45 routing
# --------------------------------------------------------------------------- #
def test_fused_variant_hn_rule():
    """``select_fused_variant`` is the H*N tile rule: <=32 bv16, <=80 bv32, else
    bv64w8; None only if BV illegal for V."""
    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
        select_fused_variant,
    )

    V = 128  # all BVs legal
    # (H, N, expected tag) spanning both sides of the 32 and 80 boundaries.
    cases = [
        (4, 1, "bv16"),
        (32, 1, "bv16"),
        (8, 4, "bv16"),  # H*N <= 32
        (8, 8, "bv32"),
        (16, 4, "bv32"),
        (12, 4, "bv32"),  # 32 < H*N <= 80
        (12, 6, "bv32"),
        (16, 5, "bv32"),  # H*N = 72 / 80: bv32, not bv64w8
        (16, 8, "bv64w8"),
        (96, 1, "bv64w8"),
        (24, 4, "bv64w8"),  # H*N > 80
    ]
    for H, N, exp in cases:
        got = select_fused_variant(H=H, N=N, V=V)
        assert got == exp, f"H={H} N={N} H*N={H * N}: got {got}, want {exp}"
    # Legality fallback: bv64w8 needs a legal BV=64; V=16 makes it illegal.
    assert select_fused_variant(H=96, N=8, V=16) is None


def test_fused_selection_heuristic():
    """``should_use_fused_gfx942`` fuses iff ceil(V/BV_run)*N*H/CU >= 0.45, with
    BV_run from the same H*N variant rule the launcher uses."""
    import math

    from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
        select_fused_variant,
    )
    from aiter.ops.flydsl.kernels.k5_variants import _bv_of_variant
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _ARCH,
        _FUSED_MIN_FILL,
        _device_cu_count,
        should_use_fused_gfx942,
    )

    if _ARCH != "gfx942":
        pytest.skip(f"fusion heuristic is gfx942-only; arch={_ARCH}")

    # Pin the calibrated threshold: a silent change here would reroute shapes.
    assert _FUSED_MIN_FILL == 0.45

    cu = _device_cu_count()
    V = 128
    for H, N in [
        (24, 8),
        (12, 8),
        (24, 4),
        (12, 4),
        (4, 8),
        (96, 1),
        (12, 1),
        (8, 1),
    ]:
        tag = select_fused_variant(H=H, N=N, V=V)
        bv = _bv_of_variant(tag) if tag else 64
        fill = math.ceil(V / bv) * N * H / cu
        expect = fill >= _FUSED_MIN_FILL
        got = should_use_fused_gfx942(H=H, N=N, V=V)
        assert (
            got == expect
        ), f"H={H} N={N} bv={bv} fill={fill:.2f}: got {got}, want {expect}"


@pytest.mark.parametrize(
    "H,Hg,seq_lens,expect_fused",
    [
        (24, 24, [512] * 8, True),  # bv64w8 fill=2.53 -> fused
        (8, 4, [512] * 1, False),  # bv16 N=1 fill=0.21 -> separate
    ],
)
def test_fused_auto_routing(H, Hg, seq_lens, expect_fused):
    """fusion=AUTO (with use_chunk_flydsl) routes by the heuristic, stays correct.

    High-fill routes to the fused kernel (matches the pure-Triton baseline within
    tolerance). Low-fill routes to the separate FlyDSL-K5 + Triton-K6 path, which
    differs from pure Triton only in the K5 backend -- still within tolerance.
    Both must match; the routing itself is asserted via should_use_fused_gfx942.
    """
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _ARCH,
        should_use_fused_gfx942,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import K5K6Fusion

    N = len(seq_lens)
    if _ARCH == "gfx942":
        assert (
            should_use_fused_gfx942(H=H, N=N, V=128) is expect_fused
        ), "heuristic prediction changed; update the test expectation"

    common = _pipeline_inputs(H, Hg, seq_lens)
    _, o_base, _ = chunk_gated_delta_rule_fwd_opt_vk(**common)  # pure Triton
    _, o_auto, _ = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl=True, fusion=K5K6Fusion.AUTO, **common
    )
    ratio = _rmse_ratio(o_auto, o_base)
    assert ratio < _RMSE_TOL, f"auto-routed o mismatch: rmse={ratio:.3e}"


# --------------------------------------------------------------------------- #
# Perf sweep (run via __main__)
# --------------------------------------------------------------------------- #
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
]

SUPPORTED_GFX = ["gfx942"]

_USE_EXP2 = False  # gates are natural-log domain; see bench_chunk_gdn_fwd.py


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


# Set by _bench_candidates whenever a kernel we own disagrees with the Triton
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
    extra=None,
):
    """Time each candidate against ``baseline``, collect one row.

    ``baseline`` names the reference candidate explicitly (it must be a key of
    ``candidates``) rather than relying on dict order -- every correctness and
    speedup number in the row is relative to it, so it should not be implicit.
    It is run first regardless of where it sits in ``candidates``.

    ``strict`` is the set of candidate names we own. Only those gate the run:

      * ``strict`` candidate outside tolerance -> **FAIL** (logged at error
        level, recorded in ``_STRICT_FAILURES``, nonzero exit from ``main``).
      * any other candidate outside tolerance -> **warn** only. ``hip`` and
        ``flydsl_opt`` are upstream implementations.

    ``out_of`` maps a candidate's return value to the tensor to compare; K5
    returns ``(h, v_new, final_state)`` while the fused path returns ``o``.
    ``extra`` is merged into the row first (e.g. the resolved variant tag).

    Column scheme: two columns per candidate -- ``us`` and ``vs <baseline>``.
    TFLOPS / TB/s are deliberately NOT per-candidate: ``flops`` and ``nbytes``
    are constant for the row (both reported once), so those are pure functions
    of ``us`` and would double the width for no information. Correctness
    collapses to a single ``check`` column; ``checkAllclose`` already logs the
    per-candidate detail, and its return value is the RATIO of elements outside
    tolerance (0 == clean), not an error magnitude.
    """
    if baseline not in candidates:
        raise KeyError(f"baseline {baseline!r} is not one of {list(candidates)}")
    unknown = set(strict) - set(candidates)
    if unknown:
        raise KeyError(f"strict names not in candidates: {sorted(unknown)}")

    ret = {"gfx": get_gfx(), "baseline": baseline, **(extra or {})}
    ref_out = None
    base_us = None
    errs = {}
    # Baseline first, whatever the dict order -- it defines ref_out and base_us.
    for name in [baseline] + [n for n in candidates if n != baseline]:
        fn = candidates[name]
        # Correctness: minimal eager run just to get the output tensor.
        out, _ = run_perftest(fn, num_iters=2, num_warmup=_NUM_WARMUP)
        out = out_of(out)
        if ref_out is None:
            ref_out = out
        errs[name] = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} vs {baseline}: {label}",
        )

        us = _graph_time_us(fn)
        aiter.logger.info(f"avg: {us:.3f} us/iter with hipgraph")
        if base_us is None:
            base_us = us
        ret[f"{name} us"] = us
        ret[f"{name} vs {baseline}"] = (
            f"{(base_us / us):.2f}x" if us > 0 else float("nan")
        )

    # flops/bytes once per row, so any candidate's TFLOPS/TB-s is one division.
    ret["flops"] = flops
    ret["bytes"] = nbytes

    over = {n: e for n, e in errs.items() if e > _ERR_RATIO_TOL}
    failed = {n: e for n, e in over.items() if n in strict}
    warned = {n: e for n, e in over.items() if n not in strict}
    parts = []
    if failed:
        msg = ", ".join(f"{n}={e:.2g}" for n, e in failed.items())
        aiter.logger.error("FAIL vs %s: %s (%s)", baseline, msg, label)
        _STRICT_FAILURES.append(f"{label}: {msg}")
        parts.append("FAIL (tolerance exceeded): " + msg)
    if warned:
        msg = ", ".join(f"{n}={e:.2g}" for n, e in warned.items())
        aiter.logger.warning(
            "known upstream mismatch vs %s: %s (%s)", baseline, msg, label
        )
        parts.append("warn (tolerance exceeded): " + msg)
    if not parts:
        worst = max(errs.values(), default=0.0)
        parts.append("ok" if worst == 0 else f"ok(max {worst:.1g})")
    ret[f"check (vs {baseline})"] = " | ".join(parts)
    return ret


@benchmark()
def bench_fused_k5k6(model_tag, H, Hg, T_flat, N, gate):
    """Triton K5+K6 baseline vs HIP K5+K6, FlyDSL fused always, and FlyDSL fused auto.

    All candidates are timed with HIP graph capture, which eliminates Python
    dispatch overhead and measures pure GPU kernel time.

    Triton K5, Triton K6, and HIP K5 all call prefill_metadata.validate() on
    every invocation, which raises during graph capture.  A _CaptureSafeMeta
    proxy pre-validates once before capture and no-ops subsequent calls so all
    four candidates can be graph-captured uniformly.
    """
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_o_auto,
        chunk_gated_delta_rule_fwd_h_o_flydsl,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt_vk as k5_triton,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk as k6_triton,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import K5K6Fusion

    K = V = 128
    BT = 64
    scale = K**-0.5

    seq_lens = [T_flat // N] * (N - 1) + [T_flat - (T_flat // N) * (N - 1)]
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)
    q = torch.randn(1, T_flat, Hg, K, dtype=torch.bfloat16, device="cuda") * 0.1
    # The HIP K5 and the FlyDSL "opt" port both want g as 3-D [B, H, T]
    # head-major; ``inp["g"]`` is the 2-D [H, T_flat] form the vk kernel takes.
    g_hm3 = inp["g"].unsqueeze(0) if inp["g"] is not None else None
    o_triton = inp["u_tm"].new_empty(1, T_flat, H, V)
    o_hip = inp["u_tm"].new_empty(1, T_flat, H, V)
    o_opt = inp["u_tm"].new_empty(1, T_flat, H, V)
    o_always = inp["u_tm"].new_empty(1, T_flat, H, V)
    o_auto = inp["u_tm"].new_empty(1, T_flat, H, V)

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
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o_triton,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o_triton

    def _run_hip():
        h, v_new, _ = chunk_gated_delta_rule_fwd_h_hip_fn(
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=g_hip,
            gk=inp["gk"],
            initial_state=inp["h0"],
            output_final_state=True,
            save_new_value=True,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            g_head_major=True,
            prefill_metadata=safe_meta,
        )
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o_hip,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o_hip

    def _run_flydsl_opt():
        # Kernel (1) K5 + Triton K6 -- the SEPARATE path that production runs
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
        k6_triton(
            q=q,
            k=inp["k"],
            v=v_new,
            o=o_opt,
            h=h,
            g=inp["g"],
            scale=scale,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            prefill_metadata=safe_meta,
        )
        return o_opt

    def _run_fused_always():
        chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=q,
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=inp["g"],
            gk=inp["gk"],
            scale=scale,
            initial_state=inp["h0"],
            output_final_state=False,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            o=o_always,
        )
        return o_always

    def _run_fused_auto():
        chunk_gated_delta_rule_fwd_h_o_auto(
            q=q,
            k=inp["k"],
            w=inp["w_hm"],
            u=inp["u_hm"],
            g=inp["g"],
            gk=inp["gk"],
            scale=scale,
            initial_state=inp["h0"],
            output_final_state=False,
            cu_seqlens=inp["cu"],
            use_exp2=_USE_EXP2,
            o=o_auto,
            fusion=K5K6Fusion.AUTO,
        )
        return o_auto

    # FLOPs: K5 (GEMM1+GEMM2) + K6 (GEMM3+GEMM4a+GEMM4b) per chunk per head
    n_chunks = sum(-(-s // BT) for s in seq_lens)
    per_chunk = 4 * K * V + 2 * K * V + 2 * BT * K + 2 * BT * V
    flops = H * n_chunks * BT * per_chunk
    # bytes: k,w,u (read) + v_new,h (K5→K6 handoff) + q (read) + o (written)
    nbytes = (
        T_flat * Hg * K  # k
        + T_flat * H * K  # w
        + T_flat * H * V  # u
        + T_flat * H * V  # v_new (fused keeps in LDS; baseline spills to HBM)
        + N * H * V * K  # h state
        + T_flat * Hg * K  # q
        + T_flat * H * V  # o
    ) * 2  # bf16

    candidates = {
        "triton+triton": _run_triton,  # baseline
        "hip+triton": _run_hip,
        "flydsl_opt+triton": _run_flydsl_opt,
        "fused_always": _run_fused_always,
        "fused_auto": _run_fused_auto,
    }
    # What the fused candidates will actually run: the H*N tile that
    # fused_always launches, and whether fused_auto routes to it at all.
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        _fused_bv_for_shape,
        should_use_fused_gfx942,
    )

    bv, num_waves = _fused_bv_for_shape(H=H, V=V, N=N, variant=None)
    fused_tag = f"bv{bv}w{num_waves}" if num_waves > 4 else f"bv{bv}"

    return _bench_candidates(
        candidates,
        flops=flops,
        nbytes=nbytes,
        label=f"fused_k5k6 H={H} N={N} T={T_flat}",
        baseline="triton+triton",
        # Ours: the fused K5+K6 kernel. hip+triton / flydsl_opt+triton are
        # upstream and only warn (see _bench_candidates).
        strict=("fused_always", "fused_auto"),
        extra={
            "HxN": H * N,
            "fused variant": fused_tag,
            "auto routes": (
                "fused" if should_use_fused_gfx942(H=H, N=N, V=V) else "separate"
            ),
        },
    )


@benchmark()
def bench_k5(model_tag, H, Hg, T_flat, N, gate, variants=None):
    """Triton K5 baseline vs HIP K5 and both FlyDSL K5 kernels -- the state
    scan in isolation.

    Candidates: ``triton`` (baseline), ``hip`` (hand-written HIP/C++),
    ``flydsl_opt`` (kernel 1, the HIP-aligned FlyDSL port) and ``flydsl_vk``
    (kernel 2, the gfx942-tuned build reached through
    ``chunk_gated_delta_rule_fwd_h_flydsl``). ``flydsl_opt`` vs ``flydsl_vk``
    is the comparison that decides whether kernel (2) should be routed into
    production.

    The K5-only counterpart of ``bench_fused_k5k6``, mirroring the ``k5``
    subcommand of ``op_tests/op_benchmarks/flydsl/bench_chunk_gdn_fwd.py``: no K6
    output, so the h snapshots and v_new are drained to HBM as the separate
    pipeline requires. That HBM traffic is exactly what fusing K6 removes, so
    these numbers are the baseline the fused path is trying to beat -- they are
    NOT comparable to the fused rows (different work, different byte counts).

    ``variants`` adds explicit kernel-(2) tile tags (e.g. ``["bv16",
    "bv64w8"]``) as extra ``flydsl_vk:<tag>`` candidates alongside the
    auto-selected one; useful for re-checking the H*N selection rule. Kernel
    (1) takes no tile argument -- it resolves BV internally -- so it always
    appears exactly once. Default: auto only.
    """
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        chunk_gated_delta_rule_fwd_h_hip_fn,
    )
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl,
        chunk_gated_delta_rule_fwd_h_flydsl_opt,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h_opt_vk as k5_triton,
    )

    K = V = 128
    BT = 64

    seq_lens = [T_flat // N] * (N - 1) + [T_flat - (T_flat // N) * (N - 1)]
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)
    # The HIP K5 and the FlyDSL "opt" port both want g as 3-D [B, H, T]
    # head-major; ``inp["g"]`` is the 2-D [H, T_flat] form the vk kernel takes.
    g_hm3 = inp["g"].unsqueeze(0) if inp["g"] is not None else None

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

    common = {
        "k": inp["k"],
        "w": inp["w_hm"],
        "u": inp["u_hm"],
        "gk": inp["gk"],
        "initial_state": inp["h0"],
        "output_final_state": True,
        "save_new_value": True,
        "cu_seqlens": inp["cu"],
        "use_exp2": _USE_EXP2,
        "prefill_metadata": safe_meta,
    }

    def _run_triton():
        return k5_triton(g=inp["g"], **common)

    def _run_hip():
        return chunk_gated_delta_rule_fwd_h_hip_fn(g=g_hm3, g_head_major=True, **common)

    def _run_flydsl_opt():
        # Kernel (1): the HIP-aligned FlyDSL port. No ``variant=`` -- it picks
        # BV internally (tuned CSV -> _hipeq_select_bv).
        return chunk_gated_delta_rule_fwd_h_flydsl_opt(
            g=g_hm3, g_head_major=True, **common
        )

    def _make_flydsl_vk(tag):
        def _run():
            return chunk_gated_delta_rule_fwd_h_flydsl(
                g=inp["g"], variant=tag, **common
            )

        return _run

    candidates = {
        "triton": _run_triton,  # baseline
        "hip": _run_hip,
        "flydsl_opt": _run_flydsl_opt,
        "flydsl_vk": _make_flydsl_vk(None),  # None -> the auto (H*N rule) pick
    }
    for tag in variants or ():
        candidates[f"flydsl_vk:{tag}"] = _make_flydsl_vk(tag)

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

    # Record what the "flydsl" (auto) candidate will actually run. 
    # _resolve_variant is the same chain the launcher uses.
    from aiter.ops.flydsl.linear_attention_prefill_kernels import _resolve_variant

    auto_tag = _resolve_variant(
        None, H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=inp["cu"] is not None
    )

    return _bench_candidates(
        candidates,
        flops=flops,
        nbytes=nbytes,
        label=f"k5 H={H} N={N} T={T_flat}",
        baseline="triton",
        # Ours: the vk kernel and any explicitly pinned vk tile. hip and
        # flydsl_opt are upstream and only warn (see _bench_candidates).
        strict=tuple(n for n in candidates if n.startswith("flydsl_vk")),
        out_of=lambda ret: ret[0],  # (h, v_new, final_state) -> compare h
        extra={"HxN": H * N, "flydsl_vk variant": auto_tag},
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fused GDN K5+K6 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL fused GDN K5+K6 correctness + perf sweep",
    )
    parser.add_argument(
        "--gate",
        type=str,
        nargs="*",
        default=["g", "gk"],
        choices=["g", "gk"],
        help="Gate type(s) to include in the sweep.",
    )
    parser.add_argument(
        "--T",
        type=int,
        nargs="*",
        default=[8192, 32768],
        help="T_flat values to include.",
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
    parser.add_argument(
        "--k5-variants",
        type=str,
        nargs="*",
        default=None,
        help=(
            f"Extra explicit FlyDSL K5 variant tags to time alongside the\n"
            f"auto-selected one, or 'all'. Choices: {list(K5_VARIANTS)}.\n"
            "Only affects --kernel k5."
        ),
    )
    args = parser.parse_args()

    gate_set = set(args.gate)
    t_set = set(args.T)

    k5_variants = args.k5_variants
    if k5_variants == ["all"]:
        k5_variants = list(K5_VARIANTS)
    for tag in k5_variants or ():
        if tag not in K5_VARIANTS:
            parser.error(
                f"unknown FlyDSL K5 variant {tag!r}; choices: {list(K5_VARIANTS)}"
            )

    shapes = [s for s in _SWEEP_SHAPES if s[5] in gate_set and s[3] in t_set]

    for kernel in args.kernel:
        rows = []
        for model_tag, H, Hg, T_flat, N, gate in shapes:
            if kernel == "fused":
                rows.append(bench_fused_k5k6(model_tag, H, Hg, T_flat, N, gate))
            else:
                row = bench_k5(
                    model_tag, H, Hg, T_flat, N, gate, variants=k5_variants
                )
                # ``variants`` is a callarg so @benchmark puts it in the row;
                # it is already encoded in the flydsl_vk:<tag> column names.
                row.pop("variants", None)
                rows.append(row)
        title = "fused GDN K5+K6" if kernel == "fused" else "GDN K5 (state scan)"
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            title,
            pd.DataFrame(rows).to_markdown(index=False),
        )

    # Correctness gate: only flydsl_vk kernels can fail the run. 
    # Upstream hip / flydsl_opt mismatches are reported as warnings and
    # do not affect the exit status.
    if _STRICT_FAILURES:
        aiter.logger.error(
            "%d shape(s) where our kernel disagreed with the Triton baseline:\n  %s",
            len(_STRICT_FAILURES),
            "\n  ".join(_STRICT_FAILURES),
        )
        return 1
 
    return 0


if __name__ == "__main__":
    sys.exit(main())
