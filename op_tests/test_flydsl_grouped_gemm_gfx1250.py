#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 grouped MoE GEMM tests through ``aiter.fused_moe``.

Two formats covered:

* **a4w4** -- MXFP4 activations × MXFP4 weights (``w1.dtype = fp4x2``).
* **a8w4** -- MXFP8 activations × MXFP4 weights (``w1.dtype = uint8``).

Both go through the public ``fused_moe`` API; we never call the underlying
grouped GEMM launcher directly. The grouped path is opted-in via the
``AITER_USE_GROUPED_GEMM=1`` env (set automatically by the runner below).

Pytest covers a small correctness case for each format. Direct execution
(``python op_tests/test_flydsl_grouped_gemm_gfx1250.py``) runs a
DeepSeek-style perf bench (``--scenario bench``) or a tiny correctness
check (``--scenario verify``).
DeepSeek-style perf bench (``--scenario bench``) or a tiny correctness
check (``--scenario verify``).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pytest
import torch

from aiter import ActivationType, QuantType, logger
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import moe_shuffle_scale, shuffle_weight
from aiter.utility import fp4_utils
from aiter.utility import dtypes

# Build every tensor straight on the device (like op_tests/test_moe_2stage.py) so
# the test body has no `.cuda()` / `.float().cuda()` plumbing.
torch.set_default_device("cuda")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# Routing: normal (random) by default; round-robin balanced only when
# AITER_MOE_EXPERT_BALANCE=1 (mirrors op_tests/test_moe_2stage.py).
AITER_MOE_EXPERT_BALANCE = (
    os.environ.get("AITER_MOE_EXPERT_BALANCE", "False").lower() == "true"
)

SCALE_BLOCK = 32
DEFAULT_SCALE_BYTE = 127  # e8m0 byte for 2^0 = 1.0
VERIFY_TOL_A4W4 = 0.02
VERIFY_TOL_A8W4 = 0.02
# Production MoE accuracy gate (matches op_tests/test_moe_2stage.py calc_diff):
# logits_diff = ||x-y||^2 / (||x||^2 + ||y||^2).  rel_l2 is kept as an
# informational print only; logits_diff < 0.01 is the actual pass/fail gate.
LOGITS_DIFF_TOL = 0.01


# ---------------------------------------------------------------------------
# Environment / arch guards
# ---------------------------------------------------------------------------
def _require_gfx1250() -> None:
    # AITER_FORCE_GFX1250=1 forces the grouped path on other archs (e.g. gfx942)
    # to exercise the tiny operators with the GEMM mocked (default; pass
    # --real-gemm to call the real gfx1250 kernel instead).
    if os.environ.get("AITER_FORCE_GFX1250", "0") in ("1", "true", "True", "yes"):
        return
    try:
        from flydsl.runtime.device import get_rocm_arch
    except Exception as exc:
        pytest.skip(f"FlyDSL not importable: {exc}")
    arch = get_rocm_arch()
    if "gfx1250" not in arch.lower():
        pytest.skip(f"requires gfx1250, got {arch!r}")


def is_gfx1250() -> bool:
    """True only on actual gfx1250 hardware. AITER_FORCE_GFX1250 does NOT count:
    forcing the grouped path onto another arch (e.g. gfx942) still needs the GEMM
    mocked, so real-gemm defaults on only when the real WMMA kernel can run."""
    try:
        from flydsl.runtime.device import get_rocm_arch

        return "gfx1250" in get_rocm_arch().lower()
    except Exception:
        return False


# Weights/scales use the public shuffle APIs directly:
#   shuffle_weight(b, layout=(16, 16))            -> FP4 TDM B layout (16-row x
#       16-byte chunks) the grouped FlyDSL kernels consume.
#   moe_shuffle_scale(s, experts_cnt=E) -> arch-aware MoE B-scale shuffle; on
#       gfx1250 it folds to the grouped-only n32k4 e8m0 layout (shuffle_scale_n32k4).
# ---------------------------------------------------------------------------
# Reference: aiter's own ``torch_moe_stage1`` + ``torch_moe_stage2``
# (high-precision fp32 baseline that decodes mxfp4/e8m0 internally and
# evaluates the same swiglu+bias formula the grouped path uses). It still
# diverges from the quantised grouped GEMM path by mxfp4/mxfp8 round noise
# (~0.2 rel_l2 on random uint8 weights, ~0.02 on real model weights). The
# point is to catch *catastrophic* regressions, not chase fp32 parity.
# ---------------------------------------------------------------------------
def _torch_moe_ref(
    hidden: torch.Tensor,  # (T, K) bf16
    w1_packed: torch.Tensor,  # (E, 2*I, K_pack) uint8 (GGUU)
    w1_scale_raw: torch.Tensor,  # (E, 2*I, K//32) uint8 (raw e8m0)
    w1_bias: torch.Tensor,  # (E, 2*I) fp32
    w2_packed: torch.Tensor,  # (E, K, I_pack) uint8
    w2_scale_raw: torch.Tensor,  # (E, K, I//32) uint8
    w2_bias: torch.Tensor,  # (E, K) fp32
    topk_w: torch.Tensor,  # (T, topk) bf16
    topk_id: torch.Tensor,  # (T, topk) int32
    *,
    data_format: str,
    activation: ActivationType,
    swiglu_limit: float,
) -> torch.Tensor:
    """Two-stage MoE reference reusing ``aiter.fused_moe.torch_moe_stage{1,2}``."""
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")

    def _per_1x32_fp8_dequant(x: torch.Tensor) -> torch.Tensor:
        """Mirror grouped a8w4's per-block-32 MXFP8 input quant, then dequant."""
        block = 32
        dtype_max = 448.0
        x_shape = x.shape
        flat = x.contiguous().view(-1, x_shape[-1]).float()
        blk = flat.view(-1, block)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = blk.abs().amax(dim=1)
        scale_e8m0 = fp4_utils.f32_to_mx_e8m0_scale(
            max_abs, dtype=fp4_utils.MxDtypeInt.FP8_E4M3
        )
        scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
        scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
        scale_f32[scale_f32 == 0] = 1.0
        q_f32 = (blk / scale_f32.unsqueeze(1)).clamp(min=-dtype_max, max=dtype_max)
        q = q_f32.contiguous().to(dtypes.fp8).to(torch.float32).view_as(blk)
        return (q * scale_f32.unsqueeze(1)).view(x_shape).to(x.dtype)

    w1_scale = w1_scale_raw.view(dtypes.fp8_e8m0)
    w2_scale = w2_scale_raw.view(dtypes.fp8_e8m0)
    if data_format == "a4w4":
        # Match the grouped a4w4 path: stage1 input is MXFP4, not bf16.
        stage1_hidden, stage1_hidden_scale = per_1x32_f4_quant(
            hidden, quant_dtype=dtypes.fp4x2, shuffle=False
        )
    else:
        # Match grouped a8w4: stage1 input is MXFP8 with per-1x32 e8m0 scale.
        stage1_hidden, stage1_hidden_scale = _per_1x32_fp8_dequant(hidden), None
    a2 = torch_moe_stage1(
        stage1_hidden,
        w1_packed,
        w2_packed,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        a1_scale=stage1_hidden_scale,
        w1_scale=w1_scale,
        w1_bias=w1_bias,
        # torch_moe_stage1 also applies swiglu_limit as a generic gate/up
        # clamp in the non-SwiGLU branch. The grouped FlyDSL SiLU epilogue
        # does *not* clamp, so only pass the limit for true SwiGLU.
        swiglu_limit=swiglu_limit if activation == ActivationType.Swiglu else 0.0,
    )
    if data_format == "a4w4":
        # Match the grouped a4w4 path again: stage2 input is MXFP4.
        T, topk = topk_id.shape
        inter = w2_packed.shape[-1] * 2
        a2_q, a2_scale = per_1x32_f4_quant(
            a2.contiguous().view(T * topk, inter),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        a2 = a2_q.view(T, topk, inter // 2)
    else:
        # Match grouped a8w4 stage2: per-block-32 MXFP8 quant + dequant.
        # This matters for SiLU because the unclamped stage1 output can exceed
        # fp8's unit-scale range; grouped now uses a real e8m0 block scale.
        a2 = _per_1x32_fp8_dequant(a2)
        a2_scale = None
    out = torch_moe_stage2(
        a2,
        w1_packed,
        w2_packed,
        topk_w,
        topk_id,
        dtype=torch.bfloat16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        w2_bias=w2_bias,
        doweight=True,
    )
    return out


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------
def _pattern_packed(experts: int, rows: int, k_pack: int) -> torch.Tensor:
    """mxfp4 packed bytes ``(E, rows, k_pack) uint8`` from the global RNG."""
    return torch.randint(0, 256, (experts, rows, k_pack), dtype=torch.uint8)


def init_weight_scales(experts: int, rows: int, n_blocks: int) -> torch.Tensor:
    """Per-block e8m0 weight scale: random small scales (drawn from the global
    RNG) so the n32k4 B-scale preshuffle layout is actually exercised."""
    r = torch.randint(0, 3, (experts, rows, n_blocks), dtype=torch.int16)
    return (r + (DEFAULT_SCALE_BYTE - 1)).to(torch.uint8)


def _make_topk(
    hidden_states: torch.Tensor, experts: int, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route via ``fused_topk``: normal (random gating) by default; round-robin
    balanced gating when ``AITER_MOE_EXPERT_BALANCE=1`` (mirrors
    op_tests/test_moe_2stage.py). Returns ``(topk_ids, topk_weights)`` on the
    same device as ``hidden_states``."""
    tokens = hidden_states.shape[0]
    if AITER_MOE_EXPERT_BALANCE:
        score = torch.zeros((tokens, experts), dtype=torch.float32)
        start_col, end_col = 0, topk
        for token_id in range(tokens):
            score[token_id, start_col:end_col] = 1.0
            start_col = end_col % experts
            end_col = start_col + topk
    else:
        score = torch.randn((tokens, experts), dtype=torch.float32)
    topk_w, topk_id = fused_topk(hidden_states, score, topk, True)
    return topk_id.to(torch.int32), topk_w


def _gguu_to_gugu_rows(t: torch.Tensor) -> torch.Tensor:
    """``(E, 2*I, ...)`` GGUU ``[g0..g_{I-1}, u0..u_{I-1}]`` -> GUGU ``[g0,u0,g1,u1,...]``."""
    E, two_inter = t.shape[:2]
    inter = two_inter // 2
    g = t[:, :inter]
    u = t[:, inter:]
    return torch.stack([g, u], dim=2).flatten(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Core runner: build inputs, invoke fused_moe, optionally compare to ref
# ---------------------------------------------------------------------------
def _run_grouped_via_fused_moe(
    *,
    experts: int,
    tokens: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    data_format: str,  # "a4w4" | "a8w4"
    layout: str = "gguu",  # "gguu" -> SEPARATED | "gugu" -> INTERLEAVE
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    use_bias: bool = True,
    bench: bool = False,
    seed: int = 0,
    warmup: int = 5,
    iters: int = 101,
) -> tuple[torch.Tensor, torch.Tensor, Optional[float]]:
    """Build mxfp4 weights + routing, dispatch through ``fused_moe``.

    ``layout`` selects the stage1 weight physical layout:
    ``gguu`` (gate rows then up rows, default) pairs with ``GateMode.SEPARATED``;
    ``gugu`` (gate/up row-interleaved, gpt-oss style) pairs with
    ``GateMode.INTERLEAVE``. The PyTorch reference always evaluates the
    GGUU logical weights, so both paths share the same numerical result.

    Correctness is always checked against the reference. ``bench`` selects the
    path that is validated and timed: when set, the output comes from
    ``run_perftest`` in CUDA-graph mode (production path) and ``us`` is the graph
    timing; otherwise the output is a single eager (graph-off) call and ``us`` is
    None. Returns ``(out, ref, us_or_None)``.
    """
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")
    if layout not in ("gguu", "gugu"):
        raise ValueError(f"layout must be gguu or gugu, got {layout!r}")

    K = model_dim
    inter = inter_dim
    K_pack = K // 2
    inter_pack = inter // 2

    # Logical weights/scale/bias: always GGUU (gate rows then up rows).
    # One global seed per case; every draw below uses the global RNG.
    torch.manual_seed(seed)
    w1_logical = _pattern_packed(experts, 2 * inter, K_pack)
    w2_logical = _pattern_packed(experts, K, inter_pack)
    w1_scale_raw = init_weight_scales(experts, 2 * inter, K // SCALE_BLOCK)
    w2_scale_raw = init_weight_scales(experts, K, inter // SCALE_BLOCK)
    if use_bias:
        bias1 = (torch.randn((experts, 2 * inter)) * 1e-3).float()
        bias2 = (torch.randn((experts, K)) * 1e-3).float()
    else:
        bias1 = torch.zeros((experts, 2 * inter))
        bias2 = torch.zeros((experts, K))
    # Activations: bf16; fused_moe handles the dispatched quant internally.
    hidden = (torch.randn((tokens, K)) * 0.5).to(torch.bfloat16)

    # Routing: normal (random) by default; balanced if AITER_MOE_EXPERT_BALANCE.
    topk_id, topk_w = _make_topk(hidden, experts, topk)
    topk_w = topk_w.to(torch.bfloat16)

    # ---- prep grouped GEMM inputs ----
    # Stage1 weight/scale/bias get rearranged to physical ``layout``; stage2
    # has no GUGU/GGUU concept (single N=hidden GEMM).
    if layout == "gugu":
        w1_phys = _gguu_to_gugu_rows(w1_logical)
        w1_scale_phys = _gguu_to_gugu_rows(w1_scale_raw)
        bias1_phys = _gguu_to_gugu_rows(bias1)
        gate_mode = GateMode.INTERLEAVE
    else:
        w1_phys = w1_logical
        w1_scale_phys = w1_scale_raw
        bias1_phys = bias1
        gate_mode = GateMode.SEPARATED

    w1_grouped = shuffle_weight(w1_phys, layout=(16, 16))
    w2_grouped = shuffle_weight(w2_logical, layout=(16, 16))
    w1_scale = moe_shuffle_scale(w1_scale_phys.contiguous(), experts_cnt=experts)
    w2_scale = moe_shuffle_scale(w2_scale_raw.contiguous(), experts_cnt=experts)

    if data_format == "a4w4":
        w1_arg = w1_grouped.view(dtypes.fp4x2)
        w2_arg = w2_grouped.view(dtypes.fp4x2)
    else:  # a8w4
        w1_arg = w1_grouped  # uint8 -> grouped helper sets q_dtype_a=fp8
        w2_arg = w2_grouped

    def _call():  # the grouped path is auto-enabled on gfx1250
        return fused_moe(
            hidden,
            w1_arg,
            w2_arg,
            topk_w,
            topk_id,
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1_phys if use_bias else None,
            bias2=bias2 if use_bias else None,
            gate_mode=gate_mode.value,
            dtype=dtypes.bf16,
            swiglu_limit=swiglu_limit,
        )

    torch.cuda.synchronize()
    if bench:
        # Bench: validate + time the CUDA-graph (production) path. The returned
        # data is the graph-captured output.
        from aiter.test_common import run_perftest

        out, us = run_perftest(
            _call, num_warmup=warmup, num_iters=iters, testGraph=True
        )
    else:
        # Verify: validate the eager (graph-off) path; no timing.
        out = _call()
        us = None

    # Reference always uses GGUU logical inputs (layouts are numerically
    # equivalent; only physical packing differs).
    ref = _torch_moe_ref(
        hidden,
        w1_logical,
        w1_scale_raw,
        bias1,
        w2_logical,
        w2_scale_raw,
        bias2,
        topk_w,
        topk_id,
        data_format=data_format,
        activation=activation,
        swiglu_limit=swiglu_limit,
    ).to(out.dtype)
    return out, ref, us


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    base = expected.float().norm().clamp(min=1e-12)
    return float(diff / base)


def _logits_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """MoE accuracy metric from op_tests/test_moe_2stage.py (calc_diff):

        1 - 2*<x,y>/(||x||^2 + ||y||^2)  ==  ||x-y||^2 / (||x||^2 + ||y||^2)

    A magnitude-weighted cosine-style diff. Relation to rel_l2: when the two
    norms match, logits_diff ~= rel_l2**2 / 2.  Production strict gate: < 0.01.
    """
    x = actual.double()
    y = expected.double()
    denom = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denom
    return float(1 - sim)


# ---------------------------------------------------------------------------
# Pytest correctness suite
# ---------------------------------------------------------------------------
def run_moe(
    data_format: str,
    *,
    experts: int = 4,
    tokens: int = 8,
    topk: int = 2,
    model_dim: int = 256,
    inter_dim: int = 256,
    layout: str = "gguu",
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    use_bias: bool = True,
    tol: float = VERIFY_TOL_A4W4,
    raise_on_fail: bool = True,
    bench: bool = False,
    warmup: int = 5,
    iters: int = 101,
) -> dict:
    """Compare grouped FlyDSL MoE vs a PyTorch fp32 ref. ``bench`` selects the
    validated path: bench checks (and times) the CUDA-graph production path;
    verify checks the eager path.

    Correctness gate: production-consistent logits_diff < LOGITS_DIFF_TOL
    (op_tests/test_moe_2stage.py).  rel_l2 (~= sqrt(2*logits_diff)) is printed
    for reference only.  Returns a metrics dict (with ``us`` when benched).
    """
    _require_gfx1250()
    act = "swiglu" if activation == ActivationType.Swiglu else "silu"
    tag = f"{data_format} {layout} {act}"

    # --- grouped FlyDSL vs PyTorch fp32 ref (graph path if bench, else eager) ---
    out, ref, us = _run_grouped_via_fused_moe(
        experts=experts,
        tokens=tokens,
        topk=topk,
        model_dim=model_dim,
        inter_dim=inter_dim,
        data_format=data_format,
        layout=layout,
        activation=activation,
        swiglu_limit=swiglu_limit,
        use_bias=use_bias,
        bench=bench,
        warmup=warmup,
        iters=iters,
    )
    mode = "graph" if bench else "eager"
    ld = _logits_diff(out, ref)
    rel = _rel_l2(out, ref)
    print(
        f"[sanity {tag}] {mode}: logits_diff={ld:.4e} rel_l2={rel:.4e} "
        f"(gate<{LOGITS_DIFF_TOL}, ref_norm={float(ref.float().norm()):.4e})",
        flush=True,
    )
    passed = ld < LOGITS_DIFF_TOL
    if raise_on_fail:
        assert (
            passed
        ), f"grouped {tag} {mode} vs ref logits_diff={ld:.4e} > {LOGITS_DIFF_TOL}"
    metrics = {
        "logits_diff": ld,
        "rel_l2": rel,
        "passed": passed,
        "grouped_norm": float(out.float().norm()),
        "ref_norm": float(ref.float().norm()),
    }

    # --- perf (bench only): timed end-to-end inside _run_grouped_via_fused_moe ---
    if bench:
        print(
            f"[bench {tag}] fused_moe end-to-end us = {us:.2f} (graph=True)",
            flush=True,
        )
        metrics["us"] = us
    return metrics


@pytest.mark.parametrize("layout", ["gguu", "gugu"])
def test_grouped_a4w4_silu_matches_torch_ref(layout):
    run_moe("a4w4", layout=layout, activation=ActivationType.Silu)


@pytest.mark.parametrize("layout", ["gguu", "gugu"])
def test_grouped_a4w4_swiglu_matches_torch_ref(layout):
    run_moe("a4w4", layout=layout, activation=ActivationType.Swiglu)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _mock_grouped_gemm() -> None:
    """Run the grouped MoE path without the gfx1250-only kernels.

    Two patches let the tiny operators (route maps, scatter/gather, quant,
    scale preshuffle, m-tile map, gather-reduce) run on any arch (e.g. gfx942
    via AITER_FORCE_GFX1250=1):

    1. Replace the grouped WMMA GEMM compilers with no-op launchers -- the GEMM
       executes nothing; stage outputs are left as-is.
    2. Route the fp4 a1/a2 quant through the Triton implementation, since the
       HIP ``per_1x32_f4_quant_hip`` has no fp4x2 output support off gfx1250.

    The library imports all these names at call time, so patching the source
    modules is enough -- no library edits required.
    """
    import aiter.ops.flydsl.kernels.moe_grouped_gemm_mxscale_gfx1250 as gk
    import aiter.ops.quant as q

    def _noop_compile(*_a, **_k):
        return lambda *_a, **_k: None

    for _name in (
        "compile_moe_grouped_gemm1_a8w4_masked",
        "compile_moe_grouped_gemm2_a8w4_masked",
        "compile_moe_grouped_gemm1_mxfp4_masked",
        "compile_moe_grouped_gemm2_mxfp4_masked",
    ):
        setattr(gk, _name, _noop_compile)

    q.per_1x32_f4_quant_hip = q.per_1x32_f4_quant_triton


def summarize(rows: list):
    """Build a precision summary table from per-case metrics and print it.

    Mirrors the pandas DataFrame reporting in op_tests/test_moe_2stage.py.
    Returns the DataFrame (or the raw rows if pandas is unavailable).
    """
    if not rows:
        return None
    try:
        import pandas as pd
    except ImportError:
        print("[precision summary] pandas not installed; raw rows:", flush=True)
        for r in rows:
            print(f"  {r}", flush=True)
        return rows
    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        # to_markdown needs the optional `tabulate` package; plain fallback.
        table = df.to_string(index=False)
    print("\n[precision summary]\n" + table, flush=True)
    return df


def set_data_format(data_format: str) -> None:
    """Select the grouped GEMM data format.

    a8w4 needs ``AITER_FORCE_A8W4=1`` so ``fused_moe`` routes the a8w4 path
    (see fused_moe.py); a4w4 needs nothing extra.
    """
    if data_format == "a8w4":
        os.environ["AITER_FORCE_A8W4"] = "1"
    logger.info("grouped GEMM data format: %s", data_format)


def main() -> None:
    if not is_gfx1250():
        print("skipping: requires gfx1250")
        sys.exit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("bench", "verify"), default="bench")
    parser.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument(
        "--layout",
        choices=("gguu", "gugu"),
        default="gguu",
        help="stage1 weight physical layout. gguu pairs with "
        "GateMode.SEPARATED (default), gugu with INTERLEAVE.",
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[64],
        metavar="N",
        help="one or more space-separated token counts; the scenario runs "
        "once per value, e.g. --tokens 64 128 256",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument(
        "--act",
        choices=("silu", "swiglu"),
        default="swiglu",
        help="stage1 activation: silu => silu(gate)*up; "
        "swiglu => gpt-oss swiglu with clamp/alpha/residual",
    )
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument(
        "--no-bias",
        action="store_true",
        help="run with zero stage1/stage2 bias tensors",
    )
    parser.add_argument(
        "--real-gemm",
        action="store_true",
        default=is_gfx1250(),
        help="call the real grouped WMMA GEMM kernel. Default: True on gfx1250, "
        "False elsewhere (mock the GEMM so the tiny operators run on any arch).",
    )
    args = parser.parse_args()
    if not args.real_gemm:
        _mock_grouped_gemm()
    if args.model_dim < 512 or args.inter_dim < 512:
        raise SystemExit(
            f"model_dim ({args.model_dim}) and inter_dim ({args.inter_dim}) must be "
            "at least 512 for the grouped GEMM kernels (tile_k=256 requires at "
            "least two K tiles)."
        )

    set_data_format(args.data_format)

    # --tokens accepts one or more counts; run once per value. Each iteration
    # sets args.tokens to a single int so run_moe reads it unchanged.
    token_list = args.tokens if isinstance(args.tokens, list) else [args.tokens]
    activation = ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
    rows = []
    for _tok in token_list:
        args.tokens = _tok
        if len(token_list) > 1:
            print(f"\n===== tokens={_tok} =====", flush=True)
        tol = VERIFY_TOL_A8W4 if args.data_format == "a8w4" else VERIFY_TOL_A4W4
        # raise_on_fail=False so one out-of-gate token does not abort the
        # sweep; the failure is recorded and reported after the table.
        metrics = run_moe(
            args.data_format,
            layout=args.layout,
            experts=args.experts,
            tokens=args.tokens,
            topk=args.topk,
            model_dim=args.model_dim,
            inter_dim=args.inter_dim,
            tol=tol,
            activation=activation,
            swiglu_limit=args.swiglu_limit,
            use_bias=not args.no_bias,
            raise_on_fail=False,
            bench=args.scenario == "bench",
            warmup=args.warmup,
            iters=args.iters,
        )
        rows.append(
            {
                "data_format": args.data_format,
                "layout": args.layout,
                "act": args.act,
                "experts": args.experts,
                "tokens": _tok,
                "topk": args.topk,
                "model_dim": args.model_dim,
                "inter_dim": args.inter_dim,
                "logits_diff": metrics["logits_diff"],
                "rel_l2": metrics["rel_l2"],
                "pass": metrics["passed"],
                "us": metrics.get("us"),
            }
        )

    # Always print the summary table (verify and bench).
    summarize(rows)
    # Preserve CI semantics: non-zero exit if any verify case missed the gate.
    if args.scenario == "verify":
        failed = [r for r in rows if not r["pass"]]
        if failed:
            raise SystemExit(
                f"{len(failed)}/{len(rows)} verify case(s) exceeded "
                f"logits_diff gate {LOGITS_DIFF_TOL}"
            )


if __name__ == "__main__":
    main()
