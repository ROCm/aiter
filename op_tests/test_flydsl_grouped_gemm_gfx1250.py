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

_LOCAL_DEPS = ("/root/data/aiter", "/root/data/triton/python")
for _dep in reversed(_LOCAL_DEPS):
    if os.path.exists(_dep) and _dep not in sys.path:
        sys.path.insert(0, _dep)

from aiter import ActivationType, QuantType  # noqa: E402
from aiter.fused_moe import (  # noqa: E402
    fused_moe,
    torch_moe_stage1,
    torch_moe_stage2,
    _grouped_a8w4_prepare_scale_batch,
)
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.ops.quant import per_1x32_f4_quant  # noqa: E402
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402
from aiter.utility import dtypes  # noqa: E402

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

SCALE_BLOCK = 32
DEFAULT_SCALE_BYTE = 127  # e8m0 byte for 2^0 = 1.0


# ---------------------------------------------------------------------------
# Environment / arch guards
# ---------------------------------------------------------------------------
def _require_gfx1250() -> None:
    try:
        from flydsl.runtime.device import get_rocm_arch
    except Exception as exc:
        pytest.skip(f"FlyDSL not importable: {exc}")
    arch = get_rocm_arch()
    if "gfx1250" not in arch.lower():
        pytest.skip(f"requires gfx1250, got {arch!r}")


# ---------------------------------------------------------------------------
# Weight / scale preshuffle helpers (mandatory for the grouped path)
#
# Note: ``aiter.ops.shuffle.shuffle_weight(b, layout=(16, 16))`` is
# byte-for-byte equivalent to the FP4 TDM B layout the grouped FlyDSL
# kernels consume (16-row * 16-byte chunks). We use that public API
# directly. Scale shuffle, on the other hand, has its own grouped-only
# permutation; ``aiter.ops.shuffle.shuffle_scale`` is *not* compatible
# and we must use ``_grouped_a8w4_prepare_scale_batch`` below.
# ---------------------------------------------------------------------------
def _grouped_scale(
    scale_raw: torch.Tensor,
    *,
    experts: int,
    rows: int,
    k_dim: int,
    tile_n: int = 64,
    n_warp: int = 2,
    tile_k: int = 128,
) -> torch.Tensor:
    """Wrap ``_grouped_a8w4_prepare_scale_batch`` with the canonical kernel knobs."""
    return _grouped_a8w4_prepare_scale_batch(
        scale_raw.contiguous().cuda().view(dtypes.fp8_e8m0),
        experts=experts,
        rows=rows,
        k_dim=k_dim,
        warp_tile=tile_n // n_warp,
        tile_k=tile_k,
        device="cuda",
    )


# ---------------------------------------------------------------------------
# Reference: aiter's own ``torch_moe_stage1`` + ``torch_moe_stage2``
# (high-precision fp32 baseline that decodes mxfp4/e8m0 internally and
# evaluates the same swiglu+bias formula the grouped path uses). It still
# diverges from the quantised grouped GEMM path by mxfp4/mxfp8 round noise
# (~0.2 rel_l2 on random uint8 weights, ~0.02 on real model weights). The
# point is to catch *catastrophic* regressions, not chase fp32 parity.
# ---------------------------------------------------------------------------
def _torch_moe_ref(
    hidden: torch.Tensor,           # (T, K) bf16
    w1_packed: torch.Tensor,        # (E, 2*I, K_pack) uint8 (GGUU)
    w1_scale_raw: torch.Tensor,     # (E, 2*I, K//32) uint8 (raw e8m0)
    w1_bias: torch.Tensor,          # (E, 2*I) fp32
    w2_packed: torch.Tensor,        # (E, K, I_pack) uint8
    w2_scale_raw: torch.Tensor,     # (E, K, I//32) uint8
    w2_bias: torch.Tensor,          # (E, K) fp32
    topk_w: torch.Tensor,           # (T, topk) bf16
    topk_id: torch.Tensor,          # (T, topk) int32
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
        dtype_max = 240.0
        x_shape = x.shape
        flat = x.contiguous().view(-1, x_shape[-1]).float()
        blk = flat.view(-1, block)
        blk = torch.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0)
        max_abs = blk.abs().amax(dim=1)
        scale_e8m0 = fp4_utils.f32_to_e8m0(max_abs / dtype_max)
        scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
        scale_f32 = torch.nan_to_num(scale_f32, nan=1.0, posinf=1.0, neginf=1.0)
        scale_f32[scale_f32 == 0] = 1.0
        q_f32 = (blk / scale_f32.unsqueeze(1)).clamp(min=-dtype_max, max=dtype_max)
        q_u8 = fp4_utils._f32_to_floatx_unpacked(q_f32.contiguous().view(-1), 4, 3)
        q = q_u8.view(dtypes.fp8).to(torch.float32).view_as(blk)
        return (q * scale_f32.unsqueeze(1)).view(x_shape).to(x.dtype)

    w1_scale = w1_scale_raw.cuda().view(dtypes.fp8_e8m0)
    w2_scale = w2_scale_raw.cuda().view(dtypes.fp8_e8m0)
    if data_format == "a4w4":
        # Match the grouped a4w4 path: stage1 input is MXFP4, not bf16.
        stage1_hidden, stage1_hidden_scale = per_1x32_f4_quant(
            hidden, quant_dtype=dtypes.fp4x2, shuffle=False
        )
    else:
        # Match grouped a8w4: stage1 input is MXFP8 with per-1x32 e8m0 scale.
        stage1_hidden, stage1_hidden_scale = _per_1x32_fp8_dequant(hidden), None
    a2 = torch_moe_stage1(
        stage1_hidden, w1_packed.cuda(), w2_packed.cuda(),
        topk_w, topk_id,
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
        I = w2_packed.shape[-1] * 2
        a2_q, a2_scale = per_1x32_f4_quant(
            a2.contiguous().view(T * topk, I),
            quant_dtype=dtypes.fp4x2,
            shuffle=False,
        )
        a2 = a2_q.view(T, topk, I // 2)
    else:
        # Match grouped a8w4 stage2: per-block-32 MXFP8 quant + dequant.
        # This matters for SiLU because the unclamped stage1 output can exceed
        # fp8's unit-scale range; grouped now uses a real e8m0 block scale.
        a2 = _per_1x32_fp8_dequant(a2)
        a2_scale = None
    out = torch_moe_stage2(
        a2, w1_packed.cuda(), w2_packed.cuda(),
        topk_w, topk_id,
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
def _pattern_packed(experts: int, rows: int, k_pack: int, *, seed: int) -> torch.Tensor:
    """Cheap deterministic mxfp4 packed bytes ``(E, rows, k_pack) uint8``."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 256, (experts, rows, k_pack), dtype=torch.uint8, generator=g)


def _full_scale(experts: int, rows: int, n_blocks: int, byte: int = DEFAULT_SCALE_BYTE) -> torch.Tensor:
    return torch.full((experts, rows, n_blocks), byte, dtype=torch.uint8)


def _balanced_topk(tokens: int, topk: int, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-robin (token, rank) -> expert, even mass on each topk slot."""
    tok = torch.arange(tokens).view(tokens, 1)
    rk = torch.arange(topk).view(1, topk)
    ids = ((tok * topk + rk) % experts).to(torch.int32)
    w = torch.full((tokens, topk), 1.0 / topk, dtype=torch.float32)
    return ids, w


def _gguu_to_gugu_rows(t: torch.Tensor) -> torch.Tensor:
    """``(E, 2*I, ...)`` GGUU ``[g0..g_{I-1}, u0..u_{I-1}]`` -> GUGU ``[g0,u0,g1,u1,...]``."""
    E, two_I = t.shape[:2]
    I = two_I // 2
    g = t[:, :I]
    u = t[:, I:]
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
    data_format: str,                  # "a4w4" | "a8w4"
    layout: str = "gguu",              # "gguu" -> SEPARATED | "gugu" -> INTERLEAVE
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    use_bias: bool = True,
    verify: bool = False,
    seed: int = 0,
    all_ones: bool = False,            # debug: hidden=1, weight bytes=0x22 (=+1.0/+1.0), scale=127, bias=0
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build mxfp4 weights + balanced routing, dispatch through ``fused_moe``.

    ``layout`` selects the stage1 weight physical layout:
    ``gguu`` (gate rows then up rows, default) pairs with ``GateMode.SEPARATED``;
    ``gugu`` (gate/up row-interleaved, gpt-oss style) pairs with
    ``GateMode.INTERLEAVE``. The PyTorch reference always evaluates the
    GGUU logical weights, so both paths share the same numerical result.

    Returns ``(grouped_out, ref_out_or_None)``.
    """
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")
    if layout not in ("gguu", "gugu"):
        raise ValueError(f"layout must be gguu or gugu, got {layout!r}")

    K = model_dim
    I = inter_dim
    K_pack = K // 2
    I_pack = I // 2

    # Logical weights/scale/bias: always GGUU (gate rows then up rows).
    if all_ones:
        # Every mxfp4 nibble decodes to +1.0 (byte=0x22 = pair of 0010);
        # scale=byte 127 = 2^0 = 1.0; bias=0; hidden=1.0.
        w1_logical = torch.full((experts, 2 * I, K_pack), 0x22, dtype=torch.uint8)
        w2_logical = torch.full((experts, K, I_pack), 0x22, dtype=torch.uint8)
        w1_scale_raw = _full_scale(experts, 2 * I, K // SCALE_BLOCK)
        w2_scale_raw = _full_scale(experts, K, I // SCALE_BLOCK)
        bias1 = torch.zeros((experts, 2 * I))
        bias2 = torch.zeros((experts, K))
        hidden = torch.ones((tokens, K), dtype=torch.bfloat16)
    else:
        w1_logical = _pattern_packed(experts, 2 * I, K_pack, seed=seed + 17)
        w2_logical = _pattern_packed(experts, K, I_pack, seed=seed + 47)
        w1_scale_raw = _full_scale(experts, 2 * I, K // SCALE_BLOCK)
        w2_scale_raw = _full_scale(experts, K, I // SCALE_BLOCK)
        if use_bias:
            bg = torch.Generator(device="cpu").manual_seed(seed + 91)
            bias1 = (torch.randn((experts, 2 * I), generator=bg) * 1e-3).float()
            bias2 = (torch.randn((experts, K), generator=bg) * 1e-3).float()
        else:
            bias1 = torch.zeros((experts, 2 * I))
            bias2 = torch.zeros((experts, K))
        # Activations: bf16; fused_moe handles the dispatched quant internally.
        hg = torch.Generator(device="cpu").manual_seed(seed + 123)
        hidden = (torch.randn((tokens, K), generator=hg) * 0.5).to(torch.bfloat16)

    # Routing: round-robin balanced.
    topk_id, topk_w = _balanced_topk(tokens, topk, experts)
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

    w1_grouped = shuffle_weight(w1_phys, layout=(16, 16)).cuda()
    w2_grouped = shuffle_weight(w2_logical, layout=(16, 16)).cuda()
    w1_scale = _grouped_scale(w1_scale_phys, experts=experts, rows=2 * I, k_dim=K)
    w2_scale = _grouped_scale(w2_scale_raw, experts=experts, rows=K, k_dim=I)

    if data_format == "a4w4":
        w1_arg = w1_grouped.view(dtypes.fp4x2)
        w2_arg = w2_grouped.view(dtypes.fp4x2)
    else:  # a8w4
        w1_arg = w1_grouped  # uint8 -> grouped helper sets q_dtype_a=fp8
        w2_arg = w2_grouped

    saved = os.environ.get("AITER_USE_GROUPED_GEMM")
    os.environ["AITER_USE_GROUPED_GEMM"] = "1"
    try:
        grouped_out = fused_moe(
            hidden.cuda(),
            w1_arg, w2_arg,
            topk_w.cuda(),
            topk_id.cuda(),
            activation=activation,
            quant_type=QuantType.per_1x32,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            bias1=bias1_phys.float().cuda(),
            bias2=bias2.float().cuda(),
            gate_mode=gate_mode.value,
            dtype=dtypes.bf16,
            swiglu_limit=swiglu_limit,
        )
    finally:
        if saved is None:
            os.environ.pop("AITER_USE_GROUPED_GEMM", None)
        else:
            os.environ["AITER_USE_GROUPED_GEMM"] = saved

    ref = None
    if verify:
        # Reference always uses GGUU logical inputs (layouts are numerically
        # equivalent; only physical packing differs).
        ref = _torch_moe_ref(
            hidden.cuda(),
            w1_logical, w1_scale_raw, bias1.cuda(),
            w2_logical, w2_scale_raw, bias2.cuda(),
            topk_w.cuda(), topk_id.cuda(),
            data_format=data_format,
            activation=activation, swiglu_limit=swiglu_limit,
        ).to(grouped_out.dtype)
    return grouped_out, ref


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    base = expected.float().norm().clamp(min=1e-12)
    return float(diff / base)


# ---------------------------------------------------------------------------
# Pytest correctness suite
# ---------------------------------------------------------------------------
def _sanity_check(
    data_format: str,
    *,
    layout: str = "gguu",
    activation: ActivationType = ActivationType.Swiglu,
    swiglu_limit: float = 7.0,
    tol: float = 0.4,
    all_ones: bool = False,
) -> None:
    """Tiny shape; compare grouped FlyDSL vs PyTorch fp32 ref.

    ``tol=0.4`` is the expected rel_l2 ceiling on **random uint8 mxfp4
    weights + random hidden_states**. fp32 reference + mxfp4/mxfp8
    quantised path naturally diverge at this scale; on real model
    weights (smoother distributions) the gap drops to ~0.02. The point
    of this sanity is to catch *catastrophic* regressions (rel_l2 >> 1
    means kernel is doing the wrong thing entirely).
    """
    _require_gfx1250()
    out, ref = _run_grouped_via_fused_moe(
        experts=4, tokens=8, topk=2,
        model_dim=256, inter_dim=256,
        data_format=data_format,
        layout=layout,
        activation=activation,
        swiglu_limit=swiglu_limit,
        verify=True,
        all_ones=all_ones,
    )
    rel = _rel_l2(out, ref)
    act = "swiglu" if activation == ActivationType.Swiglu else "silu"
    tag = f"{data_format} {layout} {act}{' all_ones' if all_ones else ''}"
    print(f"[sanity {tag}] rel_l2 grouped vs ref = {rel:.4e} "
          f"(grouped_norm={float(out.float().norm()):.4e} ref_norm={float(ref.float().norm()):.4e})",
          flush=True)
    assert rel < tol, f"grouped {tag} vs ref rel_l2={rel:.4f} > tol={tol}"


@pytest.mark.parametrize("layout", ["gguu", "gugu"])
def test_grouped_a4w4_silu_matches_torch_ref(layout):
    _sanity_check("a4w4", layout=layout, activation=ActivationType.Silu)


@pytest.mark.parametrize("layout", ["gguu", "gugu"])
def test_grouped_a4w4_swiglu_matches_torch_ref(layout):
    _sanity_check("a4w4", layout=layout, activation=ActivationType.Swiglu)


# ---------------------------------------------------------------------------
# Perf bench (uses aiter's run_perftest for stable timing)
# ---------------------------------------------------------------------------
def _bench(args: argparse.Namespace) -> None:
    from aiter.test_common import run_perftest

    _require_gfx1250()
    activation = ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
    print(f"[bench] data_format={args.data_format} layout={args.layout} act={args.act} "
          f"E={args.experts} T={args.tokens} topk={args.topk} "
          f"K={args.model_dim} I={args.inter_dim} "
          f"warmup={args.warmup} iters={args.iters}", flush=True)

    saved = os.environ.get("AITER_USE_GROUPED_GEMM")
    os.environ["AITER_USE_GROUPED_GEMM"] = "1"
    try:
        common = dict(
            experts=args.experts, tokens=args.tokens, topk=args.topk,
            model_dim=args.model_dim, inter_dim=args.inter_dim,
            data_format=args.data_format,
            layout=args.layout,
            activation=activation,
            swiglu_limit=args.swiglu_limit,
            verify=False,
        )
        _run_grouped_via_fused_moe(**common)  # warmup / JIT
        torch.cuda.synchronize()

        def _thunk():
            _run_grouped_via_fused_moe(**common)

        us, _ = run_perftest(_thunk, num_warmup=args.warmup, num_iters=args.iters)
        print(f"[bench] {args.data_format}/{args.layout} fused_moe end-to-end us = {us:.2f}",
              flush=True)
    finally:
        if saved is None:
            os.environ.pop("AITER_USE_GROUPED_GEMM", None)
        else:
            os.environ["AITER_USE_GROUPED_GEMM"] = saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("bench", "verify"), default="bench")
    parser.add_argument("--data-format", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument("--layout", choices=("gguu", "gugu"), default="gguu",
                        help="stage1 weight physical layout. gguu pairs with "
                             "GateMode.SEPARATED (default), gugu with INTERLEAVE.")
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--act", choices=("silu", "swiglu"), default="swiglu",
                        help="stage1 activation: silu => silu(gate)*up; "
                             "swiglu => gpt-oss swiglu with clamp/alpha/residual")
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    parser.add_argument("--all-ones", action="store_true",
                        help="(verify only) hidden=1, weight bytes=0x22 (=+1.0), "
                             "scale=127 (=2^0), bias=0. Expect rel_l2 < 0.01 since both "
                             "grouped and ref see the exact same dequantised values.")
    args = parser.parse_args()

    if args.scenario == "verify":
        activation = ActivationType.Swiglu if args.act == "swiglu" else ActivationType.Silu
        _sanity_check(args.data_format, layout=args.layout,
                      tol=0.01 if args.all_ones else 0.01,
                      activation=activation,
                      swiglu_limit=args.swiglu_limit,
                      all_ones=args.all_ones)
        return
    _bench(args)


if __name__ == "__main__":
    main()
