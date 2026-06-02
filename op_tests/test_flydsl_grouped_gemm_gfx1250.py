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
from aiter.fused_moe import fused_moe, _grouped_a8w4_prepare_scale_batch  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
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
# ---------------------------------------------------------------------------
def _preshuffle_b_n16_packed(b: torch.Tensor) -> torch.Tensor:
    """Pack ``(E, N, K_pack)`` mxfp4 bytes into the FP4 TDM B physical layout.

    The kernel views B as rows of 16 output columns, with packed K tiled in
    16-byte chunks (each chunk holding all 16 N lanes)::

        logical  (N, K_pack)
        physical (N/16, K_pack/16, 16 lanes, 16 bytes)
    """
    experts, rows, k_pack = b.shape
    if rows % 16 != 0 or k_pack % 16 != 0:
        raise ValueError(f"need rows%16==0 and k_pack%16==0, got {b.shape}")
    return (
        b.contiguous()
        .view(experts, rows // 16, 16, k_pack // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, rows, k_pack)
    )


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
# Reference helpers (mxfp4 / e8m0 -> fp32 for PyTorch baseline)
# ---------------------------------------------------------------------------
_MXFP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _mxfp4_to_f32(packed: torch.Tensor) -> torch.Tensor:
    """Decode mxfp4 packed bytes ``(..., K_pack)`` -> fp32 ``(..., 2*K_pack)``."""
    lut = _MXFP4_LUT.to(packed.device)
    u = packed.contiguous().view(torch.uint8)
    lo = (u & 0xF).long()
    hi = ((u >> 4) & 0xF).long()
    pair = torch.stack([lut[lo], lut[hi]], dim=-1)
    return pair.flatten(-2)


def _e8m0_to_f32(byte: torch.Tensor) -> torch.Tensor:
    u = byte.contiguous().view(torch.uint8).to(torch.int32)
    expo = u - 127
    return torch.pow(torch.full_like(expo, 2, dtype=torch.float32), expo.float())


def _dequant_weight(packed: torch.Tensor, scale_raw: torch.Tensor) -> torch.Tensor:
    """``packed: (E, N, K_pack) uint8``, ``scale_raw: (E, N, K//32) uint8`` -> ``(E, N, K) fp32``."""
    w = _mxfp4_to_f32(packed)
    s = _e8m0_to_f32(scale_raw)
    E, N, K = w.shape
    nblk = K // SCALE_BLOCK
    return (w.view(E, N, nblk, SCALE_BLOCK) * s.view(E, N, nblk, 1)).view(E, N, K)


def _torch_moe_ref(
    hidden: torch.Tensor,           # (T, K) bf16
    w1_packed: torch.Tensor,        # (E, 2*I, K_pack) uint8 (GGUU layout)
    w1_scale: torch.Tensor,         # (E, 2*I, K//32) uint8 (raw e8m0)
    w1_bias: torch.Tensor,          # (E, 2*I) fp32
    w2_packed: torch.Tensor,        # (E, K, I_pack) uint8
    w2_scale: torch.Tensor,         # (E, K, I//32) uint8
    w2_bias: torch.Tensor,          # (E, K) fp32
    topk_w: torch.Tensor,           # (T, topk) bf16
    topk_id: torch.Tensor,          # (T, topk) int32
    *,
    activation: ActivationType,
    swiglu_limit: float,
) -> torch.Tensor:
    """High-precision PyTorch reference (GGUU layout)."""
    T, K = hidden.shape
    E, two_I = w1_packed.shape[:2]
    I = two_I // 2
    h = hidden.float()
    w1_f32 = _dequant_weight(w1_packed, w1_scale)
    w2_f32 = _dequant_weight(w2_packed, w2_scale)
    out = torch.zeros((T, K), dtype=torch.float32, device=hidden.device)
    for e in range(E):
        mask = topk_id == e
        if not mask.any():
            continue
        rows, ranks = mask.nonzero(as_tuple=True)
        x = h[rows] @ w1_f32[e].t()
        x = x + w1_bias[e].float().view(1, -1)
        gate, up = x[:, :I], x[:, I:]
        if activation == ActivationType.Swiglu:
            limit = swiglu_limit if swiglu_limit > 0 else 7.0
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            act = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        else:
            act = torch.nn.functional.silu(gate) * up
        y = act @ w2_f32[e].t() + w2_bias[e].float().view(1, -1)
        out.index_add_(0, rows, y * topk_w[rows, ranks].float().unsqueeze(-1))
    return out.to(hidden.dtype)


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
    activation: ActivationType = ActivationType.Swiglu,
    gate_mode: GateMode = GateMode.SEPARATED,
    swiglu_limit: float = 7.0,
    use_bias: bool = True,
    verify: bool = False,
    seed: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build mxfp4 weights + balanced routing, dispatch through ``fused_moe``.

    Returns ``(grouped_out, ref_out_or_None)``.
    """
    if data_format not in ("a4w4", "a8w4"):
        raise ValueError(f"data_format must be a4w4 or a8w4, got {data_format!r}")

    K = model_dim
    I = inter_dim
    K_pack = K // 2
    I_pack = I // 2

    # Weights: mxfp4 packed bytes, GGUU layout (gate rows then up rows).
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
    w1_grouped = _preshuffle_b_n16_packed(w1_logical).cuda()
    w2_grouped = _preshuffle_b_n16_packed(w2_logical).cuda()
    w1_scale = _grouped_scale(w1_scale_raw, experts=experts, rows=2 * I, k_dim=K)
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
            bias1=bias1.cuda(),
            bias2=bias2.cuda(),
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
        ref = _torch_moe_ref(
            hidden.cuda(),
            w1_logical.cuda(), w1_scale_raw.cuda(), bias1.cuda(),
            w2_logical.cuda(), w2_scale_raw.cuda(), bias2.cuda(),
            topk_w.cuda(), topk_id.cuda(),
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
def _sanity_check(data_format: str, *, tol: float = 0.4) -> None:
    """Tiny shape; compare grouped vs PyTorch ref. ``model_dim/inter_dim``
    must be at least ``num_buffers * tile_k`` (defaults: 2 * 128 = 256)."""
    _require_gfx1250()
    out, ref = _run_grouped_via_fused_moe(
        experts=4, tokens=8, topk=2,
        model_dim=256, inter_dim=256,
        data_format=data_format,
        verify=True,
    )
    rel = _rel_l2(out, ref)
    print(f"[sanity {data_format}] rel_l2 grouped vs ref = {rel:.4e}", flush=True)
    assert rel < tol, f"grouped {data_format} vs ref rel_l2={rel:.4f} > tol={tol}"


def test_grouped_a4w4_matches_torch_ref():
    _sanity_check("a4w4")


def test_grouped_a8w4_matches_torch_ref():
    _sanity_check("a8w4")


# ---------------------------------------------------------------------------
# Perf bench (uses aiter's run_perftest for stable timing)
# ---------------------------------------------------------------------------
def _bench(args: argparse.Namespace) -> None:
    from aiter.test_common import run_perftest

    _require_gfx1250()
    print(f"[bench] data_format={args.data_format} E={args.experts} T={args.tokens} "
          f"topk={args.topk} K={args.model_dim} I={args.inter_dim} "
          f"warmup={args.warmup} iters={args.iters}", flush=True)

    saved = os.environ.get("AITER_USE_GROUPED_GEMM")
    os.environ["AITER_USE_GROUPED_GEMM"] = "1"
    try:
        # First call: warm caches, JIT compile.
        _run_grouped_via_fused_moe(
            experts=args.experts, tokens=args.tokens, topk=args.topk,
            model_dim=args.model_dim, inter_dim=args.inter_dim,
            data_format=args.data_format,
            verify=False,
        )
        torch.cuda.synchronize()

        def _thunk():
            _run_grouped_via_fused_moe(
                experts=args.experts, tokens=args.tokens, topk=args.topk,
                model_dim=args.model_dim, inter_dim=args.inter_dim,
                data_format=args.data_format,
                verify=False,
            )

        us, _ = run_perftest(_thunk, num_warmup=args.warmup, num_iters=args.iters)
        print(f"[bench] {args.data_format} fused_moe end-to-end us = {us:.2f}", flush=True)
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
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--swiglu-limit", type=float, default=7.0)
    args = parser.parse_args()

    if args.scenario == "verify":
        _sanity_check(args.data_format)
        return
    _bench(args)


if __name__ == "__main__":
    main()
