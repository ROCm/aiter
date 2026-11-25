# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP6 grouped MoE (FlyDSL, gfx950): the routed-expert span of one SwiGLU MoE layer.

    y[t] = sum_j w[t, j] * W2[e_tj] @ (silu(W1g[e_tj] @ x[t]) * (W1u[e_tj] @ x[t]))

Weights MXFP6 (E2M3 or E3M2); activations MXFP6 (A6W6: E2M3 or E3M2) or MXFP8 E4M3
(A8W6); E8M0 scale per 32 along K. E2M3 quantization is bitwise identical to the usual
reference (scale 2^ceil(log2(amax / 7.5)), round-half-even onto the grid, clamp).

`prepare` quantizes and preshuffles a layer's weights once; `forward` runs a call (all
FlyDSL, no host syncs, bitwise deterministic):
  1. route: counting sort by expert (one block, or histogram / scan / scatter above
     32k entries) -> gather index, tile -> expert map, used-tile count, pos[t, j] (the
     slot order within an expert does not affect bits)
  2. quant rows: x -> MX rows + scales
  3. stage 1 (grouped GEMM, "swiglu_q"): gathered x @ W1^T, SwiGLU, MX quantization of
     h in the epilogue
  4. stage 2 (grouped GEMM, "store"): h @ W2^T -> bf16 rows
  5. combine: y[t] = sum_j w[t, j] * o[pos[t, j]] in fixed j order (fp32), no atomics
Constraints: hidden and inter multiples of 128; any expert count, top-k and token
count; weights > 4 GiB ok.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_cu_num

from . import mxfp6_moe_quant as q
from .mxfp6_moe_kernels import (
    flydsl_mxfp6_moe_combine,
    flydsl_mxfp6_moe_quant_rows,
    flydsl_mxfp6_moe_route,
    flydsl_mxfp6_moe_stage1,
    flydsl_mxfp6_moe_stage2,
)
from .mxfp6_moe_policy import Mxfp6MoeConfig, default_config, stage_bn

SUPPORTED_GFX = ("gfx950",)  # scaled F8F6F4 MFMA + 160 KiB LDS


@dataclass(frozen=True, eq=False)
class Mxfp6MoeWeights:
    """One layer's weights as `prepare` lays them out for the kernels.

    w1_frags [E][2I/BN1][H/128][BN1/16][1536] uint8 (gate / up fragments interleaved)
    and w1_scales [E][2I/16][KQ(H)][64] int32; w2_frags [E][H/BN2][I/128][BN2/16][1536]
    and w2_scales [E][H/16][KQ(I)][64]. slab_n1 / slab_n2 (BN1 / BN2): the slab widths
    the fragments are grouped in; w1_bytes / w2_bytes: bytes of each stage's fragments
    and scales."""

    experts: int
    model_dim: int
    inter_dim: int
    a_fmt: str
    w_fmt: str
    w1_frags: torch.Tensor
    w1_scales: torch.Tensor
    w2_frags: torch.Tensor
    w2_scales: torch.Tensor
    slab_n1: int
    slab_n2: int
    w1_bytes: int
    w2_bytes: int


def prepare(
    w1: torch.Tensor, w2: torch.Tensor, *, a_fmt: str = "e2m3", w_fmt: str = "e2m3"
) -> Mxfp6MoeWeights:
    """Quantize a layer's weights to MX w_fmt ("e2m3" or "e3m2") and preshuffle them
    into MFMA fragments, for calls with a_fmt activations ("e2m3", "e3m2" or "e4m3").

    w1: [E, 2I, H] (gate rows first), w2: [E, H, I]; H and I multiples of 128."""
    E, N1, H = w1.shape
    I = w2.shape[2]
    if N1 != 2 * I or w2.shape[:2] != (E, H) or H % 128 or I % 128:
        raise ValueError(
            "w1 must be [E, 2I, H] and w2 [E, H, I] with H and I multiples of 128, "
            f"got {tuple(w1.shape)} and {tuple(w2.shape)}"
        )
    if a_fmt not in q.FORMATS or w_fmt not in ("e2m3", "e3m2"):
        raise ValueError(
            f"a_fmt {a_fmt!r} must be one of {tuple(q.FORMATS)}, w_fmt {w_fmt!r} "
            "'e2m3' or 'e3m2'"
        )
    dev = w1.device
    bn1, bn2 = stage_bn(N1, a_fmt), stage_bn(H, a_fmt)
    b1 = torch.empty(
        E, N1 // bn1, H // 128, bn1 // 16, 64 * 24, dtype=torch.uint8, device=dev
    )
    s1 = torch.empty(E, N1 // 16, q.kq(H), 64, dtype=torch.int32, device=dev)
    b2 = torch.empty(
        E, H // bn2, I // 128, bn2 // 16, 64 * 24, dtype=torch.uint8, device=dev
    )
    s2 = torch.empty(E, H // 16, q.kq(I), 64, dtype=torch.int32, device=dev)
    # per expert: the fp32 working copy of all experts does not fit
    for e in range(E):
        c, s = q.quant_mx(w1[e : e + 1], w_fmt)
        b1[e : e + 1], s1[e : e + 1] = q.preshuffle_b(
            c, s, interleave_gate_up=True, bn=bn1
        )
        c, s = q.quant_mx(w2[e : e + 1], w_fmt)
        b2[e : e + 1], s2[e : e + 1] = q.preshuffle_b(c, s, bn=bn2)
    return Mxfp6MoeWeights(
        experts=E,
        model_dim=H,
        inter_dim=I,
        a_fmt=a_fmt,
        w_fmt=w_fmt,
        w1_frags=b1,
        w1_scales=s1,
        w2_frags=b2,
        w2_scales=s2,
        slab_n1=bn1,
        slab_n2=bn2,
        w1_bytes=b1.nbytes + s1.nbytes,
        w2_bytes=b2.nbytes + s2.nbytes,
    )


def forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    weights: Mxfp6MoeWeights,
    config: Mxfp6MoeConfig | None = None,
) -> torch.Tensor:
    """x [T, H] bf16, topk_ids [T, K] int32, topk_weight [T, K] fp32 -> y [T, H] bf16.

    config: the kernels to run (mxfp6_moe_policy); None = the heuristic's
    (default_config)."""
    T, K = topk_ids.shape
    E, H, I = weights.experts, weights.model_dim, weights.inter_dim
    af, wf = weights.a_fmt, weights.w_fmt
    if config is None:
        config = default_config(
            tokens=T,
            topk=K,
            experts=E,
            model_dim=H,
            inter_dim=I,
            a_fmt=af,
            slab_n1=weights.slab_n1,
            slab_n2=weights.slab_n2,
            w1_bytes=weights.w1_bytes,
            w2_bytes=weights.w2_bytes,
            cu_num=get_cu_num(),
        )
    elif (
        config.stage1.route_m != config.route_m
        or config.stage2.route_m != config.route_m
    ):
        raise ValueError(
            f"config: both stages must run on the {config.route_m}-row routing tiles, "
            f"got {config.stage1} and {config.stage2}"
        )
    ridx, pos, texp, M = flydsl_mxfp6_moe_route(topk_ids, E, config.route_m)
    xa, xs = flydsl_mxfp6_moe_quant_rows(x, af)
    # stage 1 writes h already MX-quantized (stage-2 A rows + [M, KQ, 4] scales)
    ha = torch.empty(M, I * q.FORMATS[af][3] // 8, dtype=torch.uint8, device=x.device)
    hs = torch.empty(M, q.kq(I), 4, dtype=torch.int32, device=x.device)
    flydsl_mxfp6_moe_stage1(
        x_rows=xa,
        x_scales=xs,
        ridx=ridx,
        texp=texp,
        w1_frags=weights.w1_frags,
        w1_scales=weights.w1_scales,
        h_rows=ha,
        h_scales=hs,
        model_dim=H,
        inter_dim=I,
        a_fmt=af,
        w_fmt=wf,
        config=config.stage1,
        h_layout=config.h_layout,
    )
    o = torch.empty(M, H, dtype=torch.bfloat16, device=x.device)
    flydsl_mxfp6_moe_stage2(
        h_rows=ha,
        h_scales=hs,
        ridx=ridx,
        texp=texp,
        w2_frags=weights.w2_frags,
        w2_scales=weights.w2_scales,
        out=o,
        model_dim=H,
        inter_dim=I,
        a_fmt=af,
        w_fmt=wf,
        config=config.stage2,
        h_layout=config.h_layout,
    )
    return flydsl_mxfp6_moe_combine(o, pos, topk_weight)
