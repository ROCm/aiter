# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP6 grouped MoE (FlyDSL, gfx950): the routed-expert span of one SwiGLU MoE layer.

    y[t] = sum_j w[t, j] * W2[e_tj] @ (silu(W1g[e_tj] @ x[t]) * (W1u[e_tj] @ x[t]))

Weights MXFP6 (E2M3 or E3M2); activations MXFP6 (A6W6: E2M3 or E3M2) or MXFP8 E4M3
(A8W6); E8M0 scale per 32 along K. E2M3 quantization is bitwise identical to the usual
reference (scale 2^ceil(log2(amax / 7.5)), round-half-even onto the grid, clamp).

Usage:
    weights = prepare_mxfp6_moe_weights(w1, w2, a_fmt="e2m3")  # once per layer
    y = flydsl_mxfp6_moe(x, weights, topk_weight, topk_ids)   # per step

`prepare_mxfp6_moe_weights` quantizes and preshuffles a layer's weights (torch only);
`flydsl_mxfp6_moe` runs a call (all FlyDSL, on the current stream, no host syncs,
bitwise deterministic):
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

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from . import mxfp6_moe_quant as q
from .mxfp6_moe_kernels import (
    flydsl_mxfp6_moe_combine,
    flydsl_mxfp6_moe_quant_rows,
    flydsl_mxfp6_moe_route,
    flydsl_mxfp6_moe_stage1,
    flydsl_mxfp6_moe_stage2,
)
from .mxfp6_moe_policy import Mxfp6MoeConfig, select_config, stage_bn

__all__ = [
    "SUPPORTED_GFX",
    "Mxfp6MoeWeights",
    "flydsl_mxfp6_moe",
    "prepare_mxfp6_moe_weights",
]

SUPPORTED_GFX = ("gfx950",)  # scaled F8F6F4 MFMA + 160 KiB LDS
_ID_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True, eq=False)
class Mxfp6MoeWeights:
    """One layer's weights as prepare_mxfp6_moe_weights lays them out for the kernels.

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


def prepare_mxfp6_moe_weights(
    w1: torch.Tensor, w2: torch.Tensor, *, a_fmt: str = "e2m3", w_fmt: str = "e2m3"
) -> Mxfp6MoeWeights:
    """Quantize a layer's weights to MX w_fmt and preshuffle them into MFMA fragments,
    for calls with a_fmt activations. Torch only: runs on the weights' device, on any
    architecture.

    Args:
        w1: float [E, 2I, H], gate rows first.
        w2: float [E, H, I], on the w1 device.
        a_fmt: "e2m3" or "e3m2" (A6W6), or "e4m3" (A8W6). The weights are grouped for
            the kernels of this activation format, so it is fixed per prepared layer.
        w_fmt: "e2m3" or "e3m2".

    Raises:
        ValueError: w1 / w2 not float tensors of these shapes on one device, with H
            and I multiples of 128; an unknown format.
    """
    shapes_ok = w1.dim() == 3 and w2.dim() == 3
    if shapes_ok:
        E, N1, H = w1.shape
        I = w2.shape[2]
        shapes_ok = N1 == 2 * I and w2.shape[:2] == (E, H) and min(E, H, I) > 0
        shapes_ok = shapes_ok and not (H % 128 or I % 128)
    if not shapes_ok:
        raise ValueError(
            "w1 must be [E, 2I, H] and w2 [E, H, I] with H and I multiples of 128, "
            f"got {tuple(w1.shape)} and {tuple(w2.shape)}"
        )
    if not (w1.is_floating_point() and w2.is_floating_point()):
        raise ValueError(f"w1 and w2 must be float, got {w1.dtype} and {w2.dtype}")
    if w1.device != w2.device:
        raise ValueError(
            f"w1 and w2 must be on one device, got {w1.device} and {w2.device}"
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


def _check_call(x, weights, topk_weight, topk_ids, config) -> None:
    """Raises for a call outside flydsl_mxfp6_moe's contract, before anything is
    launched: the kernels read raw pointers, so a tensor of another dtype, shape or
    device is read as garbage, not refused."""
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        raise RuntimeError(
            f"flydsl_mxfp6_moe needs {' or '.join(SUPPORTED_GFX)}, got {gfx}"
        )
    if not isinstance(weights, Mxfp6MoeWeights):
        raise TypeError(
            "weights must come from prepare_mxfp6_moe_weights(), got "
            f"{type(weights).__name__}"
        )
    if config is not None and not isinstance(config, Mxfp6MoeConfig):
        raise TypeError(f"config must be Mxfp6MoeConfig, got {type(config).__name__}")
    H = weights.model_dim
    shape = x.shape
    if (
        len(shape) != 2
        or shape[1] != H
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        raise ValueError(
            f"hidden_states must be contiguous bf16 [T, {H}], got {x.dtype} "
            f"{tuple(shape)}"
        )
    # float ids are refused, not cast: they are the routing weights of a call with
    # its two routing arguments swapped
    ids_shape = topk_ids.shape
    if (
        len(ids_shape) != 2
        or ids_shape[0] != shape[0]
        or ids_shape[1] < 1
        or topk_ids.dtype not in _ID_DTYPES
    ):
        raise ValueError(
            f"topk_ids must be int32 or int64 [{shape[0]}, topk], got "
            f"{topk_ids.dtype} {tuple(ids_shape)}"
        )
    if topk_weight.shape != ids_shape or not topk_weight.is_floating_point():
        raise ValueError(
            f"topk_weight must be float {tuple(ids_shape)}, got "
            f"{topk_weight.dtype} {tuple(topk_weight.shape)}"
        )
    dev = weights.w1_frags.device
    if not (x.device == topk_weight.device == topk_ids.device == dev):
        raise ValueError(
            f"hidden_states ({x.device}), topk_weight ({topk_weight.device}) and "
            f"topk_ids ({topk_ids.device}) must be on the weights' device, {dev}"
        )


def flydsl_mxfp6_moe(
    hidden_states: torch.Tensor,
    weights: Mxfp6MoeWeights,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    config: Mxfp6MoeConfig | None = None,
) -> torch.Tensor:
    """The routed-expert sum of one MoE layer (no shared expert, no routed scaling).

    Args:
        hidden_states: contiguous bf16 [T, H].
        weights: the layer, from prepare_mxfp6_moe_weights.
        topk_weight: float [T, topk] routing weights (cast to fp32).
        topk_ids: int32 or int64 [T, topk] expert ids in [0, E) (not checked: that
            would read the tensor on the host).
        config: the kernels to run (mxfp6_moe_policy); None = the tuned row of this
            call's shape where the tuned config has one, else the heuristic's
            (select_config).

    Returns:
        bf16 [T, H] on the hidden_states device.

    Raises:
        RuntimeError: not on gfx950; tuned config files that repeat a shape.
        TypeError: weights not a Mxfp6MoeWeights, config not a Mxfp6MoeConfig.
        ValueError: a tensor of another dtype, shape or device than above; a config
            whose stages do not run on its routing tiles; a tuned config file with a
            malformed row, or a row that does not fit the layer; from the stage wrappers
            (flydsl_mxfp6_moe_stage1), a config that does not fit the prepared
            weights, or token / routed rows of 4 GiB - 4 KiB or more.
    """
    x = hidden_states
    _check_call(x, weights, topk_weight, topk_ids, config)
    T, K = topk_ids.shape
    E, H, I = weights.experts, weights.model_dim, weights.inter_dim
    af, wf = weights.a_fmt, weights.w_fmt
    if T == 0:  # nothing to launch: HIP refuses an empty grid
        return torch.empty(0, H, dtype=torch.bfloat16, device=x.device)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if topk_weight.dtype != torch.float32:
        topk_weight = topk_weight.float()
    if config is None:
        config = select_config(
            tokens=T,
            topk=K,
            experts=E,
            model_dim=H,
            inter_dim=I,
            a_fmt=af,
            w_fmt=wf,
            slab_n1=weights.slab_n1,
            slab_n2=weights.slab_n2,
            w1_bytes=weights.w1_bytes,
            w2_bytes=weights.w2_bytes,
            cu_num=get_cu_num(),
            gfx=get_gfx_runtime(),
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
