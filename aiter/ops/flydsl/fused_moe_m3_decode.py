# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whole-graph ``fused_moe`` implementation of the decode-shape (8-128 row) MoE with a folded shared expert:
E128 top-4, H6144 / I768, MXFP4 routed experts and an MXFP8 shared expert (MiniMax-M3.1 at TP4), gfx950.

Registered as ``impl__flydsl_m3decode__<ExpertCfg name>`` (``aiter.fused_moe_registry``) and reached from
``fused_moe(..., shared_folded=True)`` through a ``tuned_shared_fmoe.csv`` row; it runs, for the caller's routes
(``topk_ids`` / ``topk_weight``, the routed scale already in the weights):

1. the MXFP4 (routed) + MXFP8 (shared) input quant of ``hidden_states`` (skipped for a router entry that already
   produced both: ``fused_moe(..., hidden_states=<fp4 codes>, a1_scale=..., shared_a1=(codes, scales))``),
2. routed GEMM1 (+ SwiGLU-OAI + MXFP4 quant) and GEMM2 (+ top-k weighting) with the shared expert folded into both
   launches, no sort launch (GEMM1 builds its expert blocks from the ids),
3. the four weighted slot rows plus the shared expert summed to BF16.

Weights are the ones the model already holds: the routed ``w1`` / ``w2`` and the shared weights in aiter's (16, 16)
preshuffle, the routed scales in ``shuffle_scale`` layout and the shared scales row-major ``[N, K / 32]``. The
kernels read the routed w13 in place; the e8m0 scales of the routed w13 and of the shared expert are re-laid once
into the tile layouts and cached (build them before CUDA-graph capture: ``prepare_decode_weights`` or a first eager
call).
"""

import torch

from aiter import dtypes
from aiter.fused_moe_registry import FusedMoeRequest
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_moe_cfg import (
    ROWS,
    TOPK,
    E,
    ExpertCfg,
    H,
    I,
)

SWIGLU_LIMIT = 7.0
_DERIVED: dict = {}


def routed_w13_scale_layout(w13_scale: torch.Tensor) -> torch.Tensor:
    """e8m0 routed w13 scales in ``shuffle_scale`` layout ([E, 2I, H / 32] gate rows then up rows) -> the gate / up
    interleaved shuffle GEMM1 reads: undo the shuffle to row-major, shuffle again with ``is_guinterleave``.
    """
    from aiter.ops.shuffle import shuffle_scale

    e, n, k32 = w13_scale.shape
    t = w13_scale.reshape(e * n // 32, k32 // 8, 4, 16, 2, 2)
    raw = t.permute(0, 5, 3, 1, 4, 2).reshape(e, n, k32)
    return shuffle_scale(
        raw.reshape(e * n, k32), e, is_guinterleave=True, gate_up=True
    ).view(torch.uint8)


def _shared_s13_layout(s13: torch.Tensor) -> torch.Tensor:
    b13 = s13.view(torch.uint8).reshape(48, 2, 16, 24, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return b13.contiguous()


def _shared_s2_layout(s2: torch.Tensor) -> torch.Tensor:
    b2 = s2.view(torch.uint8).reshape(192, 2, 16, 3, 2, 4).permute(0, 3, 5, 2, 4, 1)
    return b2.contiguous()


def shared_scale_layouts(s13: torch.Tensor, s2: torch.Tensor):
    """Row-major e8m0 scales of the shared gate_up [2I, H / 32] and down [H, I / 32] -> the tile layouts (pure
    permutations)."""
    return _shared_s13_layout(s13), _shared_s2_layout(s2)


def _derived(kind: str, src: torch.Tensor, build):
    try:
        version = src._version
    except RuntimeError:  # inference tensors have no version counter
        version = -1
    key = (kind, src.data_ptr(), tuple(src.shape), src.device, version)
    hit = _DERIVED.get(key)
    if hit is None:
        # keep the source alive with its layout: while the entry exists its storage address cannot be
        # recycled by another tensor (same shape and version counter would alias the key)
        hit = _DERIVED[key] = (build(src), src)
    return hit[0]


def router_lines(router_weight: torch.Tensor) -> torch.Tensor:
    """The line-layout copy of the fp32 gate the 1-32 row router kernels read (cached)."""
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import (
        router_line_layout,
    )

    return _derived("lines", router_weight, router_line_layout)


def clear_derived_weight_cache() -> None:
    """Forget the re-laid scales (call after weights are replaced in place)."""
    _DERIVED.clear()


def prepare_decode_weights(w1_scale, shared_w1_scale, shared_w2_scale):
    """Build and cache the re-laid scales now (before graph capture); returns them (s13, ss13, ss2)."""
    s13 = _derived(
        "s13",
        w1_scale.view(torch.uint8).view(E, 2 * I, H // 32),
        routed_w13_scale_layout,
    )
    # the two shared layouts depend on different source tensors: cache them independently
    ss13 = _derived(
        "ss13",
        shared_w1_scale.view(torch.uint8).reshape(2 * I, H // 32),
        _shared_s13_layout,
    )
    ss2 = _derived(
        "ss2",
        shared_w2_scale.view(torch.uint8).reshape(H, I // 32),
        _shared_s2_layout,
    )
    return s13, ss13, ss2


def check_request(req: FusedMoeRequest) -> str | None:
    """Why the request is outside the kernels' contract, or None."""
    x, w1, w2 = req.hidden_states, req.w1, req.w2
    n = req.topk_ids.shape[0]
    if not (ROWS[0] <= n <= ROWS[1]):
        return f"{n} rows (covers {ROWS[0]}-{ROWS[1]})"
    if tuple(req.topk_ids.shape) != (n, TOPK) or req.topk_ids.dtype != torch.int32:
        return "topk_ids must be int32 [rows, 4]"
    if req.topk_weight.dtype != torch.float32 or tuple(req.topk_weight.shape) != (
        n,
        TOPK,
    ):
        return "topk_weight must be float32 [rows, 4]"
    if not (req.topk_ids.is_contiguous() and req.topk_weight.is_contiguous()):
        return "topk_ids / topk_weight must be contiguous"
    if w1.numel() != E * 2 * I * H // 2 or w2.numel() != E * H * I // 2:
        return "routed weights are not E128 H6144 I768 MXFP4"
    if not (getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        return "routed weights must be preshuffled (is_shuffled=True)"
    if req.w1_scale is None or req.w2_scale is None:
        return "routed weight scales are required"
    if req.w1_scale.numel() != E * 2 * I * H // 32:
        return "w1_scale shape"
    if req.w2_scale.numel() != E * H * I // 32:
        return "w2_scale shape"
    sw1, sw2 = req.shared_w1, req.shared_w2
    if sw1 is None or sw2 is None or req.shared_w1_scale is None:
        return "shared expert weights and scales are required"
    if req.shared_w2_scale is None:
        return "shared expert weights and scales are required"
    if sw1.numel() != 2 * I * H or sw2.numel() != H * I:
        return "shared expert is not MXFP8 [1536, 6144] / [6144, 768]"
    if sw1.dtype != dtypes.fp8 or sw2.dtype != dtypes.fp8:
        return "shared expert weights must be float8_e4m3fn (MXFP8)"
    if not (getattr(sw1, "is_shuffled", False) and getattr(sw2, "is_shuffled", False)):
        return "shared expert weights must be preshuffled (is_shuffled=True)"
    if req.shared_w1_scale.numel() != 2 * I * H // 32:
        return "shared_w1_scale shape"
    if req.shared_w2_scale.numel() != H * I // 32:
        return "shared_w2_scale shape"
    if req.swiglu_limit not in (None, SWIGLU_LIMIT) or req.beta not in (None, 1.0):
        return "kernels hard-code SwiGLU-OAI alpha=1.702 limit=7 beta=1"
    if req.expert_mask is not None or req.bias1 is not None or req.bias2 is not None:
        return "expert_mask / bias"
    if req.hidden_pad or req.intermediate_pad or req.doweight_stage1:
        return "padding / doweight_stage1"
    if req.shared_a1 is None:
        if x.dtype != torch.bfloat16 or tuple(x.shape) != (n, H):
            return "hidden_states must be bf16 [rows, 6144]"
    elif x.numel() != n * H // 2 or req.a1_scale is None:
        return "prequantized input needs fp4 codes [rows, 3072] and a1_scale"
    else:
        # moe_route_quant returns opaque uint8 codes; typed tensors must carry the encoding the kernels read
        if x.dtype not in (torch.uint8, dtypes.fp4x2):
            return "prequantized routed codes must be uint8 or float4_e2m1fn_x2 (MXFP4)"
        if req.shared_a1[0].dtype not in (torch.uint8, dtypes.fp8):
            return "prequantized shared codes must be uint8 or float8_e4m3fn (MXFP8)"
    return None


def run_m3_decode(req: FusedMoeRequest, cfg: ExpertCfg) -> torch.Tensor:
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_expert_gemms import (
        decode_expert_gemms,
        get_plan,
    )
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_topk_quant import (
        quant_mxfp4_mxfp8,
    )
    from aiter.ops.flydsl.kernels.moe_slots_shared_reduce import (
        moe_slots_shared_reduce,
    )

    why = check_request(req)
    if why is not None:
        raise NotImplementedError(f"flydsl_m3decode: {why}")
    n = req.topk_ids.shape[0]
    s13, ss13, ss2 = prepare_decode_weights(
        req.w1_scale, req.shared_w1_scale, req.shared_w2_scale
    )
    if req.shared_a1 is None:
        (xq, xs), (q8, s8) = quant_mxfp4_mxfp8(req.hidden_states.contiguous())
    else:
        xq, xs = req.hidden_states, req.a1_scale
        q8, s8 = req.shared_a1
    plan = get_plan(n, w13_split=True, tiles=cfg.to_tiles())
    u8 = lambda t: t.view(torch.uint8).view(-1)
    slots, shared = decode_expert_gemms(
        plan,
        u8(xq),
        u8(xs),
        u8(q8),
        u8(s8),
        (u8(req.w1), u8(s13), u8(req.w2), u8(req.w2_scale)),
        (u8(req.shared_w1), u8(ss13), u8(req.shared_w2), u8(ss2)),
        req.topk_ids,
        req.topk_weight,
    )
    # dropped routes (ids outside [0, E)) leave their GEMM2 slot rows unwritten: the reduction masks them
    return moe_slots_shared_reduce(slots, shared, topk_ids=req.topk_ids, num_experts=E)


def run_flydsl_m3decode_impl(req: FusedMoeRequest, config: str) -> torch.Tensor:
    """``fused_moe_registry`` entry: ``config`` is the ``ExpertCfg`` name of the tuned row."""
    return run_m3_decode(req, ExpertCfg.parse(config))
