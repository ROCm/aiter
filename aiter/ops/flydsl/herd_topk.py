# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD routing orchestration around native Top-(K+1) selection."""

import torch

from .kernels.herd_topk import build_herd_finalize_module
from .kernels.tensor_shim import _run_compiled, wave_size_of

_DSV4_EXPERTS = 384
_DSV4_TOPK = 6
_KIMI_K3_EXPERTS = 896
_KIMI_K3_TOPK = 16
_MINIMAX_M3_EXPERTS = 128
_MINIMAX_M3_TOPK = 4
_HERD_MAX_TOKENS = 128


def _profile(score_func: str, experts: int, topk: int) -> str | None:
    if (score_func, experts, topk) == (
        "sqrtsoftplus",
        _DSV4_EXPERTS,
        _DSV4_TOPK,
    ):
        return "dsv4"
    if (score_func, experts, topk) == (
        "sigmoid",
        _KIMI_K3_EXPERTS,
        _KIMI_K3_TOPK,
    ):
        return "kimi_k3"
    if (score_func, experts, topk) == (
        "sigmoid",
        _MINIMAX_M3_EXPERTS,
        _MINIMAX_M3_TOPK,
    ):
        return "minimax_m3"
    return None


def _unsupported_reason(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    score_func: str,
) -> str | None:
    if gating_output.dim() != 2:
        return f"gating_output must be 2-D, got {tuple(gating_output.shape)}"
    rows, experts = gating_output.shape
    if not 1 <= rows <= _HERD_MAX_TOKENS:
        return f"tokens must be in [1, {_HERD_MAX_TOKENS}], got {rows}"
    topk = topk_ids.shape[1] if topk_ids.dim() == 2 else -1
    profile = _profile(score_func, experts, topk)
    if profile is None:
        return (
            "unsupported HERD profile: "
            f"score_func={score_func}, experts={experts}, topk={topk}"
        )
    expected_dtype = torch.bfloat16 if profile == "dsv4" else torch.float32
    if gating_output.dtype != expected_dtype:
        return f"gating_output must be {expected_dtype}, got {gating_output.dtype}"
    if gating_output.stride(1) != 1:
        return "gating_output must have contiguous expert columns"
    if topk_weights.shape != (rows, topk) or topk_weights.dtype != torch.float32:
        return f"topk_weights must be float32 [{rows}, {topk}]"
    if topk_ids.shape != (rows, topk) or topk_ids.dtype != torch.int32:
        return f"topk_ids must be int32 [{rows}, {topk}]"
    if topk_weights.stride(1) != 1 or topk_ids.stride(1) != 1:
        return "Top-K outputs must have inner stride 1"
    if correction_bias.shape != (experts,):
        return f"correction_bias must have shape [{experts}]"
    if correction_bias.dtype not in (torch.float32, torch.bfloat16):
        return f"unsupported correction_bias dtype {correction_bias.dtype}"
    if correction_bias.stride(0) != 1:
        return "correction_bias must be contiguous"
    tensors = (topk_weights, topk_ids, gating_output, correction_bias)
    if not all(t.is_cuda for t in tensors):
        return "every tensor must be on the GPU"
    if len({t.device for t in tensors}) != 1:
        return "every tensor must be on the same GPU"
    if wave_size_of(gating_output.device.index) != 64:
        return "the FlyDSL HERD kernels require a wave64 GPU"
    return None


def herd_topk_gating_supported(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    score_func: str,
) -> bool:
    return (
        _unsupported_reason(
            topk_weights,
            topk_ids,
            gating_output,
            correction_bias,
            score_func,
        )
        is None
    )


def herd_topk_gating(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float,
    score_func: str,
) -> None:
    """Apply Top-(K+1), batch popularity, and min-unique drop.

    Output routes are canonicalized to ascending expert id, matching the
    Triton HERD output contract.
    """
    reason = _unsupported_reason(
        topk_weights,
        topk_ids,
        gating_output,
        correction_bias,
        score_func,
    )
    if reason is not None:
        raise ValueError(f"[FlyDSL HERD Top-K] {reason}")

    rows = gating_output.shape[0]
    experts = gating_output.shape[1]
    topk = topk_ids.shape[1]
    profile = _profile(score_func, experts, topk)
    assert profile is not None
    device = gating_output.device
    stream = torch.cuda.current_stream(device)
    # Both native candidate selectors assume packed rows. Some model router
    # outputs are leading slices of a larger tensor, so inner stride 1 alone is
    # not sufficient.
    packed_gating = (
        gating_output if gating_output.is_contiguous() else gating_output.contiguous()
    )
    kp1 = topk + 1
    candidate_ids = torch.empty((rows, kp1), dtype=torch.int32, device=device)
    candidate_values = torch.empty((rows, kp1), dtype=torch.float32, device=device)

    if profile == "dsv4":
        # The dedicated selector reproduces Triton's BF16 score/bias rounding
        # and tie ordering exactly for DeepSeek-V4.
        from ..topk import topk_gating_herd_candidates_fwd

        topk_gating_herd_candidates_fwd(
            candidate_values,
            candidate_ids,
            packed_gating,
            correction_bias,
        )
    else:
        # Kimi-K3 and MiniMax-M3 expose FP32 sigmoid router logits. Reuse the
        # native generic selector for Top-(K+1).
        from ..topk import topk_gating_fwd

        topk_gating_fwd(
            candidate_values,
            candidate_ids,
            packed_gating,
            correction_bias,
            False,
            1.0,
            score_func,
        )

    finalize_launch = build_herd_finalize_module(
        topk,
        experts,
        bool(need_renorm),
        profile == "dsv4",
    )

    _run_compiled(
        finalize_launch,
        candidate_ids,
        candidate_values,
        topk_weights,
        topk_ids,
        rows,
        float(routed_scaling_factor),
        stream,
    )
