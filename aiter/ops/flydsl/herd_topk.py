# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD routing orchestration with FlyDSL candidate selection + finalize."""

import torch

from .kernels.herd_topk import (
    build_herd_finalize_module,
    build_herd_fused_softmax_module,
    build_herd_fused_sqrtsoftplus_module,
    sqrtsoftplus_workspace_elems,
)
from .kernels.tensor_shim import _run_compiled, wave_size_of
from .kernels.topk.topk_per_row_small_k import build_topk_per_row_small_k_module

_KIMI_K3_EXPERTS = 896
_KIMI_K3_TOPK = 16
_MINIMAX_M3_EXPERTS = 128
_MINIMAX_M3_TOPK = 4
_GPT_OSS_EXPERTS = 128
_GPT_OSS_TOPK = 4
_DSV4_FLASH_EXPERTS = 256
_DSV4_FLASH_TOPK = 6
_HERD_MAX_TOKENS = 128
_FUSED_ROW_BUCKETS = (32, 64, _HERD_MAX_TOKENS)
_DEFAULT_FINALIZE_THREADS = 256
_K3_CANDIDATE_THREADS = 256
_K3_FINALIZE_THREADS = 1024
_SQRTSOFTPLUS_ROWS_PER_BLOCK = 4
_SQRTSOFTPLUS_WORKSPACE: dict[tuple[int, int, str], torch.Tensor] = {}


def _run_sigmoid_candidate(
    packed_gating: torch.Tensor,
    correction_bias: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_values: torch.Tensor,
    rows: int,
    experts: int,
    kp1: int,
    block_threads: int,
    stream: torch.cuda.Stream,
) -> None:
    candidate_launch = build_topk_per_row_small_k_module(
        kp1,
        experts,
        wave_size_of(packed_gating.device.index),
        block_threads=block_threads,
        forced_blocks=False,
        score_mode="sigmoid_bias",
        bias_bf16=correction_bias.dtype == torch.bfloat16,
        score_bf16=packed_gating.dtype == torch.bfloat16,
        write_values=True,
        fixed_row_len=experts,
    )
    _run_compiled(
        candidate_launch,
        packed_gating,
        candidate_ids,
        candidate_values,
        correction_bias,
        rows,
        stream,
    )


def _run_fused_softmax(
    packed_gating: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    rows: int,
    experts: int,
    topk: int,
    routed_scaling_factor: float,
    stream: torch.cuda.Stream,
) -> None:
    # A smaller row bound gives each row more lanes and a shorter per-lane scan.
    max_rows = next(b for b in _FUSED_ROW_BUCKETS if rows <= b)
    launch = build_herd_fused_softmax_module(
        topk,
        experts,
        max_rows,
        wave_size_of(packed_gating.device.index),
    )
    _run_compiled(
        launch,
        packed_gating,
        topk_weights,
        topk_ids,
        int(topk_weights.stride(0)),
        int(topk_ids.stride(0)),
        rows,
        float(routed_scaling_factor),
        stream,
    )


def _run_fused_sqrtsoftplus(
    packed_gating: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    rows: int,
    experts: int,
    topk: int,
    need_renorm: bool,
    routed_scaling_factor: float,
    stream: torch.cuda.Stream,
) -> None:
    # Zeroed once; every launch leaves it zeroed again for the next.
    key = (experts, topk, str(packed_gating.device))
    workspace = _SQRTSOFTPLUS_WORKSPACE.get(key)
    if workspace is None:
        workspace = torch.zeros(
            sqrtsoftplus_workspace_elems(experts, _HERD_MAX_TOKENS, topk),
            dtype=torch.int32,
            device=packed_gating.device,
        )
        _SQRTSOFTPLUS_WORKSPACE[key] = workspace
    launch = build_herd_fused_sqrtsoftplus_module(
        topk,
        experts,
        _HERD_MAX_TOKENS,
        wave_size_of(packed_gating.device.index),
        bool(need_renorm),
        _SQRTSOFTPLUS_ROWS_PER_BLOCK,
    )
    _run_compiled(
        launch,
        packed_gating,
        correction_bias,
        topk_weights,
        topk_ids,
        workspace,
        int(topk_weights.stride(0)),
        int(topk_ids.stride(0)),
        rows,
        -(-rows // _SQRTSOFTPLUS_ROWS_PER_BLOCK),
        float(routed_scaling_factor),
        stream,
    )


def _run_finalize(
    candidate_ids: torch.Tensor,
    candidate_values: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    rows: int,
    experts: int,
    topk: int,
    need_renorm: bool,
    routed_scaling_factor: float,
    round_to_bf16: bool,
    block_threads: int,
    stream: torch.cuda.Stream,
) -> None:
    finalize_launch = build_herd_finalize_module(
        topk,
        experts,
        bool(need_renorm),
        round_to_bf16,
        block_threads=block_threads,
    )
    _run_compiled(
        finalize_launch,
        candidate_ids,
        candidate_values,
        topk_weights,
        topk_ids,
        int(topk_weights.stride(0)),
        int(topk_ids.stride(0)),
        rows,
        float(routed_scaling_factor),
        stream,
    )


def _profile(score_func: str, experts: int, topk: int) -> str | None:
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
    if (score_func, experts, topk) == (
        "softmax",
        _GPT_OSS_EXPERTS,
        _GPT_OSS_TOPK,
    ):
        return "gpt_oss"
    if (score_func, experts, topk) == (
        "sqrtsoftplus",
        _DSV4_FLASH_EXPERTS,
        _DSV4_FLASH_TOPK,
    ):
        return "dsv4_flash"
    return None


def _unsupported_reason(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    score_func: str,
    need_renorm: bool = True,
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
    if gating_output.dtype not in (torch.float32, torch.bfloat16):
        return f"gating_output must be float32 or bfloat16, got {gating_output.dtype}"
    if gating_output.stride(1) != 1:
        return "gating_output must have contiguous expert columns"
    if topk_weights.shape != (rows, topk) or topk_weights.dtype != torch.float32:
        return f"topk_weights must be float32 [{rows}, {topk}]"
    if topk_ids.shape != (rows, topk) or topk_ids.dtype != torch.int32:
        return f"topk_ids must be int32 [{rows}, {topk}]"
    if topk_weights.stride(1) != 1 or topk_ids.stride(1) != 1:
        return "Top-K outputs must have inner stride 1"
    tensors = (topk_weights, topk_ids, gating_output)
    if profile == "gpt_oss":
        # Top-K then softmax over the kept K: without renormalization the
        # weights would be full-row probabilities, which K+1 candidates cannot
        # reproduce.
        if not need_renorm:
            return "softmax HERD requires need_renorm"
        if correction_bias is not None and correction_bias.numel() != 0:
            return "softmax HERD does not take a correction_bias"
        # The fused selector packs a score and its expert id into one 32-bit
        # key, which only a 16-bit score leaves room for.
        if gating_output.dtype != torch.bfloat16:
            return "softmax HERD requires bfloat16 gating_output"
    else:
        if correction_bias is None or correction_bias.shape != (experts,):
            return f"correction_bias must have shape [{experts}]"
        if correction_bias.dtype not in (torch.float32, torch.bfloat16):
            return f"unsupported correction_bias dtype {correction_bias.dtype}"
        if correction_bias.stride(0) != 1:
            return "correction_bias must be contiguous"
        tensors += (correction_bias,)
    if not all(t.is_cuda for t in tensors):
        return "every tensor must be on the GPU"
    if len({t.device for t in tensors}) != 1:
        return "every tensor must be on the same GPU"
    if wave_size_of(gating_output.device.index) not in (32, 64):
        return "the FlyDSL HERD kernels require a wave32 or wave64 GPU"
    return None


def herd_topk_gating_supported(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    score_func: str,
    need_renorm: bool = True,
) -> bool:
    return (
        _unsupported_reason(
            topk_weights,
            topk_ids,
            gating_output,
            correction_bias,
            score_func,
            need_renorm,
        )
        is None
    )


def herd_topk_gating(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor | None,
    need_renorm: bool,
    routed_scaling_factor: float,
    score_func: str,
) -> None:
    """Apply Top-(K+1), batch popularity, and min-unique drop.

    Output routes are canonicalized to ascending expert id, matching the
    Triton HERD output contract. The softmax profile selects on raw logits and
    weights the kept K by their softmax, as Triton's flat-topk HERD does.
    """
    reason = _unsupported_reason(
        topk_weights,
        topk_ids,
        gating_output,
        correction_bias,
        score_func,
        need_renorm,
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
    if profile == "gpt_oss":
        _run_fused_softmax(
            packed_gating,
            topk_weights,
            topk_ids,
            rows,
            experts,
            topk,
            routed_scaling_factor,
            stream,
        )
        return
    if (
        profile == "dsv4_flash"
        and packed_gating.dtype == torch.bfloat16
        and correction_bias.dtype == torch.float32
    ):
        _run_fused_sqrtsoftplus(
            packed_gating,
            correction_bias,
            topk_weights,
            topk_ids,
            rows,
            experts,
            topk,
            need_renorm,
            routed_scaling_factor,
            stream,
        )
        return
    kp1 = topk + 1
    candidate_ids = torch.empty((rows, kp1), dtype=torch.int32, device=device)
    candidate_values = torch.empty((rows, kp1), dtype=torch.float32, device=device)
    if profile == "kimi_k3":
        _run_sigmoid_candidate(
            packed_gating,
            correction_bias,
            candidate_ids,
            candidate_values,
            rows,
            experts,
            kp1,
            _K3_CANDIDATE_THREADS,
            stream,
        )
        _run_finalize(
            candidate_ids,
            candidate_values,
            topk_weights,
            topk_ids,
            rows,
            experts,
            topk,
            need_renorm,
            routed_scaling_factor,
            False,
            _K3_FINALIZE_THREADS,
            stream,
        )
        return

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
    finalize_threads = _DEFAULT_FINALIZE_THREADS
    if profile == "minimax_m3" and (rows <= 16 or rows > 64):
        finalize_threads = 512
    _run_finalize(
        candidate_ids,
        candidate_values,
        topk_weights,
        topk_ids,
        rows,
        experts,
        topk,
        need_renorm,
        routed_scaling_factor,
        False,
        finalize_threads,
        stream,
    )
