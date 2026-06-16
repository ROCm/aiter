# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import os
from typing import Optional, Tuple

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..utility import dtypes


@compile_ops("module_moe_asm", fc_name="biased_grouped_topk")
def biased_grouped_topk_hip(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_grp: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
) -> None: ...


@compile_ops("module_moe_asm", fc_name="grouped_topk")
def grouped_topk_hip(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    is_softmax: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None: ...


# Enable the FlyDSL grouped_topk gating kernel as a drop-in replacement for the
# HIP op. Defaults to OFF (opt-in): set AITER_USE_FLYDSL_GROUPED_TOPK=1 to use
# the FlyDSL kernel. For vLLM, this var is registered in ``vllm/envs.py`` so it
# is propagated to the spawned EngineCore/worker processes (plain AITER_* vars
# are not). Unsupported configs and any compile/launch failure still fall back
# to HIP automatically.
def _use_flydsl_grouped_topk() -> bool:
    # Evaluated at call time (not import time): vLLM injects this env var into the
    # spawned worker processes at runtime, which may happen after aiter.ops.topk is
    # first imported. Reading it lazily avoids that import-vs-injection race.
    return os.environ.get("AITER_USE_FLYDSL_GROUPED_TOPK", "0") in (
        "1",
        "true",
        "True",
        "yes",
        "on",
    )


_FLYDSL_WARP_SIZE = 64


def _flydsl_grouped_topk_supported(
    gating_output: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    """The FlyDSL kernel specializes per-config and has launch constraints
    (one wavefront per token). Guard the cases it supports; otherwise the
    caller falls back to the HIP op."""
    num_experts = gating_output.shape[1]
    G = int(num_expert_group)
    K = int(topk_ids.shape[1])
    if G <= 0 or num_experts % G != 0:
        return False
    if not (1 <= int(topk_group) <= G):
        return False
    if K > _FLYDSL_WARP_SIZE or G > _FLYDSL_WARP_SIZE:
        return False
    if gating_output.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    return True


def _flydsl_biased_grouped_topk_supported(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    """Same launch constraints as the non-biased FlyDSL kernel, plus: the
    group-limited (``G > 1``) biased path scores a group by the sum of its
    top-2 experts, which needs at least 2 experts per group."""
    if not _flydsl_grouped_topk_supported(
        gating_output, topk_ids, num_expert_group, topk_group
    ):
        return False
    num_experts = gating_output.shape[1]
    G = int(num_expert_group)
    K = int(topk_ids.shape[1])
    if G > 1 and (num_experts // G) < 2:
        return False
    if correction_bias is None:
        return False
    # The HIP op uses a dedicated sort-based biased kernel for exactly this
    # 256-expert config; that kernel deliberately drops ``routed_scaling_factor``
    # when ``need_renorm`` is false, so it is not bit-compatible with our FlyDSL
    # path. Defer to HIP there to guarantee identical results.
    if K == 8 and G == 8 and num_experts == 256 and int(topk_group) == 4:
        return False
    return True


def grouped_topk(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    is_softmax: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Grouped (group-limited) top-k expert routing.

    Defaults to the HIP ``grouped_topk_hip`` kernel. When
    ``AITER_USE_FLYDSL_GROUPED_TOPK=1`` and the configuration is supported,
    dispatches to the FlyDSL ``flydsl_grouped_topk`` kernel instead. Both write
    their result in place into ``topk_weights`` / ``topk_ids``."""
    if _use_flydsl_grouped_topk() and _flydsl_grouped_topk_supported(
        gating_output, topk_ids, num_expert_group, topk_group
    ):
        try:
            from .flydsl import flydsl_grouped_topk

            flydsl_grouped_topk(
                gating_output,
                topk_weights,
                topk_ids,
                num_expert_group,
                topk_group,
                need_renorm,
                is_softmax,
                routed_scaling_factor,
            )
            return
        except Exception:
            # Any FlyDSL compile/launch failure falls back to the HIP op.
            pass
    return grouped_topk_hip(
        gating_output,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        need_renorm,
        is_softmax,
        routed_scaling_factor,
    )


def gen_moe_fused_gate_fake_tensor(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(
        topk_weights, dtype=topk_weights.dtype, device=topk_weights.device
    )

    indices = torch.empty_like(topk_ids, dtype=topk_ids.dtype, device=topk_ids.device)

    return [output, indices]


@compile_ops("module_moe_asm", gen_fake=gen_moe_fused_gate_fake_tensor)
def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    n_share_experts_fusion: int,
    routed_scaling_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]: ...


def biased_grouped_topk(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,  # mul to topk_weights
):
    token_num = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    cu_num = get_cu_num()
    if token_num <= cu_num * 212 or num_experts // num_expert_group > 32:
        # Optional FlyDSL replacement for the HIP biased op, gated by the same
        # env switch as the non-biased grouped_topk (default: HIP). Falls back
        # to HIP on any unsupported config or compile/launch failure.
        if _use_flydsl_grouped_topk() and _flydsl_biased_grouped_topk_supported(
            gating_output, correction_bias, topk_ids, num_expert_group, topk_group
        ):
            try:
                from .flydsl import flydsl_biased_grouped_topk

                flydsl_biased_grouped_topk(
                    gating_output,
                    correction_bias,
                    topk_weights,
                    topk_ids,
                    num_expert_group,
                    topk_group,
                    need_renorm,
                    routed_scaling_factor,
                )
                return topk_weights, topk_ids
            except Exception:
                pass
        return biased_grouped_topk_hip(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            need_renorm,
            routed_scaling_factor,
        )
    else:
        topk = topk_ids.shape[1]
        assert need_renorm, "Renormalization is required for moe_fused_gate."
        return moe_fused_gate(
            gating_output,
            correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            topk,
            n_share_experts_fusion=0,
            routed_scaling_factor=routed_scaling_factor,
        )


# this one copied from sglang
def biased_grouped_topk_torch(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    return_score: bool = False,
):
    scores = gating_output.to(dtypes.fp32).sigmoid()
    num_token = scores.shape[0]

    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)

    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if return_score:
        return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32), scores
    else:
        return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32)


# this one copied from sglang
def grouped_topk_torch(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
):
    gating_output = gating_output.to(dtypes.fp32)
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Scoring function '{scoring_func}' is not supported.")

    num_token = scores.shape[0]
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(dtypes.fp32), topk_ids.to(dtypes.i32)


@compile_ops("module_top_k_per_row")
def top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: Optional[torch.Tensor],
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...


@compile_ops("module_top_k_per_row", ffi_type="ctypes")
def top_k_per_row_prefill_fast(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: Optional[torch.Tensor],
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...


@compile_ops("module_top_k_per_row")
def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...


@compile_ops("module_top_k_per_row", ffi_type="ctypes")
def top_k_per_row_decode_fast(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
) -> None: ...
