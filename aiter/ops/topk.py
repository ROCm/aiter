# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
from typing import Optional, Tuple

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num
from ..utility import dtypes


# DEPRECATED: low-level binding kept for backward compatibility only.
# Will be removed once all callers have migrated to topk_gating() below.
# New code should use topk_gating(), which:
#   - accepts an Optional[Tensor] correction_bias (None => no bias)
#   - validates score_func string
#   - exposes the same C++ kernel under a more accurate name
@compile_ops("module_moe_topk")
def topk_softplus(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    need_renorm: bool,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sqrtsoftplus",
) -> None: ...


_VALID_SCORE_FUNCS = {"sqrtsoftplus", "sigmoid", "softmax"}


def topk_gating(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: Optional[torch.Tensor] = None,
    need_renorm: bool = True,
    routed_scaling_factor: float = 1.0,
    score_func: str = "sqrtsoftplus",
) -> None:
    """Unified fused topk gating for MoE routing.

    Args:
        score_func: one of {"sqrtsoftplus" (DeepSeek V4-Pro default),
                            "sigmoid" (Llama4),
                            "softmax" (DeepSeek V3 / classic MoE)}.
        correction_bias: optional bias tensor, pass None for no bias.

    Note: softmax is already normalized, so renorm is forced off.
    """
    assert (
        score_func in _VALID_SCORE_FUNCS
    ), f"Unknown score_func '{score_func}', expected one of {_VALID_SCORE_FUNCS}"
    if correction_bias is None:
        # Match gating dtype/device so dispatch picks DTYPE_B == DTYPE_I,
        # avoiding extra kernel template instantiations.
        correction_bias = torch.empty(
            0, dtype=gating_output.dtype, device=gating_output.device
        )
    if score_func == "softmax":
        need_renorm = False
    topk_softplus(
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        need_renorm,
        routed_scaling_factor,
        score_func,
    )


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


@compile_ops("module_moe_asm")
def grouped_topk(
    gating_output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    need_renorm: bool,
    is_softmax: bool = True,
    routed_scaling_factor: float = 1.0,
) -> None: ...


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


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_prefill")
def _top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: Optional[torch.Tensor],
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: Optional[torch.Tensor] = None,
) -> None: ...


@compile_ops("module_top_k_per_row")
def topk_mb_workspace_size(
    numRows: int, stride0: int, k: int, is_decode: bool
) -> int: ...


@compile_ops("module_top_k_per_row")
def topk_use_mulblocks(numRows: int, stride0: int) -> bool: ...


@functools.lru_cache(maxsize=16)
def _get_topk_mb_workspace_keyed(
    device: torch.device, stream_id: int, size: int
) -> torch.Tensor:
    return torch.zeros(size, dtype=torch.uint8, device=device)


def get_topk_mb_workspace(device: torch.device, size: int) -> torch.Tensor:
    """Return a per-(device, stream, bucketed-size) zero-initialized workspace
    for the multi-block radix top-k path.

    The mb kernel uses cross-block atomic counters / histograms that must start
    at zero; instead of a per-call ``hipMemset`` the kernel resets the scratch
    back to zero after each launch, so a cached zeroed buffer can be reused.
    Concurrent launches on different streams must not share the buffer, or their
    atomic counters get mixed. Do not call from paths that violate the kernel's
    self-reset invariant.

    ``size`` is data-dependent (batch / seq_len / k), so it is rounded up to the
    next power of two before keying/allocating. That bounds the number of
    distinct cached buffers to ~log2(max_size) magnitudes (and the LRU cap of 16
    bounds it further) instead of one buffer per exact shape, trading <=2x size
    per buffer for far fewer retained buffers. The C++ side lays out its scratch
    within the first ``size`` bytes, so a larger (rounded) buffer is fine.
    """
    # Round up to the next power of two (size >= 1) to bucket nearby shapes.
    alloc = 1 if size <= 1 else 1 << (int(size) - 1).bit_length()
    stream = torch.cuda.current_stream(device)
    return _get_topk_mb_workspace_keyed(device, stream.cuda_stream, alloc)


def top_k_per_row_prefill(
    logits: torch.Tensor,
    rowStarts: torch.Tensor,
    rowEnds: torch.Tensor,
    indices: torch.Tensor,
    values: Optional[torch.Tensor],
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
) -> None:
    """Per-row top-k (prefill). The multi-block path runs on a persistent,
    zero-initialized workspace (memset-free; see get_topk_mb_workspace); the
    one-block path allocates its own scratch internally."""
    workspace = None
    if topk_use_mulblocks(numRows, stride0):
        size = topk_mb_workspace_size(numRows, stride0, k, False)
        workspace = get_topk_mb_workspace(logits.device, size)
    return _top_k_per_row_prefill(
        logits,
        rowStarts,
        rowEnds,
        indices,
        values,
        numRows,
        stride0,
        stride1,
        k,
        workspace,
    )


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


@compile_ops("module_top_k_per_row", fc_name="top_k_per_row_decode")
def _top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    workspace: Optional[torch.Tensor] = None,
) -> None: ...


def top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seqLens: torch.Tensor,
    indices: torch.Tensor,
    numRows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
) -> None:
    """Per-row top-k (decode). Multi-block path uses a persistent, zeroed
    workspace (memset-free); one-block path allocates internally."""
    workspace = None
    if topk_use_mulblocks(numRows, stride0):
        size = topk_mb_workspace_size(numRows, stride0, k, True)
        workspace = get_topk_mb_workspace(logits.device, size)
    return _top_k_per_row_decode(
        logits, next_n, seqLens, indices, numRows, stride0, stride1, k, workspace
    )


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
