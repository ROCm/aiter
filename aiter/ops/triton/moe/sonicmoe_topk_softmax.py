# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.sonicmoe.topk_softmax import (
    _softmax_over_topk_bwd_kernel,
    _topk_over_softmax_bwd_kernel,
)
from aiter.ops.triton.moe.sonicmoe_token_gather import (
    token_gather_and_sum_varlen_K_triton,
)

LIBRARY_NAME = "aiter_sonicmoe"


@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward_rocm", mutates_args={"o"})
def _router_forward(
    y: torch.Tensor,
    o: torch.Tensor,
    topk_scores: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    token_gather_and_sum_varlen_K_triton(
        y,
        topk_scores,
        o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_fwd_rocm",
    mutates_args={"topk_router_score", "topk_router_indices"},
)
def _topk_softmax_fwd(
    router_logits: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
) -> None:
    if is_softmax_over_topk:
        topk_results = router_logits.topk(K, dim=-1)
        vals = topk_results.values.softmax(dim=-1, dtype=torch.float32)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))
    else:
        probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        topk_results = probs.topk(K, dim=-1)
        vals = topk_results.values
        if norm_topk_probs:
            vals = vals / vals.sum(dim=-1, keepdim=True)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_topk_softmax_bwd_rocm", mutates_args={"dlogits_full"}
)
def _topk_softmax_bwd(
    router_logits: torch.Tensor,
    dlogits_full: torch.Tensor,
    dlogits: torch.Tensor | None,
    dtopk_score: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
) -> None:
    T = dtopk_score.shape[0]

    if is_softmax_over_topk:
        _softmax_over_topk_bwd_kernel[T,](
            dlogits,
            dlogits_full,
            topk_router_score,
            dtopk_score,
            topk_router_indices,
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            K,
            triton.next_power_of_2(K),
            (dlogits is None),
        )
    else:
        _topk_over_softmax_bwd_kernel[T,](
            router_logits,
            dlogits_full,
            dtopk_score,
            topk_router_indices,
            topk_router_score,
            router_logits.stride(0),
            router_logits.stride(1),
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            E,
            K,
            triton.next_power_of_2(E),
            triton.next_power_of_2(K),
            norm_topk_probs,
        )
