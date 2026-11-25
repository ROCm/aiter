# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Top-k routing: softmax over router logits, then top-k experts per token.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py, which re-exports it for its callers
(`from aiter.fused_moe import fused_topk`). Changes must keep the behavior those callers
see and should loop in the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.utility import dtypes


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    num_valid_rows: torch.Tensor | None = None,  # see aiter.topk_gating
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape
    expert = gating_output.shape[1]

    token_expert_indicies = torch.empty(
        M, topk, dtype=dtypes.i32, device=hidden_states.device
    )

    # The asm kernel takes no pad-row count.
    if (
        num_valid_rows is None
        and get_gfx() in ["gfx942", "gfx950"]
        and (expert, topk)
        in [
            (128, 4),
            (128, 6),
            (128, 8),
            (256, 6),
            (256, 8),
            (384, 8),
            (640, 8),
        ]
        and gating_output.dtype in [dtypes.bf16, dtypes.fp32]
        and gating_output.is_contiguous()
    ):
        if topk_weights is None:
            topk_weights = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                (M + 3) // 4 * 4, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax_asm(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )
        topk_weights = topk_weights[:M, :]
        topk_ids = topk_ids[:M, :]
    else:
        if topk_weights is None:
            topk_weights = torch.empty(
                M, topk, dtype=dtypes.fp32, device=hidden_states.device
            )
        if topk_ids is None:
            topk_ids = torch.empty(
                M, topk, dtype=dtypes.i32, device=hidden_states.device
            )
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
            num_valid_rows=num_valid_rows,
        )

    del token_expert_indicies  # Not used. Will be used in the future.

    # if renormalize:
    #     topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
