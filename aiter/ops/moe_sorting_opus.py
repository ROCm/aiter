# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops

MD_NAME = "module_moe_sorting_opus"


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_get_workspace_size(
    tokens: int,
    num_experts: int,
    topk: int,
    dispatch_policy: int = 0,
) -> int: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int,
    local_expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    dispatch_policy: int = 0,
    local_topk_ids: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    reverse_sorted: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def mxfp4_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale_token: torch.Tensor,
    num_experts: int,
) -> None:
    """Fused route sort + compact MXFP4 activation quantization.

    gfx950 only, ``model_dim=7168``, ``block_m=32``, ``M`` in ``[1, 128]``.
    ``(num_experts, topk)`` must be one of ``(256, 8)``, ``(384, 8)`` or
    ``(385, 9)`` -- the expert histogram is a kernel template parameter.
    """
