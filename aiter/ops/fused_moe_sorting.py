# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops

MD_NAME = "module_fused_moe_sorting"


@compile_ops("module_fused_moe_sorting", develop=True)
def fused_moe_sorting_get_workspace_size(
    tokens: int,
    num_experts: int,
    topk: int,
    unit_size: int,
) -> int: ...


@compile_ops("module_fused_moe_sorting", develop=True)
def fused_moe_sorting_is_supported(
    tokens: int,
    num_experts: int,
    topk: int,
    unit_size: int,
) -> bool: ...


@compile_ops("module_fused_moe_sorting", develop=True)
def fused_moe_sorting_fwd(
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
) -> None: ...


@compile_ops("module_fused_moe_sorting", develop=True)
def fused_moe_sorting_topk_is_supported(
    tokens: int,
    num_experts: int,
    topk: int,
    unit_size: int,
    router_experts: int,
) -> bool: ...


@compile_ops("module_fused_moe_sorting", develop=True)
def fused_moe_sorting_topk_fwd(
    gating_output: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    topk: int,
    unit_size: int,
    need_renorm: bool,
    is_softmax: bool,
    routed_scaling_factor: float,
    local_expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    sync: torch.Tensor | None = None,
) -> None: ...
