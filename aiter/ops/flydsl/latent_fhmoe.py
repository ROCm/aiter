# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL bridge for the Kimi-K3 latent FHMoE contract."""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class LatentFHMoELayout:
    """Compile-time geometry shared by both integrated stage launches."""

    routed_model_dim: int = 3584
    routed_inter_dim: int = 384
    shared_model_dim: int = 7168
    shared_inter_dim: int = 768

    @property
    def max_model_dim(self) -> int:
        return max(self.routed_model_dim, self.shared_model_dim)

    @property
    def max_inter_dim(self) -> int:
        return max(self.routed_inter_dim, self.shared_inter_dim)


@dataclass
class _LatentFHMoEWorkspace:
    all_ids: torch.Tensor
    all_weights: torch.Tensor
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    sort_zero: torch.Tensor
    routed_a: torch.Tensor
    routed_a_scale: torch.Tensor
    inter_storage: torch.Tensor
    routed_inter_scale: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    empty_u8: torch.Tensor
    empty_f32: torch.Tensor


# CUDA graphs replay raw addresses. Keep every kernel-visible temporary alive
# for the lifetime of the layer weight that owns this workspace.
_WORKSPACES: dict[tuple[int, int, int, int], _LatentFHMoEWorkspace] = {}


def _get_workspace(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_topk: int,
) -> _LatentFHMoEWorkspace:
    from aiter import dtypes

    m = routed_input.shape[0]
    total_topk = routed_topk + 1
    routed_experts = routed_w1.shape[0]
    num_tasks = m * total_topk
    max_active_experts = min(routed_experts + 1, num_tasks)
    max_sorted = (
        num_tasks + max_active_experts * 31 + 31
    ) // 32 * 32
    max_blocks = max_sorted // 32
    scale_rows = (max_sorted + 255) // 256 * 256
    device_index = routed_input.device.index
    assert device_index is not None
    key = (device_index, routed_w1.data_ptr(), m, routed_topk)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        device = routed_input.device
        workspace = _LatentFHMoEWorkspace(
            all_ids=torch.empty(
                (m, total_topk), dtype=torch.int32, device=device
            ),
            all_weights=torch.empty(
                (m, total_topk), dtype=torch.float32, device=device
            ),
            sorted_ids=torch.empty(max_sorted, dtype=torch.int32, device=device),
            sorted_weights=torch.empty(
                max_sorted, dtype=torch.float32, device=device
            ),
            sorted_expert_ids=torch.empty(
                max_blocks, dtype=torch.int32, device=device
            ),
            num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
            sort_zero=torch.empty(
                (m, 3584), dtype=torch.bfloat16, device=device
            ),
            routed_a=torch.empty((m, 3584), dtype=dtypes.fp8, device=device),
            routed_a_scale=torch.empty(
                (max_sorted, 112), dtype=dtypes.fp8_e8m0, device=device
            ),
            inter_storage=torch.empty(
                (m, total_topk, 1536), dtype=torch.uint8, device=device
            ),
            routed_inter_scale=torch.empty(
                scale_rows * 16, dtype=torch.uint8, device=device
            ),
            routed_output=torch.empty(
                (m, 3584), dtype=torch.bfloat16, device=device
            ),
            shared_output=torch.empty(
                (m, 7168), dtype=torch.bfloat16, device=device
            ),
            empty_u8=torch.empty(0, dtype=torch.uint8, device=device),
            empty_f32=torch.empty(0, dtype=torch.float32, device=device),
        )
        _WORKSPACES[key] = workspace
    return workspace


def integrated_route_domain(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    routed_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Append one shared task per token to the routed sorting domain.

    The returned synthetic expert id is used only as the format/task selector.
    Both stage kernels must consume the resulting single sorted domain.
    """
    m = topk_ids.shape[0]
    shared_ids = torch.full(
        (m, 1), routed_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    shared_weights = torch.ones(
        (m, 1), dtype=topk_weight.dtype, device=topk_weight.device
    )
    return (
        torch.cat((topk_ids, shared_ids), dim=1),
        torch.cat((topk_weight, shared_weights), dim=1),
        routed_experts,
    )


def run_latent_fhmoe(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one integrated stage1 kernel and one integrated stage2 kernel."""
    from aiter import dtypes
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_mixed_latent_fhmoe_gemm1,
        compile_mixed_latent_fhmoe_gemm2,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from aiter.ops.flydsl.moe_kernels import _run_compiled
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip

    m = routed_input.shape[0]
    routed_experts = routed_w1.shape[0]
    routed_topk = topk_ids.shape[1]
    total_topk = routed_topk + 1
    block_m = 32
    workspace = _get_workspace(routed_input, routed_w1, routed_topk)
    all_ids = workspace.all_ids
    all_weights = workspace.all_weights
    all_ids[:, :routed_topk].copy_(topk_ids)
    all_ids[:, routed_topk].fill_(routed_experts)
    all_weights[:, :routed_topk].copy_(topk_weight)
    all_weights[:, routed_topk].fill_(1.0)

    num_experts = routed_experts + 1
    # Stage kernels may speculatively read metadata for grid-tail blocks before
    # applying num_valid_ids. Keep those entries mapped to expert/token zero.
    sorted_ids = workspace.sorted_ids
    sorted_weights = workspace.sorted_weights
    sorted_expert_ids = workspace.sorted_expert_ids
    num_valid_ids = workspace.num_valid_ids
    sorted_ids.zero_()
    sorted_weights.zero_()
    sorted_expert_ids.zero_()
    flydsl_moe_sorting_fwd(
        all_ids,
        all_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        workspace.sort_zero,
        num_experts,
        block_m,
        None,
        None,
    )

    routed_a = workspace.routed_a
    routed_a_scale = workspace.routed_a_scale
    fused_dynamic_mx_quant_moe_sort_hip(
        routed_a,
        routed_a_scale,
        routed_input,
        sorted_ids,
        num_valid_ids,
        m,
        block_m,
        32,
        sorted_weights,
    )

    inter_storage = workspace.inter_storage
    routed_inter_scale = workspace.routed_inter_scale
    empty_u8 = workspace.empty_u8
    empty_f32 = workspace.empty_f32
    num_blocks = sorted_expert_ids.shape[0]

    stage1 = compile_mixed_latent_fhmoe_gemm1(experts=routed_experts, topk=routed_topk)
    _run_compiled(
        stage1,
        (
            ptr_arg(inter_storage),
            ptr_arg(routed_a),
            ptr_arg(shared_input),
            ptr_arg(routed_w1),
            ptr_arg(routed_a_scale),
            ptr_arg(routed_w1_scale),
            ptr_arg(shared_w1),
            ptr_arg(empty_u8),
            ptr_arg(sorted_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(sorted_weights),
            ptr_arg(num_valid_ids),
            ptr_arg(empty_f32),
            ptr_arg(routed_inter_scale),
            m,
            2 * 384,
            3584,
            num_blocks,
            beta,
            1.0 / beta,
            linear_beta,
            1.0 / linear_beta,
            float("inf"),
            torch.cuda.current_stream(),
        ),
    )
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("latent stage1 complete", flush=True)

    routed_output = workspace.routed_output
    shared_output = workspace.shared_output
    routed_output.zero_()
    shared_output.zero_()
    stage2 = compile_mixed_latent_fhmoe_gemm2(experts=routed_experts, topk=routed_topk)
    _run_compiled(
        stage2,
        (
            ptr_arg(routed_output),
            ptr_arg(shared_output),
            ptr_arg(inter_storage),
            ptr_arg(routed_w2),
            ptr_arg(routed_inter_scale.view(dtypes.fp8_e8m0)),
            ptr_arg(routed_w2_scale),
            ptr_arg(shared_w2),
            ptr_arg(empty_u8),
            ptr_arg(sorted_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(sorted_weights),
            ptr_arg(num_valid_ids),
            ptr_arg(empty_f32),
            m,
            3584,
            384,
            num_blocks,
            torch.cuda.current_stream(),
        ),
    )
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("latent stage2 complete", flush=True)
    return routed_output, shared_output
