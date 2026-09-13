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
    from aiter import dtypes, fused_dynamic_mxfp8_quant_moe_sort
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_mixed_latent_fhmoe_gemm1,
        compile_mixed_latent_fhmoe_gemm2,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from aiter.ops.flydsl.moe_kernels import _run_compiled
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd

    m = routed_input.shape[0]
    routed_experts = routed_w1.shape[0]
    routed_topk = topk_ids.shape[1]
    all_ids, all_weights, _ = integrated_route_domain(
        topk_ids, topk_weight, routed_experts
    )
    total_topk = routed_topk + 1
    block_m = 32

    num_experts = routed_experts + 1
    max_sorted = int(all_ids.numel() + num_experts * block_m - total_topk)
    max_blocks = (max_sorted + block_m - 1) // block_m
    sorted_ids = torch.empty(max_sorted, dtype=torch.int32, device=routed_input.device)
    sorted_weights = torch.empty(
        max_sorted, dtype=torch.float32, device=routed_input.device
    )
    sorted_expert_ids = torch.empty(
        max_blocks, dtype=torch.int32, device=routed_input.device
    )
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=routed_input.device)
    sort_zero = torch.empty((m, 3584), dtype=torch.bfloat16, device=routed_input.device)
    flydsl_moe_sorting_fwd(
        all_ids,
        all_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sort_zero,
        num_experts,
        block_m,
        None,
        None,
    )

    routed_a, routed_a_scale = fused_dynamic_mxfp8_quant_moe_sort(
        routed_input,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=m,
        topk=total_topk,
        block_size=block_m,
        sorted_weights=sorted_weights,
    )

    slot_bytes = 768 * 2
    inter_storage = torch.empty(
        (m, total_topk, slot_bytes), dtype=torch.uint8, device=routed_input.device
    )
    sorted_rows = max(sorted_ids.shape[0], sorted_expert_ids.shape[0] * block_m)
    scale_rows = (sorted_rows + 255) // 256 * 256
    scale_cols = ((384 // 32 + 7) // 8) * 8
    routed_inter_scale = torch.empty(
        scale_rows * scale_cols, dtype=torch.uint8, device=routed_input.device
    )
    empty_u8 = torch.empty(0, dtype=torch.uint8, device=routed_input.device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=routed_input.device)
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

    routed_output = torch.zeros(
        (m, 3584), dtype=torch.bfloat16, device=routed_input.device
    )
    shared_output = torch.zeros(
        (m, 7168), dtype=torch.bfloat16, device=routed_input.device
    )
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
