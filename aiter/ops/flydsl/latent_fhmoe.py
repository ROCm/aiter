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
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    routed_a: torch.Tensor
    routed_a_scale: torch.Tensor
    inter_storage: torch.Tensor
    shared_inter: torch.Tensor
    routed_inter_scale: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    shared_sorted_ids: torch.Tensor
    shared_sorted_weights: torch.Tensor
    shared_expert_ids: torch.Tensor
    shared_num_valid_ids: torch.Tensor
    empty_u8: torch.Tensor
    empty_f32: torch.Tensor
    shared_stream: torch.cuda.Stream
    shared_start: torch.cuda.Event
    shared_done: torch.cuda.Event


@dataclass
class _IntegratedLatentFHMoEWorkspace:
    all_ids: torch.Tensor
    all_weights: torch.Tensor
    sorted_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    routed_a: torch.Tensor
    routed_a_scale: torch.Tensor
    inter_storage: torch.Tensor
    shared_inter: torch.Tensor
    routed_inter_scale: torch.Tensor
    routed_output: torch.Tensor
    shared_output: torch.Tensor
    empty_u8: torch.Tensor
    empty_f32: torch.Tensor


# CUDA graphs replay raw addresses. Keep every kernel-visible temporary alive
# for the lifetime of the layer weight that owns this workspace.
_WORKSPACES: dict[tuple[int, int, int, int, int], _LatentFHMoEWorkspace] = {}
_INTEGRATED_WORKSPACES: dict[
    tuple[int, int, int, int, int], _IntegratedLatentFHMoEWorkspace
] = {}
_LATENT_BLOCK_M = 32


def _get_workspace(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_topk: int,
    block_m: int,
) -> _LatentFHMoEWorkspace:
    from aiter import dtypes

    m = routed_input.shape[0]
    routed_experts = routed_w1.shape[0]
    num_tasks = m * routed_topk
    max_active_experts = min(routed_experts, num_tasks)
    max_sorted = (
        num_tasks
        + max_active_experts * (block_m - 1)
        + (block_m - 1)
    ) // block_m * block_m
    max_blocks = max_sorted // block_m
    scale_rows = (max_sorted + 255) // 256 * 256
    device_index = routed_input.device.index
    assert device_index is not None
    key = (device_index, routed_w1.data_ptr(), m, routed_topk, block_m)
    workspace = _WORKSPACES.get(key)
    if workspace is None:
        device = routed_input.device
        workspace = _LatentFHMoEWorkspace(
            sorted_ids=torch.zeros(max_sorted, dtype=torch.int32, device=device),
            sorted_weights=torch.zeros(
                max_sorted, dtype=torch.float32, device=device
            ),
            sorted_expert_ids=torch.zeros(
                max_blocks, dtype=torch.int32, device=device
            ),
            num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
            routed_a=torch.empty((m, 3584), dtype=dtypes.fp8, device=device),
            routed_a_scale=torch.empty(
                (max_sorted, 112), dtype=dtypes.fp8_e8m0, device=device
            ),
            inter_storage=torch.empty(
                (max_sorted, 1536), dtype=torch.uint8, device=device
            ),
            shared_inter=torch.empty(
                (max(m, block_m), 768),
                dtype=torch.bfloat16,
                device=device,
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
            shared_sorted_ids=torch.full(
                (block_m,), m, dtype=torch.int32, device=device
            ),
            shared_sorted_weights=torch.zeros(
                block_m, dtype=torch.float32, device=device
            ),
            shared_expert_ids=torch.zeros(1, dtype=torch.int32, device=device),
            shared_num_valid_ids=torch.full(
                (2,), block_m, dtype=torch.int32, device=device
            ),
            empty_u8=torch.empty(0, dtype=torch.uint8, device=device),
            empty_f32=torch.empty(0, dtype=torch.float32, device=device),
            shared_stream=torch.cuda.Stream(device=device),
            shared_start=torch.cuda.Event(),
            shared_done=torch.cuda.Event(),
        )
        workspace.shared_sorted_ids[:m].copy_(
            torch.arange(m, dtype=torch.int32, device=device)
        )
        workspace.shared_sorted_weights[:m].fill_(1.0)
        _WORKSPACES[key] = workspace
    return workspace


def _get_integrated_workspace(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_topk: int,
    block_m: int,
) -> _IntegratedLatentFHMoEWorkspace:
    from aiter import dtypes

    m = routed_input.shape[0]
    total_topk = routed_topk + 1
    routed_experts = routed_w1.shape[0]
    num_tasks = m * total_topk
    max_active_experts = min(routed_experts + 1, num_tasks)
    max_sorted = (
        num_tasks
        + max_active_experts * (block_m - 1)
        + (block_m - 1)
    ) // block_m * block_m
    max_blocks = max_sorted // block_m
    scale_rows = (max_sorted + 255) // 256 * 256
    device_index = routed_input.device.index
    assert device_index is not None
    key = (device_index, routed_w1.data_ptr(), m, routed_topk, block_m)
    workspace = _INTEGRATED_WORKSPACES.get(key)
    if workspace is None:
        device = routed_input.device
        workspace = _IntegratedLatentFHMoEWorkspace(
            all_ids=torch.empty(
                (m, total_topk), dtype=torch.int32, device=device
            ),
            all_weights=torch.empty(
                (m, total_topk), dtype=torch.float32, device=device
            ),
            sorted_ids=torch.zeros(max_sorted, dtype=torch.int32, device=device),
            sorted_weights=torch.zeros(
                max_sorted, dtype=torch.float32, device=device
            ),
            sorted_expert_ids=torch.zeros(
                max_blocks, dtype=torch.int32, device=device
            ),
            num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
            routed_a=torch.empty((m, 3584), dtype=dtypes.fp8, device=device),
            routed_a_scale=torch.empty(
                (max_sorted, 112), dtype=dtypes.fp8_e8m0, device=device
            ),
            inter_storage=torch.empty(
                (max_sorted, 1536), dtype=torch.uint8, device=device
            ),
            shared_inter=torch.empty(
                (m * total_topk, 768), dtype=torch.bfloat16, device=device
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
        workspace.all_ids[:, routed_topk].fill_(routed_experts)
        workspace.all_weights[:, routed_topk].fill_(1.0)
        _INTEGRATED_WORKSPACES[key] = workspace
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


def _run_latent_fhmoe_integrated(
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
    """Run routed MXFP4 and shared BF16 tasks in one sorted persistent grid."""
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
    if m != 8:
        raise ValueError(
            "integrated latent FHMoE is currently correctness-validated only for M=8"
        )
    routed_experts = routed_w1.shape[0]
    routed_topk = topk_ids.shape[1]
    block_m = _LATENT_BLOCK_M
    workspace = _get_integrated_workspace(
        routed_input, routed_w1, routed_topk, block_m
    )
    all_ids = workspace.all_ids
    all_weights = workspace.all_weights
    all_ids[:, :routed_topk].copy_(topk_ids)
    all_weights[:, :routed_topk].copy_(topk_weight)

    sorted_ids = workspace.sorted_ids
    sorted_weights = workspace.sorted_weights
    sorted_expert_ids = workspace.sorted_expert_ids
    num_valid_ids = workspace.num_valid_ids
    routed_output = workspace.routed_output
    flydsl_moe_sorting_fwd(
        all_ids,
        all_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        routed_output,
        routed_experts + 1,
        block_m,
        None,
        None,
        last_expert_after=routed_experts // 4,
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
    shared_inter = workspace.shared_inter
    routed_inter_scale = workspace.routed_inter_scale
    empty_u8 = workspace.empty_u8
    empty_f32 = workspace.empty_f32
    num_blocks = sorted_expert_ids.shape[0]
    stream = torch.cuda.current_stream()

    stage1_config = os.environ.get("AITER_K3_LATENT_INTEGRATED_STAGE1_CONFIG")
    if stage1_config:
        s1 = tuple(int(value) for value in stage1_config.split(","))
        if len(s1) != 6:
            raise ValueError(
                "AITER_K3_LATENT_INTEGRATED_STAGE1_CONFIG requires "
                "tile_m,tile_n,tile_k,persist_m,waves_per_eu,xcd_swizzle"
            )
        stage1 = compile_mixed_latent_fhmoe_gemm1(
            experts=routed_experts,
            topk=routed_topk,
            tile_m=s1[0],
            tile_n=s1[1],
            tile_k=s1[2],
            persist_m=s1[3],
            waves_per_eu=s1[4] or None,
            xcd_swizzle=s1[5],
        )
    else:
        stage1 = compile_mixed_latent_fhmoe_gemm1(
            experts=routed_experts, topk=routed_topk
        )
    _run_compiled(
        stage1,
        (
            ptr_arg(inter_storage),
            ptr_arg(shared_inter),
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
            stream,
        ),
    )
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("integrated latent stage1 complete", flush=True)

    shared_output = workspace.shared_output
    shared_output.zero_()
    stage2_config = os.environ.get("AITER_K3_LATENT_INTEGRATED_STAGE2_CONFIG")
    if stage2_config:
        s2 = tuple(int(value) for value in stage2_config.split(","))
        if len(s2) != 7:
            raise ValueError(
                "AITER_K3_LATENT_INTEGRATED_STAGE2_CONFIG requires "
                "tile_m,tile_n,tile_k,persist_m,sort_block_m,"
                "waves_per_eu,xcd_swizzle"
            )
        stage2 = compile_mixed_latent_fhmoe_gemm2(
            experts=routed_experts,
            topk=routed_topk,
            tile_m=s2[0],
            tile_n=s2[1],
            tile_k=s2[2],
            persist_m=s2[3],
            sort_block_m=s2[4],
            waves_per_eu=s2[5] or None,
            xcd_swizzle=s2[6],
        )
    else:
        stage2 = compile_mixed_latent_fhmoe_gemm2(
            experts=routed_experts, topk=routed_topk
        )
    _run_compiled(
        stage2,
        (
            ptr_arg(routed_output),
            ptr_arg(shared_output),
            ptr_arg(inter_storage),
            ptr_arg(shared_inter),
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
            stream,
        ),
    )
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("integrated latent stage2 complete", flush=True)
    return routed_output, shared_output


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
    """Overlap routed MXFP8/MXFP4 and shared BF16 expert pipelines."""
    if os.environ.get("AITER_K3_LATENT_INTEGRATED", "0") == "1":
        return _run_latent_fhmoe_integrated(
            routed_input,
            routed_w1,
            routed_w2,
            routed_w1_scale,
            routed_w2_scale,
            topk_weight,
            topk_ids,
            shared_input,
            shared_w1,
            shared_w2,
            beta=beta,
            linear_beta=linear_beta,
        )

    from aiter import dtypes
    from aiter.ops.flydsl.kernels.fhmoe import (
        compile_latent_routed_gemm1,
        compile_latent_routed_gemm2,
    )
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import (
        flydsl_a16w4_gemm1,
        flydsl_a16w4_gemm2,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from aiter.ops.flydsl.moe_kernels import _run_compiled
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort_hip

    m = routed_input.shape[0]
    routed_experts = routed_w1.shape[0]
    routed_topk = topk_ids.shape[1]
    block_m = _LATENT_BLOCK_M
    workspace = _get_workspace(routed_input, routed_w1, routed_topk, block_m)
    main_stream = torch.cuda.current_stream()
    shared_stream = workspace.shared_stream
    shared_tile_n = int(os.environ.get("AITER_K3_LATENT_SHARED_TILE_N", "64"))
    shared_num_waves = int(
        os.environ.get("AITER_K3_LATENT_SHARED_NUM_WAVES", "2")
    )
    workspace.shared_start.record(main_stream)
    shared_stream.wait_event(workspace.shared_start)
    shared_output = workspace.shared_output
    with torch.cuda.stream(shared_stream):
        shared_output.zero_()
        flydsl_a16w4_gemm1(
            a_bf16=shared_input,
            w1_u8=shared_w1,
            w1_scale_u8=workspace.empty_u8,
            sorted_expert_ids=workspace.shared_expert_ids,
            cumsum_tensor=workspace.shared_num_valid_ids,
            m_indices=workspace.shared_sorted_ids,
            inter_sorted_bf16=workspace.shared_inter,
            n_tokens=m,
            NE=1,
            D_HIDDEN=7168,
            D_INTER=768,
            topk=1,
            tile_m=32,
            tile_n=shared_tile_n,
            tile_k=256,
            num_waves=shared_num_waves,
            act="situv2",
            situ_beta=beta,
            situ_linear_beta=linear_beta,
            w_dtype="bf16",
            stream=shared_stream,
        )
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=workspace.shared_inter,
            w2_u8=shared_w2,
            w2_scale_u8=workspace.empty_u8,
            sorted_expert_ids=workspace.shared_expert_ids,
            cumsum_tensor=workspace.shared_num_valid_ids,
            sorted_token_ids=workspace.shared_sorted_ids,
            sorted_weights=workspace.shared_sorted_weights,
            flat_out=shared_output,
            M_logical=m,
            max_sorted=workspace.shared_inter.shape[0],
            NE=1,
            D_HIDDEN=7168,
            D_INTER=768,
            topk=1,
            tile_m=32,
            tile_n=256,
            tile_k=128,
            xcd_swizzle=1,
            w_dtype="bf16",
            persist=False,
            stream=shared_stream,
        )
        workspace.shared_done.record(shared_stream)

    if os.environ.get("AITER_K3_LATENT_USE_FUSED_ROUTED", "0") == "1":
        from aiter import ActivationType, QuantType
        from aiter.fused_moe import fused_moe

        routed_output = fused_moe(
            routed_input,
            routed_w1,
            routed_w2,
            topk_weight,
            topk_ids,
            activation=ActivationType.Situv2,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=routed_w1_scale.view(-1, routed_w1_scale.shape[-1]),
            w2_scale=routed_w2_scale.view(-1, routed_w2_scale.shape[-1]),
            dtype=torch.bfloat16,
            beta=beta,
            linear_beta=linear_beta,
            gate_mode="interleave",
        )
        main_stream.wait_event(workspace.shared_done)
        return routed_output, shared_output

    num_experts = routed_experts
    # Tail entries are initialized to valid expert/token zero once. Sort may
    # leave values from a prior replay there, but all remain in bounds and
    # num_valid_ids prevents them from contributing to the result.
    sorted_ids = workspace.sorted_ids
    sorted_weights = workspace.sorted_weights
    sorted_expert_ids = workspace.sorted_expert_ids
    num_valid_ids = workspace.num_valid_ids
    routed_output = workspace.routed_output
    flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weight,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        routed_output,
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

    stage1 = compile_latent_routed_gemm1(
        experts=routed_experts, topk=routed_topk
    )
    _run_compiled(
        stage1,
        (
            ptr_arg(inter_storage),
            ptr_arg(routed_a),
            ptr_arg(routed_w1),
            ptr_arg(routed_a_scale),
            ptr_arg(routed_w1_scale),
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
            main_stream,
        ),
    )
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("latent stage1 complete", flush=True)

    stage2 = compile_latent_routed_gemm2(
        experts=routed_experts, topk=routed_topk
    )
    _run_compiled(
        stage2,
        (
            ptr_arg(routed_output),
            ptr_arg(inter_storage),
            ptr_arg(routed_w2),
            ptr_arg(routed_inter_scale.view(dtypes.fp8_e8m0)),
            ptr_arg(routed_w2_scale),
            ptr_arg(sorted_ids),
            ptr_arg(sorted_expert_ids),
            ptr_arg(sorted_weights),
            ptr_arg(num_valid_ids),
            ptr_arg(empty_f32),
            m,
            inter_storage.shape[0],
            3584,
            384,
            num_blocks,
            main_stream,
        ),
    )
    main_stream.wait_event(workspace.shared_done)
    if os.environ.get("AITER_LATENT_DEBUG_SYNC", "0") == "1":
        torch.cuda.synchronize()
        print("latent stage2 complete", flush=True)
    return routed_output, shared_output
