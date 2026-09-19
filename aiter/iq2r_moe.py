# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-safe GPT-OSS TP1 orchestration for native-basis IQ2R experts."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor

from .ops.iq2r import (
    iq2r_route_gather_quant_out,
    iq2r_route_direct_gather_quant_out,
    iq2r_route_reduce_add_rmsnorm_indexed_out,
    iq2r_route_reduce_indexed_out,
    iq2r_route_sort_tasks_out,
    iq2r_route_topk_direct_gather_quant_out,
    iq2r_route_topk_sort_gather_quant_out,
    iq2r_swiglu_quant_out,
    iq2r_task_capacity,
    iq2r_task_gemm_out,
)
from .ops.iq2r_format import (
    IQ2R_GPT_OSS_DOWN_N,
    IQ2R_GPT_OSS_EXPERTS,
    IQ2R_GPT_OSS_GATE_UP_N,
    IQ2R_GPT_OSS_K,
    IQ2R_GPT_OSS_TOP_K,
    IQ2R_SCALE_BLOCK,
    IQ2RMetadata,
    iq2r_validate_expert_weights,
)

IQ2R_DEFAULT_TASK_ROWS = 16
# GPT-OSS top-k=4 with the production 16K-token prefill budget.  The route
# sorter is grid-stride and the workspace is caller-owned, so this is a memory
# contract rather than a kernel launch-shape limit.
IQ2R_MAX_ROUTES = 65536
IQ2R_DIRECT_ROUTE_MAX = int(os.environ.get("IQ2R_DIRECT_ROUTE_MAX", "16"))
if not 0 <= IQ2R_DIRECT_ROUTE_MAX <= 16:
    raise ValueError("IQ2R_DIRECT_ROUTE_MAX must be in [0,16]")
IQ2R_FUSED_SORT_MAX_TOKENS = 16


@dataclass(slots=True)
class IQ2RMoeWorkspace:
    """Caller-owned, fixed-capacity buffers for capture-safe IQ2R MoE."""

    max_tokens: int
    topk: int
    max_experts: int
    task_rows: int
    scale_layout: str
    route_input_fp8: Tensor
    route_input_scales: Tensor
    gate_up: Tensor
    intermediate_fp8: Tensor
    intermediate_scales: Tensor
    route_output: Tensor
    sorted_expert_ids: Tensor
    gather_indices: Tensor
    scatter_indices: Tensor
    tasks: Tensor
    task_count: Tensor
    topk_weights: Tensor
    topk_ids: Tensor

    @property
    def max_routes(self) -> int:
        return self.max_tokens * self.topk

    @classmethod
    def allocate(
        cls,
        max_tokens: int,
        topk: int,
        *,
        device: torch.device | str,
        max_experts: int = IQ2R_GPT_OSS_EXPERTS,
        task_rows: int = IQ2R_DEFAULT_TASK_ROWS,
        scale_layout: str = "row_major",
    ) -> "IQ2RMoeWorkspace":
        for name, value in (
            ("max_tokens", max_tokens),
            ("topk", topk),
            ("max_experts", max_experts),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if topk != IQ2R_GPT_OSS_TOP_K:
            raise ValueError(f"GPT-OSS IQ2R requires topk={IQ2R_GPT_OSS_TOP_K}")
        if max_experts > IQ2R_GPT_OSS_EXPERTS:
            raise ValueError(
                f"GPT-OSS IQ2R supports at most {IQ2R_GPT_OSS_EXPERTS} experts"
            )
        if scale_layout not in ("row_major", "tile16"):
            raise ValueError("scale_layout must be 'row_major' or 'tile16'")
        routes = max_tokens * topk
        if routes > IQ2R_MAX_ROUTES:
            raise ValueError(
                f"IQ2R workspace supports at most {IQ2R_MAX_ROUTES} routes, got {routes}"
            )
        task_capacity = iq2r_task_capacity(routes, max_experts, task_rows)
        bf16 = {"dtype": torch.bfloat16, "device": device}
        fp8 = {"dtype": torch.float8_e4m3fn, "device": device}
        u8 = {"dtype": torch.uint8, "device": device}
        i32 = {"dtype": torch.int32, "device": device}
        scale_columns = IQ2R_GPT_OSS_K // IQ2R_SCALE_BLOCK
        scale_shape = (
            (routes, scale_columns)
            if scale_layout == "row_major"
            else (
                (scale_columns + 3) // 4,
                (routes + 15) // 16,
                4,
                16,
            )
        )
        return cls(
            max_tokens=max_tokens,
            topk=topk,
            max_experts=max_experts,
            task_rows=task_rows,
            scale_layout=scale_layout,
            route_input_fp8=torch.empty((routes, IQ2R_GPT_OSS_K), **fp8),
            route_input_scales=torch.empty(scale_shape, **u8),
            gate_up=torch.empty((routes, IQ2R_GPT_OSS_GATE_UP_N), **bf16),
            intermediate_fp8=torch.empty((routes, IQ2R_GPT_OSS_K), **fp8),
            intermediate_scales=torch.empty(scale_shape, **u8),
            route_output=torch.empty((routes, IQ2R_GPT_OSS_DOWN_N), **bf16),
            sorted_expert_ids=torch.empty((routes,), **i32),
            gather_indices=torch.empty((routes,), **i32),
            scatter_indices=torch.empty((routes,), **i32),
            tasks=torch.empty((task_capacity, 3), **i32),
            task_count=torch.empty((1,), **i32),
            topk_weights=torch.empty(
                (max_tokens, topk), dtype=torch.float32, device=device
            ),
            topk_ids=torch.empty((max_tokens, topk), **i32),
        )


def _validate_bias(
    name: str,
    value: Tensor | None,
    experts: int,
    width: int,
    device: torch.device,
) -> None:
    if value is None:
        return
    if value.dtype != torch.bfloat16 or tuple(value.shape) != (experts, width):
        raise ValueError(f"{name} must be BF16 [{experts},{width}]")
    if value.device != device or not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _validate_workspace(
    workspace: IQ2RMoeWorkspace,
    device: torch.device,
) -> None:
    routes = workspace.max_routes
    scale_columns = IQ2R_GPT_OSS_K // IQ2R_SCALE_BLOCK
    task_capacity = iq2r_task_capacity(
        routes, workspace.max_experts, workspace.task_rows
    )
    scale_shape = (
        (routes, scale_columns)
        if workspace.scale_layout == "row_major"
        else (
            (scale_columns + 3) // 4,
            (routes + 15) // 16,
            4,
            16,
        )
    )
    expected = (
        (
            "route_input_fp8",
            workspace.route_input_fp8,
            torch.float8_e4m3fn,
            (routes, 2880),
        ),
        (
            "route_input_scales",
            workspace.route_input_scales,
            torch.uint8,
            scale_shape,
        ),
        ("gate_up", workspace.gate_up, torch.bfloat16, (routes, 5760)),
        (
            "intermediate_fp8",
            workspace.intermediate_fp8,
            torch.float8_e4m3fn,
            (routes, 2880),
        ),
        (
            "intermediate_scales",
            workspace.intermediate_scales,
            torch.uint8,
            scale_shape,
        ),
        ("route_output", workspace.route_output, torch.bfloat16, (routes, 2880)),
        (
            "sorted_expert_ids",
            workspace.sorted_expert_ids,
            torch.int32,
            (routes,),
        ),
        ("gather_indices", workspace.gather_indices, torch.int32, (routes,)),
        ("scatter_indices", workspace.scatter_indices, torch.int32, (routes,)),
        ("tasks", workspace.tasks, torch.int32, (task_capacity, 3)),
        ("task_count", workspace.task_count, torch.int32, (1,)),
        (
            "topk_weights",
            workspace.topk_weights,
            torch.float32,
            (workspace.max_tokens, workspace.topk),
        ),
        (
            "topk_ids",
            workspace.topk_ids,
            torch.int32,
            (workspace.max_tokens, workspace.topk),
        ),
    )
    for name, tensor, dtype, shape in expected:
        if (
            tensor.device != device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"workspace.{name} must be contiguous {dtype} {shape} on {device}"
            )


def iq2r_fused_moe_out(
    hidden_states: Tensor,
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    output: Tensor,
    *,
    gate_up_metadata: IQ2RMetadata,
    down_metadata: IQ2RMetadata,
    gate_up_tile_n: int,
    down_tile_n: int,
    gate_up_bias: Tensor | None,
    down_bias: Tensor | None,
    workspace: IQ2RMoeWorkspace,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    renormalize: bool = True,
    residual: Tensor | None = None,
    norm_weight: Tensor | None = None,
    residual_out: Tensor | None = None,
    norm_epsilon: float = 1e-5,
    norm_block_size: int = 1024,
) -> None:
    """Execute GPT-OSS IQ2R MoE, optionally fusing the following add/RMSNorm."""

    gate_up_metadata.validate_gpt_oss()
    down_metadata.validate_gpt_oss()
    if (
        gate_up_metadata.logical_n != IQ2R_GPT_OSS_GATE_UP_N
        or down_metadata.logical_n != IQ2R_GPT_OSS_DOWN_N
    ):
        raise ValueError("gate_up_metadata and down_metadata are reversed")
    iq2r_validate_expert_weights(
        gate_up_data,
        gate_up_auxiliary,
        gate_up_metadata,
        verify_reserved_zero=False,
    )
    iq2r_validate_expert_weights(
        down_data,
        down_auxiliary,
        down_metadata,
        verify_reserved_zero=False,
    )
    if gate_up_data.shape[0] != down_data.shape[0]:
        raise ValueError("gate/up and down IQ2R expert counts differ")
    experts = gate_up_data.shape[0]
    if experts > workspace.max_experts:
        raise ValueError(
            f"workspace supports at most {workspace.max_experts} experts, got {experts}"
        )
    if gate_up_tile_n != 128 or down_tile_n != 64:
        raise ValueError("GPT-OSS IQ2R requires gate/up tile_n=128 and down tile_n=64")
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.ndim != 2
        or hidden_states.shape[1] != IQ2R_GPT_OSS_K
    ):
        raise ValueError("hidden_states must be BF16 [tokens,2880]")
    if (
        not hidden_states.is_cuda
        or hidden_states.stride(-1) != 1
        or hidden_states.stride(0) < IQ2R_GPT_OSS_K
    ):
        raise ValueError(
            "hidden_states must be on a GPU with contiguous columns and "
            "non-overlapping rows"
        )
    device = hidden_states.device
    weight_tensors = (
        gate_up_data,
        gate_up_auxiliary,
        down_data,
        down_auxiliary,
    )
    if any(t.device != device for t in weight_tensors):
        raise ValueError("all IQ2R weights must be on the hidden-state GPU")
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        raise TypeError("topk_ids must be int32 and topk_weights must be float32")
    if topk_ids.ndim != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same rank-2 shape")
    tokens, topk = topk_ids.shape
    if tokens != hidden_states.shape[0] or topk != IQ2R_GPT_OSS_TOP_K:
        raise ValueError("IQ2R routing must match tokens and use GPT-OSS top-k 4")
    if any(t.device != device for t in (topk_ids, topk_weights)):
        raise ValueError("routing tensors must be on the hidden-state GPU")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("routing tensors must be contiguous")
    if output.dtype != torch.bfloat16 or tuple(output.shape) != tuple(
        hidden_states.shape
    ):
        raise ValueError("output must be BF16 with the hidden-state shape")
    if output.device != device or not output.is_contiguous():
        raise ValueError("output must be contiguous on the hidden-state GPU")
    fused_norm_args = (residual, norm_weight, residual_out)
    if any(value is not None for value in fused_norm_args):
        if any(value is None for value in fused_norm_args):
            raise ValueError(
                "residual, norm_weight, and residual_out must be provided together"
            )
        assert residual is not None
        assert norm_weight is not None
        assert residual_out is not None
        if (
            residual.dtype != torch.bfloat16
            or tuple(residual.shape) != tuple(hidden_states.shape)
            or residual.device != device
            or not residual.is_contiguous()
        ):
            raise ValueError("residual must be contiguous BF16 [tokens,2880]")
        if (
            norm_weight.dtype != torch.bfloat16
            or tuple(norm_weight.shape) != (IQ2R_GPT_OSS_K,)
            or norm_weight.device != device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError("norm_weight must be contiguous BF16 [2880]")
        if (
            residual_out.dtype != torch.bfloat16
            or tuple(residual_out.shape) != tuple(hidden_states.shape)
            or residual_out.device != device
            or not residual_out.is_contiguous()
        ):
            raise ValueError("residual_out must be contiguous BF16 [tokens,2880]")
    _validate_bias("gate_up_bias", gate_up_bias, experts, 5760, device)
    _validate_bias("down_bias", down_bias, experts, 2880, device)
    if workspace.topk != topk or tokens > workspace.max_tokens:
        raise ValueError(
            f"workspace supports max_tokens={workspace.max_tokens}, "
            f"topk={workspace.topk}; got tokens={tokens}, topk={topk}"
        )
    _validate_workspace(workspace, device)

    if router_logits is not None:
        if router_logits.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("router_logits must be BF16 or FP32")
        if tuple(router_logits.shape) != (tokens, experts):
            raise ValueError(f"router_logits must have shape [{tokens},{experts}]")
        if router_logits.device != device:
            raise ValueError("router_logits must be on the hidden-state GPU")
        if router_logits.stride(-1) != 1 or router_logits.stride(0) < experts:
            raise ValueError("router_logits must have contiguous non-overlapping rows")
    if router_bias is not None:
        if router_logits is None:
            raise ValueError("router_bias requires router_logits")
        if router_bias.dtype != torch.bfloat16 or tuple(router_bias.shape) != (
            experts,
        ):
            raise ValueError(f"router_bias must be BF16 [{experts}]")
        if router_bias.device != device or not router_bias.is_contiguous():
            raise ValueError("router_bias must be contiguous on the hidden-state GPU")

    routes = tokens * topk
    # The pinned high-M prefetch kernel is validated for the production
    # 1024-token prefill shape (GPT-OSS top-k=4 => 4096 routed rows). Matching
    # its 32-row tile there avoids serial work for a hot expert while leaving
    # larger prefills on the established task policy; stitched M=2048 captures
    # regress when this specialization is applied beyond its measured shape.
    task_rows = 32 if routes == 4096 else workspace.task_rows
    regular_task_capacity = iq2r_task_capacity(routes, experts, workspace.task_rows)
    direct_task_mode = (
        routes <= IQ2R_DIRECT_ROUTE_MAX and workspace.tasks.shape[0] >= routes
    )
    task_capacity = routes if direct_task_mode else regular_task_capacity
    route_input_fp8 = workspace.route_input_fp8[:routes]
    route_input_scales = (
        workspace.route_input_scales[:routes]
        if workspace.scale_layout == "row_major"
        else workspace.route_input_scales
    )
    gate_up = workspace.gate_up[:routes]
    intermediate_fp8 = workspace.intermediate_fp8[:routes]
    intermediate_scales = (
        workspace.intermediate_scales[:routes]
        if workspace.scale_layout == "row_major"
        else workspace.intermediate_scales
    )
    route_output = workspace.route_output[:routes]
    sorted_expert_ids = workspace.sorted_expert_ids[:routes]
    gather_indices = workspace.gather_indices[:routes]
    scatter_indices = workspace.scatter_indices[:routes]
    tasks = workspace.tasks[:task_capacity]

    if router_logits is not None:
        if tokens > IQ2R_FUSED_SORT_MAX_TOKENS:
            raise ValueError("fused IQ2R top-k routing is available only for M<=16")
        if direct_task_mode:
            iq2r_route_topk_direct_gather_quant_out(
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
                sorted_expert_ids,
                gather_indices,
                scatter_indices,
                tasks,
                workspace.task_count,
                route_input_fp8,
                route_input_scales,
                renormalize=renormalize,
                router_bias=router_bias,
            )
        else:
            iq2r_route_topk_sort_gather_quant_out(
                hidden_states,
                router_logits,
                topk_weights,
                topk_ids,
                sorted_expert_ids,
                gather_indices,
                scatter_indices,
                tasks,
                workspace.task_count,
                route_input_fp8,
                route_input_scales,
                task_rows=task_rows,
                renormalize=renormalize,
                router_bias=router_bias,
            )
    elif direct_task_mode:
        iq2r_route_direct_gather_quant_out(
            hidden_states,
            topk_ids.reshape(-1),
            sorted_expert_ids,
            gather_indices,
            scatter_indices,
            tasks,
            workspace.task_count,
            route_input_fp8,
            route_input_scales,
            topk=topk,
            expert_count=experts,
        )
    else:
        iq2r_route_sort_tasks_out(
            topk_ids.reshape(-1),
            sorted_expert_ids,
            gather_indices,
            scatter_indices,
            tasks,
            workspace.task_count,
            expert_count=experts,
            task_rows=task_rows,
        )
        iq2r_route_gather_quant_out(
            hidden_states,
            gather_indices,
            route_input_fp8,
            route_input_scales,
            topk=topk,
        )
    iq2r_task_gemm_out(
        route_input_fp8,
        route_input_scales,
        gate_up_data,
        gate_up_auxiliary,
        tasks,
        workspace.task_count,
        gate_up_metadata,
        gate_up,
        tile_n=gate_up_tile_n,
        bias=gate_up_bias,
    )
    iq2r_swiglu_quant_out(
        gate_up,
        intermediate_fp8,
        intermediate_scales,
    )
    iq2r_task_gemm_out(
        intermediate_fp8,
        intermediate_scales,
        down_data,
        down_auxiliary,
        tasks,
        workspace.task_count,
        down_metadata,
        route_output,
        tile_n=down_tile_n,
        bias=down_bias,
    )
    if residual is None:
        iq2r_route_reduce_indexed_out(
            route_output,
            topk_weights,
            scatter_indices,
            output,
            topk=topk,
        )
    else:
        assert norm_weight is not None
        assert residual_out is not None
        iq2r_route_reduce_add_rmsnorm_indexed_out(
            route_output,
            topk_weights,
            scatter_indices,
            residual,
            norm_weight,
            output,
            residual_out,
            topk=topk,
            epsilon=norm_epsilon,
            block_size=norm_block_size,
        )


def iq2r_fused_moe(
    hidden_states: Tensor,
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    *,
    gate_up_metadata: IQ2RMetadata,
    down_metadata: IQ2RMetadata,
    gate_up_tile_n: int,
    down_tile_n: int,
    gate_up_bias: Tensor | None,
    down_bias: Tensor | None,
    workspace: IQ2RMoeWorkspace,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    renormalize: bool = True,
    output: Tensor | None = None,
) -> Tensor:
    """Wrapper for :func:`iq2r_fused_moe_out` with optional caller output."""

    # ``hidden_states`` may be the logical 2880-column view of ATOM's
    # 3072-stride TP1 layernorm output.  The result is a dense logical tensor;
    # preserving the input stride would waste storage and violate the out-op
    # contract.
    if output is None:
        output = torch.empty(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    iq2r_fused_moe_out(
        hidden_states,
        gate_up_data,
        gate_up_auxiliary,
        down_data,
        down_auxiliary,
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=gate_up_metadata,
        down_metadata=down_metadata,
        gate_up_tile_n=gate_up_tile_n,
        down_tile_n=down_tile_n,
        gate_up_bias=gate_up_bias,
        down_bias=down_bias,
        workspace=workspace,
        router_logits=router_logits,
        router_bias=router_bias,
        renormalize=renormalize,
    )
    return output


def iq2r_fused_moe_add_rmsnorm(
    hidden_states: Tensor,
    gate_up_data: Tensor,
    gate_up_auxiliary: Tensor,
    down_data: Tensor,
    down_auxiliary: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    *,
    gate_up_metadata: IQ2RMetadata,
    down_metadata: IQ2RMetadata,
    gate_up_tile_n: int,
    down_tile_n: int,
    gate_up_bias: Tensor | None,
    down_bias: Tensor | None,
    workspace: IQ2RMoeWorkspace,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    renormalize: bool = True,
    norm_epsilon: float = 1e-5,
    norm_block_size: int = 1024,
) -> tuple[Tensor, Tensor]:
    """Execute IQ2R MoE and the following residual-add RMSNorm in one tail op."""

    output = torch.empty(
        hidden_states.shape,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    residual_out = torch.empty_like(output)
    iq2r_fused_moe_out(
        hidden_states,
        gate_up_data,
        gate_up_auxiliary,
        down_data,
        down_auxiliary,
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=gate_up_metadata,
        down_metadata=down_metadata,
        gate_up_tile_n=gate_up_tile_n,
        down_tile_n=down_tile_n,
        gate_up_bias=gate_up_bias,
        down_bias=down_bias,
        workspace=workspace,
        router_logits=router_logits,
        router_bias=router_bias,
        renormalize=renormalize,
        residual=residual,
        norm_weight=norm_weight,
        residual_out=residual_out,
        norm_epsilon=norm_epsilon,
        norm_block_size=norm_block_size,
    )
    return output, residual_out


__all__ = [
    "IQ2R_DEFAULT_TASK_ROWS",
    "IQ2R_DIRECT_ROUTE_MAX",
    "IQ2R_FUSED_SORT_MAX_TOKENS",
    "IQ2R_MAX_ROUTES",
    "IQ2RMoeWorkspace",
    "iq2r_fused_moe",
    "iq2r_fused_moe_add_rmsnorm",
    "iq2r_fused_moe_out",
]
