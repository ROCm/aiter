# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph-safe TP1 orchestration for native-basis IQ2R experts."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor

from .ops.iq2r import (
    IQ2R_MAX_ROUTES,
    iq2r_route_direct_gather_quant_out,
    iq2r_route_gather_quant_out,
    iq2r_route_reduce_add_indexed_out,
    iq2r_route_reduce_add_rmsnorm_indexed_out,
    iq2r_route_reduce_indexed_out,
    iq2r_route_scatter_quant_out,
    iq2r_route_sort_tasks_out,
    iq2r_route_topk_direct_gather_quant_out,
    iq2r_route_topk_sort_gather_quant_out,
    iq2r_swiglu_quant_out,
    iq2r_swiglu_quant_scatter_out,
    iq2r_task_capacity,
    iq2r_task_gemm_out,
    iq2r_down_sparse_scheduled_out,
    iq2r_down_shortk_out,
    iq2r_down_token_fused48_out,
    iq2r_down_token_route9_out,
    iq2r_down_token_pair9_out,
    iq2r_down_token_adaptive9_out,
    iq2r_gate_quad_scheduled_out,
    iq2r_glm53_dense_gate_out,
    iq2r_glm53_tp4_gate_out,
    iq2r_glm53_tp4_indexed_gate_out,
    iq2r_glm53_tp4_large_down_out,
    iq2r_glm53_tp4_down_out,
    iq2r_glm53_tp4_route9_out,
    iq2r_down_sparse_large32_out,
    iq2r_task_gemm_indexed_out,
    iq2r_gate_quad_fused_out,
    iq2r_gate_quad_sparse_out,
    iq2r_gate_quad_splitk_out,
    iq2r_gate_quad_route_fused_out,
)
from .ops.iq2r_format import (
    IQ2R_GPT_OSS_EXPERTS,
    IQ2R_GPT_OSS_K,
    IQ2R_GPT_OSS_TOP_K,
    IQ2R_SCALE_BLOCK,
    IQ2RMetadata,
    iq2r_validate_expert_weights,
)

_indexed_setting = os.environ.get("IQ2R_GLM53_INDEXED_INPUT", "0")
if _indexed_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_INDEXED_INPUT must be 0 or 1")
IQ2R_GLM53_INDEXED_INPUT = _indexed_setting == "1"
_IQ2R_INDEXED_AUDIT = os.environ.get("ATOM_IQ2R_AUDIT", "0") == "1"
_IQ2R_INDEXED_AUDITED = set()
_quad_setting = os.environ.get("IQ2R_GLM53_QUAD_GATE", "0")
if _quad_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_QUAD_GATE must be 0 or 1")
IQ2R_GLM53_QUAD_GATE = _quad_setting == "1"
_IQ2R_QUAD_AUDITED = set()
_sparse_quad_setting = os.environ.get("IQ2R_GLM53_SPARSE_QUAD_GATE", "0")
if _sparse_quad_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_SPARSE_QUAD_GATE must be 0 or 1")
IQ2R_GLM53_SPARSE_QUAD_GATE = _sparse_quad_setting == "1"
_IQ2R_SPARSE_QUAD_AUDITED = set()
_split4_setting = os.environ.get("IQ2R_GLM53_SPLIT4_GATE", "0")
if _split4_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_SPLIT4_GATE must be 0 or 1")
IQ2R_GLM53_SPLIT4_GATE = _split4_setting == "1"
_IQ2R_SPLIT4_AUDITED = set()
_small_quad_setting = os.environ.get("IQ2R_GLM53_SMALL_QUAD_GATE", "0")
if _small_quad_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_SMALL_QUAD_GATE must be 0 or 1")
IQ2R_GLM53_SMALL_QUAD_GATE = _small_quad_setting == "1"
_IQ2R_SMALL_QUAD_AUDITED = set()

_sparse_down_setting = os.environ.get("IQ2R_GLM53_SPARSE_DOWN", "0")
if _sparse_down_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_SPARSE_DOWN must be 0 or 1")
IQ2R_GLM53_SPARSE_DOWN = _sparse_down_setting == "1"
_IQ2R_SPARSE_DOWN_AUDITED = set()

_scheduled_setting = os.environ.get("IQ2R_GLM53_SCHEDULED", "0")
if _scheduled_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_SCHEDULED must be 0 or 1")
IQ2R_GLM53_SCHEDULED = _scheduled_setting == "1"
_IQ2R_SCHEDULED_AUDITED = set()
_IQ2R_SCHEDULED_POLICY = {
    1: {"gate": "register", "down": "register_single_epilogue"},
    2: {"gate": "register", "down": "register_single_epilogue"},
    4: {"gate": "wait3", "down": "register_single_epilogue"},
    8: {"gate": "wait3", "down": "register_single_epilogue"},
    16: {"gate": "wait3", "down": "register_single_epilogue"},
    256: {"gate": "unchanged", "down": "register_prefetch_uniform_deferred_scale"},
}
_IQ2R_SMALL_DOWN_VARIANTS = {
    "single_epilogue": 0,
    "single_epilogue_deferred": 1,
    "register_single_epilogue": 2,
}
_IQ2R_LARGE_DOWN_VARIANTS = {
    "fixed_kn": 0,
    "register_prefetch_uniform": 1,
    "fixed_kn_deferred_scale": 2,
    "register_prefetch_uniform_deferred_scale": 3,
}

_fused_down_setting = os.environ.get("IQ2R_GLM53_FUSED_DOWN", "0")
if _fused_down_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_FUSED_DOWN must be 0 or 1")
IQ2R_GLM53_FUSED_DOWN = _fused_down_setting == "1"
_IQ2R_FUSED_DOWN_AUDITED = set()

_c32_setting = os.environ.get("IQ2R_GLM53_C32", "0")
if _c32_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_C32 must be 0 or 1")
IQ2R_GLM53_C32 = _c32_setting == "1"
_IQ2R_C32_AUDITED = set()

_route9_setting = os.environ.get("IQ2R_GLM53_ROUTE9_DOWN", "0")
if _route9_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_ROUTE9_DOWN must be 0 or 1")
IQ2R_GLM53_ROUTE9_DOWN = _route9_setting == "1"

_direct_gate_setting = os.environ.get("IQ2R_GLM53_DIRECT_GATE", "0")
if _direct_gate_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_DIRECT_GATE must be 0 or 1")
IQ2R_GLM53_DIRECT_GATE = _direct_gate_setting == "1"

_codebook_batch_setting = os.environ.get("IQ2R_GLM53_CODEBOOK_BATCH", "0")
if _codebook_batch_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_CODEBOOK_BATCH must be 0 or 1")
IQ2R_GLM53_CODEBOOK_BATCH = _codebook_batch_setting == "1"

_tp4_setting = os.environ.get("IQ2R_GLM53_TP4", "0")
if _tp4_setting not in ("0", "1"):
    raise ValueError("IQ2R_GLM53_TP4 must be 0 or 1")
IQ2R_GLM53_TP4 = _tp4_setting == "1"

IQ2R_GLM53_PAIR_DOWN = int(os.environ.get("IQ2R_GLM53_PAIR_DOWN", "0"))
if IQ2R_GLM53_PAIR_DOWN not in (0, 1, 2, 4):
    raise ValueError("IQ2R_GLM53_PAIR_DOWN must be 0, 1, 2 or 4")
_dense_gate_setting = os.environ.get("IQ2R_GLM53_DENSE_GATE", "0")
if _dense_gate_setting not in ("0", "1", "2"):
    raise ValueError("IQ2R_GLM53_DENSE_GATE must be 0, 1 or 2")
IQ2R_GLM53_DENSE_GATE = int(_dense_gate_setting)
_IQ2R_DENSE_GATE_AUDITED = set()
IQ2R_GLM53_ADAPTIVE_DOWN = os.environ.get("IQ2R_GLM53_ADAPTIVE_DOWN", "0") == "1"
IQ2R_DEFAULT_TASK_ROWS = 16
IQ2R_MAX_EXPERTS = 512
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
    hidden_size: int
    intermediate_size: int
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
    token_identity: Tensor | None = None
    small_gate_partials: Tensor | None = None

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
        hidden_size: int = IQ2R_GPT_OSS_K,
        intermediate_size: int = IQ2R_GPT_OSS_K,
        task_rows: int = IQ2R_DEFAULT_TASK_ROWS,
        scale_layout: str = "row_major",
    ) -> IQ2RMoeWorkspace:
        for name, value in (
            ("max_tokens", max_tokens),
            ("topk", topk),
            ("max_experts", max_experts),
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if max_experts > IQ2R_MAX_EXPERTS:
            raise ValueError(
                f"IQ2R routing supports at most {IQ2R_MAX_EXPERTS} experts"
            )
        if hidden_size % IQ2R_SCALE_BLOCK or intermediate_size % IQ2R_SCALE_BLOCK:
            raise ValueError(
                f"hidden_size and intermediate_size must be divisible by "
                f"{IQ2R_SCALE_BLOCK}"
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

        def scale_shape(width: int) -> tuple[int, ...]:
            scale_columns = width // IQ2R_SCALE_BLOCK
            if scale_layout == "row_major":
                return (routes, scale_columns)
            return (
                (scale_columns + 3) // 4,
                (routes + 15) // 16,
                4,
                16,
            )

        return cls(
            max_tokens=max_tokens,
            topk=topk,
            max_experts=max_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            task_rows=task_rows,
            scale_layout=scale_layout,
            route_input_fp8=torch.empty((routes, hidden_size), **fp8),
            route_input_scales=torch.empty(scale_shape(hidden_size), **u8),
            gate_up=torch.empty((routes, 2 * intermediate_size), **bf16),
            intermediate_fp8=torch.empty((routes, intermediate_size), **fp8),
            intermediate_scales=torch.empty(scale_shape(intermediate_size), **u8),
            route_output=torch.empty((routes, hidden_size), **bf16),
            sorted_expert_ids=torch.empty((routes,), **i32),
            gather_indices=torch.empty((routes,), **i32),
            scatter_indices=torch.empty((routes,), **i32),
            tasks=torch.empty((task_capacity, 3), **i32),
            task_count=torch.empty((1,), **i32),
            topk_weights=torch.empty(
                (max_tokens, topk), dtype=torch.float32, device=device
            ),
            topk_ids=torch.empty((max_tokens, topk), **i32),
            token_identity=torch.arange(max_tokens, **i32),
            # Shared temporary sums, never persistent per-layer weights.
            # Bound storage to M16 even when the workspace also serves prefill.
            small_gate_partials=(
                torch.empty(
                    8 * min(max_tokens, 16) * 9 * 512,
                    dtype=torch.float32,
                    device=device,
                )
                if (topk, max_experts, hidden_size, intermediate_size, scale_layout)
                == (9, 257, 6144, 256, "row_major")
                else None
            ),
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
    task_capacity = iq2r_task_capacity(
        routes, workspace.max_experts, workspace.task_rows
    )

    def scale_shape(width: int) -> tuple[int, ...]:
        scale_columns = width // IQ2R_SCALE_BLOCK
        if workspace.scale_layout == "row_major":
            return (routes, scale_columns)
        return (
            (scale_columns + 3) // 4,
            (routes + 15) // 16,
            4,
            16,
        )

    expected = (
        (
            "route_input_fp8",
            workspace.route_input_fp8,
            torch.float8_e4m3fn,
            (routes, workspace.hidden_size),
        ),
        (
            "route_input_scales",
            workspace.route_input_scales,
            torch.uint8,
            scale_shape(workspace.hidden_size),
        ),
        (
            "gate_up",
            workspace.gate_up,
            torch.bfloat16,
            (routes, 2 * workspace.intermediate_size),
        ),
        (
            "intermediate_fp8",
            workspace.intermediate_fp8,
            torch.float8_e4m3fn,
            (routes, workspace.intermediate_size),
        ),
        (
            "intermediate_scales",
            workspace.intermediate_scales,
            torch.uint8,
            scale_shape(workspace.intermediate_size),
        ),
        (
            "route_output",
            workspace.route_output,
            torch.bfloat16,
            (routes, workspace.hidden_size),
        ),
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
    gate_quad_data: Tensor | None = None,
    expert_map: Tensor | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    global_expert_count: int | None = None,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    router_scoring_func: str = "softmax",
    router_routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    residual: Tensor | None = None,
    norm_weight: Tensor | None = None,
    residual_out: Tensor | None = None,
    shared_output: Tensor | None = None,
    pre_reduce_stream: torch.cuda.Stream | None = None,
    norm_epsilon: float = 1e-5,
    norm_block_size: int = 1024,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    swiglu_up_offset: float = 1.0,
) -> None:
    """Execute IQ2R MoE, optionally fusing the following add/RMSNorm."""

    gate_up_metadata.validate_layout()
    down_metadata.validate_layout()
    hidden_size = gate_up_metadata.logical_k
    intermediate_size = down_metadata.logical_k
    if (
        gate_up_metadata.logical_n != 2 * intermediate_size
        or down_metadata.logical_n != hidden_size
    ):
        raise ValueError(
            "IQ2R metadata must describe gate/up [2*intermediate,hidden] and "
            "down [hidden,intermediate] matrices"
        )
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
    if global_expert_count is None:
        global_expert_count = experts
    if (
        isinstance(expert_start, bool)
        or not isinstance(expert_start, int)
        or expert_start < 0
        or isinstance(expert_stride, bool)
        or not isinstance(expert_stride, int)
        or expert_stride <= 0
        or isinstance(global_expert_count, bool)
        or not isinstance(global_expert_count, int)
        or global_expert_count < experts
        or global_expert_count > 512
        or expert_start + (experts - 1) * expert_stride >= global_expert_count
    ):
        raise ValueError(
            "expert_start, expert_stride, and global_expert_count must define "
            "a valid strided local expert range within 512"
        )
    use_expert_parallel = global_expert_count > experts
    if experts > workspace.max_experts:
        raise ValueError(
            f"workspace supports at most {workspace.max_experts} experts, got {experts}"
        )
    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.ndim != 2
        or hidden_states.shape[1] != hidden_size
    ):
        raise ValueError(f"hidden_states must be BF16 [tokens,{hidden_size}]")
    if (
        not hidden_states.is_cuda
        or hidden_states.stride(-1) != 1
        or hidden_states.stride(0) < hidden_size
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
    if tokens != hidden_states.shape[0] or topk <= 0:
        raise ValueError("IQ2R routing must match tokens and use a positive top-k")
    if any(t.device != device for t in (topk_ids, topk_weights)):
        raise ValueError("routing tensors must be on the hidden-state GPU")
    if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
        raise ValueError("routing tensors must be contiguous")
    if expert_map is not None:
        if (
            expert_map.dtype != torch.int32
            or expert_map.ndim != 1
            or expert_map.numel() < global_expert_count
        ):
            raise ValueError("expert_map must be int32 and cover every global expert")
        if expert_map.device != device or not expert_map.is_contiguous():
            raise ValueError("expert_map must be contiguous on the hidden-state GPU")
        expert_map = expert_map[:global_expert_count]
    if output.dtype != torch.bfloat16 or tuple(output.shape) != tuple(
        hidden_states.shape
    ):
        raise ValueError("output must be BF16 with the hidden-state shape")
    if output.device != device or not output.is_contiguous():
        raise ValueError("output must be contiguous on the hidden-state GPU")
    if shared_output is not None:
        if (
            shared_output.dtype != torch.bfloat16
            or tuple(shared_output.shape) != tuple(hidden_states.shape)
            or shared_output.device != device
            or not shared_output.is_contiguous()
        ):
            raise ValueError(
                f"shared_output must be contiguous BF16 [tokens,{hidden_size}]"
            )
    elif pre_reduce_stream is not None:
        raise ValueError("pre_reduce_stream requires shared_output")
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
            raise ValueError(f"residual must be contiguous BF16 [tokens,{hidden_size}]")
        if (
            norm_weight.dtype != torch.bfloat16
            or tuple(norm_weight.shape) != (hidden_size,)
            or norm_weight.device != device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError(f"norm_weight must be contiguous BF16 [{hidden_size}]")
        if (
            residual_out.dtype != torch.bfloat16
            or tuple(residual_out.shape) != tuple(hidden_states.shape)
            or residual_out.device != device
            or not residual_out.is_contiguous()
        ):
            raise ValueError(
                f"residual_out must be contiguous BF16 [tokens,{hidden_size}]"
            )
        if shared_output is not None:
            raise ValueError("shared_output cannot be combined with fused RMSNorm")
    _validate_bias("gate_up_bias", gate_up_bias, experts, 2 * intermediate_size, device)
    _validate_bias("down_bias", down_bias, experts, hidden_size, device)
    if workspace.topk != topk or tokens > workspace.max_tokens:
        raise ValueError(
            f"workspace supports max_tokens={workspace.max_tokens}, "
            f"topk={workspace.topk}; got tokens={tokens}, topk={topk}"
        )
    _validate_workspace(workspace, device)
    if (
        workspace.hidden_size != hidden_size
        or workspace.intermediate_size != intermediate_size
    ):
        raise ValueError(
            "workspace hidden/intermediate dimensions do not match IQ2R metadata"
        )

    if router_logits is not None:
        gpt_oss_router = (
            global_expert_count == IQ2R_GPT_OSS_EXPERTS
            and experts == IQ2R_GPT_OSS_EXPERTS
            and topk == IQ2R_GPT_OSS_TOP_K
            and router_scoring_func == "softmax"
            and router_routed_scaling_factor == 1.0
            and not use_expert_parallel
        )
        glm53_router = (
            global_expert_count == 256
            and topk == 8
            and router_scoring_func == "sigmoid"
            and router_routed_scaling_factor == 2.5
        )
        if not (gpt_oss_router or glm53_router):
            raise ValueError(
                "the fused IQ2R router supports unsharded GPT-OSS "
                "(128 experts, top-k 4, softmax, scale 1) or GLM-5.3 "
                "(256 global experts, top-k 8, biased sigmoid, scale 2.5)"
            )
        if glm53_router and not renormalize:
            raise ValueError("the fused GLM-5.3 IQ2R router requires renormalization")
        if router_logits.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("router_logits must be BF16 or FP32")
        if tuple(router_logits.shape) != (tokens, global_expert_count):
            raise ValueError(
                f"router_logits must have shape [{tokens},{global_expert_count}]"
            )
        if router_logits.device != device:
            raise ValueError("router_logits must be on the hidden-state GPU")
        if (
            router_logits.stride(-1) != 1
            or router_logits.stride(0) < global_expert_count
        ):
            raise ValueError("router_logits must have contiguous non-overlapping rows")
    if router_bias is not None:
        if router_logits is None:
            raise ValueError("router_bias requires router_logits")
        expected_bias_dtype = (
            router_logits.dtype if router_scoring_func == "sigmoid" else torch.bfloat16
        )
        if router_bias.dtype != expected_bias_dtype or tuple(router_bias.shape) != (
            global_expert_count,
        ):
            raise ValueError(
                f"router_bias must be {expected_bias_dtype} [{global_expert_count}]"
            )
        if router_bias.device != device or not router_bias.is_contiguous():
            raise ValueError("router_bias must be contiguous on the hidden-state GPU")
    if (
        router_logits is not None
        and router_scoring_func == "sigmoid"
        and router_bias is None
    ):
        raise ValueError("the fused biased-sigmoid router requires correction bias")

    routes = tokens * topk
    scheduled_c32 = (
        IQ2R_GLM53_C32
        and IQ2R_GLM53_SCHEDULED
        and tokens == 32
        and IQ2R_GLM53_INDEXED_INPUT
        and IQ2R_GLM53_QUAD_GATE
        and IQ2R_GLM53_SPARSE_QUAD_GATE
        and hidden_size == 6144
        and (intermediate_size == 256 or (IQ2R_GLM53_TP4 and intermediate_size == 512))
        and gate_up_metadata.logical_n == 2 * intermediate_size
        and down_metadata.logical_n == 6144
        and (experts, global_expert_count, topk) == (257, 257, 9)
        and not use_expert_parallel
        and workspace.scale_layout == "row_major"
        and router_logits is None
        and gate_up_bias is None
        and down_bias is None
        and (swiglu_limit, swiglu_alpha, swiglu_up_offset) == (0.0, 1.0, 0.0)
    )
    use_glm53_tp_m32 = (
        (64 <= tokens <= 256 or scheduled_c32)
        and hidden_size == 6144
        and (intermediate_size == 256 or (IQ2R_GLM53_TP4 and intermediate_size == 512))
        and gate_up_metadata.logical_n == 2 * intermediate_size
        and down_metadata.logical_n == 6144
        and (global_expert_count, topk) in ((256, 8), (257, 9))
        and experts == global_expert_count
        and workspace.scale_layout == "row_major"
    )
    use_glm53_tp_prefill_m64 = (
        (tokens == 1536 or 2048 <= tokens <= 4096)
        and hidden_size == 6144
        and (intermediate_size == 256 or (IQ2R_GLM53_TP4 and intermediate_size == 512))
        and gate_up_metadata.logical_n == 2 * intermediate_size
        and down_metadata.logical_n == 6144
        and (global_expert_count, topk) == (257, 9)
        and experts == global_expert_count
        and not use_expert_parallel
        and workspace.scale_layout == "row_major"
    )
    # The pinned high-M prefetch kernel is validated for the production
    # 1024-token prefill shape (GPT-OSS top-k=4 => 4096 routed rows). Matching
    # its 32-row tile there avoids serial work for a hot expert while leaving
    # larger prefills on the established task policy; stitched M=2048 captures
    # regress when this specialization is applied beyond its measured shape.
    task_rows = (
        64
        if use_glm53_tp_prefill_m64
        else (
            32
            if (
                routes == 4096
                and hidden_size == IQ2R_GPT_OSS_K
                and topk == IQ2R_GPT_OSS_TOP_K
            )
            or use_glm53_tp_m32
            else workspace.task_rows
        )
    )
    regular_task_capacity = iq2r_task_capacity(routes, experts, task_rows)
    # The compact EP activation kernels are specialized for the production
    # top-k values. Keep the established zero-task fallback for other top-k
    # contracts instead of routing them into an unsupported specialization.
    compact_expert_parallel = use_expert_parallel and topk in (4, 8)
    # Large fused GLM prefills amortize quantization by broadcasting each
    # source token to its nine already-sorted destinations. Exact production
    # captures also favor this path at M256; preserve route-major gathering
    # below that boundary.
    glm53_fused_broadcast_quant = (
        routes >= 2304
        and hidden_size == 6144
        and intermediate_size == 256
        and gate_up_metadata.logical_n == 512
        and down_metadata.logical_n == 6144
        and global_expert_count == 257
        and experts == 257
        and topk == 9
        and not use_expert_parallel
        and workspace.scale_layout == "row_major"
    )
    # Fused shared-expert GLM decode has 9 routes per token, so M>=2 no
    # longer fits the generic <=16-route direct path.  Keep expert-grouped
    # tasks, but let the native front end build their permutation while it
    # quantizes each source row once.  This preserves GEMM codebook reuse and
    # removes the separate sorter/quantizer boundary.
    glm53_grouped_decode_mode = (
        2 <= tokens <= 16
        and hidden_size == 6144
        and (intermediate_size == 256 or (IQ2R_GLM53_TP4 and intermediate_size == 512))
        and gate_up_metadata.logical_n == 2 * intermediate_size
        and down_metadata.logical_n == 6144
        and global_expert_count == 257
        and experts == 257
        and topk == 9
        and not use_expert_parallel
        and workspace.scale_layout == "row_major"
        and router_logits is None
    )
    direct_task_mode = (
        routes <= IQ2R_DIRECT_ROUTE_MAX
        and workspace.tasks.shape[0] >= routes
        and (not use_expert_parallel or compact_expert_parallel)
    ) or glm53_grouped_decode_mode
    task_capacity = routes if direct_task_mode else regular_task_capacity
    # Match the native E060 cooperative guard, including its 228..255 gap.
    # Keep direct routing, EP, other layouts, and unsupported shapes unchanged.
    indexed_input = (
        IQ2R_GLM53_INDEXED_INPUT
        and (
            (use_glm53_tp_m32 and (tokens <= 227 or tokens == 256))
            or use_glm53_tp_prefill_m64
        )
        and topk == 9
        and experts == 257
        and not use_expert_parallel
        and router_logits is None
        and not direct_task_mode
    )
    input_rows = tokens if indexed_input else routes
    if indexed_input:
        identity = workspace.token_identity
        if (
            identity is None
            or identity.dtype != torch.int32
            or identity.device != hidden_states.device
            or tuple(identity.shape) != (workspace.max_tokens,)
            or not identity.is_contiguous()
        ):
            raise ValueError(
                "indexed gate requires the allocated int32 token identity buffer"
            )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_INDEXED_AUDITED:
            _IQ2R_INDEXED_AUDITED.add(tokens)
            print(
                f"IQ2R_INDEXED_INPUT pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} task_rows={task_rows} "
                "input_layout=token_major",
                flush=True,
            )
    route_input_fp8 = workspace.route_input_fp8[:input_rows]
    route_input_scales = (
        workspace.route_input_scales[:input_rows]
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
                scoring_func=router_scoring_func,
                routed_scaling_factor=router_routed_scaling_factor,
                expert_map=expert_map,
                expert_count=experts,
                expert_start=expert_start,
                expert_stride=expert_stride,
                drop_nonlocal_routes=compact_expert_parallel,
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
                scoring_func=router_scoring_func,
                routed_scaling_factor=router_routed_scaling_factor,
                expert_map=expert_map,
                expert_count=experts,
                expert_start=expert_start,
                expert_stride=expert_stride,
                drop_nonlocal_routes=compact_expert_parallel,
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
            expert_map=expert_map,
            expert_start=expert_start,
            expert_stride=expert_stride,
            drop_nonlocal_routes=compact_expert_parallel,
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
            expert_map=expert_map,
            expert_start=expert_start,
            expert_stride=expert_stride,
            task_rows=task_rows,
            drop_nonlocal_tasks=compact_expert_parallel,
        )
        if indexed_input:
            iq2r_route_gather_quant_out(
                hidden_states,
                workspace.token_identity[:tokens],
                route_input_fp8,
                route_input_scales,
                topk=1,
            )
        elif compact_expert_parallel or glm53_fused_broadcast_quant:
            iq2r_route_scatter_quant_out(
                hidden_states,
                scatter_indices,
                route_input_fp8,
                route_input_scales,
                topk=topk,
            )
        else:
            iq2r_route_gather_quant_out(
                hidden_states,
                gather_indices,
                route_input_fp8,
                route_input_scales,
                topk=topk,
            )
    scheduled_small = (
        IQ2R_GLM53_SCHEDULED
        and tokens in (1, 2, 4, 8, 16)
        and hidden_size == 6144
        and (intermediate_size == 256 or (IQ2R_GLM53_TP4 and intermediate_size == 512))
        and gate_up_metadata.logical_n == 2 * intermediate_size
        and down_metadata.logical_n == 6144
        and (experts, global_expert_count, topk) == (257, 257, 9)
        and not use_expert_parallel
        and workspace.scale_layout == "row_major"
        and router_logits is None
        and gate_up_bias is None
        and down_bias is None
        and (swiglu_limit, swiglu_alpha, swiglu_up_offset) == (0.0, 1.0, 0.0)
    )
    selected_gate = _IQ2R_SCHEDULED_POLICY[tokens]["gate"] if scheduled_small else None
    quad_gate = (
        IQ2R_GLM53_QUAD_GATE
        and indexed_input
        and (
            use_glm53_tp_m32 or (intermediate_size == 512 and use_glm53_tp_prefill_m64)
        )
    )
    small_split4 = (selected_gate == "split4") or (
        selected_gate is None
        and IQ2R_GLM53_SPLIT4_GATE
        and intermediate_size == 256
        and tokens == 4
        and glm53_grouped_decode_mode
    )
    small_quad = (selected_gate == "quad") or (
        selected_gate is None
        and IQ2R_GLM53_SMALL_QUAD_GATE
        and intermediate_size == 256
        and tokens == 4
        and glm53_grouped_decode_mode
    )
    if selected_gate in ("wait3", "register"):
        if gate_quad_data is None:
            raise ValueError(
                "scheduled gate requires current-layer quad weights before capture"
            )
        (
            iq2r_glm53_tp4_gate_out
            if intermediate_size == 512
            else iq2r_gate_quad_scheduled_out
        )(
            route_input_fp8,
            route_input_scales,
            gate_quad_data,
            gate_up_auxiliary,
            tasks,
            workspace.task_count,
            intermediate_fp8,
            intermediate_scales,
            variant=(0 if selected_gate == "wait3" else 1)
            + (
                (4 if IQ2R_GLM53_CODEBOOK_BATCH else 2) if IQ2R_GLM53_DIRECT_GATE else 0
            ),
        )
    elif small_quad:
        if small_split4:
            raise ValueError("Select only one experimental M4 gate")
        if gate_quad_data is None:
            raise ValueError("small quad requires current-layer weights before capture")
        if gate_up_bias is not None or (
            swiglu_limit,
            swiglu_alpha,
            swiglu_up_offset,
        ) != (0.0, 1.0, 0.0):
            raise ValueError("small quad requires bias-free plain GLM SwiGLU")
        iq2r_gate_quad_route_fused_out(
            route_input_fp8,
            route_input_scales,
            gate_quad_data,
            gate_up_auxiliary,
            tasks,
            workspace.task_count,
            intermediate_fp8,
            intermediate_scales,
        )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_SMALL_QUAD_AUDITED:
            _IQ2R_SMALL_QUAD_AUDITED.add(tokens)
            print(
                f"IQ2R_SMALL_QUAD_GATE pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} "
                "kernel=iq2r_gate_quad_route_fused_kernel",
                flush=True,
            )
    elif small_split4:
        if gate_quad_data is None:
            raise ValueError(
                "split4 requires current-layer quad weights before capture"
            )
        if gate_up_bias is not None or (
            swiglu_limit,
            swiglu_alpha,
            swiglu_up_offset,
        ) != (0.0, 1.0, 0.0):
            raise ValueError("split4 requires bias-free plain GLM SwiGLU")
        partials = workspace.small_gate_partials
        if (
            partials is None
            or partials.dtype != torch.float32
            or partials.device != hidden_states.device
            or partials.ndim != 1
            or not partials.is_contiguous()
            or partials.numel() < 8 * routes * 512
        ):
            raise ValueError("split4 requires preallocated contiguous FP32 scratch")
        # A prefix of flat storage keeps the native view contiguous even when
        # workspace.max_tokens is larger than this captured shape.
        partials = partials[: 8 * routes * 512].view(8, routes, 512)
        iq2r_gate_quad_splitk_out(
            route_input_fp8,
            route_input_scales,
            gate_quad_data,
            gate_up_auxiliary,
            tasks,
            workspace.task_count,
            partials,
            intermediate_fp8,
            intermediate_scales,
            physical_waves=4,
        )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_SPLIT4_AUDITED:
            _IQ2R_SPLIT4_AUDITED.add(tokens)
            print(
                f"IQ2R_SPLIT4_GATE pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} physical_waves=4 "
                "kernel=iq2r_gate_quad_splitk_kernel",
                flush=True,
            )
    elif (
        IQ2R_GLM53_DENSE_GATE
        and indexed_input
        and intermediate_size == 256
        and use_glm53_tp_prefill_m64
    ):
        if (
            gate_quad_data is None
            or gate_up_bias is not None
            or (swiglu_limit, swiglu_alpha, swiglu_up_offset) != (0.0, 1.0, 0.0)
        ):
            raise ValueError(
                "dense gate requires quad weights and plain bias-free SwiGLU"
            )
        iq2r_glm53_dense_gate_out(
            route_input_fp8,
            route_input_scales,
            gate_quad_data,
            gate_up_auxiliary,
            tasks,
            workspace.task_count,
            gather_indices,
            intermediate_fp8,
            intermediate_scales,
            rows_per_cta=64 if use_glm53_tp_prefill_m64 else 32,
            variant=1 if IQ2R_GLM53_DENSE_GATE == 2 and use_glm53_tp_prefill_m64 else 0,
        )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_DENSE_GATE_AUDITED:
            _IQ2R_DENSE_GATE_AUDITED.add(tokens)
            print(
                f"IQ2R_DENSE_GATE device={hidden_states.device.index} tokens={tokens} tp=8 "
                f"pipeline={IQ2R_GLM53_DENSE_GATE} rows=64",
                flush=True,
            )
    elif quad_gate:
        if gate_quad_data is None:
            raise ValueError("quad gate requires weights repacked before graph capture")
        if gate_up_bias is not None or (
            swiglu_limit,
            swiglu_alpha,
            swiglu_up_offset,
        ) != (0.0, 1.0, 0.0):
            raise ValueError("quad gate requires bias-free plain GLM SwiGLU")
        quad_op = (
            iq2r_glm53_tp4_indexed_gate_out
            if intermediate_size == 512
            else (
                iq2r_gate_quad_sparse_out
                if IQ2R_GLM53_SPARSE_QUAD_GATE
                else iq2r_gate_quad_fused_out
            )
        )
        quad_op(
            route_input_fp8,
            route_input_scales,
            gate_quad_data,
            gate_up_auxiliary,
            tasks,
            workspace.task_count,
            gather_indices,
            intermediate_fp8,
            intermediate_scales,
            rows_per_cta=64 if use_glm53_tp_prefill_m64 else 32,
        )
        if (
            _IQ2R_INDEXED_AUDIT
            and IQ2R_GLM53_SPARSE_QUAD_GATE
            and tokens not in _IQ2R_SPARSE_QUAD_AUDITED
        ):
            _IQ2R_SPARSE_QUAD_AUDITED.add(tokens)
            print(
                f"IQ2R_SPARSE_QUAD_GATE pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} kernel=iq2r_gate_quad_sparse_kernel",
                flush=True,
            )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_QUAD_AUDITED:
            _IQ2R_QUAD_AUDITED.add(tokens)
            print(
                f"IQ2R_QUAD_GATE pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} sparse={int(IQ2R_GLM53_SPARSE_QUAD_GATE)}",
                flush=True,
            )
    else:
        if indexed_input:
            iq2r_task_gemm_indexed_out(
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
                gather_indices=gather_indices,
            )
        else:
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
        if compact_expert_parallel:
            iq2r_swiglu_quant_scatter_out(
                gate_up,
                scatter_indices,
                intermediate_fp8,
                intermediate_scales,
                topk=topk,
                limit=swiglu_limit,
                alpha=swiglu_alpha,
                up_offset=swiglu_up_offset,
            )
        else:
            iq2r_swiglu_quant_out(
                gate_up,
                intermediate_fp8,
                intermediate_scales,
                limit=swiglu_limit,
                alpha=swiglu_alpha,
                up_offset=swiglu_up_offset,
            )
    sparse_down = (
        IQ2R_GLM53_SPARSE_DOWN
        and tokens == 256
        and use_glm53_tp_m32
        and (experts, topk) == (257, 9)
        and not use_expert_parallel
        and down_bias is None
    )
    selected_down = _IQ2R_SCHEDULED_POLICY[tokens]["down"] if scheduled_small else None
    scheduled_large = (
        IQ2R_GLM53_SCHEDULED
        and sparse_down
        and _IQ2R_SCHEDULED_POLICY[256]["down"] != "baseline"
    )
    pair_group = 0
    if intermediate_size == 256:
        if IQ2R_GLM53_PAIR_DOWN == 1:
            pair_group = {8: 2, 16: 4}.get(tokens, 0)
        elif tokens in (4, 8, 16):
            pair_group = IQ2R_GLM53_PAIR_DOWN
    if IQ2R_GLM53_ADAPTIVE_DOWN and intermediate_size == 256 and tokens == 8:
        pair_group = 2
    fused_down = (
        IQ2R_GLM53_FUSED_DOWN
        and scheduled_small
        and (tokens in (1, 2, 4) or pair_group)
        and (intermediate_size == 256 or IQ2R_GLM53_ROUTE9_DOWN)
        and residual is None
        and shared_output is None
        and pre_reduce_stream is None
    )
    if scheduled_c32:
        (iq2r_glm53_tp4_down_out if intermediate_size == 512 else iq2r_down_shortk_out)(
            intermediate_fp8,
            intermediate_scales,
            down_data,
            down_auxiliary,
            tasks,
            workspace.task_count,
            route_output,
            grid_multiplier=4,
            variant=2,
        )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_C32_AUDITED:
            _IQ2R_C32_AUDITED.add(tokens)
            print(
                f"IQ2R_C32 pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} gate=sparse_quad down=register_single_epilogue",
                flush=True,
            )
    elif fused_down:
        if IQ2R_GLM53_ADAPTIVE_DOWN and tokens == 8 and intermediate_size == 256:
            iq2r_down_token_adaptive9_out(
                intermediate_fp8,
                intermediate_scales,
                down_data,
                down_auxiliary,
                topk_ids,
                scatter_indices,
                topk_weights,
                output,
                workspace.task_count,
                tasks,
            )
        elif pair_group:
            iq2r_down_token_pair9_out(
                intermediate_fp8,
                intermediate_scales,
                down_data,
                down_auxiliary,
                topk_ids,
                scatter_indices,
                topk_weights,
                output,
                group_tokens=pair_group,
            )
        else:
            fused_down_op = (
                (
                    iq2r_glm53_tp4_route9_out
                    if intermediate_size == 512
                    else iq2r_down_token_route9_out
                )
                if IQ2R_GLM53_ROUTE9_DOWN
                else iq2r_down_token_fused48_out
            )
            fused_down_op(
                intermediate_fp8,
                intermediate_scales,
                down_data,
                down_auxiliary,
                topk_ids,
                scatter_indices,
                topk_weights,
                output,
            )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_FUSED_DOWN_AUDITED:
            _IQ2R_FUSED_DOWN_AUDITED.add(tokens)
            print(
                f"IQ2R_FUSED_DOWN pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} "
                f"kernel={'e172_compact_down_routes' if IQ2R_GLM53_ADAPTIVE_DOWN and tokens == 8 and intermediate_size == 256 else ('e141_down_grouped_routes' if pair_group else ('iq2r_down_token_route9_kernel' if IQ2R_GLM53_ROUTE9_DOWN else 'iq2r_down_token_fused48_kernel'))} group_tokens={'adaptive' if IQ2R_GLM53_ADAPTIVE_DOWN and tokens == 8 and intermediate_size == 256 else pair_group}",
                flush=True,
            )
    elif scheduled_small and selected_down in _IQ2R_SMALL_DOWN_VARIANTS:
        (iq2r_glm53_tp4_down_out if intermediate_size == 512 else iq2r_down_shortk_out)(
            intermediate_fp8,
            intermediate_scales,
            down_data,
            down_auxiliary,
            tasks,
            workspace.task_count,
            route_output,
            grid_multiplier=2 if tokens == 1 else 4,
            variant=_IQ2R_SMALL_DOWN_VARIANTS[selected_down],
        )
    elif scheduled_large:
        (
            iq2r_glm53_tp4_large_down_out
            if intermediate_size == 512
            else iq2r_down_sparse_scheduled_out
        )(
            intermediate_fp8,
            intermediate_scales,
            down_data,
            down_auxiliary,
            tasks,
            workspace.task_count,
            route_output,
            grid_multiplier=4,
            variant=_IQ2R_LARGE_DOWN_VARIANTS[_IQ2R_SCHEDULED_POLICY[256]["down"]],
        )
    elif sparse_down and intermediate_size == 256:
        iq2r_down_sparse_large32_out(
            intermediate_fp8,
            intermediate_scales,
            down_data,
            down_auxiliary,
            tasks,
            workspace.task_count,
            route_output,
            grid_multiplier=4,
        )
        if _IQ2R_INDEXED_AUDIT and tokens not in _IQ2R_SPARSE_DOWN_AUDITED:
            _IQ2R_SPARSE_DOWN_AUDITED.add(tokens)
            print(
                f"IQ2R_SPARSE_DOWN pid={os.getpid()} device={hidden_states.device.index} "
                f"tokens={tokens} routes={routes} grid_multiplier=4 "
                "kernel=iq2r_down_sparse_large32_kernel",
                flush=True,
            )
    else:
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
    if (
        (scheduled_small or scheduled_large)
        and _IQ2R_INDEXED_AUDIT
        and tokens not in _IQ2R_SCHEDULED_AUDITED
    ):
        _IQ2R_SCHEDULED_AUDITED.add(tokens)
        choice = dict(_IQ2R_SCHEDULED_POLICY[tokens])
        if IQ2R_GLM53_DIRECT_GATE and scheduled_small:
            choice["gate"] += (
                "_direct_batch" if IQ2R_GLM53_CODEBOOK_BATCH else "_direct"
            )
        if fused_down:
            choice["down"] = (
                "token_route9" if IQ2R_GLM53_ROUTE9_DOWN else "token_fused48"
            )
        print(
            f"IQ2R_SCHEDULED pid={os.getpid()} device={hidden_states.device.index} "
            f"tokens={tokens} routes={routes} gate={choice['gate']} down={choice['down']}",
            flush=True,
        )
    if fused_down:
        return
    if residual is None:
        if shared_output is None:
            iq2r_route_reduce_indexed_out(
                route_output,
                topk_weights,
                scatter_indices,
                output,
                topk=topk,
            )
        else:
            if pre_reduce_stream is not None:
                torch.cuda.current_stream(device).wait_stream(pre_reduce_stream)
            iq2r_route_reduce_add_indexed_out(
                route_output,
                topk_weights,
                scatter_indices,
                shared_output,
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
    gate_quad_data: Tensor | None = None,
    expert_map: Tensor | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    global_expert_count: int | None = None,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    router_scoring_func: str = "softmax",
    router_routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    output: Tensor | None = None,
    shared_output: Tensor | None = None,
    pre_reduce_stream: torch.cuda.Stream | None = None,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    swiglu_up_offset: float = 1.0,
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
        gate_quad_data=gate_quad_data,
        expert_map=expert_map,
        expert_start=expert_start,
        expert_stride=expert_stride,
        global_expert_count=global_expert_count,
        router_logits=router_logits,
        router_bias=router_bias,
        router_scoring_func=router_scoring_func,
        router_routed_scaling_factor=router_routed_scaling_factor,
        renormalize=renormalize,
        shared_output=shared_output,
        pre_reduce_stream=pre_reduce_stream,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_up_offset=swiglu_up_offset,
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
    gate_quad_data: Tensor | None = None,
    expert_map: Tensor | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    global_expert_count: int | None = None,
    router_logits: Tensor | None = None,
    router_bias: Tensor | None = None,
    router_scoring_func: str = "softmax",
    router_routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    norm_epsilon: float = 1e-5,
    norm_block_size: int = 1024,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    swiglu_up_offset: float = 1.0,
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
        gate_quad_data=gate_quad_data,
        expert_map=expert_map,
        expert_start=expert_start,
        expert_stride=expert_stride,
        global_expert_count=global_expert_count,
        router_logits=router_logits,
        router_bias=router_bias,
        router_scoring_func=router_scoring_func,
        router_routed_scaling_factor=router_routed_scaling_factor,
        renormalize=renormalize,
        residual=residual,
        norm_weight=norm_weight,
        residual_out=residual_out,
        norm_epsilon=norm_epsilon,
        norm_block_size=norm_block_size,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_up_offset=swiglu_up_offset,
    )
    return output, residual_out


__all__ = [
    "IQ2R_DEFAULT_TASK_ROWS",
    "IQ2R_DIRECT_ROUTE_MAX",
    "IQ2R_FUSED_SORT_MAX_TOKENS",
    "IQ2R_MAX_EXPERTS",
    "IQ2R_MAX_ROUTES",
    "IQ2RMoeWorkspace",
    "iq2r_fused_moe",
    "iq2r_fused_moe_add_rmsnorm",
    "iq2r_fused_moe_out",
]
