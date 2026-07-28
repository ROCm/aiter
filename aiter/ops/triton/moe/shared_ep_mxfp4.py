# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Safe gfx950 MXFP4 compute adapter for SharedEP.

SharedEP route IDs are canonical global route slots.  This adapter keeps that
identity throughout both projections:

* W13 reads owner row ``route_id // top_k`` and writes the fused
  ``silu(gate) * up`` result to row ``route_id``.
* W2 reads intermediate row ``route_id``, applies the canonical (unsorted)
  route weight, and writes row ``route_id`` directly to the owner route slot.

Only canonical ``[expert, output, packed_k]`` MXFP4 weights and canonical
``[expert, output, k // 32]`` E8M0 scales are accepted.  Preshuffled or
swizzled layouts intentionally fail closed.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ContextManager, Literal, Mapping, Optional, Union

import torch


SharedEPMXFP4Stage = Literal["w13", "w2"]


class SharedEPMXFP4WeightLayout(str, Enum):
    """Weight layouts understood by the SharedEP adapter."""

    CANONICAL = "canonical"


class SharedEPMXFP4ScaleLayout(str, Enum):
    """E8M0 scale layouts understood by the SharedEP adapter."""

    CANONICAL = "canonical"


@dataclass(frozen=True)
class SharedEPMXFP4Profile:
    """Shape and routing metadata passed to config/profile hooks."""

    stage: SharedEPMXFP4Stage
    num_owner_rows: int
    route_capacity: int
    num_active_routes: Optional[int]
    num_padded_routes: int
    route_block_size: int
    num_experts: int
    input_dim: int
    output_dim: int
    top_k: int


SharedEPMXFP4ConfigHook = Callable[[SharedEPMXFP4Profile], Optional[Mapping[str, Any]]]
SharedEPMXFP4ProfileHook = Callable[
    [SharedEPMXFP4Profile, Mapping[str, Any]], Optional[ContextManager[Any]]
]

_FP4_PACKED_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_INTEGER_DTYPES = (torch.int32, torch.int64)
_ROUTE_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_REQUIRED_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
)


def _normalize_stage(stage: str) -> SharedEPMXFP4Stage:
    normalized = stage.lower()
    if normalized not in ("w13", "w2"):
        raise ValueError(f"stage must be 'w13' or 'w2', got {stage!r}")
    return normalized  # type: ignore[return-value]


def _normalize_layout(
    layout: Union[str, SharedEPMXFP4WeightLayout],
) -> SharedEPMXFP4WeightLayout:
    try:
        normalized = SharedEPMXFP4WeightLayout(layout)
    except ValueError as exc:
        raise ValueError(
            "SharedEP MXFP4 only supports canonical unswizzled weights; "
            f"got weight_layout={layout!r}"
        ) from exc
    if normalized is not SharedEPMXFP4WeightLayout.CANONICAL:
        raise ValueError("SharedEP MXFP4 only supports canonical unswizzled weights")
    return normalized


def _normalize_scale_layout(
    layout: Union[str, SharedEPMXFP4ScaleLayout],
) -> SharedEPMXFP4ScaleLayout:
    try:
        normalized = SharedEPMXFP4ScaleLayout(layout)
    except ValueError as exc:
        raise ValueError(
            "SharedEP MXFP4 only supports canonical unswizzled E8M0 scales; "
            f"got scale_layout={layout!r}"
        ) from exc
    if normalized is not SharedEPMXFP4ScaleLayout.CANONICAL:
        raise ValueError(
            "SharedEP MXFP4 only supports canonical unswizzled E8M0 scales"
        )
    return normalized


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_tensor(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _owner_rows_view(
    activations: torch.Tensor, stage: SharedEPMXFP4Stage
) -> torch.Tensor:
    _require_tensor(activations, "activations")
    if activations.ndim < 2:
        raise ValueError(
            f"{stage} activations must have at least 2 dimensions, got "
            f"shape={tuple(activations.shape)}"
        )
    if stage == "w2" and activations.ndim != 2:
        raise ValueError(
            "W2 intermediate storage must be the 2D route-indexed tensor "
            f"[route_capacity, K], got shape={tuple(activations.shape)}"
        )
    if activations.shape[-1] <= 0:
        raise ValueError("activations must have a non-empty feature dimension")
    rows = activations.numel() // activations.shape[-1]
    try:
        view = activations.view(rows, activations.shape[-1])
    except RuntimeError as exc:
        raise ValueError(
            "rank-major activations must be viewable as [owner_rows, K] "
            "without a copy"
        ) from exc
    if view.stride(1) != 1 or view.stride(0) < view.shape[1]:
        raise ValueError(
            "activations must have contiguous K and non-overlapping row-major rows; "
            f"got stride={view.stride()}"
        )
    return view


def _route_weights_view(route_weights: torch.Tensor, top_k: int) -> torch.Tensor:
    _require_tensor(route_weights, "route_weights")
    if route_weights.ndim < 2 or route_weights.shape[-1] != top_k:
        raise ValueError(
            "route_weights must use canonical rank-major owner layout "
            f"[..., top_k={top_k}], got shape={tuple(route_weights.shape)}"
        )
    owner_rows = route_weights.numel() // top_k
    try:
        view = route_weights.view(owner_rows, top_k)
    except RuntimeError as exc:
        raise ValueError(
            "route_weights must be viewable as [owner_rows, top_k] without a copy"
        ) from exc
    if not view.is_contiguous():
        raise ValueError("route_weights must be contiguous in canonical route order")
    return view


def _valid_fp4_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.uint8 or (
        _FP4_PACKED_DTYPE is not None and dtype == _FP4_PACKED_DTYPE
    )


def _valid_e8m0_dtype(dtype: torch.dtype) -> bool:
    return dtype == torch.uint8 or (_E8M0_DTYPE is not None and dtype == _E8M0_DTYPE)


def _as_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def shared_ep_mxfp4_route_rows(
    route_ids: torch.Tensor,
    *,
    stage: SharedEPMXFP4Stage,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_rows, output_rows)`` for canonical SharedEP route IDs.

    This helper is device agnostic and is also the executable CPU contract for
    route indexing.  W13 maps its input through ``route_id // top_k``; W2 and
    both outputs use ``route_id`` unchanged.
    """

    normalized_stage = _normalize_stage(stage)
    _require_positive_int(top_k, "top_k")
    _require_tensor(route_ids, "route_ids")
    if route_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"route_ids must have int32/int64 dtype, got {route_ids.dtype}")
    if route_ids.numel() and bool((route_ids < 0).any().item()):
        raise ValueError("route_ids must be non-negative")
    input_rows = route_ids // top_k if normalized_stage == "w13" else route_ids
    return input_rows, route_ids


def validate_shared_ep_mxfp4_contract(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    stage: SharedEPMXFP4Stage,
    top_k: int,
    route_block_size: int,
    route_weights: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    weight_layout: Union[
        str, SharedEPMXFP4WeightLayout
    ] = SharedEPMXFP4WeightLayout.CANONICAL,
    scale_layout: Union[
        str, SharedEPMXFP4ScaleLayout
    ] = SharedEPMXFP4ScaleLayout.CANONICAL,
    check_route_values: bool = True,
    require_cuda: bool = False,
) -> SharedEPMXFP4Profile:
    """Validate the complete SharedEP MXFP4 tensor and route contract.

    ``check_route_values=True`` performs a synchronizing validation of route
    bounds, canonical padding, uniqueness, and local expert ranges.  Launch
    APIs enable it by default so malformed remote-write metadata fails before
    a kernel can access memory.
    """

    normalized_stage = _normalize_stage(stage)
    _normalize_layout(weight_layout)
    _normalize_scale_layout(scale_layout)
    top_k = _require_positive_int(top_k, "top_k")
    route_block_size = _require_positive_int(route_block_size, "route_block_size")
    if not _is_power_of_two(route_block_size):
        raise ValueError(
            f"route_block_size must be a power of two, got {route_block_size}"
        )

    activations_2d = _owner_rows_view(activations, normalized_stage)
    if activations_2d.dtype != torch.bfloat16:
        raise TypeError(
            "the functional SharedEP MXFP4 path requires BF16 activations, "
            f"got {activations_2d.dtype}"
        )

    _require_tensor(weight, "weight")
    _require_tensor(weight_scales, "weight_scales")
    if weight.ndim != 3:
        raise ValueError(
            "weight must have canonical shape [experts, output, packed_k], "
            f"got shape={tuple(weight.shape)}"
        )
    if not _valid_fp4_dtype(weight.dtype):
        raise TypeError(
            "weight must be packed OCP E2M1 (torch.uint8 or "
            f"torch.float4_e2m1fn_x2), got {weight.dtype}"
        )
    if not weight.is_contiguous():
        raise ValueError(
            "weight must be contiguous canonical [expert, output, packed_k]; "
            "shuffled/strided weights are unsupported"
        )
    if weight_scales.ndim != 3:
        raise ValueError(
            "weight_scales must have canonical shape [experts, output, k // 32], "
            f"got shape={tuple(weight_scales.shape)}"
        )
    if not _valid_e8m0_dtype(weight_scales.dtype):
        raise TypeError(
            "weight_scales must contain E8M0 bytes (torch.uint8 or "
            f"torch.float8_e8m0fnu), got {weight_scales.dtype}"
        )
    if not weight_scales.is_contiguous():
        raise ValueError(
            "weight_scales must be contiguous canonical E8M0 scales; "
            "swizzled scales are unsupported"
        )

    num_experts, kernel_output_dim, packed_k = weight.shape
    input_dim = activations_2d.shape[1]
    if num_experts <= 0 or kernel_output_dim <= 0:
        raise ValueError("weight expert and output dimensions must be non-empty")
    if input_dim % 32 != 0:
        raise ValueError(
            f"input K={input_dim} must be divisible by the MXFP4 group size 32"
        )
    if packed_k != input_dim // 2:
        raise ValueError(
            f"weight packed K must be K/2={input_dim // 2}, got {packed_k}"
        )
    expected_scale_shape = (num_experts, kernel_output_dim, input_dim // 32)
    if tuple(weight_scales.shape) != expected_scale_shape:
        raise ValueError(
            "weight_scales must provide one E8M0 byte per 32 logical K values: "
            f"expected {expected_scale_shape}, got {tuple(weight_scales.shape)}"
        )
    if normalized_stage == "w13":
        if kernel_output_dim % 2:
            raise ValueError(
                f"W13 packed gate/up output dimension must be even, got {kernel_output_dim}"
            )
        output_dim = kernel_output_dim // 2
        num_owner_rows = activations_2d.shape[0]
        route_capacity = num_owner_rows * top_k
        if route_weights is not None:
            raise ValueError("W13 does not consume route weights")
    else:
        output_dim = kernel_output_dim
        route_capacity = activations_2d.shape[0]
        if route_capacity % top_k:
            raise ValueError(
                f"W2 route capacity {route_capacity} must be divisible by top_k={top_k}"
            )
        num_owner_rows = route_capacity // top_k
        if route_weights is None:
            raise ValueError("W2 requires canonical unsorted route_weights")
        route_weights_2d = _route_weights_view(route_weights, top_k)
        if tuple(route_weights_2d.shape) != (num_owner_rows, top_k):
            raise ValueError(
                "route_weights capacity does not match W2 intermediate rows: "
                f"expected {(num_owner_rows, top_k)}, "
                f"got {tuple(route_weights_2d.shape)}"
            )
        if route_weights_2d.dtype not in _ROUTE_WEIGHT_DTYPES:
            raise TypeError(
                "route_weights must be BF16/FP16/FP32, " f"got {route_weights_2d.dtype}"
            )

    if route_capacity <= 0:
        raise ValueError("route capacity must be positive")

    _require_tensor(sorted_route_ids, "sorted_route_ids")
    _require_tensor(expert_ids, "expert_ids")
    _require_tensor(num_tokens_post_padded, "num_tokens_post_padded")
    if sorted_route_ids.ndim != 1 or sorted_route_ids.dtype != torch.int32:
        raise TypeError(
            "sorted_route_ids must be a 1D int32 tensor of canonical route IDs"
        )
    if sorted_route_ids.stride(0) != 1:
        raise ValueError("sorted_route_ids must be contiguous")
    if expert_ids.ndim != 1 or expert_ids.dtype != torch.int32:
        raise TypeError("expert_ids must be a 1D int32 tensor of local expert IDs")
    if expert_ids.stride(0) != 1:
        raise ValueError("expert_ids must be contiguous")
    if (
        num_tokens_post_padded.numel() != 1
        or num_tokens_post_padded.dtype != torch.int32
    ):
        raise TypeError("num_tokens_post_padded must contain one int32 value")

    tensors = [
        activations,
        weight,
        weight_scales,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
    ]
    if route_weights is not None:
        tensors.append(route_weights)
    if out is not None:
        _require_tensor(out, "out")
        tensors.append(out)
    expected_device = activations.device
    for tensor in tensors:
        if tensor.device != expected_device:
            raise ValueError(
                "all SharedEP MXFP4 tensors must be on one device; "
                f"activations are on {expected_device}, got {tensor.device}"
            )
        if require_cuda and not tensor.is_cuda:
            raise ValueError("SharedEP MXFP4 launch tensors must be on a ROCm device")

    if out is not None:
        if out.ndim != 2 or out.shape[1] != output_dim:
            raise ValueError(
                f"out must have shape [route_capacity, {output_dim}], "
                f"got {tuple(out.shape)}"
            )
        if out.shape[0] < route_capacity:
            raise ValueError(
                f"out has {out.shape[0]} rows but route capacity is {route_capacity}"
            )
        if out.dtype != torch.bfloat16:
            raise TypeError(f"out must be BF16, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError(
                "out must be contiguous route-major storage for direct route writes"
            )

    # Avoid a device-to-host read during HIP Graph capture. The launch grid is
    # allowed to cover the static route buffer; the kernel reads the device
    # scalar and returns for blocks beyond the realized padded route count.
    num_padded_routes = (
        int(num_tokens_post_padded.item())
        if check_route_values
        else sorted_route_ids.numel()
    )
    num_active_routes: Optional[int] = None
    if check_route_values:
        if num_padded_routes < 0 or num_padded_routes > sorted_route_ids.numel():
            raise ValueError(
                "num_tokens_post_padded must be within sorted_route_ids capacity; "
                f"got {num_padded_routes} for {sorted_route_ids.numel()} entries"
            )
        if num_padded_routes % route_block_size:
            raise ValueError(
                f"num_tokens_post_padded={num_padded_routes} is not divisible by "
                f"route_block_size={route_block_size}"
            )
        num_route_blocks = num_padded_routes // route_block_size
        if expert_ids.numel() < num_route_blocks:
            raise ValueError(
                f"expert_ids needs at least {num_route_blocks} entries, "
                f"got {expert_ids.numel()}"
            )
        launched_routes = sorted_route_ids[:num_padded_routes]
        if launched_routes.numel():
            if bool((launched_routes < 0).any().item()):
                raise ValueError("launched route IDs must be non-negative")
            valid_mask = launched_routes < route_capacity
            padding = launched_routes[~valid_mask]
            if padding.numel() and bool((padding != route_capacity).any().item()):
                raise ValueError(
                    "padded route IDs must use the canonical route-capacity sentinel "
                    f"{route_capacity}"
                )
            valid_routes = launched_routes[valid_mask]
            num_active_routes = valid_routes.numel()
            if valid_routes.numel() != torch.unique(valid_routes).numel():
                raise ValueError(
                    "valid route IDs must be unique to guarantee deterministic writes"
                )
        else:
            num_active_routes = 0

        launched_experts = expert_ids[:num_route_blocks]
        if launched_experts.numel() and bool(
            ((launched_experts < 0) | (launched_experts >= num_experts)).any().item()
        ):
            raise ValueError(
                "expert_ids must contain local weight indices in "
                f"[0, {num_experts}); -1/non-local blocks are unsafe for SharedEP"
            )

    return SharedEPMXFP4Profile(
        stage=normalized_stage,
        num_owner_rows=num_owner_rows,
        route_capacity=route_capacity,
        num_active_routes=num_active_routes,
        num_padded_routes=num_padded_routes,
        route_block_size=route_block_size,
        num_experts=num_experts,
        input_dim=input_dim,
        output_dim=output_dim,
        top_k=top_k,
    )


def get_shared_ep_mxfp4_config(
    profile: SharedEPMXFP4Profile,
    *,
    config: Optional[Mapping[str, Any]] = None,
    config_hook: Optional[SharedEPMXFP4ConfigHook] = None,
) -> dict[str, Any]:
    """Resolve and validate a gfx950 kernel config for a SharedEP profile.

    An explicit ``config`` or per-profile ``config_hook`` can supply tuned
    values.  With neither, AITER's ``gfx950-MOE-MX_FP4.json`` selector is used.
    The selected ``BLOCK_SIZE_M`` must equal the block size used by route
    preparation.
    """

    if not isinstance(profile, SharedEPMXFP4Profile):
        raise TypeError("profile must be SharedEPMXFP4Profile")
    if config is not None and config_hook is not None:
        raise ValueError("pass either config or config_hook, not both")

    selected: Optional[Mapping[str, Any]] = config
    if config_hook is not None:
        selected = config_hook(profile)
    if selected is None:
        from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config

        selected = get_optimal_moe_config(
            torch.bfloat16,
            use_mxfp4=True,
            M=max(1, profile.num_owner_rows),
        )
    if selected is None:
        raise RuntimeError("no gfx950 MXFP4 MoE config is available")

    resolved = dict(selected)
    missing = [key for key in _REQUIRED_CONFIG_KEYS if key not in resolved]
    if missing:
        raise ValueError(f"SharedEP MXFP4 config is missing keys: {missing}")
    for key in _REQUIRED_CONFIG_KEYS:
        value = resolved[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"config[{key!r}] must be a positive integer")
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"):
        if not _is_power_of_two(resolved[key]):
            raise ValueError(f"config[{key!r}] must be a power of two")
    if resolved["BLOCK_SIZE_M"] != profile.route_block_size:
        raise ValueError(
            "route preparation block size must match config BLOCK_SIZE_M: "
            f"{profile.route_block_size} != {resolved['BLOCK_SIZE_M']}"
        )
    if resolved["BLOCK_SIZE_K"] % 32:
        raise ValueError("config BLOCK_SIZE_K must be divisible by 32")
    if profile.input_dim % resolved["BLOCK_SIZE_K"]:
        raise ValueError(
            f"functional BF16->MXFP4 path requires K={profile.input_dim} to be "
            f"divisible by BLOCK_SIZE_K={resolved['BLOCK_SIZE_K']}"
        )
    kernel_output_dim = (
        2 * profile.output_dim if profile.stage == "w13" else profile.output_dim
    )
    if kernel_output_dim % resolved["BLOCK_SIZE_N"]:
        raise ValueError(
            f"kernel output N={kernel_output_dim} must be divisible by "
            f"BLOCK_SIZE_N={resolved['BLOCK_SIZE_N']}"
        )
    if profile.stage == "w13" and resolved["BLOCK_SIZE_N"] % 2:
        raise ValueError("W13 config BLOCK_SIZE_N must be even")
    return resolved


def _require_gfx950() -> None:
    from aiter.ops.triton.utils._triton import arch_info

    arch = str(arch_info.get_arch()).split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"SharedEP MXFP4 requires gfx950, got {arch}")


def _profile_context(
    profile_hook: Optional[SharedEPMXFP4ProfileHook],
    profile: SharedEPMXFP4Profile,
    config: Mapping[str, Any],
) -> ContextManager[Any]:
    if profile_hook is None:
        return nullcontext()
    context = profile_hook(profile, MappingProxyType(dict(config)))
    if context is None:
        return nullcontext()
    if not hasattr(context, "__enter__") or not hasattr(context, "__exit__"):
        raise TypeError("profile_hook must return a context manager or None")
    return context


def _launch_shared_ep_mxfp4(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    route_weights: Optional[torch.Tensor],
    out: torch.Tensor,
    profile: SharedEPMXFP4Profile,
    config: Mapping[str, Any],
    swiglu_limit: Optional[float] = None,
) -> None:
    from aiter.ops.triton.utils.types import torch_to_triton_dtype

    activations_2d = _owner_rows_view(activations, profile.stage)
    weight_u8 = _as_uint8(weight)
    weight_scales_u8 = _as_uint8(weight_scales)
    unit_a_scale = torch.ones((1,), dtype=torch.float32, device=activations.device)
    unit_weight_scale = torch.ones(
        (profile.num_experts,), dtype=torch.float32, device=activations.device
    )

    if profile.stage == "w13":
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import (
            fused_moe_mxfp4_silu,
        )

        # Neither legacy argument is read for W13.  The route-capacity bound is
        # passed explicitly, and routed-weight multiplication is disabled.
        route_metadata_proxy = out[:, :1]
        fused_moe_mxfp4_silu(
            activations_2d,
            weight_u8,
            out,
            unit_a_scale,
            unit_weight_scale,
            None,
            weight_scales_u8,
            route_metadata_proxy,
            route_metadata_proxy,
            sorted_route_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            profile.top_k,
            False,
            False,
            dict(config),
            torch_to_triton_dtype[torch.bfloat16],
            input_row_divisor=profile.top_k,
            num_valid_routes=profile.route_capacity,
            quantize_a_to_mxfp4=True,
            swiglu_limit=swiglu_limit,
        )
    else:
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

        assert route_weights is not None
        route_weights_2d = _route_weights_view(route_weights, profile.top_k)
        fused_moe_mxfp4(
            activations_2d,
            weight_u8,
            out.unsqueeze(1),
            unit_a_scale,
            unit_weight_scale,
            None,
            weight_scales_u8,
            route_weights_2d,
            route_weights_2d,
            sorted_route_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            profile.top_k,
            False,
            False,
            dict(config),
            torch_to_triton_dtype[torch.bfloat16],
            input_row_divisor=1,
            num_valid_routes=profile.route_capacity,
            quantize_a_to_mxfp4=True,
        )


def shared_ep_mxfp4_w13(
    activations: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    route_block_size: int,
    out: Optional[torch.Tensor] = None,
    weight_layout: Union[
        str, SharedEPMXFP4WeightLayout
    ] = SharedEPMXFP4WeightLayout.CANONICAL,
    scale_layout: Union[
        str, SharedEPMXFP4ScaleLayout
    ] = SharedEPMXFP4ScaleLayout.CANONICAL,
    config: Optional[Mapping[str, Any]] = None,
    config_hook: Optional[SharedEPMXFP4ConfigHook] = None,
    profile_hook: Optional[SharedEPMXFP4ProfileHook] = None,
    check_route_values: bool = True,
    swiglu_limit: Optional[float] = None,
) -> torch.Tensor:
    """Run SharedEP W13 and return route-indexed BF16 intermediates.

    ``activations`` may have rank-major leading dimensions and must be viewable
    as ``[global_owner_rows, hidden]`` without a copy.  Canonical route
    ``r`` reads activation row ``r // top_k``.  The canonical W13 weight stores
    gate channels followed by up channels in its output dimension.  The result
    is ``[global_owner_rows * top_k, intermediate]`` and row ``r`` contains
    ``silu(gate_r) * up_r``.

    Only valid local routes are overwritten when ``out`` is supplied; other
    route slots are left untouched so disjoint SharedEP writers can share the
    owner buffer.
    """

    normalized_stage: SharedEPMXFP4Stage = "w13"
    _normalize_layout(weight_layout)
    _normalize_scale_layout(scale_layout)
    activations_2d = _owner_rows_view(activations, normalized_stage)
    _require_tensor(weight, "weight")
    if weight.ndim != 3 or weight.shape[1] % 2:
        raise ValueError("W13 weight must have shape [experts, 2 * intermediate, K/2]")
    route_capacity = activations_2d.shape[0] * _require_positive_int(top_k, "top_k")
    output_dim = weight.shape[1] // 2
    if out is None:
        out = torch.zeros(
            (route_capacity, output_dim),
            dtype=torch.bfloat16,
            device=activations.device,
        )
    profile = validate_shared_ep_mxfp4_contract(
        activations,
        weight,
        weight_scales,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
        stage=normalized_stage,
        top_k=top_k,
        route_block_size=route_block_size,
        out=out,
        weight_layout=weight_layout,
        scale_layout=scale_layout,
        check_route_values=check_route_values,
        require_cuda=True,
    )
    _require_gfx950()
    resolved_config = get_shared_ep_mxfp4_config(
        profile, config=config, config_hook=config_hook
    )
    with _profile_context(profile_hook, profile, resolved_config):
        if profile.num_padded_routes:
            _launch_shared_ep_mxfp4(
                activations,
                weight,
                weight_scales,
                sorted_route_ids,
                expert_ids,
                num_tokens_post_padded,
                None,
                out,
                profile,
                resolved_config,
                swiglu_limit,
            )
    return out


def shared_ep_mxfp4_w2(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    route_weights: torch.Tensor,
    sorted_route_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    route_block_size: int,
    out: Optional[torch.Tensor] = None,
    weight_layout: Union[
        str, SharedEPMXFP4WeightLayout
    ] = SharedEPMXFP4WeightLayout.CANONICAL,
    scale_layout: Union[
        str, SharedEPMXFP4ScaleLayout
    ] = SharedEPMXFP4ScaleLayout.CANONICAL,
    config: Optional[Mapping[str, Any]] = None,
    config_hook: Optional[SharedEPMXFP4ConfigHook] = None,
    profile_hook: Optional[SharedEPMXFP4ProfileHook] = None,
    check_route_values: bool = True,
) -> torch.Tensor:
    """Run SharedEP W2 directly into canonical global owner route slots.

    ``intermediate`` is the W13 route-indexed tensor.  ``route_weights`` must
    be the canonical unsorted rank-major owner tensor ``[..., top_k]``; it is
    indexed by the global route ID, not by sorted position.  Route ``r`` reads
    intermediate row ``r``, multiplies by ``route_weights.view(-1)[r]``, and
    writes output row ``r``.  The returned tensor is an uncombined route
    payload ``[route_capacity, hidden]``; no local combine buffer is produced.
    """

    normalized_stage: SharedEPMXFP4Stage = "w2"
    _normalize_layout(weight_layout)
    _normalize_scale_layout(scale_layout)
    intermediate_2d = _owner_rows_view(intermediate, normalized_stage)
    _require_tensor(weight, "weight")
    if weight.ndim != 3:
        raise ValueError("W2 weight must have shape [experts, hidden, K/2]")
    if out is None:
        out = torch.zeros(
            (intermediate_2d.shape[0], weight.shape[1]),
            dtype=torch.bfloat16,
            device=intermediate.device,
        )
    profile = validate_shared_ep_mxfp4_contract(
        intermediate,
        weight,
        weight_scales,
        sorted_route_ids,
        expert_ids,
        num_tokens_post_padded,
        stage=normalized_stage,
        top_k=top_k,
        route_block_size=route_block_size,
        route_weights=route_weights,
        out=out,
        weight_layout=weight_layout,
        scale_layout=scale_layout,
        check_route_values=check_route_values,
        require_cuda=True,
    )
    _require_gfx950()
    resolved_config = get_shared_ep_mxfp4_config(
        profile, config=config, config_hook=config_hook
    )
    with _profile_context(profile_hook, profile, resolved_config):
        if profile.num_padded_routes:
            _launch_shared_ep_mxfp4(
                intermediate,
                weight,
                weight_scales,
                sorted_route_ids,
                expert_ids,
                num_tokens_post_padded,
                route_weights,
                out,
                profile,
                resolved_config,
            )
    return out


__all__ = [
    "SharedEPMXFP4ConfigHook",
    "SharedEPMXFP4Profile",
    "SharedEPMXFP4ProfileHook",
    "SharedEPMXFP4ScaleLayout",
    "SharedEPMXFP4Stage",
    "SharedEPMXFP4WeightLayout",
    "get_shared_ep_mxfp4_config",
    "shared_ep_mxfp4_route_rows",
    "shared_ep_mxfp4_w13",
    "shared_ep_mxfp4_w2",
    "validate_shared_ep_mxfp4_contract",
]
