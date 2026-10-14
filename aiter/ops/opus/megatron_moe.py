# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Megatron local-expert adapter for the Opus BF16 MoE backward.

Megatron's token dispatcher has already grouped local routes by expert before
``TEGroupedMLP`` runs.  Opus uses the same expert-major order, but pads each
expert to 32 routes.  This module builds that metadata on the GPU and attaches
the native Opus backward to tensors produced by the existing Megatron forward.
Expert-parallel communication remains owned by Megatron.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...jit.core import compile_ops
from .moe_backward import (
    OpusMoeFixedMetadata,
    _select_saved_a_scaled_down_kernel,
    opus_moe_backward,
)


def _gen_megatron_metadata_fake(
    tokens_per_expert: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    reverse_sorted: Tensor,
    expert_padded_offsets: Tensor,
) -> Tensor:
    return sorted_token_ids


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_megatron_moe_metadata",
    gen_fake=_gen_megatron_metadata_fake,
    develop=True,
)
def _opus_megatron_moe_metadata_raw(
    tokens_per_expert: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    reverse_sorted: Tensor,
    expert_padded_offsets: Tensor,
    route_count: int,
    block_m: int,
) -> Tensor: ...


def _gen_megatron_pad_rows_fake(
    compact: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    padded: Tensor,
) -> Tensor:
    return padded


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_megatron_moe_pad_rows",
    gen_fake=_gen_megatron_pad_rows_fake,
    develop=True,
)
def _opus_megatron_moe_pad_rows_raw(
    compact: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    padded: Tensor,
) -> Tensor: ...


def _gen_megatron_poison_backward_fake(
    num_valid_ids: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
    d_scores: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
) -> Tensor:
    return d_x


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_megatron_moe_poison_invalid_backward",
    gen_fake=_gen_megatron_poison_backward_fake,
    develop=True,
)
def _opus_megatron_moe_poison_invalid_backward_raw(
    num_valid_ids: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
    d_scores: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
) -> Tensor: ...


def _check_grouped_weights(w1: Tensor, w2: Tensor, local_experts: int) -> None:
    """Require TE's zero-copy ``single_grouped_weight`` representation."""

    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            "Opus requires contiguous grouped weights [E,2I,D] and [E,D,I]; "
            "enable Megatron --moe-single-grouped-weight"
        )
    if w1.shape[0] != local_experts or w2.shape[0] != local_experts:
        raise ValueError("weight expert dimension must match tokens_per_expert")
    if not w1.is_contiguous() or not w2.is_contiguous():
        raise ValueError("grouped weights must be contiguous")


def _resolve_main_grad(
    weight: Tensor,
    explicit: Tensor | None,
    name: str,
) -> Tensor | None:
    main_grad = explicit
    if main_grad is None:
        candidate = getattr(weight, "main_grad", None)
        main_grad = candidate if isinstance(candidate, Tensor) else None
    if main_grad is None:
        return None
    if main_grad.shape != weight.shape:
        raise ValueError(f"{name} must have the same shape as its grouped weight")
    if main_grad.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be BF16 or FP32")
    if main_grad.device != weight.device or not main_grad.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the weight device")
    if main_grad.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    return main_grad


def _main_grad_dummy(weight: Tensor) -> Tensor | None:
    """Mirror Megatron's fused-wgrad dummy needed by its DDP post-hook."""

    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    main_grad = getattr(weight, "main_grad", None)
    factory = main_grad if isinstance(main_grad, Tensor) else weight
    if getattr(weight, "zero_out_wgrad", False):
        return factory.new_zeros(weight.shape, dtype=weight.dtype)
    return factory.new_empty(weight.shape, dtype=weight.dtype)


def _can_cache_activated_scaled(inter_dim: int, metadata: OpusMoeFixedMetadata) -> bool:
    """Check whether K1 has a no-store instance for this route geometry."""

    try:
        _select_saved_a_scaled_down_kernel(
            inter_dim,
            metadata.sorted_token_ids.numel(),
            metadata.sorted_expert_ids,
            metadata.expert_padded_offsets,
            -1,
        )
    except ValueError:
        return False
    return True


def build_megatron_local_metadata(
    tokens_per_expert: Tensor,
    route_count: int,
    *,
    block_m: int = 32,
    synchronize_validation: bool = False,
) -> OpusMoeFixedMetadata:
    """Build K=1 Opus metadata for already expert-grouped Megatron routes.

    One HIP launch constructs compact and padded expert offsets plus the K=1
    route maps.  The allocation uses the Opus sorter's exact worst-case bound,
    so no count is copied to the CPU.
    """

    if tokens_per_expert.dtype != torch.int32 or tokens_per_expert.ndim != 1:
        raise TypeError("tokens_per_expert must be a rank-1 int32 tensor")
    if not tokens_per_expert.is_cuda or not tokens_per_expert.is_contiguous():
        raise ValueError("tokens_per_expert must be contiguous on a GPU")
    if tokens_per_expert.numel() < 1:
        raise ValueError("tokens_per_expert must contain at least one expert")
    if route_count < 0 or route_count >= (1 << 24):
        raise ValueError("route_count must fit the Opus 24-bit token field")
    if block_m != 32:
        raise ValueError("current gfx950 Opus backward requires block_m=32")

    num_experts = tokens_per_expert.numel()
    capacity = route_count + num_experts * block_m - 1
    max_blocks = (capacity + block_m - 1) // block_m
    options = {"dtype": torch.int32, "device": tokens_per_expert.device}
    sorted_token_ids = torch.empty(capacity, **options)
    sorted_expert_ids = torch.empty(max_blocks, **options)
    # [padded_count, validation_status, max_padded_expert_routes].  The third
    # value lets K5 select a skew-safe launch entirely on device.
    num_valid_ids = torch.empty(3, dtype=torch.int32, device=tokens_per_expert.device)
    reverse_sorted = torch.empty(route_count, **options)
    expert_padded_offsets = torch.empty(num_experts + 1, **options)
    _opus_megatron_moe_metadata_raw(
        tokens_per_expert,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        reverse_sorted,
        expert_padded_offsets,
        route_count,
        block_m,
    )
    if synchronize_validation and int(num_valid_ids[1].item()) != route_count:
        raise ValueError("tokens_per_expert must be nonnegative and sum to route_count")
    return OpusMoeFixedMetadata(
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        reverse_sorted,
        expert_padded_offsets,
        block_m,
    )


def pad_megatron_local_rows(compact: Tensor, metadata: OpusMoeFixedMetadata) -> Tensor:
    """Copy compact expert-major BF16 rows into Opus' padded route layout."""

    if compact.dtype != torch.bfloat16 or compact.ndim != 2:
        raise TypeError("compact expert rows must be a rank-2 BF16 tensor")
    if not compact.is_cuda or not compact.is_contiguous():
        raise ValueError("compact expert rows must be contiguous on a GPU")
    if compact.shape[1] % 8:
        raise ValueError("compact expert row width must be divisible by 8")
    padded = compact.new_empty((metadata.sorted_token_ids.numel(), compact.shape[1]))
    _opus_megatron_moe_pad_rows_raw(
        compact, metadata.sorted_token_ids, metadata.num_valid_ids, padded
    )
    return padded


class _OpusMegatronGroupedMlpFunction(torch.autograd.Function):
    """One autograd node around the optimized native K1--K5 full chain."""

    @staticmethod
    def forward(
        ctx,
        output: Tensor,
        padded_z: Tensor,
        local_hidden_states: Tensor,
        fc1_weight: Tensor,
        fc2_weight: Tensor,
        route_weights: Tensor,
        metadata: OpusMoeFixedMetadata,
        saved_a_scaled: Tensor | None,
        saved_x_sorted: Tensor | None,
        fc1_main_grad: Tensor | None,
        fc2_main_grad: Tensor | None,
        accumulate_into_main_grad: bool,
    ) -> Tensor:
        saved_tensors = [
            padded_z,
            local_hidden_states,
            fc1_weight,
            fc2_weight,
            route_weights,
        ]
        optional_tensors = (
            saved_a_scaled,
            saved_x_sorted,
            fc1_main_grad,
            fc2_main_grad,
        )
        ctx.optional_tensor_mask = tuple(
            tensor is not None for tensor in optional_tensors
        )
        saved_tensors.extend(
            tensor for tensor in optional_tensors if tensor is not None
        )
        ctx.save_for_backward(*saved_tensors)
        ctx.opus_metadata = metadata
        ctx.accumulate_into_main_grad = bool(accumulate_into_main_grad)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor | None):
        saved_tensors = ctx.saved_tensors
        (
            padded_z,
            local_hidden_states,
            fc1_weight,
            fc2_weight,
            route_weights,
        ) = saved_tensors[:5]
        optional_tensors: list[Tensor | None] = []
        saved_index = 5
        for present in ctx.optional_tensor_mask:
            optional_tensors.append(saved_tensors[saved_index] if present else None)
            saved_index += int(present)
        saved_a, saved_x, fc1_main_grad, fc2_main_grad = optional_tensors
        if grad_output is None:
            return (None,) * 12
        result = opus_moe_backward(
            grad_output.contiguous(),
            local_hidden_states,
            padded_z,
            fc1_weight,
            fc2_weight,
            route_weights,
            ctx.opus_metadata,
            saved_a_scaled=saved_a,
            saved_x_sorted=saved_x,
        )
        _opus_megatron_moe_poison_invalid_backward_raw(
            ctx.opus_metadata.num_valid_ids,
            result.d_x,
            result.d_w1,
            result.d_w2,
            result.d_scores,
            result.d_z_sorted,
            result.a_scaled,
        )
        if ctx.accumulate_into_main_grad:
            fc1_main_grad.add_(result.d_w1)
            fc2_main_grad.add_(result.d_w2)
            d_w1 = _main_grad_dummy(fc1_weight)
            d_w2 = _main_grad_dummy(fc2_weight)
        else:
            d_w1 = result.d_w1 if ctx.needs_input_grad[3] else None
            d_w2 = result.d_w2 if ctx.needs_input_grad[4] else None
        return (
            None,
            None,
            result.d_x if ctx.needs_input_grad[2] else None,
            d_w1,
            d_w2,
            result.d_scores if ctx.needs_input_grad[5] else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def opus_megatron_grouped_mlp_attach_backward(
    output: Tensor,
    fc1_pre_activation: Tensor,
    local_hidden_states: Tensor,
    fc1_weight: Tensor,
    fc2_weight: Tensor,
    permuted_probs: Tensor,
    tokens_per_expert: Tensor,
    *,
    activated_scaled: Tensor | None = None,
    save_sorted_input: bool = False,
    fc1_main_grad: Tensor | None = None,
    fc2_main_grad: Tensor | None = None,
    accumulate_into_main_grad: bool | None = None,
    synchronize_metadata_validation: bool = False,
    expert_tensor_parallel_size: int = 1,
    activation_recompute: bool = False,
) -> Tensor:
    """Replace TEGroupedMLP's autograd backward while retaining its forward.

    Inputs are the tensors at the Megatron ``TEGroupedMLP.forward`` boundary.
    ``fc1_weight`` and ``fc2_weight`` must be TE single-grouped weights with
    layouts ``[E_local,2I,D]`` and ``[E_local,D,I]``.  ``activated_scaled`` is
    Megatron's BF16 output of weighted SwiGLU, before FC2; passing it enables
    Opus' forward-cache path.  The returned tensor has the same value as
    ``output`` and carries the Opus backward graph.
    """

    tensors = (output, fc1_pre_activation, local_hidden_states, fc1_weight, fc2_weight)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("Opus Megatron adapter requires BF16 activations and weights")
    if (
        output.ndim != 2
        or local_hidden_states.ndim != 2
        or fc1_pre_activation.ndim != 2
    ):
        raise ValueError(
            "output, local_hidden_states and fc1_pre_activation must be rank 2"
        )
    route_count, model_dim = local_hidden_states.shape
    if output.shape != (route_count, model_dim):
        raise ValueError("output must have shape [routes,D]")
    if tokens_per_expert.device != local_hidden_states.device:
        raise ValueError("tokens_per_expert must be on the activation device")
    _check_grouped_weights(fc1_weight, fc2_weight, tokens_per_expert.numel())
    if fc1_weight.shape[2] != model_dim or fc2_weight.shape[1] != model_dim:
        raise ValueError("weight model dimensions do not match local_hidden_states")
    inter_dim = fc2_weight.shape[2]
    if fc1_weight.shape[1] != 2 * inter_dim:
        raise ValueError("expected fc1_weight=[E,2I,D] and fc2_weight=[E,D,I]")
    if fc1_pre_activation.shape != (route_count, 2 * inter_dim):
        raise ValueError("fc1_pre_activation must have shape [routes,2I]")

    if permuted_probs.ndim == 1:
        route_weights = permuted_probs.view(route_count, 1)
    elif permuted_probs.shape == (route_count, 1):
        route_weights = permuted_probs
    else:
        raise ValueError("permuted_probs must have shape [routes] or [routes,1]")
    if route_weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("permuted_probs must be BF16 or FP32")
    # Megatron leaves router probabilities in the activation dtype unless
    # --moe-router-dtype=fp32 is selected.  Opus consumes FP32 route weights;
    # keep the cast in the autograd graph so dProb is converted back to the
    # router's original dtype on the way out.
    route_weights = route_weights.float().contiguous()

    if expert_tensor_parallel_size != 1:
        raise NotImplementedError(
            "Opus Megatron backward does not yet implement expert tensor "
            "parallel gradient collectives"
        )
    if activation_recompute:
        raise NotImplementedError(
            "Opus Megatron backward currently saves FC1 pre-activations and "
            "cannot preserve Megatron activation recompute semantics"
        )

    resolved_fc1_main_grad = _resolve_main_grad(
        fc1_weight, fc1_main_grad, "fc1_main_grad"
    )
    resolved_fc2_main_grad = _resolve_main_grad(
        fc2_weight, fc2_main_grad, "fc2_main_grad"
    )
    has_fc1_main_grad = resolved_fc1_main_grad is not None
    has_fc2_main_grad = resolved_fc2_main_grad is not None
    if accumulate_into_main_grad is None:
        accumulate_into_main_grad = has_fc1_main_grad and has_fc2_main_grad
    if accumulate_into_main_grad and not (has_fc1_main_grad and has_fc2_main_grad):
        raise ValueError(
            "accumulate_into_main_grad requires fc1_main_grad and fc2_main_grad"
        )
    if not accumulate_into_main_grad and (
        fc1_main_grad is not None or fc2_main_grad is not None
    ):
        raise ValueError(
            "explicit main_grad buffers require accumulate_into_main_grad=True"
        )
    if activated_scaled is not None:
        if activated_scaled.shape != (route_count, inter_dim):
            raise ValueError("activated_scaled must have shape [routes,I]")
        if activated_scaled.dtype != torch.bfloat16:
            raise TypeError("activated_scaled must be BF16")
        if activated_scaled.device != local_hidden_states.device:
            raise ValueError("activated_scaled must be on the activation device")
        if not activated_scaled.is_contiguous():
            raise ValueError("activated_scaled must be contiguous")

    metadata = build_megatron_local_metadata(
        tokens_per_expert,
        route_count,
        synchronize_validation=synchronize_metadata_validation,
    )
    # Opus computes the complete local-expert backward itself.  Keep only the
    # tensor values needed by the native kernels and sever the original TE
    # activation graph so its saved tensors can be released before backward.
    padded_z = pad_megatron_local_rows(fc1_pre_activation.detach(), metadata)
    if activated_scaled is None:
        saved_a = None
    else:
        saved_a = (
            pad_megatron_local_rows(activated_scaled.detach(), metadata)
            if _can_cache_activated_scaled(inter_dim, metadata)
            else None
        )
    saved_x = (
        pad_megatron_local_rows(local_hidden_states.detach(), metadata)
        if save_sorted_input
        else None
    )
    return _OpusMegatronGroupedMlpFunction.apply(
        output.detach(),
        padded_z,
        local_hidden_states,
        fc1_weight,
        fc2_weight,
        route_weights,
        metadata,
        saved_a,
        saved_x,
        resolved_fc1_main_grad,
        resolved_fc2_main_grad,
        bool(accumulate_into_main_grad),
    )


__all__ = [
    "build_megatron_local_metadata",
    "opus_megatron_grouped_mlp_attach_backward",
    "pad_megatron_local_rows",
]
