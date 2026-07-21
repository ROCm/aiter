# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused heterogeneous MoE (FHMoE) MXFP4/FP8 dispatch."""

import functools
from dataclasses import replace
from typing import Optional

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.moe_common import GateMode


def _validate_fhmoe_contract(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    shared_w1: Optional[torch.Tensor],
    shared_w2: Optional[torch.Tensor],
    shared_w1_scale: Optional[torch.Tensor],
    shared_w2_scale: Optional[torch.Tensor],
    shared_expert_id: int,
    activation: ActivationType,
    quant_type: QuantType,
    gate_mode: GateMode,
    expert_mask: Optional[torch.Tensor],
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
) -> None:
    shared_args = (shared_w1, shared_w2, shared_w1_scale, shared_w2_scale)
    has_shared_expert = all(arg is not None for arg in shared_args)
    if any(arg is not None for arg in shared_args) and not has_shared_expert:
        raise ValueError(
            "shared_w1, shared_w2, shared_w1_scale, and shared_w2_scale "
            "must be provided together"
        )
    if not has_shared_expert:
        raise ValueError(
            "shared_expert_id requires shared_w1, shared_w2, and their scales"
        )

    E = w1.shape[0]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2] * (model_dim // w1.shape[-1])
    if get_gfx() != "gfx950":
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 experts currently require gfx950"
        )
    if quant_type != QuantType.per_1x32:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 experts require per_1x32 quantization"
        )
    if activation != ActivationType.Silu:
        raise ValueError("Heterogeneous MXFP4/FP8 experts currently require SiLU")
    if gate_mode not in (GateMode.INTERLEAVE, GateMode.SEPARATED):
        raise ValueError(
            "Heterogeneous MXFP4/FP8 experts require interleaved or "
            "separated gate/up weights"
        )
    if expert_mask is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 experts do not yet support expert masks"
        )
    if bias1 is not None or bias2 is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 experts do not support expert biases"
        )
    if shared_expert_id != E - 1:
        raise ValueError(
            "The heterogeneous FlyDSL path requires a dummy final routed "
            f"weight row and shared_expert_id == E - 1; got {shared_expert_id=} "
            f"and E={E}"
        )
    if w1.dtype != dtypes.fp4x2 or w2.dtype != dtypes.fp4x2:
        raise ValueError("Heterogeneous routed weights must use MXFP4")
    if w1.shape[1] != 2 * inter_dim:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 experts require gate and up projections"
        )
    assert shared_w1 is not None and shared_w2 is not None
    assert shared_w1_scale is not None and shared_w2_scale is not None
    if shared_w1.dtype != dtypes.fp8 or shared_w2.dtype != dtypes.fp8:
        raise ValueError("Heterogeneous shared weights must use FP8 E4M3")
    if shared_w1.shape != (1, w1.shape[1], model_dim):
        raise ValueError(
            f"Expected shared_w1 shape {(1, w1.shape[1], model_dim)}, "
            f"got {tuple(shared_w1.shape)}"
        )
    if shared_w2.shape != (1, model_dim, inter_dim):
        raise ValueError(
            f"Expected shared_w2 shape {(1, model_dim, inter_dim)}, "
            f"got {tuple(shared_w2.shape)}"
        )

    scale_dtypes = (dtypes.fp8_e8m0, torch.uint8)
    scale_tensors = (w1_scale, w2_scale, shared_w1_scale, shared_w2_scale)
    if any(scale is None or scale.dtype not in scale_dtypes for scale in scale_tensors):
        raise ValueError(
            "Heterogeneous routed/shared scales must use FP8 E8M0 or raw "
            "uint8 E8M0 storage"
        )

    stage1_scale_shape = (
        ((2 * inter_dim + 255) // 256) * 256,
        ((model_dim // 32 + 7) // 8) * 8,
    )
    stage2_scale_k = ((inter_dim + 255) // 256) * 256
    stage2_scale_shape = (
        ((model_dim + 255) // 256) * 256,
        ((stage2_scale_k // 32 + 7) // 8) * 8,
    )
    if tuple(shared_w1_scale.shape) != stage1_scale_shape:
        raise ValueError(
            f"Expected preshuffled shared_w1_scale shape {stage1_scale_shape}, "
            f"got {tuple(shared_w1_scale.shape)}"
        )
    if tuple(shared_w2_scale.shape) != stage2_scale_shape:
        raise ValueError(
            f"Expected preshuffled shared_w2_scale shape {stage2_scale_shape}, "
            f"got {tuple(shared_w2_scale.shape)}"
        )

    assert w1_scale is not None and w2_scale is not None
    expected_w1_scale_numel = E * 2 * inter_dim * (model_dim // 32)
    expected_w2_scale_numel = E * model_dim * (stage2_scale_k // 32)
    if w1_scale.numel() != expected_w1_scale_numel:
        raise ValueError(
            "Expected preshuffled routed w1_scale to contain "
            f"{expected_w1_scale_numel} elements, got {w1_scale.numel()}"
        )
    if w2_scale.numel() != expected_w2_scale_numel:
        raise ValueError(
            "Expected preshuffled routed w2_scale to contain "
            f"{expected_w2_scale_numel} elements, got {w2_scale.numel()}"
        )

    tensors = (
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
    )
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError(
            "Heterogeneous routed/shared weights and scales must be on the "
            "hidden-state device"
        )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "Heterogeneous routed/shared weights and scales must be contiguous"
        )


def _flydsl_fhmoe_stage1_wrapper(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    activation=ActivationType.Silu,
    w1_scale=None,
    a1_scale=None,
    sorted_weights=None,
    out_scale=None,
    out_scale_sorted=None,
    bias1=None,
    topk_ids=None,
    swiglu_limit: Optional[float] = None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    shared_w1=None,
    shared_w1_scale=None,
    shared_expert_id: int = -1,
    **_kwargs,
):
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage1

    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    act = "swiglu" if activation == ActivationType.Swiglu else "silu"
    return flydsl_fhmoe_stage1(
        a=hidden_states,
        w1=w1,
        shared_w1=shared_w1,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        act=act,
        w1_scale=w1_scale,
        shared_w1_scale=shared_w1_scale,
        a1_scale=a1_scale,
        sorted_weights=sorted_weights,
        use_async_copy=True,
        k_batch=parsed.get("k_batch", 1),
        waves_per_eu=parsed.get("waves_per_eu", 3),
        b_nt=parsed.get("b_nt", 2),
        gate_mode=parsed.get("gate_mode", "separated"),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        bias=bias1,
        topk_ids=topk_ids,
        a_scale_one=parsed.get("a_scale_one", False),
        swiglu_limit=swiglu_limit,
        k_wave=parsed.get("k_wave", 1),
        shared_expert_id=shared_expert_id,
    )


def _flydsl_fhmoe_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    bias2=None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    expert_mask=None,
    topk_ids=None,
    shared_w2=None,
    shared_w2_scale=None,
    shared_expert_id: int = -1,
    **_kwargs,
):
    from aiter.ops.flydsl import moe_kernels
    from aiter.ops.flydsl.fhmoe import flydsl_fhmoe_stage2

    parsed = moe_kernels.get_flydsl_kernel_params(kernelName)
    if parsed is None:
        raise ValueError(f"Invalid FlyDSL kernel name: {kernelName}")
    return flydsl_fhmoe_stage2(
        inter_states=inter_states,
        w2=w2,
        shared_w2=shared_w2,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        topk=topk,
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        out_dtype=parsed["out_dtype"],
        mode=parsed.get("mode", "atomic"),
        w2_scale=w2_scale,
        shared_w2_scale=shared_w2_scale,
        a2_scale=a2_scale,
        sorted_weights=sorted_weights,
        sort_block_m=parsed.get("sort_block_m", 0),
        waves_per_eu=parsed.get("waves_per_eu", None),
        use_async_copy=parsed.get("use_async_copy", False),
        cu_num_mul=parsed.get("cu_num_mul", 1),
        b_nt=parsed.get("b_nt", 0),
        persist=parsed.get("persist", None),
        inter_dim_pad=inter_dim_pad,
        model_dim_pad=model_dim_pad,
        bias=bias2,
        expert_mask=expert_mask,
        topk_ids=topk_ids,
        shared_expert_id=shared_expert_id,
    )


_flydsl_fhmoe_stage2_wrapper._is_flydsl_stage2 = True


def _use_fhmoe_wrappers(metadata):
    from aiter.fused_moe import _flydsl_stage1_wrapper, _flydsl_stage2_wrapper

    stage1_func = getattr(metadata.stage1, "func", metadata.stage1)
    stage2_func = getattr(metadata.stage2, "func", metadata.stage2)
    if (
        metadata.run_1stage
        or stage1_func is not _flydsl_stage1_wrapper
        or stage2_func is not _flydsl_stage2_wrapper
    ):
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 experts require the two-stage FlyDSL path"
        )
    stage1 = functools.partial(
        _flydsl_fhmoe_stage1_wrapper,
        *metadata.stage1.args,
        **metadata.stage1.keywords,
    )
    stage2 = functools.partial(
        _flydsl_fhmoe_stage2_wrapper,
        *metadata.stage2.args,
        **metadata.stage2.keywords,
    )
    return replace(metadata, stage1=stage1, stage2=stage2)


def fhmoe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: Optional[torch.Tensor] = None,
    shared_w2: Optional[torch.Tensor] = None,
    shared_w1_scale: Optional[torch.Tensor] = None,
    shared_w2_scale: Optional[torch.Tensor] = None,
    shared_expert_id: int = -1,
) -> torch.Tensor:
    del (
        w1,
        topk_weight,
        expert_mask,
        activation,
        quant_type,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        block_size_M,
        num_local_tokens,
        moe_sorting_dispatch_policy,
        hidden_pad,
        intermediate_pad,
        bias1,
        bias2,
        swiglu_limit,
        gate_mode,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shared_expert_id,
    )
    output_dtype = hidden_states.dtype if dtype is None else dtype
    return torch.empty(
        (topk_ids.shape[0], w2.shape[1]),
        dtype=output_dtype,
        device=topk_ids.device,
    )


@torch_compile_guard(gen_fake=fhmoe_fake)
def fhmoe_(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_size_M: int = -1,
    num_local_tokens: Optional[torch.Tensor] = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: Optional[torch.dtype] = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: Optional[torch.Tensor] = None,
    bias2: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: Optional[torch.Tensor] = None,
    shared_w2: Optional[torch.Tensor] = None,
    shared_w1_scale: Optional[torch.Tensor] = None,
    shared_w2_scale: Optional[torch.Tensor] = None,
    shared_expert_id: int = -1,
) -> torch.Tensor:
    from aiter.fused_moe import _fused_moe_impl

    activation_enum = ActivationType(activation)
    quant_type_enum = QuantType(quant_type)
    gate_mode_enum = GateMode(gate_mode)
    _validate_fhmoe_contract(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w2,
        shared_w1_scale,
        shared_w2_scale,
        shared_expert_id,
        activation_enum,
        quant_type_enum,
        gate_mode_enum,
        expert_mask,
        bias1,
        bias2,
    )
    q_dtype_a = dtypes.fp8 if gate_mode_enum == GateMode.INTERLEAVE else dtypes.fp4x2
    return _fused_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        expert_mask=expert_mask,
        activation=activation,
        quant_type=quant_type,
        doweight_stage1=doweight_stage1,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_size_M=block_size_M,
        num_local_tokens=num_local_tokens,
        moe_sorting_dispatch_policy=moe_sorting_dispatch_policy,
        dtype=dtype,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=bias1,
        bias2=bias2,
        swiglu_limit=swiglu_limit,
        gate_mode=gate_mode,
        _q_dtype_a=q_dtype_a,
        _metadata_transform=_use_fhmoe_wrappers,
        _stage1_extra_args={
            "shared_w1": shared_w1,
            "shared_w1_scale": shared_w1_scale,
            "shared_expert_id": shared_expert_id,
            "swiglu_limit": swiglu_limit,
        },
        _stage2_extra_args={
            "shared_w2": shared_w2,
            "shared_w2_scale": shared_w2_scale,
            "shared_expert_id": shared_expert_id,
        },
    )


def _fhmoe(**kwargs) -> torch.Tensor:
    """Call the heterogeneous custom op using the public wrapper arguments."""
    kwargs["activation"] = kwargs["activation"].value
    kwargs["quant_type"] = kwargs["quant_type"].value
    block_size_M = kwargs.get("block_size_M")
    kwargs["block_size_M"] = -1 if not block_size_M else block_size_M
    return fhmoe_(**kwargs)
