# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused heterogeneous MoE (FHMoE) MXFP4/FP8 and MXFP8/FP8 dispatch.

The notation names routed/shared storage respectively. Shared ``FP8`` means
E4M3 payloads with per-32 E8M0 block scales, stored separately from routed
weights.

FHMoE requires gfx950, SiLU, per-1x32 quantization, contiguous preshuffled
weights/scales, all four shared tensors, and a dummy final routed row identified
by ``shared_expert_id == E - 1``. Expert bias and expert-mask dispatch are not
supported.

Dedicated tuned contracts:

* DSV4 I384: ``(H, I, E, topk) = (7168, 384, 385, 7)``, MXFP4 routed weights,
  interleaved gate/up layout, and clamped shared SiLU.
* HY4 MXFP8: ``(6144, 256, 257, 9)``, interleaved MXFP8 routed weights, and
  ``M <= 38836``. ``clamp_shared=False`` selects its unclamped-shared
  specialization; the public default ``True`` keeps the shared SiLU clamped.
"""

import functools
import os
from dataclasses import replace

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fhmoe_contract import (
    HY4_FHMOE_MAX_TOKENS,
    _is_hy4_mxfp8_fhmoe_contract,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.moe_common import GateMode


def _validate_fhmoe_contract(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor | None,
    w2_scale: torch.Tensor | None,
    shared_w1: torch.Tensor | None,
    shared_w2: torch.Tensor | None,
    shared_w1_scale: torch.Tensor | None,
    shared_w2_scale: torch.Tensor | None,
    shared_expert_id: int,
    activation: ActivationType,
    quant_type: QuantType,
    gate_mode: GateMode,
    expert_mask: torch.Tensor | None,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
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
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts currently require gfx950"
        )
    if quant_type != QuantType.per_1x32:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "per_1x32 quantization"
        )
    if activation != ActivationType.Silu:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts currently require SiLU"
        )
    if gate_mode not in (GateMode.INTERLEAVE, GateMode.SEPARATED):
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "interleaved or separated gate/up weights"
        )
    if expert_mask is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts do not yet "
            "support expert masks"
        )
    if bias1 is not None or bias2 is not None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts do not support expert biases"
        )
    if shared_expert_id != E - 1:
        raise ValueError(
            "The heterogeneous FlyDSL path requires a dummy final routed "
            f"weight row and shared_expert_id == E - 1; got {shared_expert_id=} "
            f"and E={E}"
        )
    if w1.dtype != w2.dtype:
        raise ValueError(
            "Heterogeneous routed W1/W2 must use the same dtype, got "
            f"{w1.dtype} and {w2.dtype}"
        )
    if w1.dtype not in (dtypes.fp4x2, dtypes.fp8):
        raise ValueError("Heterogeneous routed weights must use MXFP4 or MXFP8")
    if w1.dtype == dtypes.fp8 and gate_mode != GateMode.INTERLEAVE:
        raise ValueError(
            "Heterogeneous MXFP8 routed weights require interleaved gate/up layout"
        )
    if w1.shape[1] != 2 * inter_dim:
        raise ValueError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require "
            "gate and up projections"
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
    if w1_scale.ndim not in (1, 2) or w2_scale.ndim not in (1, 2):
        raise ValueError(
            "Preshuffled routed scales must be flattened or 2D; raw "
            f"[E, N, K/32] layouts are unsupported, got {w1_scale.ndim}D/"
            f"{w2_scale.ndim}D"
        )
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
    block_m=None,
    swiglu_limit: float | None = None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    shared_w1=None,
    shared_w1_scale=None,
    shared_expert_id: int = -1,
    clamp_shared: bool = True,
    v2_output_layout: bool = False,
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
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
        swiglu_limit=swiglu_limit,
        k_wave=parsed.get("k_wave", 1),
        v2_output_layout=v2_output_layout,
        shared_expert_id=shared_expert_id,
        clamp_shared=clamp_shared,
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
    block_m=None,
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
        xcd_swizzle=parsed.get("xcd_swizzle", 0),
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
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require the "
            "two-stage FlyDSL path"
        )
    _validate_fhmoe_metadata_consistency(metadata)
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


def _validate_fhmoe_metadata_consistency(metadata):
    """Validate shared stage identity and sorting-block metadata."""
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    stage1 = get_flydsl_kernel_params(metadata.stage1.keywords.get("kernelName", ""))
    stage2 = get_flydsl_kernel_params(metadata.stage2.keywords.get("kernelName", ""))
    if stage1 is None or stage2 is None:
        raise NotImplementedError(
            "Heterogeneous MXFP4/FP8 and MXFP8/FP8 experts require valid FlyDSL kernels"
        )
    stage2_sort_block = int(stage2.get("sort_block_m", 0) or stage2["tile_m"])
    if (
        stage1["stage"] != 1
        or stage2["stage"] != 2
        or int(stage1["tile_m"]) != int(metadata.block_m)
        or stage2_sort_block != int(metadata.block_m)
    ):
        raise NotImplementedError(
            "FHMoE metadata block_m must match stage1 tile_m and stage2 sort_block_m"
        )
    return stage1, stage2


def _is_hy4_mxfp8_metadata(metadata) -> bool:
    """Validate that a dedicated HY4 row names compatible FlyDSL kernels."""
    stage1, stage2 = _validate_fhmoe_metadata_consistency(metadata)
    return (
        metadata.fuse_quant == "fp8"
        and stage1["a_dtype"] == stage1["b_dtype"] == "fp8"
        and stage2["a_dtype"] == stage2["b_dtype"] == "fp8"
        and stage2["out_dtype"] == "bf16"
        and stage1.get("gate_mode") == "interleave"
        and _valid_fhmoe_kernel_geometry(stage1, stage2, model_dim=6144)
    )


def _is_dsv4_i384_metadata(metadata) -> bool:
    """Validate that a dedicated DSV4 row names compatible FlyDSL kernels."""
    stage1, stage2 = _validate_fhmoe_metadata_consistency(metadata)
    return (
        stage1["a_dtype"] == stage2["a_dtype"] == "fp8"
        and stage1["b_dtype"] == stage2["b_dtype"] == "fp4"
        and stage2["out_dtype"] == "bf16"
        and stage1.get("gate_mode") == "interleave"
        and _valid_fhmoe_kernel_geometry(stage1, stage2, model_dim=7168)
    )


def _valid_fhmoe_kernel_geometry(stage1, stage2, *, model_dim: int) -> bool:
    """Return whether split-K/wave and stage-2 sorting geometry can compile."""
    stage1_k_wave = int(stage1.get("k_wave", 1))
    stage1_k_batch = int(stage1.get("k_batch", 1))
    if stage1_k_batch != 1 or stage1_k_wave < 1:
        return False
    stage1_k_dim = model_dim // stage1_k_batch
    stage2_sort_block = int(stage2.get("sort_block_m", 0) or stage2["tile_m"])
    return (
        stage1_k_dim % stage1_k_wave == 0
        and (stage1_k_dim // stage1_k_wave) % int(stage1["tile_k"]) == 0
        and stage2_sort_block % int(stage2["tile_m"]) == 0
    )


def _fhmoe_config_file() -> str:
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE


def _dsv4_i384_fhmoe_config_file() -> str:
    """Compatibility wrapper for existing DSV4 capability tests."""
    return _fhmoe_config_file()


def _default_fhmoe_bucket(tokens: int, get_padded_m) -> int:
    return get_padded_m(tokens)


def _supports_fhmoe_config(
    max_tokens: int,
    config_file: str,
    *,
    shape: tuple[int, int, int, int],
    q_dtype_w: torch.dtype,
    bucket_fn,
    validate_fn,
) -> bool:
    try:
        from aiter.fused_moe import get_2stage_cfgs, get_padded_M

        required_tokens = {
            bucket_fn(1 << exponent, get_padded_M)
            for exponent in range(max_tokens.bit_length())
        }
        required_tokens.add(bucket_fn(max_tokens, get_padded_M))
        model_dim, inter_dim, experts, topk = shape
        for token in required_tokens:
            metadata = get_2stage_cfgs(
                token,
                model_dim,
                inter_dim,
                experts,
                topk,
                torch.bfloat16,
                dtypes.fp8,
                q_dtype_w,
                QuantType.per_1x32,
                True,
                ActivationType.Silu,
                False,
                0,
                0,
                True,
                GateMode.INTERLEAVE,
                config_file=config_file,
            )
            metadata = _use_fhmoe_wrappers(metadata)
            if not validate_fn(metadata):
                return False
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


@functools.cache
def _supports_dsv4_i384_fhmoe_config(max_tokens: int, config_file: str) -> bool:
    return _supports_fhmoe_config(
        max_tokens,
        config_file,
        shape=(7168, 384, 385, 7),
        q_dtype_w=dtypes.fp4x2,
        bucket_fn=_default_fhmoe_bucket,
        validate_fn=_is_dsv4_i384_metadata,
    )


def supports_dsv4_i384_fhmoe(max_tokens: int) -> bool:
    """Check complete tuned DSV4 I384 CSV coverage through ``max_tokens``."""
    if type(max_tokens) is not int or max_tokens <= 0:
        return False

    try:
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _dsv4_i384_fhmoe_config_file()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return _supports_dsv4_i384_fhmoe_config(max_tokens, config_file)


@functools.cache
def _supports_hy4_mxfp8_fhmoe_config(max_tokens: int, config_file: str) -> bool:
    if max_tokens > HY4_FHMOE_MAX_TOKENS:
        return False
    return _supports_fhmoe_config(
        max_tokens,
        config_file,
        shape=(6144, 256, 257, 9),
        q_dtype_w=dtypes.fp8,
        bucket_fn=_default_fhmoe_bucket,
        validate_fn=_is_hy4_mxfp8_metadata,
    )


def supports_hy4_mxfp8_fhmoe(max_tokens: int) -> bool:
    """Check tuned HY4 coverage through the 32-bit route-output safety limit."""
    if (
        type(max_tokens) is not int
        or max_tokens <= 0
        or max_tokens > HY4_FHMOE_MAX_TOKENS
    ):
        return False
    try:
        if int(os.environ.get("AITER_BYPASS_TUNE_CONFIG", "0")) != 0:
            return False
        config_file = _fhmoe_config_file()
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return _supports_hy4_mxfp8_fhmoe_config(max_tokens, config_file)


def _is_dsv4_i384_fhmoe_contract(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
    routed_mxfp4: bool = True,
) -> bool:
    return (
        (model_dim, inter_dim, experts, topk) == (7168, 384, 385, 7)
        and routed_mxfp4
        and hidden_pad == 0
        and intermediate_pad == 0
        and gate_mode == GateMode.INTERLEAVE
        and not doweight_stage1
    )


def _uses_dsv4_fhmoe_config(
    token_num: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    hidden_pad: int,
    intermediate_pad: int,
    gate_mode: GateMode,
    doweight_stage1: bool,
    routed_mxfp4: bool = True,
) -> bool:
    """Return whether dispatch should use the dedicated DSV4 FHMoE table."""
    return _is_dsv4_i384_fhmoe_contract(
        model_dim,
        inter_dim,
        experts,
        topk,
        hidden_pad,
        intermediate_pad,
        gate_mode,
        doweight_stage1,
        routed_mxfp4,
    ) and supports_dsv4_i384_fhmoe(token_num)


def _use_hy4_mxfp8_fhmoe_contract(*, clamp_shared: bool, **geometry) -> bool:
    """Select the HY4-compatible table and validate unclamped opt-in."""
    matches_hy4_contract = _is_hy4_mxfp8_fhmoe_contract(**geometry)
    if not clamp_shared and not matches_hy4_contract:
        raise ValueError(
            "clamp_shared=False requires the HY4-compatible MXFP8 FHMoE "
            "shape and layout contract"
        )
    return matches_hy4_contract


def fhmoe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M: int = -1,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    output: torch.Tensor | None = None,
    # Model semantic forwarded by fused_moe: False is the explicit HY4
    # unclamped-shared opt-in and is rejected for other contracts.
    clamp_shared: bool = True,
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
        clamp_shared,
    )
    if output is not None:
        return output
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
    expert_mask: torch.Tensor | None = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    block_size_M: int = -1,
    num_local_tokens: torch.Tensor | None = None,
    moe_sorting_dispatch_policy: int = 0,
    dtype: torch.dtype | None = None,
    hidden_pad: int = 0,
    intermediate_pad: int = 0,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
    gate_mode: str = GateMode.SEPARATED.value,
    shared_w1: torch.Tensor | None = None,
    shared_w2: torch.Tensor | None = None,
    shared_w1_scale: torch.Tensor | None = None,
    shared_w2_scale: torch.Tensor | None = None,
    shared_expert_id: int = -1,
    output: torch.Tensor | None = None,
    # Keep this schema default aligned with fused_moe's public API.
    clamp_shared: bool = True,
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
    experts = w1.shape[0]
    model_dim = w2.shape[1]
    inter_dim = w2.shape[2] * (model_dim // w1.shape[-1])
    topk = topk_ids.shape[1]
    metadata_config_file = None
    is_hy4_contract = _use_hy4_mxfp8_fhmoe_contract(
        clamp_shared=clamp_shared,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        routed_mxfp8=w1.dtype == dtypes.fp8,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        gate_interleaved=gate_mode_enum == GateMode.INTERLEAVE,
        doweight_stage1=doweight_stage1,
        shared_expert_id=shared_expert_id,
    )
    if _is_dsv4_i384_fhmoe_contract(
        model_dim,
        inter_dim,
        experts,
        topk,
        hidden_pad,
        intermediate_pad,
        gate_mode_enum,
        doweight_stage1,
        routed_mxfp4=w1.dtype == dtypes.fp4x2,
    ):
        if not supports_dsv4_i384_fhmoe(hidden_states.shape[0]):
            raise NotImplementedError(
                "The active FHMoE config does not cover this DSV4 I384 "
                f"token shape: M={hidden_states.shape[0]}"
            )
        from aiter.jit.core import AITER_CONFIGS

        metadata_config_file = AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE
    elif is_hy4_contract:
        if not supports_hy4_mxfp8_fhmoe(hidden_states.shape[0]):
            raise NotImplementedError(
                "The active FHMoE config does not cover this HY4 MXFP8 "
                f"token shape: M={hidden_states.shape[0]}"
            )
        from aiter.jit.core import AITER_CONFIGS

        metadata_config_file = AITER_CONFIGS.AITER_CONFIG_FHMOE_FILE
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
        output=output,
        _q_dtype_a=q_dtype_a,
        _metadata_transform=_use_fhmoe_wrappers,
        _metadata_config_file=metadata_config_file,
        _enforce_metadata_block_size=True,
        _stage1_extra_args={
            "shared_w1": shared_w1,
            "shared_w1_scale": shared_w1_scale,
            "shared_expert_id": shared_expert_id,
            "swiglu_limit": swiglu_limit,
            "clamp_shared": clamp_shared,
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
