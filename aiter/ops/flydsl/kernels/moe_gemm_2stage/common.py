# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared BF16 numerics and host specialization/cache support."""

import functools
import os
from contextlib import nullcontext

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.utils import env as flydsl_env

# The default RTE path preserves NaNs. Simplified RTA/RTE require explicit opt-in.
_SIMPLIFIED_BF16_RTA = os.environ.get(
    "AITER_FLYDSL_MOE_BF16_RTA_SIMPLIFIED", "0"
).lower() in ("1", "true")
_SIMPLIFIED_BF16_RTE = os.environ.get(
    "AITER_FLYDSL_MOE_BF16_RTE_SIMPLIFIED", "0"
).lower() in ("1", "true")


def _f32_to_bf16_rta(value):
    # Matches RTA for finite values and Inf, but does not preserve all NaNs.
    return (
        ((value.bitcast(fx.Uint32) + fx.Uint32(0x8000)) >> 16)
        .to(fx.Uint16)
        .bitcast(fx.BFloat16)
    )


def _f32_to_bf16_rte(value):
    if _SIMPLIFIED_BF16_RTE:
        bits = value.bitcast(fx.Uint32)
        rounded = bits + fx.Uint32(0x7FFF) + ((bits >> 16) & fx.Uint32(1))
        return (rounded >> 16).to(fx.Uint16).bitcast(fx.BFloat16)
    # ck_tile/float_to_bf16_rtn_asm canonicalizes NaNs to 0x7FFF.
    rounded = llvm.inline_asm(
        T.i32,
        [_raw(value), _raw(fx.Uint32(0x7FFF)), _raw(fx.Uint32(0x7FFF0000))],
        "v_cmp_u_f32 vcc, $1, $1\n\t"
        "v_bfe_u32 $0, $1, 16, 1\n\t"
        "v_add3_u32 $0, $1, $0, $2\n\t"
        "v_cndmask_b32 $0, $0, $3, vcc",
        "=&v,v,v,v,~{vcc}",
        has_side_effects=False,
    )
    return (fx.Uint32(rounded) >> 16).to(fx.Uint16).bitcast(fx.BFloat16)


def _f32_to_bf16(value):
    if _SIMPLIFIED_BF16_RTA:
        return _f32_to_bf16_rta(value)
    if isinstance(value, Vec):
        return Vec.from_elements(
            [_f32_to_bf16_rte(value[i]) for i in range_constexpr(value.numel)],
            fx.BFloat16,
        )
    return _f32_to_bf16_rte(value)


_TORCH_TO_FX = {
    torch.bfloat16: fx.BFloat16,
    torch.float32: fx.Float32,
    torch.float64: fx.Float64,
    torch.int32: fx.Int32,
    torch.float8_e4m3fnuz: fx.Uint8,
    torch.float8_e4m3fn: fx.Uint8,
}


def torch_tensor_to_pointer(tensor):
    return flyc.from_c_void_p(_TORCH_TO_FX[tensor.dtype], tensor.data_ptr())


@functools.cache
def _get_device_cache_key(device):
    properties = torch.cuda.get_device_properties(device)
    return (
        device,
        properties.name,
        properties.gcnArchName,
        properties.multi_processor_count,
    )


def get_device_cache_key(device=None):
    """Separate host launch caches by physical device and compile-only target."""
    target_arch = (os.environ.get("ARCH"), os.environ.get("FLYDSL_GPU_ARCH"))
    target_cu_count = os.environ.get("CU_NUM")
    if flydsl_env.compile.compile_only:
        return "compile_only", target_arch, target_cu_count
    if not torch.cuda.is_available():
        return None, target_arch, target_cu_count
    if device is None:
        device = torch.cuda.current_device()
    elif not isinstance(device, int):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Expected a cuda device, but got: {device}")
        device = device.index
        if device is None:
            device = torch.cuda.current_device()
    return _get_device_cache_key(device), target_arch, target_cu_count


def device_context(device=None):
    # Forked AOT workers must not touch CUDA/HIP, even if the parent initialized it.
    if flydsl_env.compile.compile_only:
        return nullcontext()
    if device is not None and torch.cuda.is_available():
        if isinstance(device, int):
            index = device
        else:
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError(f"Expected a cuda device, but got: {device}")
            index = device.index
        if index is not None and index == torch.cuda.current_device():
            return nullcontext()
        return torch.cuda.device(device)
    return nullcontext()


def resolve_tile_k(tile_k):
    # Resolve before cache lookup so the environment override is part of the key.
    if tile_k is None and os.environ.get("MOE_PREFILL_TILE_K"):
        tile_k = int(os.environ["MOE_PREFILL_TILE_K"])
    return tile_k


def validate_gemm_options(
    weight_dtype,
    weight_quant_type,
    act_quant_type,
    BLOCK_TILE_SIZE_M,
    alg,
    activation,
    swiglu_limit,
):
    if act_quant_type is None:
        act_quant_type = weight_quant_type
    assert (
        BLOCK_TILE_SIZE_M <= 256
    ), "BLOCK_SIZE_M must be less than or equal to 256 due to LDS size limit for sorted ids."
    assert weight_dtype in [
        "bf16",
        "fp8",
    ], "weight_dtype must be either 'bf16' or 'fp8'"
    assert weight_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "weight_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert act_quant_type in [
        "no",
        "ptpc",
        "per_tensor",
    ], "act_quant_type must be either 'no', 'ptpc' or 'per_tensor'"
    assert activation in [
        "silu",
        "swiglu",
    ], "activation must be either 'silu' or 'swiglu'"
    if activation == "swiglu":
        swiglu_limit = float(swiglu_limit) if swiglu_limit else 7.0
    if weight_dtype == "fp8" and alg == "prefill_1x4":
        assert (weight_quant_type == "ptpc" and act_quant_type == "ptpc") or (
            weight_quant_type == "per_tensor"
            and act_quant_type in ("ptpc", "per_tensor")
        ), (
            f"unsupported prefill quant combo (weight={weight_quant_type}, "
            f"act={act_quant_type})"
        )
    return act_quant_type, swiglu_limit
