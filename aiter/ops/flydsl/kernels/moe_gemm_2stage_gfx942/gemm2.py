# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Path dispatcher for MoE stage2 down-projection builders."""

import functools

from .common import get_device_cache_key
from .gemm2_1x4 import _build_moe_gemm2_1x4
from .gemm2_1x8 import _build_moe_gemm2_1x8
from .gemm2_2x4 import _build_moe_gemm2_2x4
from .gemm2_default import _build_moe_gemm2_default

_BUILDERS = {
    "default": _build_moe_gemm2_default,
    "1x4_64x256": _build_moe_gemm2_1x4,
    "1x8": _build_moe_gemm2_1x8,
    "2x4": _build_moe_gemm2_2x4,
}


@functools.cache
def _compile_moe_gemm2_cached(
    device_cache_key,
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    del device_cache_key
    assert down_path in _BUILDERS
    builder = _BUILDERS[down_path]
    default_kwargs = {}
    if down_path == "default":
        default_kwargs = {
            "mxfp4_gate_up_interleaved": mxfp4_gate_up_interleaved,
            "fused_down_clear": fused_down_clear,
        }
    return builder(
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        stage="down",
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
        **default_kwargs,
    )


def compile_moe_gemm2(
    *,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    mxfp4_gate_up_interleaved=True,
    fused_down_clear=False,
    down_path="default",
    down_output_padding_bytes=None,
    METADATA_TILE_SIZE_M=None,
):
    """Build and cache a stage2 down-projection launcher for one static configuration."""
    return _compile_moe_gemm2_cached(
        get_device_cache_key(),
        N=N,
        K=K,
        weight_dtype=weight_dtype,
        weight_quant_type=weight_quant_type,
        TOPK=TOPK,
        BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
        BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
        alg=alg,
        E=E,
        USE_ATOMIC_WRITE=USE_ATOMIC_WRITE,
        act_quant_type=act_quant_type,
        tile_k=tile_k,
        activation=activation,
        swiglu_limit=swiglu_limit,
        mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
        fused_down_clear=fused_down_clear,
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
        METADATA_TILE_SIZE_M=METADATA_TILE_SIZE_M,
    )


compile_moe_gemm2.cache_clear = _compile_moe_gemm2_cached.cache_clear
compile_moe_gemm2.cache_info = _compile_moe_gemm2_cached.cache_info
