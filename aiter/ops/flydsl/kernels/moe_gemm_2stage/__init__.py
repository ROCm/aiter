# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture-neutral home of the baseline two-stage MoE kernels."""

import functools

from .common import device_context, get_device_cache_key, resolve_tile_k
from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2
from .moe_reduce import (
    compile_moe_reduction,
    invert_sorted_ids,
    precompile_moe_reduction_kernels,
    sorted_sum,
)
from .quant import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
    precompile_moe_quant_kernels,
)

__all__ = [
    "compile_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "compile_moe_reduction",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "invert_sorted_ids",
    "precompile_moe_quant_kernels",
    "precompile_moe_reduction_kernels",
    "sorted_sum",
]


@functools.cache
def _compile_gemm_cached(
    device_cache_key,
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage,
    alg,
    E,
    USE_ATOMIC_WRITE,
    act_quant_type,
    tile_k,
    activation,
    swiglu_limit,
):
    del device_cache_key
    if stage == "gateup":
        return compile_moe_gemm1(
            N=N,
            K=K,
            weight_dtype=weight_dtype,
            weight_quant_type=weight_quant_type,
            TOPK=TOPK,
            BLOCK_TILE_SIZE_M=BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N=BLOCK_TILE_SIZE_N,
            alg=alg,
            E=E,
            act_quant_type=act_quant_type,
            tile_k=tile_k,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
    return compile_moe_gemm2(
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
    )


def compile_gemm(
    N,
    K,
    weight_dtype,
    weight_quant_type,
    TOPK,
    BLOCK_TILE_SIZE_M,
    BLOCK_TILE_SIZE_N,
    stage="gateup",
    alg="splitk",
    E=None,
    USE_ATOMIC_WRITE=True,
    act_quant_type=None,
    tile_k=None,
    activation="silu",
    swiglu_limit=None,
    *,
    device=None,
):
    """Build a baseline stage launcher; device only selects the host cache/context."""
    with device_context(device):
        return _compile_gemm_cached(
            get_device_cache_key(device),
            N,
            K,
            weight_dtype,
            weight_quant_type,
            TOPK,
            BLOCK_TILE_SIZE_M,
            BLOCK_TILE_SIZE_N,
            stage,
            alg,
            E,
            USE_ATOMIC_WRITE,
            act_quant_type,
            resolve_tile_k(tile_k),
            activation,
            swiglu_limit,
        )


def _clear_gemm_caches():
    _compile_gemm_cached.cache_clear()
    compile_moe_gemm1.cache_clear()
    compile_moe_gemm2.cache_clear()


compile_gemm.cache_clear = _clear_gemm_caches
compile_gemm.cache_info = _compile_gemm_cached.cache_info
