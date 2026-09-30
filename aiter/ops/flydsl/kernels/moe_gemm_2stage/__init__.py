# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture-neutral home of the baseline two-stage MoE kernels."""

from .gemm1 import compile_moe_gemm1
from .gemm2 import compile_moe_gemm2
from .moe_reduce import invert_sorted_ids, sorted_sum
from .quant import (
    flydsl_absmax,
    flydsl_quant_per_tensor,
)

__all__ = [
    "compile_gemm",
    "compile_moe_gemm1",
    "compile_moe_gemm2",
    "flydsl_absmax",
    "flydsl_quant_per_tensor",
    "invert_sorted_ids",
    "sorted_sum",
]


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
    down_path="default",
    down_output_padding_bytes=0,
    fused_down_clear=False,
    situ_beta=1.0,
    situ_linear_beta=1.0,
    mxfp4_gate_up_interleaved=True,
):
    if stage == "gateup":
        if down_path != "default" or down_output_padding_bytes != 0:
            raise ValueError("Down-specific options cannot be used for Gate/Up")
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
            fused_down_clear=fused_down_clear,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            mxfp4_gate_up_interleaved=mxfp4_gate_up_interleaved,
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
        down_path=down_path,
        down_output_padding_bytes=down_output_padding_bytes,
    )
