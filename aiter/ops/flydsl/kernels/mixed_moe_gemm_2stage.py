# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Ordinary MoE facades for the shared MXFP4/FP8 kernel builders."""

import functools

from aiter.ops.flydsl.moe_common import GateMode

from .mixed_moe_gemm_2stage_common import (
    compile_a16w4_moe_gemm2,
    compile_mixed_moe_gemm1_a16w4,
    compile_mixed_moe_gemm1_common,
    compile_mixed_moe_gemm2_a16w4,
    compile_mixed_moe_gemm2_common,
    validate_moe_dtypes,
)

__all__ = [
    "GateMode",
    "compile_a16w4_moe_gemm2",
    "compile_mixed_moe_gemm1",
    "compile_mixed_moe_gemm1_a16w4",
    "compile_mixed_moe_gemm2",
    "compile_mixed_moe_gemm2_a16w4",
    "validate_moe_dtypes",
]


@functools.cache
def compile_mixed_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
    use_async_copy: bool = False,
    waves_per_eu: int = 4,
    k_batch: int = 1,
    b_nt: int = 0,
    gate_mode: GateMode = GateMode.SEPARATED,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    k_wave: int = 1,
    v2_output_layout: bool = False,
):
    """Compile an ordinary stage1 MoE kernel."""
    return compile_mixed_moe_gemm1_common(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        use_cshuffle_epilog=use_cshuffle_epilog,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        k_batch=k_batch,
        b_nt=b_nt,
        gate_mode=gate_mode,
        a_scale_one=a_scale_one,
        xcd_swizzle=xcd_swizzle,
        k_wave=k_wave,
        v2_output_layout=v2_output_layout,
    )


@functools.cache
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
    sort_block_m: int = 0,
    waves_per_eu: int | None = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    xcd_swizzle: int = 0,
):
    """Compile an ordinary stage2 MoE kernel."""
    return compile_mixed_moe_gemm2_common(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        use_cshuffle_epilog=use_cshuffle_epilog,
        accumulate=accumulate,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        persist_m=persist_m,
        sort_block_m=sort_block_m,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        cu_num_mul=cu_num_mul,
        b_nt=b_nt,
        xcd_swizzle=xcd_swizzle,
    )
