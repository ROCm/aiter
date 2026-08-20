# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Heterogeneous MoE stage1 facade for the shared MXFP4/FP8 builder."""

import functools

from aiter.ops.flydsl.moe_common import GateMode

from .mixed_moe_gemm_2stage_common import (
    compile_mixed_moe_gemm1_common,
)


@functools.cache
def compile_mixed_fhmoe_gemm1(
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
    shared_expert_id: int,
):
    """Compile a stage1 kernel with an FP8 shared expert."""
    if shared_expert_id is None:
        raise ValueError(
            "FHMoE stage1 requires shared_expert_id == experts - 1; "
            f"got {shared_expert_id=} and {experts=}"
        )
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
        shared_expert_id=shared_expert_id,
    )
