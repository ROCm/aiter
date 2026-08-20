# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Ordinary MoE facades for the shared MXFP4/FP8 kernel builders."""

import functools

from aiter.ops.flydsl.moe_common import GateMode

# a16w4 (bf16 A x mxfp4 W) SiTUv2 is now served by the ported FlyDSL kernel
# (aiter/ops/flydsl/kernels/moe_2stage_a16wmix), built via compile_flydsl_moe_stage1
# and launched by flydsl_a16w4_gemm1 from the a16w4 stage1 path in
# _flydsl_moe_stage1_impl (aiter/ops/flydsl/moe_kernels.py). The ordinary stage2
# path is served by the layout-v2 MXFP4 kernels and is intentionally not exposed here.
from .mixed_moe_gemm_2stage_common import (
    compile_mixed_moe_gemm1_common,
    validate_moe_dtypes,
)

__all__ = [
    "GateMode",
    "compile_mixed_moe_gemm1",
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
