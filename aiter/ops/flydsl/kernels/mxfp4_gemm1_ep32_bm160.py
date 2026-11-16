# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors


def compile_gemm1_ep32_bm160(
    BM=160,
    use_nt=True,
    inline_quant=False,
    *,
    prefetch_hidden=False,
    D_HIDDEN,
    D_INTER,
    NE,
    BN=256,
    BK=256,
    interleave=True,
    xcd_swizzle=0,
    a_dtype="fp8",
    out_dtype="fp8",
    act="situv2",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
    native_scale_layout=False,
    num_waves=12,
    k_wave=1,
):
    if BN != 256:
        raise ValueError("M8192 GEMM1 requires BN256")
    if num_waves != 12:
        raise ValueError("M8192 GEMM1 requires twelve waves")
    if (
        BM,
        D_HIDDEN,
        D_INTER,
        NE,
        BN,
        BK,
        interleave,
        a_dtype,
        out_dtype,
        num_waves,
        k_wave,
        inline_quant,
        prefetch_hidden,
        xcd_swizzle,
        native_scale_layout,
    ) != (
        160,
        3584,
        3072,
        28,
        BN,
        256,
        True,
        "fp8",
        "fp8",
        num_waves,
        1,
        False,
        False,
        0,
        False,
    ):
        raise ValueError("BM160 GEMM1 requires Kimi-K3 EP32 A8W4, BN256 BK256")
    if act != "situv2" or situ_beta <= 0 or situ_linear_beta <= 0:
        raise ValueError("BM160 GEMM1 requires SiTUv2 with positive beta values")
    if enable_bias:
        raise ValueError("M8192 BM160 GEMM1 does not support bias")
    from .mxfp4_gemm1_ep32_8192 import compile_gemm1_ep8192

    return compile_gemm1_ep8192(
        BN=BN, use_nt=use_nt, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta
    )
