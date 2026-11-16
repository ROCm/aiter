# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors


def compile_gemm2_ep32_bm160(
    *,
    BM=160,
    variant="native",
    use_nt=False,
    HIDDEN_MAX=8192,
    INTER_MAX=3072,
    topk=16,
    use_valid_token_count=True,
    GRID_WGS=512,
    B_NT=True,
    REDUCE=True,
    NATIVE_C=True,
    NATIVE_BF16=True,
    BN=256,
):
    if BN != 256:
        raise ValueError("M8192 GEMM2 requires BN256")
    if (
        (BM, INTER_MAX, use_nt, REDUCE, NATIVE_C, NATIVE_BF16)
        != (
            160,
            3072,
            False,
            True,
            True,
            True,
        )
        or topk not in (16, 17)
        or variant not in ("native", "w12")
    ):
        raise ValueError("BM160 GEMM2 requires native weighted BF16 Kimi-K3 staging")
    from .mxfp4_gemm2_ep32_8192 import compile_gemm2_ep8192

    return compile_gemm2_ep8192(B_NT=B_NT, GRID_WGS=GRID_WGS, topk=topk)
