# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels
from aiter.ops.flydsl.mxfp4_kname import MXFP4_G1_VARIANTS


@functools.cache
def _get_compiled_mxfp4_gemm1_port(
    BM,
    use_nt,
    inline_quant,
    D_HIDDEN,
    D_INTER,
    NE,
    BN,
    BK,
    interleave=False,
    xcd_swizzle=0,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    enable_bias=False,
    native_scale_layout=False,
    num_waves=4,
    k_wave=1,
):
    from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    return compile_gemm1_a4w4_port(
        BM,
        use_nt,
        inline_quant,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        BN=BN,
        BK=BK,
        interleave=interleave,
        xcd_swizzle=xcd_swizzle,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        enable_bias=enable_bias,
        native_scale_layout=native_scale_layout,
        num_waves=num_waves,
        k_wave=k_wave,
    )


def _assert_supported(
    *,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM,
    use_nt,
    inline_quant,
    BN=256,
    BK=256,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    interleave=False,
    num_waves=4,
    native_scale_layout=False,
    k_wave=1,
):
    if a_dtype not in MXFP4_G1_VARIANTS:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires a_dtype in {tuple(MXFP4_G1_VARIANTS)}, "
            f"got {a_dtype!r}"
        )
    if out_dtype not in ("fp4", "fp8"):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires out_dtype in ('fp4', 'fp8'), got {out_dtype!r}"
        )
    if act not in ("silu", "swiglu", "situv2"):
        raise NotImplementedError(
            "flydsl mxfp4 gemm1 requires act in "
            f"('silu', 'swiglu', 'situv2'), got {act!r}"
        )
    if act == "situv2":
        if situ_beta <= 0.0:
            raise NotImplementedError(
                f"flydsl mxfp4 gemm1 requires situ_beta > 0, got {situ_beta!r}"
            )
        if situ_linear_beta <= 0.0:
            raise NotImplementedError(
                "flydsl mxfp4 gemm1 requires situ_linear_beta > 0, "
                f"got {situ_linear_beta!r}"
            )
    if act == "swiglu" and swiglu_limit <= 0.0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires swiglu_limit > 0, got {swiglu_limit!r}"
        )
    if D_HIDDEN % BK != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires D_HIDDEN (K) % {BK} == 0, got H={D_HIDDEN}"
        )
    if (2 * D_INTER) % BN != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires 2*D_INTER (N_OUT) % {BN} == 0, "
            f"got D_INTER={D_INTER}"
        )
    if BN not in (64, 128, 256):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires BN in (64, 128, 256), got {BN}"
        )
    if BN == 64 and not (
        BM == 32
        and a_dtype == "fp4"
        and out_dtype == "fp4"
        and not inline_quant
        and not interleave
    ):
        raise NotImplementedError(
            "flydsl mxfp4 GEMM1 BN64 is restricted to BM32 A4W4 non-inline separated"
        )
    if num_waves not in (2, 4):
        raise NotImplementedError(
            f"flydsl mxfp4 GEMM1 requires num_waves in (2, 4), got {num_waves}"
        )
    if num_waves == 2 and BN != 64:
        raise NotImplementedError(
            "flydsl mxfp4 GEMM1 two-wave specialization requires effective BN64"
        )
    if native_scale_layout and not (BM == 16 and out_dtype == "fp4"):
        raise NotImplementedError(
            "flydsl mxfp4 GEMM1 native scale layout requires BM16 FP4 output"
        )
    if k_wave not in (1, 2, 4):
        raise NotImplementedError(
            f"flydsl mxfp4 GEMM1 requires k_wave in (1, 2, 4), got {k_wave}"
        )
    if k_wave > 1:
        if BM != 32:
            raise NotImplementedError("k_wave > 1 is currently restricted to BM32")
        if inline_quant:
            raise NotImplementedError("k_wave > 1 does not support inline quantization")
        if interleave:
            raise NotImplementedError("k_wave > 1 requires separated gate/up layout")
        if num_waves * k_wave > 8:
            raise NotImplementedError(
                f"k_wave creates too many waves: {num_waves} * {k_wave} > 8"
            )
        k_tiles = D_HIDDEN // BK
        if k_tiles % k_wave != 0:
            raise NotImplementedError(
                f"D_HIDDEN/BK={k_tiles} must be divisible by k_wave={k_wave}"
            )
        if k_tiles // k_wave < 2:
            raise NotImplementedError(
                f"k_wave leaves fewer than 2 K tiles per wave: {k_tiles // k_wave}"
            )
    if (BM, use_nt, inline_quant) not in MXFP4_G1_VARIANTS[a_dtype]:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 unsupported variant "
            f"(a_dtype={a_dtype}, BM={BM}, use_nt={use_nt}, "
            f"inline_quant={inline_quant})"
        )


def flydsl_mxfp4_gemm1(
    *,
    a_quant,
    a_scale_sorted_shuffled,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    hidden_states,
    n_tokens,
    BM,
    use_nt,
    inline_quant,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BN=256,
    BK=256,
    interleave=False,
    xcd_swizzle=0,
    native_scale_layout=False,
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    bias=None,
    stream=None,
    num_waves=4,
    k_wave=1,
):
    """Launch GEMM1; v2 output keeps payload rows in expert-sorted order."""
    _assert_supported(
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BM=BM,
        use_nt=use_nt,
        inline_quant=inline_quant,
        BN=BN,
        BK=BK,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        act=act,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_limit=swiglu_limit,
        interleave=interleave,
        num_waves=num_waves,
        native_scale_layout=native_scale_layout,
        k_wave=k_wave,
    )
    from .kernels.mxfp4_gemm1 import gemm1_grid

    launch = _get_compiled_mxfp4_gemm1_port(
        BM,
        use_nt,
        inline_quant,
        D_HIDDEN,
        D_INTER,
        NE,
        BN,
        BK,
        interleave,
        xcd_swizzle,
        a_dtype,
        out_dtype,
        act,
        situ_beta,
        situ_linear_beta,
        swiglu_limit,
        bias is not None,
        native_scale_layout,
        num_waves,
        k_wave,
    )
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER, BN=BN)
    _moe_kernels._run_compiled(
        launch,
        (
            a_quant.data_ptr(),
            a_scale_sorted_shuffled.data_ptr(),
            w1_u8.data_ptr(),
            w1_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            m_indices.data_ptr(),
            n_tokens,
            grid,
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            hidden_states.data_ptr(),
            0 if bias is None else bias.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
