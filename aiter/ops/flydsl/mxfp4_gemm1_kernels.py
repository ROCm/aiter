# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels

_SUPPORTED = {
    (32, True, False),
    (32, False, False),
    (64, True, False),
    (64, False, False),
    (128, False, False),
    (16, True, True),
}
_SUPPORTED_BY_DTYPE = {
    "fp4": _SUPPORTED,
    "fp8": {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    },
}


def _effective_use_nt(*, n_tokens, topk, NE, BM, use_nt, inline_quant):
    """Keep BM32 streaming loads only while each expert averages under one M tile."""
    if use_nt and not inline_quant and BM == 32:
        total_m_blocks = (int(n_tokens) * int(topk) + BM - 1) // BM
        if total_m_blocks >= int(NE):
            return False
    return use_nt


@functools.cache
def _get_compiled_mxfp4_gemm1_port(
    BM,
    use_nt,
    inline_quant,
    D_HIDDEN,
    D_INTER,
    NE,
    topk,
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
):
    from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    return compile_gemm1_a4w4_port(
        BM,
        use_nt,
        inline_quant,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
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
):
    if a_dtype not in _SUPPORTED_BY_DTYPE:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires a_dtype in {tuple(_SUPPORTED_BY_DTYPE)}, "
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
    if BN not in (128, 256):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires BN in (128, 256), got {BN}"
        )
    if (BM, use_nt, inline_quant) not in _SUPPORTED_BY_DTYPE[a_dtype]:
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
    a_dtype="fp4",
    out_dtype="fp4",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=7.0,
    bias=None,
    stream=None,
):
    use_nt = _effective_use_nt(
        n_tokens=n_tokens,
        topk=topk,
        NE=NE,
        BM=BM,
        use_nt=use_nt,
        inline_quant=inline_quant,
    )
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
    )
    from .kernels.mxfp4_gemm1 import gemm1_grid

    launch = _get_compiled_mxfp4_gemm1_port(
        BM,
        use_nt,
        inline_quant,
        D_HIDDEN,
        D_INTER,
        NE,
        topk,
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
    )
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER, BN=BN)
    if BM == 16:
        # Two M blocks atomically pack scale bytes into each dword. The GEMM
        # uses the current/default stream, so this clear is ordered before it.
        inter_sorted_shuffled_scale.zero_()
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
