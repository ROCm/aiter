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
):
    if a_dtype not in _SUPPORTED_BY_DTYPE:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires a_dtype in {tuple(_SUPPORTED_BY_DTYPE)}, "
            f"got {a_dtype!r}"
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
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
