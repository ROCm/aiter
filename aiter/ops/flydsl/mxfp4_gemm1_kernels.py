# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import functools

import torch

from aiter.ops.flydsl import moe_kernels as _moe_kernels

_SUPPORTED = {
    (32, True, False),
    (32, False, False),
    (64, False, False),
    (128, False, False),
    (16, True, True),
}


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
    interleave=True,
    xcd_swizzle=0,
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
    )


def _assert_supported(
    *, NE, D_HIDDEN, D_INTER, topk, BM, use_nt, inline_quant, BN=256, BK=256
):
    if D_HIDDEN % BK != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires D_HIDDEN (K) % {BK} == 0, got H={D_HIDDEN}"
        )
    if (2 * D_INTER) % BN != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires 2*D_INTER (N_OUT) % {BN} == 0, "
            f"got D_INTER={D_INTER}"
        )
    if (BM, use_nt, inline_quant) not in _SUPPORTED:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 unsupported variant "
            f"(BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
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
    interleave=True,
    xcd_swizzle=0,
    stream=None,
):
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
    )
    # OOB-pad sentinel contract: gemm1 sizes the load buffer extent from n_tokens
    # and relies on the sort writing pad rows' m_indices = M (= n_tokens) so the
    # load lands exactly past the extent and the HW returns 0. The extent is
    # n_tokens rows * row-stride, so the backing buffer must be contiguous with
    # exactly n_tokens rows (the inline path reads hidden_states, otherwise A_q).
    # See csrc/kernels/mxfp4_moe/moe_aux/moe_3stage_sort.cuh fill pad path and
    # kernels/mxfp4_gemm1.py (aq_rsrc / hidden_rsrc).
    if inline_quant:
        assert hidden_states.is_contiguous() and tuple(hidden_states.shape) == (
            n_tokens,
            D_HIDDEN,
        ), (
            f"inline-quant gemm1 hidden extent invariant violated: "
            f"shape={tuple(hidden_states.shape)} "
            f"contiguous={hidden_states.is_contiguous()} "
            f"(expected ({n_tokens}, {D_HIDDEN}))"
        )
    else:
        assert a_quant.is_contiguous() and tuple(a_quant.shape) == (
            n_tokens,
            D_HIDDEN // 2,
        ), (
            f"gemm1 A_q extent invariant violated: shape={tuple(a_quant.shape)} "
            f"contiguous={a_quant.is_contiguous()} "
            f"(expected ({n_tokens}, {D_HIDDEN // 2}))"
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
