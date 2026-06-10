# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL ??? mxfp4 MoE gemm1 (a4w4 up/gate-proj) ???????

? aiter/ops/flydsl/moe_kernels.py:? functools.cache ??????
(@flyc.jit launch fn),????? launch(*args)(JitFunction ?????????)?
"""

import functools

import torch

# ????? (BM, use_nt, inline_quant) ?????
_SUPPORTED = {
    (32, True, False),
    (32, False, False),
    (128, False, False),
    (16, True, True),
}


@functools.cache
def _get_compiled_mxfp4_gemm1_port(BM, use_nt, inline_quant, D_HIDDEN):
    from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    return compile_gemm1_a4w4_port(BM, use_nt, inline_quant, D_HIDDEN=D_HIDDEN)


def _assert_supported(*, NE, D_HIDDEN, D_INTER, topk, BM, use_nt, inline_quant):
    from .kernels import mxfp4_gemm1 as port

    # K (= D_HIDDEN, the contraction dim) is parametrized; only the OUTPUT side
    # (NE, INTER, TOPK) is still fixed. K must be a multiple of BK(256).
    if (NE, D_INTER, topk) != (port.NE, port.INTER, port.TOPK):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 ??? Kimi ? NE/INTER/TOPK "
            f"(NE={port.NE}, INTER={port.INTER}, TOPK={port.TOPK}),"
            f"?? (NE={NE}, INTER={D_INTER}, TOPK={topk})"
        )
    if D_HIDDEN % 256 != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 ?? D_HIDDEN (K) ? 256 ???,?? H={D_HIDDEN}"
        )
    if (BM, use_nt, inline_quant) not in _SUPPORTED:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 ????? (variant) "
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
):
    """?? FlyDSL ??? gemm1,?? inter_sorted_quant / inter_sorted_shuffled_scale?

    ???? HIP aiter.mxfp4_moe_gemm1_a4w4 ?????????;
    w1_u8 / w1_scale_u8 ?? uint8 view(FlyDSL ? DLPack ?? fp4/e8m0 dtype code)?
    """
    _assert_supported(
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BM=BM,
        use_nt=use_nt,
        inline_quant=inline_quant,
    )
    from .kernels.mxfp4_gemm1 import gemm1_grid

    launch = _get_compiled_mxfp4_gemm1_port(BM, use_nt, inline_quant, D_HIDDEN)
    grid = gemm1_grid(n_tokens, BM)
    launch(
        a_quant,
        a_scale_sorted_shuffled,
        w1_u8,
        w1_scale_u8,
        sorted_expert_ids,
        cumsum_tensor,
        m_indices,
        n_tokens,
        grid,
        inter_sorted_quant,
        inter_sorted_shuffled_scale,
        hidden_states,
        torch.cuda.current_stream(),
    )
