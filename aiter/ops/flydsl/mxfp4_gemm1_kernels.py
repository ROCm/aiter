# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL 移植版 mxfp4 MoE gemm1 (a4w4 up/gate-proj) 的运行时包装。

仿 aiter/ops/flydsl/moe_kernels.py：用 functools.cache 缓存编译产物
(@flyc.jit launch fn)，调用时直接 launch(*args)（JitFunction 内部再缓存实际编译）。
"""

import functools

import torch

# 端口支持的 (BM, use_nt, inline_quant) 变体组合。
_SUPPORTED = {
    (32, True, False),
    (32, False, False),
    (128, False, False),
    (16, True, True),
}


@functools.cache
def _get_compiled_mxfp4_gemm1_port(BM, use_nt, inline_quant):
    from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    return compile_gemm1_a4w4_port(BM, use_nt, inline_quant)


def _assert_supported(*, NE, D_HIDDEN, D_INTER, topk, BM, use_nt, inline_quant):
    from .kernels import mxfp4_gemm1 as port

    if (NE, D_HIDDEN, D_INTER, topk) != (port.NE, port.K, port.INTER, port.TOPK):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 仅支持 Kimi 形状 "
            f"(NE={port.NE}, H={port.K}, INTER={port.INTER}, TOPK={port.TOPK})，"
            f"实际 (NE={NE}, H={D_HIDDEN}, INTER={D_INTER}, TOPK={topk})"
        )
    if (BM, use_nt, inline_quant) not in _SUPPORTED:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 不支持变体 (variant) "
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
    """运行 FlyDSL 移植版 gemm1，写入 inter_sorted_quant / inter_sorted_shuffled_scale。

    输入须与 HIP aiter.mxfp4_moe_gemm1_a4w4 完全一致的字节布局；
    w1_u8 / w1_scale_u8 应为 uint8 view（FlyDSL 经 DLPack 拒绝 fp4/e8m0 dtype code）。
    """
    _assert_supported(
        NE=NE, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, topk=topk,
        BM=BM, use_nt=use_nt, inline_quant=inline_quant,
    )
    from .kernels.mxfp4_gemm1 import gemm1_grid

    launch = _get_compiled_mxfp4_gemm1_port(BM, use_nt, inline_quant)
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
