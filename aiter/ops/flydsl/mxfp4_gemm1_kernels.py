# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the mxfp4 MoE gemm1 (a4w4 up/gate-proj) kernel.

Mirrors aiter/ops/flydsl/moe_kernels.py: a functools.cache holds the compiled
launch fn (@flyc.jit launch fn); callers then invoke launch(*args) (the
JitFunction handles argument marshalling).
"""

import functools

import torch

# Shared low-overhead launch helper (flyc.compile + cached CompiledFunction).
# Imported as a module so the AOT cache-miss gate's monkeypatch is seen here.
from aiter.ops.flydsl import moe_kernels as _moe_kernels

# Supported (BM, use_nt, inline_quant) variant combinations.
_SUPPORTED = {
    (32, True, False),
    (32, False, False),
    (64, False, False),
    (128, False, False),
    (16, True, True),
}


@functools.cache
def _get_compiled_mxfp4_gemm1_port(
    BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk
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
    )


def _assert_supported(*, NE, D_HIDDEN, D_INTER, topk, BM, use_nt, inline_quant):
    # K (= D_HIDDEN, contraction), INTER (= D_INTER, output) and NE/TOPK are all
    # parametrized now. Only the real divisibility / variant constraints remain:
    #   * K must be a multiple of BK (256)
    #   * 2*D_INTER (= N_OUT) must be a multiple of BN (256), i.e. D_INTER % 128 == 0
    if D_HIDDEN % 256 != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires D_HIDDEN (K) % 256 == 0, got H={D_HIDDEN}"
        )
    if (2 * D_INTER) % 256 != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 requires 2*D_INTER (N_OUT) % 256 == 0 "
            f"(i.e. D_INTER % 128 == 0), got D_INTER={D_INTER}"
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
):
    """Run the FlyDSL port gemm1, writing inter_sorted_quant / inter_sorted_shuffled_scale.

    Same buffer I/O contract as the HIP aiter.mxfp4_moe_gemm1_a4w4 kernel;
    w1_u8 / w1_scale_u8 must be uint8 views (FlyDSL via DLPack cannot carry the
    fp4/e8m0 dtype codes).
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

    launch = _get_compiled_mxfp4_gemm1_port(
        BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk
    )
    grid = gemm1_grid(n_tokens, BM, NE=NE, TOPK=topk, INTER=D_INTER)
    # gemm1 only needs base pointers (it assumes contiguity + derives sizes
    # from n_tokens / compile-time consts), so pass raw data_ptr() addresses
    # instead of full memref descriptors -> contiguous, coalescible kernargs.
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
            torch.cuda.current_stream(),
        ),
    )
