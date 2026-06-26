# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the mxfp4 MoE gemm1 (a4w4 up/gate-proj) kernel.

Mirrors aiter/ops/flydsl/moe_kernels.py: a functools.cache holds the compiled
launch fn (@flyc.jit launch fn); callers then invoke launch(*args) (the
JitFunction handles argument marshalling).
"""

import functools
import os

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
    BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave=True,
    a_dtype="fp4", out_dtype="fp4",
):
    # Backend switch: AITER_MXFP4_GEMM1_V2=1 selects the layout-API v2 port for the
    # BM32 variants -- (BM=32, inline_quant=False) -- for BOTH use_nt (BM32_NT vs
    # BM32_CACHED cache policy), BOTH gate modes (interleave / separate), and BOTH A
    # dtypes (a4w4 / a8w4). Any other variant (or the env unset) keeps the v1 kernel.
    if (
        os.environ.get("AITER_MXFP4_GEMM1_V2") == "1"
        and (BM, inline_quant) == (32, False)
    ):
        from .kernels.mxfp4_gemm1_v2 import compile_gemm1_a4w4_port

        return compile_gemm1_a4w4_port(
            BM, use_nt, inline_quant, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER,
            NE=NE, TOPK=topk, interleave=interleave, a_dtype=a_dtype, out_dtype=out_dtype,
        )
    # v1 (this branch) is a4w4-only with fp4 intermediate -- no a_dtype/out_dtype.
    if a_dtype != "fp4" or out_dtype != "fp4":
        raise NotImplementedError(
            f"flydsl mxfp4 gemm1 a_dtype={a_dtype!r}/out_dtype={out_dtype!r} requires "
            f"the v2 backend (AITER_MXFP4_GEMM1_V2=1, BM=32, inline_quant=False)"
        )
    from .kernels.mxfp4_gemm1 import compile_gemm1_a4w4_port

    return compile_gemm1_a4w4_port(
        BM, use_nt, inline_quant, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER,
        NE=NE, TOPK=topk, interleave=interleave,
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
    interleave=True,
    a_dtype="fp4",
    out_dtype="fp4",
    stream=None,
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
        BM, use_nt, inline_quant, D_HIDDEN, D_INTER, NE, topk, interleave, a_dtype, out_dtype
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
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
