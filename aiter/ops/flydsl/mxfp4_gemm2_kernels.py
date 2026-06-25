# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the mxfp4 MoE gemm2 (a4w4 down-proj) kernel.

Mirrors aiter/ops/flydsl/mxfp4_gemm1_kernels.py: a functools.cache holds the
compiled launch fn (@flyc.jit launch fn); callers then invoke launch(*args) (the
JitFunction handles argument marshalling).

gemm2 has several epilogs (atomic, mxfp4out, cshuffle), selected by fused_moe's
kernelName2:
  * atomic           -> LDS cshuffle + global_atomic_fadd x sorted_weights (BM16/32/64)
  * nonatomic        -> flat per-sorted-row bf16 write (BM128), then scatter_reduce
  * nonatomic_mxfp4  -> flat per-sorted-row fp4 (q + e8m0 scale) write (BM128)
"""

import functools

import torch

# Shared low-overhead launch helper (see mxfp4_gemm1_kernels); imported as a
# module so the AOT cache-miss gate's monkeypatch is seen here.
from aiter.ops.flydsl import moe_kernels as _moe_kernels

# Supported (BM, use_nt, epilog) combinations (mirrors the prebuilt HIP set).
_SUPPORTED = {
    # atomic: BM in {16,32,64} x {ATOMIC, NT}
    (16, False, "atomic"),
    (16, True, "atomic"),
    (32, False, "atomic"),
    (32, True, "atomic"),
    (64, False, "atomic"),
    (64, True, "atomic"),
    # nonatomic bf16 flat (persistent grid)
    (128, False, "nonatomic"),
    # nonatomic mxfp4-out (fp4 q + e8m0 scale flat)
    (128, False, "nonatomic_mxfp4"),
    # nonatomic bf16 WITH cshuffle (coalesced flat write, BM<=64) -- fly recipe
    (32, False, "nonatomic_cshuffle"),
    (64, False, "nonatomic_cshuffle"),
    (128, False, "nonatomic_cshuffle"),  # 2-pass cshuffle (64-row scratch)
}


def _epilog_of(atomic, mxfp4out, cshuffle=False):
    if mxfp4out:
        return "nonatomic_mxfp4"
    if cshuffle:
        return "nonatomic_cshuffle"
    return "atomic" if atomic else "nonatomic"


@functools.cache
def _get_compiled_mxfp4_gemm2_port(
    BM, use_nt, NE, N_OUT, epilog, D_INTER, D_INTER_REAL=None, BN=256, BK=256
):
    from .kernels.mxfp4_gemm2 import compile_gemm2_a4w4_port

    return compile_gemm2_a4w4_port(
        BM=BM,
        use_nt=use_nt,
        NE=NE,
        N_OUT=N_OUT,
        epilog=epilog,
        D_INTER=D_INTER,
        D_INTER_REAL=D_INTER_REAL,
        BN=BN,
        BK=BK,
    )


@functools.cache
def _dummy_out_scale(device):
    return torch.empty(1, dtype=torch.uint8, device=device)


def _assert_supported(
    *,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    BM,
    use_nt,
    atomic,
    mxfp4out,
    cshuffle=False,
    BN=256,
    BK=256,
):
    # gemm2 contraction K = inter_dim = D_INTER. The kernel (BK=256) supports any
    # D_INTER % 256 == 0 (KIMI/DSR 512 keeps the original fully-unrolled fast path;
    # >512 uses the streaming K-loop). D_HIDDEN (= model_dim / output) must equal the
    # prebuilt HIP H=7168, and NE in {257,385}, TOPK=9 -- fail-loud otherwise.
    if D_INTER % BK != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 contraction D_INTER (=inter_dim) must be a "
            f"multiple of {BK}; D_INTER not divisible by {BK} (e.g. "
            f"384/192) is not supported by this BK={BK} kernel "
            f"(got D_INTER={D_INTER})"
        )
    # N_OUT (= D_HIDDEN = model_dim) and NE are compile-parametrized; gemm2's real
    # constraint is N_OUT % BN(256) == 0 (the kernel tiles the output by BN=256).
    # (Was KIMI/DSR-gated to NE in (257,385), H=7168 -- relaxed to match gemm1's
    # _assert. Pipeline-level routing limits, e.g. mxfp4_moe_sort's NE/TOPK gating,
    # are enforced by those components, not here.)
    if D_HIDDEN % BN != 0:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 requires D_HIDDEN (=N_OUT=model_dim) % {BN} == 0, "
            f"got H={D_HIDDEN}"
        )
    epilog = _epilog_of(atomic, mxfp4out, cshuffle)
    if (BM, use_nt, epilog) not in _SUPPORTED:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 unsupported variant "
            f"(BM={BM}, use_nt={use_nt}, epilog={epilog})"
        )


def flydsl_mxfp4_gemm2(
    *,
    inter_sorted_quant,
    inter_sorted_shuffled_scale,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    BM,
    use_nt,
    atomic,
    mxfp4out,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    flat_out_scale=None,
    cshuffle=False,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
    stream=None,
):
    """Run the FlyDSL port gemm2, writing flat_out (plus flat_out_scale when mxfp4out).

    Same buffer I/O contract as the HIP aiter.mxfp4_moe_gemm2_a4w4 kernel;
    w2_u8 / w2_scale_u8 must be uint8 views (FlyDSL via DLPack cannot carry the
    fp4/e8m0 dtype codes).
    """
    _assert_supported(
        NE=NE,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        topk=topk,
        BM=BM,
        use_nt=use_nt,
        atomic=atomic,
        mxfp4out=mxfp4out,
        cshuffle=cshuffle,
        BN=BN,
        BK=BK,
    )
    epilog = _epilog_of(atomic, mxfp4out, cshuffle)
    launch = _get_compiled_mxfp4_gemm2_port(
        BM, use_nt, NE, D_HIDDEN, epilog, D_INTER, D_INTER_REAL, BN, BK
    )

    # grid size = max_m_blocks (an upper bound; the kernel reads the true active
    # block count from cumsum at runtime and early-exits past it).
    max_m_blocks = (max_sorted + BM - 1) // BM

    # Only the mxfp4 epilog writes a scale; pass a dummy out_scale otherwise so the
    # launch signature stays uniform.
    out_scale = flat_out_scale if mxfp4out else _dummy_out_scale(flat_out.device)

    # gemm2 only needs base pointers (assumes contiguity + derives sizes from
    # compile-time consts), so pass raw data_ptr() addresses instead of full
    # memref descriptors -> contiguous, coalescible kernargs (ported from gemm1).
    _moe_kernels._run_compiled(
        launch,
        (
            inter_sorted_quant.data_ptr(),
            inter_sorted_shuffled_scale.data_ptr(),
            w2_u8.data_ptr(),
            w2_scale_u8.data_ptr(),
            sorted_expert_ids.data_ptr(),
            cumsum_tensor.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            M_logical,
            max_m_blocks,
            flat_out.data_ptr(),
            out_scale.data_ptr(),
            torch.cuda.current_stream() if stream is None else stream,
        ),
    )
