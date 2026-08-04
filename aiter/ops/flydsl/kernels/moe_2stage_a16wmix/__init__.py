# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4/a16wi4/a16w16 (bf16 A x mxfp4/int4/bf16 W) 2-stage MoE kernels.

CDNA MFMA pipeline. bf16 A (no A-scale), W1/W2 upconverted to bf16 in-kernel,
non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` by sorted position (no requant/scale).
  - stage2 (:mod:`gemm2`): down-proj GEMM + routing-weighted atomic bf16 scatter
    to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Shared low-level helpers live in
:mod:`gemm1` (imported by :mod:`gemm2`); host-side launch glue is defined below.

Launch args are raw device pointers (``fx.Int64``); tensors passed as
``.data_ptr()``.
"""

import functools

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import a16wmix_use_k16, compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid

__all__ = [
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
]


@functools.cache
def _get_compiled_gemm1_a16w4(
    BM,
    D_HIDDEN,
    D_INTER,
    NE,
    topk,
    TILE_N,
    TILE_K,
    act,
    b_cache_mod,
    xcd_swizzle,
    waves_per_eu,
    w_dtype="fp4",
    w_layout="standard",
    k_wave=1,
):
    # _get_compiled_* is @functools.cache'd, so this builder runs once; the
    # returned @flyc.jit launcher is then dispatched via _run_compiled.
    return compile_gemm1_a16w4_port(
        BM=BM,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        act=act,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        w_layout=w_layout,
        k_wave=k_wave,
        use_k16=a16wmix_use_k16(get_rocm_arch()),
    )


@functools.cache
def _get_compiled_gemm2_a16w4(
    BM,
    NE,
    N_OUT,
    D_INTER,
    TILE_N,
    TILE_K,
    b_cache_mod=2,
    xcd_swizzle=1,
    waves_per_eu=None,
    w_dtype="fp4",
    persist=False,
    topk=1,
):
    return compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=N_OUT,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        persist=persist,
        use_k16=a16wmix_use_k16(get_rocm_arch()),
    )


def flydsl_a16w4_gemm1(
    *,
    a_bf16,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=None,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    k_wave=1,
    b_nt=None,
    xcd_swizzle=0,
    gate_mode="separated",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
    w_dtype="fp4",
    w_layout="standard",
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="fp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (same preshuffle as mxfp4), ``w1_scale_u8`` groupwise bf16 in
    the ``(E, N_OUT, G//2, 2)`` bf16-pair layout (dword = n*(G//2)+group//2).
    ``"bf16"``: RAW bf16 W1 preshuffled ``shuffle_weight (16,16)``; ``w1_scale_u8`` unused.

    ``w_layout="standard"`` (default) consumes the N-major GGUU preshuffle. ``"guinterleave"``
    (mxfp4 only) consumes aiter's native GUGU stage1 W1+scale layout
    (``shuffle_weight_a16w4``/``shuffle_scale_a16w4``) directly, with no host relayout.

    ``a_bf16`` is bf16 ``[n_tokens, D_HIDDEN]``. Writes the bf16 intermediate
    ``[sorted_size, D_INTER]`` (by sorted position) into ``inter_sorted_bf16``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier (0=cached, 2=nt),
    ``xcd_swizzle`` -> XCD/HBM grid remap, ``k_wave`` -> intra-block slice-K ({1,2,4}).
    ``k_batch``/``gate_mode`` accepted for parity (only k_batch=1/separated supported).
    Tiles are CSV/registry-driven and always supplied by the caller; ``b_nt=None``
    falls back to the per-M U-shape (nt mid-band, cached at ends).

    ``situ_beta``/``situ_linear_beta`` (SiTUv2 only) are runtime f32 scalars (nothing baked).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    BM = tile_m
    TILE_K = tile_k
    TILE_N = tile_n
    # Tile config is fully CSV/registry-driven: the caller resolves tile_n/tile_k/
    # k_wave/b_nt/xcd_swizzle from the tuned kernelName1 and always passes them.
    b_cache_mod = (2 if (16 <= int(n_tokens) <= 1024) else 0) if b_nt is None else b_nt
    if D_HIDDEN % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_HIDDEN (K) % {TILE_K} == 0, got H={D_HIDDEN}"
        )
    if (2 * D_INTER) % 256 != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires 2*D_INTER % 256 == 0, got D_INTER={D_INTER}"
        )
    if D_INTER % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_INTER % TILE_N({TILE_N}) == 0, got D_INTER={D_INTER}"
        )

    launch = _get_compiled_gemm1_a16w4(
        BM,
        D_HIDDEN,
        D_INTER,
        NE,
        topk,
        TILE_N,
        TILE_K,
        act,
        b_cache_mod,
        xcd_swizzle,
        waves_per_eu,
        w_dtype,
        w_layout,
        k_wave,
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    # SiTUv2 beta/linear_beta + swiglu_limit -> runtime f32 scalars (host precomputes
    # reciprocals; no device rcp). swiglu_limit is the SiTUv2 clamp bound (+inf =
    # no clamp), matching the a8w4/mixed_moe situv2 clamp so a16w4 == a8w4.
    _beta = float(situ_beta)
    _lbeta = float(situ_linear_beta)
    if _beta <= 0.0 or _lbeta <= 0.0:
        raise ValueError(
            f"situ_beta/situ_linear_beta must be > 0, got {_beta!r}/{_lbeta!r}"
        )
    _run_compiled(
        launch,
        a_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        _beta,
        1.0 / _beta,
        _lbeta,
        1.0 / _lbeta,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def flydsl_a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=256,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=1,
    w_dtype="fp4",
    persist=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt=None`` keeps the per-M
    U-shape (cached at ends, nt mid-band).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")

    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
    _m = int(M_logical)
    # Tile config is CSV/registry-driven: the caller resolves tile_n/tile_k/b_nt/
    # xcd_swizzle from the tuned kernelName2 and always passes them.
    if D_INTER % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}"
        )
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}"
        )

    # B cache modifier per-token U-shape: cached (0) at both ends (small M reuse / large
    # M L2 residency), nt (2) mid-band (32..1024). Caller may override via b_nt.
    _b_cache_mod = (0 if (_m <= 16 or _m >= 2048) else 2) if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF; byte-identical when off): does NOT
    # close the E896 gap (padded launch's empty CTAs early-return ~free), kept as an
    # opt-in building block.
    _persist = False if persist is None else bool(persist)
    launch = _get_compiled_gemm2_a16w4(
        BM,
        NE,
        D_HIDDEN,
        D_INTER,
        TILE_N,
        TILE_K,
        _b_cache_mod,
        xcd_swizzle,
        waves_per_eu,
        w_dtype,
        _persist,
        topk,
    )
    grid = gemm2_a16w4_grid(
        BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks, persist=_persist
    )
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out
