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

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid

__all__ = [
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
    "mxfp4_scale_ref_exponent",
]


def mxfp4_scale_ref_exponent(scale_u8, rows, k_groups):
    """Per-row max e8m0 byte of a preshuffled MXFP4 scale, for the gfx942 fp8 mode.

    ``scale_u8`` is the scale exactly as gemm1/gemm2 read it (the standard layout,
    ``shuffle_scale_a16w4(s, E, False)``): ``rows`` weight rows (E * N) by G e8m0
    groups, G = K/32 padded to a multiple of 8 with 0x7F.
    Viewed as (rows/32, G/8, 4, 16, 2, 2) its axes are (row // 32, group // 8,
    group % 4, row % 16, (group // 4) % 2, (row // 16) % 2), the order
    ``make_b_loader.load_b_scale`` indexes. Pad groups (>= ``k_groups``) are excluded.

    Returns ``(s_ref, max_residual)``: s_ref is ``[rows]`` uint8 (logical row order),
    max_residual the largest ``s_ref - s`` over real groups. The fp8 decode is exact
    only up to ``A16W4_FP8_MAX_RESIDUAL``; the caller must not use s_ref beyond that.
    """
    flat = scale_u8.reshape(-1)
    groups = flat.numel() // rows
    if rows % 32 or groups % 8 or groups * rows != flat.numel() or groups < k_groups:
        raise ValueError(
            f"scale with {flat.numel()} bytes is not {rows} rows x 8k groups "
            f"(k_groups={k_groups})"
        )
    v = flat.view(torch.uint8).view(rows // 32, groups // 8, 4, 16, 2, 2)
    dev = flat.device
    g = (
        torch.arange(groups // 8, device=dev).view(-1, 1, 1) * 8
        + torch.arange(4, device=dev).view(1, -1, 1)
        + torch.arange(2, device=dev).view(1, 1, -1) * 4
    )  # [G/8, 4, 2]: logical group of (group // 8, group % 4, (group // 4) % 2)
    real = (g < k_groups).view(1, groups // 8, 4, 1, 2, 1)
    sref = torch.where(real, v, 0).amax(dim=(1, 2, 4))  # [rows/32, 16, 2]
    neg_r = torch.where(real, sref[:, None, None, :, None, :] - v, 0)
    max_residual = int(neg_r.amax().item())
    return sref.permute(0, 2, 1).reshape(rows).contiguous(), max_residual


# gfx942 (CDNA3) lacks K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32 -> K=16 MFMA +
# scalar dequant fallback; gfx950 (CDNA4) uses the K=32 path. compile_gemm{1,2}_a16w4_port
# are @functools.cache'd, so building through them directly (all-keyword) shares one
# compile with moe_kernels.py's AOT/runtime direct calls.


def _non_default_a_dtype(a_dtype):
    # Passed only when not bf16: an explicit a_dtype="bf16" would be a different
    # functools.cache key from moe_kernels.py's calls and compile the kernel twice.
    return {} if a_dtype == "bf16" else {"a_dtype": a_dtype}


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
    a_dtype="bf16",
    a_scale=None,
    w1_sref=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="fp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (``pack_int8_to_packed_int4(shuffle_weight(w, (16,16)))``),
    ``w1_scale_u8`` groupwise bf16 in the ``shuffle_scale_for_int4`` ``(E, G//2, N_OUT, 2)``
    bf16-pair layout (dword = e*(G//2*N_OUT) + (group//2)*N_OUT + n).
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
    Tiles are CSV/registry-driven and always supplied by the caller; ``b_nt`` is
    the W-load cache modifier (0=cached, 2=nt), taken as-is (nt when unset).

    ``situ_beta``/``situ_linear_beta`` (SiTUv2 only) are runtime f32 scalars (nothing baked).

    ``a_dtype="fp8"`` (gfx942, mxfp4 W, standard layout): ``a_bf16`` holds per-token
    e4m3fnuz activations, ``a_scale`` their ``[n_tokens]`` f32 scales and ``w1_sref``
    the ``[E, 2*D_INTER]`` u8 reference exponents of ``w1_scale_u8``
    (:func:`mxfp4_scale_ref_exponent`); the GEMM runs on fp8 MFMA.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if a_dtype == "fp8" and (a_scale is None or w1_sref is None):
        raise ValueError("a16w4 gemm1 a_dtype='fp8' needs a_scale and w1_sref")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    BM = tile_m
    TILE_K = tile_k
    TILE_N = tile_n
    # Tile config is fully CSV/registry-driven: the caller resolves tile_n/tile_k/
    # k_wave/b_nt/xcd_swizzle from the tuned kernelName1 and always passes them.
    b_cache_mod = 2 if b_nt is None else b_nt
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

    launch = compile_gemm1_a16w4_port(
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
        rocm_arch=str(get_rocm_arch()),
        **_non_default_a_dtype(a_dtype),
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
    _fp8_args = (a_scale.data_ptr(), w1_sref.data_ptr()) if a_dtype == "fp8" else ()
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
        *_fp8_args,
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
    epilog="atomic",
    a_dtype="bf16",
    a_scale=None,
    w2_sref=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt`` is the W-load cache
    modifier (0=cached, 2=nt), taken as-is (cached when unset).

    ``a_dtype="fp8"`` (gfx942, mxfp4 W, atomic): ``inter_sorted_bf16`` holds the
    intermediate quantized per sorted row to e4m3fnuz, ``a_scale`` its
    ``[max_sorted]`` f32 row scales and ``w2_sref`` the ``[E, D_HIDDEN]`` u8 reference
    exponents of ``w2_scale_u8``; the GEMM runs on fp8 MFMA.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")
    if a_dtype == "fp8" and (a_scale is None or w2_sref is None):
        raise ValueError("a16w4 gemm2 a_dtype='fp8' needs a_scale and w2_sref")
    if a_dtype == "fp8" and a_scale.numel() < max_sorted:
        raise ValueError(
            f"a16w4 gemm2 fp8 a_scale has {a_scale.numel()} rows, needs {max_sorted}"
        )

    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
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

    # W-load cache modifier (0=cached, 2=nt): CSV/registry-driven via b_nt, taken
    # as-is (cached when unset).
    _b_cache_mod = 0 if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF; byte-identical when off): does NOT
    # close the E896 gap (padded launch's empty CTAs early-return ~free), kept as an
    # opt-in building block.
    _persist = False if persist is None else bool(persist)
    launch = compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=_b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        persist=_persist,
        rocm_arch=str(get_rocm_arch()),
        epilog=epilog,
        topk=topk,
        **_non_default_a_dtype(a_dtype),
    )
    grid = gemm2_a16w4_grid(
        BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks, persist=_persist
    )
    _fp8_args = (a_scale.data_ptr(), w2_sref.data_ptr()) if a_dtype == "fp8" else ()
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
        *_fp8_args,
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out
