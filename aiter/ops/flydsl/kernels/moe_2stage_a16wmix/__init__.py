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
from .csv_dispatch import (
    _default_tile_n,
    pick_a16w4_config,
    resolve_a16w4_gemm2_config,
    resolve_a16wmix_gemm1_config,
    resolve_a16wmix_gemm2_config,
)

__all__ = [
    "a16wi4_recommend_block_m",
    "a16wi4_scale_to_kernel_layout",
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
    "pick_a16w4_config",
    "resolve_a16w4_gemm2_config",
    "resolve_a16wmix_gemm1_config",
    "resolve_a16wmix_gemm2_config",
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
    w_dtype="mxfp4",
    w_layout="standard",
    k_wave=1,
):
    # native mxfp4 guinterleave routes through the shared compile entry;
    # FlyDSL-native standard/int4/bf16 call the builder directly.
    if w_dtype == "mxfp4" and w_layout == "guinterleave":
        from aiter.ops.flydsl.moe_kernels import compile_flydsl_moe_stage1

        return compile_flydsl_moe_stage1(
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tile_m=BM,
            tile_n=TILE_N,
            tile_k=TILE_K,
            doweight_stage1=False,
            a_dtype="bf16",
            b_dtype="mxfp4",
            out_dtype="bf16",
            act=act,
            waves_per_eu=waves_per_eu,
            b_nt=b_cache_mod,
            xcd_swizzle=xcd_swizzle,
            k_wave=k_wave,
        )
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
    w_dtype="mxfp4",
    persist=False,
    topk=1,
):
    # native mxfp4 down-proj routes through the shared compile entry;
    # the persist opt-in stays on the direct path.
    if w_dtype == "mxfp4" and not persist:
        from aiter.ops.flydsl.moe_kernels import compile_flydsl_moe_stage2

        return compile_flydsl_moe_stage2(
            model_dim=N_OUT,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tile_m=BM,
            tile_n=TILE_N,
            tile_k=TILE_K,
            doweight_stage2=True,
            a_dtype="bf16",
            b_dtype="mxfp4",
            out_dtype="bf16",
            b_nt=b_cache_mod,
            xcd_swizzle=xcd_swizzle,
            waves_per_eu=waves_per_eu,
        )
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


def a16wi4_recommend_block_m(tokens, experts, topk, *, base_block_m=32):
    """Recommend the routing/gemm1 block_m (tile_m) for the a16wi4 (int4-W) stage1.

    int4 gemm1 is W-load bound (W1 re-fetched per padded m-block). At the fill point
    where each expert has exactly two half-full 32-row m-blocks, block_m=64 collapses
    them into one full 64-row block, halving W1 HBM re-reads (~2.27x fewer misses, ~7%
    stage1). Outside that band 64 wastes padding or halves grid parallelism.

    Decides on ceil(tokens*topk/experts / base_block_m) (avg padded m-blocks/expert ==
    aiter's estimated_m_per_expert): return 2*base_block_m iff that count == 2.
    base_block_m==32 only; other block_m pass through.

    IMPORTANT: block_m sizes the moe_sorting padding, so the caller MUST build the
    routing buffers with the SAME block_m passed to gemm1 (this is a dispatcher-side
    recommendation, not a gemm1-internal override).
    """
    if int(base_block_m) != 32 or int(experts) <= 0:
        return int(base_block_m)
    routes = int(tokens) * int(topk)
    # ceil(avg routes-per-expert / 32) == avg padded m-blocks/expert at block_m=32.
    m_blocks_32 = -(-routes // (int(experts) * 32))
    return 64 if m_blocks_32 == 2 else int(base_block_m)


def a16wi4_scale_to_kernel_layout(scale_ng):
    """Re-layout a logical int4 scale ``[E, N, G]`` into the ``(E, N, G//2, 2)``
    bf16-pair layout the kernel expects (dword = n*(G//2) + group//2, even/odd group ->
    lo/hi bf16). ``G`` must be even; input is already N-major.
    """
    E, N, G = scale_ng.shape
    assert G % 2 == 0, f"num_groups must be even for bf16-pair packing, got {G}"
    s = scale_ng.to(torch.bfloat16).contiguous().view(E, N, G // 2, 2).contiguous()
    return s


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
    w_dtype="mxfp4",
    w_layout="standard",
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="mxfp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (same preshuffle as mxfp4), ``w1_scale_u8`` groupwise bf16 in
    the ``(E, N_OUT, G//2, 2)`` layout (see :func:`a16wi4_scale_to_kernel_layout`).
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
    ``tile_n=None`` picks the largest N tile dividing D_INTER. ``b_nt=None`` uses the
    per-M U-shape (nt mid-band, cached at ends).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    BM = tile_m
    TILE_K = tile_k
    _m = int(n_tokens)
    TILE_N = tile_n
    # Per-token tile config from the documented heuristic. Fills only the tile args the
    # caller left at default; explicit caller overrides always win.
    _o = resolve_a16wmix_gemm1_config(
        w_dtype=w_dtype,
        model_dim=D_HIDDEN,
        inter_dim=D_INTER,
        experts=NE,
        topk=topk,
        tokens=_m,
        tile_m=BM,
    )
    if b_nt is None:
        b_nt = _o["b_nt"]
    if xcd_swizzle == 0:
        xcd_swizzle = _o["xcd_swizzle"]
    # The slice-K config (k_wave > 1: the tok<=2 4-way split) is coupled to the
    # tile_n=64 branch and to tile_k/k_wave being at defaults -- apply it as one
    # unit only when the caller left tile_n unset. The k_wave==1 tile_k bump
    # (tok>=16 shorter K-tiles) is independent of tile_n.
    if _o["k_wave"] > 1:
        if TILE_N is None and tile_k == 256 and k_wave == 1:
            TILE_K = _o["tile_k"]
            k_wave = _o["k_wave"]
    elif tile_k == 256:
        TILE_K = _o["tile_k"]
    if TILE_N is None:
        TILE_N = _o["tile_n"]
    b_cache_mod = (2 if (16 <= _m <= 1024) else 0) if b_nt is None else b_nt
    if TILE_N is None:
        TILE_N = _default_tile_n(D_INTER, w_dtype=w_dtype)
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
    w_dtype="mxfp4",
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
    # Per-token tile config from the documented heuristic. Fills only the tile args the
    # caller left at default; explicit caller overrides always win. gemm2 defaults are
    # tile_n=256/tile_k=256/xcd_swizzle=1 (fixed 4-wave N-split, no k_wave).
    _o = resolve_a16wmix_gemm2_config(
        w_dtype=w_dtype,
        model_dim=D_HIDDEN,
        inter_dim=D_INTER,
        experts=NE,
        topk=topk,
        tokens=_m,
        tile_m=BM,
    )
    # gemm2 tile_n is not token-dependent (fixed 256 default; explicit tile_n=None
    # means adaptive _default_tile_n, handled below), so only tile_k/b_nt/xcd are
    # filled from the resolver here.
    if tile_k == 256:
        TILE_K = _o["tile_k"]
    if b_nt is None:
        b_nt = _o["b_nt"]
    if xcd_swizzle == 1:
        xcd_swizzle = _o["xcd_swizzle"]
    if TILE_N is None:
        # Adaptive default: largest N tile dividing model_dim (int4 prefers 128).
        TILE_N = _default_tile_n(D_HIDDEN, w_dtype=w_dtype)
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
