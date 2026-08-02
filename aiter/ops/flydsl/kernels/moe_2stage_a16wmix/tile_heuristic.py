# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Built-in tile heuristic for the a16w4/a16wi4 fused MoE launchers.

``resolve_a16wmix_gemm{1,2}_config`` (backed by ``_default_tiles_fallback``) return
the tuned default tile geometry when the caller leaves tile args unset; never None.
Per-token CSV-tuned tiles now come from the standard get_2stage_cfgs kernelName
(parsed by _flydsl_stage{1,2}_wrapper), so no CSV parsing lives here.
"""


# ---------------------------------------------------------------------------
# Built-in tile heuristic (no CSV): the tuned defaults for the launchers.
# ---------------------------------------------------------------------------


def _default_tile_n(N, *, w_dtype="mxfp4"):
    """Adaptive N-tile default. mxfp4/bf16: 256 when N % 256 == 0 else 128 (aiter fat
    tile). int4 (bandwidth/grid-fill-bound): prefers 128, falling to 64 when N % 128 != 0.
    """
    if w_dtype == "int4":
        if N % 128 == 0:
            return 128
        return 64 if N % 64 == 0 else 128
    return 256 if N % 256 == 0 else 128


# Heuristic tile-config defaults for the a16w4/a16wi4 launchers (used when tile args
# are left at default and no tuned-CSV row applies). gemm1 uses this heuristic rather
# than the kimik3 CSV gemm1 tiles (which regress mid/high-M on the ported body); gemm2
# prefers the aiter CSV via ``resolve_a16w4_gemm2_config``.


def _default_tiles_fallback(*, D_HIDDEN, D_INTER, tokens, w_dtype, tile_m, stage):
    """Documented tile-config heuristic for one (shape, token, stage).

    Returns a tile-config dict. Assumes the caller left all tile args at their defaults
    (explicit caller overrides are applied afterwards, upstream).
    """
    BM = int(tile_m)
    _m = int(tokens)
    if stage == 1:
        TILE_K = 256
        k_wave = 1
        xcd_swizzle = 0
        b_nt = 2 if (16 <= _m <= 1024) else 0
        # mxfp4 high-token: shorter K-tiles + XCD remap from tok>=16 (K % 128 == 0).
        if w_dtype == "mxfp4" and _m >= 16 and D_HIDDEN % 128 == 0:
            TILE_K = 128
            xcd_swizzle = 1
        # TILE_N resolution (mirrors the old tile_n=None branch order).
        if w_dtype == "mxfp4" and _m <= 2 and D_INTER % 64 == 0:
            TILE_N = 64
            # tok<=2 4-way slice-K (needs K % (4*128) == 0).
            if D_HIDDEN % 512 == 0:
                TILE_K = 128
                k_wave = 4
        elif w_dtype == "int4" and BM == 64 and D_INTER % 64 == 0:
            TILE_N = 64
        elif w_dtype == "mxfp4" and D_INTER % 128 == 0:
            TILE_N = 128
        else:
            TILE_N = _default_tile_n(D_INTER, w_dtype=w_dtype)
        return {
            "tile_m": BM,
            "tile_n": TILE_N,
            "tile_k": TILE_K,
            "k_wave": k_wave,
            "xcd_swizzle": xcd_swizzle,
            "b_nt": b_nt,
        }
    # stage 2 (down-proj): fixed 4-wave N-split; tile_n=256/tile_k=256/xcd=1 defaults.
    b_nt = 0 if (_m <= 16 or _m >= 2048) else 2
    return {
        "tile_m": BM,
        "tile_n": 256,
        "tile_k": 256,
        "k_wave": 1,
        "xcd_swizzle": 1,
        "b_nt": b_nt,
    }


def resolve_a16wmix_gemm1_config(
    *, w_dtype, model_dim, inter_dim, experts, topk, tokens, tile_m
):
    """Resolve the gemm1 tile-config from the documented heuristic. Never returns None."""
    return _default_tiles_fallback(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        tokens=tokens,
        w_dtype=w_dtype,
        tile_m=tile_m,
        stage=1,
    )


def resolve_a16wmix_gemm2_config(
    *, w_dtype, model_dim, inter_dim, experts, topk, tokens, tile_m
):
    """Resolve the gemm2 tile-config from the documented heuristic. Never returns None."""
    return _default_tiles_fallback(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        tokens=tokens,
        w_dtype=w_dtype,
        tile_m=tile_m,
        stage=2,
    )
