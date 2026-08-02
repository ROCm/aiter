# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Built-in gemm1 tile heuristic for the a16w4 (bf16 A x MXFP4 W) fused MoE launcher.

``resolve_a16wmix_gemm{1,2}_config`` return the tuned default tile geometry when the
launcher leaves tile args unset; never None. gemm1 uses this heuristic rather than the
kimik3 CSV gemm1 tiles (which regress mid/high-M on the ported body); gemm2 tiles come
from the get_2stage_cfgs kernelName. mxfp4 only -- a16wi4 (int4) dispatches via its own
tuned CSV (get_2stage_cfgs), not this heuristic.
"""


def _default_tile_n(N):
    """Adaptive N-tile: 256 when N % 256 == 0 else 128 (aiter fat tile)."""
    return 256 if N % 256 == 0 else 128


def _default_tiles_fallback(*, D_HIDDEN, D_INTER, tokens, tile_m, stage):
    """Documented mxfp4 tile-config heuristic for one (shape, token, stage).

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
        # high-token: shorter K-tiles + XCD remap from tok>=16 (K % 128 == 0).
        if _m >= 16 and D_HIDDEN % 128 == 0:
            TILE_K = 128
            xcd_swizzle = 1
        if _m <= 2 and D_INTER % 64 == 0:
            TILE_N = 64
            # tok<=2 4-way slice-K (needs K % (4*128) == 0).
            if D_HIDDEN % 512 == 0:
                TILE_K = 128
                k_wave = 4
        elif D_INTER % 128 == 0:
            TILE_N = 128
        else:
            TILE_N = _default_tile_n(D_INTER)
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


def resolve_a16wmix_gemm1_config(*, model_dim, inter_dim, experts, topk, tokens, tile_m):
    """gemm1 tile-config from the mxfp4 heuristic. Never returns None."""
    return _default_tiles_fallback(
        D_HIDDEN=model_dim, D_INTER=inter_dim, tokens=tokens, tile_m=tile_m, stage=1
    )


def resolve_a16wmix_gemm2_config(*, model_dim, inter_dim, experts, topk, tokens, tile_m):
    """gemm2 tile-config from the mxfp4 heuristic. Never returns None."""
    return _default_tiles_fallback(
        D_HIDDEN=model_dim, D_INTER=inter_dim, tokens=tokens, tile_m=tile_m, stage=2
    )
