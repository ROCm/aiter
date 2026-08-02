# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tile-config resolution for the a16w4/a16wi4 fused MoE launchers.

Two config sources, both consumed by :mod:`__init__`'s ``flydsl_a16w4_gemm{1,2}``:

  - ``resolve_a16wmix_gemm{1,2}_config``: the built-in documented tile heuristic
    (``_default_tiles_fallback``), used by default. Never returns None.
  - ``resolve_a16w4_gemm{1,2}_config``: aiter's tuned fmoe CSV (opt-in / gemm2),
    parsed from the ``abf16_w*`` kernelName tile tags. Returns None on no match so
    the caller falls back to the heuristic.
"""

import csv
import functools
import os
import re

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


# ---------------------------------------------------------------------------
# aiter tuned-CSV config: parse the abf16_w* kernelName tile tags.
# ---------------------------------------------------------------------------

_A16W4_TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")
_A16W4_W_RE = re.compile(r"_w(\d+)")
_A16W4_XCD_RE = re.compile(r"_xcd(\d+)")
_A16W4_BNT_RE = re.compile(r"_bnt(\d+)")
_A16W4_KW_RE = re.compile(r"_kw(\d+)")


def _decode_a16w4_kname(kname):
    """Decode an ``abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict, or None."""
    m = _A16W4_TILE_RE.search(kname)
    if m is None:
        return None
    tile_m, tile_n, tile_k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    w = _A16W4_W_RE.search(kname)
    xcd = _A16W4_XCD_RE.search(kname)
    bnt = _A16W4_BNT_RE.search(kname)
    kw = _A16W4_KW_RE.search(kname)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "b_nt": int(bnt.group(1)) if bnt else 2,  # aiter default 2 when token absent
        "waves_per_eu": int(w.group(1)) if w else None,
        "xcd_swizzle": int(xcd.group(1)) if xcd else 0,
        "k_wave": int(kw.group(1)) if kw else 1,
    }


@functools.cache
def _load_a16w4_csv(csv_path):
    """Parse the tuned CSV into {(model_dim,inter,E,topk,stage,tokens): cfg}."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key_shape = (
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    int(row["expert"]),
                    int(row["topk"]),
                    int(row["token"]),
                )
            except (KeyError, ValueError):
                continue
            for stage, col in ((1, "kernelName1"), (2, "kernelName2")):
                kname = row.get(col, "")
                # bf16-A rows across all weight formats (fp4/int4/bf16); tile geometry only.
                if not any(
                    w in kname for w in ("abf16_wfp4", "abf16_wint4", "abf16_wbf16")
                ):
                    continue
                cfg = _decode_a16w4_kname(kname)
                if cfg is not None:
                    table[key_shape + (stage,)] = cfg
    return table


def pick_a16w4_config(csv_path, *, model_dim, inter_dim, experts, topk, tokens, stage):
    """Return aiter's tuned tile-config for one (shape, tokens, stage), or None.

    Exact ``tokens`` row if present, else nearest tuned token (largest <= requested,
    or smallest). ``stage`` is 1 (gemm1) or 2 (gemm2).
    """
    table = _load_a16w4_csv(csv_path)
    exact = table.get((model_dim, inter_dim, experts, topk, tokens, stage))
    if exact is not None:
        return exact
    cand = sorted(
        t
        for (md, i, e, k, t, s) in table
        if (md, i, e, k, s) == (model_dim, inter_dim, experts, topk, stage)
    )
    if not cand:
        return None
    le = [t for t in cand if t <= tokens]
    pick = le[-1] if le else cand[0]
    return table[(model_dim, inter_dim, experts, topk, pick, stage)]


# Candidate locations for aiter's tuned fp4 fmoe CSV (env override wins).
_A16W4_CSV_ENV = "FLYDSL_A16W4_TUNED_CSV"
_A16W4_CSV_CANDIDATES = (
    "/root/aiter/aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv",
)


@functools.cache
def _default_a16w4_csv_path():
    """Locate aiter's tuned fp4 fmoe CSV (``FLYDSL_A16W4_TUNED_CSV`` overrides), or None."""
    env = os.environ.get(_A16W4_CSV_ENV)
    if env:
        return env if os.path.isfile(env) else None
    for p in _A16W4_CSV_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def resolve_a16w4_gemm2_config(
    *, model_dim, inter_dim, experts, topk, tokens, csv_path=None
):
    """Resolve the per-token gemm2 tile-config from the tuned CSV.

    gemm2 has no k_wave (fixed 4-wave N-split). Requires D_INTER % tile_k == 0 and
    model_dim % tile_n == 0; a row violating either is skipped (None -> adaptive default).
    """
    path = csv_path or _default_a16w4_csv_path()
    if path is None:
        return None
    cfg = pick_a16w4_config(
        path,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=2,
    )
    if cfg is None:
        return None
    tile_n = int(cfg["tile_n"])
    tile_k = int(cfg["tile_k"])
    if model_dim % tile_n != 0 or inter_dim % tile_k != 0:
        return None
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": cfg.get("waves_per_eu"),
        "xcd_swizzle": int(cfg.get("xcd_swizzle", 1)),
        "b_nt": int(cfg["b_nt"]),
        "_note": "",
    }
