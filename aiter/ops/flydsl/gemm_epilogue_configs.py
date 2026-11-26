# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tile configs of the gfx950 ``gemm_a16w16`` FlyDSL GEMM with an epilogue.

Rows come from the ``flydsl_gemm_a16w16_epilogue_tuned`` family
(``AITER_CONFIGS.AITER_CONFIG_GEMM_A16W16_EPILOGUE_FILE``: the header-only
canonical CSV merged with ``model_configs/*flydsl_gemm_a16w16_epilogue_tuned*``),
written by ``csrc/flydsl_gemm_epilogue/gemm_epilogue_tune.py``. A shape is keyed
on the untuned columns ``M, N, K, layout, epilogue, dtype, outdtype`` plus
``gfx`` and ``cu_num``. The token count is M, or K for the ``tn`` weight
gradient; a lookup takes the row of the smallest tuned token count at or above
it (else the largest), so one sweep covers every batch. Shapes without a row,
or whose row the kernel cannot run, get ``default_config``.
"""

import functools

import torch

KEY_COLUMNS = ["M", "N", "K", "layout", "epilogue", "dtype", "outdtype"]
TILE_COLUMNS = [
    "block_m",
    "block_n",
    "block_k",
    "stages",
    "split_k",
    "m_waves",
    "n_waves",
    "k_waves",
    "group_m",
    "hti",
]
TUNED_COLUMNS = [
    "gfx",
    "cu_num",
    *KEY_COLUMNS,
    "libtype",
    *TILE_COLUMNS,
    "us",
    "kernelName",
    "err_ratio",
    "tflops",
    "bw",
]
# Modes whose tiles transfer when a table has no row of their own: the safe
# and softplus gates read and write the same bytes, and SIGMOID is SIGMOID_MUL
# without the aux read.
_MODE_FALLBACK = {1: 5, 5: 1, 2: 6, 6: 2, 3: 4}
_SPLIT_K_MAX_TILES = 2048


def _ceil(a, b):
    return (a + b - 1) // b


def token_axis(layout):
    """The GEMM dim that carries the tokens: K for the ``tn`` weight gradient."""
    return "K" if layout == "tn" else "M"


def config_name(cfg):
    kind = "pht" if cfg["hti"] else "pft"
    return (
        f"t{cfg['block_m']}x{cfg['block_n']}x{cfg['block_k']}x{cfg['stages']}"
        f"_ks{cfg['split_k']}_w{cfg['m_waves']}x{cfg['n_waves']}x{cfg['k_waves']}"
        f"_gm{cfg['group_m']}_{kind}"
    )


def _cfg(bm, bn, m_waves, n_waves, stages=2, split_k=1, group_m=0, hti=False):
    return {
        "block_m": bm,
        "block_n": bn,
        "block_k": 64,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": 1,
        "group_m": group_m,
        "hti": hti,
    }


def default_config(m, n, k, layout, epilogue, cu_num):
    """Untuned fallback, from the bf16 FlyDSL hgemm rows tuned on MI350X."""
    if layout == "tn":
        # weight gradient: K = tokens; HTI needs an even count of 64-wide K tiles
        if k % 128 == 0:
            return _cfg(256, 256, 2, 4, hti=True, group_m=4)
        return _cfg(128, 256, 2, 4)
    if k <= 256:
        # skinny K (the f_b decay gate): bound by the fp32 store, so enough
        # workgroups to cover every CU; the backward modes need full tiles
        if m <= 1024:
            return _cfg(64, 128, 1, 4)
        return _cfg(128, 128, 2, 4)
    if m >= 1024:
        return _cfg(256, 256, 2, 4, hti=True, group_m=4 if m >= 2048 else 0)
    # small M, large K: split K until the grid covers the CUs once
    tiles = _ceil(m, 128) * _ceil(n, 128)
    split_k = 1
    if tiles <= _SPLIT_K_MAX_TILES:
        split_k = max(1, min(8, round(cu_num / tiles), k // (64 * 8)))
    return _cfg(128, 128, 2, 2, split_k=split_k)


def to_user_kwargs(cfg):
    kw = {c: int(cfg[c]) for c in TILE_COLUMNS if c != "hti"}
    kw["use_half_tile_interleaved"] = bool(int(cfg["hti"]))
    return kw


def _dtype_name(dtype):
    return str(dtype)


@functools.lru_cache(maxsize=1)
def _table():
    """{(gfx, cu_num, layout, epilogue, dtype, outdtype, fixed dims): [(tokens, cfg)]}."""
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS

    path = AITER_CONFIGS.AITER_CONFIG_GEMM_A16W16_EPILOGUE_FILE
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    table = {}
    for row in df.to_dict("records"):
        axis = token_axis(row["layout"])
        fixed = tuple(int(row[d]) for d in ("M", "N", "K") if d != axis)
        key = (
            row["gfx"],
            int(row["cu_num"]),
            row["layout"],
            int(row["epilogue"]),
            row["dtype"],
            row["outdtype"],
            fixed,
        )
        cfg = {c: int(row[c]) for c in TILE_COLUMNS}
        table.setdefault(key, []).append((int(row[axis]), cfg))
    for rows in table.values():
        rows.sort(key=lambda r: r[0])
    return table


def lookup(m, n, k, layout, epilogue, dtype, outdtype, gfx, cu_num):
    """Tuned config of this shape, or None."""
    axis = token_axis(layout)
    dims = {"M": m, "N": n, "K": k}
    tokens = dims[axis]
    fixed = tuple(dims[d] for d in ("M", "N", "K") if d != axis)
    for mode in (epilogue, _MODE_FALLBACK.get(epilogue)):
        if mode is None:
            continue
        rows = _table().get(
            (
                gfx,
                cu_num,
                layout,
                mode,
                _dtype_name(dtype),
                _dtype_name(outdtype),
                fixed,
            )
        )
        if rows:
            for t, cfg in rows:
                if t >= tokens:
                    return dict(cfg)
            return dict(rows[-1][1])
    return None


def select_config(m, n, k, layout, epilogue, dtype, outdtype, validate):
    """Tuned row if present and runnable (``validate(user_kwargs)``), else the
    default. Returns ``(user_kwargs, source)``."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    cu_num = get_cu_num()
    try:
        cfg = lookup(
            m, n, k, layout, epilogue, dtype, outdtype, get_gfx_runtime(), cu_num
        )
    except (OSError, KeyError, ValueError):
        cfg = None
    if cfg is not None:
        kw = _sanitize(to_user_kwargs(cfg), m, n, k, layout, epilogue)
        if validate(kw):
            return kw, "tuned"
    kw = _sanitize(
        to_user_kwargs(default_config(m, n, k, layout, epilogue, cu_num)),
        m,
        n,
        k,
        layout,
        epilogue,
    )
    return kw, "default"


def _sanitize(kw, m, n, k, layout, epilogue):
    """Adjust a row tuned at another token count to this one."""
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_BWD_MODES

    kw = dict(kw)
    if kw["use_half_tile_interleaved"] and (
        epilogue in GEMM_EPI_BWD_MODES or kw["split_k"] > 1 or k % (2 * kw["block_k"])
    ):
        # HTI takes neither K tails nor the column-sum epilogues
        kw.update(use_half_tile_interleaved=False, block_m=128, block_n=256)
        kw.update(m_waves=2, n_waves=4)
    if (
        kw["split_k"] > 1
        and _ceil(m, kw["block_m"]) * _ceil(n, kw["block_n"]) > _SPLIT_K_MAX_TILES
    ):
        kw["split_k"] = 1
    return kw


def candidate_configs(m, n, k, layout, epilogue):
    """Tile sweep of the tuner, the default first."""
    from aiter.jit.utils.chip_info import get_cu_num

    seen, out = set(), []

    def add(cfg):
        key = tuple(cfg[c] for c in TILE_COLUMNS)
        if key not in seen:
            seen.add(key)
            out.append(cfg)

    add(default_config(m, n, k, layout, epilogue, get_cu_num()))
    bwd = epilogue in (2, 6)
    tiles = [
        (32, 128, 1, 4),
        (64, 64, 1, 4),
        (64, 128, 1, 4),
        (64, 256, 1, 4),
        (128, 64, 2, 2),
        (128, 128, 2, 2),
        (128, 128, 2, 4),
        (128, 256, 2, 4),
        (256, 128, 2, 4),
    ]
    big_k = k > 256 and layout != "tn" or layout == "tn" and k >= 2048
    for bm, bn, mw, nw in tiles:
        for stages in (2, 3) if big_k else (2,):
            splits = (1,)
            if big_k and m * n <= 2048 * 12288:
                splits = (1, 2, 3, 4, 6, 8)
            for split_k in splits:
                add(_cfg(bm, bn, mw, nw, stages=stages, split_k=split_k))
    if not bwd and (layout == "tn" or k > 256):
        for gm in (0, 4, 8):
            add(_cfg(256, 256, 2, 4, hti=True, group_m=gm))
            add(_cfg(128, 128, 2, 2, hti=True, group_m=gm))
    return out


def clear_cache():
    _table.cache_clear()


def torch_dtype(name):
    return {"torch.bfloat16": torch.bfloat16, "torch.float16": torch.float16}.get(
        name, torch.float32
    )
