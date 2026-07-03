# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Runtime lookup for FlyDSL a8w8 bpreshuffle tuned tile configs.

Mirrors the aiter CK/asm convention (see ``get_GEMM_config_with_quant_type`` in
``aiter/ops/gemm_op_a8w8.py``): tuned results live in
``aiter/configs/flydsl_a8w8_bpreshuffle_tuned_gemm.csv`` and are looked up at
runtime by ``(cu_num, M, N, K, q_dtype_w)``. When no exact ``M`` row exists we
fall back to the nearest tuned ``M`` for the same ``(N, K, q_dtype_w)``; when
nothing matches, the caller should fall back to ``default_kernels_dict``.

CSV columns:
    cu_num,M,N,K,q_dtype_w,tile_m,tile_n,tile_k,
    lds_stage,use_cshuffle_epilog,use_async_copy,waves_per_eu
"""
import functools
import os

import pandas as pd

from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_cu_num

_TILE_FIELDS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "lds_stage",
    "use_cshuffle_epilog",
    "use_async_copy",
    "waves_per_eu",
)

DEFAULT_TUNED_FILE = os.path.join(
    AITER_ROOT_DIR, "aiter", "configs", "flydsl_a8w8_bpreshuffle_tuned_gemm.csv"
)


@functools.lru_cache(maxsize=8)
def _load_table(tuned_file: str):
    """Return {(cu_num, M, N, K, q_dtype_w): {tile fields}} plus an index of
    tuned M values per (cu_num, N, K, q_dtype_w) for nearest-M fallback."""
    exact = {}
    by_nk = {}
    if not os.path.exists(tuned_file):
        return exact, by_nk
    df = pd.read_csv(tuned_file).drop_duplicates()
    for _, r in df.iterrows():
        key = (
            int(r["cu_num"]),
            int(r["M"]),
            int(r["N"]),
            int(r["K"]),
            str(r["q_dtype_w"]),
        )
        cfg = {f: int(r[f]) for f in _TILE_FIELDS}
        exact[key] = cfg
        by_nk.setdefault((key[0], key[2], key[3], key[4]), []).append((key[1], cfg))
    for k in by_nk:
        by_nk[k].sort(key=lambda x: x[0])
    return exact, by_nk


def get_flydsl_gemm_config(M, N, K, q_dtype_w, tuned_file: str = DEFAULT_TUNED_FILE):
    """Look up a tuned FlyDSL tile for (M, N, K, q_dtype_w) on the current GPU.

    Returns a dict with keys in ``_TILE_FIELDS``, or ``None`` if not tuned.
    """
    exact, by_nk = _load_table(tuned_file)
    cu_num = get_cu_num()
    qw = str(q_dtype_w)
    cfg = exact.get((cu_num, int(M), int(N), int(K), qw))
    if cfg is not None:
        return cfg
    cands = by_nk.get((cu_num, int(N), int(K), qw))
    if cands:
        return min(cands, key=lambda mc: abs(mc[0] - int(M)))[1]
    return None
