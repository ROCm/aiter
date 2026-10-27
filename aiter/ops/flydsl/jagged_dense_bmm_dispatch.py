# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch layer for jagged_dense_bmm_broadcast_add.

Uniform winners come from ``jagged_dense_bmm_tuned.csv`` through
``AITER_CONFIG_JAGGED_DENSE_BMM``. The key is ``gfx``, ``cu_num``, K, N,
and a power-of-two bucket of the uniform grid. A miss uses
``analytic_config``. The public wrapper runs only on ``SUPPORTED_GFX``.
Every skew launch builds a compact TILE_MAP and uses the analytic config;
skew has no measured XCD rows.
"""

from __future__ import annotations

import csv

import flydsl.expr as fx
import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .kernels.jagged_dense_bmm_gen import BLOCK_M, BLOCK_N, jagged_dense_bmm
from .kernels.jdbba_skew_tile_map import build_tile_map_device_fused

__all__ = [
    "SUPPORTED_GFX",
    "analytic_config",
    "jagged_dense_bmm_dispatched",
    "resolve_config",
    "tile_bucket",
]

# Each entry has its own measured winners. gfx942 is MI300X; gfx950 is MI350.
# An arch absent from this tuple gets an empty table.
SUPPORTED_GFX = ("gfx942", "gfx950")

_DISPATCH_TABLES: dict[tuple, dict] = {}
_DISPATCH_CACHE: dict[tuple, dict] = {}

_XCD_C_SMALL_K = 32
_XCD_C_LARGE_K = 120
_XCD_W = 8
_XCD_C_IDENTITY = 1
_THREADS = 256
_BLOCK_K = 64
_SKEW_REMAP_MIN_GROUPS = 120
_SKEW_REMAP_MIN_K = 512


def _runtime_arch() -> str:
    """Live GPU name. ``get_gfx_runtime`` ignores ``GPU_ARCHS``."""
    return get_gfx_runtime().split(":", 1)[0]


def _require_supported_arch() -> str:
    arch = _runtime_arch()
    if arch not in SUPPORTED_GFX:
        supported = ", ".join(SUPPORTED_GFX)
        raise RuntimeError(f"jagged_dense_bmm supports {supported}; got {arch}")
    return arch


def tile_bucket(n_groups: int, max_seq_len: int, output_n: int) -> int:
    """Largest power of two not above the uniform grid.

    The grid is ``ceil(max_seq_len / BLOCK_M) * (N / BLOCK_N) * n_groups``.
    Nearby batches that quantize to the same power share one tuned row.
    """
    bm_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    n_blocks = output_n // BLOCK_N
    tiles = bm_tiles * n_blocks * n_groups
    if tiles <= 1:
        return 1
    return 1 << (tiles.bit_length() - 1)


def analytic_config(*, uniform_seqlen: bool, n_groups: int, reduction_k: int) -> dict:
    """Config used when the tuned table has no row for this launch."""
    cfg = {"threads": _THREADS, "block_k": _BLOCK_K, "xcd_w": _XCD_W}
    if not uniform_seqlen:
        if n_groups > _SKEW_REMAP_MIN_GROUPS or reduction_k >= _SKEW_REMAP_MIN_K:
            cfg["xcd_c"] = _XCD_C_SMALL_K
            return cfg
        cfg["xcd_c"] = _XCD_C_IDENTITY
        cfg["xcd_w"] = _XCD_C_IDENTITY
        return cfg
    cfg["xcd_c"] = _XCD_C_LARGE_K if reduction_k > 256 else _XCD_C_SMALL_K
    return cfg


def _load_dispatch_table() -> dict:
    """Uniform winners for the live ``gfx`` and ``cu_num``."""
    arch = _runtime_arch()
    cu = int(get_cu_num())
    path = AITER_CONFIGS.AITER_CONFIG_JAGGED_DENSE_BMM_FILE
    cache_key = (arch, cu, path)
    cached = _DISPATCH_TABLES.get(cache_key)
    if cached is not None:
        return cached
    winners: dict[tuple[int, int, int], dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("gfx") != arch or int(float(row["cu_num"])) != cu:
                continue
            if row.get("regime") != "uniform":
                continue
            winners[(int(row["K"]), int(row["N"]), int(row["tile_bucket"]))] = {
                "xcd_c": int(row["xcd_c"]),
                "xcd_w": int(row["xcd_w"]),
                "threads": int(row["threads"]),
                "block_k": int(row["block_k"]),
            }
    table = {"gfx": arch if winners else None, "winners": winners}
    _DISPATCH_TABLES[cache_key] = table
    return table


def _packed_rows(jagged) -> int:
    """Host upper bound on ``seq_offsets[-1]``.

    ``A.shape[0]`` is the packed length, or larger when ``A`` is
    over-allocated. Rows past the true end stay TILE_MAP sentinels.
    """
    rows = int(jagged.shape[0])
    if rows <= 0:
        raise ValueError(
            f"packed jagged rows must be a positive host extent, got {rows}"
        )
    return rows


def _as_torch_stream(stream, device):
    """One torch stream for the TILE_MAP prep and the GEMM."""
    if stream is None:
        return torch.cuda.current_stream(device)
    raw = stream.value if isinstance(stream, fx.Stream) else stream
    if raw is None:
        return torch.cuda.current_stream(device)
    if hasattr(raw, "cuda_stream"):
        return raw
    raise TypeError(f"unsupported stream {stream!r}")


def _config_valid(cfg: dict, *, reduction_k: int) -> bool:
    bk = cfg.get("block_k")
    if bk is None:
        bk = 64
    return not (reduction_k % bk != 0 or reduction_k // bk < 2)


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    uniform_seqlen: bool = True,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
) -> dict:
    bucket = tile_bucket(n_groups, max_seq_len, output_n)
    arch = _runtime_arch()
    cu = int(get_cu_num())
    key = (
        arch,
        cu,
        uniform_seqlen,
        reduction_k,
        output_n,
        bucket,
        n_groups > _SKEW_REMAP_MIN_GROUPS,
        xcd_c,
        xcd_w,
        block_k,
    )
    cached = _DISPATCH_CACHE.get(key)
    if cached is not None:
        return cached

    cfg = analytic_config(
        uniform_seqlen=uniform_seqlen,
        n_groups=n_groups,
        reduction_k=reduction_k,
    )
    if uniform_seqlen:
        entry = _load_dispatch_table()["winners"].get((reduction_k, output_n, bucket))
        if entry is not None and _config_valid(entry, reduction_k=reduction_k):
            cfg = dict(entry)

    if xcd_c is not None:
        cfg["xcd_c"] = int(xcd_c)
    if xcd_w is not None:
        cfg["xcd_w"] = int(xcd_w)
    if block_k is not None:
        cfg["block_k"] = int(block_k)

    _DISPATCH_CACHE[key] = cfg
    return cfg


def jagged_dense_bmm_dispatched(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream=None,
    uniform_seqlen: bool = True,
    # explicit overrides (None -> tuned row, else analytic_config)
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
):
    _require_supported_arch()
    output_n = B.shape[0] // n_groups
    reduction_k = B.shape[1]
    cfg = resolve_config(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
        uniform_seqlen=uniform_seqlen,
        xcd_c=xcd_c,
        xcd_w=xcd_w,
        block_k=block_k,
    )

    # Prep and the GEMM share this queue. A per-call TILE_MAP allocated on it
    # stays live until that queue finishes, including after this function returns.
    stream = _as_torch_stream(stream, SEQ_OFFSETS.device)

    if not uniform_seqlen:
        # Rebuild every call. The map bakes seq_start/seq_end, so an in-place
        # rewrite of SEQ_OFFSETS must not reuse a previous map.
        tile_map, ub = build_tile_map_device_fused(
            SEQ_OFFSETS,
            n_groups,
            _packed_rows(A),
            max_seq_len,
            block_m=BLOCK_M,
            stream=stream,
        )
        return jagged_dense_bmm(
            C,
            A,
            B,
            BIAS,
            SEQ_OFFSETS,
            n_groups,
            max_seq_len,
            stream=stream,
            xcd_c=cfg["xcd_c"],
            xcd_w=cfg["xcd_w"],
            uniform_seqlen=False,
            block_k=cfg["block_k"],
            threads=cfg["threads"],
            tile_map=tile_map,
            total_occ_tiles=ub,
        )

    return jagged_dense_bmm(
        C,
        A,
        B,
        BIAS,
        SEQ_OFFSETS,
        n_groups,
        max_seq_len,
        stream=stream,
        xcd_c=cfg["xcd_c"],
        xcd_w=cfg["xcd_w"],
        uniform_seqlen=True,
        block_k=cfg["block_k"],
        threads=cfg["threads"],
    )
