# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch layer for jagged_dense_bmm_broadcast_add.

Resolves per-shape kernel config from ``jagged_dense_bmm_dispatch.json``
(explicit override -> table winner -> defaults). Uniform launches use the
plain grid. Every skew launch builds a compact TILE_MAP and runs the
compact kernel. Tuning rationale lives in the JSON; skew XCD remap is
gated in ``_skew_compact_xcd``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from .kernels.jagged_dense_bmm_gen import BLOCK_M, jagged_dense_bmm
from .kernels.jdbba_skew_tile_map import build_tile_map_device_fused

__all__ = [
    "jagged_dense_bmm_dispatched",
    "resolve_config",
    "shape_id",
]

_DISPATCH_TABLE: dict | None = None
_DISPATCH_CACHE: dict[tuple, dict] = {}

SKEW_COMPACT_XCD_C = 32
SKEW_COMPACT_XCD_W = 8

_SCHEMA_DEFAULTS = {
    "xcd_c": None,
    "xcd_w": None,
    "skew_xcd_c": None,
    "skew_xcd_w": None,
    "block_k": None,
    "threads": None,
}


def _dispatch_json_paths() -> tuple[Path, ...]:
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_JSON")
    if env:
        return (Path(env),)
    return (Path(__file__).resolve().parent / "jagged_dense_bmm_dispatch.json",)


def _detect_arch() -> str | None:
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_ARCH")
    if env:
        return env
    return get_rocm_arch()


def _select_arch_section(data: dict, arch: str | None) -> dict:
    by_arch = data.get("by_arch")
    if not isinstance(by_arch, dict):
        return data  # legacy flat schema (e.g. an env-var override file)
    if arch and arch in by_arch:
        return by_arch[arch]
    if arch:
        for sect in by_arch.values():
            if sect.get("gfx") == arch:
                return sect
    if len(by_arch) == 1:
        return next(iter(by_arch.values()))
    return {}


def _load_dispatch_table() -> dict:
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is not None:
        return _DISPATCH_TABLE
    arch = _detect_arch()
    for path in _dispatch_json_paths():
        if path.is_file():
            data = json.loads(path.read_text())
            section = _select_arch_section(data, arch)
            _DISPATCH_TABLE = {
                "gfx": section.get("gfx"),
                "winners": dict(section.get("winners") or {}),
            }
            return _DISPATCH_TABLE
    _DISPATCH_TABLE = {"gfx": None, "winners": {}}
    return _DISPATCH_TABLE


def shape_id(
    *, n_groups: int, reduction_k: int, output_n: int, max_seq_len: int
) -> str:
    return f"B{n_groups}D{reduction_k}K{output_n}N{max_seq_len}"


def _coerce(v):
    if v is None:
        return None
    return int(v)


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


_SKEW_XCD_REMAP_SMALL_B_THRESHOLD = 120


def _skew_compact_xcd(n_groups: int, reduction_k: int) -> tuple[int | None, int | None]:
    if n_groups > _SKEW_XCD_REMAP_SMALL_B_THRESHOLD or reduction_k >= 512:
        return SKEW_COMPACT_XCD_C, SKEW_COMPACT_XCD_W
    return None, None


def _normalize_cfg(cfg: dict) -> dict:
    return {
        key: _coerce(cfg.get(key, default)) for key, default in _SCHEMA_DEFAULTS.items()
    }


def _config_valid(cfg: dict, *, reduction_k: int) -> bool:
    bk = cfg.get("block_k")
    if bk is None:
        bk = 64
    return not (reduction_k % bk != 0 or reduction_k // bk < 2)


def _heuristic_dispatch() -> dict:
    return _normalize_cfg({})


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
) -> dict:
    key = (
        n_groups,
        reduction_k,
        output_n,
        max_seq_len,
        xcd_c,
        xcd_w,
        block_k,
    )
    cached = _DISPATCH_CACHE.get(key)
    if cached is not None:
        return cached

    explicit = {
        "xcd_c": xcd_c,
        "xcd_w": xcd_w,
        "block_k": block_k,
    }

    table = _load_dispatch_table()
    sid = shape_id(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
    )
    entry = table["winners"].get(sid)
    if entry is not None:
        cfg = _normalize_cfg(entry)
        if not _config_valid(cfg, reduction_k=reduction_k):
            cfg = _heuristic_dispatch()
    else:
        cfg = _heuristic_dispatch()

    for k, v in explicit.items():
        if v is not None:
            cfg[k] = int(v)

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
    # explicit overrides (None -> dispatch table / heuristic / kernel default)
    xcd_c: int | None = None,
    xcd_w: int | None = None,
    block_k: int | None = None,
):
    output_n = B.shape[0] // n_groups
    reduction_k = B.shape[1]
    cfg = resolve_config(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
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
        sc_xcd_c, sc_xcd_w = _skew_compact_xcd(n_groups, reduction_k)
        return jagged_dense_bmm(
            C,
            A,
            B,
            BIAS,
            SEQ_OFFSETS,
            n_groups,
            max_seq_len,
            stream=stream,
            xcd_c=sc_xcd_c,
            xcd_w=sc_xcd_w,
            uniform_seqlen=False,
            block_k=cfg.get("block_k"),
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
        block_k=cfg.get("block_k"),
        threads=_coerce(cfg.get("threads")),
    )
