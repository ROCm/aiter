# SPDX-License-Identifier: MIT
"""Per-shape tune table for the two-kernel Stage2 transport knobs.

``num_qp`` and ``return_chunk_tokens`` are genuinely shape-dependent: CHUNK sets
the rail packet size (32 beat 16 by 8.8us at token=8192) and the optimum moves
with the token count, so a single process-wide environment variable cannot hold
the right answer for every shape.

The key follows the convention ``aiter/fused_moe.py`` already uses for its own
tuned tables (``fused_moe.py:2192``), with ``token`` meaning the GEMM row count
-- ``max_tok_per_rank * topk`` under the EP16 one-route-per-rank routing, *not*
the per-rank token count.  A miss returns ``None``; the caller keeps its built-in
default, so an absent or partial table never breaks a run.
"""
from __future__ import annotations

import csv
import os
import threading

_ENV = "AITER_CONFIG_MEGAMOE_TILE_STAGE2"
_DEFAULT_NAME = "megamoe_tile_stage2_tuned.csv"

_KEY_FIELDS = ("gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
_INT_KEYS = ("cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
_VALUE_FIELDS = ("num_qp", "return_chunk_tokens")

_lock = threading.Lock()
_cache: dict | None = None
_cache_path: str | None = None


def table_path() -> str:
    """Resolve the tune table: ``$AITER_CONFIG_MEGAMOE_TILE_STAGE2`` or the default."""
    override = os.environ.get(_ENV)
    if override:
        return override
    from aiter.jit.core import AITER_ROOT_DIR

    return os.path.join(AITER_ROOT_DIR, "aiter", "configs", _DEFAULT_NAME)


def _validate(row: dict, path: str, lineno: int) -> None:
    # Reject out-of-domain values at load time rather than letting them reach
    # compile_stage2_node_combine, whose error message says nothing about which
    # CSV row produced it.
    num_qp = row["num_qp"]
    chunk = row["return_chunk_tokens"]
    if num_qp not in (1, 2, 4, 8):
        raise ValueError(
            f"{path}:{lineno}: num_qp must be one of 1,2,4,8 (got {num_qp})"
        )
    if chunk < 4:
        raise ValueError(
            f"{path}:{lineno}: return_chunk_tokens must be >= 4 (got {chunk})"
        )


def _load(path: str) -> dict:
    table: dict = {}
    if not os.path.isfile(path):
        return table
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [f for f in _KEY_FIELDS + _VALUE_FIELDS if f not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"{path}: tune table is missing columns {missing}")
        for lineno, raw in enumerate(reader, start=2):
            if not raw.get("gfx") or raw["gfx"].lstrip().startswith("#"):
                continue
            key = tuple(
                raw["gfx"].strip() if f == "gfx" else int(raw[f])
                for f in _KEY_FIELDS
            )
            row = {f: int(raw[f]) for f in _VALUE_FIELDS}
            _validate(row, path, lineno)
            table[key] = row
    return table


def _table() -> dict:
    global _cache, _cache_path
    path = table_path()
    with _lock:
        if _cache is None or _cache_path != path:
            _cache = _load(path)
            _cache_path = path
        return _cache


def reset_cache() -> None:
    """Drop the memoised table (tests / sweeps that rewrite the CSV in-process)."""
    global _cache, _cache_path
    with _lock:
        _cache = None
        _cache_path = None


def lookup_stage2_tune(
    *,
    gfx: str,
    cu_num: int,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
) -> dict | None:
    """Return ``{"num_qp": int, "return_chunk_tokens": int}`` or ``None`` on a miss.

    ``expert`` is the per-rank routed-expert count and ``inter_dim`` the global
    one, matching ``kimik3_a4w4_tuned_fmoe.csv``.
    """
    key = (
        str(gfx),
        int(cu_num),
        int(token),
        int(model_dim),
        int(inter_dim),
        int(expert),
        int(topk),
    )
    return _table().get(key)
