# SPDX-License-Identifier: MIT
"""Per-shape tune table for the two-kernel Stage2 transport and GEMM2 knobs.

``num_qp`` and ``return_chunk_tokens`` are genuinely shape-dependent: CHUNK sets
the rail packet size (32 beat 16 by 8.8us at token=8192) and the optimum moves
with the token count, so a single process-wide environment variable cannot hold
the right answer for every shape.

The key follows the convention ``aiter/fused_moe.py`` already uses for its own
tuned tables (``fused_moe.py:2192``), with ``token`` meaning the GEMM row count
-- ``max_tok_per_rank * topk`` under the EP16 one-route-per-rank routing, *not*
the per-rank token count.  A miss returns ``None``; the caller keeps its built-in
default, so an absent or partial table never breaks a run.

``gemm2_bn`` (kernel1's GEMM2 N tile, 128 or 256) is an *optional* column: a
table without it, or a row with the cell left empty, falls back to the built-in
default, so older tables keep loading.  BM stays 32: kernel1 requires
``SBM % BM == 0`` and the arena tile (SBM) is 32 rows.
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
_OPTIONAL_FIELDS = ("gemm2_bn",)

ENV_GEMM2_BN = "MEGAMOE_TK_BN"
GEMM2_BNS = (128, 256)
DEFAULT_GEMM2_BN = 128

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
    bn = row.get("gemm2_bn")
    if bn is not None and bn not in GEMM2_BNS:
        raise ValueError(
            f"{path}:{lineno}: gemm2_bn must be one of {GEMM2_BNS} (got {bn})"
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
            for f in _OPTIONAL_FIELDS:
                cell = (raw.get(f) or "").strip()
                if cell:
                    row[f] = int(cell)
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
    """Return ``{"num_qp", "return_chunk_tokens"[, "gemm2_bn"]}`` or ``None`` on a miss.

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


def resolve_gemm2_bn(tuned: dict | None) -> tuple[int, str]:
    """GEMM2 N tile and its source: env > table > default.

    ``tuned`` is the ``lookup_stage2_tune`` result (``None`` on a miss); the
    caller already has it, so this does not look the table up again.
    """
    raw = os.environ.get(ENV_GEMM2_BN, "")
    if raw:
        bn, source = int(raw), "env"
    elif tuned and tuned.get("gemm2_bn") is not None:
        bn, source = int(tuned["gemm2_bn"]), "table"
    else:
        bn, source = DEFAULT_GEMM2_BN, "default"
    if bn not in GEMM2_BNS:
        raise ValueError(f"gemm2_bn must be one of {GEMM2_BNS} (got {bn}, from {source})")
    return bn, source
