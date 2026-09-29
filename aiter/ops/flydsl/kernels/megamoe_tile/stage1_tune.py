# SPDX-License-Identifier: MIT
"""Per-shape tune table for the fused Stage1 GMM1 tiling.

Two knobs decide how GMM1 tiles its work:

``tile_group`` (G)
    The arena tile stays ``block_m`` = 32 rows (Stage2 and MegaMoEv2's SBM depend
    on that grain), but ``tile_alloc`` hands out G physically adjacent tiles per
    claim and GMM1 runs one m_block of ``32 * G`` rows over them.  G is also an
    arena-layout parameter (map slots are rounded up to G), so it must be known
    before ``Stage1ArenaLayout.create``.
``gmm1_bn``
    GMM1's own N block (128 or 256), decoupled from the arena ``block_n``.

The table follows ``stage2_tune.py`` (and ``aiter/fused_moe.py``): same key, with
``token`` = GEMM rows = ``max_tok_per_rank * topk`` and ``expert`` the per-rank
count.  Resolution order is env > table > built-in rule; the env vars stay
because a single-variable sweep must be able to override the table, otherwise
the table silently changes a second variable.

The built-in rule is a fallback fitted to measurements, not a model.  The
weight-re-read model (smallest G minimising ``ceil(rows / 32G)``) was measured
wrong on the current Stage1: K3 2026-09-29 (47+50, fused_stage1 us)

    rows/expert   G=1     G=2     G=4
    36.6 (TPR128) 296.3   348.4   344.3   <- model picked G=2, the slowest
    73   (TPR256) 493.5   378.4   368.9
    146  (TPR512) 685.2   617.6   407.6

so the rule is G=1 up to ``_G1_MAX_ROWS`` rows per expert and G=4 above it.
G=2 is never picked by default (it also hung once on DSV4 TPR512).  G=8 does
not fit: the 256-row accumulator needs 256 KB of LDS.
"""
from __future__ import annotations

import csv
import os
import threading

_ENV = "AITER_CONFIG_MEGAMOE_TILE_STAGE1"
_DEFAULT_NAME = "megamoe_tile_stage1_tuned.csv"

# Per-field overrides; env > table > rule.
ENV_TILE_GROUP = "MEGAMOE_TK_S1_TILE_GROUP"
ENV_GMM1_BN = "MEGAMOE_TK_S1_GMM1_BN"

_KEY_FIELDS = ("gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
_VALUE_FIELDS = ("tile_group", "gmm1_bn")

TILE_GROUPS = (1, 2, 4)
GMM1_BNS = (128, 256)
_ARENA_BM = 32
# Between the measured 36.6 (G=1 best) and 73 (G=4 best) rows per expert.
_G1_MAX_ROWS = 48
_DEFAULT_BN = 256

_lock = threading.Lock()
_cache: dict | None = None
_cache_path: str | None = None


def table_path() -> str:
    """Resolve the tune table: ``$AITER_CONFIG_MEGAMOE_TILE_STAGE1`` or the default."""
    override = os.environ.get(_ENV)
    if override:
        return override
    from aiter.jit.core import AITER_ROOT_DIR

    return os.path.join(AITER_ROOT_DIR, "aiter", "configs", _DEFAULT_NAME)


def _validate(row: dict, where: str) -> None:
    # Reject out-of-domain values here rather than in the Stage1 factory, whose
    # error names neither the CSV row nor the env var that produced them.
    if row["tile_group"] not in TILE_GROUPS:
        raise ValueError(
            f"{where}: tile_group must be one of {TILE_GROUPS} (got {row['tile_group']})"
        )
    if row["gmm1_bn"] not in GMM1_BNS:
        raise ValueError(
            f"{where}: gmm1_bn must be one of {GMM1_BNS} (got {row['gmm1_bn']})"
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
            _validate(row, f"{path}:{lineno}")
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


def lookup_stage1_tune(
    *,
    gfx: str,
    cu_num: int,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
) -> dict | None:
    """Return ``{"tile_group": int, "gmm1_bn": int}`` or ``None`` on a miss."""
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


def default_tile_group(token: int, expert: int) -> int:
    """G=1 for few rows per expert, G=4 otherwise (fitted, see module docstring).

    ``token / expert`` is the balanced rows-per-expert estimate (``token`` is the
    GEMM row count, ``expert`` the per-rank count).
    """
    rows = int(token) / max(1, int(expert))
    return 1 if rows <= _G1_MAX_ROWS else 4


def resolve_stage1_tile(
    *,
    token: int,
    model_dim: int,
    inter_dim: int,
    expert: int,
    topk: int,
    gfx: str | None = None,
    cu_num: int | None = None,
) -> dict:
    """Return ``{"tile_group", "gmm1_bn", "source"}`` for this shape.

    ``source`` is ``env`` / ``table`` / ``default`` for the whole tuple, or a
    ``field=source`` list when the fields came from different places.  The
    kernel name (``_tg{G}``, ``_gbn128``) remains the ground truth for what ran.
    """
    tuned = None
    try:
        if gfx is None or cu_num is None:
            from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

            gfx = get_gfx_runtime() if gfx is None else gfx
            cu_num = get_cu_num() if cu_num is None else cu_num
        tuned = lookup_stage1_tune(
            gfx=gfx,
            cu_num=cu_num,
            token=token,
            model_dim=model_dim,
            inter_dim=inter_dim,
            expert=expert,
            topk=topk,
        )
    except ValueError:
        # A malformed table is a configuration error, not a miss.
        raise
    except Exception:
        # Anything else (no device, no aiter root) falls back to the rule.
        tuned = None
    rule = {"tile_group": default_tile_group(token, expert), "gmm1_bn": _DEFAULT_BN}
    out, sources = {}, {}
    for field, env in (("tile_group", ENV_TILE_GROUP), ("gmm1_bn", ENV_GMM1_BN)):
        raw = os.environ.get(env, "")
        # MEGAMOE_TK_S1_GMM1_BN=0 historically meant "use the default"; keep it
        # a non-override so old scripts do not pin the value.
        if raw and not (field == "gmm1_bn" and int(raw) == 0):
            out[field], sources[field] = int(raw), "env"
        elif tuned is not None:
            out[field], sources[field] = int(tuned[field]), "table"
        else:
            out[field], sources[field] = rule[field], "default"
    _validate(out, "stage1 tile config (" + ", ".join(
        f"{f}={sources[f]}" for f in _VALUE_FIELDS) + ")")
    kinds = set(sources.values())
    out["source"] = (
        kinds.pop() if len(kinds) == 1
        else ",".join(f"{f}={sources[f]}" for f in _VALUE_FIELDS)
    )
    return out
