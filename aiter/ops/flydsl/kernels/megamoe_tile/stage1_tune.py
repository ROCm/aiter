# SPDX-License-Identifier: MIT
"""Per-shape tune table for the fused Stage1 GMM1 tiling.

Three columns decide how Stage1 runs a shape:

``tile_group`` (G)
    The arena tile stays ``block_m`` = 32 rows (Stage2 and MegaMoEv2's SBM depend
    on that grain), but ``tile_alloc`` hands out G physically adjacent tiles per
    claim and GMM1 runs one m_block of ``32 * G`` rows over them.  G is also an
    arena-layout parameter (map slots are rounded up to G), so it must be known
    before ``Stage1ArenaLayout.create``.
``split_local`` (0/1)
    Which GMM1 path runs (below).
``fanout_shards``
    Fan-out CTAs per destination GPU (32 on the split path, 16 unsplit).

Knobs every measured shape set to the same value are constants, not columns:
GMM1's N block ``GMM1_BN`` = 256 (128 was slower on every shape measured,
e.g. +32us at K3 TPR128) and the first GMM1 worker ticket ``COMPUTE_FIRST`` = 8
(the old -1 = essential-CTA count measured 44-87us slower on every shape
tried).

The table follows ``stage2_tune.py`` (and ``aiter/fused_moe.py``): same key, with
``token`` = GEMM rows = ``max_tok_per_rank * topk`` and ``expert`` the per-rank
count.  Resolution is table > built-in rule; there is no per-field env
override.  A sweep points ``AITER_CONFIG_MEGAMOE_TILE_STAGE1`` at its own copy
of the table, so every swept value is recorded in a file.

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

``split_local``
    Whether Stage1 splits each expert into a local-source and a remote-source
    group and starts GMM1 on the local group before the remote rows arrive
    (split_local + early_local_gmm and the switches that depend on it).  Each
    group gets its own tiles, so GMM1 reads every expert's weights once per
    group.  When one m_block (32 * G <= 128 rows) holds all of an expert's rows,
    the unsplit path reads the weights once and that beats the overlap; above
    that both paths re-read and the overlap wins.  46+47, 2026-10-06, Stage1 us:

        shape     rows/expert   split (G table)   unsplit (G)
        K3 128      36.6          297.8 (1)         242.4 (2)
        K3 256      73            370.1 (4)         281.3 (4)
        K3 512      146           395.1 (4)         459.2 (4)
        DSV4 128    32            269.7 (1)         207.9 (1)
        DSV4 256    64            304.1 (4)         262.7 (2)
        DSV4 512    128           331.5 (4)         283.5 (4)

    The G=1/G=4 rule above was measured on the split path (a G=2 split pads each
    half to 64 rows, which is why it lost); unsplit takes the smallest G whose
    m_block holds the rows.  The operator derives each path's companion
    switches from this value (split: early_local_gmm, fan2_shards, lazy_pad,
    extra_consumers; unsplit: static_g0, seal_fast); split without
    early_local_gmm was measured a net loss.
"""
from __future__ import annotations

import csv
import os
import threading

_ENV = "AITER_CONFIG_MEGAMOE_TILE_STAGE1"
_DEFAULT_NAME = "megamoe_tile_stage1_tuned.csv"

_KEY_FIELDS = ("gfx", "cu_num", "token", "model_dim", "inter_dim", "expert", "topk")
_VALUE_FIELDS = ("tile_group", "split_local", "fanout_shards")

GMM1_BN = 256
COMPUTE_FIRST = 8

TILE_GROUPS = (1, 2, 4)
_ARENA_BM = 32
# Between the measured 36.6 (G=1 best) and 73 (G=4 best) rows per expert.
_G1_MAX_ROWS = 48
# Unsplit wins while one m_block holds an expert's rows: 32 * max(TILE_GROUPS).
_UNSPLIT_MAX_ROWS = _ARENA_BM * max(TILE_GROUPS)
# K3 TPR128 unsplit: 16 shards 234 vs 32 shards 239us; the split path was tuned at 32.
_FANOUT_SHARDS = {0: 16, 1: 32}
# fan2 takes 16 of the split path's shards; t0_no_fanout keeps one more.
FANOUT_SHARDS_MIN, FANOUT_SHARDS_MAX = 2, 64

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
    # error does not name the CSV row that produced them.
    if row["tile_group"] not in TILE_GROUPS:
        raise ValueError(
            f"{where}: tile_group must be one of {TILE_GROUPS} (got {row['tile_group']})"
        )
    if row["split_local"] not in (0, 1):
        raise ValueError(f"{where}: split_local must be 0 or 1 (got {row['split_local']})")
    fs = row["fanout_shards"]
    if not FANOUT_SHARDS_MIN <= fs <= FANOUT_SHARDS_MAX:
        raise ValueError(
            f"{where}: fanout_shards must be in [{FANOUT_SHARDS_MIN}, {FANOUT_SHARDS_MAX}] (got {fs})"
        )
    if row["split_local"] and fs < 17:
        raise ValueError(f"{where}: the split path needs fanout_shards >= 17 (fan2 uses 16)")


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
    """Return ``{"tile_group", "split_local", "fanout_shards"}`` or ``None`` on a miss."""
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


def default_split_local(token: int, expert: int) -> int:
    """Split only when one m_block cannot hold an expert's rows (see docstring)."""
    rows = int(token) / max(1, int(expert))
    return 0 if rows <= _UNSPLIT_MAX_ROWS else 1


def default_tile_group(token: int, expert: int, split_local: int = 1) -> int:
    """Split path: G=1 for few rows per expert, G=4 otherwise (fitted).
    Unsplit path: the smallest G whose 32*G-row m_block holds the rows.

    ``token / expert`` is the balanced rows-per-expert estimate (``token`` is the
    GEMM row count, ``expert`` the per-rank count).
    """
    rows = int(token) / max(1, int(expert))
    if not split_local:
        return next((g for g in TILE_GROUPS if _ARENA_BM * g >= rows), max(TILE_GROUPS))
    return 1 if rows <= _G1_MAX_ROWS else 4


def default_fanout_shards(split_local: int) -> int:
    return _FANOUT_SHARDS[int(bool(split_local))]


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
    """Return ``{"tile_group", "split_local", "fanout_shards", "gmm1_bn",
    "compute_first", "source"}`` for this shape; ``source`` is ``table`` or
    ``default`` (the built-in rule).  The kernel name (``_tg{G}``, ``_fos{N}``,
    ``_slg``) remains the ground truth for what ran.
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
    if tuned is not None:
        out, source = dict(tuned), "table"
    else:
        split = default_split_local(token, expert)
        out = {"split_local": split,
               "tile_group": default_tile_group(token, expert, split),
               "fanout_shards": default_fanout_shards(split)}
        source = "default"
        _validate(out, "stage1 tile rule")
    out.update(gmm1_bn=GMM1_BN, compute_first=COMPUTE_FIRST, source=source)
    return out
