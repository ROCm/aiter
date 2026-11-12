# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Select measured kernel configurations by architecture and tensor shape.

Untuned token counts use the nearest measured count in log space. Missing
architecture or kernel tables return ``None`` so callers can use heuristics.

Table schema (``aiter/configs/model_configs/hc_gated_residual_tuned.json``)::

    {
      "gfx950": {
        "up_gate_mix": {"256": {"block_m": 64, "block_n": 32,
                                 "m_waves": 1, "n_waves": 2, "_us": ...}, ...},
        "up_gate_mix_n336": {"256": {"block_m": 32, "block_n": 16,
                                      "m_waves": 1, "n_waves": 1, ...}, ...},
        "k1":          {"256": {"method": "splitk", ...}, ...},
        "k1_n336":     {"256": {"method": "splitk", ...}, ...},
        "seam": {"combine1_next1_n384": {"128": {"k1": {...}, "k2": {...},
                                                  "_us": ...}, ...}, ...}
      }
    }

The ``seam`` namespace is ``combine{0|1}_next{0|1}_n{width}`` -> token count ->
``{k1, k2}`` geometry; see :func:`seam_config`.

The ``_us`` field is provenance only (the measured latency) and is ignored at
runtime.
"""

import functools
import json
import math
import os

_TABLE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "configs",
        "model_configs",
        "hc_gated_residual_tuned.json",
    )
)


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    """Load and cache the tuned-config table; empty dict if absent/invalid."""
    try:
        with open(_TABLE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _nearest_tokens(shape_map: dict, tokens: int):
    """Tuned token count closest to ``tokens`` in log space (exact if present)."""
    keys = sorted(int(k) for k in shape_map)
    if not keys:
        return None
    if tokens in keys:
        return tokens
    ref = math.log(max(tokens, 1))
    return min(keys, key=lambda k: abs(math.log(k) - ref))


def _entry(arch: str, kernel: str, tokens: int):
    tbl = _table().get(arch, {}).get(kernel)
    if not tbl:
        return None
    key = _nearest_tokens(tbl, tokens)
    return tbl[str(key)] if key is not None else None


def up_gate_mix_config(arch: str, tokens: int, packed_width: int | None = None):
    """Tuned ``{block_m, block_n, m_waves, n_waves}`` or ``None``."""
    arch_table = _table().get(arch, {})
    specific = (
        arch_table.get(f"up_gate_mix_n{packed_width}")
        if packed_width is not None
        else None
    )
    # K1's packed row stride changes K2's memory access pattern. Apply
    # stride-specific entries only to measured token counts.
    e = specific.get(str(tokens)) if specific else None
    if not e:
        e = _entry(arch, "up_gate_mix", tokens)
    if not e:
        return None
    return {k: e[k] for k in ("block_m", "block_n", "m_waves", "n_waves")}


def k1_plan(arch: str, tokens: int, n_pad: int | None = None):
    """Tuned fused-K1 (combine+norm+down) plan or ``None``.

    Entry is a flat dict: ``method`` (``"decouple"`` or ``"splitk"``) plus the
    kwargs :func:`flydsl_k1_combine_norm_down` consumes -- ``split_k`` (split-K
    only), ``stages``, ``block_k``, ``sk_block_m`` (split-K partial tile height),
    ``dn_block_n``/``dn_block_m``/``dn_m_waves``/``dn_n_waves``. Missing keys
    keep the kernel's heuristic default. ``_us`` is provenance only.

    Counts below the smallest measured entry return ``None`` to preserve the
    low-token split-K heuristic, as does an entry with ``method="heuristic"``
    (a count where the tuner measured the heuristic as fastest).
    """
    arch_table = _table().get(arch, {})
    # A shape-specific table must remain active outside its measured range
    # because the generic table may use incompatible N-wave geometry.
    specific = arch_table.get(f"k1_n{n_pad}") if n_pad is not None else None
    tbl = specific or arch_table.get("k1")
    if not tbl:
        return None
    keys = sorted(int(k) for k in tbl)
    if tokens < keys[0]:
        return None
    key = _nearest_tokens(tbl, tokens)
    plan = tbl[str(key)] if key is not None else None
    if plan is not None and plan.get("method") == "heuristic":
        return None
    return plan


SEAM_SKINNY_KNOBS = {
    "threshold": {
        "domain": (2, 3),
        "default": 3,
        "class": "safe",
        "constraints": "rows above it start the deferred default range",
    },
    "chunk_m": {
        "domain": (1, 2, 4, 8),
        "default": 2,
        "class": "safe",
        "constraints": "divides M; M <= 8",
    },
    "waves_per_block": {
        "domain": (2, 4, 8),
        "default": 4,
        "class": "safe",
        "constraints": "divides 2560 and merged width/columns",
    },
    "cols_per_wave": {
        "domain": (1, 2, 4),
        "default": 2,
        "class": "safe",
        "constraints": "waves*columns divides merged width",
    },
}


def seam_skinny_config(rows):
    """Measured skinny geometry for M <= 8; the default dispatch uses it for M <= 3."""
    if not 1 <= rows <= 8:
        return None
    return {
        "chunk_m": 2 if rows % 2 == 0 else 1,
        "waves_per_block": 4,
        "cols_per_wave": 2,
    }


def seam_geometry_legal(config, width, padded):
    """Validate gfx950 seam ownership without importing GPU/compiler code."""
    try:
        return seam_k1_legal(config["k1"], width, padded) and seam_k2_legal(
            config["k2"], padded
        )
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def seam_k1_legal(k1, width, padded):
    try:
        bm, bn, mw, nw = (
            k1[k] for k in ("block_m", "dn_block_n", "dn_m_waves", "dn_n_waves")
        )
        sk = k1["split_k"]
        bk = k1["block_k"]
        if any(not isinstance(v, int) or v <= 0 for v in (bm, bn, mw, nw, sk, bk)):
            return False
        if (
            bm not in (16, 32, 64)
            or padded % bm
            or bm % (16 * mw)
            or bn % (16 * nw)
            or width % bn
        ):
            return False
        if k1.get("use_tuned") is not False:
            return False
        if k1.get("sk_block_m") != bm or k1.get("dn_block_m") != bm:
            return False
        if k1.get("m_waves") != mw or k1.get("n_waves") != nw:
            return False
        if bk != 64 or 160 % sk or (sk > 1 and 160 // sk < k1["stages"] - 1):
            return False
        return bm * bk >= mw * nw * 64 * 8 and k1["stages"] == 2
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def seam_k2_legal(k2, padded):
    try:
        bm, bn, mw, nw = (k2[k] for k in ("block_m", "block_n", "m_waves", "n_waves"))
        if any(not isinstance(v, int) or v <= 0 for v in (bm, bn, mw, nw)):
            return False
        gs = k2.get("grid_swap", 0)
        if type(gs) is not int or gs not in (0, 1):
            return False
        if gs and k2.get("branch_mode") not in ("separate", "combined"):
            return False
        return (
            bm in (16, 32, 64)
            and padded % bm == 0
            and bm % (16 * mw) == 0
            and bn % (16 * nw) == 0
            and 2560 % bn == 0
            and bn % 8 == 0
            and (
                k2.get("branch_mode", "branchwise") != "branchwise"
                or (bm * (bn // 8)) % (mw * nw * 64) == 0
            )
            and (k2.get("branch_mode") != "combined" or nw == 1)
        )
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def seam_padded_extent(tokens, config):
    """Common extent for both stages: a multiple of the larger row tile."""
    tile = max(config["k1"]["block_m"], config["k2"]["block_m"])
    return ((tokens + tile - 1) // tile) * tile


def seam_default_tile(tokens):
    """Heuristic row tile used when no measured seam row applies."""
    return 16 if tokens <= 16 else 32 if tokens <= 32 else 64


def seam_heuristic_config(tokens, width):
    """Geometry used when no measured seam row applies (no table access)."""
    small = tokens <= 64
    # Heuristic fallback (no measured row): K1 and K2 at
    # BM=32 above 16 rows, since BM=64 halves the workgroup count of both GEMMs.
    bm = 16 if tokens <= 16 else (32 if tokens < 256 else 64)
    nw = 1 if bm == 16 else 2
    mw = 2 if bm == 64 else 1
    k1 = {
        "use_tuned": False,
        "block_m": bm,
        "sk_block_m": bm,
        "dn_block_m": bm,
        "dn_block_n": 64 if width == 320 else (96 if small else 128),
        "dn_m_waves": mw,
        "dn_n_waves": nw,
        "m_waves": mw,
        "n_waves": nw,
        "block_k": 64,
        "stages": 2,
        "split_k": 32 if small else 20,
    }
    k2 = {
        "block_m": 16 if tokens <= 16 else 32,
        "block_n": 32 if tokens <= 16 else 16,
        "m_waves": 1,
        "n_waves": 1,
    }
    return {"k1": k1, "k2": k2}


def seam_config(arch, tokens, combine, next_gate, width):
    """Separate semantic/width namespace; illegal nearest rows are excluded.

    Schema: arch.seam.combine{0|1}_next{0|1}_n{width}.{M}.{k1,k2}. K1 and K2
    carry independent row tiles; a row is eligible only if it is legal at the
    extent round_up(tokens, max(BM1, BM2)). No legacy table participates.
    """
    default = seam_heuristic_config(tokens, width)
    namespace = f"combine{int(combine)}_next{int(next_gate)}_n{width}"
    tbl = _table().get(arch, {}).get("seam", {}).get(namespace, {})
    legal = {
        key: value
        for key, value in tbl.items()
        if key.isdigit() and _row_legal(value, width, tokens)
    }
    nearest = _nearest_tokens(legal, tokens)
    selected = legal[str(nearest)] if nearest is not None else default
    return {"k1": dict(selected["k1"]), "k2": dict(selected["k2"])}


def _row_legal(value, width, tokens):
    try:
        padded = seam_padded_extent(tokens, value)
    except (KeyError, TypeError):
        return False
    return seam_geometry_legal(value, width, padded)


SEAM_KNOBS = {
    "k1_a_batch": {
        "domain": (0, 1),
        "default": 0,
        "constraints": "explicit deferred; independent A-group registers; reject spills",
        "class": "safe",
    },
    "k1_grid_swap": {
        "domain": (0, 1),
        "default": 0,
        "constraints": "explicit deferred; grid (split,N,M) is a bijection",
        "class": "safe",
    },
    "k2_grid_swap": {
        "domain": (0, 1),
        "default": 0,
        "constraints": "explicit deferred; separate or combined; grid (N,M) is a bijection",
        "class": "safe",
    },
    "k2_branch_mode": {
        "domain": ("branchwise", "combined", "separate"),
        "default": None,
        "constraints": "None picks per M (seam_k2_default / seam_deferred_k2_config); combined requires N waves=1; 16-channel panels",
        "class": "semantic",
    },
    "k2_slp": {
        "domain": (False, True),
        "default": True,
        "constraints": "combined/separate K2 only; ignored by branchwise and skinny",
        "class": "safe",
    },
    "deferred": {"domain": (False, True), "default": False, "class": "semantic"},
    "row_tile": {"domain": (16, 32, 64), "default": None, "class": "safe"},
    "k1_row_tile": {"domain": (16, 32, 64), "default": None, "class": "safe"},
    "k2_row_tile": {"domain": (16, 32, 64), "default": None, "class": "safe"},
    "k1_n_tile": {
        "domain": (16, 32, 48, 64, 96, 128, 192, 320, 384),
        "default": None,
        "class": "safe",
    },
    "k1_m_waves": {"domain": (1, 2, 4), "default": None, "class": "safe"},
    "k1_n_waves": {"domain": (1, 2, 4), "default": None, "class": "safe"},
    "k2_n_tile": {"domain": (16, 32, 64, 128), "default": None, "class": "safe"},
    "k2_m_waves": {"domain": (1, 2, 4), "default": None, "class": "safe"},
    "k2_n_waves": {"domain": (1, 2, 4), "default": None, "class": "safe"},
    "padded_extent": {
        "domain": "positive multiples of max(BM1, BM2) >= M; multiples of 64 >= M when deferred",
        "default": None,
        "class": "safe",
    },
    "fallback": {
        "domain": ("splitk", "decouple", "skinny"),
        "default": None,
        "class": "semantic",
    },
    "split_k": {
        "domain": (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160),
        "default": None,
        "class": "semantic",
    },
}
# None selects the heuristic; seam_geometry_legal is the shared pruning predicate.


# Force-selected sweep candidates; these do not change the branchwise defaults.
_K2_BRANCH_CANDIDATES = {
    "combined": {
        1: {"block_m": 16, "block_n": 16, "m_waves": 1, "n_waves": 1, "slp": False},
        16: {"block_m": 16, "block_n": 16, "m_waves": 1, "n_waves": 1, "slp": False},
        32: {"block_m": 32, "block_n": 16, "m_waves": 2, "n_waves": 1, "slp": True},
        64: {"block_m": 16, "block_n": 64, "m_waves": 1, "n_waves": 1, "slp": True},
        128: {"block_m": 32, "block_n": 16, "m_waves": 2, "n_waves": 1, "slp": True},
        256: {"block_m": 32, "block_n": 16, "m_waves": 2, "n_waves": 1, "slp": True},
    },
    "separate": {
        1: {"block_m": 16, "block_n": 32, "m_waves": 1, "n_waves": 1, "slp": True},
        16: {"block_m": 16, "block_n": 32, "m_waves": 1, "n_waves": 1, "slp": True},
        32: {"block_m": 16, "block_n": 32, "m_waves": 1, "n_waves": 1, "slp": True},
        64: {"block_m": 32, "block_n": 32, "m_waves": 2, "n_waves": 1, "slp": False},
        128: {"block_m": 32, "block_n": 64, "m_waves": 1, "n_waves": 4, "slp": False},
        256: {"block_m": 16, "block_n": 32, "m_waves": 1, "n_waves": 1, "slp": True},
    },
}


def seam_k2_branch_config(rows, mode):
    table = _K2_BRANCH_CANDIDATES.get(mode, {})
    if not table:
        return {}
    key = min(table, key=lambda m: abs(m - rows))
    return dict(table[key])


# Default K2 schedule per M. Measured (prepared seam, b=24, same GPU):
# the register-only "separate" K2 beats the LDS "branchwise" K2 up to M=96
# (M80 -1.0us, M96 -1.1us) and loses from M=112 (+0.3us). Above the threshold
# the branchwise K2 keeps its tuned geometry from seam_config.
SEAM_K2_SEPARATE_MAX_ROWS = 96


def seam_k2_default(rows):
    """Default ``(branch_mode, k2 overrides)`` for ``rows`` tokens.

    Rows at or below the threshold use the "separate" candidate row at or
    above ``rows`` (the M64 row covers 65..threshold). Larger counts return
    ``("branchwise", {})``, i.e. the seam_config geometry unchanged.
    """
    if rows > SEAM_K2_SEPARATE_MAX_ROWS:
        return "branchwise", {}
    table = _K2_BRANCH_CANDIDATES["separate"]
    key = min((m for m in table if rows <= m <= 64), default=64)
    return "separate", dict(table[key])


DEFERRED_K1_KNOBS = {
    "stages": {
        "domain": (2, 3),
        "default": 2,
        "class": "safe",
        "constraints": "k_local >= stages-1; LDS <= 163840",
    },
    "block_m": {"domain": (16, 32, 64), "default": 16, "class": "safe"},
    "block_n": {"domain": (32, 64), "default": 64, "class": "safe"},
    "block_k": {
        "domain": (64, 128),
        "default": 128,
        "class": "semantic",
        "constraints": "chunk_width divisible by block_k",
    },
    "slp": {"domain": (False, True), "default": True, "class": "safe"},
    "waves": {
        "domain": ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1)),
        "default": (1, 4),
        "class": "safe",
    },
}


def seam_deferred_k1_config(rows):
    """Branch-aligned K1 geometry selected by full prepared-seam sweeps."""
    if rows <= 32:
        return {
            "block_m": 16,
            "block_n": 64,
            "block_k": 128,
            "m_waves": 1,
            "n_waves": 4,
            "slp": True,
        }
    if rows <= 96:
        return {
            "block_m": 32,
            "block_n": 64,
            "block_k": 128,
            "m_waves": 1,
            "n_waves": 4,
            "slp": False,
        }
    return {
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "m_waves": 4,
        "n_waves": 1,
        "slp": False,
    }


def seam_deferred_k1_legal(k1):
    """Legality of a deferred K1 geometry (DEFERRED_K1_KNOBS domains + wave split).

    The deferred builder takes its own geometry, so seam_geometry_legal's K1
    rules (block_k == 64, split-K, dn_* keys) do not apply to it.
    """
    try:
        bm, bn, bk = k1["block_m"], k1["block_n"], k1["block_k"]
        waves = (k1["m_waves"], k1["n_waves"])
        mw, nw = waves
        chunk = k1.get("chunk_width", 256)
        stages = k1.get("stages", 2)
        ab, gs = k1.get("k1_a_batch", 0), k1.get("k1_grid_swap", 0)
        if any(type(v) is not int or v not in (0, 1) for v in (ab, gs)):
            return False
        threads = mw * nw * 64
        if any(
            type(v) is not int or v <= 0 for v in (bm, bn, bk, mw, nw, chunk, stages)
        ):
            return False
        local = chunk // bk
        lds = stages * (bm + bn) * bk * 2 + bm * local * 4
        return (
            bm in DEFERRED_K1_KNOBS["block_m"]["domain"]
            and bn in DEFERRED_K1_KNOBS["block_n"]["domain"]
            and bk in DEFERRED_K1_KNOBS["block_k"]["domain"]
            and waves in DEFERRED_K1_KNOBS["waves"]["domain"]
            and stages in DEFERRED_K1_KNOBS["stages"]["domain"]
            and type(k1.get("slp", False)) is bool
            and chunk in (256, 512, 640, 1280, 2560)
            and 2560 % chunk == 0
            and chunk % bk == 0
            and bm % (16 * mw) == 0
            and bn % (16 * nw) == 0
            and 320 % bn == 0
            and (bm * bk) % (threads * 8) == 0
            and (bn * bk) % (threads * 8) == 0
            and threads % (bk // 8) == 0
            and local >= stages - 1
            and lds <= 163840
        )
    except (KeyError, TypeError, ZeroDivisionError):
        return False


def seam_deferred_k2_config(rows):
    """Registered K2 candidates for the deferred path."""
    if rows <= 32:
        return "combined", {
            "block_m": 32,
            "block_n": 16,
            "m_waves": 1,
            "n_waves": 1,
            "slp": False,
        }
    if rows <= 64:
        return "separate", {
            "block_m": 16,
            "block_n": 16,
            "m_waves": 1,
            "n_waves": 1,
            "slp": False,
        }
    if rows <= 96:
        return "separate", {
            "block_m": 32,
            "block_n": 32,
            "m_waves": 1,
            "n_waves": 2,
            "slp": False,
        }
    return "separate", {
        "block_m": 32,
        "block_n": 16,
        "m_waves": 1,
        "n_waves": 1,
        "slp": False,
    }


# Starts right above the skinny threshold so the two default ranges tile M.
SEAM_DEFERRED_DEFAULT_RANGES = ((SEAM_SKINNY_KNOBS["threshold"]["default"] + 1, 128),)


def seam_deferred_default(rows):
    """Row ranges where the three-launch deferred path is the default."""
    return any(lo <= rows <= hi for lo, hi in SEAM_DEFERRED_DEFAULT_RANGES)


# Row ranges where the structural knobs are on by default for the deferred path.
SEAM_DEFERRED_STRUCTURAL_RANGES = ((64, 128),)


def seam_deferred_structural_default(rows, combine, next_gate):
    """Default structural knobs for the default-selected deferred path.

    Admission evidence covers only combine=1/next_gate=1, so other
    configurations keep the knobs off.
    """
    on = (
        combine
        and next_gate
        and any(lo <= rows <= hi for lo, hi in SEAM_DEFERRED_STRUCTURAL_RANGES)
    )
    v = int(bool(on))
    return {"k1_a_batch": v, "k1_grid_swap": v, "k2_grid_swap": v}
