# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QSA index scorer profiles: exact supported rows, loader, matcher and typed reasons.

Pure Python (no torch, no FlyDSL, no device): importable and testable anywhere.

A *profile* (``aiter/configs/qsa_index_scorer/<name>.json``) lists the exact
call classes the fused FlyDSL gfx1250 scorer is allowed to serve. It is written
by :mod:`gen_profile` and every row carries the result of its golden check. A
call that matches no row whose golden status is ``pass`` is *uncovered* and the
caller runs its native scorer; the matcher always says why (:class:`Reason`).

Row keys (all must match):

* ``arch`` / ``cu``           -- device (``gfx1250`` / compute-unit count);
* ``cache_dtype``             -- ``bf16`` or ``fp8`` (E4M3 + fp32 per-row scale);
* ``regime``                  -- kernel regime derived from (rows, requests), see
                                 :func:`regime_of` (mirrors the kernel router);
* ``rows`` / ``requests``     -- inclusive ranges ``{"min", "max"[, "multiple_of"]}``
                                 (``min == max`` = exact row);
* ``columns``                 -- the effective column count (``max_columns`` bucket);
* ``table_width``             -- ``page_table.shape[1]``;
* ``vid`` / ``params``        -- kernel variant the kernel router selects for the
                                 row (checked against the router at call time);
* ``golden``                  -- ``{"status": "pass"|"pending"|"fail", ...}``.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
API_VERSION = "1.0"
FEATURE = "qsa_index_scorer"
BACKEND = "flydsl-gfx1250"
DEFAULT_PROFILE = "qsa_page4_gfx1250_v1"
#: Explicit opt-in: a profile name (or a path to a profile JSON). Unset = off.
PROFILE_ENV = "AITER_QSA_INDEX_SCORER_PROFILE"

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "configs", FEATURE))
KERNEL_MODULE = "aiter.ops.flydsl.kernels.mqa_logits.qsa_paged_mqa_logits_page4_gfx1250"
KERNEL_REL = "aiter/ops/flydsl/kernels/mqa_logits/qsa_paged_mqa_logits_page4_gfx1250.py"
KERNEL_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "flydsl", "kernels", "mqa_logits",
                 "qsa_paged_mqa_logits_page4_gfx1250.py"))

# Kernel contract (the only geometry the gfx1250 kernel implements).
INDEX_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 4            # compressed rows per index-cache page (KV block 16 / ratio 4)
COMPRESS_RATIO = 4
CACHE_ALIGN_BYTES = 128  # TDM routes need a 128 B cache base; others run native by policy
MAX_ROWS = 65535
MAX_PLANE_BYTES = 2**31 - 1

#: Import-time kernel switches and the values every profiled row was checked
#: with (the champion defaults). Any other effective value -> KERNEL_ENV_MISMATCH.
#: ``None`` = must be unset or equal to the inherited default (QSA_FP8_K_LOAD
#: inherits QSA_BF16_K_LOAD inside the kernel).
KERNEL_ENV_DEFAULTS: dict[str, str] = {
    "QSA_G2_ACCESS": "raw",
    "QSA_BF16_PATH": "wmma",
    "QSA_FP8_PATH": "wmma",
    "QSA_BF16_K_LOAD": "auto",
    "QSA_FP8_K_LOAD": "auto",
    "QSA_BF16_R64": "0",
    "QSA_BF16_R16_TDM_COLS": "32",
    "QSA_R16_NOBAR": "1",
    "QSA_BF16_PERSIST": "1",
    "QSA_PERSIST_COLS": "8",
    "QSA_FP8_CVT": "fast",
    "QSA_DRAFT_PATH": "1",
    "QSA_DRAFT_K_LOAD": "tdm",
    "QSA_DRAFT_COLS": "auto",
    "QSA_DECODE_GROUP": "all",
    "QSA_DECODE_GROUP_COLS": "32",
    "QSA_PAD_BUCKET": "0",
    "QSA_PAD_INT32_GUARD": "0",
    "QSA_STRICT_I32": "0",
}
_LOWERED = {"QSA_G2_ACCESS", "QSA_BF16_PATH", "QSA_FP8_PATH", "QSA_BF16_K_LOAD",
            "QSA_FP8_K_LOAD", "QSA_FP8_CVT", "QSA_DRAFT_K_LOAD", "QSA_DRAFT_COLS",
            "QSA_DECODE_GROUP"}

REGIMES = ("draft", "decode", "prefill")
CACHE_DTYPES = ("bf16", "fp8")
GOLDEN_STATUSES = ("pass", "pending", "fail")


class Reason(str, enum.Enum):
    """Why a call is (not) served by the fused scorer. Values are a stable API."""

    OK = "ok"
    # enablement / profile
    DISABLED = "disabled"                              # no profile selected (default)
    PROFILE_NOT_FOUND = "profile_not_found"
    PROFILE_INVALID = "profile_invalid"
    PROFILE_KERNEL_MISMATCH = "profile_kernel_mismatch"  # kernel source != profiled sha
    KERNEL_ENV_MISMATCH = "kernel_env_mismatch"          # QSA_* switches != profiled values
    KERNEL_UNAVAILABLE = "kernel_unavailable"            # FlyDSL/kernel import failed
    # device
    DEVICE_NOT_GPU = "device_not_gpu"
    DEVICE_MISMATCH = "device_mismatch"
    ARCH_UNSUPPORTED = "arch_unsupported"
    CU_UNSUPPORTED = "cu_unsupported"
    # tensor contract
    COMPRESS_RATIO = "compress_ratio_unsupported"
    Q_DTYPE = "q_dtype_unsupported"
    Q_SHAPE = "q_shape_unsupported"
    Q_LAYOUT = "q_layout_unsupported"
    CACHE_FORMAT = "cache_format_unsupported"            # mxfp8 / fp4 / other dtypes
    CACHE_SHAPE = "cache_shape_unsupported"
    CACHE_LAYOUT = "cache_layout_unsupported"
    CACHE_ALIGNMENT = "cache_alignment_unsupported"
    SCALE_LAYOUT = "scale_layout_unsupported"
    METADATA = "metadata_unsupported"
    OUTPUT_LAYOUT = "output_layout_unsupported"
    EMPTY = "empty_call"
    DIVISOR = "divisor_uncovered"
    MAX_COLUMNS = "max_columns_invalid"
    PLANE_TOO_LARGE = "plane_too_large"
    # profile coverage
    ROWS_RULE = "rows_rule_uncovered"
    COLUMNS = "columns_uncovered"
    TABLE_WIDTH = "table_width_uncovered"
    ROW_UNVERIFIED = "row_unverified"                    # matched, golden status != pass
    ROUTE_MISMATCH = "route_mismatch"                    # kernel router != row vid

    def __str__(self) -> str:  # noqa: D105
        return self.value


#: Reasons that depend only on persistent state (device, profile, kernel build,
#: cache tensors): evaluate once when a bucket is prepared (rank-uniform).
STATIC_REASONS = frozenset({
    Reason.DISABLED, Reason.PROFILE_NOT_FOUND, Reason.PROFILE_INVALID,
    Reason.PROFILE_KERNEL_MISMATCH, Reason.KERNEL_ENV_MISMATCH, Reason.KERNEL_UNAVAILABLE,
    Reason.ARCH_UNSUPPORTED, Reason.CU_UNSUPPORTED, Reason.CACHE_FORMAT, Reason.CACHE_SHAPE,
    Reason.CACHE_LAYOUT, Reason.CACHE_ALIGNMENT, Reason.SCALE_LAYOUT,
})


class ProfileError(ValueError):
    def __init__(self, reason: Reason, detail: str):
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Regime / variant rules (must mirror the kernel router; CPU tests cross-check)
# ---------------------------------------------------------------------------
def regime_of(rows: int, requests: int) -> str | None:
    """Kernel regime of a call shape (same thresholds as the kernel router).

    draft:   rows <= requests                       (one-row-per-wave draft kernel)
    decode:  requests < rows < 32*requests, rows%4==0 (decode row-group kernel)
    prefill: rows >= 32 and rows >= 32*requests     (r32 / persistent r32 kernels)
    None:    anything else (r16 tails: correct in-kernel but not a profiled class)
    """
    rows, requests = int(rows), int(requests)
    if rows < 1 or requests < 1:
        return None
    if rows <= requests:
        return "draft"
    if rows >= 32 and rows >= 32 * requests:
        return "prefill"
    if rows % 4 == 0:
        return "decode"
    return None


PERSIST_MIN_COLUMNS = 64 * 8   # WMMA_COLS * QSA_PERSIST_COLS (default)


def expected_variant(cache_dtype: str, regime: str, columns: int) -> tuple[str, dict]:
    """(vid, params) the default-env kernel router selects for an aligned cache."""
    if regime == "draft":
        return "qsa.page4.draft.v1", {"k_load": "tdm", "wave_cols": 32}
    if regime == "decode":
        return "qsa.page4.decg.v1", {"k_load": "tdm", "wave_cols": 32, "barrier": False}
    if regime == "prefill":
        if cache_dtype == "fp8":
            return "qsa.page4.r32_fp8.v1", {"k_load": "tdm", "rows_per_block": 32, "cvt": "fast"}
        if int(columns) >= PERSIST_MIN_COLUMNS:
            return "qsa.page4.r32_persist.v1", {"k_load": "tdm", "rows_per_block": 32,
                                                "persist_cols": 8, "buffers": 2}
        return "qsa.page4.r32.v1", {"k_load": "tdm", "rows_per_block": 32}
    raise ValueError(f"unknown regime {regime!r}")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
_SIG_KEYS = ("arch", "cu", "cache_dtype", "regime", "rows", "requests", "columns", "table_width")


def row_signature(row: Mapping[str, Any]) -> str:
    """Stable 16-hex signature of a row's match keys (``sig`` field)."""
    key = {k: row[k] for k in _SIG_KEYS}
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _range_ok(rng: Mapping[str, Any], value: int) -> bool:
    if value < int(rng["min"]) or value > int(rng["max"]):
        return False
    m = int(rng.get("multiple_of", 1))
    return value % m == 0


@dataclass(frozen=True)
class ProfileRow:
    raw: Mapping[str, Any]

    @property
    def sig(self) -> str:
        return self.raw["sig"]

    @property
    def cell(self) -> str:
        return self.raw["cell"]

    @property
    def vid(self) -> str:
        return self.raw["vid"]

    @property
    def params(self) -> Mapping[str, Any]:
        return self.raw["params"]

    @property
    def verified(self) -> bool:
        return self.raw["golden"]["status"] == "pass"

    def shape_ok(self, rows: int, requests: int) -> bool:
        return _range_ok(self.raw["rows"], rows) and _range_ok(self.raw["requests"], requests)


def validate_row(row: Mapping[str, Any]) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, f"row {row.get('cell', '?')}: {msg}")

    for key in ("sig", "arch", "cu", "cache_dtype", "regime", "rows", "requests", "columns",
                "table_width", "vid", "params", "cell", "golden", "generated_by", "notes"):
        if key not in row:
            bad(f"missing {key!r}")
    if row["cache_dtype"] not in CACHE_DTYPES:
        bad(f"cache_dtype {row['cache_dtype']!r}")
    if row["regime"] not in REGIMES:
        bad(f"regime {row['regime']!r}")
    for key in ("rows", "requests"):
        rng = row[key]
        if not isinstance(rng, Mapping) or "min" not in rng or "max" not in rng:
            bad(f"{key} must be {{min, max[, multiple_of]}}")
        if not 1 <= int(rng["min"]) <= int(rng["max"]):
            bad(f"{key} range {rng}")
    if int(row["rows"]["max"]) > MAX_ROWS:
        bad("rows above the kernel grid limit")
    if int(row["columns"]) < 1 or int(row["columns"]) > int(row["table_width"]) * PAGE_SIZE:
        bad("columns outside [1, table_width*4]")
    if row["golden"].get("status") not in GOLDEN_STATUSES:
        bad(f"golden.status {row['golden'].get('status')!r}")
    if row["sig"] != row_signature(row):
        bad(f"sig {row['sig']} != {row_signature(row)}")
    # every (rows, requests) the row admits must be the row's own regime
    for r in {int(row["rows"]["min"]), int(row["rows"]["max"])}:
        for q in {int(row["requests"]["min"]), int(row["requests"]["max"])}:
            if _range_ok(row["rows"], r) and regime_of(r, q) not in (row["regime"], None):
                bad(f"rows={r} requests={q} is regime {regime_of(r, q)}, not {row['regime']}")
    vid, params = expected_variant(row["cache_dtype"], row["regime"], int(row["columns"]))
    if row["vid"] != vid or dict(row["params"]) != params:
        bad(f"variant {row['vid']} {row['params']} != router {vid} {params}")


@dataclass
class Profile:
    name: str
    path: str
    data: Mapping[str, Any]
    rows: list[ProfileRow] = field(default_factory=list)
    _index: dict = field(default_factory=dict, repr=False)

    @property
    def kernel_sha256(self) -> str:
        return self.data["kernel"]["sha256"]

    @property
    def kernel_env(self) -> Mapping[str, str]:
        return self.data["kernel_env"]

    def candidates(self, arch: str, cu: int, cache_dtype: str, regime: str) -> list[ProfileRow]:
        return self._index.get((arch, int(cu), cache_dtype, regime), [])

    def archs(self) -> set:
        return {r.raw["arch"] for r in self.rows}

    def cus(self, arch: str) -> set:
        return {int(r.raw["cu"]) for r in self.rows if r.raw["arch"] == arch}


def validate_profile(data: Mapping[str, Any], name: str | None = None) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, msg)

    for key in ("schema_version", "feature", "profile", "api_version", "backend", "kernel",
                "kernel_env", "contract", "golden", "generated_by", "rows"):
        if key not in data:
            bad(f"missing top-level {key!r}")
    if data["schema_version"] != SCHEMA_VERSION:
        bad(f"schema_version {data['schema_version']} != {SCHEMA_VERSION}")
    if data["feature"] != FEATURE or data["backend"] != BACKEND:
        bad(f"feature/backend {data['feature']}/{data['backend']}")
    if str(data["api_version"]).split(".")[0] != API_VERSION.split(".")[0]:
        bad(f"api_version {data['api_version']} incompatible with {API_VERSION}")
    if name is not None and data["profile"] != name:
        bad(f"profile name {data['profile']!r} != file name {name!r}")
    if set(data["kernel_env"]) != set(KERNEL_ENV_DEFAULTS):
        bad("kernel_env keys differ from KERNEL_ENV_DEFAULTS")
    c = data["contract"]
    if (c.get("index_heads"), c.get("head_dim"), c.get("page_size"), c.get("compress_ratio")) != (
            INDEX_HEADS, HEAD_DIM, PAGE_SIZE, COMPRESS_RATIO):
        bad(f"contract geometry {c}")
    sigs = set()
    for row in data["rows"]:
        validate_row(row)
        if row["sig"] in sigs:
            bad(f"duplicate row sig {row['sig']} ({row['cell']})")
        sigs.add(row["sig"])


def resolve_profile_path(name_or_path: str) -> str:
    if os.sep in name_or_path or "/" in name_or_path or name_or_path.endswith(".json"):
        return os.path.abspath(name_or_path)
    return os.path.join(CONFIG_DIR, f"{name_or_path}.json")


_CACHE: dict[str, Profile] = {}


def load_profile(name_or_path: str, *, use_cache: bool = True) -> Profile:
    """Load + validate a profile; raises :class:`ProfileError` with a typed reason."""
    path = resolve_profile_path(name_or_path)
    if use_cache and path in _CACHE:
        return _CACHE[path]
    if not os.path.isfile(path):
        raise ProfileError(Reason.PROFILE_NOT_FOUND, path)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ProfileError(Reason.PROFILE_INVALID, f"{path}: {exc}") from exc
    stem = os.path.splitext(os.path.basename(path))[0]
    validate_profile(data, stem)
    prof = Profile(name=data["profile"], path=path, data=data,
                   rows=[ProfileRow(r) for r in data["rows"]])
    for row in prof.rows:
        key = (row.raw["arch"], int(row.raw["cu"]), row.raw["cache_dtype"], row.raw["regime"])
        prof._index.setdefault(key, []).append(row)
    if use_cache:
        _CACHE[path] = prof
    return prof


def clear_cache() -> None:
    _CACHE.clear()


def effective_kernel_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Effective value of every import-time kernel switch (kernel parsing rules)."""
    env = os.environ if environ is None else environ
    out = {}
    for key, default in KERNEL_ENV_DEFAULTS.items():
        if key == "QSA_FP8_K_LOAD":
            default = out.get("QSA_BF16_K_LOAD", KERNEL_ENV_DEFAULTS["QSA_BF16_K_LOAD"])
        val = env.get(key, default).strip()
        out[key] = val.lower() if key in _LOWERED else val
    return out


def kernel_env_diff(profile: Profile, effective: Mapping[str, str]) -> dict:
    return {k: (v, effective.get(k)) for k, v in profile.kernel_env.items()
            if effective.get(k) != v}


@dataclass(frozen=True)
class Match:
    ok: bool
    reason: Reason
    detail: str = ""
    row: ProfileRow | None = None


def match(profile: Profile, *, arch: str, cu: int, cache_dtype: str, rows: int, requests: int,
          columns: int, table_width: int, allow_unverified: bool = False) -> Match:
    """Match a (validated) call shape against the profile's exact rows."""
    if arch not in profile.archs():
        return Match(False, Reason.ARCH_UNSUPPORTED, f"arch {arch} not in {sorted(profile.archs())}")
    if int(cu) not in profile.cus(arch):
        return Match(False, Reason.CU_UNSUPPORTED, f"cu {cu} not in {sorted(profile.cus(arch))}")
    regime = regime_of(rows, requests)
    if regime is None:
        return Match(False, Reason.ROWS_RULE,
                     f"rows={rows} requests={requests}: no profiled regime (r16 tail class)")
    cands = profile.candidates(arch, cu, cache_dtype, regime)
    cands = [r for r in cands if r.shape_ok(rows, requests)]
    if not cands:
        return Match(False, Reason.ROWS_RULE,
                     f"{cache_dtype} {regime} rows={rows} requests={requests}: no row")
    by_width = [r for r in cands if int(r.raw["table_width"]) == int(table_width)]
    if not by_width:
        widths = sorted({int(r.raw["table_width"]) for r in cands})
        return Match(False, Reason.TABLE_WIDTH, f"table_width={table_width} not in {widths}")
    by_cols = [r for r in by_width if int(r.raw["columns"]) == int(columns)]
    if not by_cols:
        cols = sorted({int(r.raw["columns"]) for r in by_width})
        return Match(False, Reason.COLUMNS, f"columns={columns} not in {cols}")
    verified = [r for r in by_cols if r.verified]
    if verified:
        return Match(True, Reason.OK, verified[0].cell, verified[0])
    if allow_unverified:
        return Match(True, Reason.OK, f"{by_cols[0].cell} (unverified, allowed)", by_cols[0])
    return Match(False, Reason.ROW_UNVERIFIED,
                 f"{by_cols[0].cell}: golden status {by_cols[0].raw['golden']['status']}",
                 by_cols[0])


def iter_reasons() -> Iterable[str]:
    return (r.value for r in Reason)
