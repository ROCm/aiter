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
* ``golden``                  -- ``{"status": "pass"|"pending"|"fail", ...}``; a
                                 ``pass`` names the kernel sha it was GPU-verified on
                                 (``kernel_sha256``) and, if that is not the profiled
                                 kernel, the ISA-identity evidence (``isa_identity``).

Schema 2 adds verified **range rules** (``rules``), matched only when no exact
row serves the call (exact rows keep precedence). A rule admits every call with

* ``arch`` / ``cu`` / ``cache_dtype`` / ``regime`` equal;
* ``relation`` -- rows/requests relation (:data:`RELATIONS`, implies ``regime``);
* ``rows`` / ``requests`` -- inclusive ranges (``rows`` may carry ``multiple_of``);
* ``columns``  -- ``{"pow2": {"min", "max"}}`` (every power of two in range) or
                  ``{"set": [...]}``; always also ``columns <= table_width * 4``;
* ``table_width`` -- inclusive range ``{"min", "max"}``;
* ``rows_x_columns_max`` (optional) -- ``rows * columns <= rows_x_columns_max``: ties
  the rows bound to the column count (ATOM's select loop chunks rows so one fp32
  logits chunk stays <= 128 MiB, i.e. ``rows <= 2**25 // columns``);
* ``page_size`` in :data:`PAGE_SIZES`, ``divisor`` == ``"sqrt(head_dim)"`` (the call-time
  contract); ``columns <= table_width * page_size``.

ROUND-PAGE16: every row and rule carries the index page size ``page_size``
(``compressed_k_cache.shape[1]`` = KV block / 4: block 16 -> 4, block 64 -> 16).
Rows without the key are page 4 (and a page-4 row signature is unchanged, so
existing evidence carries over); calls only match rows/rules of their own page
size. A page size outside :data:`PAGE_SIZES` is ``cache_shape_unsupported``.

and maps to one kernel variant ``vid`` / ``params`` (must equal
:func:`expected_variant` for every admitted column; the call-time router guard
``route_mismatch`` still applies). Its ``samples`` are the deterministic shapes
``gen_profile check`` runs against the frozen golden; ``golden.status == "pass"``
requires ``kernel_sha256`` and a passing ``golden.checked`` record for every sample.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 2
#: schema 1 = exact rows only (no ``rules``); still loadable
SUPPORTED_SCHEMAS = (1, 2)
API_VERSION = "1.2"
#: api_version stamped into generated profiles: the oldest API the profile needs (1.1 is
#: additive -- prewarm/launch -- and needs no profile change). Compatibility is by major.
PROFILE_API_VERSION = "1.0"
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
    os.path.join(
        _HERE,
        "..",
        "flydsl",
        "kernels",
        "mqa_logits",
        "qsa_paged_mqa_logits_page4_gfx1250.py",
    )
)

# Kernel contract (the only geometry the gfx1250 kernel implements).
INDEX_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 4  # default index page (KV block 16 / ratio 4); rows without page_size
#: index page sizes profiles may carry (KV block 16 -> 4, block 32 -> 8, block 64 -> 16).
#: R2: page 8 is plumbed (rows/rules/buckets, matcher, prewarm); whether the loaded
#: kernel implements a page size is probed at call time (``SUPPORTED_PAGE_SIZES``).
PAGE_SIZES = (4, 8, 16)
COMPRESS_RATIO = 4
CACHE_ALIGN_BYTES = (
    128  # TDM routes need a 128 B cache base; others run native by policy
)
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
    # 82c4dec5: per-dtype slab ('auto' = bf16 32 / fp8 64); '32'/'64' force both dtypes
    "QSA_DECODE_GROUP_COLS": "auto",
    # 82c4dec5 fp8 WMMA-hazard fix: compiler-visible (intrinsic) pk8 converter by
    # default; '1' restores the inline-asm converter (fp8 decode-group back to w32)
    "QSA_FP8_CVT_ASM": "0",
    "QSA_DRAFT_FP8_SCHED": "qpre",
    "QSA_DECG_FP8_SCHED": "base",
    "QSA_PAD_BUCKET": "0",
    "QSA_PAD_INT32_GUARD": "0",
    "QSA_STRICT_I32": "0",
    # ROUND-R2 kernel (d438d2b3) wrapper knobs: defaults = the 536667d5 launches.
    # Per-cell values are pinned through row/rule/bucket ``params`` (LAUNCH_KNOBS),
    # never through the process environment.
    "QSA_DECG_ROWS3": "0",
    "QSA_DECG_MODE": "base",
    "QSA_DECG_WG_WAVES": "4",
    # ROUND-R2E: per-dtype draft workgroup waves ('auto' = bf16 4, fp8 2)
    "QSA_DRAFT_WG_WAVES": "auto",
    # ROUND-R2B kernel: persist raw fallback as a runtime loop (compile time) and the
    # data-driven prefill route table ('legacy' = the d438d2b3 persist >= 512 columns).
    # ROUND-R2E default preset 'r2e': lean ktile 32x2 for columns <= 2048 (rows <= 16384),
    # persist above.
    "QSA_PERSIST_FALLBACK": "loop",
    "QSA_PREFILL_ROUTE": "r2e",
    # ROUND-R2C kernel: fp8 r32 TDM prefill early-prefetch kernel (opt-in; "0" = kernel_r32).
    "QSA_FP8_EARLY": "0",
    "QSA_FP8_KT": "",
    # ROUND-R2E: fp8 prefill route table, preset 'r2e': lean+kconv ktile 32x2 (rows <= 2048)
    # / 32x4 (rows 2049..16384) for columns <= 2048, r32 fp8 above.
    "QSA_FP8_PREFILL_ROUTE": "r2e",
}
_LOWERED = {
    "QSA_G2_ACCESS",
    "QSA_BF16_PATH",
    "QSA_FP8_PATH",
    "QSA_BF16_K_LOAD",
    "QSA_FP8_K_LOAD",
    "QSA_FP8_CVT",
    "QSA_DRAFT_K_LOAD",
    "QSA_DRAFT_COLS",
    "QSA_DECODE_GROUP",
    "QSA_DECODE_GROUP_COLS",
    "QSA_DRAFT_FP8_SCHED",
    "QSA_DECG_FP8_SCHED",
    "QSA_DECG_MODE",
    "QSA_PERSIST_FALLBACK",
    "QSA_PREFILL_ROUTE",
    "QSA_FP8_PREFILL_ROUTE",
    "QSA_DRAFT_WG_WAVES",
}
#: Kernel switches whose default in the kernel source is a module constant, not
#: a literal (CPU drift test resolves them on the imported kernel).
KERNEL_ENV_SYMBOLIC_DEFAULTS = {
    "QSA_FP8_K_LOAD": "_BF16_K_LOAD",
    "QSA_DRAFT_FP8_SCHED": "_DRAFT_FP8_SCHED_DEFAULT",
    "QSA_DECG_FP8_SCHED": "_DECG_FP8_SCHED_DEFAULT",
    "QSA_PREFILL_ROUTE": "PREFILL_ROUTE_DEFAULT",
    "QSA_FP8_PREFILL_ROUTE": "FP8_PREFILL_ROUTE_DEFAULT",
}

#: Default-env per-dtype variant facts of the profiled kernel (82c4dec5).
DECG_WAVE_COLS = {"bf16": 32, "fp8": 64}  # QSA_DECODE_GROUP_COLS=auto
DECG_G3_WAVE_COLS = {"bf16": 32, "fp8": 32}  # ROUND-R2E: auto slab of group_rows=3
FP8_PK8 = "intrin"  # QSA_FP8_CVT_ASM=0
DRAFT_FP8_SCHED = KERNEL_ENV_DEFAULTS["QSA_DRAFT_FP8_SCHED"]
DECG_FP8_SCHED = KERNEL_ENV_DEFAULTS["QSA_DECG_FP8_SCHED"]

REGIMES = ("draft", "decode", "prefill", "tail")
CACHE_DTYPES = ("bf16", "fp8")
GOLDEN_STATUSES = ("pass", "pending", "fail")


class Reason(str, enum.Enum):
    """Why a call is (not) served by the fused scorer. Values are a stable API."""

    OK = "ok"
    # enablement / profile
    DISABLED = "disabled"  # no profile selected (default)
    PROFILE_NOT_FOUND = "profile_not_found"
    PROFILE_INVALID = "profile_invalid"
    PROFILE_KERNEL_MISMATCH = "profile_kernel_mismatch"  # kernel source != profiled sha
    KERNEL_ENV_MISMATCH = "kernel_env_mismatch"  # QSA_* switches != profiled values
    KERNEL_UNAVAILABLE = "kernel_unavailable"  # FlyDSL/kernel import failed
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
    CACHE_FORMAT = "cache_format_unsupported"  # mxfp8 / fp4 / other dtypes
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
    ROW_UNVERIFIED = "row_unverified"  # matched, golden status != pass
    ROUTE_MISMATCH = "route_mismatch"  # kernel router != row vid

    def __str__(self) -> str:
        return self.value


#: Reasons that depend only on persistent state (device, profile, kernel build,
#: cache tensors): evaluate once when a bucket is prepared (rank-uniform).
STATIC_REASONS = frozenset(
    {
        Reason.DISABLED,
        Reason.PROFILE_NOT_FOUND,
        Reason.PROFILE_INVALID,
        Reason.PROFILE_KERNEL_MISMATCH,
        Reason.KERNEL_ENV_MISMATCH,
        Reason.KERNEL_UNAVAILABLE,
        Reason.ARCH_UNSUPPORTED,
        Reason.CU_UNSUPPORTED,
        Reason.CACHE_FORMAT,
        Reason.CACHE_SHAPE,
        Reason.CACHE_LAYOUT,
        Reason.CACHE_ALIGNMENT,
        Reason.SCALE_LAYOUT,
    }
)


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
    tail:    requests < rows < 32*requests, rows%4!=0 (r16 WMMA kernels)
    None:    empty shapes only
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
    return "tail"


#: rows/requests relation of a range rule -> the regime it implies
RELATIONS = {
    "rows<=requests": "draft",
    "rows==requests": "draft",
    "rows==4*requests": "decode",
    "decode_band": "decode",  # requests < rows < 32*requests, rows % 4 == 0
    "tail_band": "tail",  # requests < rows < 32*requests, rows % 4 != 0
    "rows>=32*requests": "prefill",  # rows >= 32 and rows >= 32*requests
}


def relation_ok(relation: str, rows: int, requests: int) -> bool:
    """True when (rows, requests) satisfies ``relation`` (and hence its regime)."""
    rows, requests = int(rows), int(requests)
    if regime_of(rows, requests) != RELATIONS.get(relation):
        return False
    if relation == "rows==requests":
        return rows == requests
    if relation == "rows==4*requests":
        return rows == 4 * requests
    return True


PERSIST_MIN_COLUMNS = 64 * 8  # WMMA_COLS * QSA_PERSIST_COLS (default)


def variant_prefix(page_size: int = PAGE_SIZE) -> str:
    """``qsa.page4`` / ``qsa.page16``: the vid namespace of an index page size."""
    return f"qsa.page{int(page_size)}"


def page_of(entry: Mapping[str, Any]) -> int:
    """Index page size of a row / rule / plan entry (absent = 4)."""
    return int(entry.get("page_size", PAGE_SIZE))


#: ROUND-R2 launch knobs a row / rule / bucket may pin in ``params`` on top of its
#: variant: builder keywords of the draft / decode-group kernels (absent = kernel
#: default, i.e. the default-env launch). The router check (``route_mismatch``)
#: compares the variant WITHOUT them (:func:`split_knobs`); the plan then builds the
#: pinned specialization. ``group_rows`` 3 (kernel-native MTP-verify 3-row groups)
#: is bucket-only (rows == 3*requests); ``row_mode`` is never pinned by a profile
#: ('pad' changes invalid token_to_request semantics: op-internal padded calls only).
LAUNCH_KNOBS: dict[str, dict[str, tuple]] = {
    "draft": {"wg_waves": (4, 2, 1)},
    "decg": {"wg_waves": (4, 2, 1), "group_rows": (4, 3)},
}
KNOB_DEFAULTS = {"wg_waves": 4, "group_rows": 4}


def variant_kind(vid: str) -> str:
    """'draft' / 'decg' / ... : the variant name without page namespace and version."""
    parts = str(vid).split(".")
    return parts[2] if len(parts) >= 4 else ""


def split_knobs(vid: str, params: Mapping[str, Any]) -> tuple[dict, dict]:
    """(variant params, pinned launch knobs) of a row/rule/bucket ``params``."""
    names = LAUNCH_KNOBS.get(variant_kind(vid), {})
    base = {k: v for k, v in dict(params).items() if k not in names}
    knobs = {k: v for k, v in dict(params).items() if k in names}
    return base, knobs


def knob_errors(
    vid: str, params: Mapping[str, Any], allow_group_rows: bool = False
) -> list[str]:
    errs = []
    names = LAUNCH_KNOBS.get(variant_kind(vid), {})
    for k, v in split_knobs(vid, params)[1].items():
        if v not in names[k] or v == KNOB_DEFAULTS[k]:
            errs.append(f"knob {k}={v!r} (allowed non-default {names[k][1:]})")
        if k == "group_rows" and not allow_group_rows:
            errs.append("group_rows is bucket-only (kind group)")
    return errs


#: ROUND-R2E default prefill route tables (mirror of the kernel's 'r2e' presets
#: ``PREFILL_ROUTE_PRESETS`` / ``FP8_PREFILL_ROUTE_PRESETS``; CPU tests cross-check every
#: rule sample against the kernel router). bf16: lean ktile RB32 x C2 for columns <=
#: PREFILL_KT_MAX_COLUMNS and rows <= PREFILL_KT_MAX_ROWS, persist above (columns >= 512),
#: r32 TDM otherwise. fp8: lean+kconv ktile RB32 x C2 (rows <= FP8_KT_C2_MAX_ROWS) / x C4
#: (rows <= PREFILL_KT_MAX_ROWS) for columns <= PREFILL_KT_MAX_COLUMNS, r32 fp8 otherwise.
PREFILL_KT_MAX_COLUMNS = 2048
PREFILL_KT_MAX_ROWS = 16384
FP8_KT_C2_MAX_ROWS = 2048


def default_knobs(cache_dtype: str, regime: str, group_rows: int = 4) -> dict:
    """ROUND-R2E launch knobs the default-env kernel wrapper applies (and a rule / bucket
    therefore pins): fp8 draft wg_waves 2 (QSA_DRAFT_WG_WAVES=auto); decode-group
    group_rows 3 for kernel-native verify3 buckets."""
    out = {}
    if regime == "draft" and cache_dtype == "fp8":
        out["wg_waves"] = 2
    if regime == "decode" and int(group_rows) == 3:
        out["group_rows"] = 3
    return out


def expected_variant(
    cache_dtype: str,
    regime: str,
    columns: int,
    page_size: int = PAGE_SIZE,
    rows: int | None = None,
    group_rows: int = 4,
) -> tuple[str, dict]:
    """(vid, variant params) the default-env kernel router selects (no launch knobs:
    :func:`default_knobs`). ``rows`` matters for prefill only (ROUND-R2E fp8 ktile column
    tiles / the ktile row bound; ``None`` = 32); ``group_rows`` 3 = the kernel-native
    3-row decode-group launch (ROUND-R2E: 'auto' slab 32 for both dtypes)."""
    vid, params = _expected_variant4(cache_dtype, regime, columns, rows, group_rows)
    if int(page_size) != PAGE_SIZE:
        vid = variant_prefix(page_size) + vid[len("qsa.page4") :]
    return vid, params


def _expected_variant4(
    cache_dtype: str,
    regime: str,
    columns: int,
    rows: int | None = None,
    group_rows: int = 4,
) -> tuple[str, dict]:
    """(vid, params) the default-env kernel router selects for an aligned cache.

    fp8 variants carry the converter build (``pk8``) and, for the draft /
    decode-group kernels, the fp8 tile schedule (``sched``); bf16 kernels have
    neither (their ISA is unchanged by the 82c4dec5 fp8 work).
    """
    fp8 = cache_dtype == "fp8"
    if regime == "draft":
        params = {"k_load": "tdm", "wave_cols": 32}
        if fp8:
            params.update(sched=DRAFT_FP8_SCHED, pk8=FP8_PK8)
        return "qsa.page4.draft.v1", params
    if regime == "decode":
        wc = (
            DECG_G3_WAVE_COLS[cache_dtype]
            if int(group_rows) == 3
            else DECG_WAVE_COLS[cache_dtype]
        )
        params = {"k_load": "tdm", "wave_cols": wc, "barrier": False}
        if fp8:
            params.update(sched=DECG_FP8_SCHED, pk8=FP8_PK8)
        return "qsa.page4.decg.v1", params
    if regime == "prefill":
        r = 32 if rows is None else int(rows)
        kt = (
            int(columns) <= PREFILL_KT_MAX_COLUMNS and r <= PREFILL_KT_MAX_ROWS
        )  # ROUND-R2E
        if fp8:
            if kt:
                return "qsa.page4.r32_kt_fp8.v1", {
                    "k_load": "tdm",
                    "rows_per_block": 32,
                    "kt_rows": 32,
                    "col_tiles": 2 if r <= FP8_KT_C2_MAX_ROWS else 4,
                    "cvt": "fast",
                    "pk8": FP8_PK8,
                    "lean": True,
                    "kconv": True,
                }
            return "qsa.page4.r32_fp8.v1", {
                "k_load": "tdm",
                "rows_per_block": 32,
                "cvt": "fast",
                "pk8": FP8_PK8,
            }
        if kt:
            return "qsa.page4.r32_kt.v1", {
                "k_load": "tdm",
                "rows_per_block": 32,
                "kt_rows": 32,
                "col_tiles": 2,
                "lean": True,
            }
        if int(columns) >= PERSIST_MIN_COLUMNS:
            return "qsa.page4.r32_persist.v1", {
                "k_load": "tdm",
                "rows_per_block": 32,
                "persist_cols": 8,
                "buffers": 2,
            }
        return "qsa.page4.r32.v1", {"k_load": "tdm", "rows_per_block": 32}
    if regime == "tail":
        if fp8:  # fp8 r16 never uses TDM (_resolve_fp8_k_load)
            return "qsa.page4.r16_fp8.v1", {
                "k_load": "raw",
                "rows_per_block": 16,
                "pk8": FP8_PK8,
            }
        return "qsa.page4.r16.v1", {"k_load": "tdm", "rows_per_block": 16}
    raise ValueError(f"unknown regime {regime!r}")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
_SIG_KEYS = (
    "arch",
    "cu",
    "cache_dtype",
    "regime",
    "rows",
    "requests",
    "columns",
    "table_width",
)


def _sig_knobs(key: dict, entry: Mapping[str, Any]) -> None:
    """Pinned launch knobs join a signature only when present (R2), so every entry
    without knobs keeps its signature and evidence, and a re-pinned entry never
    inherits the golden record of another launch configuration."""
    knobs = split_knobs(entry.get("vid", ""), entry.get("params") or {})[1]
    if knobs:
        key["knobs"] = knobs


def row_signature(row: Mapping[str, Any]) -> str:
    """Stable 16-hex signature of a row's match keys (``sig`` field).

    ``page_size`` joins the key only when it is not 4, so every page-4 row keeps
    the signature (and hence the golden evidence) it had before ROUND-PAGE16.
    """
    key = {k: row[k] for k in _SIG_KEYS}
    if page_of(row) != PAGE_SIZE:
        key["page_size"] = page_of(row)
    _sig_knobs(key, row)
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
        return _range_ok(self.raw["rows"], rows) and _range_ok(
            self.raw["requests"], requests
        )


def validate_row(row: Mapping[str, Any]) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, f"row {row.get('cell', '?')}: {msg}")

    for key in (
        "sig",
        "arch",
        "cu",
        "cache_dtype",
        "regime",
        "rows",
        "requests",
        "columns",
        "table_width",
        "vid",
        "params",
        "cell",
        "golden",
        "generated_by",
        "notes",
    ):
        if key not in row:
            bad(f"missing {key!r}")
    if page_of(row) not in PAGE_SIZES:
        bad(f"page_size {row.get('page_size')!r} not in {PAGE_SIZES}")
    page = page_of(row)
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
    if int(row["columns"]) < 1 or int(row["columns"]) > int(row["table_width"]) * page:
        bad("columns outside [1, table_width*page_size]")
    if row["golden"].get("status") not in GOLDEN_STATUSES:
        bad(f"golden.status {row['golden'].get('status')!r}")
    if row["sig"] != row_signature(row):
        bad(f"sig {row['sig']} != {row_signature(row)}")
    # every (rows, requests) the row admits must be the row's own regime
    for r in {int(row["rows"]["min"]), int(row["rows"]["max"])}:
        for q in {int(row["requests"]["min"]), int(row["requests"]["max"])}:
            if _range_ok(row["rows"], r) and regime_of(r, q) not in (
                row["regime"],
                None,
            ):
                bad(
                    f"rows={r} requests={q} is regime {regime_of(r, q)}, not {row['regime']}"
                )
    for rr in sorted({int(row["rows"]["min"]), int(row["rows"]["max"])}):
        vid, params = expected_variant(
            row["cache_dtype"], row["regime"], int(row["columns"]), page, rows=rr
        )
        if row["vid"] != vid or split_knobs(vid, row["params"])[0] != params:
            bad(
                f"variant {row['vid']} {row['params']} != router {vid} {params} at rows={rr}"
            )
    for e in knob_errors(vid, row["params"]):
        bad(e)


# ---------------------------------------------------------------------------
# Range rules (schema 2)
# ---------------------------------------------------------------------------
_RULE_SIG_KEYS = (
    "arch",
    "cu",
    "cache_dtype",
    "regime",
    "relation",
    "rows",
    "requests",
    "columns",
    "table_width",
    "page_size",
    "divisor",
)
RULE_DIVISOR = "sqrt(head_dim)"
RULE_KEYS = (
    "id",
    "sig",
    "arch",
    "cu",
    "cache_dtype",
    "regime",
    "relation",
    "rows",
    "requests",
    "columns",
    "table_width",
    "page_size",
    "divisor",
    "vid",
    "params",
    "samples",
    "golden",
    "generated_by",
    "notes",
)
#: optional match keys (part of the signature when present)
RULE_OPTIONAL_SIG_KEYS = ("rows_x_columns_max",)


def rule_signature(rule: Mapping[str, Any]) -> str:
    """Stable 16-hex signature of a rule's match keys (``sig`` field)."""
    key = {k: rule[k] for k in _RULE_SIG_KEYS}
    key.update({k: rule[k] for k in RULE_OPTIONAL_SIG_KEYS if k in rule})
    _sig_knobs(key, rule)
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def column_values(spec: Mapping[str, Any]) -> list[int]:
    """Every column count a rule's ``columns`` spec admits (ascending)."""
    if "set" in spec:
        return sorted({int(c) for c in spec["set"]})
    lo, hi = int(spec["pow2"]["min"]), int(spec["pow2"]["max"])
    out, c = [], 1
    while c <= hi:
        if c >= lo:
            out.append(c)
        c *= 2
    return out


def columns_admitted(spec: Mapping[str, Any], columns: int) -> bool:
    columns = int(columns)
    if "set" in spec:
        return columns in {int(c) for c in spec["set"]}
    rng = spec["pow2"]
    return (
        int(rng["min"]) <= columns <= int(rng["max"])
        and columns > 0
        and columns & (columns - 1) == 0
    )


def sample_key(s: Mapping[str, Any]) -> tuple:
    return (
        int(s["rows"]),
        int(s["requests"]),
        int(s["columns"]),
        int(s["table_width"]),
    )


@dataclass(frozen=True)
class ProfileRule:
    raw: Mapping[str, Any]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def sig(self) -> str:
        return self.raw["sig"]

    @property
    def cell(self) -> str:  # Decision.cell of a rule match
        return f"rule:{self.raw['id']}"

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
        return (
            _range_ok(self.raw["rows"], rows)
            and _range_ok(self.raw["requests"], requests)
            and relation_ok(self.raw["relation"], rows, requests)
        )

    def width_ok(self, table_width: int) -> bool:
        return _range_ok(self.raw["table_width"], table_width)

    def columns_ok(self, columns: int, table_width: int) -> bool:
        return columns_admitted(self.raw["columns"], columns) and int(columns) <= int(
            table_width
        ) * page_of(self.raw)

    def rows_max_at(self, columns: int) -> int:
        """Largest rows the rule admits at ``columns`` (``rows_x_columns_max`` bound)."""
        hi = int(self.raw["rows"]["max"])
        rxc = self.raw.get("rows_x_columns_max")
        return hi if rxc is None else min(hi, int(rxc) // max(int(columns), 1))

    def plane_ok(self, rows: int, columns: int) -> bool:
        return int(rows) <= self.rows_max_at(columns)

    def admits(self, rows: int, requests: int, columns: int, table_width: int) -> bool:
        return (
            self.shape_ok(rows, requests)
            and self.width_ok(table_width)
            and self.columns_ok(columns, table_width)
            and self.plane_ok(rows, columns)
        )


def validate_rule(rule: Mapping[str, Any], kernel_sha256: str | None = None) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, f"rule {rule.get('id', '?')}: {msg}")

    for key in RULE_KEYS:
        if key not in rule:
            bad(f"missing {key!r}")
    if rule["cache_dtype"] not in CACHE_DTYPES:
        bad(f"cache_dtype {rule['cache_dtype']!r}")
    if rule["relation"] not in RELATIONS:
        bad(f"relation {rule['relation']!r} not in {sorted(RELATIONS)}")
    if rule["regime"] != RELATIONS[rule["relation"]]:
        bad(
            f"regime {rule['regime']!r} != relation {rule['relation']} regime "
            f"{RELATIONS[rule['relation']]!r}"
        )
    for key in ("rows", "requests", "table_width"):
        rng = rule[key]
        if not isinstance(rng, Mapping) or "min" not in rng or "max" not in rng:
            bad(f"{key} must be {{min, max[, multiple_of]}}")
        if not 1 <= int(rng["min"]) <= int(rng["max"]):
            bad(f"{key} range {rng}")
    if int(rule["rows"]["max"]) > MAX_ROWS:
        bad("rows above the kernel grid limit")
    if int(rule["page_size"]) not in PAGE_SIZES or rule["divisor"] != RULE_DIVISOR:
        bad(
            f"page_size/divisor {rule['page_size']}/{rule['divisor']} not in "
            f"{PAGE_SIZES}/{RULE_DIVISOR}"
        )
    page = int(rule["page_size"])
    cs = rule["columns"]
    if not isinstance(cs, Mapping) or len(cs) != 1 or not ("set" in cs or "pow2" in cs):
        bad("columns must be {'pow2': {min, max}} or {'set': [...]}")
    cols = column_values(cs)
    if not cols or cols[0] < 1:
        bad(f"columns {cs} admit nothing")
    if cols[0] > int(rule["table_width"]["max"]) * page:
        bad("no admitted column fits table_width.max*page_size")
    unknown = set(rule) - set(RULE_KEYS) - set(RULE_OPTIONAL_SIG_KEYS)
    if unknown:
        bad(f"unknown keys {sorted(unknown)}")
    if "rows_x_columns_max" in rule:
        rxc = rule["rows_x_columns_max"]
        if isinstance(rxc, bool) or not isinstance(rxc, int) or rxc < 1:
            bad(f"rows_x_columns_max {rxc!r} must be a positive int")
    r = ProfileRule(rule)
    for (
        c
    ) in cols:  # every admitted column admits rows.min and fits int32 byte addressing
        if r.rows_max_at(c) < int(rule["rows"]["min"]):
            bad(f"rows_x_columns_max leaves no rows at columns={c}")
        if r.rows_max_at(c) * c * 4 > MAX_PLANE_BYTES:
            bad(
                f"rows {r.rows_max_at(c)} x columns {c} logits plane exceeds int32 byte "
                "addressing"
            )
    for c in cols:  # one variant for the whole rule (the router may not split it)
        for rr in sorted({int(rule["rows"]["min"]), r.rows_max_at(c)}):
            vid, params = expected_variant(
                rule["cache_dtype"], rule["regime"], c, page, rows=rr
            )
            if rule["vid"] != vid or split_knobs(vid, rule["params"])[0] != params:
                bad(
                    f"variant {rule['vid']} {rule['params']} != router {vid} {params} at "
                    f"columns={c} rows={rr}"
                )
    for e in knob_errors(rule["vid"], rule["params"]):
        bad(e)
    if rule["sig"] != rule_signature(rule):
        bad(f"sig {rule['sig']} != {rule_signature(rule)}")
    samples = rule["samples"]
    if not samples:
        bad("no golden samples")
    keys = set()
    for s in samples:
        k = sample_key(s)
        if not r.admits(*k):
            bad(f"sample {k} outside the rule")
        if k in keys:
            bad(f"duplicate sample {k}")
        keys.add(k)
    g = rule["golden"]
    if g.get("status") not in GOLDEN_STATUSES:
        bad(f"golden.status {g.get('status')!r}")
    if g["status"] == "pass":
        vsha = g.get("kernel_sha256")
        if not vsha:
            bad("pass without golden.kernel_sha256")
        if (
            kernel_sha256 is not None
            and vsha != kernel_sha256
            and not g.get("isa_identity")
        ):
            bad(
                f"pass verified on kernel {str(vsha)[:12]} != profiled {str(kernel_sha256)[:12]} "
                "without golden.isa_identity"
            )
        checked = {sample_key(c): c for c in g.get("checked") or []}
        missing = sorted(keys - set(checked))
        if missing:
            bad(f"pass without a golden record for samples {missing[:4]}")
        failed = sorted(k for k in keys if not checked[k].get("pass"))
        if failed:
            bad(f"pass with failing samples {failed[:4]}")


# ---------------------------------------------------------------------------
# Buckets (R2, optional top-level ``buckets`` list; opt-in at call time)
# ---------------------------------------------------------------------------
# A bucket maps a call class onto a padded *target* shape that an exact row or
# range rule serves with the decode-group kernel:
#
# * kind ``group``  -- ``relation`` ``rows==3*requests`` (MTP verify bench3, rows of a
#   request contiguous): every request's ``group.real`` rows are padded to
#   ``group.padded`` (= 4, the decode-group kernel's row group); target rows =
#   ``padded * requests`` (``rows==4*requests``, decode regime). The op pads through
#   pre-allocated staging planes and returns exactly the real rows. R2-MERGE: a group
#   bucket with ``group.kernel_group_rows`` = 3 and ``params.group_rows`` = 3 instead
#   launches the decode-group kernel's native 3-row groups on the caller's own
#   tensors (zero copy; preferred, listed first); the staging bucket stays as the
#   fallback for kernels without ``DECG_REAL_ROWS_PER_GROUP``.
# * kind ``rows``   -- ``relation`` ``tail_band`` (rows % 4 != 0): rows rounded up to a
#   multiple of ``round_to`` (4) while staying in the decode band; the kernel runs
#   with the real row count as its row bound (it masks the partial last group), so
#   nothing is padded and nothing beyond the real rows is written.
#
# Match keys as rules (arch, cu, cache_dtype, page_size, requests, rows, columns,
# table_width); ``vid``/``params`` = the target variant; ``samples`` + ``golden`` as
# rules (a bucket is used only when it is ``pass`` AND its target shape matches a
# verified row/rule).
BUCKET_KINDS = {"group": "rows==3*requests", "rows": "tail_band"}
BUCKET_KEYS = (
    "id",
    "sig",
    "kind",
    "arch",
    "cu",
    "cache_dtype",
    "relation",
    "rows",
    "requests",
    "columns",
    "table_width",
    "page_size",
    "divisor",
    "target_regime",
    "vid",
    "params",
    "samples",
    "golden",
    "generated_by",
    "notes",
)
BUCKET_OPTIONAL_KEYS = ("group", "round_to")
_BUCKET_SIG_KEYS = (
    "kind",
    "arch",
    "cu",
    "cache_dtype",
    "relation",
    "rows",
    "requests",
    "columns",
    "table_width",
    "page_size",
    "divisor",
    "target_regime",
)


def bucket_signature(b: Mapping[str, Any]) -> str:
    key = {k: b[k] for k in _BUCKET_SIG_KEYS}
    key.update({k: b[k] for k in BUCKET_OPTIONAL_KEYS if k in b})
    _sig_knobs(key, b)
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def bucket_target(
    b: Mapping[str, Any], rows: int, requests: int
) -> tuple[int, tuple[int, int]] | None:
    """(target rows, (lowest, highest) caller rows one decision may serve) or None."""
    rows, requests = int(rows), int(requests)
    if b["kind"] == "group":
        real, padded = int(b["group"]["real"]), int(b["group"]["padded"])
        if rows != real * requests:
            return None
        t = padded * requests
        return (
            (t, (rows, rows)) if regime_of(t, requests) == b["target_regime"] else None
        )
    if b["kind"] == "rows":
        m = int(b["round_to"])
        t = -(-rows // m) * m
        if regime_of(t, requests) != b["target_regime"]:
            return None
        return t, (max(t - m + 1, requests + 1), t)
    return None


@dataclass(frozen=True)
class ProfileBucket:
    raw: Mapping[str, Any]

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def cell(self) -> str:
        return f"bucket:{self.raw['id']}"

    @property
    def vid(self) -> str:
        return self.raw["vid"]

    @property
    def params(self) -> Mapping[str, Any]:
        return self.raw["params"]

    @property
    def verified(self) -> bool:
        return self.raw["golden"]["status"] == "pass"

    def admits(self, rows: int, requests: int, columns: int, table_width: int) -> bool:
        b = self.raw
        rel = b["relation"]
        if rel == "rows==3*requests":
            rel_ok = int(rows) == 3 * int(requests)
        else:
            rel_ok = relation_ok(rel, rows, requests)
        return (
            rel_ok
            and _range_ok(b["rows"], rows)
            and _range_ok(b["requests"], requests)
            and _range_ok(b["table_width"], table_width)
            and columns_admitted(b["columns"], columns)
            and int(columns) <= int(table_width) * page_of(b)
            and bucket_target(b, rows, requests) is not None
        )


def validate_bucket(b: Mapping[str, Any], kernel_sha256: str | None = None) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, f"bucket {b.get('id', '?')}: {msg}")

    for key in BUCKET_KEYS:
        if key not in b:
            bad(f"missing {key!r}")
    unknown = set(b) - set(BUCKET_KEYS) - set(BUCKET_OPTIONAL_KEYS)
    if unknown:
        bad(f"unknown keys {sorted(unknown)}")
    if b["kind"] not in BUCKET_KINDS or b["relation"] != BUCKET_KINDS[b["kind"]]:
        bad(f"kind/relation {b['kind']}/{b['relation']}")
    if b["kind"] == "group":
        g = b.get("group") or {}
        if not (1 <= int(g.get("real", 0)) < int(g.get("padded", 0)) == 4):
            bad(f"group {g} (real < padded == 4)")
        if int(g["real"]) != 3:
            bad("only rows==3*requests groups are defined")
        if set(g) - {"real", "padded", "kernel_group_rows"}:
            bad(f"group keys {sorted(g)}")
        kg = g.get("kernel_group_rows")
        if kg is not None and int(kg) != int(g["real"]):
            bad(f"group.kernel_group_rows {kg} != group.real {g['real']}")
        if split_knobs(b["vid"], b["params"])[1].get("group_rows") != kg:
            bad(
                f"params group_rows {dict(b['params']).get('group_rows')} != "
                f"group.kernel_group_rows {kg}"
            )
    else:
        if int(b.get("round_to", 0)) != 4:
            bad("round_to must be 4 (decode-group row group)")
    if b["target_regime"] != "decode":
        bad("target_regime must be decode (the decode-group kernel)")
    if b["cache_dtype"] not in CACHE_DTYPES:
        bad(f"cache_dtype {b['cache_dtype']!r}")
    if int(b["page_size"]) not in PAGE_SIZES or b["divisor"] != RULE_DIVISOR:
        bad(f"page_size/divisor {b['page_size']}/{b['divisor']}")
    for key in ("rows", "requests", "table_width"):
        rng = b[key]
        if not isinstance(rng, Mapping) or not 1 <= int(rng["min"]) <= int(rng["max"]):
            bad(f"{key} range {rng}")
    for c in column_values(b["columns"]):
        vid, params = expected_variant(
            b["cache_dtype"],
            "decode",
            c,
            int(b["page_size"]),
            group_rows=int(dict(b["params"]).get("group_rows", 4)),
        )
        if b["vid"] != vid or split_knobs(vid, b["params"])[0] != params:
            bad(f"variant {b['vid']} {b['params']} != decode-group {vid} {params}")
    for e in knob_errors(b["vid"], b["params"], allow_group_rows=b["kind"] == "group"):
        bad(e)
    if b["sig"] != bucket_signature(b):
        bad(f"sig {b['sig']} != {bucket_signature(b)}")
    R = ProfileBucket(b)
    keys = set()
    for s in b["samples"]:
        k = sample_key(s)
        if not R.admits(*k):
            bad(f"sample {k} outside the bucket")
        if k in keys:
            bad(f"duplicate sample {k}")
        keys.add(k)
    g = b["golden"]
    if g.get("status") not in GOLDEN_STATUSES:
        bad(f"golden.status {g.get('status')!r}")
    if g["status"] == "pass":
        if not g.get("kernel_sha256"):
            bad("pass without golden.kernel_sha256")
        if (
            kernel_sha256 is not None
            and g["kernel_sha256"] != kernel_sha256
            and not g.get("isa_identity")
        ):
            bad("pass verified on another kernel without golden.isa_identity")
        checked = {sample_key(c): c for c in g.get("checked") or []}
        if keys - set(checked) or any(not checked[k].get("pass") for k in keys):
            bad("pass without a passing golden record for every sample")


@dataclass
class Profile:
    name: str
    path: str
    data: Mapping[str, Any]
    rows: list[ProfileRow] = field(default_factory=list)
    rules: list[ProfileRule] = field(default_factory=list)
    buckets: list[ProfileBucket] = field(default_factory=list)
    _index: dict = field(default_factory=dict, repr=False)
    _rule_index: dict = field(default_factory=dict, repr=False)
    _bucket_index: dict = field(default_factory=dict, repr=False)
    _sets: dict = field(default_factory=dict, repr=False)

    def bucket_candidates(
        self, arch: str, cu: int, cache_dtype: str, page_size: int = PAGE_SIZE
    ) -> list[ProfileBucket]:
        return self._bucket_index.get((arch, int(cu), cache_dtype, int(page_size)), [])

    @property
    def kernel_sha256(self) -> str:
        return self.data["kernel"]["sha256"]

    @property
    def kernel_env(self) -> Mapping[str, str]:
        return self.data["kernel_env"]

    def candidates(
        self,
        arch: str,
        cu: int,
        cache_dtype: str,
        regime: str,
        page_size: int = PAGE_SIZE,
    ) -> list[ProfileRow]:
        return self._index.get((arch, int(cu), cache_dtype, int(page_size), regime), [])

    def rule_candidates(
        self,
        arch: str,
        cu: int,
        cache_dtype: str,
        regime: str,
        page_size: int = PAGE_SIZE,
    ) -> list[ProfileRule]:
        return self._rule_index.get(
            (arch, int(cu), cache_dtype, int(page_size), regime), []
        )

    def page_sizes(self) -> set:
        """Index page sizes any row or rule of the profile serves."""
        s = self._sets.get("page_sizes")
        if s is None:
            s = self._sets["page_sizes"] = frozenset(
                page_of(r.raw) for r in [*self.rows, *self.rules]
            )
        return s

    def archs(self) -> set:
        s = self._sets.get(None)
        if (
            s is None
        ):  # rows/rules are fixed after profile_from_data: memoise (hot path)
            s = self._sets[None] = frozenset(
                r.raw["arch"] for r in [*self.rows, *self.rules]
            )
        return s

    def cus(self, arch: str) -> set:
        s = self._sets.get(arch)
        if s is None:
            s = self._sets[arch] = frozenset(
                int(r.raw["cu"])
                for r in [*self.rows, *self.rules]
                if r.raw["arch"] == arch
            )
        return s


def validate_profile(data: Mapping[str, Any], name: str | None = None) -> None:
    def bad(msg):
        raise ProfileError(Reason.PROFILE_INVALID, msg)

    for key in (
        "schema_version",
        "feature",
        "profile",
        "api_version",
        "backend",
        "kernel",
        "kernel_env",
        "contract",
        "golden",
        "generated_by",
        "rows",
    ):
        if key not in data:
            bad(f"missing top-level {key!r}")
    if data["schema_version"] not in SUPPORTED_SCHEMAS:
        bad(f"schema_version {data['schema_version']} not in {SUPPORTED_SCHEMAS}")
    if data["schema_version"] >= 2 and not isinstance(data.get("rules"), list):
        bad("schema 2 requires a 'rules' list")
    if data["schema_version"] < 2 and data.get("rules"):
        bad("rules need schema_version 2")
    if data["feature"] != FEATURE or data["backend"] != BACKEND:
        bad(f"feature/backend {data['feature']}/{data['backend']}")
    if str(data["api_version"]).split(".")[0] != API_VERSION.split(".")[0]:
        bad(f"api_version {data['api_version']} incompatible with {API_VERSION}")
    if name is not None and data["profile"] != name:
        bad(f"profile name {data['profile']!r} != file name {name!r}")
    if set(data["kernel_env"]) != set(KERNEL_ENV_DEFAULTS):
        bad("kernel_env keys differ from KERNEL_ENV_DEFAULTS")
    c = data["contract"]
    if (
        c.get("index_heads"),
        c.get("head_dim"),
        c.get("page_size"),
        c.get("compress_ratio"),
    ) != (INDEX_HEADS, HEAD_DIM, PAGE_SIZE, COMPRESS_RATIO):
        bad(f"contract geometry {c}")
    if not set(c.get("page_sizes", [PAGE_SIZE])) <= set(PAGE_SIZES):
        bad(f"contract page_sizes {c.get('page_sizes')} not within {PAGE_SIZES}")
    sigs = set()
    ksha = data["kernel"].get("sha256")
    for e in [*data["rows"], *(data.get("rules") or []), *(data.get("buckets") or [])]:
        # s_prefetch_inst (ipf) kernel variants fault on GPU: a profile cannot route them
        if "ipf" in (e.get("params") or {}):
            bad(f"{e.get('cell') or e.get('id')}: ipf variants are not routable")
    for row in data["rows"]:
        validate_row(row)
        g = row["golden"]
        if g["status"] == "pass":
            # every pass names the kernel it was GPU-verified on; evidence from an
            # older kernel is only valid with a recorded ISA-identity justification
            vsha = g.get("kernel_sha256")
            if not vsha:
                bad(f"row {row['cell']}: pass without golden.kernel_sha256")
            if vsha != ksha and not g.get("isa_identity"):
                bad(
                    f"row {row['cell']}: pass verified on kernel {str(vsha)[:12]} != profiled "
                    f"{str(ksha)[:12]} without golden.isa_identity"
                )
        if row["sig"] in sigs:
            bad(f"duplicate row sig {row['sig']} ({row['cell']})")
        sigs.add(row["sig"])
    ids = set()
    for rule in data.get("rules") or []:
        validate_rule(rule, ksha)
        if rule["id"] in ids or rule["sig"] in sigs:
            bad(f"duplicate rule id/sig {rule['id']} {rule['sig']}")
        ids.add(rule["id"])
        sigs.add(rule["sig"])
    if "buckets" in data and not isinstance(data["buckets"], list):
        bad("'buckets' must be a list")
    for b in data.get("buckets") or []:
        validate_bucket(b, ksha)
        if b["id"] in ids or b["sig"] in sigs:
            bad(f"duplicate bucket id/sig {b['id']} {b['sig']}")
        ids.add(b["id"])
        sigs.add(b["sig"])


def resolve_profile_path(name_or_path: str) -> str:
    if os.sep in name_or_path or "/" in name_or_path or name_or_path.endswith(".json"):
        return os.path.abspath(name_or_path)
    return os.path.join(CONFIG_DIR, f"{name_or_path}.json")


def profile_from_data(
    data: Mapping[str, Any], path: str = "<memory>", name: str | None = None
) -> Profile:
    """Validate a profile dict and build its matcher indexes."""
    validate_profile(data, name)
    prof = Profile(
        name=data["profile"],
        path=path,
        data=data,
        rows=[ProfileRow(r) for r in data["rows"]],
        rules=[ProfileRule(r) for r in data.get("rules") or []],
        buckets=[ProfileBucket(b) for b in data.get("buckets") or []],
    )
    for b in prof.buckets:
        key = (b.raw["arch"], int(b.raw["cu"]), b.raw["cache_dtype"], page_of(b.raw))
        prof._bucket_index.setdefault(key, []).append(b)
    for row in prof.rows:
        key = (
            row.raw["arch"],
            int(row.raw["cu"]),
            row.raw["cache_dtype"],
            page_of(row.raw),
            row.raw["regime"],
        )
        prof._index.setdefault(key, []).append(row)
    for rule in prof.rules:
        key = (
            rule.raw["arch"],
            int(rule.raw["cu"]),
            rule.raw["cache_dtype"],
            page_of(rule.raw),
            rule.raw["regime"],
        )
        prof._rule_index.setdefault(key, []).append(rule)
    return prof


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
    prof = profile_from_data(data, path, stem)
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
    return {
        k: (v, effective.get(k))
        for k, v in profile.kernel_env.items()
        if effective.get(k) != v
    }


@dataclass(frozen=True)
class Match:
    ok: bool
    reason: Reason
    detail: str = ""
    row: ProfileRow | ProfileRule | None = None  # the exact row or range rule selected


#: how far a failed match got (deeper = more specific reason)
_DEPTH = {
    Reason.ROWS_RULE: 0,
    Reason.TABLE_WIDTH: 1,
    Reason.COLUMNS: 2,
    Reason.ROW_UNVERIFIED: 3,
}


def _select(kind: str, by_cols: list, allow_unverified: bool) -> Match:
    verified = [r for r in by_cols if r.verified]
    if verified:
        return Match(True, Reason.OK, verified[0].cell, verified[0])
    if allow_unverified:
        return Match(
            True, Reason.OK, f"{by_cols[0].cell} (unverified, allowed)", by_cols[0]
        )
    return Match(
        False,
        Reason.ROW_UNVERIFIED,
        f"{kind} {by_cols[0].cell}: golden status {by_cols[0].raw['golden']['status']}",
        by_cols[0],
    )


def _match_rows(
    profile: Profile,
    arch,
    cu,
    cache_dtype,
    regime,
    rows,
    requests,
    columns,
    table_width,
    allow_unverified,
    page_size=PAGE_SIZE,
) -> Match:
    cands = [
        r
        for r in profile.candidates(arch, cu, cache_dtype, regime, page_size)
        if r.shape_ok(rows, requests)
    ]
    if not cands:
        return Match(
            False,
            Reason.ROWS_RULE,
            f"{cache_dtype} page{page_size} {regime} rows={rows} requests={requests}: "
            "no row",
        )
    by_width = [r for r in cands if int(r.raw["table_width"]) == int(table_width)]
    if not by_width:
        widths = sorted({int(r.raw["table_width"]) for r in cands})
        return Match(
            False, Reason.TABLE_WIDTH, f"table_width={table_width} not in rows {widths}"
        )
    by_cols = [r for r in by_width if int(r.raw["columns"]) == int(columns)]
    if not by_cols:
        cols = sorted({int(r.raw["columns"]) for r in by_width})
        return Match(False, Reason.COLUMNS, f"columns={columns} not in rows {cols}")
    return _select("row", by_cols, allow_unverified)


def _match_rules(
    profile: Profile,
    arch,
    cu,
    cache_dtype,
    regime,
    rows,
    requests,
    columns,
    table_width,
    allow_unverified,
    page_size=PAGE_SIZE,
) -> Match:
    cands = [
        r
        for r in profile.rule_candidates(arch, cu, cache_dtype, regime, page_size)
        if r.shape_ok(rows, requests)
    ]
    if not cands:
        return Match(
            False,
            Reason.ROWS_RULE,
            f"{cache_dtype} page{page_size} {regime} rows={rows} requests={requests}: "
            "no rule",
        )
    by_width = [r for r in cands if r.width_ok(table_width)]
    if not by_width:
        return Match(
            False,
            Reason.TABLE_WIDTH,
            f"table_width={table_width} outside rules " f"{[r.id for r in cands]}",
        )
    by_cols = [r for r in by_width if r.columns_ok(columns, table_width)]
    if not by_cols:
        return Match(
            False,
            Reason.COLUMNS,
            f"columns={columns} outside rules " f"{[r.id for r in by_width]}",
        )
    by_plane = [r for r in by_cols if r.plane_ok(rows, columns)]
    if not by_plane:  # rows bound tied to the column count (rows_x_columns_max)
        return Match(
            False,
            Reason.COLUMNS,
            f"rows={rows} x columns={columns} above "
            f"rows_x_columns_max of rules {[r.id for r in by_cols]}",
        )
    return _select("rule", by_plane, allow_unverified)


def match(
    profile: Profile,
    *,
    arch: str,
    cu: int,
    cache_dtype: str,
    rows: int,
    requests: int,
    columns: int,
    table_width: int,
    allow_unverified: bool = False,
    page_size: int = PAGE_SIZE,
) -> Match:
    """Match a (validated) call shape: exact rows first, then verified range rules.

    An exact row that serves the call always wins over a rule. When neither
    serves it, the reason of whichever got further (rows -> table_width ->
    columns -> golden status) is returned; ties report the exact-row reason.
    """
    if arch not in profile.archs():
        return Match(
            False,
            Reason.ARCH_UNSUPPORTED,
            f"arch {arch} not in {sorted(profile.archs())}",
        )
    if int(cu) not in profile.cus(arch):
        return Match(
            False, Reason.CU_UNSUPPORTED, f"cu {cu} not in {sorted(profile.cus(arch))}"
        )
    if int(page_size) not in PAGE_SIZES:
        return Match(
            False,
            Reason.CACHE_SHAPE,
            f"index page size {page_size} not in {PAGE_SIZES}",
        )
    regime = regime_of(rows, requests)
    if regime is None:
        return Match(
            False, Reason.ROWS_RULE, f"rows={rows} requests={requests}: empty shape"
        )
    a = (
        profile,
        arch,
        cu,
        cache_dtype,
        regime,
        int(rows),
        int(requests),
        int(columns),
        int(table_width),
        allow_unverified,
        int(page_size),
    )
    m_row = _match_rows(*a)
    if m_row.ok:
        return m_row
    m_rule = _match_rules(*a)
    if m_rule.ok:
        return m_rule
    return m_rule if _DEPTH[m_rule.reason] > _DEPTH[m_row.reason] else m_row


@dataclass(frozen=True)
class BucketMatch:
    bucket: ProfileBucket
    target_rows: int
    call_rows: tuple[int, int]
    target: Match  # match of the padded target shape (ok)


def match_bucket(
    profile: Profile,
    *,
    arch: str,
    cu: int,
    cache_dtype: str,
    rows: int,
    requests: int,
    columns: int,
    table_width: int,
    allow_unverified: bool = False,
    page_size: int = PAGE_SIZE,
    native_group_rows: tuple | None = None,
) -> BucketMatch | None:
    """First bucket that admits the call, is verified (or ``allow_unverified``) and whose
    target shape matches a row/rule serving the bucket's variant (launch knobs aside);
    else None (the caller then matches the call itself, unpadded).

    ``native_group_rows`` (R2): real rows per group the loaded kernel launches natively
    (``DECG_REAL_ROWS_PER_GROUP``); a kernel-native group bucket
    (``group.kernel_group_rows``) is skipped when its value is not in it (``None`` =
    no capability filter), so the next (staging) bucket of the class applies."""
    for b in profile.bucket_candidates(arch, cu, cache_dtype, page_size):
        if not (b.verified or allow_unverified) or not b.admits(
            rows, requests, columns, table_width
        ):
            continue
        if (
            b.raw.get("golden", {}).get("status") == "fail"
        ):  # ROUND-R2E: never a failed bucket
            continue
        kg = (b.raw.get("group") or {}).get("kernel_group_rows")
        if (
            kg is not None
            and native_group_rows is not None
            and int(kg) not in native_group_rows
        ):
            continue
        tgt = bucket_target(b.raw, rows, requests)
        m = match(
            profile,
            arch=arch,
            cu=cu,
            cache_dtype=cache_dtype,
            rows=tgt[0],
            requests=requests,
            columns=columns,
            table_width=table_width,
            allow_unverified=allow_unverified,
            page_size=page_size,
        )
        tb, bb = split_knobs(b.vid, m.row.params)[0] if m.ok else None, split_knobs(
            b.vid, b.params
        )[0]
        if kg is not None and tb is not None:
            # ROUND-R2E: a kernel-native group launch uses the router's group_rows slab
            # (fp8 g3 w32 vs the 4-row fp8 decode w64): the target serves the same variant
            # up to wave_cols (the call-time guard checks the real-rows slab)
            tb = {k: v for k, v in tb.items() if k != "wave_cols"}
            bb = {k: v for k, v in bb.items() if k != "wave_cols"}
        if m.ok and m.row.vid == b.vid and tb == bb:
            return BucketMatch(b, tgt[0], tgt[1], m)
    return None


def iter_reasons() -> Iterable[str]:
    return (r.value for r in Reason)
