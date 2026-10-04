#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Merge one or more `mha_tuned_*.csv` files into a single JSON that can be
plugged into CK's `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`, while also
embedding runtime-dispatch information as `cpp_constraint` on each tile.

Design (approved 2026-08-25, see history):

1. Overall approach = B1: we do NOT modify CK codegen. The JSON stays fully
   forward-compatible with `_parse_tune_config_tiles` in CK's
   `codegen/ops/fmha_fwd.py` (unknown fields on each tile are silently
   ignored). `cpp_constraint` is therefore transparent to CK codegen; it is
   meant to be consumed by the host-side dispatcher in aiter (or any
   downstream integrator) at runtime.

2. Tile bucket key uses `best_hdim_q, best_hdim_v` (i.e. the *compiled*
   hdim, 80/96 or 256/256), NOT the raw CSV hdim_q/hdim_v (72/72 or 256/256),
   because CK codegen keys tiles on compiled hdim.

3. Buckets: for each source CSV, sort rows by max_seqlen ascending; the
   boundary between two neighboring rows i / i+1 is
   `(M_i + M_{i+1} + 1) // 2` (integer midpoint rounded up, so that
   M_i < mid_i <= M_{i+1} and every sample M_i lies inside its own
   interval, even for adjacent samples such as 512 / 513). The i-th row
   therefore owns the half-open interval
   `[low_i, high_i)`:
     * low_0    = -inf (represented as absence of the lower bound)
     * low_i    = mid_{i-1}         for i >= 1
     * high_i   = mid_i             for i <  N-1
     * high_{N-1} = +inf            (absence of the upper bound)
   Consecutive rows sharing the same tile are folded into one big interval.
   Runs of the same tile that are non-consecutive (tile-A, tile-B, tile-A)
   produce two intervals on tile-A that are OR-ed together in its
   cpp_constraint.

4. Kernel signatures must agree inside a bucket. Tiles of one bucket are
   picked at runtime by `cpp_constraint`, which can only talk about
   `max_seqlen_q`; two different signatures (mask_type, nhead, bias, lse,
   dropout, raw hdim, ...) landing in the same bucket are therefore
   undecidable at runtime, and the merge is rejected instead of silently
   mixing them. The same holds for a `max_seqlen` tuned twice: such shapes
   belong to one group and must be tuned together (see
   `mha_count_shape.py group`).

5. The output does not depend on the `--in` order: buckets, dtypes and
   `meta.sources` are sorted, so two runs over the same set of CSVs differ
   only in `meta.generated_at`.

6. Output path is mandatory: `--out <path>` must be provided, typically named
   after the model (e.g. `caption5p1_gfx942.json`).

Usage:
    python mha_gen_runtime_json.py \\
        --in  mha_tuned_0_group_bf16_hq72_hv72_mask0.csv \\
        --in  mha_tuned_1_group_bf16_hq256_hv256_mask2.csv \\
        --out caption5p1_gfx942.json \\
        --target gfx942
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

# CK's FmhaFwdTileSize field order (19 fields), must stay in sync with
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py
# (see FmhaFwdTileSize dataclass) and mha_tune.py's TileSize._ORDERED_FIELDS.
_ORDERED_FIELDS: tuple[str, ...] = (
    "F_bm0",
    "F_bn0",
    "F_bk0",
    "F_bn1",
    "F_bk1",
    "F_bk0max",
    "F_rm0",
    "F_rn0",
    "F_rk0",
    "F_rm1",
    "F_rn1",
    "F_rk1",
    "F_wm0",
    "F_wn0",
    "F_wk0",
    "F_wm1",
    "F_wn1",
    "F_wk1",
    "F_occupancy",
)

# Regex to pull the 19 ints out of a `best_tile_expr` string of the form:
#   (80, 96) : [FmhaFwdTileSize(128,  64,  64,  96,  32, 128,  4, 1, 1,  4, 1, 1,  32, 32, 16,  32, 32, 16,   2)]
_TILE_EXPR_RE = re.compile(
    r"FmhaFwdTileSize\(\s*" + r"\s*,\s*".join([r"(-?\d+)"] * 19) + r"\s*\)"
)

# Untune / tuned filename pattern (validation only). Matches:
#   mha_tuned_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>[_<trait>...].csv
# The optional trait tail (nh/nhk + CK pipeline tokens such as
# _nlogits_nbias_nlse_ndropout_skip_nqscale_nsink) is written by
# mha_count_shape.py; legacy names ending at `mask<M>` still match.
_TUNED_NAME_RE = re.compile(
    r"^mha_tuned_(?P<gid>\d+)_"
    r"(?P<mode>[a-zA-Z0-9]+)_"
    r"(?P<dtype>[a-zA-Z0-9]+)_"
    r"hq(?P<hq>\d+)_hv(?P<hv>\d+)_"
    r"mask(?P<mask>\d+)(?P<traits>(?:_[a-zA-Z0-9]+)*)\.csv$"
)

# Full group signature columns written by mha_tune.py into every tuned csv
# row. Must stay identical to mha_count_shape.GROUP_COLS and
# mha_tune.SIGNATURE_COLS (asserted by the unit tests).
SIGNATURE_COLS: tuple[str, ...] = (
    "mode",
    "dtype",
    "hdim_q",
    "hdim_v",
    "mask_type",
    "nhead_q",
    "nhead_k",
    "has_logits_soft_cap",
    "bias_type",
    "has_lse",
    "has_dropout",
    "skip_min_seqlen_q",
    "qscale_type",
    "has_sink",
)
_STR_SIGNATURE_COLS = frozenset({"mode", "dtype"})

# Columns that must agree between tuned csvs merged into the same output
# bucket. `dtype` is excluded because it is part of the bucket key itself;
# `bench_variant` is included because a downgraded run (skip -> nskip) is
# not comparable with a native one. `best_hdim_q/v` are the bucket key, so
# they are not conflict dimensions either.
CONFLICT_COLS: tuple[str, ...] = tuple(c for c in SIGNATURE_COLS if c != "dtype") + (
    "bench_variant",
)


# ===========================================================================
# CSV loading
# ===========================================================================


def _parse_tile_expr(expr: str) -> dict[str, int]:
    """Extract 19 ints from `best_tile_expr` and return a dict keyed by
    _ORDERED_FIELDS."""
    m = _TILE_EXPR_RE.search(expr)
    if not m:
        raise ValueError(f"unrecognized tile expr: {expr!r}")
    values = [int(v) for v in m.groups()]
    return dict(zip(_ORDERED_FIELDS, values))


def _tile_signature(tile: dict[str, int]) -> tuple[int, ...]:
    """Immutable key used to compare tiles (all 19 fields in order)."""
    return tuple(tile[k] for k in _ORDERED_FIELDS)


class TunedCsv:
    """Loaded, sorted, ok-only rows from one `mha_tuned_*.csv` file."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.mode: str = ""
        self.dtype: str = ""
        self.orig_hdim_q: int = 0
        self.orig_hdim_v: int = 0
        self.compiled_hdim_q: int = 0
        self.compiled_hdim_v: int = 0
        self.mask_type: int = 0
        # Full SIGNATURE_COLS values of this file (every tuned csv carries
        # them; see mha_tune.TUNED_CSV_COLS).
        self.signature: dict[str, Any] = {}
        # Kernel variant actually benched ("skip" / "nskip").
        self.bench_variant: str = ""
        # Each entry: {"max_seqlen": int, "tile": {F_*: int, ...}}
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def load(path: Path) -> TunedCsv:
        obj = TunedCsv(path)

        # sanity-check the filename shape (non-fatal if it differs, just warn).
        m = _TUNED_NAME_RE.match(obj.path.name)
        if not m:
            print(
                f"[warn] {obj.path.name} does not match "
                f"mha_tuned_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>.csv; "
                f"reading anyway.",
                file=sys.stderr,
            )

        with obj.path.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            required = {
                "max_seqlen",
                *SIGNATURE_COLS,
                "bench_variant",
                "best_hdim_q",
                "best_hdim_v",
                "best_tile_expr",
                "status",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{obj.path}: missing required columns: {sorted(missing)}. "
                    f"The full kernel signature is needed to detect merge "
                    f"conflicts; re-generate the tuned csv with the current "
                    f"`mha_tune.py bench`."
                )

            group_key_seen: tuple[Any, ...] | None = None
            for row in reader:
                if row.get("status", "").strip() != "ok":
                    continue

                try:
                    max_seqlen = int(row["max_seqlen"])
                    row_mode = row["mode"].strip()
                    row_dtype = row["dtype"].strip()
                    row_hq = int(row["hdim_q"])
                    row_hv = int(row["hdim_v"])
                    row_mask = int(row["mask_type"])
                    row_bhq = int(row["best_hdim_q"])
                    row_bhv = int(row["best_hdim_v"])
                    tile = _parse_tile_expr(row["best_tile_expr"])
                    row_sig = {
                        c: (row[c].strip() if c in _STR_SIGNATURE_COLS else int(row[c]))
                        for c in SIGNATURE_COLS
                    }
                    row_variant = row["bench_variant"].strip()
                except (KeyError, ValueError, AttributeError) as e:
                    raise ValueError(
                        f"{obj.path}: cannot parse row {row!r}: {e}"
                    ) from e

                gk = (
                    row_bhq,
                    row_bhv,
                    tuple(row_sig[c] for c in SIGNATURE_COLS),
                    row_variant,
                )
                if group_key_seen is None:
                    group_key_seen = gk
                    obj.mode = row_mode
                    obj.dtype = row_dtype
                    obj.orig_hdim_q = row_hq
                    obj.orig_hdim_v = row_hv
                    obj.compiled_hdim_q = row_bhq
                    obj.compiled_hdim_v = row_bhv
                    obj.mask_type = row_mask
                    obj.signature = row_sig
                    obj.bench_variant = row_variant
                elif gk != group_key_seen:
                    raise ValueError(
                        f"{obj.path}: multiple group signatures found "
                        f"({group_key_seen} vs {gk}); a single tuned csv "
                        f"is expected to represent one shape group."
                    )

                obj.rows.append({"max_seqlen": max_seqlen, "tile": tile})

        if not obj.rows:
            raise ValueError(
                f"{obj.path}: no rows with status=='ok'; nothing to merge."
            )

        # Sort ascending by max_seqlen. A max_seqlen may appear only once:
        # picking a winner between two rows of the same shape would be
        # arbitrary, so this is reported instead of silently deduplicated.
        obj.rows.sort(key=lambda r: r["max_seqlen"])
        dups = sorted(
            {
                r["max_seqlen"]
                for prev, r in zip(obj.rows, obj.rows[1:])
                if prev["max_seqlen"] == r["max_seqlen"]
            }
        )
        if dups:
            raise ValueError(
                f"{obj.path}: duplicate max_seqlen {dups}; every tuned row "
                f"must describe a distinct max_seqlen."
            )
        return obj


# ===========================================================================
# Bucketing / constraint generation
# ===========================================================================


def _row_intervals(
    rows: list[dict[str, Any]],
) -> list[tuple[int | None, int | None]]:
    """Given rows sorted by max_seqlen, return per-row half-open interval
    [low, high) where boundaries are integer midpoints between neighbors.

    The first row gets `low = None` (meaning -inf), the last row gets
    `high = None` (meaning +inf).
    """
    n = len(rows)
    if n == 0:
        return []
    if n == 1:
        # single row covers the entire real line
        return [(None, None)]

    mids: list[int] = []
    for i in range(n - 1):
        a = rows[i]["max_seqlen"]
        b = rows[i + 1]["max_seqlen"]
        # Round the midpoint UP so that a < mid <= b for every a < b.
        # With half-open intervals [low, high) this guarantees that each
        # sample `a` falls inside its own row's interval (never empty),
        # including adjacent samples (b == a + 1 -> mid == b). A tie (an
        # exact-middle length for odd gaps) goes to the larger sample.
        mids.append((a + b + 1) // 2)

    intervals: list[tuple[int | None, int | None]] = []
    for i in range(n):
        low = None if i == 0 else mids[i - 1]
        high = None if i == n - 1 else mids[i]
        intervals.append((low, high))
    return intervals


def _fold_same_tile(
    rows: list[dict[str, Any]],
    intervals: list[tuple[int | None, int | None]],
) -> list[dict[str, Any]]:
    """Group entries by tile signature, folding contiguous same-tile runs
    into one interval, but preserving non-contiguous runs as separate
    intervals under the same tile.

    Returns a list of dicts:
        {
            "tile":       {F_*: int, ...},
            "intervals":  [(low, high), ...],  # may be > 1 entry
            "samples":    [max_seqlen1, max_seqlen2, ...]  # debug
        }
    Preserves the encounter order of tiles.
    """
    # Step 1: fold contiguous same-tile runs.
    folded_runs: list[dict[str, Any]] = []
    for row, (low, high) in zip(rows, intervals):
        sig = _tile_signature(row["tile"])
        if folded_runs and folded_runs[-1]["_sig"] == sig:
            prev = folded_runs[-1]
            prev_low, _prev_high = prev["_interval"]
            prev["_interval"] = (prev_low, high)
            prev["_samples"].append(row["max_seqlen"])
        else:
            folded_runs.append(
                {
                    "_sig": sig,
                    "_tile": row["tile"],
                    "_interval": (low, high),
                    "_samples": [row["max_seqlen"]],
                }
            )

    # Step 2: group by tile signature, keeping first-seen order.
    order: list[tuple[int, ...]] = []
    grouped: dict[tuple[int, ...], dict[str, Any]] = {}
    for run in folded_runs:
        sig = run["_sig"]
        if sig not in grouped:
            order.append(sig)
            grouped[sig] = {
                "tile": run["_tile"],
                "intervals": [run["_interval"]],
                "samples": list(run["_samples"]),
            }
        else:
            grouped[sig]["intervals"].append(run["_interval"])
            grouped[sig]["samples"].extend(run["_samples"])

    return [grouped[s] for s in order]


def _interval_to_cpp(
    low: int | None, high: int | None, var: str = "a.max_seqlen_q"
) -> str:
    """Render one half-open interval [low, high) as a C++ boolean expr."""
    if low is None and high is None:
        return "true"
    if low is None:
        return f"{var} < {high}"
    if high is None:
        return f"{var} >= {low}"
    return f"({var} >= {low} && {var} < {high})"


def _intervals_to_cpp(
    intervals: list[tuple[int | None, int | None]], var: str = "a.max_seqlen_q"
) -> str:
    """OR multiple intervals into one C++ boolean expression."""
    parts = [_interval_to_cpp(lo, hi, var) for (lo, hi) in intervals]
    if len(parts) == 1:
        return parts[0]
    # Wrap each disjunct in parens for readability, then join with " || ".
    wrapped = [p if p.startswith("(") and p.endswith(")") else f"({p})" for p in parts]
    return " || ".join(wrapped)


# ===========================================================================
# Merge / dump
# ===========================================================================


def _bucket_name(key: tuple[str, int, int]) -> str:
    dtype, hq, hv = key
    return f"dtype={dtype} hdim=({hq},{hv})"


def _conflict_key(lc: TunedCsv) -> tuple[Any, ...]:
    """Signature values that must agree between csvs of one bucket."""
    sig = {**lc.signature, "bench_variant": lc.bench_variant}
    return tuple(sig[c] for c in CONFLICT_COLS)


def _check_bucket_signature(key: tuple[str, int, int], sources: list[TunedCsv]) -> None:
    """Reject a bucket fed by tuned csvs with different kernel signatures.

    Tiles of one bucket are selected at runtime by `cpp_constraint`, which
    only knows `max_seqlen_q`, so two signatures sharing a bucket cannot be
    told apart. The report is sorted, hence independent of the `--in` order.
    """
    by_sig: dict[tuple[Any, ...], list[TunedCsv]] = {}
    for lc in sources:
        by_sig.setdefault(_conflict_key(lc), []).append(lc)
    if len(by_sig) <= 1:
        return

    diff_idx = [
        i for i in range(len(CONFLICT_COLS)) if len({sig[i] for sig in by_sig}) > 1
    ]
    lines = []
    for sig in sorted(by_sig, key=lambda s: tuple(str(v) for v in s)):
        shown = " ".join(f"{CONFLICT_COLS[i]}={sig[i]}" for i in diff_idx)
        files = ", ".join(sorted(lc.path.name for lc in by_sig[sig]))
        lines.append(f"  {shown} : {files}")
    raise ValueError(
        f"bucket {_bucket_name(key)} is fed by {len(by_sig)} incompatible "
        f"tuned csvs (differing in: "
        f"{', '.join(CONFLICT_COLS[i] for i in diff_idx)})\n"
        + "\n".join(lines)
        + "\nTiles of one bucket are selected at runtime by max_seqlen_q "
        "only, so these cannot be merged. Emit one JSON per signature."
    )


def _check_bucket_max_seqlens(
    key: tuple[str, int, int], tagged: list[tuple[dict[str, Any], TunedCsv]]
) -> None:
    """Reject the same max_seqlen coming from two csvs of one bucket."""
    by_len: dict[int, list[TunedCsv]] = {}
    for row, lc in tagged:
        by_len.setdefault(row["max_seqlen"], []).append(lc)
    dups = {M: srcs for M, srcs in by_len.items() if len(srcs) > 1}
    if not dups:
        return
    lines = [
        f"  max_seqlen={M} : " + ", ".join(sorted(lc.path.name for lc in dups[M]))
        for M in sorted(dups)
    ]
    raise ValueError(
        f"bucket {_bucket_name(key)} gets the same max_seqlen from several "
        f"tuned csvs\n"
        + "\n".join(lines)
        + "\nThese shapes share one kernel signature, so they belong to a "
        "single group and must be tuned together (see "
        "`mha_count_shape.py group`)."
    )


def build_merged_payload(
    csv_paths: list[Path],
    target: str,
    schema_version: int,
    constraint_var: str,
) -> dict[str, Any]:
    """Read every csv, group by (dtype, best_hq, best_hv), and produce the
    final JSON payload (as a Python dict).

    Raises ValueError when one bucket is fed by csvs with different kernel
    signatures, or when a max_seqlen is tuned more than once.
    """
    loaded: list[TunedCsv] = [TunedCsv.load(p) for p in csv_paths]
    # Sort the sources so that every output (payload key order, warnings and
    # error reports) is independent of the `--in` order.
    loaded.sort(
        key=lambda lc: (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v, str(lc.path))
    )

    # Group by (dtype, compiled_hq, compiled_hv), i.e. by the key CK uses to
    # look tiles up. Several csvs may feed one bucket only when they share
    # the full kernel signature (item 4 of the spec), which is what
    # _check_bucket_signature enforces below.
    grouped: dict[tuple[str, int, int], list[tuple[dict[str, Any], TunedCsv]]] = {}
    grouped_sources: dict[tuple[str, int, int], list[TunedCsv]] = {}
    for lc in loaded:
        key = (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v)
        grouped.setdefault(key, []).extend((r, lc) for r in lc.rows)
        grouped_sources.setdefault(key, []).append(lc)

    tiles_by_dtype: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for key in sorted(grouped):
        dtype, hq, hv = key
        _check_bucket_signature(key, grouped_sources[key])
        _check_bucket_max_seqlens(key, grouped[key])

        merged = sorted((r for r, _ in grouped[key]), key=lambda r: r["max_seqlen"])
        intervals = _row_intervals(merged)
        folded = _fold_same_tile(merged, intervals)

        # Assemble tile objects for this (dtype, hq, hv) bucket.
        tile_objs: list[dict[str, Any]] = []
        for entry in folded:
            tile: dict[str, Any] = dict(entry["tile"])  # copy F_* ints
            tile["cpp_constraint"] = _intervals_to_cpp(
                entry["intervals"], var=constraint_var
            )
            tile["_max_seqlen_samples"] = sorted(entry["samples"])
            tile["_intervals"] = [
                {"low": lo, "high": hi} for (lo, hi) in entry["intervals"]
            ]
            tile_objs.append(tile)

        tiles_by_dtype.setdefault(dtype, {})[f"{hq},{hv}"] = tile_objs

    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "target": target,
        # `grouped` is walked in sorted order, so both dtypes and the
        # per-dtype "HQ,HV" keys come out sorted as well.
        "dtypes": list(tiles_by_dtype),
        "tiles": tiles_by_dtype,
        "meta": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generator": "mha_gen_runtime_json.py",
            "constraint_var": constraint_var,
            "sources": [
                {
                    "csv": str(lc.path),
                    "mode": lc.mode,
                    "dtype": lc.dtype,
                    "orig_hdim_q": lc.orig_hdim_q,
                    "orig_hdim_v": lc.orig_hdim_v,
                    "compiled_hdim_q": lc.compiled_hdim_q,
                    "compiled_hdim_v": lc.compiled_hdim_v,
                    "mask_type": lc.mask_type,
                    "signature": lc.signature,
                    "bench_variant": lc.bench_variant,
                    "row_count": len(lc.rows),
                }
                for lc in loaded
            ],
        },
    }
    return payload


# ===========================================================================
# CLI
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mha_gen_runtime_json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument(
        "--in",
        dest="inputs",
        action="append",
        type=Path,
        required=True,
        metavar="CSV",
        help="Path to a `mha_tuned_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_"
        "mask<M>[_<trait>...].csv` produced by mha_tune.py. May be "
        "repeated to merge multiple tuned shapes into one JSON. Two csvs "
        "that compile to the same (dtype, best_hdim_q, best_hdim_v) bucket "
        "must share the full kernel signature, otherwise the merge is "
        "rejected. The result does not depend on the order of --in.",
    )
    p.add_argument(
        "--out",
        dest="out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output JSON path (mandatory). Recommend naming after the "
        "target model, e.g. `caption5p1_gfx942.json`.",
    )
    p.add_argument(
        "--target",
        default="gfx942",
        help="GPU target string written into the JSON `target` field "
        "(default: %(default)s).",
    )
    p.add_argument(
        "--schema-version",
        type=int,
        default=2,
        help="Value of top-level `schema_version` in the output JSON "
        "(default: %(default)s). CK's _parse_tune_config_tiles ignores "
        "unknown fields, so bumping this does not break codegen.",
    )
    p.add_argument(
        "--constraint-var",
        default="a.max_seqlen_q",
        help="C++ expression used inside cpp_constraint to reference the "
        "runtime max_seqlen_q. Default matches CK's fmha_fwd_args "
        "(`a` in the generated `fmha_fwd_v2/v3` dispatcher), i.e. "
        "`a.max_seqlen_q`. Override only when consuming the JSON from "
        "code that binds a different variable name.",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation (default: %(default)s; use 0 for compact).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="After writing the JSON, print a human-readable summary of "
        "each (dtype, hq, hv) bucket to stdout. Uses the full payload "
        "regardless of --compact, so debug fields are always visible.",
    )
    p.add_argument(
        "--compact",
        action="store_true",
        help="Strip debug and redundant fields, producing a minimal "
        "deployment JSON. Kept: top-level `schema_version`, `target` "
        "(REQUIRED by CK's _build_custom_tune_factory), and "
        '`tiles.<dtype>."HQ,HV".[i].{F_*, cpp_constraint}`. '
        "Removed: top-level `dtypes` (CK derives it from tiles keys), "
        "`meta`; per-tile `_max_seqlen_samples`, `_intervals`. "
        "--print-summary still sees the full payload.",
    )
    return p


def _strip_for_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-ish copy of `payload` containing ONLY the fields that
    CK actually consumes when loading a custom tune-config JSON.

    Kept:
        * top-level `target`         (REQUIRED: CK's `_build_custom_tune_factory`
          calls `_tune_config_pick_base(cfg["target"])` to select the arch
          factory, and raises `ValueError` if it is missing)
        * top-level `schema_version` (small forward-compat tag; CK doesn't
          check it today but keeping it costs almost nothing and guards
          against future CK versions enforcing a version)
        * `tiles.<dtype>."<HQ,HV>".[i].F_*`                (19 int fields)
        * `tiles.<dtype>."<HQ,HV>".[i].cpp_constraint`     (optional string)

    Dropped:
        * top-level `dtypes` (CK derives dtype set from `tiles` keys)
        * top-level `meta`   (audit / debug only)
        * per-tile `_max_seqlen_samples` and `_intervals` (debug only)
        * any other unknown field (CK silently ignores them, but we
          proactively strip to shrink the deployment JSON)
    """
    kept_tile_keys = set(_ORDERED_FIELDS) | {"cpp_constraint"}
    stripped_tiles: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dtype, hmap in payload.get("tiles", {}).items():
        stripped_hmap: dict[str, list[dict[str, Any]]] = {}
        for hkey, tile_list in hmap.items():
            stripped_hmap[hkey] = [
                {k: v for k, v in t.items() if k in kept_tile_keys} for t in tile_list
            ]
        stripped_tiles[dtype] = stripped_hmap

    out: dict[str, Any] = {}
    # Preserve schema_version if present, for forward-compat.
    if "schema_version" in payload:
        out["schema_version"] = payload["schema_version"]
    # Required by CK's _build_custom_tune_factory; carry it through unchanged.
    if "target" in payload:
        out["target"] = payload["target"]
    out["tiles"] = stripped_tiles
    return out


def _print_summary(payload: dict[str, Any]) -> None:
    print()
    print("# ==== merge summary ====")
    for dtype, hkey_map in payload["tiles"].items():
        for hkey, tiles in hkey_map.items():
            print(f"[{dtype}] ({hkey})   #tiles={len(tiles)}")
            for i, t in enumerate(tiles):
                sig = ",".join(str(t[k]) for k in _ORDERED_FIELDS)
                print(f"  #{i}  cpp_constraint = {t['cpp_constraint']}")
                print(f"       samples       = {t.get('_max_seqlen_samples')}")
                print(f"       tile          = {sig}")
    print()


def main() -> int:
    args = _build_parser().parse_args()

    for p in args.inputs:
        if not p.is_file():
            print(f"[error] not a file: {p}", file=sys.stderr)
            return 2

    try:
        payload = build_merged_payload(
            csv_paths=args.inputs,
            target=args.target,
            schema_version=args.schema_version,
            constraint_var=args.constraint_var,
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    indent = args.indent if args.indent and args.indent > 0 else None

    # `--compact` strips debug/meta fields for deployment; keep the full
    # payload around so --print-summary can still show samples/meta.
    to_write = _strip_for_runtime(payload) if args.compact else payload
    args.out.write_text(json.dumps(to_write, indent=indent), encoding="utf-8")
    print(
        f"[done] wrote {args.out} "
        f"(dtypes={payload['dtypes']}, "
        f"pairs={sum(len(v) for v in payload['tiles'].values())}, "
        f"mode={'compact' if args.compact else 'full'})"
    )

    if args.print_summary:
        _print_summary(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
