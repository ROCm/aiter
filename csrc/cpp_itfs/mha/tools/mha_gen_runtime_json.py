#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
   boundary between two neighboring rows i / i+1 is `(M_i + M_{i+1}) // 2`
   (integer midpoint). The i-th row therefore owns the half-open interval
   `[low_i, high_i)`:
     * low_0    = -inf (represented as absence of the lower bound)
     * low_i    = mid_{i-1}         for i >= 1
     * high_i   = mid_i             for i <  N-1
     * high_{N-1} = +inf            (absence of the upper bound)
   Consecutive rows sharing the same tile are folded into one big interval.
   Runs of the same tile that are non-consecutive (tile-A, tile-B, tile-A)
   produce two intervals on tile-A that are OR-ed together in its
   cpp_constraint.

4. mask_type / bias / lse / dropout are ignored: CK will emit binaries for
   every combination anyway; the constraint only talks about `max_seqlen`.

5. Output path is mandatory: `--out <path>` must be provided, typically named
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
from typing import Any, Dict, List, Optional, Tuple

# CK's FmhaFwdTileSize field order (19 fields), must stay in sync with
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py
# (see FmhaFwdTileSize dataclass) and mha_tune.py's TileSize._ORDERED_FIELDS.
_ORDERED_FIELDS: Tuple[str, ...] = (
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
#   mha_tuned_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>.csv
_TUNED_NAME_RE = re.compile(
    r"^mha_tuned_(?P<gid>\d+)_"
    r"(?P<mode>[a-zA-Z0-9]+)_"
    r"(?P<dtype>[a-zA-Z0-9]+)_"
    r"hq(?P<hq>\d+)_hv(?P<hv>\d+)_"
    r"mask(?P<mask>\d+)\.csv$"
)


# ===========================================================================
# CSV loading
# ===========================================================================


def _parse_tile_expr(expr: str) -> Dict[str, int]:
    """Extract 19 ints from `best_tile_expr` and return a dict keyed by
    _ORDERED_FIELDS."""
    m = _TILE_EXPR_RE.search(expr)
    if not m:
        raise ValueError(f"unrecognized tile expr: {expr!r}")
    values = [int(v) for v in m.groups()]
    return dict(zip(_ORDERED_FIELDS, values))


def _tile_signature(tile: Dict[str, int]) -> Tuple[int, ...]:
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
        # Each entry: {"max_seqlen": int, "tile": {F_*: int, ...}}
        self.rows: List[Dict[str, Any]] = []

    @staticmethod
    def load(path: Path) -> "TunedCsv":
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
                "mode",
                "dtype",
                "hdim_q",
                "hdim_v",
                "mask_type",
                "best_hdim_q",
                "best_hdim_v",
                "best_tile_expr",
                "status",
            }
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{obj.path}: missing required columns: {sorted(missing)}"
                )

            group_key_seen: Optional[Tuple[str, str, int, int, int, int, int]] = None
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
                except (KeyError, ValueError) as e:
                    raise ValueError(
                        f"{obj.path}: cannot parse row {row!r}: {e}"
                    ) from e

                gk = (row_mode, row_dtype, row_hq, row_hv, row_mask, row_bhq, row_bhv)
                if group_key_seen is None:
                    group_key_seen = gk
                    obj.mode = row_mode
                    obj.dtype = row_dtype
                    obj.orig_hdim_q = row_hq
                    obj.orig_hdim_v = row_hv
                    obj.compiled_hdim_q = row_bhq
                    obj.compiled_hdim_v = row_bhv
                    obj.mask_type = row_mask
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

        # Sort ascending by max_seqlen; also deduplicate exact-duplicate
        # max_seqlen entries (keeping the first, warn on rest).
        obj.rows.sort(key=lambda r: r["max_seqlen"])
        deduped: List[Dict[str, Any]] = []
        seen: Dict[int, Dict[str, Any]] = {}
        for r in obj.rows:
            M = r["max_seqlen"]
            if M in seen:
                if _tile_signature(r["tile"]) != _tile_signature(seen[M]["tile"]):
                    print(
                        f"[warn] {obj.path.name}: duplicate max_seqlen={M} "
                        f"with different tiles; keeping the first.",
                        file=sys.stderr,
                    )
                continue
            seen[M] = r
            deduped.append(r)
        obj.rows = deduped
        return obj


# ===========================================================================
# Bucketing / constraint generation
# ===========================================================================


def _row_intervals(
    rows: List[Dict[str, Any]],
) -> List[Tuple[Optional[int], Optional[int]]]:
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

    mids: List[int] = []
    for i in range(n - 1):
        a = rows[i]["max_seqlen"]
        b = rows[i + 1]["max_seqlen"]
        # (a + b) // 2 is well-defined and strictly greater than a whenever
        # b > a + 1; for adjacent (b == a + 1) it equals a, giving an empty
        # interval on the smaller side, which is harmless because ranges
        # are half-open and the boundary flips to the next row anyway.
        mids.append((a + b) // 2)

    intervals: List[Tuple[Optional[int], Optional[int]]] = []
    for i in range(n):
        low = None if i == 0 else mids[i - 1]
        high = None if i == n - 1 else mids[i]
        intervals.append((low, high))
    return intervals


def _fold_same_tile(
    rows: List[Dict[str, Any]],
    intervals: List[Tuple[Optional[int], Optional[int]]],
) -> List[Dict[str, Any]]:
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
    folded_runs: List[Dict[str, Any]] = []
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
    order: List[Tuple[int, ...]] = []
    grouped: Dict[Tuple[int, ...], Dict[str, Any]] = {}
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
    low: Optional[int], high: Optional[int], var: str = "a.max_seqlen_q"
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
    intervals: List[Tuple[Optional[int], Optional[int]]], var: str = "a.max_seqlen_q"
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


def build_merged_payload(
    csv_paths: List[Path],
    target: str,
    schema_version: int,
    constraint_var: str,
) -> Dict[str, Any]:
    """Read every csv, group by (dtype, best_hq, best_hv), and produce the
    final JSON payload (as a Python dict).
    """
    loaded: List[TunedCsv] = [TunedCsv.load(p) for p in csv_paths]

    # Group by (dtype, compiled_hq, compiled_hv). Each group can contain
    # multiple csv sources (e.g. same shape tuned twice with different mask
    # types); item 4 of the spec says we ignore mask/bias/lse/dropout so
    # we merge them into one tile pool.
    grouped_rows: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}
    grouped_sources: Dict[Tuple[str, int, int], List[TunedCsv]] = {}
    for lc in loaded:
        key = (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v)
        grouped_rows.setdefault(key, []).extend(lc.rows)
        grouped_sources.setdefault(key, []).append(lc)

    tiles_by_dtype: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    dtypes_seen: List[str] = []

    for (dtype, hq, hv), rows in grouped_rows.items():
        # Re-sort merged rows by max_seqlen and dedupe on max_seqlen (same
        # rules as inside TunedCsv.load, only that here duplicates may come
        # from different csv sources).
        rows.sort(key=lambda r: r["max_seqlen"])
        dedup: List[Dict[str, Any]] = []
        seen: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            M = r["max_seqlen"]
            if M in seen:
                if _tile_signature(r["tile"]) != _tile_signature(seen[M]["tile"]):
                    print(
                        f"[warn] group=(dtype={dtype}, hq={hq}, hv={hv}): "
                        f"duplicate max_seqlen={M} across csv sources with "
                        f"different tiles; keeping the first-encountered.",
                        file=sys.stderr,
                    )
                continue
            seen[M] = r
            dedup.append(r)

        intervals = _row_intervals(dedup)
        folded = _fold_same_tile(dedup, intervals)

        # Assemble tile objects for this (dtype, hq, hv) bucket.
        tile_objs: List[Dict[str, Any]] = []
        for entry in folded:
            tile: Dict[str, Any] = dict(entry["tile"])  # copy F_* ints
            tile["cpp_constraint"] = _intervals_to_cpp(
                entry["intervals"], var=constraint_var
            )
            tile["_max_seqlen_samples"] = sorted(entry["samples"])
            tile["_intervals"] = [
                {"low": lo, "high": hi} for (lo, hi) in entry["intervals"]
            ]
            tile_objs.append(tile)

        tiles_by_dtype.setdefault(dtype, {})[f"{hq},{hv}"] = tile_objs
        if dtype not in dtypes_seen:
            dtypes_seen.append(dtype)

    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "target": target,
        "dtypes": dtypes_seen,
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
        "mask<M>.csv` produced by mha_tune.py. May be repeated to "
        "merge multiple tuned shapes into one JSON.",
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


def _strip_for_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    stripped_tiles: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for dtype, hmap in payload.get("tiles", {}).items():
        stripped_hmap: Dict[str, List[Dict[str, Any]]] = {}
        for hkey, tile_list in hmap.items():
            stripped_hmap[hkey] = [
                {k: v for k, v in t.items() if k in kept_tile_keys} for t in tile_list
            ]
        stripped_tiles[dtype] = stripped_hmap

    out: Dict[str, Any] = {}
    # Preserve schema_version if present, for forward-compat.
    if "schema_version" in payload:
        out["schema_version"] = payload["schema_version"]
    # Required by CK's _build_custom_tune_factory; carry it through unchanged.
    if "target" in payload:
        out["target"] = payload["target"]
    out["tiles"] = stripped_tiles
    return out


def _print_summary(payload: Dict[str, Any]) -> None:
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
