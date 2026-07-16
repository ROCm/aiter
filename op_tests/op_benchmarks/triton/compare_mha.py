#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Main-vs-branch A/B for MHA bench CSVs written by bench_mha.py -o.

Diffs two wide-table CSVs (one metric column, the rest identify the shape) and prints a
per-shape table: the metric in each CSV and the signed percent change (current vs
baseline). It reports the numbers only -- whether a change is a regression or an
improvement is left to the reader. The metric's direction is noted so the sign reads
correctly (TFLOPS / GB/s higher=faster, ms / us lower=faster), but no verdict is computed.

Resilient to schema differences: rows are matched on the columns the two CSVs SHARE, so
a CSV missing some columns (e.g. an older bench predating sliding-window, with no
window_left/window_right) still compares against a newer one. Columns present in only one
CSV are shown in the table so each match is disambiguated; when a shared key matches
several rows (the extra columns distinguish them) all matches are shown.

Stdlib only, no GPU: runs on a plain CPU runner or locally. Always exits 0.

Usage: compare_mha.py BASELINE_CSV CURRENT_CSV
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import List, NamedTuple, Tuple

# The metric column is the unit header bench_mha's _CsvWriter writes as the last column.
_METRIC_UNITS = ("TFLOPS", "GB/s", "ms", "us")
# Units where a LOWER value is faster; TFLOPS / GB/s are higher=faster. Used only to NOTE
# the metric's direction so the reader can read the ratio -- no verdict is computed.
_LOWER_IS_BETTER = frozenset({"ms", "us"})

Row = dict


class BenchTable(NamedTuple):
    """A parsed bench CSV: its columns, its metric column, and its raw rows."""

    metric: str  # metric (unit) column name: TFLOPS / GB/s / ms / us
    fields: Tuple[str, ...]  # all column names, in file order
    rows: List[Row]  # raw rows (all values are strings, as read)


def _detect_metric_col(fieldnames: List[str]) -> str:
    """The metric column is the unit header (TFLOPS/GB/s/ms/us); everything else is a key."""
    found = [c for c in fieldnames if c in _METRIC_UNITS]
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one metric column from {_METRIC_UNITS}, "
            f"found {found} in columns {fieldnames}"
        )
    return found[0]


def _parse_metric(raw: str) -> float | None:
    """Metric cell -> positive float, or None for missing/invalid."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def parse_csv(path: str) -> BenchTable:
    """Read a bench CSV verbatim (csv.DictReader). Pure: no filtering, no reshaping."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = tuple(reader.fieldnames or [])
    metric = _detect_metric_col(list(fields))
    return BenchTable(metric, fields, rows)


def _group(rows: List[Row], key_cols: List[str]) -> dict:
    """Group rows by their value tuple over key_cols (a key may map to several rows)."""
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(tuple(r.get(c, "") for c in key_cols), []).append(r)
    return grouped


def compare_csvs(baseline_csv: str, current_csv: str) -> None:
    """Print the per-shape table: the metric in each CSV + the signed percent change.
    No verdict -- the numbers are the output; the reader decides what a change means."""
    base = parse_csv(baseline_csv)
    cur = parse_csv(current_csv)
    if base.metric != cur.metric:
        raise SystemExit(
            f"metric columns differ: baseline={base.metric}, current={cur.metric}; "
            "the two CSVs measure different things and cannot be compared."
        )
    metric = cur.metric
    lower_better = metric in _LOWER_IS_BETTER

    # Match on the columns the two CSVs SHARE (resilient to either side missing some).
    # Columns present in only one CSV are kept for display (they disambiguate matches),
    # but do not gate matching.
    base_keys = [c for c in base.fields if c != metric]
    cur_keys = [c for c in cur.fields if c != metric]
    key_cols = [c for c in cur_keys if c in base_keys]
    extra_cols = [c for c in (*base_keys, *cur_keys) if c not in key_cols]
    extra_cols = list(dict.fromkeys(extra_cols))  # de-dupe, preserve order
    display_cols = key_cols + extra_cols

    bg, cg = _group(base.rows, key_cols), _group(cur.rows, key_cols)

    def _val(r):
        return _parse_metric(r.get(metric, ""))

    def _cells(brow, crow):
        # Common columns agree; take a column from whichever row carries it.
        src = {**(brow or {}), **(crow or {})}
        return [src.get(c, "") for c in display_cols]

    # Each entry: (delta_pct, cur_val, base_val, cells). delta_pct is None when a row is
    # present in only one CSV (or a value didn't parse) -- shown as blanks, nothing inferred.
    matched = only_cur = only_base = 0
    multi = 0  # shared keys that matched more than one pair (extra columns distinguish)
    entries = []
    for key in bg.keys() & cg.keys():
        pairs = [(b, c) for b in bg[key] for c in cg[key]]
        if len(pairs) > 1:
            multi += 1
        for brow, crow in pairs:
            b, c = _val(brow), _val(crow)
            matched += 1
            delta = ((c - b) / b * 100) if (b is not None and c is not None) else None
            entries.append((delta, c, b, _cells(brow, crow)))
    for key in cg.keys() - bg.keys():
        for crow in cg[key]:
            only_cur += 1
            entries.append((None, _val(crow), None, _cells(None, crow)))
    for key in bg.keys() - cg.keys():
        for brow in bg[key]:
            only_base += 1
            entries.append((None, None, _val(brow), _cells(brow, None)))

    # Order by the magnitude of the change (biggest first, either direction) -- ordering
    # only, no judgement; rows without a delta go last, by shape.
    entries.sort(key=lambda e: (e[0] is None, -abs(e[0] or 0.0), e[3]))

    direction = "lower is faster" if lower_better else "higher is faster"
    print(f"=== MHA bench: current vs baseline (metric={metric}, {direction}) ===")
    print(f"  baseline: {baseline_csv} ({len(base.rows)} rows)")
    print(f"  current:  {current_csv} ({len(cur.rows)} rows)")
    print(
        "  delta% = (current - baseline) / baseline; interpretation left to the reader."
    )
    if extra_cols:
        print(
            f"  note: columns {extra_cols} are present in only one CSV; matched on the "
            f"{len(key_cols)} shared columns (extra columns shown to disambiguate)."
        )
    if multi:
        print(
            f"  note: {multi} shared key(s) matched multiple rows (see extra columns)."
        )
    print()

    # Shape columns first, then the numbers (cur, base, delta%) -- read the config, then
    # its values.
    header = [*display_cols, f"cur({metric})", f"base({metric})", "delta%"]
    body = []
    for delta, c, b, cells in entries:
        body.append(
            [
                *cells,
                f"{c:.3f}" if c is not None else "-",
                f"{b:.3f}" if b is not None else "-",
                f"{delta:+.1f}" if delta is not None else "-",
            ]
        )
    widths = [
        max([len(header[i])] + [len(r[i]) for r in body]) for i in range(len(header))
    ]

    def _line(cells):
        # the last three columns (cur, base, delta%) are numeric -> right-justify
        n = len(cells)
        return "  ".join(
            cells[i].rjust(widths[i]) if i >= n - 3 else cells[i].ljust(widths[i])
            for i in range(len(cells))
        )

    print(_line(header))
    print("  ".join("-" * w for w in widths))
    for row in body:
        print(_line(row))

    print("\nSummary:")
    print(f"  matched:          {matched}")
    print(f"  only in current:  {only_cur}")
    print(f"  only in baseline: {only_base}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline_csv")
    parser.add_argument("current_csv")
    args = parser.parse_args(argv)
    try:
        compare_csvs(args.baseline_csv, args.current_csv)
    except (SystemExit, Exception) as e:
        # Report-only: a malformed or incompatible CSV (bad metric column, mismatched
        # metrics, unreadable file) must not fail the caller -- the CI compare step runs
        # under `set -euo pipefail`. Warn to stderr and still exit 0.
        print(f"compare_mha: skipping compare ({e})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
