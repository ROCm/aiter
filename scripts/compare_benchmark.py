#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare two tuned operator benchmark CSVs (wide-table, metric=us, lower is better).

Used by .github/workflows/aiter-test.yaml tuned_op_bench job to flag tuned operator
performance regressions between a PR and main.

CSV schema:
    - Columns derived from tuned_fmoe.csv shape/params (dtype, token,
      model_dim, inter_dim, E, topk, actType, ...) form the JOIN KEY.
    - `us` is the metric (microseconds, lower = faster).
    - `kernelName1`, `kernelName2` are NOT part of the key (kernel choice
      may differ between baseline and current); they are shown on a
      follow-up line when they differ.
    - Any other column is treated as a key column.

Status legend (all rows are printed, prefixed with status tag):
    [REGRESS]  ratio = current_us / baseline_us > FAIL threshold
    [WARN]     ratio > WARN threshold (and <= FAIL)
    [OK]       ratio <= WARN (including faster-than-baseline)
    [NEW]      shape present in current only (no baseline)
    [REMOVED]  shape present in baseline only (current missing)
    [SKIPPED]  missing or invalid us value on either side

Rows are sorted worst-first: REGRESS, WARN, OK, NEW, REMOVED, SKIPPED.

Exit code:
    0      always, unless --fail-on-regress is set AND >= 1 REGRESS row exists.
    1      with --fail-on-regress when REGRESS detected.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

METRIC = "us"
KERNEL_COLS = ("kernelName1", "kernelName2")
NON_KEY = {METRIC, *KERNEL_COLS}

# Cols kept in the join key (for correct shape matching) but hidden from
# the printed table (low signal — usually constant across runs).
HIDE_DISPLAY_COLS = (
    "preshuffle",
    "strict_accuracy",
    "check_aot_cache",
    "swiglu_limit",
    # Source cols folded into derived `hip` column below
    "hidden_pad",
    "intermediate_pad",
)

# Derived display columns. Each entry: derived_name -> (source_col_a, source_col_b)
# Value rendered as "(a, b)" tuple. Source cols stay in the join key; they're
# just hidden from display (covered by HIDE_DISPLAY_COLS above) and a synthetic
# tuple-valued col is inserted in their place.
DERIVED_TUPLE_COLS = {
    "hip": ("hidden_pad", "intermediate_pad"),  # hidden_pad / intermediate_pad
}

# Display-only abbreviations. Applied at print time; underlying join key
# still uses full strings, so matching across files is unaffected.
# NOTE: `torch.float8_e4m3fnuz` is AMD's default fp8, mapped to `fp8` so it
# stays consistent with `torch.fp8` alias. The OCP / e5m2 variants keep
# the suffix so they remain distinguishable.
_VALUE_ABBREV = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float8_e4m3fnuz": "fp8",  # AMD default
    "torch.float8_e4m3fn": "fp8e4m3fn",  # OCP
    "torch.float8_e5m2": "fp8e5m2",
    "torch.float4_e2m1fn_x2": "fp4",  # x2 = packed (2 elems per byte)
    "torch.fp8": "fp8",
    "torch.fp4x2": "fp4",
    "torch.int8": "i8",
    "torch.int4": "i4",
    "torch.i4x2": "i4",
    # Booleans
    "True": "T",
    "False": "F",
}
# Enum class prefixes to strip ("ActivationType.Silu" -> "Silu")
_STRIP_PREFIXES = ("ActivationType.", "QuantType.", "GateMode.")


def _abbreviate(val: str) -> str:
    """Shorten verbose enum/dtype/bool values for table display."""
    if val in _VALUE_ABBREV:
        return _VALUE_ABBREV[val]
    for prefix in _STRIP_PREFIXES:
        if val.startswith(prefix):
            return val[len(prefix) :]
    return val


def _natural_key(val: str):
    """Cast numeric strings to numbers for natural sort (token=2 < 16 < 128)."""
    try:
        return (0, int(val))
    except ValueError:
        try:
            return (0, float(val))
        except ValueError:
            return (1, val)


Row = Dict[str, str]
Key = Tuple[Tuple[str, str], ...]


def _read_csv(path: Path) -> Tuple[Dict[Key, Row], Tuple[str, ...]]:
    """Return ({key: row}, key_col_order). Whitespace stripped from values."""
    rows: Dict[Key, Row] = {}
    if not path.exists():
        raise SystemExit(f"input csv not found: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or METRIC not in reader.fieldnames:
            raise SystemExit(
                f"{path} missing required column `{METRIC}`; "
                f"got columns: {reader.fieldnames}"
            )
        key_cols = tuple(c for c in reader.fieldnames if c not in NON_KEY)
        for raw in reader:
            # Strip whitespace from every value to avoid silent join misses
            # caused by trailing/leading spaces.
            raw = {k: (v.strip() if isinstance(v, str) else v) for k, v in raw.items()}
            key = tuple(sorted((c, raw.get(c, "")) for c in key_cols))
            rows[key] = raw
    return rows, key_cols


def _parse_us(raw: Row) -> float | None:
    val = raw.get(METRIC, "")
    if val in ("", "-", "skip", "nan", "NaN"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _fmt_key_compact(
    key: Key, key_cols_order: Tuple[str, ...], constants: Dict[str, str]
) -> str:
    """Format key showing only cols whose value is NOT in `constants`."""
    d = dict(key)
    parts = []
    for c in key_cols_order:
        if c in d and c not in constants:
            parts.append(f"{c}={d[c]}")
    return " ".join(parts) if parts else "(common)"


def _find_constants(keys: list[Key], key_cols_order: Tuple[str, ...]) -> Dict[str, str]:
    """Return cols whose value is identical across all `keys`."""
    if not keys:
        return {}
    first = dict(keys[0])
    constants = {}
    for c in key_cols_order:
        if c not in first:
            continue
        v = first[c]
        if all(dict(k).get(c) == v for k in keys):
            constants[c] = v
    return constants


def _kernel_diff(base_row: Row, cur_row: Row) -> list[str]:
    """Return list of `kernelNameX: <base> -> <cur>` for cols that differ."""
    diffs = []
    for c in KERNEL_COLS:
        b, k = base_row.get(c, ""), cur_row.get(c, "")
        if b != k:
            diffs.append(f"{c}: {b}  ->  {k}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("baseline_csv", type=Path)
    parser.add_argument("current_csv", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--current-label", default="current")
    parser.add_argument(
        "--warn",
        type=float,
        default=1.10,
        help="warn threshold: ratio > this is `warn` (default 1.10 = 10%% slower)",
    )
    parser.add_argument(
        "--fail",
        type=float,
        default=1.15,
        help="regress threshold: ratio > this is `REGRESS` (default 1.15 = 15%% slower)",
    )
    parser.add_argument(
        "--fail-on-regress",
        action="store_true",
        help="exit 1 if any REGRESS row found (default: report only, exit 0)",
    )
    args = parser.parse_args()

    if args.warn >= args.fail:
        raise SystemExit(f"--warn ({args.warn}) must be < --fail ({args.fail})")

    baseline, _ = _read_csv(args.baseline_csv)
    current, key_cols = _read_csv(args.current_csv)

    print(f"=== Tuned op bench: {args.current_label} vs {args.baseline_label} ===")
    print(f"  baseline: {args.baseline_csv}  ({len(baseline)} rows)")
    print(f"  current:  {args.current_csv}  ({len(current)} rows)")
    print(f"  thresholds: warn>{args.warn:.2f}, fail>{args.fail:.2f}")
    print()

    common = sorted(baseline.keys() & current.keys())
    only_curr = sorted(current.keys() - baseline.keys())
    only_base = sorted(baseline.keys() - current.keys())

    # Classify every row. Each entry: (sort_rank, status_tag, key, base, cur, ratio)
    # sort_rank: 0=REGRESS, 1=WARN, 2=OK, 3=NEW, 4=REMOVED, 5=SKIPPED
    entries: list[tuple[int, str, Key, float | None, float | None, float | None]] = []
    n_regress = n_warn = n_ok = n_skip = 0
    for key in common:
        b_us = _parse_us(baseline[key])
        c_us = _parse_us(current[key])
        if b_us is None or c_us is None or b_us <= 0:
            n_skip += 1
            entries.append((5, "SKIPPED", key, b_us, c_us, None))
            continue
        ratio = c_us / b_us
        if ratio > args.fail:
            rank, tag = 0, "REGRESS"
            n_regress += 1
        elif ratio > args.warn:
            rank, tag = 1, "WARN"
            n_warn += 1
        else:
            rank, tag = 2, "OK"
            n_ok += 1
        entries.append((rank, tag, key, b_us, c_us, ratio))
    for key in only_curr:
        c_us = _parse_us(current[key])
        entries.append((3, "NEW", key, None, c_us, None))
    for key in only_base:
        b_us = _parse_us(baseline[key])
        entries.append((4, "REMOVED", key, b_us, None, None))

    # Sort worst-first, then by key (natural sort: token=2 < 16 < 128)
    def _entry_sort_key(e):
        rank, _tag, key, *_ = e
        return (rank, [_natural_key(v) for _, v in key])

    entries.sort(key=_entry_sort_key)

    # ── Build proper tabular output ──
    # Columns: status, ratio, cur(us), base(us), *display_cols
    # display_cols = key_cols minus HIDE_DISPLAY_COLS, with each derived
    # tuple col inserted at the position of its first source col.
    # Hidden source cols still contribute to the join key.
    # Kernel diffs (not in table) go on indented ↳ sub-lines below each row.
    METRIC_HDRS = ("ratio", "cur(us)", "base(us)")

    # Build display_cols: walk key_cols, drop hidden, splice derived in place
    _derived_sources = {
        src for sources in DERIVED_TUPLE_COLS.values() for src in sources
    }
    _derived_first_src = {
        sources[0]: name for name, sources in DERIVED_TUPLE_COLS.items()
    }
    display_cols: list[str] = []
    for c in key_cols:
        if c in _derived_first_src:
            display_cols.append(_derived_first_src[c])
        if c not in HIDE_DISPLAY_COLS:
            display_cols.append(c)

    def _cell_value(c: str, d: Dict[str, str]) -> str:
        if c in DERIVED_TUPLE_COLS:
            srcs = DERIVED_TUPLE_COLS[c]
            return "(" + ", ".join(d.get(s, "") for s in srcs) + ")"
        return _abbreviate(d.get(c, ""))

    def _row_cells(rank, tag, key, b_us, c_us, ratio):
        d = dict(key)
        cells = [f"[{tag}]"]
        cells.append(f"{ratio:.3f}" if ratio is not None else "-")
        cells.append(f"{c_us:.2f}" if c_us is not None else "-")
        cells.append(f"{b_us:.2f}" if b_us is not None else "-")
        for c in display_cols:
            cells.append(_cell_value(c, d))
        return cells

    header = ["status", *METRIC_HDRS, *display_cols]
    body = [_row_cells(*e) for e in entries]

    # Column widths = max(header, max value)
    widths = [
        max(len(header[i]), *(len(r[i]) for r in body)) if body else len(header[i])
        for i in range(len(header))
    ]

    # Right-justify the 3 metric cols (numbers), left-justify the rest.
    def _fmt_row(cells):
        out = []
        for i, c in enumerate(cells):
            justify = str.rjust if 1 <= i <= 3 else str.ljust
            out.append(justify(c, widths[i]))
        return "  ".join(out)

    print(_fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for cells, e in zip(body, entries):
        print(_fmt_row(cells))
        # Kernel-diff sub-lines (indented to align under shape columns)
        tag = e[1]
        key = e[2]
        if tag in ("REGRESS", "WARN", "OK"):
            for d in _kernel_diff(baseline[key], current[key]):
                # Indent past the status column for visual hierarchy
                print(" " * (widths[0] + 2) + "↳  " + d)

    print()
    print("Summary:")
    print(f"  compared: {len(common)}")
    print(f"  REGRESS:  {n_regress}")
    print(f"  WARN:     {n_warn}")
    print(f"  OK:       {n_ok}")
    print(f"  NEW:      {len(only_curr)}")
    print(f"  REMOVED:  {len(only_base)}")
    print(f"  SKIPPED:  {n_skip}  (missing/invalid us value)")

    if args.fail_on_regress and n_regress > 0:
        print(f"\nFAIL: {n_regress} regression(s) above threshold.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
