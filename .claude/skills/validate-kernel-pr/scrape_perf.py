#!/usr/bin/env python3
"""Compare two benchmark logs and decide whether head regressed against base.

Reads the timing tables aiter targets print -- `df.to_markdown(index=False)` from the
`@benchmark`/`run_perftest` pair, or the equivalent table a `--scenario bench` sweep
emits -- from a base log and a head log, matches rows by their non-numeric key columns,
and reduces the pair to one number: `median_ratio`, the head speedup over base.

Ratio orientation is fixed so that **larger is always better and < 1 always means head
got worse**, whichever way the column reads:

    latency column (us, ms, ns, ...)      ratio = base / head
    throughput column (TFLOPS, TB/s, ...) ratio = head / base

`median_ratio` is the *minimum* over per-column medians, not the mean over all of them.
An aiter timing table routinely carries reference columns the PR does not touch
(`torch us`, `triton us`); averaging them in pulls a real regression in the one column
that moved back toward 1.0 and hides it. Gating on the worst column means those
reference columns sit near 1.0 and cannot mask anything.

Each side takes MULTIPLE logs -- repeat runs of the same code -- and each cell is reduced
to its best sample (min for latency, max for throughput) before any ratio is formed. That
is not a refinement, it is what makes the 0.95 threshold usable at all. Measured on this
box, five warm repeat runs of an unchanged op_tests/test_layernorm2d.py gave:

    torch avg (reference column) : 14.24 14.64 14.18 14.23 14.31   -> 1.03x spread
    ck avg    (the aiter kernel) : 13.10 20.98 20.70 13.28 13.17   -> 1.60x spread

The kernel column is bimodal, not noisy-around-a-mean. Comparing one run against one run
would put a base/head ratio anywhere in [0.62, 1.60] on code that did not change, so a
0.95 gate would fire a false regression roughly half the time. Reducing by the minimum over
three repeats collapses that same data to a 1.014x spread and a worst-case ratio of 0.986 --
comfortably inside the threshold. Minimum is the right estimator because scheduling noise,
clock ramp and contention only ever ADD time; the fastest observed run is the closest thing
to the kernel's actual cost.

Deliberately conservative, because a regression verdict here writes a `should-fix`
finding and flips the deterministic verdict to NEEDS_WORK:

  * fewer than `--min-rows` matched rows  -> `insufficient`, never `regression`
  * a column with fewer than `--min-rows` usable samples cannot set the verdict
  * a row present on only one side        -> dropped, not counted as a change
  * a non-numeric, zero, or negative cell -> dropped for that column only
  * no timing column common to both sides -> `insufficient`

Callers must additionally require that every run exited cleanly; this script sees only
text and cannot tell a truncated log from a complete one.

usage: scrape_perf.py --base b1.log [b2.log ...] --head h1.log [h2.log ...]
                      [--threshold 0.95] [--min-rows 3] [--out perf.json]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

# Column-name classification. Matched against the lowercased header cell, on word
# boundaries, so `aiter us` and `us` both read as latency while `status` does not.
LATENCY_RE = re.compile(
    r"(?:^|[^a-z])(us|usec|usecs|microseconds?|ms|msec|msecs|milliseconds?|ns|nsec|"
    r"seconds?|sec|secs|latency|time|elapsed|duration)(?:$|[^a-z])"
)
THROUGHPUT_RE = re.compile(
    r"(?:^|[^a-z])(tflops?|gflops?|mflops?|flops|tb/s|gb/s|mb/s|bw|bandwidth|"
    r"throughput|tokens/s|samples/s|ops/s)(?:$|[^a-z])"
)
# µ is not [a-z], so the boundary class above already isolates it; spell the variants out.
MICRO_RE = re.compile(r"(?:^|[^a-z])(?:µs|μs)(?:$|[^a-z])")

SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def classify(name):
    """Return 'latency', 'throughput', or 'key' for a header cell."""
    low = name.strip().lower()
    if MICRO_RE.search(low) or LATENCY_RE.search(low):
        return "latency"
    if THROUGHPUT_RE.search(low):
        return "throughput"
    return "key"


def split_row(line):
    """Split one markdown table line into its cells."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def to_number(cell):
    """Parse a table cell as a positive float, or return None."""
    text = cell.strip().replace(",", "").replace("%", "")
    # `1.23 us` and `1.23us` both appear when a target formats units into the cell.
    text = re.sub(r"[a-zA-Zµμ/]+$", "", text).strip()
    if not NUMERIC_RE.match(text):
        return None
    value = float(text)
    # A zero or negative timing is a harness artefact (unfilled cell, failed run), not a
    # measurement; dividing by it would manufacture an infinite ratio.
    if value <= 0.0:
        return None
    return value


def parse_tables(text):
    """Yield (headers, rows) for every markdown table in `text`.

    `rows` is a list of cell lists. Tables that carry no separator line are ignored:
    a bare `|`-containing log line is far more often prose than a table.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines) - 1:
        line = lines[index]
        if line.count("|") < 2:
            index += 1
            continue
        if not SEPARATOR_RE.match(lines[index + 1]):
            index += 1
            continue
        headers = split_row(line)
        if lines[index + 1].count("|") + 1 < len(headers):
            index += 1
            continue
        rows = []
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].count("|") >= 2:
            cells = split_row(lines[cursor])
            if len(cells) == len(headers):
                rows.append(cells)
            cursor += 1
        if rows:
            yield headers, rows
        index = max(cursor, index + 1)


# aiter's second output convention. The `@benchmark`/`run_perftest` pair prints a markdown
# table, but the older bare `perftest` decorator's callers hand-roll an f-string per test:
#   [perf] dim: (128, 8192)   , dtype: torch.bfloat16, torch avg: 14.55 us, ck avg: 20.52 us
# Eleven real kernel targets print only this -- test_moe.py, test_pa_v1.py, test_rope.py,
# test_layernorm2d.py among them. Without this parser the stage detects a harness, spends a
# full base+head run, then reports "no parseable timing table" and measures nothing.
PERF_LINE_RE = re.compile(r"\[perf\]\s*(.*)")
PERF_METRIC_RE = re.compile(
    r"([A-Za-z_][\w .+\-]*?)\s+avg:\s*([0-9]*\.?[0-9]+)\s*(us|ms|ns|µs|μs)\b"
)


def parse_perf_lines(text):
    """Yield (row_key, {column: (kind, value)}) for aiter's `[perf] ...` lines."""
    for line in text.splitlines():
        marker = PERF_LINE_RE.search(line)
        if not marker:
            continue
        body = marker.group(1)
        metrics = list(PERF_METRIC_RE.finditer(body))
        if not metrics:
            continue
        # Everything before the first metric identifies the case (dim, dtype, ...) and is
        # stable across runs, which is exactly what a row key has to be.
        row_key = " ".join(body[: metrics[0].start()].split()).strip(" ,")
        values = {}
        for metric in metrics:
            value = to_number(metric.group(2))
            if value is not None:
                values[f"{metric.group(1).strip()} {metric.group(3)}"] = (
                    "latency",
                    value,
                )
        if row_key and values:
            yield row_key, values


def is_identity_cell(cell):
    """Is this cell safe to use as part of a row's identity?

    Non-numeric cells (`fn`, `causal`, `False`, `ok`) always are. A numeric cell is only an
    identity if it is integral: shape and count columns (`s_q`, `num_heads`) are integers
    and reproduce exactly across two runs, while an unlabeled MEASUREMENT column -- the
    `flydsl rel`, `triton err` and `speedup` columns an aiter bench table routinely prints --
    is a float that differs by construction between base and head. Treating those as part of
    the key gives every row a unique name on each side and matches nothing.
    """
    text = cell.strip().replace(",", "").replace("%", "")
    text = re.sub(r"[a-zA-Zµμ/]+$", "", text).strip()
    if not NUMERIC_RE.match(text):
        return True
    try:
        value = float(text)
    except ValueError:
        return True
    return value.is_integer()


def scrape(text, stable_keys_only=False):
    """Reduce a log to {(table_signature, row_key): {column: value}}.

    With `stable_keys_only`, identity columns whose cells are non-integral numbers are
    dropped from the signature and the row key. See `is_identity_cell`.
    """
    measurements = {}
    for row_key, values in parse_perf_lines(text):
        # Distinct signature so a `[perf]` line can never collide with a table row.
        measurements[("[perf]", row_key)] = values
    for headers, rows in parse_tables(text):
        kinds = [classify(header) for header in headers]
        if not any(kind != "key" for kind in kinds):
            continue
        # The signature keeps two tables with different schemas from colliding on a
        # shared row key -- a sweep printed per-dtype produces several such tables. Only
        # the *key* columns go into it, never the timing ones: a PR that adds a variant
        # column to the bench table would otherwise change every signature, drop the
        # match count to zero, and report `insufficient` for exactly the kind of change
        # most worth measuring. Timing columns are intersected per-column further down,
        # so an added one is dropped there instead.
        key_indices = [i for i, kind in enumerate(kinds) if kind == "key"]
        if stable_keys_only:
            key_indices = [
                i
                for i in key_indices
                if all(is_identity_cell(cells[i]) for cells in rows if i < len(cells))
            ]
        signature = "|".join(headers[i].strip().lower() for i in key_indices)
        for cells in rows:
            key_parts = [cells[i] for i in key_indices if i < len(cells)]
            values = {}
            for i, kind in enumerate(kinds):
                if kind == "key":
                    continue
                value = to_number(cells[i])
                if value is not None:
                    values[headers[i].strip()] = (kind, value)
            if not values:
                continue
            # A row without key columns can only be identified by position; number it so
            # a single-column table still matches across the two sides.
            row_key = " / ".join(key_parts) if key_parts else f"#{len(measurements)}"
            # Later occurrences win: a repeated sweep re-prints the same rows warm, and
            # the warm number is the one a reader would quote.
            measurements[(signature, row_key)] = values
    return measurements


def reduce_repeats(runs):
    """Collapse repeat runs of the same code into one best-sample measurement.

    Minimum for latency, maximum for throughput: contention, clock ramp and scheduling
    noise only ever move a measurement the wrong way, so the best observed sample is the
    closest estimate of what the kernel actually costs. See the module docstring for the
    measurement that makes this mandatory rather than optional.
    """
    merged = {}
    for run in runs:
        for key, values in run.items():
            slot = merged.setdefault(key, {})
            for column, (kind, value) in values.items():
                if column not in slot:
                    slot[column] = (kind, value, 1)
                    continue
                prev_kind, prev_value, seen = slot[column]
                if prev_kind != kind:
                    continue
                best = (
                    min(prev_value, value)
                    if kind == "latency"
                    else max(prev_value, value)
                )
                slot[column] = (kind, best, seen + 1)
    return {
        key: {column: (kind, value) for column, (kind, value, _) in values.items()}
        for key, values in merged.items()
    }, {
        key: {column: seen for column, (_, _, seen) in values.items()}
        for key, values in merged.items()
    }


def compare(base_texts, head_texts, threshold, min_rows):
    # Strict identity first, so a table whose key columns are all genuinely identifying keeps
    # exactly the behaviour it had. Only when that matches NOTHING is the relaxed key tried:
    # a table carrying an unlabeled measurement column (an error, a relative error, a
    # speedup) gives every row a unique name on each side, so `matched_rows: 0` there is an
    # artefact of the row-key rule and not a fact about the two runs. Reporting
    # `insufficient` in that case silently disables the perf gate for every target that
    # prints such a column -- which is most aiter bench tables.
    base, base_repeats = reduce_repeats([scrape(text) for text in base_texts])
    head, head_repeats = reduce_repeats([scrape(text) for text in head_texts])
    shared = sorted(set(base) & set(head))
    row_key_basis = "all key columns"
    if not shared:
        relaxed_base, relaxed_base_repeats = reduce_repeats(
            [scrape(text, stable_keys_only=True) for text in base_texts]
        )
        relaxed_head, relaxed_head_repeats = reduce_repeats(
            [scrape(text, stable_keys_only=True) for text in head_texts]
        )
        relaxed_shared = sorted(set(relaxed_base) & set(relaxed_head))
        if relaxed_shared:
            base, base_repeats = relaxed_base, relaxed_base_repeats
            head, head_repeats = relaxed_head, relaxed_head_repeats
            shared = relaxed_shared
            row_key_basis = (
                "key columns excluding non-integral numeric cells, because the strict key "
                "matched no row across the two sides"
            )

    per_column = {}
    rows = []
    for key in shared:
        base_values, head_values = base[key], head[key]
        for column, (kind, base_value) in base_values.items():
            if column not in head_values:
                continue
            head_kind, head_value = head_values[column]
            if head_kind != kind:
                continue
            ratio = (
                base_value / head_value
                if kind == "latency"
                else head_value / base_value
            )
            per_column.setdefault(column, []).append(ratio)
            rows.append(
                {
                    "row": key[1],
                    "column": column,
                    "kind": kind,
                    "base": base_value,
                    "head": head_value,
                    "ratio": round(ratio, 4),
                    "base_repeats": base_repeats[key][column],
                    "head_repeats": head_repeats[key][column],
                }
            )

    columns = {
        column: {
            "median_ratio": round(statistics.median(ratios), 4),
            "samples": len(ratios),
        }
        for column, ratios in per_column.items()
    }
    matched_rows = len({row["row"] for row in rows})

    result = {
        "base_rows": len(base),
        "head_rows": len(head),
        "base_runs": len(base_texts),
        "head_runs": len(head_texts),
        "matched_rows": matched_rows,
        "row_key_basis": row_key_basis,
        "threshold": threshold,
        "min_rows": min_rows,
        "columns": columns,
        "worst_column": None,
        "median_ratio": None,
        "status": "insufficient",
        "reason": None,
        "regressed_rows": [],
    }

    if not columns:
        # Three different failures land here and a reader has to tell them apart: an
        # empty log, two logs that measured different rows, and two logs that measured
        # the same rows under different column names.
        if not base and not head:
            result["reason"] = "neither log contains a parseable timing table"
        elif not base or not head:
            side = "base" if not base else "head"
            result["reason"] = f"the {side} log contains no parseable timing table"
        elif not shared:
            result["reason"] = (
                f"no row is present in both logs ({len(base)} base row(s), "
                f"{len(head)} head row(s)); the two sides measured different shapes"
            )
        else:
            result["reason"] = (
                f"{len(shared)} row(s) matched but no timing column is common to both logs"
            )
        return result
    if matched_rows < min_rows:
        result["reason"] = (
            f"only {matched_rows} row(s) matched across base and head; "
            f"{min_rows} are required before a ratio is trusted"
        )
        return result

    # The row floor has to hold per column, not just overall. A column whose cells are
    # mostly `nan` in one side still gets a median -- from one sample -- and because the
    # verdict takes the *minimum* across columns, that single sample would be exactly the
    # one to set it. Columns stay in the report either way; they just cannot decide.
    eligible = [
        column for column, stats in columns.items() if stats["samples"] >= min_rows
    ]
    for column, stats in columns.items():
        stats["eligible"] = stats["samples"] >= min_rows
    if not eligible:
        result["reason"] = (
            f"{matched_rows} row(s) matched but no timing column has {min_rows} usable "
            "samples on both sides"
        )
        return result

    worst = min(eligible, key=lambda column: columns[column]["median_ratio"])
    result["worst_column"] = worst
    result["median_ratio"] = columns[worst]["median_ratio"]
    if result["median_ratio"] < threshold:
        result["status"] = "regression"
        result["regressed_rows"] = sorted(
            (
                row
                for row in rows
                if row["column"] == worst and row["ratio"] < threshold
            ),
            key=lambda row: row["ratio"],
        )[:5]
        result["reason"] = (
            f"{worst}: median head/base speedup {result['median_ratio']:.3f} "
            f"< {threshold} over {matched_rows} matched row(s)"
        )
    else:
        result["status"] = "ok"
        result["reason"] = (
            f"{worst}: median head/base speedup {result['median_ratio']:.3f} "
            f">= {threshold} over {matched_rows} matched row(s)"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        nargs="+",
        help="benchmark log(s) from the base side; repeats are reduced to the best sample",
    )
    parser.add_argument(
        "--head",
        required=True,
        nargs="+",
        help="benchmark log(s) from the head side; repeats are reduced to the best sample",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="median_ratio below this is a regression (default: 0.95)",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=3,
        help="matched rows required before any verdict (default: 3)",
    )
    parser.add_argument("--out", help="write the JSON result here as well as to stdout")
    args = parser.parse_args(argv)

    sides = {"base": [], "head": []}
    for side, paths in (("base", args.base), ("head", args.head)):
        for path in paths:
            candidate = Path(path)
            if not candidate.is_file():
                result = {
                    "status": "insufficient",
                    "reason": f"{side} log is missing: {path}",
                    "median_ratio": None,
                    "matched_rows": 0,
                    "columns": {},
                }
                print(json.dumps(result, indent=2))
                if args.out:
                    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
                return 0
            sides[side].append(candidate.read_text(errors="replace"))

    result = compare(sides["base"], sides["head"], args.threshold, args.min_rows)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
