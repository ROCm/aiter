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
import ast
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


# --------------------------------------------------------------------------------------
# The decisions that surround a timing run.
#
# The runs themselves stay in the entry point, for the same reason the target launch does:
# they need the locked GPU, the phase's warm cache root, and a timeout the entry point owns.
# What does not need to be there is the reasoning -- which harness the target has, what a
# timing run is allowed to leave behind, and whether a measured difference is attributable.
# Those were bash heredocs, which is to say they were untestable, and they are here instead.


def detect_harness(text):
    """Which benchmark entry point the target's own source offers, if any.

    Returns {"args": ..., "basis": ...}, or None when the target exposes no harness.

    Keep this in step with perf_command() in review-pr/SKILL.md, which computes the manual
    fallback recipe. If the two disagree, that step prints a recipe for a harness this stage
    declined to use, or "no benchmark entry point" for a target this stage happily timed.
    A test asserts they agree, because a comment cannot enforce it.

    A harness cannot be inferred from the diff, only from the target's text, and aiter
    carries three conventions for it. Getting this wrong is survivable in one direction
    only, which is why the tests below matter: a MISSED harness reports `skip` when there
    was something to measure, while a FALSE harness produces a run with no timing table --
    which lands on `skip` as well. Neither can manufacture a regression, because the
    comparison is the ledger and it only ever counts rows it actually parsed.
    """
    if "--scenario" in text and "bench" in text:
        return {"args": "--scenario bench", "basis": "target exposes --scenario bench"}
    if "perftest" in text or "@benchmark" in text:
        # `perftest`, not `run_perftest`. The bare decorator is one of aiter's three timing
        # conventions; matching only the longer name misses 12 of the 123 targets in
        # op_tests/, 11 with live `perftest` usage. Reporting those as "no benchmark entry
        # point" reads as "there was nothing to measure" when the truth is that the detector
        # was too narrow -- the failure mode this whole stage exists to avoid.
        #
        # A substring test, not a parse, so it also matches a commented-out import (the
        # 12th target). That error is the safe one, per the docstring above.
        return {"args": "", "basis": "target uses the perftest/@benchmark harness"}
    if "triton.testing.perf_report" in text or "triton.testing.do_bench" in text:
        # aiter's fourth convention, and the one its DEDICATED benchmark directory is written
        # in. Measured: 58 of the 67 files under op_tests/op_benchmarks/ time themselves this
        # way and exactly one of the 59 repo-wide also matches a rule above -- so every rule
        # this detector had was blind to almost the whole of op_benchmarks/. Pointing
        # --perf-target straight at bench_gemm_a8w8.py reported "no benchmark entry point".
        #
        # Which is precisely the failure this docstring names: a missed harness says "there
        # was nothing to measure" when the truth is that the detector was too narrow.
        return {
            "args": "",
            "basis": "target uses the triton.testing perf_report/do_bench harness",
        }
    return None


# How the validator came to be timing the file it timed. `same-as-correctness-target` is the
# fallback and is an INFERENCE -- nobody read the diff and concluded that file was the right
# one to time; it is simply the only file the caller named.
BASIS_CALLER = "declared-by-caller"
BASIS_FALLBACK = "same-as-correctness-target"
BASIS_SHIPPED = "discovered-pr-shipped"
BASIS_REPO = "discovered-repo-bench"

# The namespace a kernel change lives in. Discovery's second path asks which benches import
# what the patch changed, and a change outside this prefix is not a kernel change: editing a
# test's input generator would otherwise match the bench that imports that test, and the perf
# stage would report on a file whose kernel nobody touched.
KERNEL_PREFIX = "aiter"


def discover_shipped(status_text, read_text):
    """Files the PATCH ITSELF wrote that carry a benchmark harness.

    The first of the two places a perf target comes from: a PR that means to be faster
    usually says so by bringing a bench along. This is the cheap half -- the patch already
    told us which files it touched, and `detect_harness` already knows what a bench looks
    like, so the only new work is asking the second question of the answers to the first.

    `read_text` is injected rather than opened here, for the same reason `restore_worktree`
    injects its three filesystem effects: the decision is then testable without a worktree,
    and a decision that decides what gets EXECUTED deserves that.

    Deletions are skipped. A file the patch removes is not on head and cannot be timed, and
    the resulting `no such file` would arrive as a mysterious perf skip rather than as the
    plain fact that the bench is gone. Git-quoted paths are skipped and returned separately:
    this list feeds a `cp`, and a path we cannot spell is a path we must not act on.
    """
    candidates, unspellable = [], []
    for line in status_text.splitlines():
        if len(line) <= 3:
            continue
        code, path = line[:2], line[3:]
        if path.startswith('"'):
            unspellable.append(path)
            continue
        if "D" in code or not path.endswith(".py"):
            continue
        text = read_text(path)
        if text is not None and detect_harness(text) is not None:
            candidates.append(path)
    return {"candidates": sorted(candidates), "unspellable": unspellable}


def changed_kernel_modules(status_text):
    """Dotted module names for the kernel sources the patch changed.

    `aiter/ops/triton/gemm/basic/gemm_a8w8.py` -> `aiter.ops.triton.gemm.basic.gemm_a8w8`,
    and a package's `__init__.py` -> the package. Deletions are skipped: a module that is not
    on head cannot be imported by anything we are about to run.
    """
    modules = []
    for line in status_text.splitlines():
        if len(line) <= 3:
            continue
        code, path = line[:2], line[3:]
        if path.startswith('"') or "D" in code or not path.endswith(".py"):
            continue
        parts = path[: -len(".py")].split("/")
        if parts[-1] == "__init__":
            parts.pop()
        if not parts or parts[0] != KERNEL_PREFIX:
            continue
        modules.append(".".join(parts))
    return sorted(set(modules))


def imported_modules(text):
    """Every module name a file imports, however it spells the import.

    Parsed, not matched. aiter benches spell the same edge three ways --
    `import aiter.ops.mha`, `from aiter.ops.triton.attention.mla import mla_decode_fwd`, and
    `from aiter.ops.triton.attention import extend_attention` where the imported name is
    itself a module. The last form is common enough in op_benchmarks/ that a regex over the
    dotted path alone misses it, and a regex loose enough to catch it also matches
    `gemm_a8w8_preshuffle` when the patch touched `gemm_a8w8`. The parse has neither problem.

    A file that does not parse contributes nothing. That is the safe direction: it costs a
    candidate, where a wrong candidate costs a finding against a PR author.

    Relative imports are skipped -- resolving them needs the importing package, and a bench
    that reaches a kernel through one is not something to guess at.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def discover_repo_benches(modules, walk, read_text, exclude=()):
    """Files ALREADY in the repo that both carry a harness and import what changed.

    The second of the two places a perf target comes from, and the one that covers the
    ordinary kernel PR: the author changed a kernel and the repository already owns the bench
    that measures it. `op_tests/op_benchmarks/triton/bench_gemm_a8w8.py` imports
    `aiter.ops.triton.gemm.basic.gemm_a8w8` -- that import is an edge that can be checked,
    where a matching filename is only a resemblance.

    What the edge proves is that the bench REFERENCES the changed module. It does not prove
    the bench executes the changed line: run_perf deliberately injects no probe, because a
    traced kernel is not the kernel whose latency we are reporting, so nothing downstream can
    close that gap. It is bounded instead by declining whenever the answer is not unique.
    """
    found = []
    for path in walk():
        if path in exclude:
            continue
        text = read_text(path)
        if text is None or detect_harness(text) is None:
            continue
        if imported_modules(text) & set(modules):
            found.append(path)
    return sorted(found)


# Where aiter keeps the files whose only job is timing. Used to break a tie between several
# benches that all import the changed module, and ONLY to break a tie -- never to find a
# candidate that the import edge did not already prove.
#
# This is not the filename-resemblance guess the import edge exists to avoid. `bench_gemm_a8w8`
# looking like `gemm_a8w8` is a resemblance; a file living in the directory the project set
# aside for benchmarks is a fact about how the project organises itself. Measured on 40 random
# triton kernel modules: 10 resolved to one bench and 6 declined as ambiguous -- and all 6 had
# a candidate here, so this converts every ambiguous case in the sample without touching the
# refusals that matter. A change to the shared aiter.ops.triton.utils.types still declines,
# because 13 of its 14 candidates are under this prefix and a tie is still a tie.
BENCH_HOME = "op_tests/op_benchmarks/"


def _resolve(candidates, correctness_target, kind, advice):
    """One path's candidate list reduced to a target, or to the reason there is none.

    Every outcome that is not "exactly one, and it is new information" resolves to None. That
    asymmetry is the whole safety argument for discovery: a target declined costs a
    measurement, while a target picked WRONG spends a should-fix finding on a PR author whose
    code may be innocent -- and nothing downstream can tell the two apart, because run_perf
    injects no probe and no evidence exists that the bench executed the changed line.

    Refusing to choose between several is the rule --runner established: a caller who can see
    the diff decides, and the report says a choice was owed rather than inventing one.
    """
    if not candidates:
        return None, f"nothing is {kind}"
    if correctness_target in candidates:
        return None, f"the correctness target is itself {advice}"
    if len(candidates) > 1:
        # One tie-break before declining, and only among candidates the import edge already
        # proved: if exactly one of them lives where the project keeps its benchmarks, that is
        # the benchmark. Anything else and the choice is a reading of the diff.
        home = [path for path in candidates if path.startswith(BENCH_HOME)]
        if len(home) == 1:
            return home[0], ""
        return None, (
            "%d files are %s (%s); which of them measures this change is a reading of the "
            "diff and not a fact about it, so name one with --perf-target"
            % (len(candidates), kind, ", ".join(candidates))
        )
    return candidates[0], ""


def choose_perf_target(shipped, repo_benches, correctness_target):
    """Which of the two discovered targets to time, or why the fallback stands.

    A repository bench wins over one the PR ships, and the reason is mechanical rather than a
    preference. A pre-existing bench is on BOTH sides of the patch, so the baseline is this
    worktree with the patch reversed -- one tree, no extra burden of proof. A bench the PR
    adds is absent from base and forces the target-transplant baseline, which spans two trees
    and is only attributable when --perf-control-column reproduces across them. Choosing the
    cheaper, more attributable comparison is not taste.
    """
    repo_target, repo_reason = _resolve(
        repo_benches,
        correctness_target,
        "a benchmark already in the repository importing what the patch changed",
        "the repository's benchmark for what the patch changed",
    )
    ship_target, ship_reason = _resolve(
        shipped["candidates"],
        correctness_target,
        "a file the patch ships carrying a benchmark harness",
        "the benchmark the patch ships",
    )
    candidates = sorted(set(repo_benches) | set(shipped["candidates"]))

    if repo_target is not None:
        return {
            "basis": BASIS_REPO,
            "target": repo_target,
            "reason": (
                "the repository already owns exactly one benchmark importing what this patch "
                "changed, and it is on both sides of the patch so the baseline needs no "
                "transplant"
            ),
            "candidates": candidates,
        }
    if ship_target is not None:
        return {
            "basis": BASIS_SHIPPED,
            "target": ship_target,
            "reason": (
                "the patch ships exactly one file carrying a benchmark harness, and the "
                "repository offers none: %s" % repo_reason
            ),
            "candidates": candidates,
        }
    return {
        "basis": BASIS_FALLBACK,
        "target": correctness_target,
        "reason": f"{repo_reason}; and {ship_reason}",
        "candidates": candidates,
    }


def _status_entries(text):
    """`git status --porcelain` output as {path: two-letter code}."""
    return {line[3:]: line[:2] for line in text.splitlines() if len(line) > 3}


def restore_worktree(root, before_text, current_text, unlink, rmtree, checkout):
    """Undo what the timing run left in the worktree, and only that.

    A bench harness routinely writes its results next to the code -- aiter targets drop a
    tuned_op_bench.csv in the repo root. The baseline phase asserts a CLEAN worktree after
    the base runs, so an artifact left by the timing run sets BASE_READY=0 and skips the
    entire head correctness phase: measured, the same target went PASS with --no-perf and
    INCONCLUSIVE with perf on, with head correctness never executed. A perf stage that
    silently disables correctness validation is far worse than no perf stage.

    Scoped deliberately: only paths whose status CHANGED across the run are touched.
    Anything already dirty beforehand is somebody else's and is left alone. Anything this
    function cannot be sure of -- a git-quoted path, a path that resolves outside the
    worktree -- is reported as skipped rather than guessed at, because this code deletes
    files and a wrong guess is unrecoverable.

    The three filesystem effects are injected so the decision can be tested without a
    worktree to wreck.
    """
    root = Path(root).resolve()
    before = _status_entries(before_text)
    removed, reverted, skipped = [], [], []
    for path, code in _status_entries(current_text).items():
        if before.get(path) == code:
            continue
        if path.startswith('"'):
            skipped.append(path)
            continue
        try:
            resolved = (root / path).resolve()
        except OSError:
            skipped.append(path)
            continue
        if root != resolved and root not in resolved.parents:
            skipped.append(path)
            continue
        if code == "??":
            if resolved.is_dir() and not resolved.is_symlink():
                rmtree(resolved)
            else:
                try:
                    unlink(resolved)
                except OSError:
                    skipped.append(path)
                    continue
            removed.append(path)
        else:
            checkout(path)
            reverted.append(path)
    return {"removed": removed, "reverted": reverted, "skipped": skipped}


def attribute(result, baseline_method, control_column, control_tol):
    """Whether a measured difference can be charged to the patch.

    Only meaningful for a transplanted baseline, where the two sides are two different
    trees. There the difference could be anything -- a different harness path, a different
    allocation, a different clock state -- unless a column the patch does not touch
    reproduces across both. A number nobody can attribute is worse than no number, so
    without that agreement the comparison is downgraded to `insufficient` and says why.

    Mutates `result` (the comparison's own verdict) and returns the note, or "" when there
    was nothing to attribute.
    """
    if baseline_method != "target-transplant":
        return "", None
    columns = result.get("columns") or {}
    match = next(
        (name for name in columns if control_column.lower() in name.lower()), None
    )
    if match is None:
        note = (
            f"the named control column {control_column!r} is not present in both logs, so "
            "this cross-tree comparison cannot be attributed"
        )
        result["status"] = "insufficient"
        result["reason"] = note
        return note, None
    ratio = columns[match].get("median_ratio")
    tolerance = float(control_tol)
    if ratio is None or abs(ratio - 1.0) > tolerance:
        moved = "unknown" if ratio is None else f"{abs(ratio - 1.0):.1%}"
        note = (
            f"the control column {match!r} moved by {moved} across the two trees "
            f"(tolerance {tolerance:.0%}); the patch does not touch it, so the two runs "
            "are not comparable and no ratio is reported"
        )
        result["status"] = "insufficient"
        result["reason"] = note
        return note, ratio
    return (
        f"control column {match!r} reproduced within {abs(ratio - 1.0):.1%} "
        "across the two trees"
    ), ratio


def perf_stage(result, context):
    """The perf stage entry and any finding it earns, from a finished comparison.

    Returns (stage, findings). The verdict mapping is the narrow one on purpose: only
    `regression` may fail and only `ok` may pass. A timeout, a crash, a missing harness and
    a one-row table all land on `skip`, because a false regression here blocks a good PR
    and would get the stage switched off within a week.
    """
    control_note, control_ratio = attribute(
        result,
        context["baseline_method"],
        context["control_column"],
        context["control_tol"],
    )
    baseline_method = context["baseline_method"]
    base_sha = context["base_sha"]
    stage = {
        "status": {"regression": "fail", "ok": "pass"}.get(result["status"], "skip"),
        "baseline_method": baseline_method,
        "baseline": (
            f"{base_sha} with the candidate patch reversed, same worktree and GPU"
            if baseline_method != "target-transplant"
            else (
                f"{base_sha} with the candidate patch reversed and this PR's own target "
                "file copied in, same worktree and GPU; the target drives an entry point "
                "that exists on both sides, so this times the pre-PR implementation "
                "through the same harness"
            )
        ),
        "command": context["command"] or "(target's default entry point)",
        "harness": context["basis"],
        "threshold": result.get("threshold"),
        "matched_rows": result.get("matched_rows", 0),
        # How rows were paired across the two sides. A relaxed key is a fact a reader
        # needs: it means the target printed an unlabeled measurement column that the
        # strict key would have treated as part of each row's identity.
        "row_key_basis": result.get("row_key_basis", "unknown"),
        "columns": result.get("columns", {}),
        # Repeat count is part of the claim, not trivia: the threshold is only defensible
        # because each cell is a best-of-N, so a reader has to be able to see N.
        "repeats": {
            "base": result.get("base_runs", 1),
            "head": result.get("head_runs", 1),
            "reduction": "best sample per cell (min latency / max throughput)",
        },
        "base_log": context["base_log"],
        "head_log": context["head_log"],
        "note": result.get("reason") or "",
    }
    if control_note:
        stage["control_column"] = context["control_column"]
        stage["control_note"] = control_note
        if control_ratio is not None:
            stage["control_ratio"] = control_ratio
        # A stage the control gate rejected must not carry the numbers it rejected.
        # Publishing a median_ratio and a regressed_rows list beside `status: skip` reads
        # as a regression that was merely not acted on, when what happened is that the
        # comparison was found unattributable and no ratio is claimed at all.
        if result["status"] == "insufficient":
            for field in ("median_ratio", "worst_column", "regressed_rows"):
                result.pop(field, None)
    # median_ratio is omitted, never nulled, when there is no measurement:
    # report_schema.json types it as a number, and a null would fail validation at
    # review-pr's identity gate -- turning "we could not measure" into "this report is
    # malformed".
    if result.get("median_ratio") is not None:
        stage["median_ratio"] = result["median_ratio"]
    if result.get("worst_column"):
        stage["worst_column"] = result["worst_column"]
    if result.get("regressed_rows"):
        stage["regressed_rows"] = result["regressed_rows"]

    findings = []
    if result["status"] == "regression":
        rows = ", ".join(
            f"{row['row']}: {row['base']:g} -> {row['head']:g}"
            for row in result.get("regressed_rows", [])[:3]
        )
        findings.append(
            {
                "severity": "should-fix",
                "stage": "perf",
                "detail": (
                    "head is slower than base on the same locked GPU -- "
                    + result["reason"]
                    + (f"; worst rows: {rows}" if rows else "")
                ),
            }
        )
    elif result["status"] == "insufficient":
        findings.append(
            {
                "severity": "note",
                "stage": "perf",
                "detail": f"no perf comparison was made: {result['reason']}",
            }
        )
    return stage, findings


def cmd_detect(args):
    """Exit 3, not 1, when there is no harness: 1 is what a crashed detector returns.

    Arguments first, basis second, one per line -- and the arguments line is empty for the
    decorator harness, which is why it goes first: the caller splits on the first newline,
    and a leading empty field survives command substitution where a trailing one does not.
    """
    harness = detect_harness(Path(args.target).read_text(errors="replace"))
    if harness is None:
        return 3
    print(harness["args"])
    print(harness["basis"])
    return 0


def cmd_discover(args):
    """One JSON object on stdout, and exit 0 whether or not anything was found.

    The opposite convention to cmd_detect above, for a reason: this command ALWAYS has an
    answer -- the fallback is an answer -- so a nonzero exit here is unambiguously a crash
    and never a shrug. The caller reads fields out of the blob with `target_run.py
    stats-field`, the same generic reader the target stats already go through.
    """
    root = Path(args.root).resolve()

    def read_text(path):
        candidate = (root / path).resolve()
        # A patch could name a path outside the worktree. Reading one would be a file
        # disclosure through a report; declining is free and there is nothing to lose.
        if root not in candidate.parents or not candidate.is_file():
            return None
        return candidate.read_text(errors="replace")

    def walk():
        # Every .py in the worktree, not a curated list of test directories. aiter keeps
        # benches in op_tests/op_benchmarks/, but 119 files elsewhere under op_tests/ carry a
        # timing harness too, and a hardcoded directory would quietly decide that those are
        # not perf tests. Measured at 1530 files and 0.3s on this repository, which is not a
        # cost worth buying a guess with. .git is skipped because it holds no source.
        for candidate in sorted(root.rglob("*.py")):
            if ".git" in candidate.parts:
                continue
            yield str(candidate.relative_to(root))

    status = sys.stdin.read()
    shipped = discover_shipped(status, read_text)
    repo_benches = discover_repo_benches(
        changed_kernel_modules(status),
        walk,
        read_text,
        exclude=set(shipped["candidates"]),
    )
    decision = choose_perf_target(shipped, repo_benches, args.correctness_target)
    decision["unspellable"] = shipped["unspellable"]
    print(json.dumps(decision))
    return 0


def cmd_restore(args):
    import shutil
    import subprocess

    root = Path(args.root).resolve()
    current = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    outcome = restore_worktree(
        root,
        Path(args.before).read_text(errors="replace"),
        current,
        unlink=lambda path: path.unlink(),
        rmtree=lambda path: shutil.rmtree(path, ignore_errors=True),
        checkout=lambda path: subprocess.run(
            ["git", "-C", str(root), "checkout", "--", path],
            capture_output=True,
            check=False,
        ),
    )
    if any(outcome.values()):
        print(
            "timing run artifacts cleaned: "
            f"removed={outcome['removed']} reverted={outcome['reverted']} "
            f"skipped={outcome['skipped']}"
        )
    return 0


def cmd_stage(args):
    report = json.loads(Path(args.report).read_text())
    result = json.loads(Path(args.compare).read_text())
    stage, findings = perf_stage(
        result,
        {
            "base_log": args.base_log,
            "head_log": args.head_log,
            "base_sha": args.base_sha,
            "command": args.command,
            "basis": args.basis,
            "baseline_method": args.baseline_method,
            "control_column": args.control_column,
            "control_tol": args.control_tol,
        },
    )
    report["stages"]["perf"] = stage
    report["findings"].extend(findings)
    Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


def subcommand(argv):
    """The three surrounding decisions, dispatched by name.

    Kept off the main parser so that comparing two logs stays the bare `--base/--head`
    invocation it has always been -- that is the interface the entry point and the tests
    already use, and renaming it would be churn with no reader on the other end.
    """
    parser = argparse.ArgumentParser(prog="scrape_perf.py")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="which benchmark harness a target exposes")
    detect.add_argument("target")
    detect.set_defaults(func=cmd_detect)

    discover = sub.add_parser(
        "discover", help="which file to time, read from the patch and the repo"
    )
    discover.add_argument("--root", required=True)
    discover.add_argument("--correctness-target", required=True)
    discover.set_defaults(func=cmd_discover)

    restore = sub.add_parser(
        "restore-worktree", help="undo what a timing run left behind"
    )
    restore.add_argument("root")
    restore.add_argument("before")
    restore.set_defaults(func=cmd_restore)

    stage = sub.add_parser("stage", help="write the perf stage into the report")
    stage.add_argument("--report", required=True)
    stage.add_argument("--compare", required=True)
    stage.add_argument("--base-log", required=True)
    stage.add_argument("--head-log", required=True)
    stage.add_argument("--base-sha", required=True)
    stage.add_argument("--command", default="")
    stage.add_argument("--basis", default="")
    stage.add_argument("--baseline-method", required=True)
    stage.add_argument("--control-column", default="")
    stage.add_argument("--control-tol", default="0.10")
    stage.set_defaults(func=cmd_stage)

    args = parser.parse_args(argv)
    return args.func(args)


# The names main() will route to subcommand(). Written by hand and therefore checked by a test:
# it went stale the first time a subcommand was added, and the failure is quiet in the worst way
# -- `discover` was registered on the subparser, reached this list's `not in`, and fell through
# to the bare comparison parser, which exited 2 complaining about a missing --base. A caller
# reading that has no reason to suspect the subcommand exists.
SUBCOMMANDS = ("detect", "discover", "restore-worktree", "stage")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in SUBCOMMANDS:
        return subcommand(argv)

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
