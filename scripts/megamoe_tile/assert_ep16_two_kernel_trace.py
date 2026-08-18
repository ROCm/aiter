# SPDX-License-Identifier: MIT
"""Fail unless an EP16 candidate hot loop is exactly Stage1, Stage2.

This consumes the unfiltered rocprofv3 kernel-trace CSV.  It deliberately does
not use a profiler kernel filter: a filter would hide an accidental quant,
memset, pack, sort, wait, or copy kernel and turn a launch-count check into a
false pass.

The final ``--iterations`` Stage1/Stage2 pairs are audited.  Initialization,
JIT and explicit warmup launches before that tail are outside the contract.
Within the selected tail every adjacent pair must be exactly Stage1 followed
by Stage2; any third kernel is a hard error.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class KernelLaunch:
    name: str
    start_ns: int
    end_ns: int
    source: str
    row: int

    @property
    def duration_us(self) -> float:
        return (self.end_ns - self.start_ns) / 1000.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) & 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _read_one(path: Path) -> list[KernelLaunch]:
    launches: list[KernelLaunch] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing kernel-trace columns {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            name = row["Kernel_Name"].strip()
            if not name:
                continue
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if end < start:
                raise ValueError(
                    f"{path}:{row_number}: End_Timestamp precedes Start_Timestamp"
                )
            launches.append(KernelLaunch(name, start, end, str(path), row_number))
    return launches


def find_kernel_trace_csvs(path: Path) -> list[Path]:
    """Return trace CSVs below *path*, excluding stats-only CSV files."""

    candidates = [path] if path.is_file() else sorted(path.rglob("*.csv"))
    selected: list[Path] = []
    for candidate in candidates:
        try:
            with candidate.open(newline="") as handle:
                fields = set(next(csv.reader(handle)))
        except (OSError, StopIteration):
            continue
        if {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}.issubset(fields):
            selected.append(candidate)
    if not selected:
        raise ValueError(f"no rocprofv3 kernel-trace CSV found below {path}")
    return selected


def load_launches(path: Path) -> tuple[list[KernelLaunch], list[Path]]:
    csvs = find_kernel_trace_csvs(path)
    launches = [launch for csv_path in csvs for launch in _read_one(csv_path)]
    # A rank-filtered run should normally produce one CSV.  Sorting also makes
    # the checker deterministic if rocprof splits traces by process/thread.
    launches.sort(key=lambda item: (item.start_ns, item.end_ns, item.name))
    if not launches:
        raise ValueError("kernel trace contains no launches")
    return launches, csvs


def _resolve_unique_name(
    launches: Iterable[KernelLaunch], pattern: str, label: str
) -> str:
    regex = re.compile(pattern)
    names = sorted({launch.name for launch in launches if regex.fullmatch(launch.name)})
    if len(names) != 1:
        raise ValueError(
            f"{label} regex {pattern!r} must resolve to one exact kernel symbol; "
            f"resolved {names}"
        )
    return names[0]


def audit_launches(
    launches: list[KernelLaunch],
    *,
    stage1_regex: str,
    stage2_regex: str,
    iterations: int,
) -> dict[str, object]:
    """Audit the final *iterations* complete target pairs.

    The first selected Stage1 is located from the tail, then the complete
    kernel stream is consumed pair-by-pair.  Requiring adjacency is what makes
    an independently launched input quantization kernel fail the contract.
    """

    if iterations < 2:
        raise ValueError("iterations must be >=2 so repeated hidden helpers are observable")
    stage1_name = _resolve_unique_name(launches, stage1_regex, "Stage1")
    stage2_name = _resolve_unique_name(launches, stage2_regex, "Stage2")
    if stage1_name == stage2_name:
        raise ValueError("Stage1 and Stage2 regexes resolve to the same symbol")

    stage1_indices = [
        index for index, launch in enumerate(launches) if launch.name == stage1_name
    ]
    if len(stage1_indices) < iterations:
        raise AssertionError(
            f"need {iterations} Stage1 launches, found {len(stage1_indices)}"
        )
    first_index = stage1_indices[-iterations]
    selected = launches[first_index:]

    pairs: list[tuple[KernelLaunch, KernelLaunch]] = []
    cursor = 0
    for iteration in range(iterations):
        if cursor + 1 >= len(selected):
            raise AssertionError(
                f"iteration {iteration}: trace ends before a complete Stage1/Stage2 pair"
            )
        first, second = selected[cursor], selected[cursor + 1]
        if first.name != stage1_name or second.name != stage2_name:
            observed = [item.name for item in selected[cursor : cursor + 4]]
            raise AssertionError(
                f"iteration {iteration}: expected [{stage1_name!r}, {stage2_name!r}] "
                f"as adjacent launches, observed {observed}"
            )
        pairs.append((first, second))
        cursor += 2

    # A kernel after the final Stage2 may be an out-of-contract result clone or
    # correctness check.  Kernels between selected pairs are impossible here:
    # they would have broken adjacency above.  Report, but do not reject, the
    # post-tail rows.
    trailing = selected[cursor:]
    stage1_us = [pair[0].duration_us for pair in pairs]
    stage2_us = [pair[1].duration_us for pair in pairs]
    pair_envelope_us = [
        (pair[1].end_ns - pair[0].start_ns) / 1000.0 for pair in pairs
    ]
    launch_gap_us = [
        (pair[1].start_ns - pair[0].end_ns) / 1000.0 for pair in pairs
    ]
    return {
        "contract": "exactly_two_gpu_kernel_launches_per_hot_iteration",
        "iterations": iterations,
        "stage1_kernel": stage1_name,
        "stage2_kernel": stage2_name,
        "selected_launches": iterations * 2,
        "sequence": [name for _ in range(iterations) for name in (stage1_name, stage2_name)],
        "stage1_duration_us": stage1_us,
        "stage2_duration_us": stage2_us,
        "pair_envelope_us": pair_envelope_us,
        "stage1_to_stage2_launch_gap_us": launch_gap_us,
        "median_us": {
            "stage1": _median(stage1_us),
            "stage2": _median(stage2_us),
            "pair_envelope": _median(pair_envelope_us),
        },
        "trailing_kernel_count_outside_selected_tail": len(trailing),
        "trailing_kernel_names_outside_selected_tail": [item.name for item in trailing],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="rocprofv3 output directory or one kernel-trace CSV",
    )
    parser.add_argument("--stage1-regex", required=True)
    parser.add_argument("--stage2-regex", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument(
        "--require-no-trailing-kernels",
        action="store_true",
        help=(
            "Also reject kernels after the final selected Stage2. Normally the "
            "driver performs an out-of-timing output clone/check, so this is off."
        ),
    )
    args = parser.parse_args()

    launches, csvs = load_launches(args.trace)
    result = audit_launches(
        launches,
        stage1_regex=args.stage1_regex,
        stage2_regex=args.stage2_regex,
        iterations=args.iterations,
    )
    result["trace_csvs"] = [str(path.resolve()) for path in csvs]
    if args.require_no_trailing_kernels and result[
        "trailing_kernel_count_outside_selected_tail"
    ]:
        raise AssertionError(
            "kernel launches follow the final selected Stage2: "
            f"{result['trailing_kernel_names_outside_selected_tail']}"
        )
    print("MEGAMOE_EP16_TWO_KERNEL_TRACE " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
