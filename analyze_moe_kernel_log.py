#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROW_RE = re.compile(
    r"^\s*(?P<idx>\d+)\s+"
    r"(?P<name>.*?)\s+"
    r"(?P<cnt>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<host_time_sum>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<device_time_sum>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<device_time_avg>[\d,]+(?:\.\d+)?)\s+"
    r"(?P<device_type>\S+)\s+"
    r"(?P<device_index>\S+)\s*$"
)

BENCH_ROW_RE = re.compile(
    r"^\s*(?P<M>\d+)\s+\|\s+"
    r"(?P<mx_us>[\d.]+)\s+\|\s+"
    r"(?P<fly_us>[\d.]+)\s+\|"
)


@dataclass
class KernelRow:
    name: str
    cnt: float
    device_time_sum: float


@dataclass
class KernelBlock:
    line_no: int
    rows: list[KernelRow]


@dataclass
class SummaryRow:
    pair: int
    M: int | None
    op: str
    name: str
    kernel_names: list[str]
    cnt: float
    device_time_sum: float
    avg_time: float


@dataclass
class ComparisonRow:
    pair: int
    M: int | None
    name: str
    mx_kernel_names: list[str]
    mx_cnt: float
    mx_device_time_sum: float
    mx_avg_time: float
    fly_kernel_names: list[str]
    fly_cnt: float
    fly_device_time_sum: float
    fly_avg_time: float
    delta_avg_time: float
    fly_over_mx: float


@dataclass
class TotalComparisonRow:
    pair: int
    M: int | None
    mx_kernel_names: list[str]
    mx_device_time_sum: float
    mx_avg_time: float
    fly_kernel_names: list[str]
    fly_device_time_sum: float
    fly_avg_time: float
    delta_avg_time: float
    fly_over_mx: float


@dataclass
class DeltaContributionRow:
    pair: int
    M: int | None
    total_mx_avg_time: float
    total_pct: float | None
    gemm1_pct: float | None
    gemm2_pct: float | None
    sort_pct: float | None


CATEGORY_ORDER = ["gemm1", "gemm2", "sort"]


def parse_float(text: str) -> float:
    return float(text.replace(",", ""))


def is_header(line: str) -> bool:
    return "device_time_sum" in line and "device_time_avg" in line


def parse_kernel_row(line: str) -> KernelRow | None:
    match = ROW_RE.match(line)
    if match is None:
        return None

    device_time_sum = parse_float(match.group("device_time_sum"))
    if device_time_sum == 0.0:
        return None

    return KernelRow(
        name=match.group("name").strip(),
        cnt=parse_float(match.group("cnt")),
        device_time_sum=device_time_sum,
    )


def parse_log(path: Path) -> tuple[list[KernelBlock], dict[int, int]]:
    blocks: list[KernelBlock] = []
    pending_rows: list[KernelRow] | None = None
    pending_line_no = 0
    bench_m_by_pair: dict[int, int] = {}
    next_bench_pair = 1

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if is_header(line):
                if pending_rows:
                    blocks.append(KernelBlock(pending_line_no, pending_rows))
                pending_rows = []
                pending_line_no = line_no
                continue

            bench_match = BENCH_ROW_RE.match(line)
            if bench_match is not None:
                bench_m_by_pair[next_bench_pair] = int(bench_match.group("M"))
                next_bench_pair += 1

            if pending_rows is None:
                continue

            row = parse_kernel_row(line)
            if row is not None:
                pending_rows.append(row)
                continue

            if "[avg us/iter]" in line:
                blocks.append(KernelBlock(pending_line_no, pending_rows))
                pending_rows = None

    if pending_rows:
        blocks.append(KernelBlock(pending_line_no, pending_rows))

    return blocks, bench_m_by_pair


def summarize_category(
    pair: int,
    M: int | None,
    op: str,
    name: str,
    rows: list[KernelRow],
) -> SummaryRow:
    if not rows:
        return SummaryRow(pair, M, op, name, [], 0.0, 0.0, 0.0)

    # Multiple sort kernels belong to the same operator iteration, so add their
    # device_time_sum values first and divide by the benchmark iteration count.
    cnt = rows[0].cnt
    device_time_sum = sum(row.device_time_sum for row in rows)
    avg_time = device_time_sum / cnt if cnt else 0.0
    kernel_names = sorted({row.name for row in rows})
    return SummaryRow(pair, M, op, name, kernel_names, cnt, device_time_sum, avg_time)


def summarize_blocks(
    blocks: list[KernelBlock],
    bench_m_by_pair: dict[int, int],
    labels: list[str],
) -> list[SummaryRow]:
    summaries: list[SummaryRow] = []

    for block_idx, block in enumerate(blocks):
        pair = block_idx // len(labels) + 1
        op = labels[block_idx % len(labels)]
        M = bench_m_by_pair.get(pair)
        valid_rows = block.rows

        if len(valid_rows) < 2:
            print(
                f"warning: block at line {block.line_no} has fewer than 2 valid "
                "device rows; skipping",
                file=sys.stderr,
            )
            continue

        sort_rows = valid_rows[:-2]
        gemm2_rows = [valid_rows[-2]]
        gemm1_rows = [valid_rows[-1]]

        summaries.append(summarize_category(pair, M, op, "sort", sort_rows))
        summaries.append(summarize_category(pair, M, op, "gemm2", gemm2_rows))
        summaries.append(summarize_category(pair, M, op, "gemm1", gemm1_rows))

    return summaries


def compare_summaries(rows: list[SummaryRow], labels: list[str]) -> list[ComparisonRow]:
    if len(labels) != 2:
        raise ValueError("comparison output requires exactly two labels")

    lhs_label, rhs_label = labels
    by_key = {(row.pair, row.name, row.op): row for row in rows}
    pairs = sorted({row.pair for row in rows})
    comparisons: list[ComparisonRow] = []

    for pair in pairs:
        for name in CATEGORY_ORDER:
            lhs = by_key.get((pair, name, lhs_label))
            rhs = by_key.get((pair, name, rhs_label))
            if lhs is None or rhs is None:
                continue

            fly_over_mx = rhs.avg_time / lhs.avg_time if lhs.avg_time else 0.0
            comparisons.append(
                ComparisonRow(
                    pair=pair,
                    M=lhs.M if lhs.M is not None else rhs.M,
                    name=name,
                    mx_kernel_names=lhs.kernel_names,
                    mx_cnt=lhs.cnt,
                    mx_device_time_sum=lhs.device_time_sum,
                    mx_avg_time=lhs.avg_time,
                    fly_kernel_names=rhs.kernel_names,
                    fly_cnt=rhs.cnt,
                    fly_device_time_sum=rhs.device_time_sum,
                    fly_avg_time=rhs.avg_time,
                    delta_avg_time=rhs.avg_time - lhs.avg_time,
                    fly_over_mx=fly_over_mx,
                )
            )

    return comparisons


def compare_totals(rows: list[ComparisonRow]) -> list[TotalComparisonRow]:
    rows_by_pair: dict[int, list[ComparisonRow]] = {}
    for row in rows:
        rows_by_pair.setdefault(row.pair, []).append(row)

    totals: list[TotalComparisonRow] = []
    for pair in sorted(rows_by_pair):
        pair_rows = [
            row
            for row in rows_by_pair[pair]
            if row.name in CATEGORY_ORDER
        ]
        if not pair_rows:
            continue

        mx_device_time_sum = sum(row.mx_device_time_sum for row in pair_rows)
        fly_device_time_sum = sum(row.fly_device_time_sum for row in pair_rows)
        mx_avg_time = sum(row.mx_avg_time for row in pair_rows)
        fly_avg_time = sum(row.fly_avg_time for row in pair_rows)
        fly_over_mx = fly_avg_time / mx_avg_time if mx_avg_time else 0.0
        mx_kernel_names = sorted(
            {kernel_name for row in pair_rows for kernel_name in row.mx_kernel_names}
        )
        fly_kernel_names = sorted(
            {kernel_name for row in pair_rows for kernel_name in row.fly_kernel_names}
        )

        totals.append(
            TotalComparisonRow(
                pair=pair,
                M=pair_rows[0].M,
                mx_kernel_names=mx_kernel_names,
                mx_device_time_sum=mx_device_time_sum,
                mx_avg_time=mx_avg_time,
                fly_kernel_names=fly_kernel_names,
                fly_device_time_sum=fly_device_time_sum,
                fly_avg_time=fly_avg_time,
                delta_avg_time=fly_avg_time - mx_avg_time,
                fly_over_mx=fly_over_mx,
            )
        )

    return totals


def compute_delta_contributions(rows: list[ComparisonRow]) -> list[DeltaContributionRow]:
    rows_by_pair: dict[int, dict[str, ComparisonRow]] = {}
    for row in rows:
        rows_by_pair.setdefault(row.pair, {})[row.name] = row

    contributions: list[DeltaContributionRow] = []
    for pair in sorted(rows_by_pair):
        pair_rows = rows_by_pair[pair]
        if not all(name in pair_rows for name in CATEGORY_ORDER):
            continue

        total_mx_avg = sum(pair_rows[name].mx_avg_time for name in CATEGORY_ORDER)

        def pct(name: str) -> float | None:
            if total_mx_avg == 0:
                return None
            return pair_rows[name].delta_avg_time / total_mx_avg * 100

        first_row = pair_rows[CATEGORY_ORDER[0]]
        contributions.append(
            DeltaContributionRow(
                pair=pair,
                M=first_row.M,
                total_mx_avg_time=total_mx_avg,
                total_pct=(
                    sum(pair_rows[name].delta_avg_time for name in CATEGORY_ORDER)
                    / total_mx_avg
                    * 100
                    if total_mx_avg
                    else None
                ),
                gemm1_pct=pct("gemm1"),
                gemm2_pct=pct("gemm2"),
                sort_pct=pct("sort"),
            )
        )

    return contributions


def format_num(value: float) -> str:
    return f"{value:,.1f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


def format_kernel_names(names: list[str]) -> str:
    if not names:
        return "{}"
    return "{" + "; ".join(names) + "}"


def format_comparison_kernel_names(row: ComparisonRow) -> str:
    return (
        f"mx={format_kernel_names(row.mx_kernel_names)}; "
        f"fly={format_kernel_names(row.fly_kernel_names)}"
    )


def format_total_kernel_names(row: TotalComparisonRow) -> str:
    return (
        f"mx={format_kernel_names(row.mx_kernel_names)}; "
        f"fly={format_kernel_names(row.fly_kernel_names)}"
    )


def print_table(rows: Iterable[SummaryRow]) -> None:
    headers = [
        "pair",
        "M",
        "op",
        "name",
        "cnt",
        "device_time_sum",
        "avg_time",
        "kernel_names",
    ]
    widths = [len(h) for h in headers]
    body: list[list[str]] = []

    for row in rows:
        values = [
            str(row.pair),
            "" if row.M is None else str(row.M),
            row.op,
            row.name,
            format_num(row.cnt),
            format_num(row.device_time_sum),
            format_num(row.avg_time),
            format_kernel_names(row.kernel_names),
        ]
        body.append(values)
        widths = [max(width, len(value)) for width, value in zip(widths, values)]

    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for values in body:
        print("  ".join(value.rjust(w) for value, w in zip(values, widths)))


def print_comparison_table(rows: Iterable[ComparisonRow]) -> None:
    rows = list(rows)
    rows_by_name = {name: [] for name in CATEGORY_ORDER}
    for row in rows:
        rows_by_name.setdefault(row.name, []).append(row)

    headers = [
        "pair",
        "M",
        "mx_cnt",
        "mx_sum",
        "mx_avg",
        "fly_cnt",
        "fly_sum",
        "fly_avg",
        "delta",
        "fly/mx",
        "kernel_names",
    ]

    first_table = True
    for name in CATEGORY_ORDER:
        table_rows = rows_by_name.get(name, [])
        if not table_rows:
            continue

        if not first_table:
            print()
        first_table = False
        print(f"[{name}]")

        widths = [len(h) for h in headers]
        body: list[list[str]] = []
        for row in table_rows:
            values = [
                str(row.pair),
                "" if row.M is None else str(row.M),
                format_num(row.mx_cnt),
                format_num(row.mx_device_time_sum),
                format_num(row.mx_avg_time),
                format_num(row.fly_cnt),
                format_num(row.fly_device_time_sum),
                format_num(row.fly_avg_time),
                format_num(row.delta_avg_time),
                f"{row.fly_over_mx:.2f}x",
                format_comparison_kernel_names(row),
            ]
            body.append(values)
            widths = [max(width, len(value)) for width, value in zip(widths, values)]

        print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
        print("  ".join("-" * w for w in widths))
        for values in body:
            print("  ".join(value.rjust(w) for value, w in zip(values, widths)))

    totals = compare_totals(rows)
    if totals:
        print()
        print_total_comparison_table(totals)

    contributions = compute_delta_contributions(rows)
    if contributions:
        print()
        print_delta_contribution_table(contributions)


def print_total_comparison_table(rows: Iterable[TotalComparisonRow]) -> None:
    headers = [
        "pair",
        "M",
        "mx_sum",
        "mx_avg",
        "fly_sum",
        "fly_avg",
        "delta",
        "fly/mx",
        "kernel_names",
    ]
    widths = [len(h) for h in headers]
    body: list[list[str]] = []

    for row in rows:
        values = [
            str(row.pair),
            "" if row.M is None else str(row.M),
            format_num(row.mx_device_time_sum),
            format_num(row.mx_avg_time),
            format_num(row.fly_device_time_sum),
            format_num(row.fly_avg_time),
            format_num(row.delta_avg_time),
            f"{row.fly_over_mx:.2f}x",
            format_total_kernel_names(row),
        ]
        body.append(values)
        widths = [max(width, len(value)) for width, value in zip(widths, values)]

    print("[total]")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for values in body:
        print("  ".join(value.rjust(w) for value, w in zip(values, widths)))


def print_delta_contribution_table(rows: Iterable[DeltaContributionRow]) -> None:
    headers = [
        "pair",
        "M",
        "total_mx_avg",
        "total_pct",
        "gemm1_pct",
        "gemm2_pct",
        "sort_pct",
    ]
    widths = [len(h) for h in headers]
    body: list[list[str]] = []

    for row in rows:
        values = [
            str(row.pair),
            "" if row.M is None else str(row.M),
            format_num(row.total_mx_avg_time),
            format_pct(row.total_pct),
            format_pct(row.gemm1_pct),
            format_pct(row.gemm2_pct),
            format_pct(row.sort_pct),
        ]
        body.append(values)
        widths = [max(width, len(value)) for width, value in zip(widths, values)]

    print("[delta_contribution]")
    print("  ".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for values in body:
        print("  ".join(value.rjust(w) for value, w in zip(values, widths)))


def print_csv(rows: Iterable[SummaryRow]) -> None:
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "pair",
            "M",
            "op",
            "name",
            "cnt",
            "device_time_sum",
            "avg_time",
            "kernel_names",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.pair,
                "" if row.M is None else row.M,
                row.op,
                row.name,
                row.cnt,
                row.device_time_sum,
                row.avg_time,
                format_kernel_names(row.kernel_names),
            ]
        )


def print_comparison_csv(rows: Iterable[ComparisonRow]) -> None:
    rows = list(rows)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "pair",
            "M",
            "name",
            "kernel_names",
            "mx_cnt",
            "mx_device_time_sum",
            "mx_avg_time",
            "fly_cnt",
            "fly_device_time_sum",
            "fly_avg_time",
            "delta_avg_time",
            "fly_over_mx",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.pair,
                "" if row.M is None else row.M,
                row.name,
                format_comparison_kernel_names(row),
                row.mx_cnt,
                row.mx_device_time_sum,
                row.mx_avg_time,
                row.fly_cnt,
                row.fly_device_time_sum,
                row.fly_avg_time,
                row.delta_avg_time,
                row.fly_over_mx,
            ]
        )
    for row in compare_totals(rows):
        writer.writerow(
            [
                row.pair,
                "" if row.M is None else row.M,
                "total",
                format_total_kernel_names(row),
                "",
                row.mx_device_time_sum,
                row.mx_avg_time,
                "",
                row.fly_device_time_sum,
                row.fly_avg_time,
                row.delta_avg_time,
                row.fly_over_mx,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize paired MoE kernel timing blocks into sort/gemm2/gemm1 "
            "device-time categories."
        )
    )
    parser.add_argument(
        "log",
        nargs="?",
        default="tt.log",
        type=Path,
        help="path to the log file, default: tt.log",
    )
    parser.add_argument(
        "--labels",
        default="mx,fly",
        help="comma-separated operator labels for each timing pair, default: mx,fly",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="print CSV instead of a fixed-width table",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="print per-operator detail rows instead of mx/fly comparison rows",
    )
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        parser.error("--labels must contain at least one non-empty label")
    if not args.detail and len(labels) != 2:
        parser.error("comparison output requires exactly two --labels entries")

    blocks, bench_m_by_pair = parse_log(args.log)
    summaries = summarize_blocks(blocks, bench_m_by_pair, labels)

    if args.detail:
        if args.csv:
            print_csv(summaries)
        else:
            print_table(summaries)
        return

    comparisons = compare_summaries(summaries, labels)
    if args.csv:
        print_comparison_csv(comparisons)
    else:
        print_comparison_table(comparisons)


if __name__ == "__main__":
    main()
