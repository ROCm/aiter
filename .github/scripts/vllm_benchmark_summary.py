#!/usr/bin/env python3
"""Convert vLLM benchmark results into CSV and GitHub job summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PERCENTILES = [10, 25, 50, 75, 90, 99]
FIELDNAMES = [
    "model",
    "kv_cache_dtype",
    "tp",
    "base_image",
    "batch_size",
    "input_len",
    "output_len",
    "warmup_iters",
    "benchmark_iters",
    "avg_latency_s",
    "p10_latency_s",
    "p25_latency_s",
    "p50_latency_s",
    "p75_latency_s",
    "p90_latency_s",
    "p99_latency_s",
]


def _format_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def _read_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _out(path: Path, line: str = "") -> None:
    with open(path, "a") as handle:
        handle.write(line + "\n")


def _table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    _out(path, "| " + " | ".join(headers) + " |")
    _out(path, "| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        _out(path, "| " + " | ".join(row) + " |")
    _out(path)


def _percentile_value(percentiles: dict, key: int) -> str:
    return _format_float(percentiles.get(key, percentiles.get(str(key))))


def _build_row(results: dict, args: argparse.Namespace) -> dict[str, str]:
    percentiles = results.get("percentiles", {})
    return {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tp": str(args.tp),
        "base_image": args.base_image,
        "batch_size": str(args.batch_size),
        "input_len": str(args.input_len),
        "output_len": str(args.output_len),
        "warmup_iters": str(args.num_iters_warmup),
        "benchmark_iters": str(args.num_iters),
        "avg_latency_s": _format_float(results.get("avg_latency")),
        "p10_latency_s": _percentile_value(percentiles, 10),
        "p25_latency_s": _percentile_value(percentiles, 25),
        "p50_latency_s": _percentile_value(percentiles, 50),
        "p75_latency_s": _percentile_value(percentiles, 75),
        "p90_latency_s": _percentile_value(percentiles, 90),
        "p99_latency_s": _percentile_value(percentiles, 99),
    }


def _summary_path() -> Path | None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    return Path(summary) if summary else None


def single_summary(args: argparse.Namespace) -> None:
    results = _read_json(Path(args.input_json))
    row = _build_row(results, args)
    _write_csv(Path(args.output_csv), [row])

    summary = _summary_path()
    if not summary:
        return

    _out(summary, f"## vLLM Benchmark - {args.model}")
    _out(summary)
    _table(
        summary,
        ["Item", "Value"],
        [
            ["KV cache dtype", f"`{args.kv_cache_dtype}`"],
            ["Base image", f"`{args.base_image}`"],
            ["AITER commit", f"`{args.commit_sha}`"],
            ["Batch size", str(args.batch_size)],
            ["Input length", str(args.input_len)],
            ["Output length", str(args.output_len)],
            ["Warmup iterations", str(args.num_iters_warmup)],
            ["Benchmark iterations", str(args.num_iters)],
        ],
    )
    _out(summary, "### Benchmark CSV")
    _table(summary, FIELDNAMES, [[row[field] for field in FIELDNAMES]])


def _load_csv_rows(input_dir: Path, output_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(input_dir.rglob("*.csv")):
        if csv_path.resolve() == output_csv.resolve():
            continue
        with open(csv_path, newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for row in reader:
                rows.append({field: row.get(field, "") for field in FIELDNAMES})
    rows.sort(key=lambda row: (row["model"], row["kv_cache_dtype"]))
    return rows


def merge_summary(args: argparse.Namespace) -> None:
    output_csv = Path(args.output_csv)
    rows = _load_csv_rows(Path(args.input_dir), output_csv)
    _write_csv(output_csv, rows)

    summary = _summary_path()
    if not summary:
        return

    _out(summary, "## vLLM Benchmark Summary")
    _out(summary)
    summary_rows = [
        ["Base image", f"`{args.base_image}`"],
        ["AITER commit", f"`{args.commit_sha}`"],
        ["Result rows", str(len(rows))],
        ["Merged CSV", f"`{output_csv.name}`"],
    ]
    _table(summary, ["Item", "Value"], summary_rows)

    if not rows:
        _out(summary, "No benchmark CSV files were found.")
        _out(summary)
        return

    _out(summary, "### Benchmark CSV")
    _table(
        summary,
        FIELDNAMES,
        [[row[field] for field in FIELDNAMES] for row in rows],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--input-json", required=True)
    single.add_argument("--output-csv", required=True)
    single.add_argument("--model", required=True)
    single.add_argument("--kv-cache-dtype", required=True)
    single.add_argument("--base-image", required=True)
    single.add_argument("--commit-sha", required=True)
    single.add_argument("--tp", type=int, required=True)
    single.add_argument("--batch-size", type=int, required=True)
    single.add_argument("--input-len", type=int, required=True)
    single.add_argument("--output-len", type=int, required=True)
    single.add_argument("--num-iters-warmup", type=int, required=True)
    single.add_argument("--num-iters", type=int, required=True)
    single.set_defaults(func=single_summary)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input-dir", required=True)
    merge.add_argument("--output-csv", required=True)
    merge.add_argument("--base-image", required=True)
    merge.add_argument("--commit-sha", required=True)
    merge.set_defaults(func=merge_summary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
