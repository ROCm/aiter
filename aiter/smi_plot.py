# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic, offline plots of AITER_SMI_RESULT summaries. No GPU required."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PREFIX = "AITER_SMI_RESULT "
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PARAM_ORDER = ("M", "N", "K", "T", "H", "D")
FUNCTION_NAMES = {
    "run_gemm_bpreshuffle": "B-preshuffle",
    "run_gemm_abpreshuffle": "A+B-preshuffle",
}
# Fixed metric colors and markers, independent of the files or their order.
PANELS = (
    (
        "Clocks (MHz)",
        (
            ("gfx_clk_mhz", "Engine", "#2378d4", "o"),
            ("fclk_mhz", "Fabric", "#ee7733", "D"),
            ("soc_clk_mhz", "SoC", "#228833", "s"),
        ),
    ),
    ("Power (W)", (("power_w", "Socket power", "#aa3377", "o"),)),
    (
        "Activity (%)",
        (
            ("gfx_activity_pct", "GPU", "#2378d4", "o"),
            ("umc_activity_pct", "Memory controller", "#ee7733", "D"),
        ),
    ),
    ("Temperature (°C)", (("temp_hotspot_c", "Hotspot", "#cc3311", "o"),)),
)


def natural(value):
    return tuple(
        (0, int(p)) if p.isdigit() else (1, p.casefold())
        for p in re.split(r"(\d+)", str(value))
    )


def param_key(item):
    key, value = item
    return (
        PARAM_ORDER.index(key) if key in PARAM_ORDER else len(PARAM_ORDER),
        natural(key),
        natural(value),
    )


def split_label(label):
    parts = label.split("/")
    test = parts[0]
    params = []
    functions = []
    for part in parts[1:]:
        if "=" in part:
            pair = tuple(part.split("=", 1))
            if pair not in params:
                params.append(pair)
        elif part:
            functions.append(part)
    function = "/".join(functions) or test
    match = re.fullmatch(r"(.+)#(\d+)", function)
    occurrence = int(match[2]) if match else 1
    function = match[1] if match else function
    return test, tuple(sorted(params, key=param_key)), function, occurrence


@dataclass
class Record:
    source: str
    line: int
    data: dict
    test: str
    params: tuple
    function: str
    occurrence: int

    @property
    def case(self):
        return self.test, self.params, str(self.data.get("device", "unknown"))

    @property
    def order(self):
        priority = {"run_gemm_bpreshuffle": 0, "run_gemm_abpreshuffle": 1}
        return (
            natural(self.test),
            tuple(param_key(p) for p in self.params),
            natural(self.case[2]),
            priority.get(self.function, 2),
            natural(self.function),
            natural(self.source),
            self.occurrence,
            self.line,
        )

    @property
    def function_label(self):
        return FUNCTION_NAMES.get(self.function, self.function)


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_records(input_path, output_path, patterns):
    """Accept mixed console logs and bare JSONL; never merge repeated results."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path.is_file():
        paths, base = [input_path], input_path.parent
    elif input_path.is_dir():
        paths = sorted(
            {
                p
                for pattern in patterns
                for p in input_path.rglob(pattern)
                if p.is_file() and not p.resolve().is_relative_to(output_path)
            },
            key=lambda p: p.relative_to(input_path).as_posix(),
        )
        base = input_path
    else:
        raise ValueError(f"Input does not exist: {input_path}")
    records, sources, warnings = [], [], []
    for path in paths:
        source = path.relative_to(base).as_posix()
        raw = path.read_bytes()
        count = 0
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            line = ANSI.sub("", line).strip()
            marked = PREFIX in line
            if marked:
                line = line.split(PREFIX, 1)[1]
            elif not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                if marked or ('"metrics"' in line and '"label"' in line):
                    raise ValueError(
                        f"{source}:{line_number}: invalid SMI JSON: {error}"
                    ) from error
                continue
            if not marked and not (
                isinstance(data, dict) and "metrics" in data and "label" in data
            ):
                continue
            where = f"{source}:{line_number}"
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("label"), str)
                or not data["label"]
                or not isinstance(data.get("metrics"), dict)
            ):
                raise ValueError(
                    f"{where}: SMI record needs a nonempty label and metrics object"
                )
            metrics = {}
            for key, stats in sorted(data["metrics"].items()):
                if not isinstance(stats, dict) or not all(
                    finite_number(stats.get(k)) for k in ("min", "mean", "max")
                ):
                    warnings.append(f"{where}: omitted unavailable metric {key}")
                    continue
                lo, mean, hi = (stats[k] for k in ("min", "mean", "max"))
                tolerance = 1e-9 * max(1, abs(lo), abs(hi))
                if lo > hi or mean < lo - tolerance or mean > hi + tolerance:
                    raise ValueError(f"{where}: invalid min/mean/max for {key}")
                metrics[key] = stats
            data = {**data, "metrics": metrics}
            if data.get("sample_status") != "ok":
                warnings.append(
                    f"{where}: sample_status={data.get('sample_status', 'unknown')}"
                )
            records.append(
                Record(source, line_number, data, *split_label(data["label"]))
            )
            count += 1
        sources.append(
            {
                "file": source,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "records": count,
            }
        )
    if not records:
        raise ValueError(
            "No SMI records found. Expected AITER_SMI_RESULT {...} or bare JSONL records."
        )
    return sorted(records, key=lambda r: r.order), sources, warnings


def slug(value):
    readable = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")[:80] or "case"
    return readable + "_" + hashlib.sha256(value.encode()).hexdigest()[:10]


def common_params(records):
    common = set(records[0].params)
    for record in records[1:]:
        common.intersection_update(record.params)
    return common


def descriptions(records, *, detailed=False):
    common = common_params(records)
    multiple_tests = len({r.test for r in records}) > 1
    multiple_sources = len({r.source for r in records}) > 1
    multiple_devices = len({r.case[2] for r in records}) > 1
    labels = []
    for r in records:
        varying = [
            (k, v) for k, v in r.params if (k, v) not in common or k in PARAM_ORDER
        ]
        shape = " ".join(f"{k}={v}" for k, v in varying)
        parts = ([r.test] if multiple_tests else []) + (
            [shape] if shape and not detailed else []
        )
        parts.append(r.function_label)
        if multiple_devices:
            parts.append(f"GPU {r.case[2]}")
        if multiple_sources:
            parts.append(r.source)
        labels.append(" | ".join(parts))
    counts = {label: labels.count(label) for label in set(labels)}
    for i, record in enumerate(records):
        if counts[labels[i]] > 1:
            labels[i] += f" | {record.source}:{record.line} (#{record.occurrence})"
        if record.data.get("sample_status") != "ok":
            labels[i] += (
                " [sample quality: "
                + str(record.data.get("sample_status", "unknown"))
                + "]"
            )
    return labels


def axis_limits(records):
    limits = {}
    for title, metrics in PANELS:
        values = [
            r.data["metrics"][key][bound]
            for r in records
            for key, *_ in metrics
            if key in r.data["metrics"]
            for bound in ("min", "max")
        ]
        if title == "Activity (%)":
            limits[title] = (0, 100)
        elif values:
            lo, hi = min(values), max(values)
            pad = max((hi - lo) * 0.07, abs(hi) * 0.01, 1)
            limits[title] = (max(0, lo - pad), hi + pad)
    return limits


def draw_panel(ax, records, panel, limits, labels, *, annotate=False):
    from matplotlib.lines import Line2D

    title, metrics = panel
    present = [m for m in metrics if any(m[0] in r.data["metrics"] for r in records)]
    for i in range(len(records)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f3f5f7", zorder=0)
        if i and records[i].case != records[i - 1].case:
            ax.axhline(i - 0.5, color="#c7cdd4", linewidth=0.8)
    for j, (key, name, color, marker) in enumerate(present):
        offset = (j - (len(present) - 1) / 2) * 0.22
        for i, record in enumerate(records):
            stats = record.data["metrics"].get(key)
            if stats is None:
                continue
            y = i + offset
            good = record.data.get("sample_status") == "ok"
            ax.hlines(
                y,
                stats["min"],
                stats["max"],
                color=color,
                alpha=0.38,
                linewidth=3,
                zorder=2,
            )
            ax.plot(
                stats["mean"],
                y,
                marker=marker,
                color=color,
                markersize=6,
                markerfacecolor=color if good else "white",
                markeredgewidth=1.1,
                zorder=3,
            )
            if annotate:
                ax.annotate(
                    f"{stats['mean']:,.1f}",
                    (stats["mean"], y),
                    xytext=(0, -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color,
                )
    ax.set_yticks(range(len(records)), labels)
    ax.set_ylim(len(records) - 0.45, -0.7)
    ax.set_xlabel(title)
    ax.grid(axis="x", color="#dce1e5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if title in limits:
        ax.set_xlim(*limits[title])
    if present:
        handles = [
            Line2D([], [], color=color, marker=marker, linestyle="None", label=name)
            for _, name, color, marker in present
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=len(handles),
            frameon=False,
            borderaxespad=0,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Metric unavailable",
            transform=ax.transAxes,
            ha="center",
            color="#777777",
        )


def save_figure(fig, stem, formats):
    import matplotlib.pyplot as plt

    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        metadata = (
            {"Date": None, "Creator": "smi_plot.py"}
            if extension == "svg"
            else {"Software": "smi_plot.py"}
        )
        fig.savefig(
            stem.with_suffix("." + extension),
            dpi=160,
            metadata=metadata,
            facecolor="white",
        )
    plt.close(fig)


def draw_plot(records, stem, title, limits, formats, *, detailed=False):
    import matplotlib.pyplot as plt

    labels = descriptions(records, detailed=detailed)
    width = 40 if detailed else 65
    labels = [
        "\n".join(textwrap.wrap(label, width, break_long_words=True))
        for label in labels
    ]
    max_lines = max(label.count("\n") + 1 for label in labels)
    common = sorted(common_params(records), key=param_key)
    subtitle = "; ".join(f"{k}={v}" for k, v in common)
    devices = ", ".join(sorted({r.case[2] for r in records}, key=natural))
    subtitle += f" | GPU {devices} | {len(records)} recorded runs"
    note = "Marker = mean; bar = observed min–max. Shared axes across this report; summaries include ramp-up."
    if any(r.data.get("sample_status") != "ok" for r in records):
        note += " Hollow markers = unverified/insufficient samples."
    header = title + "\n" + textwrap.fill(subtitle, 140)
    header_lines = header.count("\n") + 1
    if detailed:
        fig, axes = plt.subplots(
            2, 2, figsize=(16, max(8, len(records) * max_lines * 0.7 + 4))
        )
        for ax, panel in zip(axes.flat, PANELS):
            draw_panel(ax, records, panel, limits, labels, annotate=len(records) <= 8)
        fig.tight_layout(rect=(0, 0.1, 1, 1 - header_lines * 0.026), h_pad=4, w_pad=3)
        durations = [
            f"{r.function_label}: {r.data.get('duration_s', 0):.2f}s, "
            f"{r.data.get('samples', '?')} samples, {r.data.get('interval_s', '?')}s interval"
            for r in records
        ]
        footer = note + "\n" + " | ".join(durations[:4])
        if len(durations) > 4:
            footer += " | Full metadata in index.html and records.csv."
    else:
        height = max(
            5.5, len(records) * (0.29 + 0.14 * max_lines) + 1.9 + header_lines * 0.15
        )
        fig, ax = plt.subplots(figsize=(16, height))
        draw_panel(ax, records, PANELS[0], limits, labels)
        fig.tight_layout(rect=(0, 0.055, 1, 1 - (header_lines * 0.22 + 0.35) / height))
        footer = note
    fig.suptitle(header, x=0.02, y=0.985, ha="left", fontsize=12)
    fig.text(
        0.02,
        0.015,
        textwrap.fill(footer, 180),
        fontsize=8,
        color="#555555",
        va="bottom",
    )
    save_figure(fig, stem, formats)


def export_csv(records, path):
    metric_keys = sorted({key for r in records for key in r.data["metrics"]})
    fields = [
        "source",
        "line",
        "test",
        "function",
        "occurrence",
        "label",
        "device",
        "duration_s",
        "interval_s",
        "launches",
        "samples",
        "sample_status",
    ]
    fields += [
        f"{key}.{stat}"
        for key in metric_keys
        for stat in ("min", "mean", "median", "max", "n")
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {key: record.data.get(key, "") for key in fields[:12]}
            row.update(
                source=record.source,
                line=record.line,
                test=record.test,
                function=record.function,
                occurrence=record.occurrence,
            )
            for key in metric_keys:
                for stat in ("min", "mean", "median", "max", "n"):
                    row[f"{key}.{stat}"] = (
                        record.data["metrics"].get(key, {}).get(stat, "")
                    )
            writer.writerow(row)


def write_index(output, records, plots, warnings, title):
    escape = lambda value: html.escape(str(value), quote=True)
    parts = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        (
            "<style>body{font:15px system-ui;margin:32px auto;max-width:1500px;padding:0;color:#26313d}"
            "img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}"
            "td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}"
            "a{color:#1766b0}summary{cursor:pointer;padding:12px;background:#f3f5f7}"
            "details{margin:12px 0}code{overflow-wrap:anywhere}</style>"
        ),
        (
            f"<h1>{escape(title)}</h1><p>{len(records)} recorded runs. Marker = mean; bar = observed min–max. "
            "These are monitoring-window summaries, including ramp-up, not time-series plots or confidence intervals. "
            "Repeated results remain separate. Axes are shared across the report.</p>"
        ),
        "<p><a href='records.csv'>All metrics (CSV)</a> · <a href='manifest.json'>Inputs, hashes and plot index</a></p>",
    ]
    if warnings:
        parts.append(
            "<details><summary>Data notices</summary><ul>"
            + "".join(f"<li>{escape(w)}</li>" for w in warnings)
            + "</ul></details>"
        )
    for group in ("overview", "test", "case"):
        parts.append(
            f"<h2>{ {'overview': 'All tests', 'test': 'Per test', 'case': 'Per case'}[group]}</h2>"
        )
        for plot in plots:
            if plot["kind"] != group:
                continue
            links = " · ".join(
                f"<a href='{escape(path)}'>{escape(Path(path).suffix[1:].upper())}</a>"
                for path in plot["files"]
            )
            image_path = next(
                (p for p in plot["files"] if p.endswith(".png")), plot["files"][0]
            )
            body = f"<p>{links}</p><img loading='lazy' src='{escape(image_path)}' alt='{escape(plot['title'])}'>"
            if group == "overview":
                parts.append(body)
            else:
                parts.append(
                    f"<details><summary>{escape(plot['title'])}</summary>{body}</details>"
                )
    parts.append(
        "<h2>Run metadata</h2><table><tr><th>Source</th><th>Full label</th><th>GPU</th><th>Duration (s)</th>"
        "<th>Interval (s)</th><th>Samples</th><th>Status</th></tr>"
    )
    for r in records:
        values = (
            f"{r.source}:{r.line}",
            r.data["label"],
            r.data.get("device", "?"),
            r.data.get("duration_s", "?"),
            r.data.get("interval_s", "?"),
            r.data.get("samples", "?"),
            r.data.get("sample_status", "unknown"),
        )
        parts.append(
            "<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>"
        )
    parts.append("</table></html>\n")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="One log/JSONL file, or a directory scanned recursively",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: INPUT_plots for a file; INPUT/smi_plots for a directory)",
    )
    parser.add_argument(
        "--glob",
        action="append",
        help="Directory file pattern; repeatable (default: *.log, *.jsonl, *.txt)",
    )
    parser.add_argument(
        "--title",
        default="AITER GPU monitoring",
        help="Report title; hardware/date are never guessed",
    )
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "svg"), default=["png", "svg"]
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=40,
        help="Maximum rows per plot (default: 40)",
    )
    args = parser.parse_args(argv)
    if args.rows_per_page < 1:
        parser.error("--rows-per-page must be positive")
    output = args.output or (
        args.input.with_name(args.input.stem + "_plots")
        if args.input.is_file()
        else args.input / "smi_plots"
    )
    if args.input.resolve() == output.resolve() or args.input.resolve().is_relative_to(
        output.resolve()
    ):
        parser.error("Output must not be the input or an ancestor of it")
    try:
        records, sources, warnings = read_records(
            args.input, output, args.glob or ["*.log", "*.jsonl", "*.txt"]
        )
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    try:
        import matplotlib
    except ImportError:
        parser.error("Matplotlib is required: python3 -m pip install matplotlib")
    matplotlib.use("Agg")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": "aiter-smi-plot-v1",
            "svg.fonttype": "none",
            "axes.formatter.useoffset": False,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    limits = axis_limits(records)
    plots = []
    formats = sorted(set(args.formats))

    def render(group, group_records, name, title):
        pages = math.ceil(len(group_records) / args.rows_per_page)
        for page in range(pages):
            subset = group_records[
                page * args.rows_per_page : (page + 1) * args.rows_per_page
            ]
            stem = name + (f"_{page + 1:03d}" if pages > 1 else "")
            page_title = title + (f" — page {page + 1}/{pages}" if pages > 1 else "")
            draw_plot(
                subset,
                output / stem,
                page_title,
                limits,
                formats,
                detailed=group == "case",
            )
            plots.append(
                {
                    "kind": group,
                    "title": page_title,
                    "files": [stem + "." + f for f in formats],
                    "records": [{"source": r.source, "line": r.line} for r in subset],
                }
            )

    render("overview", records, "all_tests", args.title + " — all tests")
    tests, cases = defaultdict(list), defaultdict(list)
    for record in records:
        tests[record.test].append(record)
        cases[record.case].append(record)
    for test, group_records in tests.items():
        render("test", group_records, "per_test/" + slug(test), test)
    for case, group_records in cases.items():
        test, params, device = case
        shape = " ".join(f"{k}={v}" for k, v in params if k in PARAM_ORDER)
        identity = json.dumps(case, ensure_ascii=True)
        name = (
            slug(test + "_" + (shape or "case"))
            + "_"
            + hashlib.sha256(identity.encode()).hexdigest()[:10]
        )
        render(
            "case",
            group_records,
            "per_case/" + name,
            test + " | " + (shape or "case") + f" | GPU {device}",
        )
    export_csv(records, output / "records.csv")
    manifest = {
        "schema_version": 1,
        "title": args.title,
        "records": len(records),
        "sources": sources,
        "warnings": warnings,
        "axis_limits": limits,
        "plots": plots,
        "matplotlib_version": matplotlib.__version__,
        "formats": formats,
        "rows_per_page": args.rows_per_page,
        "statistics": "Marker=mean; bar=observed min-max; no pooling or deduplication; includes ramp-up",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_index(output, records, plots, warnings, args.title)
    for warning in warnings:
        print("Warning: " + warning, file=sys.stderr)
    print(
        f"Read {len(records)} records from {sum(s['records'] > 0 for s in sources)} files; "
        f"wrote {len(plots)} plots ({', '.join(formats)}) to {output}"
    )
    print(f"Open {output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
