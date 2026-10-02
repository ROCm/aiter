# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Plot every timestamped AITER_SMI_TRACE sample, grouped by source and case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PREFIX = "AITER_SMI_TRACE "
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FUNCTIONS = {
    "run_gemm_bpreshuffle": "B-preshuffle",
    "run_gemm_abpreshuffle": "A+B-preshuffle",
}
PANELS = (
    (
        "Clocks (MHz)",
        (
            ("gfx_clk_mhz", "GFX", "#1764b4"),
            ("fclk_mhz", "Fabric", "#d16c14"),
            ("soc_clk_mhz", "SoC", "#23844b"),
        ),
    ),
    ("Power (W)", (("power_w", "Socket power", "#983e94"),)),
    (
        "Activity (%)",
        (
            ("gfx_activity_pct", "GFX", "#1764b4"),
            ("umc_activity_pct", "Memory controller", "#d16c14"),
        ),
    ),
    ("Temperature (°C)", (("temp_hotspot_c", "Hotspot", "#b93d35"),)),
    ("VRAM used (MB)", (("vram_used_mb", "VRAM", "#367d84"),)),
)
METRIC_KEYS = tuple(key for _, metrics in PANELS for key, _, _ in metrics)
EXPLANATION = (
    "Every retained raw observation is plotted at timestamp_s minus its own "
    "window_start_monotonic_s. Columns are separate runs aligned to independent "
    "relative starts; this does not imply concurrent execution. Markers show actual "
    "observations; thin lines connect adjacent available observations as a visual "
    "guide. Missing or null metric values break the line. No resampling, smoothing, "
    "aggregation, or sample dropping is applied. Metric colors and y limits are "
    "consistent across this report; each case shares x limits across its run columns."
)


@dataclass
class Trace:
    source: str
    line: int
    data: dict
    test: str
    params: tuple
    function: str

    @property
    def case_key(self):
        return (
            self.source,
            self.test,
            tuple(sorted(self.params)),
            str(self.data["device"]),
        )

    @property
    def function_label(self):
        match = re.fullmatch(r"(.+?)(#\d+)?", self.function)
        name, occurrence = match.groups()
        return FUNCTIONS.get(name, name) + (
            occurrence if occurrence and occurrence != "#1" else ""
        )

    @property
    def elapsed(self):
        return [
            sample["timestamp_s"] - self.data["window_start_monotonic_s"]
            for sample in self.data["samples"]
        ]


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def unavailable(value):
    # AMD SMI can return the literal firmware sentinel "N/A". Keep it in
    # the source/sample export; only the plotting arrays convert it to NaN.
    return value is None or (isinstance(value, str) and value.strip().upper() == "N/A")


def parse_label(label):
    parts, params, functions = label.split("/"), [], []
    for part in parts[1:]:
        if "=" in part:
            pair = tuple(part.split("=", 1))
            if pair not in params:
                params.append(pair)
        elif part:
            functions.append(part)
    return parts[0], tuple(params), "/".join(functions) or parts[0]


def read_inputs(input_path, output_path, patterns):
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path.is_file():
        paths, base = [input_path], input_path.parent
    elif input_path.is_dir():
        paths = sorted(
            {
                path.resolve()
                for pattern in patterns
                for path in input_path.rglob(pattern)
                if path.is_file() and not path.resolve().is_relative_to(output_path)
            }
        )
        base = input_path
    else:
        raise ValueError(f"Input does not exist: {input_path}")
    records, sources, warnings = [], [], []
    for path in paths:
        raw = path.read_bytes()
        source = (
            path.relative_to(base).as_posix()
            if path.is_relative_to(base)
            else str(path)
        )
        record_count = sample_count = 0
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            line = ANSI.sub("", line).strip()
            marked = PREFIX in line
            if marked:
                line = line.split(PREFIX, 1)[1]
            elif not line.startswith("{"):
                continue
            where = f"{source}:{line_number}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                if marked or (
                    '"window_start_monotonic_s"' in line and '"samples"' in line
                ):
                    raise ValueError(f"{where}: invalid trace JSON: {error}") from error
                continue
            if not marked and not (
                isinstance(data, dict)
                and "window_start_monotonic_s" in data
                and "samples" in data
            ):
                continue
            if not isinstance(data, dict) or data.get("schema_version") != 1:
                raise ValueError(f"{where}: expected trace schema_version=1")
            if not isinstance(data.get("label"), str) or not data["label"]:
                raise ValueError(f"{where}: a nonempty label is required")
            for key in (
                "window_start_monotonic_s",
                "window_end_monotonic_s",
                "window_start_unix_s",
                "duration_s",
                "interval_s",
            ):
                if not finite(data.get(key)):
                    raise ValueError(
                        f"{where}: {key} must be finite; timestamps are never inferred"
                    )
            if (
                data["window_end_monotonic_s"] < data["window_start_monotonic_s"]
                or data["duration_s"] < 0
                or data["interval_s"] <= 0
            ):
                raise ValueError(
                    f"{where}: invalid window bounds, duration, or interval"
                )
            if "device" not in data or not isinstance(data.get("metrics"), dict):
                raise ValueError(f"{where}: device and summary metrics are required")
            samples = data.get("samples")
            if (
                not isinstance(samples, list)
                or type(data.get("sample_count")) is not int
                or data["sample_count"] != len(samples)
            ):
                raise ValueError(
                    f"{where}: sample_count must equal the number of raw samples"
                )
            previous = None
            for index, sample in enumerate(samples):
                if not isinstance(sample, dict) or not finite(
                    sample.get("timestamp_s")
                ):
                    raise ValueError(
                        f"{where}: sample {index} has no finite timestamp_s"
                    )
                timestamp = sample["timestamp_s"]
                if previous is not None and timestamp < previous:
                    raise ValueError(
                        f"{where}: sample timestamps must retain nondecreasing capture order"
                    )
                previous = timestamp
                for metric in METRIC_KEYS:
                    if not unavailable(sample.get(metric)) and not finite(
                        sample[metric]
                    ):
                        raise ValueError(
                            f"{where}: sample {index} has a nonfinite/nonnumeric {metric}"
                        )
            outside = sum(
                s["timestamp_s"] < data["window_start_monotonic_s"]
                or s["timestamp_s"] > data["window_end_monotonic_s"]
                for s in samples
            )
            if outside:
                warnings.append(
                    f"{where}: {outside} samples outside recorded bounds; retained at their actual timestamps"
                )
            if not samples:
                warnings.append(f"{where}: no samples; empty run retained")
            if data.get("sample_status") != "ok":
                warnings.append(
                    f"{where}: sample_status={data.get('sample_status', 'unknown')}"
                )
            record_count += 1
            sample_count += len(samples)
            records.append(
                Trace(source, line_number, data, *parse_label(data["label"]))
            )
        sources.append(
            {
                "file": source,
                "absolute_path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "records": record_count,
                "samples": sample_count,
            }
        )
    if not records:
        raise ValueError(
            "No raw AITER_SMI_TRACE records found. Aggregate AITER_SMI_RESULT logs do not contain sample traces."
        )
    return records, sources, warnings


def metric_limits(records):
    limits = []
    for title, metrics in PANELS:
        values = [
            sample[key]
            for record in records
            for sample in record.data["samples"]
            for key, _, _ in metrics
            if finite(sample.get(key))
        ]
        if not values:
            limits.append((0.0, 100.0) if title == "Activity (%)" else (0.0, 1.0))
            continue
        low, high = min(values), max(values)
        padding = max((high - low) * 0.07, abs(high) * 0.025, 1)
        if title == "Activity (%)":
            limits.append((min(0, low - padding), max(102, high + padding)))
        elif title in ("Power (W)", "VRAM used (MB)"):
            limits.append((min(0, low - padding), high + padding))
        else:
            limits.append((max(0, low - padding), high + padding))
    return limits


def series(record, metric):
    # Keep arrays the same length as the sample list. NaN breaks the line at every missing value.
    return record.elapsed, [
        sample[metric] if finite(sample.get(metric)) else math.nan
        for sample in record.data["samples"]
    ]


def case_description(record):
    params = dict(record.params)
    shape = "  ·  ".join(
        f"{key}={params[key]}"
        for key in ("M", "N", "K", "T", "H", "D")
        if key in params
    )
    settings = "  ·  ".join(
        f"{key}={value}"
        for key, value in record.params
        if key not in ("M", "N", "K", "T", "H", "D")
    )
    return shape or "Recorded case", settings


def plot_case(records, path_base, limits, *, synthetic=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "aiter-raw-samples-v1",
            "path.simplify": False,
            "agg.path.chunksize": 0,
            "text.color": "#203047",
            "axes.labelcolor": "#203047",
            "axes.edgecolor": "#c6cfda",
            "xtick.color": "#52637a",
            "ytick.color": "#52637a",
            "savefig.facecolor": "white",
        }
    )
    columns = len(records)
    fig, axes = plt.subplots(
        len(PANELS),
        columns,
        figsize=(7.7 * columns + 0.7, 15.5),
        squeeze=False,
        gridspec_kw={
            "left": 0.075,
            "right": 0.98,
            "bottom": 0.11,
            "top": 0.785,
            "hspace": 0.28,
            "wspace": 0.13,
            "height_ratios": [1.35, 1, 1, 1, 0.9],
        },
    )
    max_time = max(
        [record.data["duration_s"] for record in records]
        + [
            record.data["window_end_monotonic_s"]
            - record.data["window_start_monotonic_s"]
            for record in records
        ]
        + [t for record in records for t in record.elapsed]
        + [0.001]
    )
    min_time = min([0.0] + [t for record in records for t in record.elapsed])
    for column, record in enumerate(records):
        first = axes[0, column]
        first.set_title(
            f"{record.function_label}\n{len(record.data['samples']):,} raw observations · {record.data['duration_s']:.3f} s · source line {record.line}\nIndependent run start; requested interval {record.data['interval_s']:g} s",
            fontsize=11,
            pad=15,
            linespacing=1.6,
        )
        for row, ((title, metrics), y_limits) in enumerate(zip(PANELS, limits)):
            ax = axes[row, column]
            available = 0
            for key, label, color in metrics:
                x, y = series(record, key)
                count = sum(math.isfinite(value) for value in y)
                if not count:
                    continue
                available += count
                ax.plot(
                    x,
                    y,
                    color=color,
                    linewidth=0.55,
                    marker="o",
                    markersize=1.9,
                    markeredgewidth=0,
                    alpha=0.9,
                    label=f"{label} ({count:,})",
                    zorder=3,
                )
            if not available:
                ax.text(
                    0.5,
                    0.5,
                    "No recorded values",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="#7d8897",
                )
            ax.set_xlim(min_time, max_time)
            ax.set_ylim(*y_limits)
            ax.grid(color="#e2e8ef", linewidth=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(length=0, pad=6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
            if column == 0:
                ax.set_ylabel(title, fontweight="bold", labelpad=13)
            else:
                ax.tick_params(axis="y", labelleft=False)
            if row != len(PANELS) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(
                    "Elapsed time from this run’s start (s)",
                    labelpad=10,
                    fontweight="bold",
                )
            if available:
                ax.legend(
                    loc="upper right",
                    fontsize=8,
                    framealpha=0.9,
                    edgecolor="none",
                    ncol=len(metrics),
                )
    shape, settings = case_description(records[0])
    prefix = "SYNTHETIC DEVELOPMENT FIXTURE · " if synthetic else ""
    fig.text(
        0.075,
        0.96,
        prefix + "SMI raw samples by case",
        fontsize=21 if not synthetic else 15,
        fontweight="bold",
    )
    fig.text(0.075, 0.93, records[0].test, fontsize=12)
    fig.text(
        0.075,
        0.905,
        shape + f"  ·  GPU {records[0].data['device']}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.882,
        "\n".join(textwrap.wrap(settings, width=145 if columns > 1 else 75)),
        fontsize=9,
        color="#52637a",
        linespacing=1.5,
    )
    fig.text(
        0.075, 0.843, "Source: " + records[0].source, fontsize=9.5, color="#52637a"
    )
    fig.text(
        0.075,
        0.063,
        "Each marker is a recorded observation. Thin lines connect adjacent values; missing values leave gaps. Legend counts are available values per metric.",
        fontsize=9,
    )
    fig.text(
        0.075,
        0.043,
        "Separate runs use their own recorded starts. All samples and original timestamps are retained; there is no smoothing, averaging, or downsampling.",
        fontsize=9,
        color="#52637a",
    )
    fig.text(
        0.075,
        0.024,
        "Y limits are shared across all cases in this report. Full run labels, source hashes, and original timestamps: index.html, manifest.json, raw_points.csv.",
        fontsize=8.5,
        color="#6c7a8c",
    )
    metadata = {"Creator": "smi_trace_plot.py", "Description": EXPLANATION}
    fig.savefig(path_base.with_suffix(".png"), dpi=160, metadata=metadata)
    fig.savefig(path_base.with_suffix(".svg"), metadata={**metadata, "Date": None})
    plt.close(fig)


def build(input_path, output_path, patterns=None, *, synthetic=False, max_columns=2):
    input_path, output_path = Path(input_path), Path(output_path)
    records, sources, warnings = read_inputs(
        input_path, output_path, patterns or ["*.log", "*.jsonl"]
    )
    output_path.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for record in records:
        groups[record.case_key].append(record)
    limits = metric_limits(records)
    charts, runs = [], []
    for record in records:
        metadata = {
            key: value for key, value in record.data.items() if key != "samples"
        }
        runs.append(
            {
                "source": record.source,
                "source_line": record.line,
                "metadata": metadata,
                "elapsed_first_s": record.elapsed[0] if record.elapsed else None,
                "elapsed_last_s": record.elapsed[-1] if record.elapsed else None,
                "available_values": {
                    key: sum(
                        finite(sample.get(key)) for sample in record.data["samples"]
                    )
                    for key in METRIC_KEYS
                },
            }
        )
    for key, case_records in groups.items():
        shape, settings = case_description(case_records[0])
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False).encode()
        ).hexdigest()[:10]
        readable = re.sub(r"[^a-zA-Z0-9_-]+", "_", shape).strip("_")[:75]
        for page_start in range(0, len(case_records), max_columns):
            page = case_records[page_start : page_start + max_columns]
            suffix = (
                f"_page{page_start // max_columns + 1}"
                if len(case_records) > max_columns
                else ""
            )
            stem = "case_" + readable + "_" + digest + suffix
            plot_case(page, output_path / stem, limits, synthetic=synthetic)
            charts.append(
                {
                    "source": key[0],
                    "test": key[1],
                    "parameters": list(key[2]),
                    "device": key[3],
                    "shape": shape,
                    "settings": settings,
                    "png": stem + ".png",
                    "svg": stem + ".svg",
                    "source_lines": [r.line for r in page],
                    "sample_count": sum(len(r.data["samples"]) for r in page),
                }
            )
    csv_columns = [
        "source",
        "source_line",
        "label",
        "device",
        "sample_index",
        "timestamp_s",
        "elapsed_s",
        "window_start_monotonic_s",
        "window_end_monotonic_s",
        "window_start_unix_s",
        *METRIC_KEYS,
        "raw_sample_json",
    ]
    with (output_path / "raw_points.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_columns)
        writer.writeheader()
        for record in records:
            for sample_index, sample in enumerate(record.data["samples"]):
                writer.writerow(
                    {
                        "source": record.source,
                        "source_line": record.line,
                        "label": record.data["label"],
                        "device": record.data["device"],
                        "sample_index": sample_index,
                        "timestamp_s": sample["timestamp_s"],
                        "elapsed_s": sample["timestamp_s"]
                        - record.data["window_start_monotonic_s"],
                        **{
                            key: record.data[key]
                            for key in (
                                "window_start_monotonic_s",
                                "window_end_monotonic_s",
                                "window_start_unix_s",
                            )
                        },
                        **{key: sample.get(key, "") for key in METRIC_KEYS},
                        "raw_sample_json": json.dumps(
                            sample, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
    manifest = {
        "schema_version": 1,
        "synthetic_development_fixture": synthetic,
        "input": str(input_path.resolve()),
        "interpretation": EXPLANATION,
        "record_count": len(records),
        "case_count": len(groups),
        "sample_count": sum(len(r.data["samples"]) for r in records),
        "sources": sources,
        "panels": [
            {
                "title": title,
                "limits": y_limits,
                "metrics": [key for key, _, _ in metrics],
            }
            for (title, metrics), y_limits in zip(PANELS, limits)
        ],
        "charts": charts,
        "runs": runs,
        "warnings": warnings,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_html(output_path, manifest)
    return manifest


def write_html(output_path, manifest):
    pieces = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SMI raw samples by case</title>',
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1550px;margin:36px auto;padding:0 24px;color:#203047}img{width:100%;border:1px solid #dbe3ec}a{color:#1764b4}code{overflow-wrap:anywhere}td,th{text-align:left;padding:8px;vertical-align:top;border-bottom:1px solid #e5eaf0}table{border-collapse:collapse;width:100%;font-size:13px}.note{background:#edf3f8;padding:16px}nav{display:flex;flex-wrap:wrap;gap:8px}nav a{background:#edf3f8;padding:7px 12px;text-decoration:none}</style>",
        "<h1>"
        + (
            "SYNTHETIC DEVELOPMENT FIXTURE · "
            if manifest["synthetic_development_fixture"]
            else ""
        )
        + "SMI raw samples by case</h1>",
        f'<p><strong>{manifest["case_count"]} cases · {manifest["record_count"]} runs · {manifest["sample_count"]:,} raw observations</strong></p>',
        '<p class="note">' + html.escape(EXPLANATION) + "</p>",
        '<p><a href="raw_points.csv">Every original sample (CSV)</a> · <a href="manifest.json">Run metadata, source hashes and counts (JSON)</a></p><nav>',
    ]
    for index, chart in enumerate(manifest["charts"], 1):
        pieces.append(f'<a href="#case-{index}">{html.escape(chart["shape"])}</a>')
    pieces.append("</nav>")
    for index, chart in enumerate(manifest["charts"], 1):
        pieces.extend(
            [
                f'<h2 id="case-{index}">{html.escape(chart["shape"])}</h2>',
                f'<p><code>{html.escape(chart["test"])}</code> · GPU {html.escape(chart["device"])}<br>{html.escape(chart["settings"])}</p>',
                f'<p>Source: <code>{html.escape(chart["source"])}</code> · {chart["sample_count"]:,} observations · <a href="{chart["png"]}">PNG</a> · <a href="{chart["svg"]}">SVG</a></p>',
                f'<a href="{chart["svg"]}"><img loading="lazy" src="{chart["png"]}" alt="All timestamped raw clock, power, activity, temperature and VRAM samples for separate function runs"></a>',
                "<details><summary>Original run labels and timing origins</summary><table><tr><th>Source line</th><th>Raw observations</th><th>Recorded origins</th><th>Full label</th></tr>",
            ]
        )
        for run in manifest["runs"]:
            if (
                run["source"] == chart["source"]
                and run["source_line"] in chart["source_lines"]
            ):
                data = run["metadata"]
                pieces.append(
                    f'<tr><td>{run["source_line"]}</td><td>{data["sample_count"]}</td><td>Monotonic: {data["window_start_monotonic_s"]}<br>Unix: {data["window_start_unix_s"]}</td><td><code>{html.escape(data["label"])}</code></td></tr>'
                )
        pieces.append("</table></details>")
    if manifest["warnings"]:
        pieces.append(
            "<h2>Data warnings</h2><ul>"
            + "".join("<li>" + html.escape(w) + "</li>" for w in manifest["warnings"])
            + "</ul>"
        )
    pieces.append("</html>\n")
    (output_path / "index.html").write_text("\n".join(pieces), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="Raw trace log/JSONL file, or recursively searched directory",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "report",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        help="Directory glob, repeatable (default: *.log and *.jsonl)",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=2,
        choices=range(1, 5),
        help="Separate run columns per case page",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Mark development fixture output prominently as synthetic",
    )
    args = parser.parse_args()
    try:
        manifest = build(
            args.input,
            args.output,
            args.pattern,
            synthetic=args.synthetic,
            max_columns=args.max_columns,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(
        f"Created {len(manifest['charts'])} case plots: {manifest['record_count']} runs, {manifest['sample_count']:,} raw observations"
    )
    print(f"Report: {args.output.resolve() / 'index.html'}")
    for warning in manifest["warnings"]:
        print("Warning: " + warning)


if __name__ == "__main__":
    main()
