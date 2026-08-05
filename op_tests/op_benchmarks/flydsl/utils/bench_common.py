# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared CLI / output helpers for FlyDSL A/B bench scripts.

Problem-agnostic utilities that any bench script under ``op_tests/op_benchmarks/flydsl/``
can import, covering timing arg parsing, Markdown report writing, and environment info
collection.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import torch

from ._bench_timing import MeasureConfig


# --------------------------------------------------------------------------- #
# argparse helpers
# --------------------------------------------------------------------------- #
def add_timing_args(parser) -> None:
    """Add ``--mode``, ``--warmup``, ``--bench-iters``, ``--graph-replay-iters``,
    ``--replay-iters`` to *parser*."""
    parser.add_argument(
        "--mode",
        default="graph",
        choices=("eager", "graph", "all"),
        help="Timing strategy: eager (device+host), graph (device only), or both.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        metavar="N",
        help="Warm-up iterations before timing (default: 10).",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=20,
        metavar="N",
        help="Number of timed samples to collect (default: 20).",
    )
    parser.add_argument(
        "--graph-replay-iters",
        type=int,
        default=50,
        metavar="N",
        help="Graph replays per timing sample (default: 50).",
    )
    parser.add_argument(
        "--replay-iters",
        type=int,
        default=50,
        metavar="N",
        help="Eager replays per timing sample (default: 50).",
    )


def add_output_args(parser) -> None:
    """Add ``--output / -o`` to *parser*."""
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help="Write Markdown report to FILE (also auto-generates a PNG plot beside it).",
    )


def add_verification_args(parser) -> None:
    """Add ``--verification`` and ``--verify`` shorthand to *parser*."""
    parser.add_argument(
        "--verification",
        default="none",
        choices=("none", "reference", "triton"),
        help=(
            "Verification mode: "
            "'reference' uses fp32 torch reference, "
            "'triton' checks against the Triton baseline, "
            "'none' skips (default)."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_const",
        const="reference",
        dest="verification",
        help="Shorthand for --verification reference.",
    )


def make_measure_config(args) -> MeasureConfig:
    """Build a :class:`MeasureConfig` from parsed CLI args."""
    return MeasureConfig(
        warmup_iters=args.warmup,
        bench_iters=args.bench_iters,
        replay_iters=args.replay_iters,
        graph_replay_iters=args.graph_replay_iters,
    )


# --------------------------------------------------------------------------- #
# Environment info
# --------------------------------------------------------------------------- #
def collect_env_info() -> dict:
    """Collect GPU / software version info as a plain dict."""
    info: dict = {}

    # GPU
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        try:
            from flydsl.runtime.device import get_rocm_arch
            info["gpu_arch"] = get_rocm_arch()
        except Exception:
            info["gpu_arch"] = "unknown"
    else:
        info["gpu_name"] = "N/A"
        info["gpu_arch"] = "N/A"

    # Torch
    info["torch_version"] = torch.__version__

    # Triton
    try:
        import triton
        info["triton_version"] = triton.__version__
    except Exception:
        info["triton_version"] = "N/A"

    # FlyDSL
    try:
        import flydsl
        info["flydsl_version"] = flydsl.__version__
    except Exception:
        info["flydsl_version"] = "N/A"

    # HIP / ROCm
    try:
        hip_ver = subprocess.check_output(
            ["hipcc", "--version"], stderr=subprocess.STDOUT, text=True
        )
        for line in hip_ver.splitlines():
            if "HIP version" in line or "ROCm" in line:
                info["hip_version"] = line.strip()
                break
        else:
            info["hip_version"] = hip_ver.splitlines()[0].strip()
    except Exception:
        info["hip_version"] = "N/A"

    # Git commit
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        info["git_commit"] = git_hash
    except Exception:
        info["git_commit"] = "N/A"

    return info


def format_env_section(env: dict) -> str:
    """Render env info as a Markdown block."""
    lines = ["## Environment\n"]
    for k, v in env.items():
        lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Console table + Markdown report writing
# --------------------------------------------------------------------------- #
@dataclass
class BenchRow:
    """One row of bench results (one shape x one impl x one mode)."""
    shape_label: str
    impl: str
    mode: str
    time_us: float | None      # median latency
    tflops: float | None
    verify: str                # "PASS", "FAIL(...)", "N/A", "OOM", ...
    vs_baseline: str           # "1.23x" or "-" for the baseline itself


def write_markdown_report(
    path: str,
    title: str,
    env_info: dict,
    shape_sections: list[tuple[str, list[BenchRow]]],
    summary_rows: list[BenchRow] | None = None,
    png_path: str | None = None,
) -> None:
    """Write a full Markdown bench report to *path*.

    Args:
        path: output file path.
        title: top-level heading.
        env_info: from :func:`collect_env_info`.
        shape_sections: list of ``(section_heading, rows)`` — one entry per shape.
        summary_rows: optional flat list of best-per-shape rows for the summary table.
        png_path: optional path to a plot PNG; embedded as a relative link if given.
    """
    lines: list[str] = [f"# {title}\n", ""]

    if png_path:
        rel = os.path.relpath(png_path, os.path.dirname(path))
        lines += [f"![Performance plot]({rel})\n", ""]

    if summary_rows:
        lines += ["## Summary — best impl per shape\n", ""]
        lines += _md_table(summary_rows)
        lines += [""]

    for heading, rows in shape_sections:
        lines += [f"### {heading}\n", ""]
        lines += _md_table(rows)
        lines += [""]

    lines += ["", format_env_section(env_info)]

    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def _md_table(rows: list[BenchRow]) -> list[str]:
    if not rows:
        return []
    has_tflops = any(r.tflops is not None for r in rows)
    header = "| impl | mode | time_us | verify | vs_baseline |"
    sep    = "|------|------|---------|--------|-------------|"
    if has_tflops:
        header = "| impl | mode | time_us | TFLOPs | verify | vs_baseline |"
        sep    = "|------|------|---------|--------|--------|-------------|"

    out = [header, sep]
    for r in rows:
        t = f"{r.time_us:.1f}" if r.time_us is not None else "—"
        tf = f"{r.tflops:.3f}" if (has_tflops and r.tflops is not None) else "—"
        if has_tflops:
            out.append(f"| {r.impl} | {r.mode} | {t} | {tf} | {r.verify} | {r.vs_baseline} |")
        else:
            out.append(f"| {r.impl} | {r.mode} | {t} | {r.verify} | {r.vs_baseline} |")
    return out


# --------------------------------------------------------------------------- #
# Generic result-row dict helpers (used by bench scripts)
#
# A "result row" produced by a bench script has this shape:
#   {
#     "label":        str,                        # shape label for display
#     "error":        str | missing,              # set only when shape failed entirely
#     "modes":        list[str],                  # e.g. ["graph"] or ["eager", "graph"]
#     "baseline_name": str,                       # e.g. "triton" or "hip"
#     "impls": {
#       impl_name: {
#         "timing": {mode: TimingStats | str},    # str = error/skip message
#         "tflops": {mode: float | None},         # pre-computed
#         "verify": str,
#       }
#     },
#     "baseline_times": {mode: float},           # baseline impl median_us per mode
#   }
# --------------------------------------------------------------------------- #

def print_result_table(row: dict) -> None:
    """Print a per-shape result dict to stdout as a fixed-width console table."""
    import sys

    if "error" in row:
        print(f"  ERROR: {row['error']}")
        return

    modes = row["modes"]
    baseline_name = row.get("baseline_name", "baseline")
    baseline_times = row.get("baseline_times", {})

    col_w = 36
    mode_cols = "".join(
        f"  {'time_' + m + ' (us)':>14}  {'TFLOPs':>8}  {'vs_' + baseline_name:>14}"
        for m in modes
    )
    header = f"  {'impl':<{col_w}}{mode_cols}  {'verify':<24}"
    print(f"\n  {row['label']}")
    print("  " + "-" * (len(header) - 2))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for impl_name, data in row.get("impls", {}).items():
        if "error" in data:
            print(f"  {impl_name:<{col_w}} {data['error']}")
            continue
        timing = data.get("timing", {})
        tflops_d = data.get("tflops", {})
        verify = data.get("verify", "N/A")
        line = f"  {impl_name:<{col_w}}"
        for mode in modes:
            t = timing.get(mode)
            tf = tflops_d.get(mode)
            base = baseline_times.get(mode)
            t_str  = f"{t.median_us:>14.1f}" if hasattr(t, "median_us") else f"{'—':>14}"
            tf_str = f"{tf:>8.3f}" if tf is not None else f"{'—':>8}"
            if hasattr(t, "median_us") and base and impl_name != baseline_name and t.median_us > 0:
                sp_str = f"×{base / t.median_us:.2f}"
            elif impl_name == baseline_name:
                sp_str = "baseline"
            else:
                sp_str = "—"
            line += f"  {t_str}  {tf_str}  {sp_str:>14}"
        line += f"  {verify:<24}"
        print(line)


def write_bench_markdown(
    path: str,
    title: str,
    all_rows: list[dict],
    env: dict,
    baseline_name: str = "triton",
    png_path: str | None = None,
) -> None:
    """Write a full Markdown bench report from a list of result-row dicts.

    Args:
        path: output file path.
        title: top-level heading.
        all_rows: list of result dicts from the bench script.
        env: from :func:`collect_env_info`.
        baseline_name: impl name that is the baseline (for speedup column label).
        png_path: if given, embedded as a relative link at the top.
    """
    import math

    lines: list[str] = [f"# {title}\n"]
    if png_path:
        rel = os.path.relpath(png_path, os.path.dirname(path))
        lines += [f"![Performance plot]({rel})\n", ""]

    good_rows = [r for r in all_rows if "error" not in r]
    if not good_rows:
        lines += ["No results.\n"]
    else:
        modes_seen = good_rows[0]["modes"]
        vs_col = f"vs_{baseline_name}"

        # ── Summary table ──────────────────────────────────────────────────
        lines += ["## Summary\n", ""]
        sum_hdr = ["| Shape"]
        for m in modes_seen:
            sum_hdr += [f" | best_{m}", f" | time_{m}_us", f" | {vs_col}_{m}"]
        sum_hdr.append(" |")
        lines.append("".join(sum_hdr))
        lines.append("|" + "|".join(["---"] * (1 + 3 * len(modes_seen))) + "|")

        for row in all_rows:
            if "error" in row:
                lines.append(f"| {row['label']} | ERROR |")
                continue
            baseline_times = row.get("baseline_times", {})
            cells = [row["label"]]
            for mode in modes_seen:
                best_impl, best_t = None, math.inf
                for impl, data in row.get("impls", {}).items():
                    t = data.get("timing", {}).get(mode)
                    if hasattr(t, "median_us") and t.median_us < best_t:
                        best_t, best_impl = t.median_us, impl
                cells.append(best_impl or "—")
                cells.append(f"{best_t:.1f}" if best_t < math.inf else "—")
                base = baseline_times.get(mode)
                if base and best_t < math.inf and best_impl != baseline_name:
                    cells.append(f"×{base / best_t:.2f}")
                elif best_impl == baseline_name:
                    cells.append("baseline")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

        # ── Per-shape detail tables ────────────────────────────────────────
        for row in all_rows:
            if "error" in row:
                lines += [f"### {row['label']}\n", f"ERROR: {row['error']}\n", ""]
                continue
            lines += [f"### {row['label']}\n", ""]
            modes_seen = row["modes"]
            baseline_times = row.get("baseline_times", {})

            hdr_cols = (
                ["impl"]
                + [f"time_{m}_us" for m in modes_seen]
                + [f"TFLOPs_{m}" for m in modes_seen]
                + ["verify"]
                + [f"{vs_col}_{m}" for m in modes_seen]
            )
            lines.append("| " + " | ".join(hdr_cols) + " |")
            lines.append("|" + "|".join(["---"] * len(hdr_cols)) + "|")

            for impl, data in row.get("impls", {}).items():
                if "error" in data:
                    lines.append(f"| {impl} | {data['error']} |")
                    continue
                timing = data.get("timing", {})
                tflops_d = data.get("tflops", {})
                verify = data.get("verify", "N/A")
                cells = [impl]
                for mode in modes_seen:
                    t = timing.get(mode)
                    cells.append(f"{t.median_us:.1f}" if hasattr(t, "median_us") else "—")
                for mode in modes_seen:
                    tf = tflops_d.get(mode)
                    cells.append(f"{tf:.3f}" if tf is not None else "—")
                cells.append(verify)
                for mode in modes_seen:
                    t = timing.get(mode)
                    base = baseline_times.get(mode)
                    if hasattr(t, "median_us") and base and impl != baseline_name and t.median_us > 0:
                        cells.append(f"×{base / t.median_us:.2f}")
                    elif impl == baseline_name:
                        cells.append("baseline")
                    else:
                        cells.append("—")
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines += ["", format_env_section(env)]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
