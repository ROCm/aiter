#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def is_ok(status: str) -> bool:
    return status == "ok" or status.startswith("ok:")


def to_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def to_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def row_weight(r: dict[str, str]) -> int:
    # Older CSVs include trace-frequency "count". Native-layout CSVs do not.
    if "count" in r:
        return max(1, to_int(r.get("count", "1")))
    return 1


def row_is_decode(r: dict[str, str]) -> bool:
    # Support both schemas:
    # - old: is_decode=true/false
    # - native-layout: phase=decode/prefill
    if "is_decode" in r:
        return r.get("is_decode", "").lower() == "true"
    return r.get("phase", "").lower() == "decode"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def plot_backend_summary(rows: list[dict[str, str]], out_dir: Path) -> None:
    by_backend_all: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_backend_ok: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        b = r["backend"]
        by_backend_all[b].append(r)
        if is_ok(r["status"]) and to_float(r["ms"]) is not None:
            by_backend_ok[b].append(r)

    backends = sorted(by_backend_all.keys())
    weighted_avg = []
    coverage = []
    for b in backends:
        ok_rows = by_backend_ok[b]
        total_rows = len(by_backend_all[b])
        den = sum(row_weight(r) for r in ok_rows)
        num = sum(to_float(r["ms"]) * row_weight(r) for r in ok_rows) if den > 0 else 0.0
        weighted_avg.append((num / den) if den > 0 else 0.0)
        coverage.append((len(ok_rows) / total_rows * 100.0) if total_rows > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(backends, weighted_avg)
    ax.set_title("Backend weighted avg latency (ms)")
    ax.set_ylabel("ms")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "backend_weighted_avg_ms.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(backends, coverage)
    ax.set_title("Backend coverage (% rows with successful result)")
    ax.set_ylabel("%")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "backend_coverage_pct.png", dpi=160)
    plt.close(fig)


def plot_top_cases(rows: list[dict[str, str]], out_dir: Path, top_n: int) -> None:
    # Case = idx in the CSV.
    case_counts: dict[str, int] = defaultdict(int)
    case_info: dict[str, dict[str, str]] = {}
    for r in rows:
        idx = r["idx"]
        case_counts[idx] = max(case_counts[idx], row_weight(r))
        case_info.setdefault(idx, r)

    top_cases = sorted(case_counts.keys(), key=lambda k: (-case_counts[k], to_int(k)))[:top_n]
    backends = sorted({r["backend"] for r in rows})
    x_labels = []
    values: dict[str, list[float]] = {b: [] for b in backends}

    for idx in top_cases:
        info = case_info[idx]
        label = f"idx{idx}|q{info['max_seqlen_q']}|k{info['max_seqlen_k']}"
        x_labels.append(label)
        rows_case = [r for r in rows if r["idx"] == idx]
        by_backend = {r["backend"]: r for r in rows_case}
        for b in backends:
            r = by_backend.get(b)
            if r is None or not is_ok(r["status"]) or to_float(r["ms"]) is None:
                values[b].append(0.0)
            else:
                values[b].append(to_float(r["ms"]))

    fig, ax = plt.subplots(figsize=(max(12, top_n * 1.1), 5))
    width = 0.8 / max(1, len(backends))
    xs = list(range(len(top_cases)))
    for bi, b in enumerate(backends):
        offs = [x + (bi - len(backends) / 2) * width + width / 2 for x in xs]
        ax.bar(offs, values[b], width=width, label=b)
    ax.set_title(f"Per-case latency by backend (top {top_n} cases by trace count)")
    ax.set_ylabel("ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "top_cases_backend_latency.png", dpi=160)
    plt.close(fig)


def plot_top_cases_by_phase(rows: list[dict[str, str]], out_dir: Path, top_n: int) -> None:
    def _plot_phase(phase_name: str, phase_rows: list[dict[str, str]], filename: str) -> None:
        if not phase_rows:
            print(f"No rows for phase={phase_name}; skipping.")
            return
        case_counts: dict[str, int] = defaultdict(int)
        case_info: dict[str, dict[str, str]] = {}
        for r in phase_rows:
            idx = r["idx"]
            case_counts[idx] = max(case_counts[idx], row_weight(r))
            case_info.setdefault(idx, r)

        top_cases = sorted(case_counts.keys(), key=lambda k: (-case_counts[k], to_int(k)))[:top_n]
        backends = sorted({r["backend"] for r in phase_rows})
        x_labels = []
        values: dict[str, list[float]] = {b: [] for b in backends}

        for idx in top_cases:
            info = case_info[idx]
            label = f"idx{idx}|q{info['max_seqlen_q']}|k{info['max_seqlen_k']}"
            x_labels.append(label)
            rows_case = [r for r in phase_rows if r["idx"] == idx]
            by_backend = {r["backend"]: r for r in rows_case}
            for b in backends:
                r = by_backend.get(b)
                if r is None or not is_ok(r["status"]) or to_float(r["ms"]) is None:
                    values[b].append(0.0)
                else:
                    values[b].append(to_float(r["ms"]))

        fig, ax = plt.subplots(figsize=(max(12, top_n * 1.1), 5))
        width = 0.8 / max(1, len(backends))
        xs = list(range(len(top_cases)))
        for bi, b in enumerate(backends):
            offs = [x + (bi - len(backends) / 2) * width + width / 2 for x in xs]
            ax.bar(offs, values[b], width=width, label=b)
        ax.set_title(f"Per-case latency by backend ({phase_name}, top {top_n} cases)")
        ax.set_ylabel("ms")
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    prefill_rows = [r for r in rows if not row_is_decode(r)]
    decode_rows = [r for r in rows if row_is_decode(r)]
    _plot_phase("prefill", prefill_rows, "top_cases_backend_latency_prefill.png")
    _plot_phase("decode", decode_rows, "top_cases_backend_latency_decode.png")


def plot_focus_backend_cases(
    rows: list[dict[str, str]], out_dir: Path, backend: str, top_n: int
) -> None:
    # Only cases where this backend has successful timing.
    focus_rows = [
        r for r in rows if r["backend"] == backend and is_ok(r["status"]) and to_float(r["ms"]) is not None
    ]
    if not focus_rows:
        print(f"No successful rows for backend={backend}; skipping focus chart.")
        return

    ok_case_ids = {r["idx"] for r in focus_rows}
    all_rows_ok_cases = [r for r in rows if r["idx"] in ok_case_ids]

    case_counts: dict[str, int] = defaultdict(int)
    case_info: dict[str, dict[str, str]] = {}
    for r in all_rows_ok_cases:
        idx = r["idx"]
        case_counts[idx] = max(case_counts[idx], row_weight(r))
        case_info.setdefault(idx, r)

    top_cases = sorted(case_counts.keys(), key=lambda k: (-case_counts[k], to_int(k)))[:top_n]
    backends = sorted({r["backend"] for r in all_rows_ok_cases})
    x_labels = []
    values: dict[str, list[float]] = {b: [] for b in backends}

    for idx in top_cases:
        info = case_info[idx]
        label = f"idx{idx}|q{info['max_seqlen_q']}|k{info['max_seqlen_k']}"
        x_labels.append(label)
        rows_case = [r for r in all_rows_ok_cases if r["idx"] == idx]
        by_backend = {r["backend"]: r for r in rows_case}
        for b in backends:
            r = by_backend.get(b)
            if r is None or not is_ok(r["status"]) or to_float(r["ms"]) is None:
                values[b].append(0.0)
            else:
                values[b].append(to_float(r["ms"]))

    fig, ax = plt.subplots(figsize=(max(12, top_n * 1.1), 5))
    width = 0.8 / max(1, len(backends))
    xs = list(range(len(top_cases)))
    for bi, b in enumerate(backends):
        offs = [x + (bi - len(backends) / 2) * width + width / 2 for x in xs]
        ax.bar(offs, values[b], width=width, label=b)
    ax.set_title(
        f"Per-case latency by backend (top {top_n} cases where {backend} is eligible)"
    )
    ax.set_ylabel("ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_name = f"top_cases_backend_latency_{backend}_eligible.png"
    fig.savefig(out_dir / out_name, dpi=160)
    plt.close(fig)
    print(f"Focus chart written: {out_dir / out_name}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot charts from attention backend CSV results.")
    p.add_argument("--csv", type=Path, required=True, help="Path to benchmark results CSV.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspaces/workspace/attn_charts"),
        help="Directory where charts are written.",
    )
    p.add_argument("--top-n", type=int, default=20, help="Top-N cases for per-case chart.")
    p.add_argument(
        "--focus-backend",
        type=str,
        default="ck_pa_naive",
        help="Also generate a top-cases chart restricted to cases where this backend succeeded.",
    )
    args = p.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        print("No rows found.")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_backend_summary(rows, args.out_dir)
    plot_top_cases(rows, args.out_dir, top_n=args.top_n)
    plot_top_cases_by_phase(rows, args.out_dir, top_n=args.top_n)
    if args.focus_backend:
        plot_focus_backend_cases(
            rows, args.out_dir, backend=args.focus_backend, top_n=args.top_n
        )

    unique_cases = len({r["idx"] for r in rows})
    total_rows = len(rows)
    total_events_weighted = sum(
        row_weight(r) for r in rows if r["backend"] == rows[0]["backend"]
    )

    print(f"CSV rows: {total_rows}")
    print(f"Unique cases (idx): {unique_cases}")
    print(f"Trace-weighted events (from one backend's count column): {total_events_weighted}")
    print(f"Charts written to: {args.out_dir}")
    print("- backend_weighted_avg_ms.png")
    print("- backend_coverage_pct.png")
    print("- top_cases_backend_latency.png")
    print("- top_cases_backend_latency_prefill.png")
    print("- top_cases_backend_latency_decode.png")
    if args.focus_backend:
        print(f"- top_cases_backend_latency_{args.focus_backend}_eligible.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
