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


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_case_map(rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, int]]:
    by_idx_backend: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    case_info: dict[str, dict[str, str]] = {}
    case_count: dict[str, int] = defaultdict(int)
    for r in rows:
        idx = r["idx"]
        by_idx_backend[idx][r["backend"]] = r
        case_info.setdefault(idx, r)
        case_count[idx] = max(case_count[idx], to_int(r["count"]))
    return by_idx_backend, case_info, case_count


def is_decode_case(r: dict[str, str]) -> bool:
    return str(r.get("is_decode", "")).lower() == "true"


def plot_phase(
    phase_name: str,
    case_ids: list[str],
    by_idx_backend: dict[str, dict[str, dict[str, str]]],
    case_info: dict[str, dict[str, str]],
    out_png: Path,
) -> None:
    if not case_ids:
        print(f"No cases for phase={phase_name}; skipping {out_png.name}")
        return

    labels: list[str] = []
    ms2d: list[float] = []
    ms3d: list[float] = []
    for idx in case_ids:
        r2 = by_idx_backend[idx].get("unified_force_2d")
        r3 = by_idx_backend[idx].get("unified_force_3d")
        if r2 is None or r3 is None:
            continue
        if not (is_ok(r2["status"]) and is_ok(r3["status"])):
            continue
        v2 = to_float(r2["ms"])
        v3 = to_float(r3["ms"])
        if v2 is None or v3 is None:
            continue
        info = case_info[idx]
        labels.append(f"idx{idx}|q{info['max_seqlen_q']}|k{info['max_seqlen_k']}")
        ms2d.append(v2)
        ms3d.append(v3)

    if not labels:
        print(f"No plottable 2d/3d rows for phase={phase_name}; skipping {out_png.name}")
        return

    xs = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.1), 5))
    ax.bar([x - width / 2 for x in xs], ms2d, width=width, label="unified_force_2d")
    ax.bar([x + width / 2 for x in xs], ms3d, width=width, label="unified_force_3d")
    ax.set_title(f"Per-case 2D vs 3D latency ({phase_name})")
    ax.set_ylabel("ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot per-case bar charts for forced 2D vs forced 3D.")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/workspaces/workspace/attn_charts"))
    ap.add_argument("--top-n", type=int, default=30, help="Top-N cases by trace count per phase.")
    ap.add_argument(
        "--only-3d-beats",
        action="store_true",
        help="Only include cases where 3D latency is lower than 2D latency.",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    by_idx_backend, case_info, case_count = build_case_map(rows)
    all_case_ids = sorted(case_info.keys(), key=lambda k: (-case_count[k], to_int(k)))

    def _eligible(case_id: str) -> bool:
        r2 = by_idx_backend[case_id].get("unified_force_2d")
        r3 = by_idx_backend[case_id].get("unified_force_3d")
        if r2 is None or r3 is None:
            return False
        if not (is_ok(r2["status"]) and is_ok(r3["status"])):
            return False
        v2 = to_float(r2["ms"])
        v3 = to_float(r3["ms"])
        if v2 is None or v3 is None:
            return False
        if args.only_3d_beats:
            return v3 < v2
        return True

    decode_ids = [cid for cid in all_case_ids if is_decode_case(case_info[cid]) and _eligible(cid)]
    prefill_ids = [cid for cid in all_case_ids if not is_decode_case(case_info[cid]) and _eligible(cid)]

    decode_ids = decode_ids[: args.top_n]
    prefill_ids = prefill_ids[: args.top_n]

    suffix = "_3d_beats_only" if args.only_3d_beats else ""
    plot_phase(
        "decode",
        decode_ids,
        by_idx_backend,
        case_info,
        args.out_dir / f"top_cases_2d_vs_3d_decode{suffix}.png",
    )
    plot_phase(
        "prefill",
        prefill_ids,
        by_idx_backend,
        case_info,
        args.out_dir / f"top_cases_2d_vs_3d_prefill{suffix}.png",
    )

    print(
        f"decode_cases_plotted={len(decode_ids)} prefill_cases_plotted={len(prefill_ids)} "
        f"only_3d_beats={args.only_3d_beats}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
