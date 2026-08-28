# SPDX-License-Identifier: MIT
"""Build a shape-locked Stage2/E2E comparison from benchmark JSON logs.

The raw benchmarks intentionally retain full distributions for diagnosis.
This reporter is the compact sign-off view: it validates that every input was
measured with the same case and sampling policy, then reports means only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CASE_LABEL = "TPR128_TopK16_E896_H7168_I3072_EP16_A4W4"
ROUTE_PATTERN = "paired-rank-half-remote"
EXPECTED_SHAPE = {
    "tokens": 128,
    "hidden": 7168,
    "inter": 3072,
    "experts": 896,
    "topk": 16,
    "ep_size": 16,
    "gpus_per_node": 8,
    "activation": "silu",
}
STAGE2_MARKER = "MEGAMOE_EP16_STAGE2_BREAKDOWN_RESULT "
E2E_MARKER = "MEGAMOE_EP16_TWO_KERNEL_BENCH "


@dataclass(frozen=True)
class ResultRow:
    path: str
    rank_max_mean_us: float
    all_rank_mean_us: float
    relative_change_pct: float


def _load_marked_json(path: Path, marker: str) -> dict[str, object]:
    text = path.read_text()
    marker_offset = text.rfind(marker)
    if marker_offset < 0:
        raise ValueError(f"{path}: missing result marker {marker.strip()!r}")
    payload = text[marker_offset + len(marker) :]
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: malformed result JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: result payload must be a JSON object")
    return value


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected NAME=LOG_PATH")
    return name.strip(), Path(raw_path).expanduser()


def _validate_contract(
    document: dict[str, object],
    *,
    path: Path,
    expected_policy: tuple[int, int, int],
) -> None:
    shape = document.get("shape")
    if not isinstance(shape, dict):
        raise ValueError(f"{path}: missing shape contract")
    mismatches = {
        key: (shape.get(key), expected)
        for key, expected in EXPECTED_SHAPE.items()
        if shape.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{path}: shape is not {CASE_LABEL}: {mismatches}")
    if document.get("route_pattern") != ROUTE_PATTERN:
        raise ValueError(
            f"{path}: route_pattern must be {ROUTE_PATTERN!r}, "
            f"got {document.get('route_pattern')!r}"
        )
    case_label = document.get("case_label")
    if case_label != CASE_LABEL:
        raise ValueError(
            f"{path}: case_label must be {CASE_LABEL!r}, got {case_label!r}; "
            "rerun with the current benchmark driver"
        )
    policy = (
        int(document.get("warmup", -1)),
        int(document.get("iterations", -1)),
        int(document.get("tail_iterations", -1)),
    )
    if policy != expected_policy:
        raise ValueError(
            f"{path}: sampling policy {policy} does not match {expected_policy}"
        )


def _relative_change(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("comparison baseline must be positive")
    return 100.0 * (value / baseline - 1.0)


def _stage2_field(
    document: dict[str, object], field: str, *, path: Path
) -> tuple[float, float]:
    timing = document.get("timing")
    if not isinstance(timing, dict):
        raise ValueError(f"{path}: missing timing object")
    rank_stats = timing.get("tail_rank_max_stats_us")
    all_rank = timing.get("tail_all_rank_sample_mean_us")
    if not isinstance(rank_stats, dict) or not isinstance(all_rank, dict):
        raise ValueError(f"{path}: missing Stage2 mean summaries")
    field_stats = rank_stats.get(field)
    if not isinstance(field_stats, dict) or "mean" not in field_stats:
        raise ValueError(f"{path}: missing rank-max mean for {field!r}")
    if field not in all_rank:
        raise ValueError(f"{path}: missing all-rank mean for {field!r}")
    return float(field_stats["mean"]), float(all_rank[field])


def _candidate_stage2_field(
    document: dict[str, object], *, path: Path
) -> str:
    timing = document.get("timing")
    if not isinstance(timing, dict) or not isinstance(timing.get("path"), str):
        raise ValueError(f"{path}: missing candidate timing path")
    return str(timing["path"])


def _row(
    name: str,
    values: tuple[float, float],
    baseline: tuple[float, float],
) -> ResultRow:
    rank_max, all_rank = values
    return ResultRow(
        path=name,
        rank_max_mean_us=rank_max,
        all_rank_mean_us=all_rank,
        relative_change_pct=_relative_change(rank_max, baseline[0]),
    )


def build_report(
    *,
    mori_stage2_path: Path,
    diagnostic_paths: Iterable[tuple[str, Path]],
    candidate_stage2_paths: Iterable[tuple[str, Path]],
    e2e_path: Path,
    expected_policy: tuple[int, int, int],
) -> dict[str, object]:
    mori = _load_marked_json(mori_stage2_path, STAGE2_MARKER)
    _validate_contract(
        mori, path=mori_stage2_path, expected_policy=expected_policy
    )
    if mori.get("path") != "mori" or mori.get("mode") != "gmm2_combine":
        raise ValueError(
            f"{mori_stage2_path}: expected MORI gmm2_combine result"
        )

    standalone_gmm2 = _stage2_field(
        mori, "standalone_fused_moe_gmm2", path=mori_stage2_path
    )
    mori_stage2 = _stage2_field(
        mori, "gmm2_plus_mori_combine", path=mori_stage2_path
    )
    gmm2_rows = [
        _row("standalone fused_moe GMM2", standalone_gmm2, standalone_gmm2)
    ]
    for name, path in diagnostic_paths:
        document = _load_marked_json(path, STAGE2_MARKER)
        _validate_contract(document, path=path, expected_policy=expected_policy)
        if document.get("path") != "candidate" or document.get("mode") == "full":
            raise ValueError(f"{path}: expected a non-full candidate diagnostic")
        field = _candidate_stage2_field(document, path=path)
        values = _stage2_field(document, field, path=path)
        gmm2_rows.append(
            _row(f"{name} ({document.get('mode')})", values, standalone_gmm2)
        )

    stage2_rows = [
        _row("fused_moe GMM2 + MORI combine", mori_stage2, mori_stage2)
    ]
    for name, path in candidate_stage2_paths:
        document = _load_marked_json(path, STAGE2_MARKER)
        _validate_contract(document, path=path, expected_policy=expected_policy)
        if document.get("path") != "candidate" or document.get("mode") != "full":
            raise ValueError(f"{path}: expected candidate full Stage2 result")
        field = _candidate_stage2_field(document, path=path)
        values = _stage2_field(document, field, path=path)
        stage2_rows.append(_row(name, values, mori_stage2))

    e2e = _load_marked_json(e2e_path, E2E_MARKER)
    _validate_contract(e2e, path=e2e_path, expected_policy=expected_policy)
    comparison = e2e.get("timing_comparison")
    if not isinstance(comparison, dict):
        raise ValueError(f"{e2e_path}: E2E log must contain both paths")
    rank_max = comparison.get("mean")
    all_rank = comparison.get("all_rank_mean")
    if not isinstance(rank_max, dict) or not isinstance(all_rank, dict):
        raise ValueError(f"{e2e_path}: missing E2E mean comparison")
    baseline_e2e = (
        float(rank_max["baseline_us"]),
        float(all_rank["baseline_us"]),
    )
    candidate_e2e = (
        float(rank_max["candidate_us"]),
        float(all_rank["candidate_us"]),
    )
    e2e_rows = [
        _row("small-op E2E", baseline_e2e, baseline_e2e),
        _row("fused Stage1 + Stage2", candidate_e2e, baseline_e2e),
    ]

    return {
        "case_label": CASE_LABEL,
        "gmm2": [asdict(row) for row in gmm2_rows],
        "stage2": [asdict(row) for row in stage2_rows],
        "e2e": [asdict(row) for row in e2e_rows],
    }


def _markdown_table(title: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| Path | 16-rank max mean | all-rank mean | relative change (rank-max) |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {path} | {rank:.2f} us | {all_rank:.2f} us | {change:+.2f}% |".format(
                path=row["path"],
                rank=float(row["rank_max_mean_us"]),
                all_rank=float(row["all_rank_mean_us"]),
                change=float(row["relative_change_pct"]),
            )
        )
    return "\n".join(lines)


def format_markdown(report: dict[str, object]) -> str:
    sections = [f"# {report['case_label']}"]
    for key, title in (
        ("gmm2", "GMM2 and fused epilogue"),
        ("stage2", "Equivalent Stage2"),
        ("e2e", "Equivalent end-to-end MoE"),
    ):
        rows = report[key]
        if not isinstance(rows, list):
            raise TypeError(f"report section {key!r} must be a list")
        sections.append(_markdown_table(title, rows))
    return "\n\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mori-stage2", type=Path, required=True)
    parser.add_argument(
        "--diagnostic",
        action="append",
        default=[],
        type=_parse_named_path,
        metavar="NAME=LOG_PATH",
    )
    parser.add_argument(
        "--candidate-stage2",
        action="append",
        required=True,
        type=_parse_named_path,
        metavar="NAME=LOG_PATH",
    )
    parser.add_argument("--e2e", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tail-iters", type=int, default=50)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    args = parser.parse_args()
    expected_policy = (args.warmup, args.iters, args.tail_iters)
    report = build_report(
        mori_stage2_path=args.mori_stage2,
        diagnostic_paths=args.diagnostic,
        candidate_stage2_paths=args.candidate_stage2,
        e2e_path=args.e2e,
        expected_policy=expected_policy,
    )
    if args.format == "json":
        print(json.dumps(report, sort_keys=True))
    else:
        print(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
