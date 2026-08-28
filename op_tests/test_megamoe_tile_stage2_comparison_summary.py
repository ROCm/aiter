# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.megamoe_tile.summarize_stage2_comparison import (
    CASE_LABEL,
    E2E_MARKER,
    EXPECTED_SHAPE,
    STAGE2_MARKER,
    build_report,
    format_markdown,
)


POLICY = {"warmup": 20, "iterations": 100, "tail_iterations": 50}


def _write(path: Path, marker: str, document: dict[str, object]) -> Path:
    path.write_text("diagnostic preface\n" + marker + json.dumps(document) + "\n")
    return path


def _contract(**extra: object) -> dict[str, object]:
    return {
        "case_label": CASE_LABEL,
        "shape": EXPECTED_SHAPE,
        "route_pattern": "paired-rank-half-remote",
        **POLICY,
        **extra,
    }


def _stage2_timing(path: str, fields: dict[str, tuple[float, float]]):
    return {
        "path": path,
        "tail_rank_max_stats_us": {
            name: {"mean": rank_max, "p50": -1.0, "p95": -1.0}
            for name, (rank_max, _all_rank) in fields.items()
        },
        "tail_all_rank_sample_mean_us": {
            name: all_rank for name, (_rank_max, all_rank) in fields.items()
        },
    }


def test_summary_reports_only_requested_means_and_rank_max_change(tmp_path: Path):
    mori = _write(
        tmp_path / "mori.log",
        STAGE2_MARKER,
        _contract(
            path="mori",
            mode="gmm2_combine",
            timing=_stage2_timing(
                "mori_gmm2_combine",
                {
                    "standalone_fused_moe_gmm2": (250.0, 230.0),
                    "gmm2_plus_mori_combine": (1250.0, 1150.0),
                },
            ),
        ),
    )
    diagnostic = _write(
        tmp_path / "diagnostic.log",
        STAGE2_MARKER,
        _contract(
            path="candidate",
            mode="gmm2_atomic_only",
            timing=_stage2_timing("candidate_diag", {"candidate_diag": (500.0, 450.0)}),
        ),
    )
    stage2 = _write(
        tmp_path / "stage2.log",
        STAGE2_MARKER,
        _contract(
            path="candidate",
            mode="full",
            timing=_stage2_timing("candidate_full", {"candidate_full": (1500.0, 1400.0)}),
        ),
    )
    e2e = _write(
        tmp_path / "e2e.log",
        E2E_MARKER,
        _contract(
            timing_comparison={
                "mean": {"baseline_us": 1800.0, "candidate_us": 2160.0},
                "p50": {"baseline_us": -1.0, "candidate_us": -1.0},
                "p95": {"baseline_us": -1.0, "candidate_us": -1.0},
                "all_rank_mean": {
                    "baseline_us": 1700.0,
                    "candidate_us": 2040.0,
                },
            }
        ),
    )

    report = build_report(
        mori_stage2_path=mori,
        diagnostic_paths=[("production", diagnostic)],
        candidate_stage2_paths=[("production", stage2)],
        e2e_path=e2e,
        expected_policy=(20, 100, 50),
    )
    rendered = format_markdown(report)

    assert rendered.startswith(f"# {CASE_LABEL}\n")
    assert "16-rank max mean" in rendered
    assert "all-rank mean" in rendered
    assert "p50" not in rendered.lower()
    assert "p95" not in rendered.lower()
    assert "| production | 1500.00 us | 1400.00 us | +20.00% |" in rendered
    assert "| fused Stage1 + Stage2 | 2160.00 us | 2040.00 us | +20.00% |" in rendered


def test_summary_rejects_sampling_policy_mismatch(tmp_path: Path):
    bad = _contract(path="mori", mode="gmm2_combine")
    bad["iterations"] = 30
    bad["timing"] = _stage2_timing(
        "mori_gmm2_combine",
        {
            "standalone_fused_moe_gmm2": (250.0, 230.0),
            "gmm2_plus_mori_combine": (1250.0, 1150.0),
        },
    )
    mori = _write(tmp_path / "mori.log", STAGE2_MARKER, bad)

    with pytest.raises(ValueError, match="sampling policy"):
        build_report(
            mori_stage2_path=mori,
            diagnostic_paths=[],
            candidate_stage2_paths=[],
            e2e_path=tmp_path / "unused.log",
            expected_policy=(20, 100, 50),
        )
