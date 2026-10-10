# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Offline checks through the calibration report's summarize(records) seam."""

import gzip
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def summary_module() -> "ModuleType":
    path = Path(__file__).with_name("summarize_calibration.py")
    spec = importlib.util.spec_from_file_location("summarize_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_records() -> "list[dict[str, object]]":
    records = []
    for token in (64, 128, 512, 1024, 2048, 4096, 8192):
        modes = (
            ("full", "prune", "staged")
            if token in (64, 2048, 8192)
            else ("prune", "staged")
        )
        for mode in modes:
            records.append(
                {
                    "id": f"minimax-{token}-{mode}",
                    "model": "minimax",
                    "token": token,
                    "config": "default",
                    "mode": mode,
                }
            )
    for config in ("default", "fp8", "fp4", "limit5", "combined5"):
        for mode in ("prune", "staged"):
            records.append(
                {
                    "id": f"gpt-{config}-{mode}",
                    "model": "gpt",
                    "token": 2048,
                    "config": config,
                    "mode": mode,
                }
            )
    return records


def update(
    records: "list[dict[str, object]]", identifier: "str", **result: "object"
) -> "None":
    next(record for record in records if record["id"] == identifier).update(result)


def test_missing_and_nonfinite_winners_keep_all_paired_outcomes() -> "None":
    records = manifest_records()
    update(records, "minimax-64-prune", outcome="valid", winner_us=10.0)
    update(records, "minimax-64-staged", outcome="valid", winner_us=float("inf"))
    update(records, "minimax-128-prune", outcome="valid", winner_us=8.0)

    report = summary_module().summarize(records)

    assert len(report["searches"]) == 27
    assert len(report["paired_rows"]) == 12
    rows = {
        row["token"]: row for row in report["paired_rows"] if row["model"] == "minimax"
    }
    assert rows[64]["prune"]["valid_winner"] is True
    assert rows[64]["staged"]["valid_winner"] is False
    assert rows[64]["prune_to_staged"]["available"] is False
    assert rows[128]["staged"]["outcome"] == "incomplete_run"
    assert report["prune_valid_staged_invalid"] == {
        "count": 2,
        "causes": {"normal_pair": 1, "incomplete_run": 1},
    }


def test_regret_separates_pruning_and_screening_on_matching_valid_winners() -> "None":
    records = manifest_records()
    update(records, "minimax-64-full", outcome="valid", winner_us=10.0)
    update(records, "minimax-64-prune", outcome="valid", winner_us=12.0)
    update(records, "minimax-64-staged", outcome="valid", winner_us=15.0)
    update(records, "gpt-default-prune", outcome="valid", winner_us=20.0)
    update(records, "gpt-fp8-staged", outcome="valid", winner_us=10.0)
    update(
        records,
        "minimax-2048-full",
        outcome="valid",
        winner_us=4.0,
        numerical_error=float("nan"),
    )
    update(records, "minimax-2048-prune", outcome="valid", winner_us=5.0)

    report = summary_module().summarize(records)

    anchor = next(
        row
        for row in report["paired_rows"]
        if row["model"] == "minimax" and row["token"] == 64
    )
    assert round(anchor["full_to_prune"]["regret_percent"], 6) == 20.0
    assert round(anchor["prune_to_staged"]["regret_percent"], 6) == 25.0
    assert round(anchor["full_to_staged"]["regret_percent"], 6) == 50.0
    extended = next(
        row
        for row in report["paired_rows"]
        if row["model"] == "gpt" and row["config"] == "default"
    )
    assert extended["prune_to_staged"]["available"] is False
    assert extended["full_to_prune"]["reason"] == "missing_search"
    nonfinite = next(
        row
        for row in report["paired_rows"]
        if row["model"] == "minimax" and row["token"] == 2048
    )
    assert nonfinite["full_to_prune"]["available"] is False


def test_default_and_extended_wall_totals_retain_incomplete_accounting() -> "None":
    records = manifest_records()
    for record in records:
        if record["mode"] != "full":
            record.update(
                outcome="valid",
                winner_us=10.0,
                validation_attempts=3,
                search_wall_seconds=10.0 if record["model"] == "minimax" else 100.0,
            )
    update(
        records,
        "minimax-8192-staged",
        outcome="incomplete_run",
        search_wall_seconds=None,
    )
    update(records, "gpt-fp4-staged", outcome="normal_pair", search_wall_seconds=120.0)

    report = summary_module().summarize(records)

    totals = report["wall_time_totals"]
    assert totals["default_minimax"]["prune"] == {
        "expected_searches": 7,
        "recorded_searches": 7,
        "observed_seconds": 70.0,
        "complete": True,
    }
    assert totals["default_minimax"]["staged"] == {
        "expected_searches": 7,
        "recorded_searches": 6,
        "observed_seconds": 60.0,
        "complete": False,
    }
    assert totals["extended_gpt"]["prune"]["observed_seconds"] == 500.0
    assert totals["extended_gpt"]["staged"]["observed_seconds"] == 520.0
    assert "full" not in totals["extended_gpt"]
    assert report["searches"][1]["validation_attempts"] == 3


def test_prune_valid_staged_invalid_counts_each_failure_cause() -> "None":
    records = manifest_records()
    for config, cause in zip(
        ("default", "fp8", "fp4", "limit5", "combined5"),
        (
            "missing_reference",
            "screening",
            "normal_pair",
            "worker_death",
            "incomplete_run",
        ),
    ):
        update(records, f"gpt-{config}-prune", outcome="valid", winner_us=10.0)
        update(records, f"gpt-{config}-staged", outcome=cause)

    report = summary_module().summarize(records)

    assert report["prune_valid_staged_invalid"] == {
        "count": 5,
        "causes": {
            "missing_reference": 1,
            "screening": 1,
            "normal_pair": 1,
            "worker_death": 1,
            "incomplete_run": 1,
        },
    }


def test_cli_reads_completed_search_artifacts_and_keeps_missing_searches(
    tmp_path: "Path",
) -> "None":
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_records()))
    gpu = tmp_path / "gpu"
    directory = gpu / "minimax-64-prune"
    directory.mkdir(parents=True)
    result = {
        **manifest_records()[1],
        "outcome": "valid",
        "winner_us": 10.0,
        "validation_attempts": 6,
        "search_wall_seconds": 25.0,
        "winner": {
            "kernelName1": "g1_selected",
            "kernelName2": "g2_selected",
            "err1": "1.12",
        },
        "finite_reference_errors": ["0.011172"],
    }
    (directory / "search.json").write_text(json.dumps(result))
    task_directory = directory / "search-task"
    task_directory.mkdir()
    (task_directory / "task.json").write_text(
        json.dumps({"exit_code": 0, "implementation_commit": "tested"})
    )
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_calibration.py")),
            "--manifest",
            str(manifest),
            "--gpu-dir",
            str(gpu),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert len(report["searches"]) == 27
    assert len(report["paired_rows"]) == 12
    selected = next(
        record for record in report["searches"] if record["id"] == "minimax-64-prune"
    )
    assert selected["finite_reference_errors"] == ["0.011172"]
    assert selected["winner"]["err1"] == "1.12"
    assert selected["task"]["implementation_commit"] == "tested"
    assert report["prune_valid_staged_invalid"]["causes"] == {"incomplete_run": 1}
    text = markdown.read_text()
    assert "g1_selected" in text
    assert "0.011172" in text
    assert "unavailable" in text
    assert "extended_gpt" in text


def test_cli_preserves_recorded_failure_cause_and_nonzero_task_exit(
    tmp_path: "Path",
) -> "None":
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_records()))
    gpu = tmp_path / "gpu"
    for mode, outcome, exit_code in (
        ("prune", "valid", 0),
        ("staged", "worker-death", -11),
    ):
        directory = gpu / f"minimax-64-{mode}"
        (directory / "search-task").mkdir(parents=True)
        (directory / "search.json").write_text(
            json.dumps({"outcome": outcome, "winner_us": 10.0})
        )
        (directory / "search-task" / "task.json").write_text(
            json.dumps({"exit_code": exit_code})
        )
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_calibration.py")),
            "--manifest",
            str(manifest),
            "--gpu-dir",
            str(gpu),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    assert report["prune_valid_staged_invalid"] == {
        "count": 1,
        "causes": {"worker_death": 1},
    }
    failed = next(
        record for record in report["searches"] if record["id"] == "minimax-64-staged"
    )
    assert failed["task"]["exit_code"] == -11
    assert failed["valid_winner"] is False


def test_rank_retention_uses_stable_nonatomic_ties_and_separate_atomic_group() -> (
    "None"
):
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu"
    g1_fast = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu_xcd2"
    g1_later = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu_xcd4"
    reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    scatter = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_scatter_sbm128"
    later_reduce = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_reduce_sbm128"
    atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_atomic_sbm128"
    atomic_fast = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_atomic_sbm128"
    console = "\n".join(
        [
            f"[mxfp4-staged] GEMM1 {g1_fast} screening us=1.0",
            f"[mxfp4-staged] GEMM1 {g1} screening us=2.0",
            f"[mxfp4-staged] GEMM1 {g1_later} screening us=2.0",
            "[mxfp4-staged] GEMM1 flydsl_mxmoe_g1_a4w4_64x128x128_f16in_nt_silu screening us=0.1",
            f"[mxfp4-staged] GEMM2 {atomic_fast} screening us=0.1",
            f"[mxfp4-staged] GEMM2 {atomic} screening us=0.2",
            f"[mxfp4-staged] GEMM2 {reduce} screening us=10.0",
            f"[mxfp4-staged] GEMM2 {scatter} screening us=10.0",
            f"[mxfp4-staged] GEMM2 {later_reduce} screening us=10.0",
            f"[mxfp4-staged] {g1} + {scatter} us=12.0",
        ]
    )

    report = summary_module().rank_retention(
        {"kernelName1": g1, "kernelName2": scatter}, console
    )

    assert report == {
        "block_m": 128,
        "accumulation_mode": "non_atomic",
        "screen_topk": 2,
        "g1_rank": 2,
        "g2_rank": 2,
        "g1_successful_candidates": 3,
        "g2_successful_candidates": 3,
        "both_retained": True,
        "pair_validated": True,
    }
    atomic_report = summary_module().rank_retention(
        {"kernelName1": g1, "kernelName2": atomic}, console
    )
    assert atomic_report["g2_rank"] == 2
    assert atomic_report["g2_successful_candidates"] == 2
    assert atomic_report["pair_validated"] is False


def test_failed_screen_has_no_rank_even_when_fallback_validates_pair() -> "None":
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu"
    g2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    console = (
        f"[mxfp4-staged] GEMM1 {g1} screening us=1.0\n"
        f"[mxfp4-staged] GEMM2 {g2} screening failed: fixture\n"
        f"[mxfp4-staged] {g1} + {g2} validation failed: returned fixture\n"
    )

    report = summary_module().rank_retention(
        {"kernelName1": g1, "kernelName2": g2}, console
    )

    assert report["g1_rank"] == 1
    assert report["g2_rank"] is None
    assert report["both_retained"] is False
    assert report["pair_validated"] is True


def test_fallback_count_infers_expansion_after_nonempty_shortlist_failure() -> "None":
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu"
    g2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    expanded = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x128_scatter_sbm128"
    console = (
        f"[mxfp4-staged] GEMM1 {g1} screening us=1.0\n"
        f"[mxfp4-staged] GEMM2 {g2} screening us=1.0\n"
        f"[mxfp4-staged] GEMM2 {expanded} screening us=2.0\n"
        f"[mxfp4-staged] {g1} + {g2} validation failed: returned fixture\n"
        f"[mxfp4-staged] {g1} + {expanded} us=4.0\n"
    )

    count = summary_module().staged_fallback_groups(console, screen_topk=1)

    assert count == 1
    assert summary_module().staged_fallback_groups(console, screen_topk=2) == 0


def test_fallback_count_unions_explicit_and_inferred_groups_once() -> "None":
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu"
    g2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    console = (
        "[mxfp4-staged] exhaustive fallback BM128 non-atomic: no usable reference/shortlist\n"
        "[mxfp4-staged] exhaustive fallback BM128 non-atomic: no usable reference/shortlist\n"
        "[mxfp4-staged] exhaustive fallback BM64 atomic: no usable reference/shortlist\n"
        f"[mxfp4-staged] {g1} + {g2} us=4.0\n"
    )

    assert summary_module().staged_fallback_groups(console) == 2


@pytest.mark.parametrize("compressed", (False, True))
def test_cli_reports_rank_retention_only_when_staged_log_exists(
    tmp_path: "Path", compressed: "bool"
) -> "None":
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(manifest_records()))
    gpu = tmp_path / "gpu"
    g1 = "flydsl_mxmoe_g1_a4w4_128x128x128_f16in_nt_silu"
    g2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"
    for token in (64, 128):
        for mode in ("prune", "staged"):
            directory = gpu / f"minimax-{token}-{mode}"
            (directory / "search-task").mkdir(parents=True)
            (directory / "search.json").write_text(
                json.dumps(
                    {
                        "outcome": "valid",
                        "winner_us": 10.0,
                        "winner": {"kernelName1": g1, "kernelName2": g2},
                    }
                )
            )
            (directory / "search-task" / "task.json").write_text(
                json.dumps({"exit_code": 0})
            )
    console = (
        f"[mxfp4-staged] GEMM1 {g1} screening us=1.0\n"
        f"[mxfp4-staged] GEMM2 {g2} screening us=2.0\n"
        f"[mxfp4-staged] {g1} + {g2} us=3.0\n"
    )
    console_path = gpu / "minimax-64-staged" / "search-task" / "console.log.txt"
    if compressed:
        with gzip.open(str(console_path) + ".gz", "wt") as stream:
            stream.write(console)
    else:
        console_path.write_text(console)
    output = tmp_path / "report.json"
    markdown = tmp_path / "report.md"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("summarize_calibration.py")),
            "--manifest",
            str(manifest),
            "--gpu-dir",
            str(gpu),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        check=True,
    )

    report = json.loads(output.read_text())
    rows = {
        row["token"]: row for row in report["paired_rows"] if row["model"] == "minimax"
    }
    assert rows[64]["rank_retention"]["both_retained"] is True
    assert rows[64]["rank_retention"]["pair_validated"] is True
    assert "rank_retention" not in rows[128]
    assert "Prune winner screening ranks" in markdown.read_text()
