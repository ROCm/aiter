# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Calibration report observes public tuner output without GPU imports."""

import importlib.util
from pathlib import Path

path = Path(__file__).with_name("run_calibration.py")
spec = importlib.util.spec_from_file_location("calibration_driver", path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def test_search_report_counts_valid_and_failed_attempts_and_actual_summary(tmp_path):
    log = tmp_path / "console.log.txt"
    log.write_text(
        "[aiter] port[g1+g2][cosine_diff=0.011174 passed~]\n"
        "[mxfp4-port] token=64 inter=384 g1 + g2 us=15.1\n"
        "[mxfp4-port] candidate failed: badg1/badg2: compilation failed\n"
        "[mxfp4-staged] GEMM1 g1 screening us=5.0\n"
        "[mxfp4-staged] GEMM2 g2 screening failed: fixture\n"
        "[mxfp4-staged] exhaustive fallback BM16 atomic: no usable reference/shortlist\n"
        "[mxfp4-staged] g1 + g2 validation failed: cosine err_ratio 0.5 > 0.1\n"
        "[aiter] Tuning Finished. tune 1 shapes, total tuning time is 107.2228 seconds\n"
    )
    report = driver.parse_search_console(log)
    assert report["validation_attempts"] == 3
    assert (
        report["g1_screens"] == report["g2_screens"] == report["fallback_groups"] == 1
    )
    assert report["finite_reference_errors"] == ["0.011174"]
    assert report["tuning_summary_seconds"] == 107.2228


def test_finalize_exited_search_records_complete_evidence_without_gpu_launch(tmp_path):
    import json

    job_dir = tmp_path / "case"
    artifact = job_dir / "search-task"
    artifact.mkdir(parents=True)
    (artifact / "task.json").write_text(
        json.dumps(
            {
                "process_start_utc": "2026-10-10T00:00:00+00:00",
                "process_exit_utc": "2026-10-10T00:00:03+00:00",
                "exit_code": 0,
                "implementation_commit": "abc",
                "environment": {},
            }
        )
    )
    (artifact / "console.log.txt").write_text(
        "[mxfp4-port] token=64 inter=384 g1 + g2 us=10.0\n"
        "[aiter] Tuning Finished. tune 1 shapes, total tuning time is 2.5 seconds\n"
    )
    (job_dir / "winner.csv").write_text("us,kernelName1,kernelName2\n10.0,g1,g2\n")
    result = driver.finalize_search(
        {"id": "case", "mode": "prune"}, job_dir, {"total_pairs": 1}
    )
    assert result["outcome"] == "valid" and result["validation_attempts"] == 1
    assert (
        result["search_wall_seconds"] == 3 and result["tuning_summary_seconds"] == 2.5
    )
    assert (job_dir / "search.json").exists()
    assert (artifact / "console.log.txt.gz").exists()
    assert not (artifact / "console.log.txt").exists()
