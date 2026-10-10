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
