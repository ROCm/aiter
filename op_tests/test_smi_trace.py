# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks for lossless trace capture and unchanged summary data."""

import copy
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "smi_trace", Path(__file__).resolve().parents[1] / "aiter/smi_trace.py"
)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TraceTests(unittest.TestCase):
    def test_lossless_export_and_unchanged_result(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"AITER_SMI_TRACE": "1", "AITER_SMI_TRACE_DIR": directory}
        ):
            trace._reset_after_fork()
            result = {
                "label": "case/kernel#1",
                "samples": 3,
                "metrics": {"gfx_clk_mhz": {"mean": 1800}},
            }
            samples = [
                {"timestamp_s": 10.012, "gfx_clk_mhz": 2000, "fclk_mhz": None},
                {"timestamp_s": 10.067, "gfx_clk_mhz": 1700},
                {"timestamp_s": 10.124, "gfx_clk_mhz": 1700, "power_w": 1200},
            ]
            before = copy.deepcopy((result, samples))
            path = trace.emit_smi_trace(
                result, samples, start_s=10, end_s=10.2, start_unix_s=123456
            )
            data = json.loads(path.read_text().removeprefix(trace.TRACE_PREFIX))
            self.assertEqual(data["samples"], samples)
            self.assertEqual(data["sample_count"], 3)
            self.assertEqual((result, samples), before)
            self.assertEqual(data["window_start_monotonic_s"], 10)
            self.assertEqual(data["window_end_monotonic_s"], 10.2)
            trace.emit_smi_trace(
                result, samples, start_s=11, end_s=11.2, start_unix_s=123457
            )
            self.assertEqual(len(path.read_text().splitlines()), 2)

    def test_disabled_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"AITER_SMI_TRACE": "0", "AITER_SMI_TRACE_DIR": directory}
        ):
            trace._reset_after_fork()
            self.assertIsNone(
                trace.emit_smi_trace({}, [], start_s=0, end_s=1, start_unix_s=2)
            )
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bad_output_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "AITER_SMI_TRACE": "1",
                "AITER_SMI_TRACE_DIR": str(Path(directory) / "file"),
            },
        ):
            trace._reset_after_fork()
            (Path(directory) / "file").write_text("existing file")
            self.assertIsNone(
                trace.emit_smi_trace({}, [], start_s=0, end_s=1, start_unix_s=2)
            )


if __name__ == "__main__":
    unittest.main()
