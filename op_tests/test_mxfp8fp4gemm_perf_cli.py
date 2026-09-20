# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for the F8GEMM benchmark's CLI and exit status."""

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from op_tests import test_mxfp8fp4gemm_perf as perf


class TestF8GemmPerfCLI(unittest.TestCase):
    def run_benchmark(self, output, verdict, events=100):
        row = {
            "apre": 1,
            "splitk": 8,
            "asm us": 10.0,
            "asm result": verdict,
            "num_iters": 100,
            "timing_scope": "gemm_only",
            "profile_gpu_kernels": {"f8gemm_test_ABpreShuffle_test": events},
        }

        def launch(command, **kwargs):
            Path(command[command.index("--json") + 1]).write_text(json.dumps([row]))
            process = MagicMock()
            process.__enter__.return_value = process
            process.stdout = ["synthetic native output\n"]
            process.returncode = 0
            return process

        chip_info = types.ModuleType("aiter.jit.utils.chip_info")
        chip_info.get_gfx_runtime = lambda: "gfx1250"
        argv = [
            "benchmark",
            "--cases",
            "wqkv_a",
            "--data-init",
            "uniform",
            "--repeat",
            "1",
            "--max-attempts",
            "2",
            "--output-dir",
            str(output),
        ]
        with (
            patch.dict(sys.modules, {"aiter.jit.utils.chip_info": chip_info}),
            patch.object(sys, "argv", argv),
            patch.object(perf.subprocess, "Popen", side_effect=launch),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            perf.main()

    def test_failed_correctness_stops_without_retry_and_preserves_diagnostics(self):
        # A numerical failure must not be retried away even if profiling is incomplete.
        for events in (100, 99):
            with self.subTest(events=events), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                with self.assertRaisesRegex(RuntimeError, "Native correctness failed"):
                    self.run_benchmark(output, "failed", events)
                attempts = json.loads((output / "attempts.json").read_text())
                self.assertEqual(len(attempts), 1)
                self.assertFalse(attempts[0]["accepted"])
                self.assertEqual(attempts[0]["correctness"], ["failed"])
                self.assertTrue((output / attempts[0]["json"]).is_file())
                self.assertTrue((output / attempts[0]["log"]).is_file())
                self.assertFalse((output / "perf.csv").exists())

    def test_pass_and_warning_are_reported_without_failure(self):
        for verdict in ("pass", "warning"):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                self.run_benchmark(output, verdict)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary[0]["correctness"], [verdict])
                self.assertTrue((output / "perf.csv").is_file())

    def test_dry_run_needs_no_gpu_and_keeps_no_reduce(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            stdout = io.StringIO()
            with (
                patch.dict(sys.modules, {"aiter.jit.utils.chip_info": None}),
                patch.object(
                    sys, "argv", ["benchmark", "--dry-run", "--output-dir", str(output)]
                ),
                patch.object(perf.subprocess, "Popen") as launch,
                contextlib.redirect_stdout(stdout),
            ):
                perf.main()
            commands = stdout.getvalue().splitlines()
            self.assertEqual(len(commands), 12)
            self.assertTrue(all("--no-reduce" in command for command in commands))
            launch.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
