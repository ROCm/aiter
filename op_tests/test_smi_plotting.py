# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for SMI plot capture, shutdown, output and failure behavior."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_DIR = Path(__file__).resolve().parents[1] / "aiter"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plotting = load_module("smi_plotting_test", "smi_plotting.py")

SAMPLE = {
    "label": "test_gemm.test_gemm/M=512/N=6144/K=7168/run_gemm_bpreshuffle#1",
    "device": 0,
    "duration_s": 2.0,
    "interval_s": 0.05,
    "launches": 1000,
    "samples": 40,
    "sample_status": "ok",
    "metrics": {
        "gfx_clk_mhz": {"min": 1800, "mean": 2000, "max": 2200, "n": 40},
        "fclk_mhz": {"min": 1950, "mean": 1950, "max": 1950, "n": 40},
    },
}
LINE = "AITER_SMI_RESULT " + json.dumps(SAMPLE, sort_keys=True)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        plotting._reset_after_fork()
        self.addCleanup(plotting._reset_after_fork)

    def enable(self):
        os.environ.update(
            AITER_SMI_PLOT="1", AITER_SMI_PLOT_DIR=str(self.root / "plots")
        )

    def test_disabled_and_directory_only_do_not_create_files(self):
        for flag in (None, "0", "true"):
            if flag is not None:
                os.environ["AITER_SMI_PLOT"] = flag
            os.environ["AITER_SMI_PLOT_DIR"] = str(self.root / "plots")
            plotting.record_smi_result(LINE)
        self.assertIsNone(plotting.flush_smi_plots())
        self.assertFalse((self.root / "plots").exists())

    def test_toggle_only_defaults_to_current_directory(self):
        os.environ["AITER_SMI_PLOT"] = "1"
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            plotting.record_smi_result(LINE)
        finally:
            os.chdir(previous)
        self.assertEqual(plotting._session.directory.parent, self.root / "smi_plots")
        self.assertEqual(plotting._session.input_path.read_text(), LINE + "\n")

    def test_empty_run_creates_nothing(self):
        self.enable()
        self.assertIsNone(plotting.flush_smi_plots())
        self.assertFalse((self.root / "plots").exists())

    def test_multiple_records_render_once_and_later_runs_stay_separate(self):
        self.enable()
        with patch.object(
            plotting.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)
        ) as run:
            plotting.record_smi_result(LINE)
            plotting.record_smi_result(LINE)
            run.assert_not_called()
            first = plotting._session.directory
            self.assertEqual(
                plotting._session.input_path.read_text().splitlines(), [LINE, LINE]
            )
            self.assertIsNotNone(plotting.flush_smi_plots())
            self.assertIsNone(plotting.flush_smi_plots())
            run.assert_called_once()
            plotting.record_smi_result(LINE)
            self.assertNotEqual(first, plotting._session.directory)

    def test_unwritable_output_warns_once_without_raising(self):
        self.enable()
        target = self.root / "plots"
        target.write_text("a file cannot contain reports")
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            plotting.record_smi_result(LINE)
            plotting.record_smi_result(LINE)
        self.assertEqual(errors.getvalue().count("could not capture"), 1)
        self.assertIsNone(plotting.flush_smi_plots())

    def test_plot_failure_keeps_input_and_does_not_raise(self):
        self.enable()
        plotting.record_smi_result(LINE)
        source = plotting._session.input_path
        errors = io.StringIO()
        with patch.object(
            plotting.subprocess, "run", return_value=types.SimpleNamespace(returncode=2)
        ), contextlib.redirect_stderr(errors):
            self.assertIsNone(plotting.flush_smi_plots())
        self.assertIn("status 2", errors.getvalue())
        self.assertEqual(source.read_text(), LINE + "\n")

    def test_spawn_error_keeps_input(self):
        self.enable()
        plotting.record_smi_result(LINE)
        source = plotting._session.input_path
        with patch.object(
            plotting.subprocess, "run", side_effect=OSError("cannot spawn")
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(plotting.flush_smi_plots())
        self.assertTrue(source.exists())

    def test_fork_reset_drops_parent_session(self):
        self.enable()
        plotting.record_smi_result(LINE)
        parent = plotting._session
        plotting._reset_after_fork()
        self.assertIsNone(plotting._session)
        plotting.record_smi_result(LINE)
        self.assertNotEqual(parent.directory, plotting._session.directory)
        with patch.object(plotting.os, "getpid", return_value=parent.pid + 1):
            self.assertIsNone(parent.finish())


class EmissionTests(unittest.TestCase):
    def test_original_stdout_and_file_sinks_are_preserved(self):
        # Stubs let us import the real monitor on CPU-only machines without AITER.
        package = types.ModuleType("aiter")
        package.__path__ = []
        stubs = {
            "aiter": package,
            "aiter.smi_plotting": plotting,
            "torch": types.ModuleType("torch"),
            "numpy": types.ModuleType("numpy"),
            "amdsmi": types.ModuleType("amdsmi"),
        }
        with patch.dict(sys.modules, stubs), tempfile.TemporaryDirectory() as directory:
            monitor = load_module("smi_monitor_test", "smi_monitor.py")
            with patch.dict(
                os.environ, {"AITER_SMI_PLOT": "1", "AITER_SMI_OUTPUT_PATH": ""}
            ), patch.object(plotting, "record_smi_result") as capture:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    monitor.emit_smi_result(SAMPLE)
                self.assertEqual(output.getvalue(), LINE + "\n")
                capture.assert_called_once_with(LINE)
                capture.reset_mock()
                path = Path(directory) / "existing.log"
                path.write_text("existing content\n")
                os.environ["AITER_SMI_OUTPUT_PATH"] = str(path)
                with contextlib.redirect_stdout(io.StringIO()) as stdout:
                    monitor.emit_smi_result(SAMPLE)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(path.read_text(), "existing content\n" + LINE + "\n")
                capture.assert_called_once_with(LINE)
            with patch.dict(
                os.environ, {"AITER_SMI_PLOT": "0", "AITER_SMI_OUTPUT_PATH": ""}
            ), patch.object(
                plotting, "record_smi_result"
            ) as capture, contextlib.redirect_stdout(
                io.StringIO()
            ):
                monitor.emit_smi_result(SAMPLE)
                capture.assert_not_called()


class ProcessTests(unittest.TestCase):
    def test_normal_exit_renders_once_without_polluting_stdout(self):
        code = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import smi_plotting
assert "matplotlib" not in sys.modules
for line in Path(sys.argv[2]).read_text().splitlines():
    smi_plotting.record_smi_result(line)
assert "matplotlib" not in sys.modules
print("benchmark complete")
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.log"
            source.write_text(
                LINE + "\n" + LINE.replace("bpreshuffle#1", "abpreshuffle#1") + "\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, str(MODULE_DIR), str(source)],
                cwd=root,
                env={**os.environ, "AITER_SMI_PLOT": "1", "AITER_SMI_PLOT_DIR": ""},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "benchmark complete\n")
            reports = list((root / "smi_plots").glob("run-*/report/manifest.json"))
            self.assertEqual(len(reports), 1, result.stderr)
            manifest = json.loads(reports[0].read_text())
            self.assertEqual(manifest["records"], 2)
            self.assertEqual(len(manifest["plots"]), 3)

    def test_missing_matplotlib_preserves_input_and_benchmark_exit_status(self):
        code = """
import sys
sys.path.insert(0, sys.argv[1])
import smi_plotting
smi_plotting.record_smi_result(sys.argv[2])
sys.exit(7)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matplotlib.py").write_text(
                "raise ImportError('test missing dependency')\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, str(MODULE_DIR), LINE],
                cwd=root,
                env={
                    **os.environ,
                    "AITER_SMI_PLOT": "1",
                    "AITER_SMI_PLOT_DIR": "",
                    "PYTHONPATH": str(root),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 7)
            runs = list((root / "smi_plots").glob("run-*"))
            self.assertEqual(len(runs), 1)
            self.assertEqual((runs[0] / "smi_results.log").read_text(), LINE + "\n")
            self.assertIn(
                "Matplotlib is required", (runs[0] / "plotter.log").read_text()
            )
            self.assertIn("Records preserved", result.stderr)


if __name__ == "__main__":
    unittest.main()
