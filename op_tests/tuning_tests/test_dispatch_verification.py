# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Level 1: the fresh-process dispatch verification (no GPU required).

A stub family stands in for a real operator: its config file is either one
backend name or an m,backend table (a shape without a row auto-selects ck, and
dispatch ignores rows naming retired or ignored), and its probe records the backend it ran the
way a real operator calls record_selection. Verification must accept only a
child that ran correctly and recorded exactly the selection the row names.
"""

import argparse
import importlib.util
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

DISPATCH_VERIFICATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "aiter"
    / "utility"
    / "dispatch_verification.py"
)


def _load_dispatch_verification():
    spec = importlib.util.spec_from_file_location(
        "dispatch_verification_under_test", DISPATCH_VERIFICATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dispatch_verification = _load_dispatch_verification()

STUB_FAMILY_MODULE = "stub_selection_family"
STUB_FAMILY_SOURCE = textwrap.dedent(f"""\
    import importlib.util
    import os
    import sys

    spec = importlib.util.spec_from_file_location(
        "dispatch_verification", {str(DISPATCH_VERIFICATION_PATH)!r}
    )
    dispatch_verification = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch_verification)


    def probe(row, *, warmup, iters):
        with open(os.environ["STUB_CONFIG"], encoding="utf-8") as file:
            lines = file.read().split()
        backend = lines[0] if lines else "ck"
        if backend == "m,backend":
            table = dict(line.split(",") for line in lines[1:])
            backend = table.get(str(row["m"]), "ck")
            backend = "ck" if backend in ("retired", "ignored") else backend
        if backend != "nothing":
            for _ in range(warmup + iters):
                dispatch_verification.record_selection("stub", backend=backend)
        if row.get("wrong_output"):
            raise AssertionError("output differs from the reference")
        if row.get("no_latency"):
            return {{}}
        return {{"latency_us": row.get("latency_us", 1.5)}}


    if __name__ == "__main__":
        if sys.argv[1:] == [dispatch_verification.SELECTION_PROBE_FLAG]:
            raise SystemExit(dispatch_verification.run_selection_probe(probe))
    """)


class _StubFamily(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = directory.name
        Path(self.root, f"{STUB_FAMILY_MODULE}.py").write_text(STUB_FAMILY_SOURCE)

    def config(self, backend):
        path = Path(self.root, f"tuned_{backend}.csv")
        path.write_text(backend)
        return str(path)

    def verify(self, backend, expected, row=None, config_env="STUB_CONFIG"):
        return dispatch_verification.verify_dispatch(
            family="stub",
            probe_module=STUB_FAMILY_MODULE,
            row=row or {"m": 128},
            expected=expected,
            config_env=config_env,
            config_file=self.config(backend),
            warmup=2,
            iters=3,
            timeout=60,
            repository_root=self.root,
        )


class TestVerifyDispatch(_StubFamily):
    def test_the_recorded_selection_matching_the_row_is_verified(self):
        verification = self.verify("triton", {"backend": "triton"})
        self.assertEqual(verification.status, "verified", verification.detail)
        self.assertEqual(verification.observed, [{"backend": "triton"}])
        self.assertEqual(verification.latency_us, 1.5)

    def test_a_different_recorded_selection_is_rejected(self):
        verification = self.verify("ck", {"backend": "triton"})
        self.assertEqual(verification.status, "wrong_selection")
        self.assertEqual(verification.observed, [{"backend": "ck"}])
        self.assertIn("observed", verification.detail)

    def test_an_operator_that_records_nothing_is_not_verified(self):
        verification = self.verify("nothing", {"backend": "triton"})
        self.assertEqual(verification.status, "wrong_selection")
        self.assertIn("no selection", verification.detail)

    def test_a_wrong_output_fails_even_with_the_right_selection(self):
        verification = self.verify(
            "triton", {"backend": "triton"}, {"wrong_output": True}
        )
        self.assertEqual(verification.status, "failed")
        self.assertIn("output differs", verification.detail)

    def test_no_expectation_is_reported_as_unchecked(self):
        verification = self.verify("ck", None)
        self.assertEqual(verification.status, "unchecked", verification.detail)
        self.assertEqual(verification.observed, [{"backend": "ck"}])

    def test_a_candidate_file_without_a_config_env_fails(self):
        verification = self.verify("triton", {"backend": "triton"}, config_env=None)
        self.assertEqual(verification.status, "failed")
        self.assertIn("config env", verification.detail)

    def test_a_probe_without_a_usable_latency_fails(self):
        for row in ({"no_latency": True}, {"latency_us": None}, {"latency_us": 0}):
            with self.subTest(row=row):
                verification = self.verify("triton", {"backend": "triton"}, row)
                self.assertEqual(verification.status, "failed")
                self.assertIn("latency_us", verification.detail)
                self.assertIsNone(verification.latency_us)


class TestCheckExpectedSelection(unittest.TestCase):
    def test_values_compare_as_csv_text(self):
        check = dispatch_verification._check_expected_selection
        for recorded in (32, 32.0, "32"):
            with self.subTest(recorded=recorded):
                self.assertEqual(
                    check({"block_m": "32"}, [{"block_m": recorded}])[0], "verified"
                )
        self.assertEqual(
            check({"block_m": "64"}, [{"block_m": 32}])[0], "wrong_selection"
        )


class TestRecordSelection(unittest.TestCase):
    def test_without_a_trace_file_recording_is_a_no_op(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(dispatch_verification.DISPATCH_TRACE_ENV, None)
            with mock.patch.object(dispatch_verification.os, "open") as open_file:
                dispatch_verification.record_selection("stub", backend="ck")
        open_file.assert_not_called()

    def test_each_distinct_selection_is_written_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trace.jsonl")
            with mock.patch.dict(
                os.environ, {dispatch_verification.DISPATCH_TRACE_ENV: path}
            ):
                for backend in ("ck", "ck", "triton", "ck"):
                    dispatch_verification.record_selection("stub-once", backend=backend)
            with open(path, encoding="utf-8") as file:
                self.assertEqual(len(file.readlines()), 2)
            self.assertEqual(
                dispatch_verification.read_selections(path, "stub-once"),
                [{"backend": "ck"}, {"backend": "triton"}],
            )


class TestTunerCommonRunConfig(_StubFamily):
    """A family that opts in gets run_config from TunerCommon unchanged."""

    def make_tuner(self):
        import pandas as pd

        from aiter.utility.base_tuner import TunerCommon

        class StubTuner(TunerCommon):
            ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
                **TunerCommon.ARG_DEFAULTS,
                "config_env_name": "STUB_CONFIG",
            }
            SELECTION_PROBE_MODULE = STUB_FAMILY_MODULE
            SELECTION_FAMILY = "stub"

            def _setup_specific_arguments(self):
                pass

            def expected_selection(self, row):
                return {"backend": row["backend"]} if "backend" in row else None

            def stale_reason(self, row):
                return "retired backend" if row["backend"] == "retired" else ""

            def get_gfx(self):
                return "gfx950"

            def get_gpu_model(self):
                return "mi355x"

            def get_cu_num(self):
                return 256

        tuner = StubTuner("stub", ["m"], ["backend"])
        tuner.untunedf = pd.DataFrame({"m": [128, 256], "backend": ["triton", "ck"]})
        return tuner

    def run_config(self, tuner, config_file):
        args = argparse.Namespace(warmup=1, iters=1, timeout=60)
        environment = {
            "STUB_CONFIG": config_file,
            "PYTHONPATH": os.pathsep.join(
                path for path in (self.root, os.environ.get("PYTHONPATH")) if path
            ),
        }
        with mock.patch.dict(os.environ, environment):
            return tuner.run_config(args)

    def test_rows_are_verified_against_the_configured_file(self):
        results = self.run_config(self.make_tuner(), self.config("triton"))
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["e2e_us"], 1.5)
        self.assertEqual(results[0]["selection"], "backend=triton")
        self.assertTrue(results[1]["status"].startswith("error:"), results[1])

    def test_each_shape_is_verified_against_its_row_in_the_config_file(self):
        import pandas as pd

        tuner = self.make_tuner()
        tuner.untunedf = pd.DataFrame({"m": [128, 256, 512, 1024]})
        table = Path(self.root, "tuned.csv")
        table.write_text("m,backend\n128,triton\n512,retired\n1024,ignored\n")
        with mock.patch.object(tuner, "_hardware_mismatch", return_value=""):
            results = self.run_config(tuner, str(table))
        by_m = {m: result for m, result in zip((128, 256, 512, 1024), results)}
        self.assertEqual(by_m[128]["status"], "ok")
        self.assertEqual(by_m[128]["selection"], "backend=triton")
        # No row: auto-select runs, and nothing is expected of it.
        self.assertEqual(by_m[256]["status"], "ok")
        self.assertEqual(by_m[256]["selection"], "backend=ck")
        self.assertEqual(by_m[512]["status"], "error: stale: retired backend")
        self.assertEqual(by_m[512]["selection"], "")
        self.assertIn("wrong_selection", by_m[1024]["status"])
        self.assertEqual(by_m[1024]["selection"], "backend=ck")

    def test_only_an_unchecked_tuned_row_warns(self):
        import pandas as pd

        from aiter.utility import base_tuner

        tuner = self.make_tuner()
        tuner.untunedf = pd.DataFrame({"m": [128, 256]})
        table = Path(self.root, "tuned.csv")
        for rows, warned in (("128,triton\n", True), ("", False)):
            table.write_text(f"m,backend\n{rows}")
            with (
                self.subTest(rows=rows),
                mock.patch.object(tuner, "_hardware_mismatch", return_value=""),
                mock.patch.object(tuner, "expected_selection", return_value=None),
                mock.patch.object(base_tuner.logger, "warning") as warning,
            ):
                results = self.run_config(tuner, str(table))
                self.assertEqual([r["status"] for r in results], ["ok", "ok"])
                self.assertEqual(warning.called, warned)
                if warned:
                    self.assertIn("stub: 1 tuned row(s)", warning.call_args[0][0])

    def test_verification_sees_the_model_configs_serving_merges(self):
        from aiter.jit import core

        tuner = self.make_tuner()
        tuner.ARG_DEFAULTS = {**tuner.ARG_DEFAULTS, "tune_file": "a/stub_tuned.csv"}
        model_configs = Path(self.root, "aiter", "configs", "model_configs")
        model_configs.mkdir(parents=True)
        merged = model_configs / "dsv3_stub_tuned.csv"
        for path in (merged, model_configs / "dsv3_stub_untuned.csv"):
            path.write_text("m,backend\n")
        candidate = self.config("triton")
        args = argparse.Namespace(warmup=1, iters=1, timeout=60)
        with (
            mock.patch.object(core, "AITER_ROOT_DIR", self.root),
            mock.patch.object(tuner, "_hardware_mismatch", return_value=""),
            mock.patch("aiter.utility.dispatch_verification.verify_dispatch") as spawn,
        ):
            for substituted, config_file in (
                (candidate, candidate),
                (str(merged), str(merged)),
                (None, candidate),
            ):
                tuner._substituted_config = substituted
                tuner.verify_dispatch({"m": 128}, config_file, args)
        # The tuner's own file stands in for the default table and is merged;
        # a model_configs file is not merged with itself; a file the user
        # set is verified as given, as serving would read it.
        self.assertEqual(
            [c.kwargs["config_file"] for c in spawn.call_args_list],
            [os.pathsep.join([candidate, str(merged)]), str(merged), candidate],
        )

    def test_a_row_keyed_on_another_gpu_fails_without_spawning(self):
        tuner = self.make_tuner()
        args = argparse.Namespace(warmup=1, iters=1, timeout=60)
        row = {"m": 128, "backend": "triton", "gpu_model": "mi300x"}
        with mock.patch("aiter.utility.dispatch_verification.verify_dispatch") as spawn:
            verification = tuner.verify_dispatch(row, self.config("triton"), args)
        spawn.assert_not_called()
        self.assertEqual(verification.status, "failed")
        self.assertIn("gpu_model=mi300x", verification.detail)

    def test_a_row_keyed_like_this_gpu_matches_despite_csv_formatting(self):
        tuner = self.make_tuner()
        row = {"gfx": "gfx950", "gpu_model": "AMD Instinct MI355X", "cu_num": 256.0}
        self.assertEqual(tuner._hardware_mismatch(row), "")
        self.assertIn("cu_num", tuner._hardware_mismatch({**row, "cu_num": 304.0}))


if __name__ == "__main__":
    unittest.main()
