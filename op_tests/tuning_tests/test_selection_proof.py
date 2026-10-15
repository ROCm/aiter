# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Level 1: the fresh-process selection proof (no GPU required).

A stub family stands in for a real operator: its "config file" names the
backend the operator dispatches to, and its probe records that backend the
way a real operator calls record_selection. The proof must accept only a
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

SELECTION_PROOF_PATH = (
    Path(__file__).resolve().parents[2] / "aiter" / "utility" / "selection_proof.py"
)


def _load_selection_proof():
    spec = importlib.util.spec_from_file_location(
        "selection_proof_under_test", SELECTION_PROOF_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection_proof = _load_selection_proof()

STUB_FAMILY_MODULE = "stub_selection_family"
STUB_FAMILY_SOURCE = textwrap.dedent(f"""\
    import importlib.util
    import os
    import sys

    spec = importlib.util.spec_from_file_location(
        "selection_proof", {str(SELECTION_PROOF_PATH)!r}
    )
    selection_proof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selection_proof)


    def probe(row, *, warmup, iters):
        with open(os.environ["STUB_CONFIG"], encoding="utf-8") as file:
            backend = file.read().strip()
        if backend != "nothing":
            for _ in range(warmup + iters):
                selection_proof.record_selection("stub", backend=backend)
        if row.get("wrong_output"):
            raise AssertionError("output differs from the reference")
        return {{"latency_us": 1.5}}


    if __name__ == "__main__":
        if sys.argv[1:] == [selection_proof.SELECTION_PROBE_FLAG]:
            raise SystemExit(selection_proof.run_selection_probe(probe))
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

    def prove(self, backend, expected, row=None):
        return selection_proof.prove_selection(
            family="stub",
            probe_module=STUB_FAMILY_MODULE,
            row=row or {"m": 128},
            expected=expected,
            config_env="STUB_CONFIG",
            config_file=self.config(backend),
            warmup=2,
            iters=3,
            timeout=60,
            repository_root=self.root,
        )


class TestProveSelection(_StubFamily):
    def test_the_recorded_selection_matching_the_row_is_verified(self):
        proof = self.prove("triton", {"backend": "triton"})
        self.assertEqual(proof.status, "verified", proof.detail)
        self.assertEqual(proof.observed, [{"backend": "triton"}])
        self.assertEqual(proof.latency_us, 1.5)

    def test_a_different_recorded_selection_is_rejected(self):
        proof = self.prove("ck", {"backend": "triton"})
        self.assertEqual(proof.status, "wrong_selection")
        self.assertEqual(proof.observed, [{"backend": "ck"}])
        self.assertIn("observed", proof.detail)

    def test_an_operator_that_records_nothing_proves_nothing(self):
        proof = self.prove("nothing", {"backend": "triton"})
        self.assertEqual(proof.status, "wrong_selection")
        self.assertIn("no selection", proof.detail)

    def test_a_wrong_output_fails_even_with_the_right_selection(self):
        proof = self.prove("triton", {"backend": "triton"}, {"wrong_output": True})
        self.assertEqual(proof.status, "failed")
        self.assertIn("output differs", proof.detail)

    def test_no_expectation_checks_only_that_the_operator_runs(self):
        proof = self.prove("ck", None)
        self.assertEqual(proof.status, "verified", proof.detail)
        self.assertEqual(proof.observed, [{"backend": "ck"}])


class TestRecordSelection(unittest.TestCase):
    def test_without_a_proof_file_recording_is_a_no_op(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(selection_proof.SELECTION_PROOF_ENV, None)
            with mock.patch.object(selection_proof.os, "open") as open_file:
                selection_proof.record_selection("stub", backend="ck")
        open_file.assert_not_called()

    def test_each_distinct_selection_is_written_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "proof.jsonl")
            with mock.patch.dict(
                os.environ, {selection_proof.SELECTION_PROOF_ENV: path}
            ):
                for backend in ("ck", "ck", "triton", "ck"):
                    selection_proof.record_selection("stub-once", backend=backend)
            with open(path, encoding="utf-8") as file:
                self.assertEqual(len(file.readlines()), 2)
            self.assertEqual(
                selection_proof.read_selections(path, "stub-once"),
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
                return {"backend": row["backend"]}

        tuner = StubTuner("stub", ["m"], ["backend"])
        tuner.untunedf = pd.DataFrame({"m": [128, 256], "backend": ["triton", "ck"]})
        return tuner

    def test_rows_are_proven_against_the_configured_file(self):
        tuner = self.make_tuner()
        args = argparse.Namespace(warmup=1, iters=1, timeout=60)
        environment = {
            "STUB_CONFIG": self.config("triton"),
            "PYTHONPATH": os.pathsep.join(
                path for path in (self.root, os.environ.get("PYTHONPATH")) if path
            ),
        }
        with mock.patch.dict(os.environ, environment):
            results = tuner.run_config(args)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["e2e_us"], 1.5)
        self.assertTrue(results[1]["status"].startswith("error:"), results[1])


if __name__ == "__main__":
    unittest.main()
