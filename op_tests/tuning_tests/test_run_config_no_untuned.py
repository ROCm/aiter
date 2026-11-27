# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: ``--run_config <tuned_csv>`` runs without the default untuned CSV (no GPU).

``--run_config <tuned_csv>`` reads its shapes from the tuned file, but ``_run_selected``
used to call ``pre_process`` first, and ``GemmCommonTuner.pre_process`` asserts that the
untuned CSV exists. Only that read may be skipped: a tuner's own ``pre_process`` sets up
state that ``run_config`` needs (``gemm_a8w8_mxfp8_bpreshuffle`` sets ``_mxscale_table``
and its tuned file there).

Run:
    python3 op_tests/tuning_tests/test_run_config_no_untuned.py
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def stub_tuner(calls):
    from aiter.utility.base_tuner import GemmCommonTuner

    class Stub(GemmCommonTuner):
        ARG_DEFAULTS: ClassVar[dict] = {
            **GemmCommonTuner.ARG_DEFAULTS,
            "untune_file": "no/such/untuned.csv",
            "tune_file": "no/such/tuned_out.csv",
        }

        def _setup_specific_arguments(self):
            pass

        def pre_process(self, args):
            calls["pre_process"] += 1
            return super().pre_process(args)

        def tune(self, *a, **kw):
            return []

        def getKernelName(self, kid):
            return f"k{kid}"

        def run_config(self, args):
            calls["run_config"] += 1
            return [
                {"shape": f"({r.M}, {r.N}, {r.K})", "e2e_us": 1.0, "status": "ok"}
                for r in self.untunedf.itertuples()
            ]

    key = ["gfx", "cu_num", "M", "N", "K"]
    result = ["kernelId", "splitK", "us", "kernelName", "tflops", "bw", "errRatio"]
    tuner = Stub("stub", key=key, resultList=result)
    tuner.get_cu_num = lambda: 256
    tuner.get_gfx = lambda: "gfx950"
    return tuner


def tuned_csv(tmp, **extra):
    path = os.path.join(tmp, "tuned.csv")
    row = {
        "gfx": "gfx950",
        "cu_num": 256,
        "M": 32,
        "N": 96,
        "K": 5120,
        "kernelId": 0,
        "splitK": 0,
        "us": 5.0,
        "kernelName": "k0",
        "tflops": 1.0,
        "bw": 1.0,
        "errRatio": 0.0,
    }
    row.update(extra)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


class TestRunConfigWithoutUntunedFile(unittest.TestCase):
    """--run_config <tuned_csv> must not need the default untuned CSV."""

    def make(self):
        calls = {"pre_process": 0, "run_config": 0}
        return stub_tuner(calls), calls

    def parse(self, tuner, *argv):
        return tuner.parser.parse_args(list(argv))

    def run_tuner(self, tuner, args):
        with mock.patch.object(tuner, "_set_config_env_for_run_config") as env:
            env.return_value = (None, 0)
            return tuner.run(args)

    def test_run_config_file_without_untuned_file(self):
        tuner, calls = self.make()
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse(tuner, "--run_config", tuned_csv(tmp))
            self.assertFalse(os.path.exists(args.untune_file))
            self.run_tuner(tuner, args)
        self.assertEqual(calls["run_config"], 1)
        # the tuner's own pre_process still runs (it sets up state for run_config)
        self.assertEqual(calls["pre_process"], 1)
        self.assertEqual(len(tuner.untunedf), 1)  # shapes come from the tuned CSV

    def test_compare_still_needs_the_untuned_file(self):
        tuner, calls = self.make()
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse(tuner, "--run_config", tuned_csv(tmp), "--compare")
            with self.assertRaisesRegex(AssertionError, "Not exist untuned file"):
                tuner.run(args)
        self.assertEqual(calls["run_config"], 0)

    def test_all_still_needs_the_untuned_file(self):
        tuner, calls = self.make()
        with tempfile.TemporaryDirectory() as tmp:
            args = self.parse(tuner, "--run_config", tuned_csv(tmp), "--all")
            with self.assertRaisesRegex(AssertionError, "Not exist untuned file"):
                tuner.run(args)
        self.assertEqual(calls["run_config"], 0)

    def test_run_config_without_file_still_needs_the_untuned_file(self):
        tuner, calls = self.make()
        args = self.parse(tuner, "--run_config")
        with self.assertRaisesRegex(AssertionError, "Not exist untuned file"):
            tuner.run(args)
        self.assertEqual(calls["run_config"], 0)

    def test_existing_untuned_file_is_still_read(self):
        tuner, _ = self.make()
        tuner._untuned_file_optional = True
        with tempfile.TemporaryDirectory() as tmp:
            untuned = os.path.join(tmp, "untuned.csv")
            pd.DataFrame([{"M": 1, "N": 96, "K": 5120}]).to_csv(untuned, index=False)
            self.assertEqual(len(tuner.get_untuned_gemm_list(untuned)), 1)
            missing = tuner.get_untuned_gemm_list(os.path.join(tmp, "missing.csv"))
        self.assertEqual(list(missing.columns), tuner.keys)
        self.assertTrue(missing.empty)

    def test_flag_is_off_outside_run_config(self):
        tuner, _ = self.make()
        args = self.parse(tuner, "--run_config")
        with self.assertRaises(AssertionError):
            tuner.run(args)
        self.assertFalse(tuner._untuned_file_optional)


class TestMxfp8TunerKeepsItsSetup(unittest.TestCase):
    """gemm_a8w8_mxfp8_bpreshuffle sets _mxscale_table in pre_process; run_config needs it."""

    def load_tuner_module(self):
        path = (
            REPO
            / "csrc"
            / "gemm_a8w8_mxfp8_bpreshuffle"
            / "gemm_a8w8_mxfp8_bpreshuffle_tune.py"
        )
        spec = importlib.util.spec_from_file_location("mxfp8_tune_under_test", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        self.addCleanup(sys.modules.pop, spec.name, None)
        spec.loader.exec_module(module)
        return module

    def test_mxscale_table_survives_a_missing_untuned_file(self):
        mod = self.load_tuner_module()
        key = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]
        result = ["libtype", "kernelId", "splitK", "us", "kernelName", "tflops", "bw"]
        tuner = mod.GemmA8W8MXFP8BpreshuffleTuner(
            "mxfp8_under_test", key=key, resultList=result + ["errRatio"]
        )
        tuner.get_cu_num = lambda: 256
        tuner.get_gfx = lambda: "gfx950"
        seen = {}

        def run_config(args):
            seen["mxscale_table"] = tuner._mxscale_table
            seen["tune_file"] = args.tune_file
            seen["shapes"] = len(tuner.untunedf)
            return []

        with tempfile.TemporaryDirectory() as tmp:
            tuned = tuned_csv(tmp, w_scale_block="128x128", libtype="flydsl")
            args = tuner.parser.parse_args(
                ["--run_config", tuned, "--mxscale_table", "-i", f"{tmp}/missing.csv"]
            )
            with (
                mock.patch.object(tuner, "run_config", side_effect=run_config),
                mock.patch.object(
                    tuner, "_set_config_env_for_run_config", return_value=(None, 0)
                ),
                mock.patch.object(tuner, "_restore_config_env"),
            ):
                tuner.run(args)
        self.assertTrue(seen["mxscale_table"])
        self.assertEqual(seen["tune_file"], mod.MXSCALE_TUNED_FILE)
        self.assertEqual(seen["shapes"], 1)


if __name__ == "__main__":
    unittest.main()
