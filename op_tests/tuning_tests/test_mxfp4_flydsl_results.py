# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks of coupled tuning from CSV input to persisted results.

The GPU candidate execution boundary supplies controlled observations. The
candidate enumeration, winner selection, batching and CSV writers stay real.
"""

import contextlib
import importlib
import io
import multiprocessing
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
RESULT_COLUMNS = [
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
]
G1 = "flydsl_mxmoe_g1_a4w4_16x128x256_f16in_nt"
G2_PREFIX = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128"
OBSERVATIONS = {
    G2_PREFIX + "_atomic_nt_sbm16": (float("nan"), 0.01),
    G2_PREFIX + "_atomic_persist_nt_sbm16": (float("inf"), 0.01),
    G2_PREFIX + "_atomic_sbm16": (0.0, 0.01),
    G2_PREFIX + "_atomic_persist_sbm16": (-2.0, 0.01),
    G2_PREFIX + "_reduce_nt_sbm16": (9.0, 0.01),
    G2_PREFIX + "_reduce_persist_nt_sbm16": (4.0, 0.5),
    G2_PREFIX + "_reduce_sbm16": (7.0, 0.03),
    G2_PREFIX + "_reduce_persist_sbm16": (1.0, float("nan")),
}


def _tuner_module():
    module = importlib.import_module("csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune")
    torch.set_default_device("cpu")
    return module


def _controlled_tuner_class(module):
    class ControlledTuner(module.Mxfp4FlydslTuner):
        def _candidate_rows(self, row, full_search=False):
            return [
                candidate
                for candidate in super()._candidate_rows(row, full_search=full_search)
                if candidate["kernelName1"] == G1
                and candidate["kernelName2"] in OBSERVATIONS
            ]

        def _run_candidate(self, row, candidate, args):
            us, error = OBSERVATIONS[candidate["kernelName2"]]
            candidate.update(
                us=us,
                us1=us,
                pipeline_us=us,
                error=error,
                err1=f"{error:.1%}",
                err2=f"{error:.1%}",
            )
            return us

    return ControlledTuner


def _input_row(token=1):
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": token,
        "model_dim": 512,
        "inter_dim": 512,
        "expert": 16,
        "topk": 4,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
    }


def _cpu_shape_process(payload, out_q, index):
    """Use the real isolated-worker aggregation without launching GPU work."""
    keys, row, args, _gpu = payload
    module = _tuner_module()

    class WorkerTuner(_controlled_tuner_class(module)):
        def _run_candidate(self, row, candidate, args):
            if row["token"] == 1 and candidate["kernelName2"].endswith("_atomic_sbm16"):
                os._exit(17)
            return super()._run_candidate(row, candidate, args)

    tuner = WorkerTuner.__new__(WorkerTuner)
    tuner.keys = keys
    out_q.put((index, tuner._tune_one_shape(row, args)))


class TestCoupledTuningResults(unittest.TestCase):
    def setUp(self):
        self.module = _tuner_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.input_file = self.root / "input.csv"
        self.output_file = self.root / "tuned.csv"
        self.profile_file = self.root / "profile.csv"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.object(torch.cuda, "device_count", return_value=0)
        )
        self.stack.enter_context(
            patch.object(torch.cuda, "current_device", return_value=0)
        )
        self.stack.enter_context(
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=256),
            )
        )
        self.stack.enter_context(
            patch.object(self.module, "get_gfx_runtime", return_value="gfx950")
        )

    def run_csv(
        self,
        rows,
        tuner_class=None,
        profile=True,
        mp=1,
        batch=2,
        timeout=0,
        existing_rows=None,
    ):
        pd.DataFrame(rows, columns=KEYS).to_csv(self.input_file, index=False)
        if existing_rows is not None:
            pd.DataFrame(existing_rows, columns=KEYS + RESULT_COLUMNS).to_csv(
                self.output_file, index=False
            )
        tuner_class = tuner_class or _controlled_tuner_class(self.module)
        tuner = tuner_class("test", KEYS, RESULT_COLUMNS)
        args = tuner.parser.parse_args(
            [
                "--mxfp4-flydsl",
                "-i",
                str(self.input_file),
                "-o",
                str(self.output_file),
                "--profile_file",
                str(self.profile_file) if profile else "",
                "--mp",
                str(mp),
                "--batch",
                str(batch),
                "--timeout",
                str(timeout),
                "--errRatio",
                "0.1",
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tuner.run(args)

    def test_csv_saves_fastest_valid_winner_and_all_candidate_observations(self):
        self.run_csv([_input_row()])

        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["kernelName2"].tolist(), [G2_PREFIX + "_reduce_sbm16"])
        self.assertEqual(tuned["us"].tolist(), [7.0])
        self.assertEqual(tuned["us1"].tolist(), [7.0])
        self.assertEqual(tuned["us2"].tolist(), [0])
        self.assertEqual(tuned.columns.tolist(), KEYS + RESULT_COLUMNS)

        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 8)
        self.assertTrue(set(KEYS).issubset(profile.columns))
        self.assertEqual(set(profile["precision"]), {"A4W4"})
        self.assertEqual(set(profile["search_mode"]), {"prune"})
        self.assertEqual(set(profile["kernelName1"]), {G1})
        observations = profile.set_index("kernelName2")
        self.assertEqual(observations.loc[G2_PREFIX + "_reduce_sbm16", "error"], 0.03)
        self.assertEqual(observations.loc[G2_PREFIX + "_reduce_sbm16", "status"], "ok")
        self.assertEqual(
            observations.loc[G2_PREFIX + "_atomic_sbm16", "status"], "invalid_time"
        )
        self.assertEqual(
            observations.loc[G2_PREFIX + "_reduce_persist_nt_sbm16", "status"],
            "accuracy_failed",
        )
        failed_candidates = profile[profile["status"] != "ok"]
        self.assertEqual(len(failed_candidates), 6)
        self.assertTrue(failed_candidates["failure_reason"].str.len().gt(0).all())
        failed_shapes = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertTrue(failed_shapes.empty)

    def test_shape_exception_preserves_later_winners_profile_and_full_failure_reason(
        self,
    ):
        failure_reason = "candidate catalog unavailable; " + "diagnostic context " * 30

        class ShapeFailure(_controlled_tuner_class(self.module)):
            def _candidate_rows(self, row, full_search=False):
                if row["token"] == 1:
                    raise RuntimeError(failure_reason)
                return super()._candidate_rows(row, full_search)

        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row(1), _input_row(2), _input_row(3)], ShapeFailure)
        self.assertEqual(stopped.exception.code, 1)

        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["token"].tolist(), [2, 3])
        self.assertEqual(tuned["us"].tolist(), [7.0, 7.0])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 16)
        self.assertEqual(set(profile["token"]), {2, 3})
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])
        self.assertTrue(set(KEYS).issubset(failures.columns))
        self.assertIn(failure_reason, failures.iloc[0]["failure_reason"])

    def test_dead_worker_preserves_completed_profiles_and_other_shapes_finish(self):
        run_isolated = self.module._run_shapes_isolated

        def cpu_workers(payloads, mp_num, _ctx):
            return run_isolated(
                payloads,
                mp_num,
                multiprocessing.get_context("spawn"),
                entry=_cpu_shape_process,
            )

        with (
            patch.object(torch.cuda, "device_count", return_value=2),
            patch.object(self.module, "_run_shapes_isolated", side_effect=cpu_workers),
            self.assertRaises(SystemExit) as stopped,
        ):
            self.run_csv([_input_row(1), _input_row(2), _input_row(3)], mp=2, batch=3)
        self.assertEqual(stopped.exception.code, 1)

        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["token"].tolist(), [2, 3])
        self.assertEqual(tuned["us"].tolist(), [7.0, 7.0])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 18)
        dead_shape = profile[profile["token"] == 1]
        self.assertEqual(len(dead_shape), 2)
        self.assertFalse(dead_shape["status"].eq("ok").any())
        self.assertNotIn(
            G2_PREFIX + "_atomic_sbm16", dead_shape["kernelName2"].tolist()
        )
        self.assertEqual(len(profile[profile["token"] == 2]), 8)
        self.assertEqual(len(profile[profile["token"] == 3]), 8)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])
        self.assertIn("worker", failures.iloc[0]["failure_reason"])

    def test_all_failing_candidates_keep_numeric_errors_and_untruncated_diagnostics(
        self,
    ):
        failure_reason = "compiler refused candidate; " + "full diagnostic detail " * 30

        class CandidateFailure(_controlled_tuner_class(self.module)):
            def _run_candidate(self, row, candidate, args):
                if row["token"] == 1:
                    candidate["error"] = 0.025
                    raise RuntimeError(failure_reason)
                return super()._run_candidate(row, candidate, args)

        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row(1), _input_row(2)], CandidateFailure)
        self.assertEqual(stopped.exception.code, 1)
        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["token"].tolist(), [2])
        profile = pd.read_csv(self.profile_file)
        rejected = profile[profile["token"] == 1]
        self.assertEqual(len(rejected), 8)
        self.assertTrue(rejected["error"].eq(0.025).all())
        self.assertTrue(rejected["pipeline_us"].isna().all())
        self.assertTrue(rejected["status"].eq("execution_failed").all())
        self.assertTrue(rejected["failure_reason"].str.contains(failure_reason).all())
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])
        self.assertIn(failure_reason, failures.iloc[0]["failure_reason"])

    def test_empty_candidate_shape_continues_without_a_profile_file(self):
        class EmptyCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(self, row, full_search=False):
                if row["token"] == 1:
                    return []
                return super()._candidate_rows(row, full_search)

        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row(1), _input_row(2)], EmptyCandidates, profile=False)
        self.assertEqual(stopped.exception.code, 1)
        self.assertFalse(self.profile_file.exists())
        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["token"].tolist(), [2])
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])
        self.assertIn("no legal coupled candidates", failures.iloc[0]["failure_reason"])

    def test_sigalrm_timeout_is_per_candidate_and_full_shape_can_run_longer(self):
        class SlowCandidates(_controlled_tuner_class(self.module)):
            def _run_candidate(self, row, candidate, args):
                if candidate["kernelName2"] == G2_PREFIX + "_atomic_nt_sbm16":
                    time.sleep(2)
                else:
                    time.sleep(0.2)
                return super()._run_candidate(row, candidate, args)

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            self.run_csv([_input_row()], SlowCandidates, timeout=1)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)
        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["us"].tolist(), [7.0])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 8)
        expired = profile[profile["kernelName2"] == G2_PREFIX + "_atomic_nt_sbm16"]
        self.assertEqual(expired["status"].tolist(), ["timeout"])
        self.assertTrue(expired["pipeline_us"].isna().all())
        self.assertTrue(expired["failure_reason"].str.contains("exceeded 1s").all())

    def test_resume_skips_only_existing_successful_rows(self):
        existing = []
        for token, us in ((1, float("nan")), (2, -1.0), (3, 0.0), (4, 11.0)):
            result = _input_row(token)
            result.update(dict.fromkeys(RESULT_COLUMNS, 0))
            result.update(
                block_m=16,
                us=us,
                us1=us,
                kernelName1=G1,
                kernelName2=G2_PREFIX + "_reduce_nt_sbm16",
                err1="1.0%",
                err2="1.0%",
            )
            existing.append(result)
        self.run_csv(
            [_input_row(token) for token in (1, 2, 3, 4)], existing_rows=existing
        )

        tuned = pd.read_csv(self.output_file).sort_values("token")
        self.assertEqual(tuned["token"].tolist(), [1, 2, 3, 4])
        self.assertEqual(tuned["us"].tolist(), [7.0, 7.0, 7.0, 11.0])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(set(profile["token"]), {1, 2, 3})
        self.assertEqual(len(profile), 24)

    def test_interrupted_run_persists_unfinished_shapes_and_exits_nonzero(self):
        class InterruptedCandidate(_controlled_tuner_class(self.module)):
            def _run_candidate(self, row, candidate, args):
                raise KeyboardInterrupt

        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row(1), _input_row(2)], InterruptedCandidate)
        self.assertEqual(stopped.exception.code, 1)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1, 2])
        self.assertTrue(failures["failure_reason"].str.contains("unfinished").all())

    def test_failed_retune_does_not_keep_an_old_invalid_winner(self):
        class EmptyCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(self, row, full_search=False):
                return []

        result = _input_row()
        result.update(dict.fromkeys(RESULT_COLUMNS, 0))
        result.update(kernelName1=G1, kernelName2=G2_PREFIX + "_atomic_sbm16")
        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row()], EmptyCandidates, existing_rows=[result])
        self.assertEqual(stopped.exception.code, 1)
        tuned = pd.read_csv(self.output_file)
        self.assertTrue(tuned.empty)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])

    def test_fully_covered_resume_refreshes_stale_failure_manifest(self):
        failure_file = self.output_file.with_suffix(".failed_shapes.csv")
        failed = _input_row()
        failed.update(status="failed", failure_reason="previous tuning failure")
        pd.DataFrame([failed]).to_csv(failure_file, index=False)
        successful = _input_row()
        successful.update(dict.fromkeys(RESULT_COLUMNS, 0))
        successful.update(
            us=7.0,
            us1=7.0,
            block_m=16,
            kernelName1=G1,
            kernelName2=G2_PREFIX + "_reduce_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        self.run_csv([_input_row()], existing_rows=[successful])

        failures = pd.read_csv(failure_file)
        self.assertTrue(failures.empty)
        self.assertEqual(failures.columns.tolist(), KEYS + ["status", "failure_reason"])
        self.assertEqual(pd.read_csv(self.output_file)["us"].tolist(), [7.0])
        self.assertFalse(self.profile_file.exists())

    def test_fully_covered_resume_creates_empty_failure_manifest(self):
        successful = _input_row()
        successful.update(dict.fromkeys(RESULT_COLUMNS, 0))
        successful.update(
            us=7.0,
            us1=7.0,
            block_m=16,
            kernelName1=G1,
            kernelName2=G2_PREFIX + "_reduce_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        self.run_csv([_input_row()], existing_rows=[successful])

        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertTrue(failures.empty)
        self.assertEqual(failures.columns.tolist(), KEYS + ["status", "failure_reason"])
        self.assertEqual(pd.read_csv(self.output_file)["us"].tolist(), [7.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
