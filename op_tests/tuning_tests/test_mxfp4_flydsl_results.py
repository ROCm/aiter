# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks of coupled tuning from CSV input to persisted results.

The GPU candidate execution boundary supplies controlled observations. The
candidate enumeration, winner selection, batching and CSV writers stay real.
"""

from __future__ import annotations

import argparse
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
from typing import Any
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


def _tuner_module() -> Any:
    module = importlib.import_module("csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune")
    torch.set_default_device("cpu")
    return module


def _controlled_tuner_class(module: Any) -> type:
    class ControlledTuner(module.Mxfp4FlydslTuner):
        def _candidate_rows(
            self, row: dict[str, Any] | pd.Series, full_search: bool = False
        ) -> list[dict[str, Any]]:
            return [
                candidate
                for candidate in super()._candidate_rows(row, full_search=full_search)
                if candidate["kernelName1"] == G1
                and candidate["kernelName2"] in OBSERVATIONS
            ]

        def _run_candidate(
            self,
            row: dict[str, Any],
            candidate: dict[str, Any],
            args: argparse.Namespace,
        ) -> float:
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


def _input_row(token: int = 1) -> dict[str, Any]:
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


def _cpu_shape_process(
    payload: tuple[list[str], dict[str, Any], argparse.Namespace, int],
    out_q: Any,
    index: int,
) -> None:
    """Use the real isolated-worker aggregation without launching GPU work."""
    keys, row, args, _gpu = payload
    module = _tuner_module()

    class WorkerTuner(_controlled_tuner_class(module)):
        def _run_candidate(
            self,
            row: dict[str, Any],
            candidate: dict[str, Any],
            args: argparse.Namespace,
        ) -> float:
            if row["token"] == 1 and candidate["kernelName2"].endswith("_atomic_sbm16"):
                os._exit(17)
            return super()._run_candidate(row, candidate, args)

    tuner = WorkerTuner.__new__(WorkerTuner)
    tuner.keys = keys
    out_q.put((index, tuner._tune_one_shape(row, args)))


class TestCoupledTuningResults(unittest.TestCase):
    def setUp(self) -> None:
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
        # CPU infrastructure tests replace the HIP capability boundary as
        # well as candidate execution; real aux build/dispatch has GPU coverage.
        from aiter.ops import moe_mxfp4_aux

        self.stack.enter_context(
            patch.object(moe_mxfp4_aux, "prepare_mxfp4_moe_aux", return_value=None)
        )

    def run_csv(
        self,
        rows: list[dict[str, Any]],
        tuner_class: Any = None,
        profile: bool = True,
        mp: int = 1,
        batch: int = 2,
        timeout: int = 0,
        existing_rows: Any = None,
    ) -> None:
        pd.DataFrame(rows, columns=KEYS).to_csv(self.input_file, index=False)
        if existing_rows is not None:
            old_results = pd.DataFrame(existing_rows)
            columns = (
                KEYS
                + RESULT_COLUMNS
                + [
                    column
                    for column in old_results
                    if column not in KEYS + RESULT_COLUMNS
                ]
            )
            old_results.reindex(columns=columns).to_csv(self.output_file, index=False)
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

    def test_csv_saves_fastest_valid_winner_and_all_candidate_observations(
        self,
    ) -> None:
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
    ) -> None:
        failure_reason = "candidate catalog unavailable; " + "diagnostic context " * 30

        class ShapeFailure(_controlled_tuner_class(self.module)):
            def _candidate_rows(
                self, row: dict[str, Any] | pd.Series, full_search: bool = False
            ) -> list[dict[str, Any]]:
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

    def test_dead_worker_preserves_completed_profiles_and_other_shapes_finish(
        self,
    ) -> None:
        run_isolated = self.module._run_shapes_isolated

        def cpu_workers(
            payloads: list[tuple], mp_num: int, _ctx: Any
        ) -> list[dict[str, Any] | None]:
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
    ) -> None:
        failure_reason = "compiler refused candidate; " + "full diagnostic detail " * 30

        class CandidateFailure(_controlled_tuner_class(self.module)):
            def _run_candidate(
                self,
                row: dict[str, Any],
                candidate: dict[str, Any],
                args: argparse.Namespace,
            ) -> float:
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

    def test_empty_candidate_shape_continues_without_a_profile_file(self) -> None:
        class EmptyCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(
                self, row: dict[str, Any] | pd.Series, full_search: bool = False
            ) -> list[dict[str, Any]]:
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

    def test_sigalrm_timeout_is_per_candidate_and_full_shape_can_run_longer(
        self,
    ) -> None:
        class SlowCandidates(_controlled_tuner_class(self.module)):
            def _run_candidate(
                self,
                row: dict[str, Any],
                candidate: dict[str, Any],
                args: argparse.Namespace,
            ) -> float:
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

    def test_resume_skips_only_existing_successful_rows(self) -> None:
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

    def test_interrupted_run_persists_unfinished_shapes_and_exits_nonzero(self) -> None:
        class InterruptedCandidate(_controlled_tuner_class(self.module)):
            def _run_candidate(
                self,
                row: dict[str, Any],
                candidate: dict[str, Any],
                args: argparse.Namespace,
            ) -> float:
                raise KeyboardInterrupt

        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row(1), _input_row(2)], InterruptedCandidate)
        self.assertEqual(stopped.exception.code, 1)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1, 2])
        self.assertTrue(failures["failure_reason"].str.contains("unfinished").all())

    def test_failed_retune_does_not_keep_an_old_invalid_winner(self) -> None:
        class EmptyCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(
                self, row: dict[str, Any] | pd.Series, full_search: bool = False
            ) -> list[dict[str, Any]]:
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

    def test_fully_covered_resume_refreshes_stale_failure_manifest(self) -> None:
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

    def test_fully_covered_resume_creates_empty_failure_manifest(self) -> None:
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

    def test_accepted_aliases_save_canonical_runtime_keys_and_deduplicate(self) -> None:
        aliases = _input_row()
        aliases.update(
            dtype="bf16",
            q_dtype_a="fp4x2",
            q_dtype_w="fp4x2",
            q_type="per_1x32",
            act_type="silu",
            use_g1u1="true",
            doweight_stage1="false",
        )
        self.run_csv([_input_row(), aliases])

        tuned = pd.read_csv(self.output_file)
        self.assertEqual(len(tuned), 1)
        self.assertEqual(tuned[KEYS].to_dict("records"), [_input_row()])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 8)
        self.assertEqual(
            profile[KEYS].drop_duplicates().to_dict("records"), [_input_row()]
        )

    def test_resume_rechecks_recorded_accuracy_failures_and_keeps_valid_legacy_rows(
        self,
    ) -> None:
        existing = []
        for token, error in ((1, "90.0%"), (2, "nan%"), (3, "inf"), (4, "1.0%")):
            row = _input_row(token)
            row.update(dict.fromkeys(RESULT_COLUMNS, 0))
            row.update(
                us=11.0,
                us1=11.0,
                block_m=16,
                kernelName1=G1,
                kernelName2=G2_PREFIX + "_reduce_nt_sbm16",
                err1=error,
                err2=error,
            )
            existing.append(row)
        self.run_csv(
            [_input_row(token) for token in (1, 2, 3, 4)],
            existing_rows=existing,
        )

        tuned = pd.read_csv(self.output_file).sort_values("token")
        self.assertEqual(tuned["us"].tolist(), [7.0, 7.0, 7.0, 11.0])
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(set(profile["token"]), {1, 2, 3})
        self.assertEqual(len(profile), 24)

    def test_failed_retune_removes_old_failed_status_even_with_good_metrics(
        self,
    ) -> None:
        class EmptyCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(
                self, row: dict[str, Any] | pd.Series, full_search: bool = False
            ) -> list[dict[str, Any]]:
                return []

        failed = _input_row()
        failed.update(dict.fromkeys(RESULT_COLUMNS, 0))
        failed.update(
            us=7.0,
            us1=7.0,
            block_m=16,
            kernelName1=G1,
            kernelName2=G2_PREFIX + "_reduce_sbm16",
            err1="1.0%",
            err2="1.0%",
            status="execution_failed",
        )
        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([_input_row()], EmptyCandidates, existing_rows=[failed])
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(pd.read_csv(self.output_file).empty)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["token"].tolist(), [1])

    def test_nonfinite_timed_output_is_failed_with_raw_error_and_no_winner(
        self,
    ) -> None:
        class RealCandidate(_controlled_tuner_class(self.module)):
            _run_candidate = self.module.Mxfp4FlydslTuner._run_candidate

        reference = torch.ones((1, 512), device="cpu")
        timed_output = torch.full((1, 512), float("nan"), device="cpu")
        test_common = importlib.import_module("aiter.test_common")
        with (
            patch.object(RealCandidate, "_prepare_case", return_value={}),
            patch.object(RealCandidate, "_port_e2e", return_value=reference),
            patch.object(RealCandidate, "_torch_ref", return_value=reference),
            patch.object(test_common, "run_perftest", return_value=(timed_output, 7.0)),
            self.assertRaises(SystemExit) as stopped,
        ):
            self.run_csv([_input_row()], RealCandidate)
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(pd.read_csv(self.output_file).empty)
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 8)
        self.assertTrue(profile["pipeline_us"].eq(7.0).all())
        self.assertTrue(profile["error"].isna().all())
        self.assertTrue(profile["status"].eq("accuracy_failed").all())
        self.assertTrue(profile["failure_reason"].str.len().gt(0).all())

    def test_finite_wrong_timed_output_saves_amplitude_error_and_rejects_pair(
        self,
    ) -> None:
        class RealCandidate(_controlled_tuner_class(self.module)):
            _run_candidate = self.module.Mxfp4FlydslTuner._run_candidate

        reference = torch.ones((1, 512), device="cpu")
        test_common = importlib.import_module("aiter.test_common")
        with (
            patch.object(RealCandidate, "_prepare_case", return_value={}),
            patch.object(RealCandidate, "_port_e2e", return_value=reference),
            patch.object(RealCandidate, "_torch_ref", return_value=reference),
            patch.object(
                test_common, "run_perftest", return_value=(reference * 2, 7.0)
            ),
            self.assertRaises(SystemExit) as stopped,
        ):
            self.run_csv([_input_row()], RealCandidate)
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(pd.read_csv(self.output_file).empty)
        profile = pd.read_csv(self.profile_file)
        self.assertEqual(len(profile), 8)
        for error in profile["error"]:
            self.assertAlmostEqual(error, 0.2)
        self.assertTrue(profile["status"].eq("accuracy_failed").all())

    def test_explicit_run_config_selects_tuned_shapes_before_coupled_validation(
        self,
    ) -> None:
        unrelated = _input_row()
        unrelated.update(
            q_dtype_a="torch.bfloat16",
            q_dtype_w="torch.bfloat16",
            q_type="QuantType.No",
        )
        pd.DataFrame([unrelated], columns=KEYS).to_csv(self.input_file, index=False)
        winner = _input_row(2)
        winner.update(dict.fromkeys(RESULT_COLUMNS, 0))
        winner.update(
            us=7.0,
            us1=7.0,
            block_m=16,
            kernelName1=G1,
            kernelName2=G2_PREFIX + "_reduce_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        pd.DataFrame([winner]).to_csv(self.output_file, index=False)

        class BenchmarkBoundary(self.module.Mxfp4FlydslTuner):
            def run_config(self, args: argparse.Namespace) -> list[dict[str, Any]]:
                return [
                    {"shape": str(row["token"]), "us": 1.0, "status": "ok"}
                    for _, row in self.untunedf.iterrows()
                ]

        tuner = BenchmarkBoundary("test", KEYS, RESULT_COLUMNS)
        args = tuner.parser.parse_args(
            [
                "--mxfp4-flydsl",
                "-i",
                str(self.input_file),
                "-o",
                str(self.output_file),
                "--run_config",
                str(self.output_file),
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tuner.run(args)
        self.assertEqual(tuner.untunedf["token"].tolist(), [2])
        self.assertFalse(self.output_file.with_suffix(".failed_shapes.csv").exists())
        self.assertEqual(pd.read_csv(self.output_file)["us"].tolist(), [7.0])

    def test_resume_rechecks_a8_row_whose_recorded_pair_executes_a4(self) -> None:
        class A8CandidateBoundary(self.module.Mxfp4FlydslTuner):
            def _run_candidate(
                self,
                row: dict[str, Any],
                candidate: dict[str, Any],
                args: argparse.Namespace,
            ) -> float:
                candidate.update(us=7.0, us1=7.0, error=0.01, err1="1.0%", err2="1.0%")
                return 7.0

        requested = _input_row()
        requested["q_dtype_a"] = "torch.float8_e4m3fn"
        stale = requested.copy()
        stale.update(dict.fromkeys(RESULT_COLUMNS, 0))
        stale.update(
            us=11.0,
            us1=11.0,
            block_m=16,
            kernelName1=G1,
            kernelName2=G2_PREFIX + "_reduce_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        self.run_csv([requested], A8CandidateBoundary, existing_rows=[stale])

        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["us"].tolist(), [7.0])
        self.assertTrue(
            tuned.iloc[0]["kernelName1"].startswith("flydsl_mxmoe_g1_a8w4_")
        )
        self.assertIn("_fp8out", tuned.iloc[0]["kernelName1"])
        self.assertTrue(
            tuned.iloc[0]["kernelName2"].startswith("flydsl_moe2_layout_afp8_")
        )
        profile = pd.read_csv(self.profile_file)
        self.assertFalse(profile.empty)
        self.assertEqual(set(profile["precision"]), {"A8W4"})

    def test_retune_discards_saved_pair_with_mismatched_activation_dtype_or_blocks(
        self,
    ) -> None:
        class NoCandidates(_controlled_tuner_class(self.module)):
            def _candidate_rows(
                self, row: dict[str, Any] | pd.Series, full_search: bool = False
            ) -> list[dict[str, Any]]:
                return []

        for change in (
            {"kernelName1": G1 + "_swiglu"},
            {"kernelName2": G2_PREFIX.replace("afp4", "afp8") + "_reduce_sbm16"},
            {"kernelName2": G2_PREFIX + "_reduce_sbm32"},
            {"block_m": 32},
            {"kernelName1": float("nan")},
            {"kernelName2": float("nan")},
        ):
            saved = _input_row()
            saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
            saved.update(
                us=7.0,
                us1=7.0,
                block_m=16,
                kernelName1=G1,
                kernelName2=G2_PREFIX + "_reduce_sbm16",
                err1="1.0%",
                err2="1.0%",
            )
            saved.update(change)
            with self.subTest(change=change):
                with self.assertRaises(SystemExit) as stopped:
                    self.run_csv([_input_row()], NoCandidates, existing_rows=[saved])
                self.assertEqual(stopped.exception.code, 1)
                self.assertTrue(pd.read_csv(self.output_file).empty)

    def test_retune_discards_saved_g1_that_shared_shape_support_rejects(self) -> None:
        class NoCandidates(self.module.Mxfp4FlydslTuner):
            def _candidate_rows(
                self, row: dict[str, Any], full_search: bool | None = None
            ) -> list[dict[str, Any]]:
                return []

        requested = _input_row()
        requested.update(q_dtype_a="torch.float8_e4m3fn", inter_dim=384)
        saved = requested.copy()
        saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
        saved.update(
            us=11.0,
            us1=11.0,
            block_m=16,
            kernelName1="flydsl_mxmoe_g1_a8w4_16x128x256_nt_fp8out",
            kernelName2="flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_nt_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([requested], NoCandidates, existing_rows=[saved])
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(pd.read_csv(self.output_file).empty)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["inter_dim"].tolist(), [384])
        self.assertTrue(failures["failure_reason"].str.contains("no legal").all())

    def test_resume_keeps_compatible_legacy_native_a4_pair(self) -> None:
        saved = _input_row()
        saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
        saved.update(
            us=7.0,
            us1=7.0,
            block_m=16,
            kernelName1=G1,
            kernelName2="flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt",
            err1="1.0%",
            err2="1.0%",
        )
        self.run_csv([_input_row()], existing_rows=[saved])
        self.assertEqual(
            pd.read_csv(self.output_file)["kernelName2"].tolist(),
            [saved["kernelName2"]],
        )
        self.assertFalse(self.profile_file.exists())
        self.assertTrue(
            pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv")).empty
        )

    def test_resume_retunes_saved_g2_with_unsupported_contraction_tile(self) -> None:
        class A8CandidateBoundary(self.module.Mxfp4FlydslTuner):
            def _run_candidate(
                self,
                row: dict[str, Any],
                candidate: dict[str, Any],
                args: argparse.Namespace,
            ) -> float:
                candidate.update(us=7.0, us1=7.0, error=0.01, err1="1.0%", err2="1.0%")
                return 7.0

        requested = _input_row()
        requested.update(q_dtype_a="torch.float8_e4m3fn", inter_dim=384)
        saved = requested.copy()
        saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
        saved.update(
            us=11.0,
            us1=11.0,
            block_m=16,
            kernelName1="flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out",
            kernelName2="flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x256_atomic_nt_sbm16",
            err1="1.0%",
            err2="1.0%",
        )
        self.run_csv([requested], A8CandidateBoundary, existing_rows=[saved])
        tuned = pd.read_csv(self.output_file)
        self.assertEqual(tuned["us"].tolist(), [7.0])
        self.assertIn("x128_", tuned.iloc[0]["kernelName2"])
        profile = pd.read_csv(self.profile_file)
        self.assertFalse(profile.empty)
        self.assertTrue(profile["kernelName2"].str.contains("x128_").all())
        self.assertEqual(set(profile["precision"]), {"A8W4"})

    def test_retune_discards_native_a4_saved_g2_with_unsupported_contraction(
        self,
    ) -> None:
        class NoCandidates(self.module.Mxfp4FlydslTuner):
            def _candidate_rows(
                self, row: dict[str, Any], full_search: bool | None = None
            ) -> list[dict[str, Any]]:
                return []

        requested = _input_row()
        requested["inter_dim"] = 384
        saved = requested.copy()
        saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
        saved.update(
            us=11.0,
            us1=11.0,
            block_m=16,
            kernelName1=G1,
            kernelName2="flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt",
            err1="1.0%",
            err2="1.0%",
        )
        with self.assertRaises(SystemExit) as stopped:
            self.run_csv([requested], NoCandidates, existing_rows=[saved])
        self.assertEqual(stopped.exception.code, 1)
        self.assertTrue(pd.read_csv(self.output_file).empty)
        failures = pd.read_csv(self.output_file.with_suffix(".failed_shapes.csv"))
        self.assertEqual(failures["inter_dim"].tolist(), [384])

    def test_retune_discards_saved_name_flags_outside_the_coupled_catalog(self) -> None:
        class NoCandidates(self.module.Mxfp4FlydslTuner):
            def _candidate_rows(
                self, row: dict[str, Any], full_search: bool | None = None
            ) -> list[dict[str, Any]]:
                return []

        requested = _input_row()
        requested["q_dtype_a"] = "torch.float8_e4m3fn"
        g1 = "flydsl_mxmoe_g1_a8w4_16x128x256_f16in_nt_fp8out"
        g2 = "flydsl_moe2_layout_afp8_wfp4_bf16_t16x128x128_atomic_nt_sbm16"
        for kernel1, kernel2 in (
            (g1 + "_sk2", g2),
            (g1 + "_bias", g2),
            (g1, g2 + "_sp1"),
            (g1, g2 + "_bf16lds"),
        ):
            saved = requested.copy()
            saved.update(dict.fromkeys(RESULT_COLUMNS, 0))
            saved.update(
                us=11.0,
                us1=11.0,
                block_m=16,
                kernelName1=kernel1,
                kernelName2=kernel2,
                err1="1.0%",
                err2="1.0%",
            )
            with self.subTest(kernel1=kernel1, kernel2=kernel2):
                with self.assertRaises(SystemExit) as stopped:
                    self.run_csv([requested], NoCandidates, existing_rows=[saved])
                self.assertEqual(stopped.exception.code, 1)
                self.assertTrue(pd.read_csv(self.output_file).empty)


if __name__ == "__main__":
    unittest.main(verbosity=2)
