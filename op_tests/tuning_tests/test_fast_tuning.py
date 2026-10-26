# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only regression tests for lightweight tuning scans."""

import csv
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# CI executes this file directly; imports and fixtures must use this checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


class TestFastScanCliSafety(unittest.TestCase):
    def test_generic_fmoe_rejects_fast_scan_before_dispatch(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.FmoeTuner.__new__(gemm_moe_tune.FmoeTuner)
        args = SimpleNamespace(fast_scan=True)
        with (
            mock.patch.object(
                gemm_moe_tune,
                "mp_tuner",
                side_effect=AssertionError("generic candidates were dispatched"),
            ),
            self.assertRaisesRegex(ValueError, "requires --mxfp4-flydsl"),
        ):
            tuner.tune(None, None, args)

    def test_generic_parser_does_not_expose_fast_scan(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.FmoeTuner("testFmoe", [], [], "test parser")

        self.assertNotIn("--fast-scan", tuner.parser._option_string_actions)

    def test_mxfp4_parser_accepts_explicit_fast_scan_controls(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner("testMxfp4", [], [], "test parser")
        argv = [
            "gemm_moe_tune.py",
            "--mxfp4-flydsl",
            "--fast-scan",
            "--fast-scan-warmup-accuracy-checks",
            "2",
            "--fast-scan-iters",
            "7",
            "--fast-scan-final-iters",
            "31",
            "--fast-scan-finalists",
            "4",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = tuner.parse_args()

        self.assertTrue(args.fast_scan)
        self.assertEqual(args.fast_scan_warmup_accuracy_checks, 2)
        self.assertEqual(args.fast_scan_iters, 7)
        self.assertEqual(args.fast_scan_final_iters, 31)
        self.assertEqual(args.fast_scan_finalists, 4)

        for option in ("--fast-scan-iters", "--fast-scan-final-iters"):
            with (
                self.subTest(option=option),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "gemm_moe_tune.py",
                        "--mxfp4-flydsl",
                        "--fast-scan",
                        option,
                        "1",
                    ],
                ),
                self.assertRaises(SystemExit),
            ):
                tuner.parse_args()


class TestMxfp4AccuracyFunnel(unittest.TestCase):
    @staticmethod
    def _candidate(name):
        return {"kernelName1": f"g1_{name}", "kernelName2": f"g2_{name}", "us": 0}

    def test_performance_finalists_keep_global_fastest(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        candidates = [self._candidate(f"candidate_{i}") for i in range(6)]
        for index, candidate in enumerate(candidates):
            candidate.update(us=100.0 + index * 0.2, err1=0.01 * (6 - index))

        finalists = gemm_moe_tune._select_mxfp4_performance_finalists(
            candidates, finalist_count=5
        )

        self.assertEqual(
            [candidate["us"] for candidate in finalists],
            [100.0, 100.2, 100.4, 100.6, 100.8],
        )
        self.assertIs(finalists[0], candidates[0])

    def test_final_global_gate_prefers_accuracy_without_chaining(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fastest = self._candidate("fastest")
        fastest.update(us=100.0, err1=0.03)
        accurate_tie = self._candidate("accurate_tie")
        accurate_tie.update(us=100.9, err1=0.01)
        chained_only = self._candidate("chained_only")
        chained_only.update(us=101.8, err1=0.001)

        selected = gemm_moe_tune._select_mxfp4_finalist(
            [chained_only, fastest, accurate_tie]
        )

        self.assertIs(selected, accurate_tie)

    def test_final_global_gate_includes_exactly_one_percent(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fastest = self._candidate("fastest")
        fastest.update(us=100.0, err1=0.03)
        accurate_boundary = self._candidate("accurate_boundary")
        accurate_boundary.update(us=101.0, err1=0.01)

        selected = gemm_moe_tune._select_mxfp4_finalist([fastest, accurate_boundary])

        self.assertIs(selected, accurate_boundary)

    def test_accuracy_promotion_selects_best_candidate_outside_speed_set(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        fast = self._candidate("fast")
        fast.update(us=100.0, err1=0.03)
        speed_finalist = self._candidate("speed_finalist")
        speed_finalist.update(us=100.5, err1=0.02)
        accuracy_candidate = self._candidate("accuracy_candidate")
        accuracy_candidate.update(us=102.0, err1=0.001)

        promoted = gemm_moe_tune._select_mxfp4_accuracy_promotion(
            [fast, speed_finalist, accuracy_candidate],
            [fast, speed_finalist],
        )

        self.assertIs(promoted, accuracy_candidate)

    def test_accuracy_promotion_skips_accurate_speed_finalist(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        globally_accurate = self._candidate("globally_accurate")
        globally_accurate.update(us=100.0, err1=0.001)
        next_accurate = self._candidate("next_accurate")
        next_accurate.update(us=102.0, err1=0.002)
        less_accurate = self._candidate("less_accurate")
        less_accurate.update(us=103.0, err1=0.01)

        promoted = gemm_moe_tune._select_mxfp4_accuracy_promotion(
            [globally_accurate, next_accurate, less_accurate],
            [globally_accurate],
        )

        self.assertIs(promoted, next_accurate)

    def test_inaccurate_candidate_is_rejected_before_timing(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        row = {
            "expert": 8,
            "model_dim": 256,
            "inter_dim": 256,
            "token": 1,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        args = SimpleNamespace(fast_scan=True, errRatio=0.1, warmup=1, iters=1)
        with (
            mock.patch.object(tuner, "_port_e2e", return_value="output"),
            mock.patch.object(gemm_moe_tune, "cosine_diff_compare", return_value=0.2),
            mock.patch(
                "aiter.test_common.run_perftest",
                side_effect=AssertionError("inaccurate candidate was timed"),
            ),
            self.assertRaisesRegex(RuntimeError, "cosine err_ratio"),
        ):
            tuner._run_candidate(
                row,
                self._candidate("bad"),
                args,
                data="fixture",
                reference="reference",
            )

    def test_finalists_are_revalidated_and_failures_are_discarded(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        fast = self._candidate("fast_bad")
        accurate = self._candidate("accurate")
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        calls = []

        def run_candidate(_row, candidate, _args, **kwargs):
            calls.append((candidate["kernelName1"], kwargs))
            if kwargs.get("num_iters") == 10:
                candidate["us"] = 1.0 if "fast_bad" in candidate["kernelName1"] else 2.0
                return candidate["us"]
            if "fast_bad" in candidate["kernelName1"]:
                raise gemm_moe_tune.Mxfp4AccuracyError("finalist accuracy regression")
            candidate["us"] = 2.0
            return candidate["us"]

        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=[fast, accurate]),
            mock.patch.object(
                tuner, "_precompile_candidates", return_value=(2, 0)
            ) as precompile,
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            best = tuner._tune_one_shape(row, args)

        self.assertEqual(best["kernelName1"], "g1_accurate")
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(kwargs["reference"] == "reference" for _, kwargs in calls))
        self.assertEqual([call[1]["num_warmup"] for call in calls], [1, 1, 0, 0])
        self.assertEqual([call[1]["num_iters"] for call in calls], [10, 10, 100, 100])
        precompile.assert_called_once()

    def test_finalist_failures_backfill_from_coarse_survivors(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        candidates = [self._candidate(f"candidate_{i}") for i in range(1, 5)]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        calls = []

        def run_candidate(_row, candidate, _args, **kwargs):
            name = candidate["kernelName1"]
            calls.append((name, kwargs["num_iters"]))
            if kwargs["num_iters"] == 10:
                order = {
                    "g1_candidate_1": (1.0, 0.01),
                    "g1_candidate_2": (2.0, 0.02),
                    "g1_candidate_3": (3.0, 0.001),
                    "g1_candidate_4": (4.0, 0.002),
                }
                candidate["us"], candidate["err1"] = order[name]
                return candidate["us"]
            if name in {"g1_candidate_1", "g1_candidate_2", "g1_candidate_3"}:
                raise gemm_moe_tune.Mxfp4AccuracyError("final validation failed")
            candidate["us"] = 4.0
            return candidate["us"]

        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=candidates),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(4, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            best = tuner._tune_one_shape(row, args)

        self.assertEqual(best["kernelName1"], "g1_candidate_4")
        self.assertEqual(
            [name for name, iters in calls if iters == 100],
            [
                "g1_candidate_1",
                "g1_candidate_2",
                "g1_candidate_3",
                "g1_candidate_4",
            ],
        )

    def test_backfill_is_latency_ordered_after_accuracy_promotion(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        candidates = [self._candidate(f"candidate_{i}") for i in range(10)]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=5,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        final_calls = []

        def run_candidate(_row, candidate, _args, **kwargs):
            name = candidate["kernelName1"]
            index = int(name.rsplit("_", 1)[1])
            if kwargs["num_iters"] == 10:
                candidate["us"] = 100.0 + index * 0.1
                candidate["err1"] = 0.001 if index == 9 else 0.01 + index * 1e-4
                return candidate["us"]

            final_calls.append(name)
            if index in {*range(5), 9}:
                raise gemm_moe_tune.Mxfp4AccuracyError("final validation failed")
            return candidate["us"]

        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=candidates),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(10, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            tuner._tune_one_shape(row, args)

        self.assertEqual(
            final_calls,
            [
                "g1_candidate_0",
                "g1_candidate_1",
                "g1_candidate_2",
                "g1_candidate_3",
                "g1_candidate_4",
                "g1_candidate_9",
                "g1_candidate_5",
                "g1_candidate_6",
                "g1_candidate_7",
                "g1_candidate_8",
            ],
        )

    def test_warmup_outputs_are_accuracy_checks_outside_timing(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        # _run_candidate finishes by building the tuned row through calculate(),
        # which needs the real 14-column key layout.
        tuner = gemm_moe_tune.Mxfp4FlydslTuner(
            "testMxfp4",
            [
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
            ],
            [],
            "test parser",
        )
        args = SimpleNamespace(fast_scan=True, errRatio=0.1, warmup=1, iters=10)
        row = {
            "gfx": "gfx950",
            "cu_num": 256,
            "expert": 8,
            "model_dim": 256,
            "inter_dim": 256,
            "token": 1,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
        }

        def run_perftest(operation, **kwargs):
            self.assertEqual(kwargs["num_warmup"], 0)
            self.assertEqual(kwargs["num_iters"], 10)
            self.assertFalse(kwargs.get("use_cuda_event", False))
            self.assertEqual(kwargs["profile_warm_iters"], 0)
            self.assertEqual(kwargs["num_rotate_args"], 1)
            return operation(), 7.0

        with (
            mock.patch.object(
                tuner, "_port_e2e", side_effect=["warm", "timed"]
            ) as port,
            mock.patch.object(
                gemm_moe_tune,
                "cosine_diff_compare",
                side_effect=[0.01, 0.02],
            ) as compare,
            mock.patch("aiter.test_common.run_perftest", side_effect=run_perftest),
        ):
            candidate = self._candidate("accurate")
            candidate.update(err1=0.03, err2=0.025, block_m=16)
            tuner._run_candidate(
                row,
                candidate,
                args,
                data="fixture",
                reference="reference",
                num_warmup=1,
                num_iters=10,
            )

        self.assertEqual(port.call_count, 2)
        self.assertEqual(compare.call_count, 2)
        # err1/err2 keep the upstream percent-string display format; the raw
        # ratio used by the funnel lives in the private _err key.
        self.assertEqual(candidate["err1"], "3.0%")
        self.assertEqual(candidate["err2"], "3.0%")
        self.assertEqual(candidate["_err"], 0.03)

    def test_shared_reference_uses_tuned_swiglu_limit(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=1,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        limits = []

        def torch_ref(_data, _topk, _dtype, _activation, swiglu_limit=None):
            limits.append(swiglu_limit)
            return "reference"

        def run_candidate(_row, candidate, _args, **kwargs):
            candidate["us"] = 1.0
            return 1.0

        with (
            mock.patch.dict(os.environ, {"AITER_MXFP4_TUNE_SWIGLU_LIMIT": "3.5"}),
            mock.patch.object(
                tuner, "_candidate_rows", return_value=[self._candidate("accurate")]
            ),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(1, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", side_effect=torch_ref),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            tuner._tune_one_shape(row, args)

        # The shared fast-scan reference must apply the same clamp that every
        # candidate's kernel launch applies, or setting the tuning limit rejects
        # all candidates against an unclamped reference.
        self.assertEqual(limits, [3.5])

    def test_candidate_kernel_and_reference_share_the_tuned_swiglu_limit(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        tuner.keys = [
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
            "act_type",
        ]
        args = SimpleNamespace(fast_scan=False, errRatio=0.1, warmup=1, iters=10)
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        limits = []

        def port_e2e(*_args, **kwargs):
            limits.append(kwargs.get("swiglu_limit"))
            return "out"

        with (
            mock.patch.dict(os.environ, {"AITER_MXFP4_TUNE_SWIGLU_LIMIT": "3.5"}),
            mock.patch.object(tuner, "_port_e2e", side_effect=port_e2e),
            mock.patch.object(tuner, "_torch_ref", return_value="ref") as torch_ref,
            mock.patch.object(gemm_moe_tune, "cosine_diff_compare", return_value=0.01),
            mock.patch.object(tuner, "calculate", return_value=(0.0, 0.0)),
            mock.patch(
                "aiter.test_common.run_perftest",
                side_effect=lambda operation, **kwargs: (operation(), 5.0),
            ),
        ):
            candidate = self._candidate("accurate")
            candidate.update(block_m=16)
            tuner._run_candidate(row, candidate, args, data="fixture")

        # Kernel launch and reference must clamp identically, otherwise the
        # accuracy gate compares two different functions.
        self.assertEqual(limits, [3.5, 3.5])
        self.assertEqual(torch_ref.call_args.kwargs["swiglu_limit"], 3.5)

    def test_finalist_retime_preserves_unrounded_coarse_error(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        tuner.keys = [
            "token",
            "model_dim",
            "inter_dim",
            "expert",
            "topk",
            "act_type",
        ]
        args = SimpleNamespace(fast_scan=False, errRatio=0.1, warmup=1, iters=10)
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        candidate = self._candidate("fastest")
        candidate.update(
            block_m=16,
            err1="1.1%",
            err2="1.1%",
            _err=0.01149,
        )
        with (
            mock.patch.dict(os.environ, {"AITER_MXFP4_TUNE_SWIGLU_LIMIT": "3.5"}),
            mock.patch.object(tuner, "_port_e2e", return_value="out"),
            mock.patch.object(tuner, "_torch_ref", return_value="ref"),
            mock.patch.object(gemm_moe_tune, "cosine_diff_compare", return_value=0.010),
            mock.patch.object(tuner, "calculate", return_value=(0.0, 0.0)),
            mock.patch(
                "aiter.test_common.run_perftest",
                side_effect=lambda operation, **kwargs: (operation(), 5.0),
            ),
        ):
            tuner._run_candidate(row, candidate, args, data="fixture")

        # The retime measured 0.010, but the coarse pass already saw 0.01149.
        # Keeping the raw value stops a 1.1% display string (0.011) from
        # rounding the candidate below a slower one with a true 0.01120.
        self.assertEqual(candidate["_err"], 0.01149)
        self.assertEqual(candidate["err1"], "1.1%")
        self.assertEqual(candidate["err2"], "1.1%")

    def test_accuracy_failure_rejects_only_the_candidate(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        candidates = [self._candidate("bad"), self._candidate("good")]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=candidates),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(2, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(tuner, "_run_candidate") as run_candidate,
            mock.patch("builtins.print"),
        ):

            def candidate_result(_row, candidate, _args, **kwargs):
                if "bad" in candidate["kernelName1"]:
                    raise gemm_moe_tune.Mxfp4AccuracyError("warmup failed")
                candidate.update(us=2.0, err1=0.01, err2=0.01)
                return 2.0

            run_candidate.side_effect = candidate_result
            result = tuner._tune_one_shape(row, args)

        self.assertEqual(run_candidate.call_count, 3)
        self.assertEqual(result["kernelName1"], "g1_good")
        self.assertEqual(result["us"], 2.0)

    def test_shape_fails_only_when_no_accurate_candidate_survives(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        tuner.keys = ["token", "model_dim", "inter_dim", "expert", "topk"]
        candidates = [self._candidate("bad-1"), self._candidate("bad-2")]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=2,
            timeout=0,
            errRatio=0.1,
            warmup=2,
            iters=101,
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
        }
        with (
            mock.patch.object(tuner, "_candidate_rows", return_value=candidates),
            mock.patch.object(tuner, "_precompile_candidates", return_value=(2, 0)),
            mock.patch.object(tuner, "_prepare_case", return_value="fixture"),
            mock.patch.object(tuner, "_torch_ref", return_value="reference"),
            mock.patch.object(
                tuner,
                "_run_candidate",
                side_effect=gemm_moe_tune.Mxfp4AccuracyError("warmup failed"),
            ) as run_candidate,
            mock.patch("builtins.print"),
        ):
            result = tuner._tune_one_shape(row, args)

        self.assertEqual(run_candidate.call_count, 2)
        self.assertEqual(result["us"], tuner.INVALID_TIME)
        self.assertTrue(result["kernelName1"].startswith("FAILED:"))


class TestGlm53TunedArtifact(unittest.TestCase):
    def test_token_one_row_matches_validated_coupled_winner(self):
        config = REPO_ROOT / "aiter/configs/model_configs/glm53_fp4_tuned_fmoe.csv"
        with config.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        token_one = next(row for row in rows if row["token"] == "1")

        self.assertEqual(len(rows), 16)
        self.assertEqual(
            token_one["kernelName1"],
            "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
        )
        self.assertEqual(
            token_one["kernelName2"],
            "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_nt_sbm16",
        )
        self.assertGreater(float(token_one["us"]), 0.0)
        self.assertEqual(float(token_one["us1"]), float(token_one["us"]))
        self.assertEqual(float(token_one["us2"]), 0.0)
        self.assertGreater(float(token_one["err1"]), 0.0)
        self.assertGreater(float(token_one["err2"]), 0.0)

    def test_in_memory_jobs_match_file_parser(self):
        from aiter.aot.flydsl.mxfp4_moe import _job_key, jobs_from_rows, parse_csv

        config = REPO_ROOT / "aiter/configs/model_configs/glm53_fp4_tuned_fmoe.csv"
        with config.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

        file_jobs = parse_csv(str(config))
        row_jobs = jobs_from_rows(rows)

        # The tuner feeds candidate rows through jobs_from_rows instead of a CSV
        # file, so the in-memory path must select exactly the jobs the shipped
        # file parser selects, stage-2 bias variants included.
        self.assertEqual(
            [_job_key(job) for job in row_jobs],
            [_job_key(job) for job in file_jobs],
        )
        no_bias_jobs = jobs_from_rows(rows, include_bias_variants=False)
        self.assertLess(len(no_bias_jobs), len(file_jobs))
        self.assertLessEqual(
            {_job_key(job) for job in no_bias_jobs},
            {_job_key(job) for job in file_jobs},
        )


class TestParallelMxfp4Precompile(unittest.TestCase):
    def test_uses_forkserver_pool_and_reports_failed_cache_jobs(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        row = {"token": 1, "model_dim": 256}
        candidates = [{"kernelName1": "g1", "kernelName2": "g2"}]
        jobs = [
            {"kernel_name": "g1"},
            {"kernel_name": "g2"},
            {"kernel_name": "g3"},
        ]
        results = [
            {
                "kernel_name": "batch 1",
                "compile_time": 1.0,
                "results": [
                    {"kernel_name": "g1", "compile_time": 0.5},
                    {"kernel_name": "g3", "compile_time": None},
                ],
            },
            {
                "kernel_name": "batch 2",
                "compile_time": 0.5,
                "results": [{"kernel_name": "g2", "compile_time": 0.5}],
            },
        ]
        with (
            mock.patch(
                "aiter.aot.flydsl.mxfp4_moe.jobs_from_rows", return_value=jobs
            ) as jobs_from_rows,
            mock.patch(
                "aiter.aot.flydsl.common.run_jobs_parallel", return_value=results
            ) as run_jobs_parallel,
            mock.patch("aiter.aot.flydsl.common.get_max_workers", return_value=2),
            mock.patch("builtins.print"),
        ):
            total, failed = tuner._precompile_candidates(row, candidates)

        self.assertEqual((total, failed), (3, 1))
        jobs_from_rows.assert_called_once_with(
            [{"token": 1, "model_dim": 256, **candidates[0]}],
            include_bias_variants=False,
        )
        self.assertEqual(
            run_jobs_parallel.call_args.args[1],
            [{"jobs": [jobs[0], jobs[2]]}, {"jobs": [jobs[1]]}],
        )
        self.assertEqual(
            run_jobs_parallel.call_args.kwargs["start_method"], "forkserver"
        )

    def test_precompile_failure_preserves_runtime_jit_fallback(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        with (
            mock.patch(
                "aiter.aot.flydsl.mxfp4_moe.jobs_from_rows",
                side_effect=RuntimeError("compiler unavailable"),
            ),
            mock.patch("builtins.print") as output,
        ):
            result = tuner._precompile_candidates({}, [{}])

        self.assertEqual(result, (0, 0))
        self.assertIn("falling back to runtime JIT", output.call_args.args[0])

    def test_flydsl_aot_pool_honors_explicit_worker_budget(self):
        from aiter.aot.flydsl import common

        with mock.patch.dict(os.environ, {"AITER_MAX_JOBS": "7"}, clear=False):
            self.assertEqual(common.get_max_workers(23), 7)

    def test_precompile_applies_tuning_swiglu_limit_to_stage1_jobs(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)
        jobs = [
            {"stage": 1, "act": "swiglu", "swiglu_limit": 7.0},
            {"stage": 1, "act": "silu", "swiglu_limit": float("inf")},
            {"stage": 2},
        ]
        with (
            mock.patch.dict(
                os.environ,
                {"AITER_MXFP4_TUNE_SWIGLU_LIMIT": "3.5"},
                clear=False,
            ),
            mock.patch("aiter.aot.flydsl.mxfp4_moe.jobs_from_rows", return_value=jobs),
            mock.patch(
                "aiter.aot.flydsl.common.run_jobs_parallel",
                return_value=[{"results": []}],
            ) as run_jobs_parallel,
            mock.patch("aiter.aot.flydsl.common.get_max_workers", return_value=1),
            mock.patch("builtins.print"),
        ):
            tuner._precompile_candidates({}, [{}])

        # The scan JITs on a cache miss, so the precompiled closure scalar must
        # match the runtime launch limit instead of jobs_from_rows' defaults.
        precompiled = run_jobs_parallel.call_args.args[1][0]["jobs"]
        self.assertEqual(precompiled[0]["swiglu_limit"], 3.5)
        self.assertEqual(precompiled[1]["swiglu_limit"], 3.5)
        self.assertNotIn("swiglu_limit", precompiled[2])


if __name__ == "__main__":
    unittest.main()
