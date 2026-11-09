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

    @staticmethod
    def _pipeline_data():
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        # Timing passes each pipeline input to run_perftest positionally.
        inputs = gemm_moe_tune.Mxfp4FlydslTuner._PORT_E2E_INPUTS
        return {key: f"fixture:{key}" for key in inputs}

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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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

        def run_perftest(operation, *inputs, **kwargs):
            self.assertEqual(kwargs["num_warmup"], 0)
            self.assertEqual(kwargs["num_iters"], 10)
            self.assertFalse(kwargs.get("use_cuda_event", False))
            self.assertEqual(kwargs["profile_warm_iters"], 0)
            self.assertEqual(kwargs["num_rotate_args"], 1)
            return operation(*inputs), 7.0

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
                data=self._pipeline_data(),
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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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
                side_effect=lambda operation, *inputs, **kwargs: (
                    operation(*inputs),
                    5.0,
                ),
            ),
        ):
            candidate = self._candidate("accurate")
            candidate.update(block_m=16)
            tuner._run_candidate(row, candidate, args, data=self._pipeline_data())

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
                side_effect=lambda operation, *inputs, **kwargs: (
                    operation(*inputs),
                    5.0,
                ),
            ),
        ):
            tuner._run_candidate(row, candidate, args, data=self._pipeline_data())

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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
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


class TestFusedStage1QuantizedCompare(unittest.TestCase):
    """The fused fp4 stage-1 check: codes and e8m0 scales against the reference
    quantizer."""

    def _quantized(self, x):
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_f4_quant

        codes, scales = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2)
        return codes, scales

    def _compare(self, ref, res):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        return gemm_moe_tune.quantized_output_compare(ref, res, printLog=False)

    def test_identical_output_passes(self):
        import torch

        codes, scales = self._quantized(torch.randn(64, 4, 256))
        self.assertEqual(self._compare(codes, codes.clone()), 0.0)
        self.assertEqual(self._compare(scales, scales.clone()), 0.0)

    def test_a_few_rounding_flips_pass(self):
        import torch

        codes, _ = self._quantized(torch.randn(64, 4, 256))
        flipped = codes.view(torch.uint8).clone()
        flipped.view(-1)[:50] ^= 0x1
        err = self._compare(codes, flipped.view(codes.dtype))
        self.assertGreater(err, 0.0)
        self.assertLess(err, 1.0)

    def test_systematic_bias_fails(self):
        import torch

        x = torch.randn(64, 4, 256)
        codes, scales = self._quantized(x)
        biased_codes, biased_scales = self._quantized(x * 1.03)
        self.assertEqual(self._compare(codes, biased_codes), 1.0)
        self.assertEqual(self._compare(scales, biased_scales), 1.0)

    def test_misplaced_scales_fail(self):
        import torch

        _, scales = self._quantized(torch.randn(64, 4, 256))
        flat = scales.view(torch.uint8).reshape(-1)
        permuted = flat[torch.randperm(flat.numel())].view(scales.dtype)
        self.assertEqual(self._compare(scales, permuted), 1.0)

    def test_scale_off_by_two_exponents_fails(self):
        import torch

        _, scales = self._quantized(torch.randn(64, 4, 256))
        off = scales.view(torch.uint8).clone()
        off.view(-1)[0] += 2
        self.assertEqual(self._compare(scales, off.view(scales.dtype)), 1.0)

    def test_unmapped_scale_positions_are_skipped(self):
        import torch

        from aiter import dtypes

        ref = torch.full((256,), 0xFF, dtype=torch.uint8)
        ref[:128] = 127
        res = torch.randint(0, 255, (256,), dtype=torch.uint8)
        res[:128] = 127
        self.assertEqual(
            self._compare(ref.view(dtypes.fp8_e8m0), res.view(dtypes.fp8_e8m0)), 0.0
        )

    def test_sorted_codes_skip_padding_rows(self):
        import torch

        from aiter import dtypes

        codes = torch.randint(0, 256, (1000, 64), dtype=torch.uint8)
        ref = torch.full((1000, 64), float("nan"), dtype=dtypes.fp16)
        ref[:8] = codes[:8].to(dtypes.fp16)
        wrong = codes.clone()
        wrong[:8] = torch.randint(0, 256, (8, 64), dtype=torch.uint8)
        self.assertEqual(self._compare(ref, wrong), 1.0)
        padding_garbage = codes.clone()
        padding_garbage[8:] = 0xAA
        self.assertEqual(self._compare(ref, padding_garbage), 0.0)

    def test_scale_layout_matches_moe_mxfp4_sort(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("moe_mxfp4_sort needs a GPU")
        from aiter import dtypes
        from aiter.utility.fp4_utils import moe_mxfp4_sort
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        token_num, topk, cols = 37, 3, 12
        routes = [t | (s << 24) for t in range(token_num) for s in range(topk)]
        routes = torch.tensor(routes, dtype=torch.int32)[torch.randperm(len(routes))]
        sorted_ids = torch.full((192,), token_num, dtype=torch.int32)
        sorted_ids[: routes.numel()] = routes
        sorted_ids = sorted_ids.cuda()
        num_valid_ids = torch.tensor([routes.numel()], dtype=torch.int32).cuda()
        scales = torch.randint(
            0, 255, (token_num * topk, cols), dtype=torch.uint8, device="cuda"
        ).view(dtypes.fp8_e8m0)

        ours = gemm_moe_tune._sorted_route_scales(
            scales, sorted_ids, num_valid_ids, token_num, topk
        ).view(torch.uint8)
        theirs = moe_mxfp4_sort(
            scales.view(token_num, topk, cols),
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=token_num,
            block_size=32,
        ).view(torch.uint8)

        mapped = ours != 0xFF
        self.assertEqual(int(mapped.sum()), token_num * topk * cols)
        self.assertTrue(torch.equal(ours[mapped], theirs.reshape(-1)[mapped]))


class TestFusedStage1Fp8AndSplitKCompare(unittest.TestCase):
    """The fused fp8 stage-1 reference quantizer, and the split-K check of
    dequantized values against the fp32 reference."""

    token_num, topk, inter = 37, 3, 256

    @staticmethod
    def _tune():
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        return gemm_moe_tune

    def _routes(self):
        import torch

        routes = [
            t | (s << 24) for t in range(self.token_num) for s in range(self.topk)
        ]
        routes = torch.tensor(routes, dtype=torch.int32)[torch.randperm(len(routes))]
        sorted_ids = torch.full((192,), self.token_num, dtype=torch.int32)
        sorted_ids[: routes.numel()] = routes
        return sorted_ids, torch.tensor([routes.numel()], dtype=torch.int32)

    def test_mxfp8_quant_uses_the_given_scale_rule(self):
        import torch

        from aiter.utility import fp4_utils

        tune = self._tune()
        x = (torch.randn(64, 256) * torch.rand(64, 1) * 10).view(64, 8, 32)
        scales_by_rule = []
        for rule in (fp4_utils.f32_to_fused_moe_mxfp8_scale, tune._roundup_mxfp8_scale):
            codes, scales = tune._mxfp8_quant(x.view(64, 256), rule)
            expected = rule(x.abs().amax(-1)).view(torch.uint8)
            self.assertTrue(torch.equal(scales.view(torch.uint8), expected))
            scale = fp4_utils.e8m0_to_f32(scales).unsqueeze(-1)
            dequant = codes.float().view(64, 8, 32) * scale
            # Round to nearest e4m3: half a step, 2^-4 relative or 2^-10 * scale
            # for subnormals.
            bound = torch.maximum(x.abs() * 2**-4, scale * 2**-10)
            self.assertTrue(torch.all((dequant - x).abs() <= bound))
            scales_by_rule.append(scales.view(torch.uint8))
        # The two kernels' rules disagree on some blocks, so each check names its own.
        self.assertFalse(torch.equal(*scales_by_rule))

    def test_fp8_codes_compare(self):
        import torch

        from aiter.utility import fp4_utils

        tune = self._tune()
        rule = fp4_utils.f32_to_fused_moe_mxfp8_scale
        x = torch.randn(64, 4, 256)
        codes, _ = tune._mxfp8_quant(x, rule)
        biased_codes, _ = tune._mxfp8_quant(x * 1.03, rule)
        compare = tune.quantized_output_compare
        self.assertEqual(compare(codes, codes.clone(), printLog=False), 0.0)
        self.assertEqual(compare(codes, biased_codes, printLog=False), 1.0)

    def test_scale_positions_invert_the_sorted_layout(self):
        import torch

        from aiter import dtypes

        tune = self._tune()
        sorted_ids, num_valid_ids = self._routes()
        cols = self.inter // 32
        scales = torch.randint(
            0, 255, (self.token_num * self.topk, cols), dtype=torch.uint8
        )
        sorted_scales = tune._sorted_route_scales(
            scales.view(dtypes.fp8_e8m0),
            sorted_ids,
            num_valid_ids,
            self.token_num,
            self.topk,
        )
        positions = tune._route_scale_positions(
            sorted_ids, num_valid_ids, self.token_num, self.topk, cols
        )
        gathered = sorted_scales.view(torch.uint8)[positions]
        self.assertTrue(torch.equal(gathered, scales))

    def test_scale_positions_reject_an_unmapped_route(self):
        tune = self._tune()
        sorted_ids, num_valid_ids = self._routes()
        sorted_ids[0] = self.token_num  # drop one route to padding
        with self.assertRaises(ValueError):
            tune._route_scale_positions(
                sorted_ids, num_valid_ids, self.token_num, self.topk, 8
            )

    def _fused_output(self, quant, scale_of=lambda x: x, x=None, split_k=True):
        """A fused stage-1 output: the result quantized by quant (rounded to bf16
        first under split_k, which sums bf16 partials), its scales in the sorted
        layout, and the reference the check reads."""
        import torch

        tune = self._tune()
        sorted_ids, num_valid_ids = self._routes()
        if x is None:
            x = torch.randn(self.token_num, self.topk, self.inter)
        quantized = scale_of(x)
        if split_k:
            quantized = quantized.bfloat16().float()
        codes, scales = quant(quantized)
        sorted_scales = tune._sorted_route_scales(
            scales, sorted_ids, num_valid_ids, self.token_num, self.topk
        )
        positions = tune._route_scale_positions(
            sorted_ids, num_valid_ids, self.token_num, self.topk, self.inter // 32
        )
        return (x, positions), (codes, sorted_scales)

    def _quantizers(self):
        from aiter import dtypes
        from aiter.ops.quant import per_1x32_f4_quant
        from aiter.utility import fp4_utils

        tune = self._tune()
        return {
            "fp4": lambda x: per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2),
            "fp8": lambda x: tune._mxfp8_quant(
                x, fp4_utils.f32_to_fused_moe_mxfp8_scale
            ),
        }

    def _split_k_compare(self):
        import functools

        return functools.partial(self._tune().dequantized_output_compare, split_k=True)

    def test_reference_rounding_passes(self):
        compare = self._tune().dequantized_output_compare
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, res = self._fused_output(quant, split_k=False)
                self.assertEqual(compare(ref, res, printLog=False), 0.0)

    def test_rounding_of_a_value_within_the_block_floor_passes(self):
        """A value on a rounding boundary may round either way: the kernel's fp32
        accumulation differs from the reference's in the last bits."""
        compare = self._tune().dequantized_output_compare
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, res = self._fused_output(
                    quant, scale_of=lambda x: x * (1 + 2**-14), split_k=False
                )
                self.assertEqual(compare(ref, res, printLog=False), 0.0)

    def test_neighbouring_code_fails_without_split_k(self):
        """One value on the other neighbouring code is within one step of the
        reference, which split-K allows and the default check does not."""
        import torch

        from aiter.utility import fp4_utils

        tune = self._tune()
        (x, positions), (codes, scales) = self._fused_output(
            self._quantizers()["fp8"], split_k=False
        )
        scale = fp4_utils.e8m0_to_f32(
            scales.view(torch.uint8)[positions.reshape(-1)]
        ).view(-1, 1)
        flat_x = x.reshape(-1, 32)
        rounded = codes.reshape(-1, 32).float() * scale
        byte = codes.view(torch.uint8).reshape(-1, 32)
        # A positive normal value well away from a rounding boundary, so the
        # other neighbour is more than the block floor from the reference.
        step = torch.exp2(torch.floor(torch.log2(flat_x.abs() / scale)) - 3) * scale
        candidates = (
            (flat_x > flat_x.abs().amax(-1, keepdim=True) * 2**-6)
            & ((flat_x - rounded).abs() < 0.25 * step)
            & (byte > 0x08)
            & (byte < 0x7E)
        ).nonzero()
        block, lane = candidates[0].tolist()
        moved = codes.clone()
        moved_bytes = moved.view(torch.uint8).reshape(-1, 32)
        moved_bytes[block, lane] += (
            1 if flat_x[block, lane] > rounded[block, lane] else -1
        )
        ref, res = (x, positions), (moved, scales)
        self.assertEqual(tune.dequantized_output_compare(ref, res, printLog=False), 1.0)
        self.assertEqual(self._split_k_compare()(ref, res, printLog=False), 0.0)

    def test_sorted_row_codes_pass(self):
        """The v2 fused stage 1 writes its codes in sorted rows."""
        import torch

        tune = self._tune()
        sorted_ids, num_valid_ids = self._routes()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                (x, positions), (codes, scales) = self._fused_output(
                    quant, split_k=False
                )
                token_major = codes.view(torch.uint8).view(
                    self.token_num, self.topk, -1
                )
                sorted_codes = torch.zeros(
                    sorted_ids.numel(), token_major.shape[-1], dtype=torch.uint8
                )
                for row, route in enumerate(sorted_ids[: num_valid_ids[0]].tolist()):
                    sorted_codes[row] = token_major[route & 0xFFFFFF, route >> 24]
                route_rows = tune._route_sorted_rows(
                    sorted_ids, num_valid_ids, self.token_num, self.topk
                )
                ref = (x, positions, route_rows, codes.dtype)
                compare = tune.dequantized_output_compare
                self.assertEqual(
                    compare(ref, (sorted_codes, scales), printLog=False), 0.0
                )
                first, second = route_rows[:2].tolist()
                sorted_codes[[first, second]] = sorted_codes[[second, first]]
                self.assertEqual(
                    compare(ref, (sorted_codes, scales), printLog=False), 1.0
                )

    def test_sorted_rows_reject_an_unmapped_route(self):
        tune = self._tune()
        sorted_ids, num_valid_ids = self._routes()
        sorted_ids[0] = self.token_num
        with self.assertRaises(ValueError):
            tune._route_sorted_rows(
                sorted_ids, num_valid_ids, self.token_num, self.topk
            )

    def test_split_k_within_one_step_passes(self):
        compare = self._split_k_compare()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, res = self._fused_output(quant)
                self.assertEqual(compare(ref, res, printLog=False), 0.0)

    def test_near_zero_value_within_the_block_floor_passes(self):
        """Near zero the fp8 grid is finer than fp32 accumulation noise: a value
        several fine steps off but tiny next to its block still passes."""
        import torch

        x = torch.randn(self.token_num, self.topk, self.inter)
        x[0, 0, 1] = 0
        (_, positions), res = self._fused_output(
            self._quantizers()["fp8"], x=x, split_k=False
        )
        for name, compare in (
            ("default", self._tune().dequantized_output_compare),
            ("split_k", self._split_k_compare()),
        ):
            with self.subTest(name):
                drifted = x.clone()
                drifted[0, 0, 1] = x[0, 0, :32].abs().max() * 2**-13
                self.assertEqual(
                    compare((drifted, positions), res, printLog=False), 0.0
                )
                drifted[0, 0, 1] = x[0, 0, :32].abs().max() * 2**-10
                self.assertEqual(
                    compare((drifted, positions), res, printLog=False), 1.0
                )

    def test_one_corrupted_route_fails(self):
        """A fault in one route of many is below any whole-output fraction, but
        each of its values is far from the reference."""
        import torch

        compare = self._split_k_compare()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, (codes, scales) = self._fused_output(quant)
                codes = codes.clone()
                codes.view(torch.uint8).view(-1, codes.shape[-1])[3] = 0
                self.assertEqual(compare(ref, (codes, scales), printLog=False), 1.0)

    def test_split_k_bias_fails(self):
        compare = self._split_k_compare()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, res = self._fused_output(quant, scale_of=lambda x: x * 1.03)
                self.assertEqual(compare(ref, res, printLog=False), 1.0)

    def test_split_k_misplaced_scales_fail(self):
        import torch

        compare = self._split_k_compare()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, (codes, scales) = self._fused_output(quant)
                flat = scales.view(torch.uint8)
                permuted = flat[torch.randperm(flat.numel())].view(scales.dtype)
                self.assertEqual(compare(ref, (codes, permuted), printLog=False), 1.0)

    def test_split_k_scale_one_exponent_off_fails(self):
        import torch

        compare = self._split_k_compare()
        for name, quant in self._quantizers().items():
            with self.subTest(name):
                ref, (codes, scales) = self._fused_output(quant)
                smaller = (scales.view(torch.uint8) - 1).view(scales.dtype)
                self.assertEqual(compare(ref, (codes, smaller), printLog=False), 1.0)


class TestStage2RelativeL2Compare(unittest.TestCase):
    """The stage-2 check: relative L2 and slope against the reference."""

    @staticmethod
    def _atomic(parts):
        """Sum the top-k partials in bf16, one add at a time, as an atomic
        stage 2 accumulates them."""
        import torch

        acc = torch.zeros(parts.shape[0], dtype=torch.bfloat16)
        for k in range(parts.shape[1]):
            acc = (acc.float() + parts[:, k].bfloat16().float()).bfloat16()
        return acc

    def _parts(self, topk=16):
        import torch

        torch.manual_seed(0)
        return torch.randn(200000, topk) * 8

    def _compare(self, ref, res):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        return gemm_moe_tune.relative_l2_compare(ref, res, printLog=False)

    def test_atomic_bf16_accumulation_passes(self):
        parts = self._parts()
        ref = parts.sum(1).bfloat16()
        err = self._compare(ref, self._atomic(parts))
        self.assertGreater(err, 0.0)
        self.assertLess(err, 1e-2)

    def test_three_percent_bias_fails(self):
        parts = self._parts()
        ref = parts.sum(1).bfloat16()
        self.assertEqual(self._compare(ref, self._atomic(parts * 1.03)), 1.0)

    def test_noise_above_the_cap_fails(self):
        import torch

        ref = self._parts().sum(1)
        noisy = ref + torch.randn_like(ref) * ref.std() * 0.02
        self.assertEqual(self._compare(ref.bfloat16(), noisy.bfloat16()), 1.0)

    def test_nan_output_fails(self):
        ref = self._parts().sum(1).bfloat16()
        res = ref.clone()
        res[0] = float("nan")
        self.assertEqual(self._compare(ref, res), 1.0)

    def test_one_wrong_row_among_many_fails(self):
        """One zeroed output row of 32768 is under the global cap but is
        entirely wrong on its own."""
        import torch

        torch.manual_seed(0)
        ref = torch.randn(32768, 128).bfloat16()
        res = ref.clone()
        res[5] = 0
        global_rel = ((res.float() - ref.float()).norm() / ref.float().norm()).item()
        self.assertLess(global_rel, 1e-2)
        self.assertEqual(self._compare(ref, res), 1.0)
        self.assertEqual(self._compare(ref, ref.clone()), 0.0)


class TestRunConfigRouting(unittest.TestCase):
    """How FmoeTuner.run_config calls fused_moe and builds its reference for a
    row: gate mode, activation dtype, fp8 scale rule and the fp32 retry. The
    kernel call and the torch reference are replaced by recorders."""

    md, inter, expert, topk = 256, 256, 4, 2

    def setUp(self):
        import torch

        from aiter.jit.utils.chip_info import get_gfx_runtime

        if not torch.cuda.is_available():
            self.skipTest("run_config builds its inputs on the GPU")
        if get_gfx_runtime() != "gfx950":
            self.skipTest("the rows and their routing are gfx950's")

    def _row(self, token, q_dtype_a, q_dtype_w, act="Silu", kernel1="", stage1=0):
        return {
            "gfx": "gfx950",
            "cu_num": 256,
            "token": token,
            "model_dim": self.md,
            "inter_dim": self.inter,
            "expert": self.expert,
            "topk": self.topk,
            "act_type": f"ActivationType.{act}",
            "dtype": "torch.bfloat16",
            "q_dtype_a": q_dtype_a,
            "q_dtype_w": q_dtype_w,
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
            "kernelName1": kernel1,
            "run_1stage": stage1,
        }

    def _run(self, row, first_reference_matches=True):
        """Run run_config on one row; return the fused_moe kwargs, the reference
        calls and the result. The kernel returns ones; the reference returns
        ones, or zeros unless it quantizes the fp32 result."""
        from unittest import mock

        import pandas as pd
        import torch

        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        kernel_calls, reference_calls = [], []

        def kernel(hidden, *args, **kwargs):
            kernel_calls.append(kwargs)
            return torch.ones(hidden.shape, dtype=torch.bfloat16, device="cuda")

        def reference(a1_qt, *args, quantize_fp32=False, **kwargs):
            reference_calls.append(
                dict(kwargs, a1_dtype=a1_qt.dtype, quantize_fp32=quantize_fp32)
            )
            fill = 1.0 if first_reference_matches or quantize_fp32 else 0.0
            return torch.full(
                (a1_qt.shape[0], self.md), fill, dtype=torch.bfloat16, device="cuda"
            )

        tuner = gemm_moe_tune.FmoeTuner.__new__(gemm_moe_tune.FmoeTuner)
        tuner.untunedf = pd.DataFrame([row])
        args = SimpleNamespace(warmup=1, iters=2, errRatio=0.05)
        with mock.patch.object(
            gemm_moe_tune.FmoeTuner, "torch_moe_2stages", staticmethod(reference)
        ):
            (result,) = tuner.run_config(args, target_fused_moe=kernel)
        self.assertEqual(result["status"], "ok", result)
        return kernel_calls[-1], reference_calls

    def test_a8w4_runs_interleaved_with_the_fp8_activation(self):
        from aiter import dtypes
        from aiter.utility import fp4_utils

        kernel, refs = self._run(
            self._row(
                256,
                "torch.float8_e4m3fn",
                "torch.float4_e2m1fn_x2",
                kernel1="flydsl_moe1_afp8_wfp4_bf16_t32x64x256_fp8",
            )
        )
        self.assertEqual(kernel["gate_mode"], "interleave")
        self.assertIsNone(kernel["quant_dtype_a"])
        self.assertEqual(refs[0]["a1_dtype"], dtypes.fp8)
        self.assertIsNotNone(refs[0]["a1_scale"])
        self.assertIs(refs[0]["fp8_scale_fn"], fp4_utils.f32_to_fused_moe_mxfp8_scale)

    def test_unfused_stage1_quantizes_stage2_input_with_roundup(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        _, refs = self._run(
            self._row(
                256,
                "torch.float8_e4m3fn",
                "torch.float4_e2m1fn_x2",
                kernel1="flydsl_moe1_afp8_wfp4_bf16_t32x64x256",
            )
        )
        self.assertIs(refs[0]["fp8_scale_fn"], gemm_moe_tune._roundup_mxfp8_scale)

    def test_a8w4_below_the_bound_warns_and_runs_bf16(self):
        from aiter import dtypes

        with self.assertLogs("aiter", level="WARNING") as logs:
            kernel, refs = self._run(
                self._row(1, "torch.float8_e4m3fn", "torch.float4_e2m1fn_x2")
            )
        self.assertEqual(kernel["gate_mode"], "interleave")
        self.assertEqual(refs[0]["a1_dtype"], dtypes.bf16)
        self.assertTrue(any("kernels are not what" in m for m in logs.output))

    def test_mxfp8_runs_interleaved_with_the_fp8_activation(self):
        from aiter import dtypes

        kernel, refs = self._run(
            self._row(256, "torch.float8_e4m3fn", "torch.float8_e4m3fn")
        )
        self.assertEqual(kernel["gate_mode"], "interleave")
        self.assertEqual(refs[0]["a1_dtype"], dtypes.fp8)

    def test_situv2_row_passes_its_tuned_activation_dtype(self):
        from aiter import dtypes

        kernel, refs = self._run(
            self._row(256, "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2", "Situv2")
        )
        self.assertEqual(kernel["gate_mode"], "separated")
        self.assertEqual(kernel["quant_dtype_a"], dtypes.fp4x2)
        self.assertEqual(refs[0]["a1_dtype"], dtypes.fp4x2)

    def test_asm_1stage_reference_quantizes_the_activation_to_fp4(self):
        from aiter import dtypes

        kernel, refs = self._run(
            self._row(
                16,
                "torch.bfloat16",
                "torch.float4_e2m1fn_x2",
                "Situv2",
                kernel1="_ZN5aiter46fmoe_bf16_pertokenMXfp4_g1u1_flat_situv2_16x32E",
                stage1=1,
            )
        )
        self.assertEqual(kernel["quant_dtype_a"], dtypes.bf16)
        self.assertEqual(refs[0]["a1_dtype"], dtypes.fp4x2)

    def test_failing_shape_is_retried_at_the_fp32_quantization_point(self):
        row = self._row(256, "torch.float4_e2m1fn_x2", "torch.float4_e2m1fn_x2")
        _, refs = self._run(row, first_reference_matches=False)
        self.assertEqual([r["quantize_fp32"] for r in refs], [False, True])
        _, refs = self._run(row)
        self.assertEqual([r["quantize_fp32"] for r in refs], [False])


class TestMoePowerRanking(unittest.TestCase):
    def test_power_rank_orders_pairs_by_stage_energy(self):
        import pandas as pd

        from aiter.utility.selection_policy import POWER
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import rank_moe_pairs

        pairs = pd.DataFrame(
            [
                {
                    "kernelName1": "fast_hot",
                    "kernelName2": "stage2_a",
                    "us1": 40.0,
                    "us2": 20.0,
                    "us": 60.0,
                    "watts1": 700.0,
                    "watts2": 700.0,
                    "tflops": 100.0,
                },
                {
                    "kernelName1": "slower_cooler",
                    "kernelName2": "stage2_b",
                    "us1": 50.0,
                    "us2": 20.0,
                    "us": 70.0,
                    "watts1": 400.0,
                    "watts2": 400.0,
                    "tflops": 80.0,
                },
                {
                    "kernelName1": "one_stage",
                    "kernelName2": None,
                    "us1": 55.0,
                    "us2": 0.0,
                    "us": 55.0,
                    "watts1": 500.0,
                    "watts2": float("nan"),
                    "tflops": 90.0,
                },
            ]
        )
        ranked = rank_moe_pairs(pairs, POWER)
        # one_stage 55*500 = 27500, slower_cooler 50*400+20*400 = 28000,
        # fast_hot 40*700+20*700 = 42000.
        self.assertEqual(
            list(ranked["kernelName1"]),
            ["one_stage", "slower_cooler", "fast_hot"],
        )
        self.assertEqual(list(ranked["rank"]), [1, 2, 3])
        self.assertEqual(float(ranked.iloc[0]["energy"]), 27500.0)
        self.assertAlmostEqual(float(ranked.iloc[0]["watts"]), 500.0, places=3)
        self.assertAlmostEqual(
            float(ranked.iloc[0]["tflops_per_w"]), 90 / 500, places=5
        )

    def test_latency_rank_ignores_energy_order(self):
        import pandas as pd

        from aiter.utility.selection_policy import LATENCY
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import rank_moe_pairs

        pairs = pd.DataFrame(
            [
                {
                    "kernelName1": "slow",
                    "us1": 80.0,
                    "us2": 0.0,
                    "us": 80.0,
                    "watts1": 100.0,
                },
                {
                    "kernelName1": "fast",
                    "us1": 40.0,
                    "us2": 0.0,
                    "us": 40.0,
                    "watts1": 600.0,
                },
            ]
        )
        ranked = rank_moe_pairs(pairs, LATENCY)
        # Latency keeps the -o2 profile exactly as main writes it: same
        # columns, same row order, no rank or energy columns.
        self.assertEqual(list(ranked.columns), list(pairs.columns))
        self.assertEqual(list(ranked["kernelName1"]), ["slow", "fast"])
        self.assertEqual(LATENCY.sort_frame(ranked).iloc[0]["kernelName1"], "fast")

    def test_unmeasured_power_sorts_last(self):
        import pandas as pd

        from aiter.utility.selection_policy import POWER
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import rank_moe_pairs

        pairs = pd.DataFrame(
            [
                {"kernelName1": "no_watts", "us1": 10.0, "us2": 10.0, "us": 20.0},
                {
                    "kernelName1": "measured",
                    "us1": 30.0,
                    "us2": 10.0,
                    "us": 40.0,
                    "watts1": 500.0,
                    "watts2": 500.0,
                },
            ]
        )
        ranked = rank_moe_pairs(pairs, POWER)
        self.assertEqual(ranked.iloc[0]["kernelName1"], "measured")
        self.assertTrue(pd.isna(ranked.iloc[1]["energy"]))


class TestPowerSelectionGaps(unittest.TestCase):
    def test_candidate_beats_uses_energy_only_in_power_mode(self):
        from aiter.utility.tune_result import candidate_beats

        hot = {"us": 10.0, "watts": 1000.0}
        cool = {"us": 20.0, "watts": 200.0}
        self.assertTrue(candidate_beats(hot, cool, True))
        self.assertFalse(candidate_beats(cool, hot, True))
        self.assertFalse(candidate_beats(hot, cool, False))
        self.assertTrue(candidate_beats(hot, {"us": 9.0, "watts": 5000.0}, False))
        self.assertTrue(candidate_beats(None, hot, True))

    def test_grouped_power_keeps_lower_energy_tile(self):
        import pandas as pd

        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.GroupedFmoeTuner.__new__(gemm_moe_tune.GroupedFmoeTuner)
        hot = {"tile_m": 128, "kernelName1": "hot"}
        cool = {"tile_m": 16, "kernelName1": "cool"}
        args = SimpleNamespace(
            fast_scan=False,
            select="power",
            warmup=1,
            iters=1,
            tune_file="unused.csv",
        )
        frame = pd.DataFrame(
            [{"token": 4, "inter_dim": 32, "q_dtype_a": "fp4", "expert": 2, "topk": 1}]
        )

        def run_candidate(_row, candidate, _args):
            if candidate["tile_m"] == 128:
                candidate["watts"] = 1000.0
                return 10.0
            candidate["watts"] = 200.0
            return 20.0

        with (
            mock.patch.object(tuner, "_write_layout", return_value=([], [], set())),
            mock.patch.object(tuner, "_candidate_rows", return_value=[hot, cool]),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            rows = tuner.tune(frame, None, args)

        self.assertEqual(rows[0]["kernelName1"], "cool")

        args.select = "latency"
        with (
            mock.patch.object(tuner, "_write_layout", return_value=([], [], set())),
            mock.patch.object(tuner, "_candidate_rows", return_value=[hot, cool]),
            mock.patch.object(tuner, "_run_candidate", side_effect=run_candidate),
            mock.patch("builtins.print"),
        ):
            rows = tuner.tune(frame, None, args)

        self.assertEqual(rows[0]["kernelName1"], "hot")

    def test_mxfp4_power_retimes_every_accurate_candidate(self):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(gemm_moe_tune.Mxfp4FlydslTuner)

        def candidate(name):
            return {"kernelName1": f"g1_{name}", "kernelName2": f"g2_{name}", "us": 0}

        candidates = [candidate(f"candidate_{i}") for i in range(4)]
        args = SimpleNamespace(
            fast_scan=True,
            fast_scan_warmup_accuracy_checks=1,
            fast_scan_iters=10,
            fast_scan_final_iters=100,
            fast_scan_finalists=1,
            timeout=0,
            errRatio=0.1,
            warmup=1,
            iters=2,
            select="power",
            mxfp4_search_mode="prune",
        )
        row = {
            "token": 1,
            "model_dim": 256,
            "inter_dim": 256,
            "expert": 8,
            "topk": 2,
            "act_type": "ActivationType.Silu",
            "dtype": "torch.bfloat16",
            "q_dtype_a": "torch.float4_e2m1fn_x2",
            "q_dtype_w": "torch.float4_e2m1fn_x2",
            "q_type": "QuantType.per_1x32",
            "use_g1u1": 1,
            "doweight_stage1": 0,
        }
        final_names = []

        def run_candidate(_row, candidate, _args, **kwargs):
            name = candidate["kernelName1"]
            if kwargs.get("num_iters") == 10:
                candidate["us"] = 10.0 + candidates.index(candidate)
                candidate["err1"] = 0.01
                return candidate["us"]
            final_names.append(name)
            # Last candidate is slower and much cooler, so it wins on energy.
            candidate["us"] = 40.0 if name.endswith("_3") else 10.0
            candidate["watts"] = 100.0 if name.endswith("_3") else 1000.0
            candidate["err1"] = 0.01
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

        self.assertEqual(
            final_names,
            [f"g1_candidate_{i}" for i in range(4)],
        )
        self.assertEqual(best["kernelName1"], "g1_candidate_3")

    def test_k5_post_process_keeps_lower_energy_duplicate(self):
        from csrc.gdn_k5.chunk_gdn_h_opt_tune import K5BvTuner

        tuner = K5BvTuner.__new__(K5BvTuner)
        tuner.keys = ["H"]
        tuner.columns = ["H", "BV", "us"]
        rows = [
            {"H": 1, "BV": 64, "us": 10.0, "watts": 1000.0},
            {"H": 1, "BV": 16, "us": 20.0, "watts": 200.0},
        ]
        power = tuner.post_process(rows, SimpleNamespace(select="power"))
        latency = tuner.post_process(rows, SimpleNamespace(select="latency"))
        self.assertEqual(int(power.iloc[0]["BV"]), 16)
        self.assertEqual(int(latency.iloc[0]["BV"]), 64)

    def test_unmeasured_power_dedup_keeps_lowest_latency(self):
        import pandas as pd

        from aiter.utility.tune_result import sort_by_energy_then_latency

        frame = pd.DataFrame(
            [
                {"name": "slow", "us": 20.0, "watts": float("nan"), "key": 1},
                {"name": "fast", "us": 10.0, "watts": float("nan"), "key": 1},
                {"name": "cool", "us": 30.0, "watts": 100.0, "key": 2},
            ]
        )
        ordered = sort_by_energy_then_latency(frame)
        same_key = ordered[ordered["key"] == 1]
        self.assertEqual(list(same_key["name"]), ["fast", "slow"])
        self.assertEqual(ordered.iloc[0]["name"], "cool")

    def test_latency_frame_does_not_gain_watts(self):
        import pandas as pd

        from aiter.utility.selection_policy import LATENCY, POWER

        frame = pd.DataFrame([{"us": 10.0, "tflops": 100.0}])
        results = [SimpleNamespace(watts=400.0)]
        self.assertNotIn("watts", LATENCY.attach_columns(frame, results).columns)
        self.assertEqual(
            float(POWER.attach_columns(frame, results).iloc[0]["watts"]), 400.0
        )
        candidate = {"us": 1.0}
        LATENCY.record(candidate, "watts", lambda: self.fail("latency measured"))
        self.assertNotIn("watts", candidate)

    def test_selection_prefers_energy_over_tflops_per_watt(self):
        import pandas as pd

        from aiter.utility.tune_result import sort_frame_for_selection

        # Same FLOPs would make the two orders agree. Different tflops here
        # shows the rank is us*watts, not the stored efficiency column.
        frame = pd.DataFrame(
            [
                {"name": "hot", "us": 10.0, "watts": 1000.0, "tflops_per_w": 9.0},
                {"name": "cool", "us": 20.0, "watts": 200.0, "tflops_per_w": 1.0},
            ]
        )
        ordered = sort_frame_for_selection(frame, select_power=True)
        self.assertEqual(list(ordered["name"]), ["cool", "hot"])

    def test_power_path_expands_per_device_and_rejects_one_file(self):
        from aiter.utility.package_power import (
            power_input_path,
            shared_power_override_rejected,
        )

        override = "AITER_POWER_PATH"
        previous = os.environ.get(override)
        try:
            os.environ[override] = "/sys/class/drm/card{device}/power1_input"
            self.assertEqual(power_input_path(3), "/sys/class/drm/card3/power1_input")
            self.assertIsNone(shared_power_override_rejected())
            os.environ[override] = "/sys/class/drm/card0/power1_input"
            self.assertEqual(power_input_path(2), "/sys/class/drm/card0/power1_input")
            self.assertIn("{device}", shared_power_override_rejected())
        finally:
            if previous is None:
                os.environ.pop(override, None)
            else:
                os.environ[override] = previous


class TestPowerTableSeparation(unittest.TestCase):
    def test_device_conditions_read_level_cap_and_sclk(self):
        import tempfile

        from aiter.utility import package_power

        with tempfile.TemporaryDirectory() as device_dir:
            os.makedirs(f"{device_dir}/hwmon/hwmon3")
            with open(f"{device_dir}/power_dpm_force_performance_level", "w") as f:
                f.write("perf_determinism\n")
            with open(f"{device_dir}/hwmon/hwmon3/power1_cap", "w") as f:
                f.write("1000000000\n")
            with open(f"{device_dir}/pp_od_clk_voltage", "w") as f:
                f.write("OD_SCLK:\n0: 500Mhz\n1: 2100Mhz\nOD_MCLK:\n0: 900Mhz\n")
            with mock.patch.object(
                package_power, "_device_sysfs_dir", return_value=device_dir
            ):
                conditions = package_power._device_conditions(0)
        self.assertEqual(
            conditions,
            {
                "perf_level": "perf_determinism",
                "power_cap_w": "1000",
                "sclk_max_mhz": "2100",
            },
        )

    def test_tune_conditions_mark_devices_that_differ(self):
        from aiter.utility import package_power

        states = [
            {"perf_level": "auto", "power_cap_w": "1400", "sclk_max_mhz": "2400"},
            {"perf_level": "auto", "power_cap_w": "1000", "sclk_max_mhz": "2400"},
        ]
        with mock.patch.object(package_power, "_device_conditions", side_effect=states):
            merged = package_power.tune_conditions([0, 1])
        self.assertEqual(merged["perf_level"], "auto")
        self.assertEqual(merged["power_cap_w"], "mixed")

    def test_power_rows_are_stamped_and_columns_extended(self):
        import pandas as pd

        from aiter.utility.base_tuner import TunerCommon

        def stamp_stub():
            stub = SimpleNamespace(
                columns=["M", "us"], keys=["M"], POWER_WATT_COLUMNS=("watts",)
            )
            stub.policy = TunerCommon.policy
            stub.policy_for = lambda args: TunerCommon.policy_for(stub, args)
            stub.power_metrics = lambda df: TunerCommon.power_metrics(stub, df)
            stub._rows_with_finite_watts = (
                lambda df: TunerCommon._rows_with_finite_watts(stub, df)
            )
            return stub

        stub = stamp_stub()
        args = SimpleNamespace(select="power", mp=1)
        frame = pd.DataFrame([{"M": 1, "us": 2.0, "tflops": 800.0, "watts": 400.0}])
        conditions = {
            "perf_level": "auto",
            "power_cap_w": "1400",
            "sclk_max_mhz": "2400",
        }
        with mock.patch(
            "aiter.utility.package_power.tune_conditions", return_value=conditions
        ):
            stamped = TunerCommon._stamp_power_rows(stub, frame, args)
        self.assertEqual(stamped.iloc[0]["select"], "power")
        self.assertEqual(stamped.iloc[0]["power_cap_w"], "1400")
        self.assertAlmostEqual(stamped.iloc[0]["tflops_per_w"], 2.0)
        self.assertEqual(
            stub.columns,
            [
                "M",
                "us",
                "watts",
                "tflops_per_w",
                "select",
                "perf_level",
                "power_cap_w",
                "sclk_max_mhz",
            ],
        )
        with mock.patch(
            "aiter.utility.package_power.tune_conditions", return_value=conditions
        ):
            unmeasured = TunerCommon._stamp_power_rows(
                stamp_stub(),
                pd.DataFrame([{"M": 1, "us": 2.0, "watts": float("nan")}]),
                args,
            )
            mixed = TunerCommon._stamp_power_rows(
                stamp_stub(),
                pd.DataFrame(
                    [
                        {"M": 1, "us": 2.0, "watts": 400.0},
                        {"M": 2, "us": 3.0, "watts": float("nan")},
                    ]
                ),
                args,
            )
        # Unmeasured shapes are latency picks: they are not written to
        # power_configs/ and stay untuned for the next power tune.
        self.assertTrue(unmeasured.empty)
        self.assertIn("M", unmeasured.columns)
        self.assertEqual(list(mixed["M"]), [1])
        self.assertEqual(list(mixed["select"]), ["power"])
        latency = TunerCommon._stamp_power_rows(
            stamp_stub(),
            frame,
            SimpleNamespace(select="latency"),
        )
        self.assertNotIn("select", latency.columns)

    def test_power_replay_launches_the_kernel_without_perftest_options(self):
        from aiter.utility.package_power import kernel_call

        calls = []

        def kernel(x, scale=1):
            calls.append((x, scale))

        call = kernel_call(
            kernel,
            (3,),
            {
                "num_warmup": 10,
                "num_iters": 101,
                "num_rotate_args": 0,
                "testGraph": False,
                "needTrace": False,
                "use_cuda_event": False,
                "profile_warm_iters": 1,
                "scale": 2,
            },
        )
        with mock.patch("aiter.test_common.run_perftest") as perftest:
            call()
            call()
        perftest.assert_not_called()
        self.assertEqual(calls, [(3, 2), (3, 2)])

    def test_power_window_keeps_few_calls_in_flight(self):
        from aiter.utility import package_power

        log = []

        class Event:
            def __init__(self):
                self.index = len(log)

            def record(self):
                log.append(("record", self.index))

            def synchronize(self):
                log.append(("wait", self.index))

        in_flight = package_power._InFlight()
        in_flight._event = Event
        for _ in range(10):
            in_flight.mark()
        waits = [index for kind, index in log if kind == "wait"]
        records = [index for kind, index in log if kind == "record"]
        depth = package_power._InFlight.DEPTH
        self.assertEqual(len(records), 10)
        # Each wait is on the call DEPTH launches back, so the GPU never drains.
        self.assertEqual(len(waits), 10 - depth)
        self.assertEqual(len(in_flight._events), depth)

        # A 10 us call gets one event per SPAN_S of work, not one per launch.
        log.clear()
        sparse = package_power._InFlight(10e-6)
        sparse._event = Event
        every = int(package_power._InFlight.SPAN_S / 10e-6)
        for _ in range(every * 3):
            sparse.mark()
        self.assertEqual(len([kind for kind, _ in log if kind == "record"]), 3)

    def test_launch_bound_window_is_scaled_to_kernel_power(self):
        import math
        import time

        from aiter.utility import package_power

        active = package_power.active_watts
        self.assertEqual(active(500.0, 300.0, 1.0), 500.0)
        self.assertAlmostEqual(active(400.0, 300.0, 0.5), 500.0)
        self.assertTrue(math.isnan(active(310.0, 300.0, 0.05)))
        self.assertEqual(active(400.0, None, 0.5), 400.0)

        def launch():
            # Host-bound: 1 ms of host time per "launch" of a 0.5 ms kernel.
            time.sleep(0.001)

        with (
            mock.patch.object(package_power, "power_input_path", return_value="x"),
            mock.patch.object(package_power, "_read_watts", return_value=400.0),
            mock.patch.object(package_power, "idle_watts", return_value=300.0),
            mock.patch.object(package_power, "_sync_cuda"),
            mock.patch.object(package_power._InFlight, "mark"),
        ):
            raw = package_power.measure_package_watts(launch, seconds=0.1)
            corrected = package_power.measure_package_watts(
                launch, seconds=0.1, kernel_us=500.0
            )
        self.assertEqual(raw, 400.0)
        # busy is about 0.5, so 300 + (400 - 300) / 0.5 = 500.
        self.assertGreater(corrected, 430.0)
        self.assertLess(corrected, 620.0)

    def test_a16w16_hands_its_policy_to_hipblaslt(self):
        from csrc.gemm_a16w16 import gemm_a16w16_tune

        tuner = gemm_a16w16_tune.GemmA16W16Tuner.__new__(
            gemm_a16w16_tune.GemmA16W16Tuner
        )
        seen = {}

        class FakeGemm:
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)

            def run_solutions(self):
                return []

            def cleanup(self):
                pass

        ds = {
            "M": 1,
            "N": 128,
            "K": 128,
            "bias": False,
            "dtype": "torch.bfloat16",
            "outdtype": "torch.bfloat16",
            "scaleAB": False,
            "bpreshuffle": False,
        }
        args = SimpleNamespace(
            select="power", mp=1, errRatio=0.05, profile_file="", timeout=0, verbose=0
        )
        with (
            mock.patch.object(gemm_a16w16_tune, "HipblasltGemm", FakeGemm),
            mock.patch.object(tuner, "get_gfx", return_value="gfx950"),
            mock.patch.object(tuner, "get_cu_num", return_value=256),
        ):
            tuner._run_hipblaslt(ds, args)
        self.assertEqual(seen["policy"].name, "power")

    def test_policy_owns_every_select_decision(self):
        from aiter.utility.selection_policy import (
            LATENCY,
            POWER,
            active_policy,
            make_policy,
            use_policy,
        )

        hot = {"us": 10.0, "watts": 1000.0}
        cool = {"us": 20.0, "watts": 200.0}
        self.assertIs(make_policy("efficiency"), POWER)
        self.assertIs(make_policy(SimpleNamespace(select="latency")), LATENCY)
        self.assertIs(make_policy(None), LATENCY)
        self.assertTrue(POWER.beats(hot, cool))
        self.assertFalse(LATENCY.beats(hot, cool))
        # Latency noise floor: 0.4 us faster is not a win at a 0.5 us margin.
        self.assertFalse(LATENCY.beats({"us": 10.0}, {"us": 9.6}, 0.5))
        self.assertTrue(LATENCY.beats({"us": 10.0}, {"us": 9.4}, 0.5))
        self.assertEqual(POWER.pair_energy(10.0, 100.0, 0.0, float("nan")), 1000.0)
        self.assertEqual(POWER.accurate_count(5, 40), 40)
        self.assertEqual(LATENCY.accurate_count(5, 40), 5)
        self.assertIs(POWER.best([hot, cool], latency_choice=lambda c: c[0]), cool)
        self.assertIs(LATENCY.best([cool, hot], latency_choice=lambda c: c[0]), cool)
        self.assertIs(active_policy(), LATENCY)
        with use_policy(POWER):
            self.assertIs(active_policy(), POWER)
        self.assertIs(active_policy(), LATENCY)

    def test_mp_tuner_reads_the_activated_policy(self):
        from aiter.utility import mp_tuner
        from aiter.utility.selection_policy import POWER, use_policy

        class PoolStarted(Exception):
            pass

        def run(fast_mode):
            mp_tuner.mp_tuner([((("k",), 0),)], [(1, ())], 2, fast_mode=fast_mode)

        # A single-file AITER_POWER_PATH is rejected only when this pass
        # measures power, so the error shows which policy mp_tuner used.
        with (
            mock.patch.dict(os.environ, {"AITER_POWER_PATH": "/tmp/one_file"}),
            mock.patch.object(mp_tuner.torch.cuda, "device_count", return_value=2),
            mock.patch.object(mp_tuner, "_new_pool", side_effect=PoolStarted),
        ):
            with self.assertRaises(PoolStarted):
                run(fast_mode=False)
            with use_policy(POWER):
                with self.assertRaises(ValueError):
                    run(fast_mode=False)
                with self.assertRaises(PoolStarted):
                    run(fast_mode=True)

    def test_tuners_do_not_decide_select_mode_themselves(self):
        import re

        tuner_files = [
            REPO_ROOT / "aiter/utility/base_tuner.py",
            REPO_ROOT / "aiter/utility/mp_tuner.py",
            REPO_ROOT / "gradlib/gradlib/GemmTuner.py",
            *sorted((REPO_ROOT / "csrc").rglob("*_tune.py")),
        ]
        forbidden = re.compile(
            r"power_selection_enabled|select_power|record_package_watts|"
            r"measure_package_watts|measure_kernel|selection_rank\(|"
            r"candidate_beats\(|sort_frame_for_selection|"
            r"os\.environ\[\"AITER_SELECT\"\]|os\.environ\.get\(\"AITER_SELECT\""
        )
        offenders = []
        for path in tuner_files:
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if forbidden.search(line):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_select_power_requires_support_and_refuses_latency_gate(self):
        from aiter.utility.base_tuner import TunerCommon

        def error(message):
            raise SystemExit(message)

        stub = SimpleNamespace(
            SUPPORTS_POWER_SELECT=False,
            parser=SimpleNamespace(error=error),
            get_out_file=lambda path: path,
        )
        previous = os.environ.get("AITER_SELECT")
        try:
            TunerCommon._apply_select(
                stub,
                SimpleNamespace(select="latency", compare=False, update_improved=False),
            )
            self.assertEqual(stub.policy.name, "latency")
            self.assertEqual(os.environ.get("AITER_SELECT"), previous)
            with self.assertRaises(SystemExit) as unsupported:
                TunerCommon._apply_select(
                    stub,
                    SimpleNamespace(
                        select="power", compare=False, update_improved=False
                    ),
                )
            self.assertIn("not supported", str(unsupported.exception))
            stub.SUPPORTS_POWER_SELECT = True
            with self.assertRaises(SystemExit) as gated:
                TunerCommon._apply_select(
                    stub,
                    SimpleNamespace(
                        select="power",
                        compare=True,
                        update_improved=True,
                        tune_file="/tmp/out.csv",
                    ),
                )
            self.assertIn("latency", str(gated.exception))
        finally:
            if previous is None:
                os.environ.pop("AITER_SELECT", None)
            else:
                os.environ["AITER_SELECT"] = previous

    def test_moe_post_process_keeps_measured_watts(self):
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import (
            GroupedFmoeTuner,
            _frame_keeping_watts,
        )

        frame = _frame_keeping_watts(
            ["token", "us"],
            [{"token": 1, "us": 2.0, "watts": 400.0}],
        )
        self.assertEqual(float(frame.iloc[0]["watts"]), 400.0)
        dropped = GroupedFmoeTuner.__new__(GroupedFmoeTuner)
        dropped.columns = ["token", "us"]
        kept = dropped.post_process([{"token": 1, "us": 2.0, "watts": 500.0}], None)
        self.assertEqual(float(kept.iloc[0]["watts"]), 500.0)

    def test_power_tune_of_shipped_table_writes_to_power_configs(self):
        import argparse
        import tempfile

        from aiter.utility.base_tuner import TunerCommon

        with tempfile.TemporaryDirectory() as root:
            shipped = f"{root}/aiter/configs/tuned_fmoe.csv"
            parser = argparse.ArgumentParser()
            parser.add_argument("-o", dest="tune_file", default=shipped)
            stub = SimpleNamespace(
                parser=parser, get_out_file=lambda path: path.split(os.pathsep)[0]
            )
            with mock.patch("aiter.jit.core.AITER_ROOT_DIR", root):
                args = SimpleNamespace(tune_file=shipped)
                TunerCommon._redirect_power_output(stub, args)
                self.assertEqual(
                    args.tune_file,
                    f"{root}/aiter/configs/power_configs/tuned_fmoe.csv",
                )
                chosen = SimpleNamespace(tune_file="/tmp/my_model_power.csv")
                TunerCommon._redirect_power_output(stub, chosen)
                self.assertEqual(chosen.tune_file, "/tmp/my_model_power.csv")
                explicit = SimpleNamespace(
                    tune_file=f"{root}/aiter/configs/model_configs/gptoss_tuned_fmoe.csv"
                )
                TunerCommon._redirect_power_output(stub, explicit)
                self.assertEqual(
                    explicit.tune_file,
                    f"{root}/aiter/configs/power_configs/gptoss_tuned_fmoe.csv",
                )

    def test_merge_refuses_power_and_latency_rows_for_one_shape(self):
        import pandas as pd

        from aiter.utility.power_configs import power_merge_conflict

        mixed = pd.DataFrame(
            [
                {"M": 1, "us": 5.0, "select": 0, "power_cap_w": 0},
                {"M": 1, "us": 9.0, "select": "power", "power_cap_w": "1400"},
            ]
        )
        self.assertIn("latency row", power_merge_conflict(mixed, ["M"]))
        caps = pd.DataFrame(
            [
                {"M": 1, "us": 5.0, "select": "power", "power_cap_w": "1400"},
                {"M": 1, "us": 9.0, "select": "power", "power_cap_w": "1000"},
            ]
        )
        self.assertIn("power_cap_w", power_merge_conflict(caps, ["M"]))
        same = caps.assign(power_cap_w="1400")
        self.assertIsNone(power_merge_conflict(same, ["M"]))
        latency_only = pd.DataFrame([{"M": 1, "us": 5.0}, {"M": 1, "us": 6.0}])
        self.assertIsNone(power_merge_conflict(latency_only, ["M"]))

    def test_power_overlay_replaces_only_covered_shapes(self):
        import tempfile

        import pandas as pd

        from aiter.jit.core import AITER_CONFIG, AITER_CONFIGS
        from aiter.utility.power_configs import _overlay_power_tables

        with tempfile.TemporaryDirectory() as tmp:
            latency_file = f"{tmp}/latency.csv"
            power_file = f"{tmp}/tuned_demo.csv"
            pd.DataFrame(
                [
                    {"M": 1, "us": 5.0, "kernel": "fast", "xbf16": 0},
                    {"M": 2, "us": 7.0, "kernel": "only_latency", "xbf16": 0},
                ]
            ).to_csv(latency_file, index=False)
            pd.DataFrame(
                [
                    {
                        "M": 1.0,
                        "us": 9.0,
                        "kernel": "cool",
                        "watts": 300.0,
                        "select": "power",
                    }
                ]
            ).to_csv(power_file, index=False)
            with (
                mock.patch.object(
                    AITER_CONFIG, "_shape_dedup_keys", return_value=["M"]
                ),
                mock.patch("tempfile.gettempdir", return_value=tmp),
            ):
                merged_path = _overlay_power_tables(
                    AITER_CONFIGS, latency_file, [power_file], "tuned_demo", None
                )
            merged = pd.read_csv(merged_path).sort_values("M")
        self.assertEqual(list(merged["kernel"]), ["cool", "only_latency"])
        self.assertEqual(list(merged["select"]), ["power", "latency"])
        self.assertEqual(list(merged["xbf16"]), [0, 0])

    def test_power_overlay_ignores_rows_not_stamped_power(self):
        import tempfile

        import pandas as pd

        from aiter.jit.core import AITER_CONFIG, AITER_CONFIGS
        from aiter.utility.power_configs import _overlay_power_tables

        with tempfile.TemporaryDirectory() as tmp:
            latency_file = f"{tmp}/latency.csv"
            stamped = f"{tmp}/tuned_demo.csv"
            unstamped = f"{tmp}/old_tuned_demo.csv"
            pd.DataFrame(
                [
                    {"M": 1, "us": 5.0, "kernel": "shipped_1"},
                    {"M": 2, "us": 7.0, "kernel": "shipped_2"},
                    {"M": 3, "us": 8.0, "kernel": "shipped_3"},
                ]
            ).to_csv(latency_file, index=False)
            pd.DataFrame(
                [
                    {"M": 1, "us": 9.0, "kernel": "cool", "select": "power"},
                    {"M": 2, "us": 6.0, "kernel": "latency_pick", "select": ""},
                ]
            ).to_csv(stamped, index=False)
            # An older or hand-copied table with no select column.
            pd.DataFrame([{"M": 3, "us": 4.0, "kernel": "hand_copied"}]).to_csv(
                unstamped, index=False
            )
            with (
                mock.patch.object(
                    AITER_CONFIG, "_shape_dedup_keys", return_value=["M"]
                ),
                mock.patch("tempfile.gettempdir", return_value=tmp),
            ):
                merged_path = _overlay_power_tables(
                    AITER_CONFIGS,
                    latency_file,
                    [stamped, unstamped],
                    "tuned_demo",
                    None,
                )
                only_unstamped = _overlay_power_tables(
                    AITER_CONFIGS, latency_file, [unstamped], "tuned_demo", None
                )
            merged = pd.read_csv(merged_path).sort_values("M")
        self.assertEqual(list(merged["kernel"]), ["cool", "shipped_2", "shipped_3"])
        self.assertEqual(only_unstamped, latency_file)

    def test_prebuild_config_load_does_not_import_aiter_package(self):
        import subprocess

        script = """
import os
import sys

sys.path.insert(0, os.path.join(os.environ["AITER_REPO"], "aiter"))
os.environ["AITER_SELECT"] = "power"
from jit import core

assert "aiter" not in sys.modules
path = core.AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE
assert path
assert "aiter" not in sys.modules
"""
        env = os.environ.copy()
        env["AITER_REPO"] = str(REPO_ROOT)
        env["AITER_SELECT"] = "power"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_power_sample_of_a_large_gemm_tracks_a_busy_gpu(self):
        """Real sensor: the tuner's sample must see the kernel, not idle time."""
        import time

        import torch

        from aiter.utility import package_power
        from aiter.utility.selection_policy import POWER

        if not torch.cuda.is_available():
            self.skipTest("needs a GPU")
        device = torch.cuda.current_device()
        path = package_power.power_input_path(device)
        if path is None or package_power._read_watts(path) is None:
            self.skipTest("package power sensor (power1_input) is not readable")

        idle = []
        deadline = time.perf_counter() + 0.4
        while time.perf_counter() < deadline:
            watts = package_power._read_watts(path)
            if watts is not None:
                idle.append(watts)
            time.sleep(0.005)
        idle_w = sum(idle) / len(idle)

        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        back_to_back = package_power.measure_package_watts(
            lambda: torch.mm(a, b), device_index=device
        )
        timed_kwargs = {"num_warmup": 2, "num_iters": 101, "num_rotate_args": 0}
        tuner_sample = POWER.measure_timed(
            torch.mm, (a, b), timed_kwargs, device_index=device
        )
        print(
            f"\npower check: idle={idle_w:.0f} W back_to_back={back_to_back:.0f} W "
            f"tuner={tuner_sample:.0f} W"
        )
        self.assertGreater(tuner_sample, 1.5 * idle_w)
        self.assertGreater(tuner_sample, 0.85 * back_to_back)

    def test_prebuild_duplicate_shapes_resolve_without_aiter_package(self):
        import subprocess
        import tempfile

        script = """
import os
import sys

sys.path.insert(0, os.path.join(os.environ["AITER_REPO"], "aiter"))
from jit import core

try:
    core.AITER_CONFIGS.AITER_CONFIG_BF16_BATCHED_GEMM_FILE
except RuntimeError as exc:
    assert "lowest 'us'" in str(exc), exc
else:
    raise AssertionError("duplicate shapes were not reported")
assert "aiter" not in sys.modules
"""
        header = "gfx,cu_num,B,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
        rows = [
            "gfx942,304,16,1,1280,8192,78,0,111.9,slow,3.0,3000.0,0.0",
            "gfx942,304,16,1,1280,8192,29,0,90.5,fast,3.5,3500.0,0.0",
        ]
        for select in ("latency", "power"):
            with tempfile.TemporaryDirectory() as tmp:
                paths = []
                for index, row in enumerate(rows):
                    path = os.path.join(tmp, f"bf16_tuned_batched_gemm_{index}.csv")
                    with open(path, "w", encoding="ascii") as handle:
                        handle.write(f"{header}\n{row}\n")
                    paths.append(path)
                env = os.environ.copy()
                env["AITER_REPO"] = str(REPO_ROOT)
                env["AITER_SELECT"] = select
                env["AITER_CONFIG_BF16_BATCHED_GEMM"] = os.pathsep.join(paths)
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    env=env,
                    capture_output=True,
                    text=True,
                    cwd=str(REPO_ROOT),
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, select + result.stderr + result.stdout
                )
                with open(paths[0], encoding="ascii") as handle:
                    self.assertNotIn("slow", handle.read())


if __name__ == "__main__":
    unittest.main()
