# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import contextlib
import io
import os
import sys
import unittest
import weakref
from collections.abc import Callable
from typing import Any
from unittest import mock

from csrc.ck_gemm_moe_2stages_codegen.mxfp4_staged_search import (
    AccumulationMode,
    GEMM1CandidateIdentity,
    GEMM1ScreeningKey,
    GEMM2ExecutionIdentity,
    GEMM2ScreeningKey,
    PairKey,
    PipelineEquivalenceKey,
    build_staged_candidate_plan,
    resolve_search_config,
)


class TestMxfp4SearchConfig(unittest.TestCase):

    def test_full_search_is_the_default(self):
        config = resolve_search_config(
            search_mode="full",
            explicit_screen_topk=None,
            gfx="gfx950",
        )

        self.assertEqual(config.mode, "full")
        self.assertEqual(config.screen_topk, 2)

    def test_staged_search_accepts_any_positive_screening_topk(self):
        for screen_topk in (1, 2, 7):
            with self.subTest(screen_topk=screen_topk):
                config = resolve_search_config(
                    search_mode="staged",
                    explicit_screen_topk=screen_topk,
                    gfx="gfx950",
                )

                self.assertEqual(config.mode, "staged")
                self.assertEqual(config.screen_topk, screen_topk)

    def test_explicit_screening_topk_requires_staged_search(self):
        with self.assertRaisesRegex(
            ValueError, "--mxfp4-screen-topk requires --mxfp4-search staged"
        ):
            resolve_search_config(
                search_mode="full",
                explicit_screen_topk=2,
                gfx="gfx950",
            )

    def test_screening_topk_must_be_positive(self):
        for screen_topk in (0, -1):
            with self.subTest(screen_topk=screen_topk):
                with self.assertRaisesRegex(
                    ValueError, "--mxfp4-screen-topk must be a positive integer"
                ):
                    resolve_search_config(
                        search_mode="staged",
                        explicit_screen_topk=screen_topk,
                        gfx="gfx950",
                    )

    def test_mxfp4_search_rejects_unsupported_devices(self):
        with self.assertRaisesRegex(
            ValueError, "--mxfp4-flydsl is only supported on gfx950"
        ):
            resolve_search_config(
                search_mode="staged",
                explicit_screen_topk=None,
                gfx="gfx942",
            )


class TestMxfp4StagedCandidatePlan(unittest.TestCase):

    G1_A = "flydsl_mxmoe_g1_a4w4_128x128x256"
    G1_B = "flydsl_mxmoe_g1_a4w4_128x256x256_xcd2"
    G1_BM32 = "flydsl_mxmoe_g1_a4w4_32x128x256_nt"
    G2_BARE = "flydsl_mxmoe_g2_a4w4_128x256x256"
    G2_F4OUT = "flydsl_mxmoe_g2_a4w4_128x256x256_f4out"
    G2_CSHUFFLE = "flydsl_mxmoe_g2_a4w4_128x256x256_cshuffle"
    G2_CSHUFFLE_BM32 = "flydsl_mxmoe_g2_a4w4_32x256x256_cshuffle"
    G2_V2_ATOMIC_A = "flydsl_moe2_layout_afp4_wfp4_bf16_" "t128x128x128_atomic_sbm128"
    G2_V2_ATOMIC_B = (
        "flydsl_moe2_layout_afp4_wfp4_bf16_" "t128x256x128_atomic_persist_nt_sbm128"
    )

    @staticmethod
    def _row(*, model_dim=7168, inter_dim=512, expert=257):
        return {
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "expert": expert,
            "act_type": "ActivationType.Silu",
        }

    @staticmethod
    def _pair(gemm1, gemm2, *, block_m=128):
        return {
            "block_m": block_m,
            "kernelName1": gemm1,
            "kernelName2": gemm2,
        }

    def _full_pairs(self):
        gemm2 = (
            self.G2_BARE,
            self.G2_F4OUT,
            self.G2_CSHUFFLE,
            self.G2_V2_ATOMIC_A,
            self.G2_V2_ATOMIC_B,
        )
        return [
            self._pair(gemm1, g2) for gemm1 in (self.G1_A, self.G1_B) for g2 in gemm2
        ]

    def test_disabled_intermediate_canonicalizes_f4out_to_bare(self):
        plan = build_staged_candidate_plan(
            self._row(),
            self._full_pairs(),
            mxfp4_intermediate=False,
        )

        native_gemm2 = {
            pair.key.gemm2.kernel_name
            for pair in plan.pairs
            if pair.key.gemm2.kernel_name.startswith("flydsl_mxmoe_g2_")
        }
        self.assertEqual(native_gemm2, {self.G2_BARE, self.G2_CSHUFFLE})

    def test_enabled_eligible_intermediate_uses_f4out_representative(self):
        plan = build_staged_candidate_plan(
            self._row(),
            self._full_pairs(),
            mxfp4_intermediate=True,
        )

        native_gemm2 = {
            pair.key.gemm2.kernel_name
            for pair in plan.pairs
            if pair.key.gemm2.kernel_name.startswith("flydsl_mxmoe_g2_")
        }
        self.assertEqual(native_gemm2, {self.G2_F4OUT})

    def test_enabled_ineligible_intermediate_keeps_cshuffle_distinct(self):
        plan = build_staged_candidate_plan(
            self._row(expert=256),
            self._full_pairs(),
            mxfp4_intermediate=True,
        )

        native_gemm2 = {
            pair.key.gemm2.kernel_name
            for pair in plan.pairs
            if pair.key.gemm2.kernel_name.startswith("flydsl_mxmoe_g2_")
        }
        self.assertEqual(native_gemm2, {self.G2_BARE, self.G2_CSHUFFLE})

    def test_enabled_intermediate_does_not_alias_ineligible_block_m(self):
        plan = build_staged_candidate_plan(
            self._row(),
            [self._pair(self.G1_BM32, self.G2_CSHUFFLE_BM32, block_m=32)],
            mxfp4_intermediate=True,
        )

        self.assertEqual(
            [pair.key.gemm2.kernel_name for pair in plan.pairs],
            [self.G2_CSHUFFLE_BM32],
        )

    def test_native_candidates_are_filtered_only_for_physical_dimensions(self):
        full_pairs = [
            self._pair(self.G1_A, self.G2_BARE),
            self._pair(self.G1_A, self.G2_V2_ATOMIC_A),
        ]

        plan = build_staged_candidate_plan(
            self._row(model_dim=640, inter_dim=384),
            full_pairs,
            mxfp4_intermediate=False,
        )

        self.assertEqual(
            [pair.key.gemm2.kernel_name for pair in plan.pairs],
            [self.G2_V2_ATOMIC_A],
        )

    def test_groups_are_the_exact_projection_of_canonical_full_pairs(self):
        plan = build_staged_candidate_plan(
            self._row(),
            self._full_pairs(),
            mxfp4_intermediate=False,
        )

        groups = {group.key: group for group in plan.pipeline_groups}
        nonatomic_key = PipelineEquivalenceKey(
            block_m=128,
            accumulation_mode=AccumulationMode.NON_ATOMIC,
        )
        atomic_key = PipelineEquivalenceKey(
            block_m=128,
            accumulation_mode=AccumulationMode.ATOMIC,
        )
        self.assertEqual(set(groups), {nonatomic_key, atomic_key})
        self.assertEqual(
            [group.key for group in plan.gemm1_groups],
            [GEMM1ScreeningKey(block_m=128)],
        )
        self.assertEqual(
            {group.key for group in plan.gemm2_groups},
            {
                GEMM2ScreeningKey(128, AccumulationMode.NON_ATOMIC),
                GEMM2ScreeningKey(128, AccumulationMode.ATOMIC),
            },
        )

        nonatomic = groups[nonatomic_key]
        expected_nonatomic_pairs = {
            PairKey(
                GEMM1CandidateIdentity(gemm1),
                GEMM2ExecutionIdentity(gemm2),
            )
            for gemm1 in (self.G1_A, self.G1_B)
            for gemm2 in (self.G2_BARE, self.G2_CSHUFFLE)
        }
        self.assertEqual(
            {pair.key for pair in nonatomic.pairs}, expected_nonatomic_pairs
        )
        self.assertEqual(
            nonatomic.gemm1_screening_key,
            GEMM1ScreeningKey(block_m=128),
        )
        self.assertEqual(
            nonatomic.gemm2_screening_key,
            GEMM2ScreeningKey(
                block_m=128,
                accumulation_mode=AccumulationMode.NON_ATOMIC,
            ),
        )
        self.assertEqual(
            [candidate.identity for candidate in nonatomic.gemm1_candidates],
            [GEMM1CandidateIdentity(self.G1_A), GEMM1CandidateIdentity(self.G1_B)],
        )
        self.assertEqual(
            [candidate.identity for candidate in nonatomic.gemm2_candidates],
            [
                GEMM2ExecutionIdentity(self.G2_BARE),
                GEMM2ExecutionIdentity(self.G2_CSHUFFLE),
            ],
        )

        atomic = groups[atomic_key]
        expected_atomic_pairs = {
            PairKey(
                GEMM1CandidateIdentity(gemm1),
                GEMM2ExecutionIdentity(gemm2),
            )
            for gemm1 in (self.G1_A, self.G1_B)
            for gemm2 in (self.G2_V2_ATOMIC_A, self.G2_V2_ATOMIC_B)
        }
        self.assertEqual({pair.key for pair in atomic.pairs}, expected_atomic_pairs)

    def test_pair_identity_keeps_kernel_identity_beyond_the_group_key(self):
        plan = build_staged_candidate_plan(
            self._row(),
            self._full_pairs(),
            mxfp4_intermediate=False,
        )

        self.assertEqual(len(plan.pairs), 8)
        self.assertEqual(len({pair.key for pair in plan.pairs}), 8)

    def test_input_quantization_contract_is_checked_before_grouping(self):
        invalid_gemm1 = "flydsl_mxmoe_g1_a4w4_32x128x256_f16in_nt"
        invalid_pair = self._pair(
            invalid_gemm1,
            "flydsl_mxmoe_g2_a4w4_32x256x256_atomic",
            block_m=32,
        )

        with self.assertRaisesRegex(
            ValueError, "violates the derived input quantization contract"
        ):
            build_staged_candidate_plan(
                self._row(),
                [invalid_pair],
                mxfp4_intermediate=False,
            )

    def test_gemm1_fixed_contract_is_checked_before_grouping(self):
        invalid_gemm1_names = {
            "a_dtype": "flydsl_mxmoe_g1_a8w4_128x128x256",
            "out_dtype": "flydsl_mxmoe_g1_a4w4_128x128x256_fp8out",
            "activation": "flydsl_mxmoe_g1_a4w4_128x128x256_situv2",
            "gate layout": "flydsl_mxmoe_g1_a4w4_128x128x256_il",
            "bias": "flydsl_mxmoe_g1_a4w4_128x128x256_bias",
        }
        for mismatch, gemm1 in invalid_gemm1_names.items():
            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(ValueError, mismatch):
                    build_staged_candidate_plan(
                        self._row(),
                        [self._pair(gemm1, self.G2_BARE)],
                        mxfp4_intermediate=False,
                    )

    def test_layout_v2_fixed_contract_is_checked_before_grouping(self):
        invalid_gemm2_names = {
            "a_dtype": (
                "flydsl_moe2_layout_afp8_wfp4_bf16_" "t128x128x128_atomic_sbm128"
            ),
            "b_dtype": (
                "flydsl_moe2_layout_afp4_wfp8_bf16_" "t128x128x128_atomic_sbm128"
            ),
            "out_dtype": (
                "flydsl_moe2_layout_afp4_wfp4_fp32_" "t128x128x128_atomic_sbm128"
            ),
            "sort block_m": (
                "flydsl_moe2_layout_afp4_wfp4_bf16_" "t128x128x128_atomic_sbm64"
            ),
        }
        for mismatch, gemm2 in invalid_gemm2_names.items():
            with self.subTest(mismatch=mismatch):
                with self.assertRaisesRegex(ValueError, mismatch):
                    build_staged_candidate_plan(
                        self._row(),
                        [self._pair(self.G1_A, gemm2)],
                        mxfp4_intermediate=False,
                    )


class TestMxfp4TunerSearchSeam(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        cls.tune_module = gemm_moe_tune
        cls.tuner_type = gemm_moe_tune.Mxfp4FlydslTuner

    def _parser_tuner(self):
        return self.tuner_type("test", [], [], "test")

    def test_parser_materializes_search_defaults_before_tuning(self):
        tuner = self._parser_tuner()
        with (
            mock.patch.object(sys, "argv", ["tune", "--mxfp4-flydsl"]),
            mock.patch.object(self.tune_module, "get_gfx", return_value="gfx950"),
        ):
            args = tuner.parse_args()

        self.assertEqual(args.mxfp4_search, "full")
        self.assertEqual(args.screen_topk, 2)

    def test_parser_accepts_explicit_staged_search_and_screening_topk(self):
        tuner = self._parser_tuner()
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "tune",
                    "--mxfp4-flydsl",
                    "--mxfp4-search",
                    "staged",
                    "--mxfp4-screen-topk",
                    "3",
                ],
            ),
            mock.patch.object(self.tune_module, "get_gfx", return_value="gfx950"),
        ):
            args = tuner.parse_args()

        self.assertEqual(args.mxfp4_search, "staged")
        self.assertEqual(args.screen_topk, 3)

    def test_parser_rejects_explicit_screening_topk_in_full_mode(self):
        tuner = self._parser_tuner()
        with (
            mock.patch.object(
                sys,
                "argv",
                ["tune", "--mxfp4-flydsl", "--mxfp4-screen-topk", "3"],
            ),
            mock.patch.object(self.tune_module, "get_gfx", return_value="gfx950"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                tuner.parse_args()

    def test_parser_rejects_unsupported_device_before_tuning(self):
        tuner = self._parser_tuner()
        with (
            mock.patch.object(sys, "argv", ["tune", "--mxfp4-flydsl"]),
            mock.patch.object(self.tune_module, "get_gfx", return_value="gfx942"),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit):
                tuner.parse_args()

    def test_default_and_explicit_full_search_have_identical_behavior(self):
        def run(search_mode):
            tuner = self.tuner_type.__new__(self.tuner_type)
            candidates = [
                {"kernelName1": "g1-a", "kernelName2": "g2", "us": 0},
                {"kernelName1": "g1-a", "kernelName2": "g2_f4out", "us": 0},
                {"kernelName1": "g1-b", "kernelName2": "g2", "us": 0},
            ]
            calls = []

            def candidate_rows(_row):
                return [dict(candidate) for candidate in candidates]

            def run_candidate(_row, candidate, _args):
                identity = (candidate["kernelName1"], candidate["kernelName2"])
                calls.append(identity)
                if identity == ("g1-a", "g2_f4out"):
                    raise RuntimeError("expected failure")
                candidate["us"] = 7.0 if identity[0] == "g1-a" else 3.0
                return candidate["us"]

            tuner._candidate_rows = mock.Mock(side_effect=candidate_rows)
            tuner._run_candidate = mock.Mock(side_effect=run_candidate)
            args = argparse.Namespace(timeout=0)
            if search_mode is not None:
                args.mxfp4_search = search_mode
            with contextlib.redirect_stdout(io.StringIO()):
                result = tuner._tune_one_shape({"token": 1, "inter_dim": 512}, args)
            return result, calls

        default_result, default_calls = run(None)
        explicit_result, explicit_calls = run("full")

        self.assertEqual(explicit_calls, default_calls)
        self.assertEqual(explicit_result, default_result)
        self.assertEqual(explicit_result["kernelName1"], "g1-b")

    def test_staged_planning_uses_candidate_rows_without_executing_pairs(self):
        tuner = self.tuner_type.__new__(self.tuner_type)
        full_pairs = [
            {
                "block_m": 128,
                "kernelName1": TestMxfp4StagedCandidatePlan.G1_A,
                "kernelName2": TestMxfp4StagedCandidatePlan.G2_BARE,
            },
            {
                "block_m": 128,
                "kernelName1": TestMxfp4StagedCandidatePlan.G1_A,
                "kernelName2": TestMxfp4StagedCandidatePlan.G2_F4OUT,
            },
        ]
        tuner._candidate_rows = mock.Mock(return_value=full_pairs)
        tuner._run_candidate = mock.Mock(side_effect=AssertionError("must not run"))

        with mock.patch.dict(os.environ, {"AITER_MXFP4_INTERMEDIATE": "0"}):
            plan = tuner._staged_candidate_plan(TestMxfp4StagedCandidatePlan._row())

        tuner._candidate_rows.assert_called_once()
        tuner._run_candidate.assert_not_called()
        self.assertEqual(
            [pair.key.gemm2.kernel_name for pair in plan.pairs],
            [TestMxfp4StagedCandidatePlan.G2_BARE],
        )

    def test_staged_search_executes_only_planned_representatives(self):
        tuner = self.tuner_type.__new__(self.tuner_type)
        full_pairs = [
            {
                "block_m": 128,
                "kernelName1": TestMxfp4StagedCandidatePlan.G1_A,
                "kernelName2": gemm2,
                "us": 0,
            }
            for gemm2 in (
                TestMxfp4StagedCandidatePlan.G2_BARE,
                TestMxfp4StagedCandidatePlan.G2_F4OUT,
                TestMxfp4StagedCandidatePlan.G2_CSHUFFLE,
                TestMxfp4StagedCandidatePlan.G2_V2_ATOMIC_A,
            )
        ]
        calls = []

        def run_candidate(_row, candidate, _args):
            calls.append(candidate["kernelName2"])
            candidate["us"] = float(len(calls))
            return candidate["us"]

        tuner._candidate_rows = mock.Mock(return_value=full_pairs)
        tuner._run_candidate = mock.Mock(side_effect=run_candidate)
        tuner._check_reference_consumer = mock.Mock(return_value=True)
        tuner._check_reference_pair = mock.Mock(return_value=True)
        tuner._screen_gemm1 = mock.Mock(return_value=1.0)
        tuner._screen_gemm2 = mock.Mock(return_value=1.0)
        row = {
            **TestMxfp4StagedCandidatePlan._row(),
            "token": 1,
        }
        args = argparse.Namespace(timeout=0, mxfp4_search="staged")

        with (
            mock.patch.dict(os.environ, {"AITER_MXFP4_INTERMEDIATE": "0"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = tuner._tune_one_shape(row, args)

        self.assertEqual(
            calls,
            [
                TestMxfp4StagedCandidatePlan.G2_BARE,
                TestMxfp4StagedCandidatePlan.G2_CSHUFFLE,
                TestMxfp4StagedCandidatePlan.G2_V2_ATOMIC_A,
            ],
        )
        self.assertEqual(result["kernelName2"], TestMxfp4StagedCandidatePlan.G2_BARE)


class TestMxfp4StagedWinner(unittest.TestCase):
    """Exercise search policy at the shape boundary with fake GPU outcomes."""

    G1_A = TestMxfp4StagedCandidatePlan.G1_A
    G1_B = TestMxfp4StagedCandidatePlan.G1_B
    G1_C = "flydsl_mxmoe_g1_a4w4_128x256x256"
    G2_REFERENCE = TestMxfp4StagedCandidatePlan.G2_V2_ATOMIC_A
    G2_ATOMIC = "flydsl_mxmoe_g2_a4w4_128x256x256_atomic"
    G2_BARE = TestMxfp4StagedCandidatePlan.G2_BARE
    G2_REDUCE = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x128x128_reduce_sbm128"

    def _run_shape(
        self,
        *,
        screen_topk: int = 1,
        configure: Callable[[Any], None] | None = None,
    ) -> tuple[dict[str, Any], Any, list[tuple[str, str]]]:
        from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner

        tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
        gemm1 = (self.G1_B, self.G1_A, self.G1_C)
        gemm2 = (self.G2_REFERENCE, self.G2_ATOMIC, self.G2_BARE, self.G2_REDUCE)
        pairs = [
            {
                "block_m": 128,
                "kernelName1": kn1,
                "kernelName2": kn2,
                "us1": 0,
                "us2": 0,
                "us": 0,
            }
            for kn1 in gemm1
            for kn2 in gemm2
        ]
        stage1_us = {self.G1_B: 1.0, self.G1_A: 3.0, self.G1_C: 2.0}
        stage2_us = {
            self.G2_REFERENCE: 11.0,
            self.G2_ATOMIC: 9.0,
            self.G2_BARE: 0.5,
            self.G2_REDUCE: 0.2,
        }
        pipeline_us = {
            (self.G1_B, self.G2_ATOMIC): 3.0,
            (self.G1_B, self.G2_REDUCE): 7.0,
            (self.G1_A, self.G2_BARE): 0.1,
        }
        validated = []

        def validate(
            _row: dict[str, Any], candidate: dict[str, Any], _args: argparse.Namespace
        ) -> float:
            pair = (candidate["kernelName1"], candidate["kernelName2"])
            validated.append(pair)
            us = pipeline_us.get(pair, 20.0)
            candidate.update(us1=us, us2=0, us=us)
            return us

        tuner._candidate_rows = mock.Mock(return_value=pairs)
        tuner._check_reference_consumer = mock.Mock(return_value=True)
        tuner._check_reference_pair = mock.Mock(return_value=True)
        tuner._screen_gemm1 = mock.Mock(
            side_effect=lambda _row, kn1, _args: stage1_us[kn1]
        )
        tuner._screen_gemm2 = mock.Mock(
            side_effect=lambda _row, _kn1, kn2, _args: stage2_us[kn2]
        )
        tuner._run_candidate = mock.Mock(side_effect=validate)
        if configure is not None:
            configure(tuner)
        row = {
            **TestMxfp4StagedCandidatePlan._row(model_dim=256, inter_dim=256, expert=2),
            "token": 1,
            "topk": 1,
        }
        args = argparse.Namespace(
            timeout=0,
            mxfp4_search="staged",
            screen_topk=screen_topk,
            warmup=3,
            iters=5,
            errRatio=0.01,
        )
        with (
            mock.patch.dict(os.environ, {"AITER_MXFP4_INTERMEDIATE": "0"}),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = tuner._tune_one_shape(row, args)
        return result, tuner, validated

    def test_topk_one_validates_one_pair_per_accumulation_mode(self) -> None:
        result, tuner, validated = self._run_shape(screen_topk=1)

        self.assertEqual(
            validated,
            [(self.G1_B, self.G2_ATOMIC), (self.G1_B, self.G2_REDUCE)],
        )
        self.assertEqual(tuner._screen_gemm1.call_count, 3)
        self.assertEqual(tuner._screen_gemm2.call_count, 4)
        self.assertEqual(result["kernelName1"], self.G1_B)
        self.assertEqual(result["kernelName2"], self.G2_ATOMIC)
        self.assertEqual((result["us1"], result["us2"], result["us"]), (3.0, 0, 3.0))

    def test_topk_two_validates_the_retained_cartesian_products(self) -> None:
        result, tuner, validated = self._run_shape(screen_topk=2)

        self.assertEqual(
            set(validated),
            {
                (self.G1_B, self.G2_REFERENCE),
                (self.G1_B, self.G2_ATOMIC),
                (self.G1_C, self.G2_REFERENCE),
                (self.G1_C, self.G2_ATOMIC),
                (self.G1_B, self.G2_BARE),
                (self.G1_B, self.G2_REDUCE),
                (self.G1_C, self.G2_BARE),
                (self.G1_C, self.G2_REDUCE),
            },
        )
        self.assertEqual(len(validated), 8)
        self.assertEqual(tuner._screen_gemm1.call_count, 3)
        self.assertEqual(result["us"], 3.0)

    def test_screening_topk_larger_than_available_uses_all_successful_candidates(
        self,
    ) -> None:
        result, _, validated = self._run_shape(screen_topk=7)

        self.assertEqual(len(validated), 12)
        self.assertEqual(len(set(validated)), 12)
        self.assertEqual(
            (result["kernelName1"], result["kernelName2"]), (self.G1_A, self.G2_BARE)
        )
        self.assertEqual((result["us1"], result["us2"], result["us"]), (0.1, 0, 0.1))

    def test_exact_screening_ties_retain_only_topk_in_original_order(self) -> None:
        def configure(tuner: Any) -> None:
            tuner._screen_gemm1.side_effect = None
            tuner._screen_gemm1.return_value = 1.0
            tuner._screen_gemm2.side_effect = None
            tuner._screen_gemm2.return_value = 1.0

        _, _, validated = self._run_shape(configure=configure)

        self.assertEqual(
            validated, [(self.G1_B, self.G2_REFERENCE), (self.G1_B, self.G2_BARE)]
        )

    def test_only_finite_positive_screening_scores_reach_validation(self) -> None:
        def configure(tuner: Any) -> None:
            tuner._screen_gemm1.side_effect = lambda _row, kn1, _args: {
                self.G1_B: 1.0,
                self.G1_A: float("inf"),
                self.G1_C: float("nan"),
            }[kn1]
            tuner._screen_gemm2.side_effect = lambda _row, _kn1, kn2, _args: {
                self.G2_REFERENCE: 0.0,
                self.G2_ATOMIC: 9.0,
                self.G2_BARE: -1.0,
                self.G2_REDUCE: 0.2,
            }[kn2]

        _, _, validated = self._run_shape(screen_topk=7, configure=configure)

        self.assertEqual(
            validated, [(self.G1_B, self.G2_ATOMIC), (self.G1_B, self.G2_REDUCE)]
        )

    def test_each_bm_bootstraps_and_shares_its_ranking_before_shape_winner_selection(
        self,
    ) -> None:
        expected_producers = {}

        def configure(tuner: Any) -> None:
            pairs = []
            for bm in (16, 32, 64, 128):
                suffix = "_f16in_nt" if bm == 16 else ""
                preferred = f"flydsl_mxmoe_g1_a4w4_{bm}x128x256{suffix}"
                other = f"flydsl_mxmoe_g1_a4w4_{bm}x256x256{suffix}_xcd2"
                expected_producers[bm] = preferred
                for kn1 in (other, preferred):
                    for mode in ("atomic", "reduce"):
                        pairs.append(
                            {
                                "block_m": bm,
                                "kernelName1": kn1,
                                "kernelName2": f"flydsl_moe2_layout_afp4_wfp4_bf16_t{bm}x128x128_{mode}_sbm{bm}",
                            }
                        )
            tuner._candidate_rows.return_value = pairs
            tuner._screen_gemm1.side_effect = None
            tuner._screen_gemm1.return_value = 1.0
            tuner._screen_gemm2.side_effect = None
            tuner._screen_gemm2.return_value = 1.0
            original_validate = tuner._run_candidate.side_effect

            def validate(
                row: dict[str, Any], candidate: dict[str, Any], args: argparse.Namespace
            ) -> float:
                original_validate(row, candidate, args)
                us = {16: 8.0, 32: 7.0, 64: 1.0, 128: 3.0}[candidate["block_m"]]
                candidate.update(us1=us, us2=0, us=us)
                return us

            tuner._run_candidate.side_effect = validate

        result, tuner, validated = self._run_shape(configure=configure)

        self.assertEqual(tuner._screen_gemm1.call_count, 8)
        self.assertEqual(tuner._screen_gemm2.call_count, 8)
        self.assertEqual(len(validated), 8)
        self.assertEqual(
            [call.args[1] for call in tuner._check_reference_pair.call_args_list],
            list(expected_producers.values()),
        )
        self.assertEqual(result["block_m"], 64)
        self.assertEqual(result["us"], 1.0)

    def test_bootstrap_prefers_reference_pair_without_ranking_it(self) -> None:
        result, tuner, validated = self._run_shape()

        self.assertEqual(tuner._check_reference_consumer.call_count, 1)
        self.assertEqual(
            tuner._check_reference_consumer.call_args.args[1], self.G2_REFERENCE
        )
        self.assertEqual(tuner._check_reference_pair.call_count, 1)
        self.assertEqual(
            tuner._check_reference_pair.call_args.args[1:3],
            (self.G1_A, self.G2_REFERENCE),
        )
        self.assertEqual(
            {call.args[1] for call in tuner._screen_gemm2.call_args_list},
            {self.G1_A},
        )
        self.assertNotIn((self.G1_A, self.G2_REFERENCE), validated)
        self.assertEqual(result["kernelName1"], self.G1_B)

    def test_failed_preferred_consumer_tries_remaining_atomic_consumers_in_order(
        self,
    ) -> None:
        first = TestMxfp4StagedCandidatePlan.G2_V2_ATOMIC_B
        second = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_atomic_sbm128"

        def configure(tuner: Any) -> None:
            pairs = tuner._candidate_rows.return_value
            alternatives = [
                {**pair, "kernelName2": kn2}
                for pair in pairs
                if pair["kernelName2"] == self.G2_REFERENCE
                for kn2 in (first, second)
            ]
            tuner._candidate_rows.return_value = alternatives + pairs
            tuner._check_reference_consumer.side_effect = [
                RuntimeError("preferred consumer failed its self-test"),
                RuntimeError("first alternative failed its self-test"),
                True,
            ]
            tuner._screen_gemm2.side_effect = None
            tuner._screen_gemm2.return_value = 1.0

        _, tuner, _ = self._run_shape(configure=configure)

        self.assertEqual(
            [call.args[1] for call in tuner._check_reference_consumer.call_args_list],
            [self.G2_REFERENCE, first, second],
        )
        self.assertEqual(
            tuner._check_reference_pair.call_args.args[1:3], (self.G1_A, second)
        )

    def test_gemm1_screening_times_only_the_kernel_for_each_input_quantization_mode(
        self,
    ) -> None:
        import torch
        from aiter import dtypes
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        data = {
            "input": torch.empty((1, 256), dtype=dtypes.bf16, device="cpu"),
            "w1_a16": torch.empty((2, 512, 128), dtype=torch.uint8, device="cpu"),
            "w2_a16": torch.empty((2, 256, 128), dtype=torch.uint8, device="cpu"),
            "w1s_a16": torch.empty((1024, 8), dtype=torch.uint8, device="cpu"),
            "topk_ids": torch.zeros((1, 1), dtype=torch.int32, device="cpu"),
            "topk_weights": torch.ones((1, 1), device="cpu"),
        }
        sorting = (
            torch.zeros(256, dtype=torch.int32, device="cpu"),
            torch.ones(256, device="cpu"),
            torch.zeros(16, dtype=torch.int32, device="cpu"),
            torch.tensor([16, 1], dtype=torch.int32, device="cpu"),
            torch.empty(0, dtype=dtypes.bf16, device="cpu"),
            torch.zeros(256, dtype=torch.int32, device="cpu"),
            torch.zeros(1, dtype=torch.int32, device="cpu"),
        )
        intermediate = (
            torch.empty((256, 128), dtype=torch.uint8, device="cpu"),
            torch.empty((512, 8), dtype=torch.uint8, device="cpu"),
        )
        packed = torch.empty((1, 128), dtype=dtypes.fp4x2, device="cpu")
        scale = torch.empty((256, 8), dtype=torch.uint8, device="cpu")

        for bm in (16, 32, 64, 128):
            with self.subTest(block_m=bm):
                timed = False
                events = []

                def setup(value: Any) -> Callable[..., Any]:
                    def run(*_args: Any, **_kwargs: Any) -> Any:
                        self.assertFalse(
                            timed, "preparation entered the timed callable"
                        )
                        return value

                    return run

                def kernel(**kwargs: Any) -> None:
                    self.assertTrue(timed)
                    self.assertEqual(kwargs["inline_quant"], bm == 16)
                    self.assertEqual(kwargs["native_scale_layout"], bm == 16)
                    self.assertIs(
                        kwargs["hidden_states"], data["input"] if bm == 16 else packed
                    )
                    self.assertIs(kwargs["inter_sorted_quant"], intermediate[0])
                    events.append("gemm1")

                def measure(
                    call: Callable[[], Any], *, num_warmup: int, num_iters: int
                ) -> tuple[None, float]:
                    nonlocal timed
                    self.assertEqual((num_warmup, num_iters), (3, 5))
                    timed = True
                    try:
                        call()
                    finally:
                        timed = False
                    return None, 1.0

                def configure(tuner: Any) -> None:
                    kn1 = f"flydsl_mxmoe_g1_a4w4_{bm}x128x256"
                    if bm == 16:
                        kn1 += "_f16in_nt"
                    kn2 = f"flydsl_moe2_layout_afp4_wfp4_bf16_t{bm}x128x128_atomic_sbm{bm}"
                    tuner._candidate_rows.return_value = [
                        {"block_m": bm, "kernelName1": kn1, "kernelName2": kn2}
                    ]
                    tuner._prepare_case = mock.Mock(side_effect=setup(data))
                    tuner._screen_gemm1 = (
                        gemm_moe_tune.Mxfp4FlydslTuner._screen_gemm1.__get__(tuner)
                    )
                    tuner._screen_gemm2 = mock.Mock(return_value=1.0)

                with (
                    mock.patch.object(
                        gemm_moe_tune, "moe_sorting", side_effect=setup(sorting)
                    ),
                    mock.patch.object(
                        gemm_moe_tune.aiter,
                        "fused_dynamic_mxfp4_quant_moe_sort",
                        side_effect=setup((packed, scale)),
                    ) as prequant,
                    mock.patch.object(
                        gemm_moe_tune,
                        "_mxfp4_a4w4_stage1_fw",
                        side_effect=setup(intermediate),
                    ),
                    mock.patch(
                        "aiter.ops.flydsl.mxfp4_gemm1_kernels.flydsl_mxfp4_gemm1",
                        side_effect=kernel,
                    ),
                    mock.patch("aiter.test_common.run_perftest", side_effect=measure),
                ):
                    self._run_shape(configure=configure)

                self.assertEqual(events, ["gemm1"])
                self.assertEqual(prequant.call_count, 0 if bm == 16 else 1)

    def test_gemm2_screening_owns_each_intermediate_and_times_through_final_output(
        self,
    ) -> None:
        import torch
        from aiter import dtypes
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        data = {
            "input": torch.empty((1, 256), dtype=dtypes.bf16, device="cpu"),
            "w1_a16": torch.empty((2, 512, 128), dtype=torch.uint8, device="cpu"),
            "w2_a16": torch.empty((2, 256, 128), dtype=torch.uint8, device="cpu"),
            "w1s_a16": torch.empty((1024, 8), dtype=torch.uint8, device="cpu"),
            "w2s_a16": torch.empty((512, 8), dtype=torch.uint8, device="cpu"),
            "topk_ids": torch.zeros((1, 1), dtype=torch.int32, device="cpu"),
            "topk_weights": torch.ones((1, 1), device="cpu"),
        }
        timed = False
        intermediates: list[weakref.ReferenceType[torch.Tensor]] = []
        producers = []
        screened = []

        def sort(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
            self.assertFalse(timed)
            return (
                torch.zeros(256, dtype=torch.int32, device="cpu"),
                torch.ones(256, device="cpu"),
                torch.zeros(2, dtype=torch.int32, device="cpu"),
                torch.tensor([128, 1], dtype=torch.int32, device="cpu"),
                torch.full((1, 256), 3.0, dtype=dtypes.bf16, device="cpu"),
                torch.zeros(256, dtype=torch.int32, device="cpu"),
                torch.zeros(1, dtype=torch.int32, device="cpu"),
            )

        def producer(*_args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            self.assertFalse(timed)
            self.assertTrue(all(ref() is None for ref in intermediates))
            producers.append(kwargs["kernelName1"])
            payload = torch.ones((256, 128), dtype=torch.uint8, device="cpu")
            scale = torch.ones((512, 8), dtype=torch.uint8, device="cpu")
            intermediates.extend((weakref.ref(payload), weakref.ref(scale)))
            return payload, scale

        def consume(
            inter_q: torch.Tensor,
            _w1: Any,
            _w2: Any,
            _sti: Any,
            _sei: Any,
            _nvi: Any,
            out: torch.Tensor,
            _topk: int,
            **kwargs: Any,
        ) -> torch.Tensor:
            self.assertTrue(timed)
            self.assertIs(inter_q, intermediates[-2]())
            self.assertIs(kwargs["a2_scale"], intermediates[-1]())
            kn2 = kwargs["kernelName2"]
            if "atomic" in kn2:
                self.assertEqual(torch.count_nonzero(out).item(), 0)
            else:
                self.assertGreater(torch.count_nonzero(out).item(), 0)
            screened.append(kn2)
            out.fill_(7.0)
            return out

        def measure(
            call: Callable[[], torch.Tensor], *, num_warmup: int, num_iters: int
        ) -> tuple[torch.Tensor, float]:
            nonlocal timed
            self.assertEqual((num_warmup, num_iters), (3, 5))
            timed = True
            try:
                for _ in range(2):
                    out = call()
                    self.assertEqual(out.dtype, dtypes.bf16)
                    self.assertTrue(torch.all(out == 7.0).item())
            finally:
                timed = False
            return out, 1.0

        def configure(tuner: Any) -> None:
            tuner._prepare_case = mock.Mock(return_value=data)
            tuner._screen_gemm2 = gemm_moe_tune.Mxfp4FlydslTuner._screen_gemm2.__get__(
                tuner
            )

        with (
            mock.patch.object(gemm_moe_tune, "moe_sorting", new=sort),
            mock.patch.object(
                gemm_moe_tune.aiter,
                "fused_dynamic_mxfp4_quant_moe_sort",
                return_value=(
                    torch.empty((1, 128), dtype=dtypes.fp4x2, device="cpu"),
                    torch.empty((256, 8), dtype=torch.uint8, device="cpu"),
                ),
            ),
            mock.patch.object(gemm_moe_tune, "_mxfp4_a4w4_stage1_fw", new=producer),
            mock.patch.object(gemm_moe_tune, "_mxfp4_a4w4_stage2_fw", new=consume),
            mock.patch("aiter.test_common.run_perftest", new=measure),
        ):
            self._run_shape(configure=configure)

        self.assertEqual(producers, [self.G1_A] * 4)
        self.assertEqual(
            screened,
            [self.G2_REFERENCE] * 2
            + [self.G2_ATOMIC] * 2
            + [self.G2_BARE] * 2
            + [self.G2_REDUCE] * 2,
        )
        self.assertTrue(all(ref() is None for ref in intermediates))

    def test_bootstrap_self_tests_torch_layout_then_checks_producer_without_timing(
        self,
    ) -> None:
        import torch
        from aiter import dtypes
        from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

        output = torch.ones((1, 256), dtype=dtypes.bf16, device="cpu")
        ref1 = torch.ones((1, 1, 256), dtype=dtypes.bf16, device="cpu")
        data = {
            key: None
            for key in (
                "a1_qt",
                "a1_scale",
                "w1_qt",
                "w2_qt",
                "w1_scale",
                "w2_scale",
                "w1_a16",
                "w2_a16",
                "w1s_a16",
                "w2s_a16",
                "topk_ids",
                "topk_weights",
            )
        }
        data["input"] = output
        sti = torch.ones(256, dtype=torch.int32, device="cpu")
        sti[0] = 0
        sorting = (
            sti,
            torch.ones(256, device="cpu"),
            torch.zeros(2, dtype=torch.int32, device="cpu"),
            torch.tensor([128, 1], dtype=torch.int32, device="cpu"),
            output.clone(),
            torch.zeros(256, dtype=torch.int32, device="cpu"),
            torch.zeros(1, dtype=torch.int32, device="cpu"),
        )
        produced = torch.full((256, 128), 0xAA, dtype=torch.uint8, device="cpu")
        sources = []

        def consume(inter_q: torch.Tensor, *_args: Any, **kwargs: Any) -> torch.Tensor:
            if inter_q is produced:
                sources.append("producer")
            else:
                sources.append("torch")
                self.assertTrue(torch.all(inter_q[0] == 7).item())
                self.assertEqual(torch.count_nonzero(inter_q[1:]).item(), 0)
                scales = kwargs["a2_scale"].flatten()
                self.assertEqual(scales.numel(), 2048)
                self.assertEqual(
                    scales[[0, 64, 128, 192, 2, 66, 130, 194]].tolist(),
                    [120, 121, 122, 123, 124, 125, 126, 127],
                )
            return output

        def configure(tuner: Any) -> None:
            tuner._prepare_case = mock.Mock(return_value=data)
            tuner._check_reference_consumer = (
                gemm_moe_tune.Mxfp4FlydslTuner._check_reference_consumer.__get__(tuner)
            )
            tuner._check_reference_pair = (
                gemm_moe_tune.Mxfp4FlydslTuner._check_reference_pair.__get__(tuner)
            )
            tuner._screen_gemm1.side_effect = lambda _row, kn1, _args: (
                0.5 if kn1 == self.G1_A else 1.0
            )
            tuner._screen_gemm2.side_effect = lambda _row, _kn1, kn2, _args: (
                0.5 if kn2 == self.G2_REFERENCE else 1.0
            )

        with (
            mock.patch.object(gemm_moe_tune, "moe_sorting", return_value=sorting),
            mock.patch.object(
                gemm_moe_tune.FmoeTuner, "run_torch_moe_stage1", return_value=ref1
            ),
            mock.patch.object(
                gemm_moe_tune.FmoeTuner, "run_torch_moe_stage2", return_value=output
            ),
            mock.patch.object(
                gemm_moe_tune,
                "_mxfp4_a4w4_stage1_fw",
                return_value=(produced, produced),
            ) as producer,
            mock.patch.object(gemm_moe_tune, "_mxfp4_a4w4_stage2_fw", new=consume),
            mock.patch.object(
                gemm_moe_tune.aiter,
                "fused_dynamic_mxfp4_quant_moe_sort",
                return_value=(produced, produced),
            ),
            mock.patch(
                "csrc.ck_gemm_moe_2stages_codegen.mxfp4_v2_tune_utils.per_1x32_f4_quant",
                return_value=(
                    torch.full((1, 128), 7, dtype=torch.uint8, device="cpu"),
                    torch.arange(120, 128, dtype=torch.uint8, device="cpu").view(1, 8),
                ),
            ),
            mock.patch("aiter.test_common.run_perftest") as measure,
        ):
            _, _, validated = self._run_shape(configure=configure)

        self.assertEqual(sources, ["torch", "producer"])
        producer.assert_called_once()
        measure.assert_not_called()
        self.assertEqual(validated.count((self.G1_A, self.G2_REFERENCE)), 1)


if __name__ == "__main__":
    unittest.main()
