# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import contextlib
import io
import os
import sys
import unittest
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
            )
        ]
        calls = []

        def run_candidate(_row, candidate, _args):
            calls.append(candidate["kernelName2"])
            candidate["us"] = float(len(calls))
            return candidate["us"]

        tuner._candidate_rows = mock.Mock(return_value=full_pairs)
        tuner._run_candidate = mock.Mock(side_effect=run_candidate)
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
            ],
        )
        self.assertEqual(result["kernelName2"], TestMxfp4StagedCandidatePlan.G2_BARE)


if __name__ == "__main__":
    unittest.main()
