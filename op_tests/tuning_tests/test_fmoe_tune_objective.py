# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for the two-stage FMoE tuning objective.

Stage-1 FlyDSL kernels whose name ends in ``_fp4``/``_fp8`` fuse the
intermediate quantization that stage 2 consumes; every other stage-1 kernel
needs a separate quant+sort launch. The tuner must therefore (a) dedupe the two
paths independently, (b) charge the non-fused path for that extra launch, and
(c) minimize the adjusted Stage1 + Stage2 latency. These tests pin that
contract.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "csrc" / "ck_gemm_moe_2stages_codegen"))

from gemm_moe_tune import (
    FUSED_INTERMEDIATE_SUFFIXES,
    FmoeTuner,
    add_intermediate_quant_cost,
    dedupe_objective_candidates,
    select_fastest_stage_combination,
)


def _candidate(kernel_name, us, stage="stage1", block_m=32, flat=0):
    return {
        "stage": stage,
        "block_m": block_m,
        "flat": flat,
        "kernelName": kernel_name,
        "us": us,
    }


class TestDedupeObjectiveCandidates(unittest.TestCase):
    def test_fused_and_non_fused_paths_survive_independently(self):
        candidates = pd.DataFrame(
            [
                _candidate("flydsl_moe1_t32x32x256_fp4", 12.0),
                _candidate("flydsl_moe1_t32x64x256_fp4", 15.0),
                _candidate("flydsl_moe1_t32x128x256", 9.0),
                _candidate("flydsl_moe1_t32x256x256", 11.0),
            ]
        )

        kept = set(dedupe_objective_candidates(candidates)["kernelName"])

        # Fastest of each path survives, even though the non-fused kernel is
        # faster on raw us than either fused candidate.
        self.assertEqual(
            kept, {"flydsl_moe1_t32x32x256_fp4", "flydsl_moe1_t32x128x256"}
        )

    def test_dedupe_is_per_stage_and_block_m(self):
        candidates = pd.DataFrame(
            [
                _candidate("k_a_fp4", 12.0, block_m=32),
                _candidate("k_b_fp4", 10.0, block_m=64),
                _candidate("k_c_fp4", 14.0, stage="stage2"),
            ]
        )

        kept = dedupe_objective_candidates(candidates)

        # Different (stage, block_m) groups are independent, so nothing is
        # dropped here despite all three being fused kernels.
        self.assertEqual(len(kept), 3)

    def test_empty_frame_is_passed_through(self):
        empty = pd.DataFrame(columns=["stage", "block_m", "flat", "kernelName", "us"])
        self.assertTrue(dedupe_objective_candidates(empty).empty)

    def test_helper_column_is_not_leaked(self):
        candidates = pd.DataFrame([_candidate("k_a_fp4", 12.0)])
        self.assertNotIn("_is_fused", dedupe_objective_candidates(candidates).columns)


class TestAddIntermediateQuantCost(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame(
            [
                {"block_m": 32, "kernelName1": "flydsl_moe1_a_fp4", "us1": 12.0},
                {"block_m": 32, "kernelName1": "flydsl_moe1_b", "us1": 9.0},
                {"block_m": 64, "kernelName1": "flydsl_moe1_c", "us1": 20.0},
            ]
        )

    def test_only_non_fused_rows_are_charged(self):
        charged = add_intermediate_quant_cost(self._frame(), "_fp4", {32: 4.0, 64: 5.0})

        # Fused row untouched; non-fused rows pay their block_m's measured cost.
        self.assertAlmostEqual(charged.loc[0, "us1"], 12.0)
        self.assertAlmostEqual(charged.loc[1, "us1"], 13.0)
        self.assertAlmostEqual(charged.loc[2, "us1"], 25.0)

    def test_penalty_can_flip_the_winner(self):
        charged = add_intermediate_quant_cost(self._frame(), "_fp4", {32: 4.0})
        block32 = charged[charged["block_m"] == 32]
        winner = block32.sort_values("us1").iloc[0]["kernelName1"]

        # Non-fused wins on raw us (9.0 < 12.0) but loses once its separate
        # quant+sort launch is accounted for.
        self.assertEqual(winner, "flydsl_moe1_a_fp4")

    def test_fp8_suffix_is_honoured(self):
        frame = pd.DataFrame(
            [
                {"block_m": 32, "kernelName1": "flydsl_moe1_a_fp8", "us1": 12.0},
                {"block_m": 32, "kernelName1": "flydsl_moe1_b", "us1": 9.0},
            ]
        )
        charged = add_intermediate_quant_cost(frame, "_fp8", {32: 4.0})

        self.assertAlmostEqual(charged.loc[0, "us1"], 12.0)
        self.assertAlmostEqual(charged.loc[1, "us1"], 13.0)

    def test_unmeasured_block_m_is_not_charged(self):
        charged = add_intermediate_quant_cost(self._frame(), "_fp4", {32: 4.0})

        # block_m=64 has no measurement; it must not become NaN.
        self.assertAlmostEqual(charged.loc[2, "us1"], 20.0)

    def test_missing_measurements_are_a_no_op(self):
        frame = self._frame()
        pd.testing.assert_frame_equal(
            add_intermediate_quant_cost(frame, "_fp4", {}), frame
        )

    def test_input_frame_is_not_mutated(self):
        frame = self._frame()
        add_intermediate_quant_cost(frame, "_fp4", {32: 4.0})
        self.assertAlmostEqual(frame.loc[1, "us1"], 9.0)


class TestObjectiveContract(unittest.TestCase):
    def test_shortest_stage_combination_wins(self):
        candidates = pd.DataFrame(
            [
                {
                    "kernelName1": "fast_stage1",
                    "kernelName2": "slow_stage2",
                    "us1": 5.0,
                    "us2": 20.0,
                },
                {
                    "kernelName1": "balanced_stage1",
                    "kernelName2": "fast_stage2",
                    "us1": 8.0,
                    "us2": 10.0,
                },
                {
                    "kernelName1": "one_stage",
                    "kernelName2": None,
                    "us1": 19.0,
                    "us2": 0.0,
                },
            ]
        )

        winner = select_fastest_stage_combination(candidates)

        self.assertEqual(winner["kernelName1"], "balanced_stage1")
        self.assertEqual(winner["kernelName2"], "fast_stage2")
        self.assertAlmostEqual(winner["us1"] + winner["us2"], 18.0)

    def test_combination_selection_precedes_display_rounding(self):
        candidates = pd.DataFrame(
            [
                {"kernelName1": "rounds_to_tie", "us1": 1.00004, "us2": 1.0},
                {"kernelName1": "actually_fastest", "us1": 1.0, "us2": 1.0},
            ]
        )

        winner = select_fastest_stage_combination(candidates)

        self.assertEqual(winner["kernelName1"], "actually_fastest")

    def test_fused_suffixes_cover_both_quant_dtypes(self):
        self.assertEqual(set(FUSED_INTERMEDIATE_SUFFIXES), {"_fp4", "_fp8"})

    def test_error_ratio_default_matches_cosine_gate(self):
        # errRatio is a max accepted cosine_diff; 0.1 rejects broken kernels
        # while tolerating lossy-but-correct fp4/fp8 MoE paths.
        self.assertEqual(FmoeTuner.ARG_DEFAULTS["errRatio"], 0.1)


if __name__ == "__main__":
    unittest.main()
