# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for the family-independent tuning thresholds.

Run: python3 -m unittest op_tests.tuning_tests.test_tuning_policy -v
"""

import dataclasses
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "aiter" / "utility" / "tuning_policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "tuning_policy_under_test", POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


policy = _load_policy()


class TestStandaloneImport(unittest.TestCase):
    def test_loads_without_torch(self):
        # The runtime and the CSV tests read these values without a device.
        script = (
            "import sys, importlib.util\n"
            "sys.modules['torch'] = None\n"
            f"spec = importlib.util.spec_from_file_location('p', {str(POLICY_PATH)!r})\n"
            "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
        )
        subprocess.run([sys.executable, "-c", script], check=True)


class TestBaseTunerDefaults(unittest.TestCase):
    def test_arg_defaults_come_from_the_policy(self):
        from aiter.utility.base_tuner import TunerCommon

        defaults = TunerCommon.ARG_DEFAULTS
        measurement = policy.DEFAULT_MEASUREMENT
        self.assertEqual(defaults["warmup"], measurement.warmup)
        self.assertEqual(defaults["iters"], measurement.iters)
        self.assertEqual(defaults["errRatio"], measurement.err_ratio)
        self.assertEqual(defaults["timeout"], measurement.timeout)
        self.assertEqual(
            defaults["min_improvement_pct"], policy.COMPARE_MIN_IMPROVEMENT_PCT
        )


class TestPromotionPolicy(unittest.TestCase):
    def test_family_override_keeps_the_other_defaults(self):
        tighter = dataclasses.replace(policy.DEFAULT_PROMOTION, indifference_delta=0.01)
        self.assertEqual(tighter.indifference_delta, 0.01)
        self.assertEqual(tighter.finalists, policy.DEFAULT_PROMOTION.finalists)

    def test_rejects_a_delta_that_could_never_be_beaten(self):
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(indifference_delta=1.0)
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(indifference_delta=-0.01)

    def test_rejects_an_empty_finalist_round(self):
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(finalists=0)
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(finalist_rounds=0)

    def test_rejects_a_significance_bar_of_zero(self):
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(significance_sigma=0.0)


class TestRacePolicy(unittest.TestCase):
    def test_rejects_an_unusable_error_budget(self):
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                policy.RacePolicy(alpha=alpha)

    def test_rejects_a_race_that_stops_before_it_may_eliminate(self):
        with self.assertRaises(ValueError):
            policy.RacePolicy(min_blocks=5, max_blocks=4)
        with self.assertRaises(ValueError):
            policy.RacePolicy(block_calls=0)


class TestGateAgainstIncumbent(unittest.TestCase):
    DELTA = policy.PromotionPolicy(indifference_delta=0.02)

    def gate(self, incumbent_us, challenger_us):
        return policy.gate_against_incumbent(incumbent_us, challenger_us, self.DELTA)

    def test_a_clear_win_is_promoted(self):
        decision = self.gate(100.0, 90.0)
        self.assertEqual(decision.outcome, policy.PROMOTE)
        self.assertAlmostEqual(decision.margin, 0.10)

    def test_a_win_inside_the_delta_is_retained(self):
        decision = self.gate(100.0, 99.0)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertAlmostEqual(decision.margin, 0.01)

    def test_a_margin_equal_to_the_delta_is_retained(self):
        # The delta is the largest gap still treated as noise.
        self.assertEqual(self.gate(100.0, 98.0).outcome, policy.RETAIN)

    def test_a_slower_challenger_is_retained(self):
        decision = self.gate(100.0, 110.0)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertLess(decision.margin, 0)

    def test_no_measured_incumbent_promotes_without_a_margin(self):
        for incumbent in (None, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(incumbent=incumbent):
                decision = self.gate(incumbent, 50.0)
                self.assertEqual(decision.outcome, policy.PROMOTE)
                self.assertIsNone(decision.margin)

    def test_a_standard_error_bar_replaces_the_delta(self):
        # 2 sigma x 1.5 us combined standard error on 100 us is a 3% bar, so a
        # 2.5% win that clears the delta is still inside the scatter.
        bar = policy.standard_error_bar(100.0, 1.5, self.DELTA)
        self.assertAlmostEqual(bar, 0.03)
        decision = policy.gate_against_incumbent(100.0, 97.5, self.DELTA, bar=bar)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertEqual(self.gate(100.0, 97.5).outcome, policy.PROMOTE)

    def test_an_unmeasured_challenger_is_a_caller_error(self):
        for challenger in (None, 0.0, -1.0, float("inf")):
            with self.subTest(challenger=challenger), self.assertRaises(ValueError):
                self.gate(100.0, challenger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
