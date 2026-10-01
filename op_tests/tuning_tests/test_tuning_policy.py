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
        self.assertEqual(defaults["batch"], policy.DEFAULT_RUN.batch)
        self.assertEqual(
            defaults["min_improvement_pct"],
            policy.DEFAULT_PROMOTION.min_improvement_pct,
        )


class TestRunPolicy(unittest.TestCase):
    def test_rejects_an_empty_batch(self):
        with self.assertRaises(ValueError):
            policy.RunPolicy(batch=0)


class TestPromotionPolicy(unittest.TestCase):
    def test_family_override(self):
        tighter = dataclasses.replace(policy.DEFAULT_PROMOTION, min_improvement_pct=5.0)
        self.assertEqual(tighter.min_improvement_pct, 5.0)

    def test_rejects_a_bar_that_could_never_be_cleared(self):
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(min_improvement_pct=100.0)
        with self.assertRaises(ValueError):
            policy.PromotionPolicy(min_improvement_pct=-1.0)


class TestFinalistPolicy(unittest.TestCase):
    def test_rejects_an_empty_finalist_round(self):
        with self.assertRaises(ValueError):
            policy.FinalistPolicy(finalists=0)
        with self.assertRaises(ValueError):
            policy.FinalistPolicy(rounds=0)

    def test_rejects_a_significance_bar_of_zero(self):
        with self.assertRaises(ValueError):
            policy.FinalistPolicy(significance_sigma=0.0)


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
    BAR = policy.PromotionPolicy(min_improvement_pct=3.0)

    def gate(self, incumbent_us, challenger_us):
        return policy.gate_against_incumbent(incumbent_us, challenger_us, self.BAR)

    def test_a_clear_win_is_promoted(self):
        decision = self.gate(100.0, 90.0)
        self.assertEqual(decision.outcome, policy.PROMOTE)
        self.assertAlmostEqual(decision.margin_pct, 10.0)

    def test_a_win_below_the_bar_is_retained(self):
        decision = self.gate(100.0, 99.0)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertAlmostEqual(decision.margin_pct, 1.0)

    def test_a_margin_equal_to_the_bar_is_promoted(self):
        # "At least this many percent", as --compare --update_improved reads it.
        self.assertEqual(self.gate(100.0, 97.0).outcome, policy.PROMOTE)

    def test_a_slower_challenger_is_retained(self):
        decision = self.gate(100.0, 110.0)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertLess(decision.margin_pct, 0)

    def test_no_measured_incumbent_promotes_without_a_margin(self):
        for incumbent in (None, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(incumbent=incumbent):
                decision = self.gate(incumbent, 50.0)
                self.assertEqual(decision.outcome, policy.PROMOTE)
                self.assertIsNone(decision.margin_pct)

    def test_the_standard_error_bar_is_in_percent(self):
        # 2 sigma x 2.5 us combined standard error on 100 us.
        self.assertAlmostEqual(policy.standard_error_bar(100.0, 2.5), 5.0)
        self.assertEqual(policy.standard_error_bar(None, 2.5), 0.0)

    def test_noise_above_the_minimum_raises_the_bar(self):
        # A 4% win clears the 3% minimum, but not a 5% noise floor.
        decision = policy.gate_against_incumbent(100.0, 96.0, self.BAR, noise_pct=5.0)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertAlmostEqual(decision.bar_pct, 5.0)
        self.assertEqual(self.gate(100.0, 96.0).outcome, policy.PROMOTE)

    def test_noise_below_the_minimum_does_not_lower_the_bar(self):
        # Well resolved but too small to be worth a row.
        decision = policy.gate_against_incumbent(100.0, 99.0, self.BAR, noise_pct=0.1)
        self.assertEqual(decision.outcome, policy.RETAIN)
        self.assertAlmostEqual(decision.bar_pct, 3.0)

    def test_an_unmeasured_challenger_is_a_caller_error(self):
        for challenger in (None, 0.0, -1.0, float("inf")):
            with self.subTest(challenger=challenger), self.assertRaises(ValueError):
                self.gate(100.0, challenger)


if __name__ == "__main__":
    unittest.main(verbosity=2)
