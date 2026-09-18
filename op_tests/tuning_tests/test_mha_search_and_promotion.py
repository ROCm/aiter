# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for how the MHA forward tuner chooses what to measure and
what to publish: the sampled search strategy, the backend restriction, and the
gate that refuses to displace a configuration it cannot beat."""

import collections
import json
import unittest

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import pandas as pd

from op_tests.tuners.tune_mha_fwd import MhaFwdTuner

from aiter.ops.mha_fwd_policy import (
    MHA_FWD_TILE_CONFIG_BACKENDS,
    enumerate_mha_fwd_candidates,
)


class TestIncumbentGate(unittest.TestCase):
    """A tuning run must be able to tell an improvement from a regression."""

    KEY = ("gfx950", 256, "mi355x")
    INCUMBENT_CONFIG = '{"BLOCK_M":128,"BLOCK_N":64}'
    CHALLENGER_CONFIG = '{"BLOCK_M":256,"BLOCK_N":64}'

    def _tuner(self):
        tuner = MhaFwdTuner.__new__(MhaFwdTuner)
        tuner._incumbents_by_key = {self.KEY: {("gluon", self.INCUMBENT_CONFIG)}}
        tuner._promotions = []
        return tuner

    def _frame(self, challenger_us, incumbent_us, samples=(1000.0, 1001.0)):
        return pd.DataFrame(
            [
                {
                    "backend": "gluon",
                    "backend_config": self.CHALLENGER_CONFIG,
                    "us": challenger_us,
                    "samples_us": json.dumps(list(samples)),
                },
                {
                    "backend": "gluon",
                    "backend_config": self.INCUMBENT_CONFIG,
                    "us": incumbent_us,
                    "samples_us": json.dumps(list(samples)),
                },
            ]
        ).sort_values("us")

    def test_a_clear_improvement_is_published(self):
        winner = self._tuner()._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=800.0, incumbent_us=1000.0)
        )
        self.assertEqual(winner["backend_config"], self.CHALLENGER_CONFIG)
        self.assertIn("beat incumbent", winner["detail"])

    def test_a_winner_inside_measurement_spread_does_not_displace_the_incumbent(self):
        """The margin here is 0.1%, far under the scatter of the samples, so
        the two configurations have not been told apart and the run should
        change nothing rather than churn the published table."""
        winner = self._tuner()._gate_against_incumbent(
            self.KEY,
            self._frame(
                challenger_us=999.0, incumbent_us=1000.0, samples=(900.0, 1100.0)
            ),
        )
        self.assertEqual(winner["backend_config"], self.INCUMBENT_CONFIG)
        self.assertIn("incumbent retained", winner["detail"])

    def test_the_incumbent_winning_outright_is_recorded_as_such(self):
        winner = self._tuner()._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=1200.0, incumbent_us=1000.0)
        )
        self.assertEqual(winner["backend_config"], self.INCUMBENT_CONFIG)
        self.assertIn("nothing measured beat it", winner["detail"])

    def test_an_unmeasured_incumbent_is_flagged_rather_than_assumed_beaten(self):
        tuner = self._tuner()
        frame = pd.DataFrame(
            [
                {
                    "backend": "gluon",
                    "backend_config": self.CHALLENGER_CONFIG,
                    "us": 800.0,
                    "samples_us": "[800.0,801.0]",
                }
            ]
        )
        winner = tuner._gate_against_incumbent(self.KEY, frame)
        self.assertEqual(winner["backend_config"], self.CHALLENGER_CONFIG)
        self.assertIn("improvement unverified", winner["detail"])

    def test_the_regression_this_gate_exists_to_stop(self):
        """The real case: a sampler gap meant the published winner was 17%
        slower than the shipped default. With the default measured in the same
        sweep the gate keeps it."""
        tuner = self._tuner()
        winner = tuner._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=2206.0, incumbent_us=1831.0)
        )
        self.assertEqual(winner["backend_config"], self.INCUMBENT_CONFIG)
        self.assertEqual(tuner._promotions[0]["decision"], "incumbent_fastest")

    def test_more_rounds_make_the_gate_more_sensitive_not_less(self):
        """A range-based threshold widens as samples are added, so gathering
        more evidence would make a real improvement harder to publish. The
        standard error has to shrink instead."""
        tight = [1000.0, 1002.0]
        many = tight * 8
        self.assertLess(
            MhaFwdTuner._standard_error_us({"samples_us": json.dumps(many)}),
            MhaFwdTuner._standard_error_us({"samples_us": json.dumps(tight)}),
        )

    def test_every_decision_is_recorded_for_the_evidence_file(self):
        tuner = self._tuner()
        tuner._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=800.0, incumbent_us=1000.0)
        )
        record = tuner._promotions[0]
        self.assertEqual(record["decision"], "promoted")
        self.assertAlmostEqual(record["margin"], 0.2, places=6)
        self.assertEqual(record["incumbent"]["us"], 1000.0)


class TestSmokeStrategy(unittest.TestCase):
    def test_smoke_is_a_strict_subset_of_the_exhaustive_catalogue(self):
        for gfx in ("gfx942", "gfx950", "gfx1250"):
            with self.subTest(gfx=gfx):
                full = {c.identity for c in enumerate_mha_fwd_candidates(gfx)}
                smoke = {c.identity for c in enumerate_mha_fwd_candidates(gfx, "smoke")}
                self.assertTrue(smoke <= full)

    def test_smoke_keeps_every_name_is_config_backend(self):
        full = collections.Counter(
            c.backend for c in enumerate_mha_fwd_candidates("gfx950")
        )
        smoke = collections.Counter(
            c.backend for c in enumerate_mha_fwd_candidates("gfx950", "smoke")
        )
        for backend, count in full.items():
            if backend not in MHA_FWD_TILE_CONFIG_BACKENDS:
                with self.subTest(backend=backend):
                    self.assertEqual(smoke[backend], count)

    def test_a_sampled_grid_still_varies_every_tuning_axis(self):
        """A stride over the flattened product pins the inner axes; if that
        regresses, a smoke run would only ever sample block sizes."""
        for backend in ("triton", "gluon"):
            values = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates("gfx950", "smoke"):
                if candidate.backend == backend:
                    for key, value in candidate.backend_config.items():
                        values[key].add(value)
            full_axes = collections.defaultdict(set)
            for candidate in enumerate_mha_fwd_candidates("gfx950"):
                if candidate.backend == backend:
                    for key, value in candidate.backend_config.items():
                        full_axes[key].add(value)
            for key, sampled in values.items():
                if len(full_axes[key]) > 1:
                    with self.subTest(backend=backend, axis=key):
                        self.assertGreater(len(sampled), 1)

    def test_a_sampled_grid_does_not_lock_axes_to_each_other(self):
        """Varying every axis is not enough. Advancing all axes together
        varies each one while walking a single diagonal, which leaves whole
        regions of the grid unreachable -- BLOCK_N=64 with num_warps=8 never
        appears. Require each pair of axes to take more joint values than
        either takes alone, which a diagonal cannot satisfy."""
        for backend in ("triton", "gluon"):
            sample = [
                c.backend_config
                for c in enumerate_mha_fwd_candidates("gfx950", "smoke")
                if c.backend == backend
            ]
            axes = sorted(sample[0])
            for i, left in enumerate(axes):
                for right in axes[i + 1 :]:
                    distinct_left = {cfg[left] for cfg in sample}
                    distinct_right = {cfg[right] for cfg in sample}
                    if len(distinct_left) < 2 or len(distinct_right) < 2:
                        continue
                    joint = {(cfg[left], cfg[right]) for cfg in sample}
                    with self.subTest(backend=backend, pair=(left, right)):
                        self.assertGreater(
                            len(joint),
                            max(len(distinct_left), len(distinct_right)),
                            f"{left} and {right} advance in lockstep",
                        )

    def test_smoke_is_deterministic(self):
        first = [c.identity for c in enumerate_mha_fwd_candidates("gfx950", "smoke")]
        second = [c.identity for c in enumerate_mha_fwd_candidates("gfx950", "smoke")]
        self.assertEqual(first, second)

    def test_an_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown MHA search strategy"):
            enumerate_mha_fwd_candidates("gfx950", "random")


class TestBackendRestriction(unittest.TestCase):
    """The restriction is a control for comparing contracts, so it has to
    actually restrict and has to be visible when it does."""

    def test_only_the_named_backends_are_measured(self):
        candidates = enumerate_mha_fwd_candidates(
            "gfx950", "exhaustive", ["triton", "gluon"]
        )
        self.assertEqual(sorted({c.backend for c in candidates}), ["gluon", "triton"])

    def test_the_restricted_catalogue_is_a_subset_of_the_full_one(self):
        full = {c.identity for c in enumerate_mha_fwd_candidates("gfx950")}
        restricted = {
            c.identity
            for c in enumerate_mha_fwd_candidates("gfx950", "exhaustive", ["gluon"])
        }
        self.assertTrue(restricted <= full)
        self.assertLess(len(restricted), len(full))

    def test_no_restriction_is_the_full_catalogue(self):
        self.assertEqual(
            [c.identity for c in enumerate_mha_fwd_candidates("gfx950", "exhaustive")],
            [
                c.identity
                for c in enumerate_mha_fwd_candidates("gfx950", "exhaustive", None)
            ],
        )

    def test_an_unknown_backend_is_rejected_rather_than_silently_empty(self):
        with self.assertRaisesRegex(ValueError, "unknown MHA backends"):
            enumerate_mha_fwd_candidates("gfx950", "exhaustive", ["trition"])

    def test_the_restriction_composes_with_the_smoke_strategy(self):
        candidates = enumerate_mha_fwd_candidates("gfx950", "smoke", ["triton"])
        self.assertTrue(candidates)
        self.assertEqual({c.backend for c in candidates}, {"triton"})


if __name__ == "__main__":
    unittest.main()
