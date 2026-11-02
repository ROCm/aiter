# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for aiter.utility.tuning_search (no GPU, no kernels).

The synthetic GEMM-like space has a known optimum (found by enumeration), so
the strategies are scored against the truth; the measurement-rule tests use
a CallableBackend over a tiny space whose fastest config is broken.
"""

import math
import tempfile
import unittest
from pathlib import Path

from aiter.utility.tuning_search import (
    Budget,
    CallableBackend,
    CompileOutcome,
    Param,
    SearchSpace,
    Status,
    SyntheticBackend,
    TuningDB,
    record_result,
    search_configs,
    tune,
)
from aiter.utility.tuning_search.synthetic import brute_force_optimum, gemm_like

SEEDS = (0, 1, 2)
BUDGET = 128


def _fraction_of_optimum(strategy, space, objective, optimum):
    fractions = []
    for seed in SEEDS:
        res = tune(
            space,
            SyntheticBackend(objective, noise=0.01, seed=seed),
            strategy=strategy,
            budget=Budget(evaluations=BUDGET),
            seed=seed,
        )
        fractions.append(optimum / objective(res.best_config))
    return sum(fractions) / len(fractions)


class TestStrategiesOnSyntheticGemm(unittest.TestCase):
    """Guided strategies reach near the exhaustive optimum on a rugged
    ~10k-config space with ~1 % of the evaluations, and beat random."""

    @classmethod
    def setUpClass(cls):
        cls.space, objective = gemm_like(rugged=0.15)
        # staticmethod: a plain function class attribute would bind to self.
        cls.objective = staticmethod(objective)
        _key, cls.optimum, cls.valid = brute_force_optimum(cls.space, objective)

    def test_space_is_large(self):
        self.assertGreater(self.valid, 50 * BUDGET)

    def test_guided_strategies(self):
        random_frac = _fraction_of_optimum(
            "random", self.space, self.objective, self.optimum
        )
        for strategy in ("hill", "de", "gp", "portfolio"):
            with self.subTest(strategy=strategy):
                frac = _fraction_of_optimum(
                    strategy, self.space, self.objective, self.optimum
                )
                self.assertGreaterEqual(frac, 0.95)
                self.assertGreater(frac, random_frac)

    def test_exhaustive_finds_optimum_when_space_fits(self):
        space = SearchSpace(
            (Param.ordinal("a", (1, 2, 3, 4)), Param.ordinal("b", (1, 2, 3))),
            name="small",
        )

        def objective(cfg):
            return 1.0 + (cfg["a"] - 3) ** 2 + (cfg["b"] - 2) ** 2

        res = tune(
            space,
            SyntheticBackend(objective, noise=0.0),
            strategy="exhaustive",
            budget=Budget(evaluations=64),
        )
        self.assertEqual((res.best_config["a"], res.best_config["b"]), (3, 2))


def _broken_fastest_backend(*, spill=False):
    """Time = x; config x=1 is fastest but fails the numeric check (or
    spills, with ``spill``)."""

    def compile_(config, key):
        resources = {"spill_vgpr": 4.0} if spill and config["x"] == 1 else {}
        return CompileOutcome(True, artifact=config["x"], resources=resources)

    def check(config, key, artifact):
        if not spill and artifact == 1:
            return False, "wrong K tail"
        return True, ""

    def measure(config, key, artifact, fidelity):
        return [float(artifact)] * max(1, fidelity.reps)

    return CallableBackend(compile=compile_, check=check, measure=measure)


class TestMeasurementRules(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace((Param.ordinal("x", (1, 2, 3, 4)),), name="x")

    def _search(self, backend):
        return tune(
            self.space,
            backend,
            strategy="exhaustive",
            budget=Budget(evaluations=16),
        )

    def test_numeric_failure_never_wins(self):
        res = self._search(_broken_fastest_backend())
        self.assertEqual(res.best_config["x"], 2)
        failed = [o for o in res.observations if o.status == Status.NUMERIC_FAIL]
        self.assertTrue(failed)
        self.assertTrue(all(math.isinf(o.value) for o in failed))

    def test_spill_is_rejected(self):
        res = self._search(_broken_fastest_backend(spill=True))
        self.assertEqual(res.best_config["x"], 2)
        self.assertTrue(any(o.status == Status.SPILL for o in res.observations))


class TestSpaceAndRecords(unittest.TestCase):
    def test_gated_knob_ignored_when_gate_off(self):
        space = SearchSpace(
            (
                Param.boolean("lds"),
                Param.ordinal("num_buffers", (2, 3, 4), gate="lds"),
            ),
            name="gated",
        )
        off_a = space.encode({"lds": False, "num_buffers": 2})
        off_b = space.encode({"lds": False, "num_buffers": 4})
        on_a = space.encode({"lds": True, "num_buffers": 2})
        on_b = space.encode({"lds": True, "num_buffers": 4})
        self.assertEqual(space.key(off_a), space.key(off_b))
        self.assertNotEqual(space.key(on_a), space.key(on_b))
        self.assertNotIn("num_buffers", space.decode(off_a))

    def test_search_configs_over_flat_list(self):
        configs = [{"bkv": b, "rpw": r} for b in (64, 128, 256) for r in (1, 2, 4)]

        def measure(cfg, fidelity):
            return abs(cfg["bkv"] - 128) / 64 + abs(cfg["rpw"] - 2) + 1.0

        winner, _res = search_configs(
            configs,
            measure=measure,
            check=lambda cfg: True,
            strategy="exhaustive",
            budget=Budget(evaluations=32),
        )
        self.assertEqual(configs[winner], {"bkv": 128, "rpw": 2})

    def test_record_roundtrip_through_tuning_db(self):
        space = SearchSpace((Param.ordinal("x", (1, 2, 3, 4)),), name="x")

        def objective(cfg):
            return float(cfg["x"])

        baseline = space.encode({"x": 4})
        res = tune(
            space,
            SyntheticBackend(objective, noise=0.0),
            strategy="exhaustive",
            budget=Budget(evaluations=32),
            baseline=baseline,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tuned.json"
            rec = record_result(
                res,
                family="fp8_mqa_logits",
                candidate="lds_pipe",
                arch="gfx950",
                problem="H64_S4096_K4096",
                tuning_id=f"lds_pipe@{res.best_key}",
                config_key=res.best_key,
                knobs=res.best_config,
                db=TuningDB(path),
            )
            self.assertIsNotNone(rec)
            again = TuningDB(path).get(
                "fp8_mqa_logits", "gfx950", "H64_S4096_K4096", "lds_pipe"
            )
            self.assertIsNotNone(again)
            self.assertEqual(again.knobs, {"x": 1})


if __name__ == "__main__":
    unittest.main()
