# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the interleaved elimination race.

Every test here drives the race from synthetic latencies, so what is under
test is the decision logic rather than a GPU. That is the point: the questions
these answer -- whether a protected incumbent can be eliminated, whether a
resumed race reaches the same verdict, how often a certified race publishes
something more than delta behind the best -- all have exact or statistical
answers that a real measurement would only obscure.
"""

import json
import os
import random
import tempfile
import unittest
from typing import ClassVar

from aiter.utility.block_race import (
    JsonlBlockJournal,
    RaceEntrant,
    RacePolicy,
    critical_t,
    measure_blocks,
    race,
)

POLICY = RacePolicy(
    alpha=0.05, block_calls=10, warmup_calls=0, min_blocks=3, max_blocks=30
)
RACE_ARGS = {
    "min_improvement_pct": 2.0,
    "policy": POLICY,
    "seed": 4242,
    "verbose": False,
}
DELTA = RACE_ARGS["min_improvement_pct"] / 100.0


def constant_timer(latency_by_label, noise=0.004, seed=1):
    """A fake GPU: each label has a true latency, plus reproducible jitter."""
    rng = random.Random(seed)

    def time_calls(entrant, count):
        true = latency_by_label[entrant.label]
        return [true * (1.0 + rng.gauss(0.0, noise)) for _ in range(count)]

    return time_calls


def noisy_timer(truth, seed, call_noise=0.005, visit_jitter=0.0, outliers=0.0):
    """Call-level noise, a per-visit shift, and a share of 1.5x outlier calls."""
    rng = random.Random(seed)

    def time_calls(entrant, count):
        shift = 1.0 + rng.gauss(0.0, visit_jitter)
        values = []
        for _ in range(count):
            value = truth[entrant.label] * shift * (1.0 + rng.gauss(0.0, call_noise))
            if rng.random() < outliers:
                value *= 1.5
            values.append(value)
        return values

    return time_calls


class TestRacePolicy(unittest.TestCase):
    def test_rejects_an_unusable_error_budget(self):
        for alpha in (0.0, 1.0):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                RacePolicy(alpha=alpha)

    def test_rejects_a_race_that_stops_before_it_may_eliminate(self):
        for kwargs in (
            {"min_blocks": 5, "max_blocks": 4},
            {"min_blocks": 1},
            {"block_calls": 0},
            {"warmup_calls": -1},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                RacePolicy(**kwargs)


class TestStatistics(unittest.TestCase):
    def test_the_hand_rolled_t_matches_published_tables(self):
        """The continued fraction is the one piece that can be wrong without
        looking wrong: a bad critical value does not raise, it just eliminates
        candidates the evidence does not support."""
        published = {
            (1, 0.05): 12.706,
            (2, 0.05): 4.303,
            (5, 0.05): 2.571,
            (10, 0.05): 2.228,
            (29, 0.05): 2.045,
            (2, 0.01): 9.925,
            (10, 0.01): 3.169,
            (29, 0.001): 3.659,
        }
        for (degrees, alpha), expected in published.items():
            with self.subTest(degrees=degrees, alpha=alpha):
                self.assertAlmostEqual(critical_t(alpha, degrees), expected, delta=1e-3)


class TestCalibration(unittest.TestCase):
    """The guarantee the module exists for, estimated over seeded races: a
    certified race publishes something more than delta behind the best at
    most alpha of the time, and drops the true best at most alpha/2 of the
    time. Each field has a rival just outside delta, so a race that publishes
    the wrong one of the top two fails here."""

    RACES = 60
    FIELDS: ClassVar[dict] = {
        "gaussian": {"call_noise": 0.005},
        "jittery_visits": {"call_noise": 0.005, "visit_jitter": 0.02},
        "outlier_calls": {"call_noise": 0.005, "outliers": 0.1},
    }

    @staticmethod
    def _truth(seed):
        rng = random.Random(seed)
        truth = {"best": 100.0, "rival": 100.0 * (1.0 + DELTA + 0.005)}
        truth.update({f"slow{i}": rng.uniform(110.0, 300.0) for i in range(6)})
        return truth

    def test_a_certified_race_is_rarely_more_than_delta_behind(self):
        for name, noise in self.FIELDS.items():
            with self.subTest(field=name):
                certified = regret = dropped_best = 0
                for seed in range(self.RACES):
                    truth = self._truth(seed)
                    result = race(
                        [RaceEntrant(label) for label in truth],
                        noisy_timer(truth, seed, **noise),
                        **dict(RACE_ARGS, seed=seed),
                    )
                    states = {v.label: v.state for v in result.verdicts}
                    dropped_best += states["best"] == "eliminated"
                    if result.certified:
                        certified += 1
                        regret += truth[result.winner] > 100.0 * (1.0 + DELTA)
                self.assertGreater(certified, self.RACES // 2, "the race never decides")
                self.assertLessEqual(regret, POLICY.alpha * certified)
                self.assertLessEqual(dropped_best, POLICY.alpha / 2 * self.RACES)


class TestElimination(unittest.TestCase):
    def test_candidates_slower_by_a_factor_are_dropped_at_the_first_look(self):
        truth = {
            "fast": 100.0,
            "slow": 400.0,
            "slower": 900.0,
        }
        entrants = [RaceEntrant(label) for label in truth]
        result = race(entrants, constant_timer(truth), **RACE_ARGS)

        dropped = {v.label: v for v in result.verdicts if v.state == "eliminated"}
        self.assertEqual(set(dropped), {"slow", "slower"})
        for verdict in dropped.values():
            self.assertEqual(
                verdict.blocks_used,
                POLICY.min_blocks,
                "a candidate this far behind should not survive past the "
                "first block at which elimination is allowed",
            )

    def test_a_candidate_inside_delta_is_kept_rather_than_separated(self):
        """Half a percent apart with delta at two percent is a finished
        question, not a harder one. Spending blocks to separate them would buy
        nothing, because either is equally fast to ship."""
        truth = {"leader": 100.0, "neighbour": 100.5, "behind": 300.0}
        entrants = [RaceEntrant(label) for label in truth]
        result = race(entrants, constant_timer(truth), **RACE_ARGS)

        self.assertIn("neighbour", result.survivors)
        self.assertNotIn(
            "neighbour",
            [v.label for v in result.verdicts if v.state == "eliminated"],
        )

    def test_a_noisy_candidate_that_merely_outlasts_the_budget_is_not_publishable(self):
        """Surviving elimination is not the same as being tied."""
        # Scripted rather than random: the leader is steady while the noisy
        # one swings either side of it, so the paired differences average
        # clearly positive but scatter far too widely to prove anything.
        scripted = {
            "leader": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "noisy": [60.0, 190.0, 60.0, 190.0, 60.0, 190.0],
        }
        visits = {label: 0 for label in scripted}

        def scripted_timer(entrant, count):
            sequence = scripted[entrant.label]
            value = sequence[visits[entrant.label] % len(sequence)]
            visits[entrant.label] += 1
            return [value] * count

        entrants = [RaceEntrant(label) for label in scripted]
        policy = RacePolicy(block_calls=10, warmup_calls=0, max_blocks=6)
        result = race(entrants, scripted_timer, **dict(RACE_ARGS, policy=policy))

        states = {v.label: v.state for v in result.verdicts}
        self.assertEqual(states["noisy"], "undecided")
        self.assertNotIn("noisy", result.survivors)
        self.assertEqual(result.winner, "leader")
        self.assertFalse(result.certified)

    def test_a_race_that_certifies_stops_before_the_ceiling(self):
        truth = {"fast": 100.0, "slow": 500.0}
        result = race(
            [RaceEntrant(label) for label in truth],
            constant_timer(truth),
            **RACE_ARGS,
        )
        self.assertTrue(result.certified)
        self.assertLess(result.blocks_run, POLICY.max_blocks)


class TestWinner(unittest.TestCase):
    def test_a_steadier_rival_just_behind_is_not_published_over_the_leader(self):
        """The leader is what the guarantee covers. A steadier candidate 2.5%
        behind with delta at 2% is a slower config, however repeatable."""
        truth = {"best": 100.0, "steady": 102.5}
        for seed in range(20):
            with self.subTest(seed=seed):
                jumpy = noisy_timer(truth, seed, visit_jitter=0.02)
                steady = noisy_timer(truth, seed + 1000, call_noise=0.001)

                def time_calls(entrant, count, jumpy=jumpy, steady=steady):
                    timer = jumpy if entrant.label == "best" else steady
                    return timer(entrant, count)

                result = race(
                    [RaceEntrant(label) for label in truth],
                    time_calls,
                    **dict(RACE_ARGS, seed=seed),
                )
                if result.certified:
                    self.assertEqual(result.winner, "best")

    def test_an_incumbent_within_delta_is_kept(self):
        truth = {"challenger": 100.0, "incumbent": 100.3}
        entrants = [
            RaceEntrant("challenger"),
            RaceEntrant("incumbent", protected=True),
        ]
        result = race(entrants, constant_timer(truth, noise=0.002), **RACE_ARGS)
        self.assertTrue(result.certified)
        self.assertEqual(result.winner, "incumbent")
        self.assertEqual(result.rule, "incumbent_retained")

    def test_an_incumbent_not_proven_within_delta_is_not_kept(self):
        """Close on the point estimate is not enough: the incumbent is kept
        only when no survivor is proven more than delta faster than it."""
        entrants = [
            RaceEntrant("challenger"),
            RaceEntrant("incumbent", protected=True),
        ]
        for name, incumbent, timer in (
            # Proven behind.
            ("behind", 103.0, lambda truth: constant_timer(truth, noise=0.002)),
            # Inside the zone on its point estimate, but too noisy to prove
            # the challenger is not more than delta faster.
            ("unproven", 101.5, lambda truth: noisy_timer(truth, 3, visit_jitter=0.01)),
        ):
            with self.subTest(name):
                truth = {"challenger": 100.0, "incumbent": incumbent}
                result = race(entrants, timer(truth), **RACE_ARGS)
                self.assertTrue(result.certified)
                self.assertEqual(result.winner, "challenger")
                self.assertEqual(result.rule, "leader")

    def test_the_promotion_bar_is_a_percentage(self):
        truth = {"fast": 100.0, "slow": 500.0}
        result = race(
            [RaceEntrant(label) for label in truth],
            constant_timer(truth),
            **dict(RACE_ARGS, min_improvement_pct=3.0),
        )
        self.assertAlmostEqual(result.delta, 0.03)
        for pct in (0.0, -1.0, 100.0):
            with self.subTest(pct=pct), self.assertRaises(ValueError):
                race(
                    [RaceEntrant(label) for label in truth],
                    constant_timer(truth),
                    **dict(RACE_ARGS, min_improvement_pct=pct),
                )


class TestProtectedIncumbent(unittest.TestCase):
    """A run that stops measuring the configuration already in use cannot tell
    an improvement from a regression."""

    ENTRANTS: ClassVar[list] = [
        RaceEntrant("challenger"),
        RaceEntrant("incumbent", protected=True),
    ]
    TRUTH: ClassVar[dict] = {"challenger": 100.0, "incumbent": 900.0}

    def test_a_protected_incumbent_is_never_eliminated(self):
        result = race(self.ENTRANTS, constant_timer(self.TRUTH), **RACE_ARGS)
        states = {v.label: v.state for v in result.verdicts}
        self.assertEqual(states["incumbent"], "protected_behind")

    def test_an_incumbent_kept_in_the_field_is_not_reported_as_a_tie(self):
        result = race(self.ENTRANTS, constant_timer(self.TRUTH), **RACE_ARGS)
        self.assertNotIn("incumbent", result.survivors)
        self.assertEqual(result.winner, "challenger")

    def test_a_protected_incumbent_is_measured_in_every_block(self):
        result = race(self.ENTRANTS, constant_timer(self.TRUTH), **RACE_ARGS)
        self.assertEqual(len(result.samples["incumbent"].blocks), result.blocks_run)


class TestStrategyAgreement(unittest.TestCase):
    """It reaches the answer exhaustive screening reaches, for fewer calls."""

    @staticmethod
    def _field(seed):
        rng = random.Random(seed)
        truth = {"fast": 100.0, "near": 101.5}
        truth.update({f"slow{i}": rng.uniform(115.0, 400.0) for i in range(18)})
        return truth

    def test_the_race_picks_what_measuring_everything_would_pick(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                truth = self._field(seed)
                entrants = [RaceEntrant(label) for label in truth]
                raced = race(entrants, constant_timer(truth, seed=seed), **RACE_ARGS)
                screened = measure_blocks(
                    entrants,
                    constant_timer(truth, seed=seed),
                    blocks=POLICY.max_blocks,
                    seed=RACE_ARGS["seed"],
                    policy=POLICY,
                )
                exhaustive = min(screened, key=lambda label: screened[label].estimate)
                self.assertLessEqual(
                    truth[raced.winner], truth[exhaustive] * (1.0 + DELTA)
                )

    def test_the_race_gets_there_on_fewer_calls(self):
        truth = self._field(0)
        entrants = [RaceEntrant(label) for label in truth]
        raced = race(entrants, constant_timer(truth, seed=0), **RACE_ARGS)
        exhaustive_calls = len(entrants) * POLICY.max_blocks * POLICY.block_calls
        self.assertLess(raced.calls_spent, exhaustive_calls / 2)

    def test_each_visit_starts_with_untimed_warmup_calls(self):
        """The previous visit ran another kernel, so a visit's first call is
        cold; it must not reach the sample."""
        truth = {"fast": 100.0, "slow": 500.0}
        requested = []

        def cold_first(entrant, count):
            requested.append(count)
            if count == 2:
                return [10_000.0] * count
            return [truth[entrant.label]] * count

        policy = RacePolicy(block_calls=5, warmup_calls=2)
        result = race(
            [RaceEntrant(label) for label in truth],
            cold_first,
            **dict(RACE_ARGS, policy=policy),
        )
        self.assertEqual(requested[:2], [2, 5])
        self.assertEqual(result.samples["fast"].estimate, 100.0)


class TestUnusableMeasurements(unittest.TestCase):
    """A candidate that cannot be timed must leave, not decide the race."""

    TRUTH: ClassVar[dict] = {"fast": 100.0, "near": 100.6, "slow": 250.0}

    def _race_with(self, broken_label, broken_timer, protected=()):
        entrants = [
            RaceEntrant(label, protected=label in protected) for label in self.TRUTH
        ]
        honest = constant_timer(self.TRUTH, seed=5)

        def time_calls(entrant, count):
            if entrant.label == broken_label:
                return broken_timer(count)
            return honest(entrant, count)

        return race(entrants, time_calls, **RACE_ARGS)

    def test_a_faulting_candidate_is_dropped_rather_than_ending_the_run(self):
        def explode(count):
            raise RuntimeError("HIP error: illegal memory access")

        result = self._race_with("slow", explode)
        states = {v.label: v.state for v in result.verdicts}
        self.assertEqual(states["slow"], "crashed")
        self.assertEqual(result.winner, "fast")
        self.assertNotIn("slow", result.survivors)

    def test_a_latency_that_is_not_positive_and_finite_cannot_win(self):
        for reading in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(reading=reading):
                result = self._race_with(
                    "slow", lambda count, value=reading: [value] * count
                )
                states = {v.label: v.state for v in result.verdicts}
                self.assertEqual(states["slow"], "crashed")
                self.assertEqual(result.winner, "fast")

    def test_one_bad_visit_is_retried_rather_than_ejecting_the_incumbent(self):
        calls = {"n": 0}

        def glitch_once(count):
            calls["n"] += 1
            if calls["n"] == 1:
                return [0.0] * count
            return [100.3] * count

        result = self._race_with("near", glitch_once, protected=("near",))
        states = {v.label: v.state for v in result.verdicts}
        self.assertNotEqual(states["near"], "crashed")
        self.assertEqual(len(result.samples["near"].blocks), result.blocks_run)

    def test_a_field_that_cannot_be_timed_at_all_fails_loudly(self):
        entrants = [RaceEntrant(label) for label in self.TRUTH]

        def explode(entrant, count):
            raise RuntimeError("HIP error: illegal memory access")

        with self.assertRaisesRegex(RuntimeError, "every entrant failed to time"):
            race(entrants, explode, **RACE_ARGS)


class ProcessDied(BaseException):
    """Stands in for a GPU fault that takes the whole process down."""


class TestJournalReplay(unittest.TestCase):
    TRUTH: ClassVar[dict] = {
        "fast": 100.0,
        "near": 100.6,
        "slow": 250.0,
        "slowest": 700.0,
    }

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "race.jsonl")
        self.entrants = [RaceEntrant(label) for label in self.TRUTH]

    def _race(self, timer, resume=False, **overrides):
        return race(
            self.entrants,
            timer,
            journal=JsonlBlockJournal(self.path, resume=resume),
            resume=resume,
            **{**RACE_ARGS, "key": "shape-a", **overrides},
        )

    def _records(self):
        with open(self.path) as handle:
            return [json.loads(line) for line in handle]

    @staticmethod
    def refuse(entrant, count):
        raise AssertionError("replay must not re-measure a finished block")

    def test_a_complete_journal_replays_without_measuring_anything(self):
        live = self._race(constant_timer(self.TRUTH, seed=9))
        replayed = self._race(self.refuse, resume=True)
        self.assertEqual(replayed.blocks_replayed, live.blocks_run)
        self.assertEqual(
            {(v.label, v.state, v.estimate) for v in live.verdicts},
            {(v.label, v.state, v.estimate) for v in replayed.verdicts},
        )
        self.assertEqual(live.winner, replayed.winner)
        self.assertEqual(live.calls_spent, replayed.calls_spent)

    def test_a_journal_from_another_race_is_refused(self):
        self._race(constant_timer(self.TRUTH, seed=9))
        for overrides in (
            {"key": "shape-b"},
            {"min_improvement_pct": 50.0},
            {"policy": RacePolicy(block_calls=1, warmup_calls=0)},
            {"seed": 7},
        ):
            with (
                self.subTest(**{k: str(v) for k, v in overrides.items()}),
                self.assertRaisesRegex(ValueError, "different race"),
            ):
                self._race(self.refuse, resume=True, **overrides)

    def test_the_journal_records_the_visit_order_it_actually_used(self):
        self._race(constant_timer(self.TRUTH, seed=9))
        blocks = [r for r in self._records() if "latencies" in r]
        self.assertTrue(blocks)
        for record in blocks:
            self.assertEqual(sorted(record["order"]), sorted(record["latencies"]))

    def test_the_entrant_running_when_the_process_died_is_not_run_again(self):
        """A kernel that took the process down would take the resume down
        too, so it is journaled before it runs and treated as crashed."""
        honest = constant_timer(self.TRUTH, seed=9)
        visits = {"slowest": 0}

        def dies_on_second_visit(entrant, count):
            if entrant.label == "slowest":
                visits["slowest"] += 1
                if visits["slowest"] == 2:
                    raise ProcessDied()
            return honest(entrant, count)

        with self.assertRaises(ProcessDied):
            self._race(dies_on_second_visit)

        def never_slowest(entrant, count):
            if entrant.label == "slowest":
                raise AssertionError("the entrant that killed the run ran again")
            return honest(entrant, count)

        resumed = self._race(never_slowest, resume=True)
        states = {v.label: v for v in resumed.verdicts}
        self.assertEqual(states["slowest"].state, "crashed")
        self.assertIn("process died", states["slowest"].note)
        self.assertEqual(resumed.blocks_replayed, 1)

        # And a second resume still knows, from the journal alone.
        again = self._race(self.refuse, resume=True)
        self.assertEqual(
            {v.label: v.state for v in again.verdicts}["slowest"], "crashed"
        )

    def test_a_crash_is_journaled_so_a_resume_does_not_rerun_it(self):
        honest = constant_timer(self.TRUTH, seed=9)

        def slow_faults(entrant, count):
            if entrant.label == "slow":
                raise RuntimeError("HIP error: illegal memory access")
            return honest(entrant, count)

        live = self._race(slow_faults)
        replayed = self._race(self.refuse, resume=True)
        self.assertEqual(replayed.blocks_replayed, live.blocks_run)
        self.assertEqual(
            {v.label: v.state for v in replayed.verdicts}["slow"], "crashed"
        )

    def test_a_torn_final_line_is_dropped_rather_than_failing_the_resume(self):
        self._race(constant_timer(self.TRUTH, seed=9))
        with open(self.path) as handle:
            lines = handle.readlines()
        with open(self.path, "w") as handle:
            handle.writelines(lines[:-1])
            handle.write('{"block": 99, "order": ["fa')

        journal = JsonlBlockJournal(self.path, resume=True)
        self.assertEqual(len(journal.records()), len(lines) - 1)

    def test_a_second_resume_keeps_what_the_first_one_finished(self):
        """The torn line has to go, not just be read around."""
        self._race(constant_timer(self.TRUTH, seed=9))
        with open(self.path) as handle:
            lines = handle.readlines()
        with open(self.path, "w") as handle:
            handle.writelines(lines[:-1])
            handle.write('{"block": 99, "order": ["fa')

        first = JsonlBlockJournal(self.path, resume=True)
        first.append({"block": len(lines), "order": [], "latencies": {}})
        second = JsonlBlockJournal(self.path, resume=True)
        self.assertEqual(len(second.records()), len(lines))

    def test_resuming_continues_instead_of_starting_over(self):
        capped = RacePolicy(block_calls=10, warmup_calls=0, max_blocks=4)
        first = self._race(constant_timer(self.TRUTH, seed=3), policy=capped)
        resumed = self._race(self.refuse, resume=True, policy=capped)
        self.assertEqual(resumed.blocks_replayed, first.blocks_run)

    def test_a_journal_missing_an_active_entrant_is_not_replayed(self):
        """Replaying blocks an entrant never ran would pair each survivor's
        block n against a block the entrant was absent from."""
        policy = RacePolicy(block_calls=10, warmup_calls=0, max_blocks=6)
        self._race(constant_timer(self.TRUTH, seed=3), policy=policy)
        records = self._records()
        blocks = [r for r in records if "latencies" in r]
        dropped = blocks[-1]["order"][0]
        for record in blocks[len(blocks) // 2 :]:
            record["order"] = [n for n in record["order"] if n != dropped]
            record["latencies"].pop(dropped, None)
        with open(self.path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        resumed = self._race(
            constant_timer(self.TRUTH, seed=11), resume=True, policy=policy
        )
        self.assertEqual(resumed.blocks_replayed, len(blocks) // 2)
        survivors = [
            v.label
            for v in resumed.verdicts
            if v.state not in ("eliminated", "crashed")
        ]
        lengths = {len(resumed.samples[label].block_medians) for label in survivors}
        self.assertEqual(lengths, {resumed.blocks_run})


if __name__ == "__main__":
    unittest.main()
