# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the interleaved elimination race.

Every test here drives the race from synthetic latencies, so what is under
test is the decision logic rather than a GPU. That is the point: the questions
these answer -- whether a protected incumbent can be eliminated, whether a
resumed race reaches the same verdict, which of several equally fast
configurations gets published -- all have exact answers that a real
measurement would only obscure.
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
    Samples,
    critical_t,
    measure_blocks,
    race,
    select_winner,
)

RACE_ARGS = {
    "delta": 0.02,
    "alpha": 0.05,
    "block_calls": 10,
    "min_blocks": 3,
    "max_blocks": 30,
    "seed": 4242,
    "verbose": False,
}


def constant_timer(latency_by_label, noise=0.004, seed=1):
    """A fake GPU: each label has a true latency, plus reproducible jitter."""
    rng = random.Random(seed)

    def time_calls(entrant, count):
        true = latency_by_label[entrant.label]
        return [true * (1.0 + rng.gauss(0.0, noise)) for _ in range(count)]

    return time_calls


def samples_from(block_medians):
    samples = Samples()
    samples.blocks = [[value] for value in block_medians]
    return samples


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
                RACE_ARGS["min_blocks"],
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
        """Surviving elimination is not the same as being tied. A candidate
        noisy enough that the race cannot prove it is worse still must not be
        published as an equal of the leader, or the tie-break would rank a
        steady-looking configuration ahead of one that is genuinely faster."""
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
        result = race(entrants, scripted_timer, **dict(RACE_ARGS, max_blocks=6))

        states = {v.label: v.state for v in result.verdicts}
        self.assertEqual(
            states["noisy"],
            "undecided",
            "the race cannot separate these, so it must not claim to have",
        )
        self.assertNotIn(
            "noisy",
            result.survivors,
            f"a candidate measured at {result.samples['noisy'].estimate:.0f} us "
            f"against a leader at {result.samples['leader'].estimate:.0f} us is "
            "not inside a 2% indifference zone",
        )
        self.assertEqual(result.winner, "leader")
        self.assertFalse(
            result.certified,
            "a race that never separated the field has not certified anything",
        )

    def test_a_race_that_certifies_stops_before_the_ceiling(self):
        truth = {"fast": 100.0, "slow": 500.0}
        result = race(
            [RaceEntrant(label) for label in truth],
            constant_timer(truth),
            **RACE_ARGS,
        )
        self.assertTrue(result.certified)
        self.assertLess(result.blocks_run, RACE_ARGS["max_blocks"])


class TestProtectedIncumbent(unittest.TestCase):
    """A run that stops measuring the configuration already in use cannot tell
    an improvement from a regression."""

    def test_a_protected_incumbent_is_never_eliminated(self):
        truth = {"challenger": 100.0, "incumbent": 900.0}
        entrants = [
            RaceEntrant("challenger"),
            RaceEntrant("incumbent", protected=True),
        ]
        result = race(entrants, constant_timer(truth), **RACE_ARGS)

        states = {v.label: v.state for v in result.verdicts}
        self.assertNotEqual(states["incumbent"], "eliminated")
        self.assertEqual(states["incumbent"], "protected_behind")

    def test_an_incumbent_kept_in_the_field_is_not_reported_as_a_tie(self):
        """Being unkillable is not the same as being close. If the incumbent
        is behind, the winner is a real improvement and the report has to say
        so, or the guard would launder a regression into a tie."""
        truth = {"challenger": 100.0, "incumbent": 900.0}
        entrants = [
            RaceEntrant("challenger"),
            RaceEntrant("incumbent", protected=True),
        ]
        result = race(entrants, constant_timer(truth), **RACE_ARGS)

        self.assertNotIn("incumbent", result.survivors)
        self.assertEqual(result.winner, "challenger")

    def test_a_protected_incumbent_is_measured_in_every_block(self):
        truth = {"challenger": 100.0, "incumbent": 900.0}
        entrants = [
            RaceEntrant("challenger"),
            RaceEntrant("incumbent", protected=True),
        ]
        result = race(entrants, constant_timer(truth), **RACE_ARGS)

        self.assertEqual(
            len(result.samples["incumbent"].blocks),
            result.blocks_run,
            "the incumbent has to keep being measured, or the comparison "
            "against it goes stale while the race continues",
        )


class TestTieBreak(unittest.TestCase):
    """Inside the indifference zone nothing is meaningfully faster, so the
    rule's job is to make a re-tune return the same answer."""

    def test_a_tied_incumbent_is_kept_over_a_nominally_faster_challenger(self):
        samples = {
            "challenger": samples_from([100.0, 100.0, 100.0]),
            "incumbent": samples_from([100.5, 100.5, 100.5]),
        }
        by_label = {
            "challenger": RaceEntrant("challenger"),
            "incumbent": RaceEntrant("incumbent", protected=True),
        }
        winner, rule = select_winner(["challenger", "incumbent"], samples, by_label)
        self.assertEqual(winner, "incumbent")
        self.assertEqual(rule, "incumbent_retained")

    def test_the_steadier_of_two_tied_challengers_wins(self):
        """Deliberately give the steady one the worse median. Inside the
        indifference zone the median is the noisier thing to choose on, and
        the steadier configuration is the one more likely to still be fast in
        the next session."""
        samples = {
            "jumpy": samples_from([95.0, 105.0, 90.0, 110.0]),
            "steady": samples_from([100.4, 100.5, 100.6, 100.5]),
        }
        by_label = {
            "jumpy": RaceEntrant("jumpy"),
            "steady": RaceEntrant("steady"),
        }
        self.assertLess(
            samples["jumpy"].estimate,
            samples["steady"].estimate,
            "the test is only meaningful if the jumpy one looks faster",
        )
        winner, rule = select_winner(["jumpy", "steady"], samples, by_label)
        self.assertEqual(winner, "steady")
        self.assertEqual(rule, "lowest_spread")

    def test_exactly_tied_spreads_fall_back_to_a_stable_order(self):
        samples = {
            "b_config": samples_from([100.0, 101.0, 100.0]),
            "a_config": samples_from([100.0, 101.0, 100.0]),
        }
        by_label = {
            "b_config": RaceEntrant("b_config"),
            "a_config": RaceEntrant("a_config"),
        }
        first, rule = select_winner(["b_config", "a_config"], samples, by_label)
        second, _ = select_winner(["a_config", "b_config"], samples, by_label)
        self.assertEqual(first, second)
        self.assertEqual(rule, "stable_sort")

    def test_the_input_order_of_survivors_does_not_change_the_pick(self):
        samples = {
            "one": samples_from([100.0, 101.0, 99.0]),
            "two": samples_from([100.2, 100.3, 100.1]),
            "three": samples_from([100.1, 102.0, 98.5]),
        }
        by_label = {label: RaceEntrant(label) for label in samples}
        picks = {
            select_winner(list(order), samples, by_label)[0]
            for order in (
                ("one", "two", "three"),
                ("three", "one", "two"),
                ("two", "three", "one"),
            )
        }
        self.assertEqual(len(picks), 1, f"selection depended on input order: {picks}")


class TestStrategyAgreement(unittest.TestCase):
    """The claim the race exists to support: it reaches the answer exhaustive
    screening reaches, for a fraction of the calls. Nothing else in the suite
    would fail if the two strategies disagreed."""

    @staticmethod
    def _field(seed):
        # One clear winner, a near-tie beside it, and a spread of slower
        # candidates: the shape of a real tile-config catalogue.
        rng = random.Random(seed)
        truth = {"fast": 100.0, "near": 101.5}
        truth.update({f"slow{i}": rng.uniform(115.0, 400.0) for i in range(18)})
        return truth

    def test_the_race_picks_what_measuring_everything_would_pick(self):
        delta = RACE_ARGS["delta"]
        for seed in range(6):
            with self.subTest(seed=seed):
                truth = self._field(seed)
                entrants = [RaceEntrant(label) for label in truth]

                raced = race(entrants, constant_timer(truth, seed=seed), **RACE_ARGS)
                screened = measure_blocks(
                    entrants,
                    constant_timer(truth, seed=seed),
                    block_calls=RACE_ARGS["block_calls"],
                    blocks=RACE_ARGS["max_blocks"],
                    seed=RACE_ARGS["seed"],
                )
                exhaustive = min(screened, key=lambda label: screened[label].estimate)

                self.assertLessEqual(
                    truth[raced.winner],
                    truth[exhaustive] * (1.0 + delta),
                    f"race published {raced.winner} where measuring every "
                    f"candidate published {exhaustive}",
                )

    def test_the_race_gets_there_on_fewer_calls(self):
        truth = self._field(0)
        entrants = [RaceEntrant(label) for label in truth]
        raced = race(entrants, constant_timer(truth, seed=0), **RACE_ARGS)
        exhaustive_calls = (
            len(entrants) * RACE_ARGS["max_blocks"] * RACE_ARGS["block_calls"]
        )
        self.assertLess(raced.calls_spent, exhaustive_calls / 2)


class TestUnusableMeasurements(unittest.TestCase):
    """A candidate that cannot be timed must leave, not decide the race."""

    TRUTH: ClassVar[dict] = {"fast": 100.0, "near": 100.6, "slow": 250.0}

    def _race_with(self, broken_label, broken_timer):
        entrants = [RaceEntrant(label) for label in self.TRUTH]
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

    def test_a_zero_latency_cannot_win(self):
        """Zero is a broken reading, not a fast one, and the leader's
        estimate divides the tolerance and every reported gap."""
        result = self._race_with("slow", lambda count: [0.0] * count)
        states = {v.label: v.state for v in result.verdicts}
        self.assertEqual(states["slow"], "crashed")
        self.assertEqual(result.winner, "fast")

    def test_a_negative_latency_cannot_win(self):
        result = self._race_with("slow", lambda count: [-1.0] * count)
        self.assertEqual(result.winner, "fast")

    def test_a_field_that_cannot_be_timed_at_all_fails_loudly(self):
        entrants = [RaceEntrant(label) for label in self.TRUTH]

        def explode(entrant, count):
            raise RuntimeError("HIP error: illegal memory access")

        with self.assertRaisesRegex(RuntimeError, "every entrant failed to time"):
            race(entrants, explode, **RACE_ARGS)


class TestJournalReplay(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "race.jsonl")

    def _run_and_journal(self):
        truth = {"fast": 100.0, "near": 100.6, "slow": 250.0, "slowest": 700.0}
        entrants = [RaceEntrant(label) for label in truth]
        live = race(
            entrants,
            constant_timer(truth, seed=9),
            journal=JsonlBlockJournal(self.path),
            **RACE_ARGS,
        )
        return entrants, live

    def test_a_complete_journal_replays_without_measuring_anything(self):
        entrants, live = self._run_and_journal()

        def refuse(entrant, count):
            raise AssertionError("replay must not re-measure a finished block")

        replayed = race(
            entrants,
            refuse,
            journal=JsonlBlockJournal(self.path, resume=True),
            resume=True,
            **RACE_ARGS,
        )
        self.assertEqual(replayed.blocks_replayed, live.blocks_run)
        self.assertEqual(
            {(v.label, v.state, v.estimate) for v in live.verdicts},
            {(v.label, v.state, v.estimate) for v in replayed.verdicts},
        )
        self.assertEqual(live.winner, replayed.winner)
        self.assertEqual(live.survivors, replayed.survivors)

    def test_the_journal_records_the_visit_order_it_actually_used(self):
        """Re-deriving the order from the seed would be one assumption away
        from a silent mismatch, and it would leave the position-effect
        analysis with nothing to check against."""
        self._run_and_journal()
        with open(self.path) as handle:
            records = [json.loads(line) for line in handle]
        self.assertTrue(records)
        for record in records:
            self.assertIn("order", record)
            self.assertEqual(
                sorted(record["order"]),
                sorted(record["latencies"]),
                "every candidate visited in a block must have latencies stored",
            )

    def test_a_torn_final_line_is_dropped_rather_than_failing_the_resume(self):
        """A process killed mid-write leaves one truncated line. Losing the
        last block is correct; refusing to resume is not."""
        self._run_and_journal()
        with open(self.path) as handle:
            lines = handle.readlines()
        with open(self.path, "w") as handle:
            handle.writelines(lines[:-1])
            handle.write('{"block": 99, "order": ["fa')

        journal = JsonlBlockJournal(self.path, resume=True)
        self.assertEqual(len(list(journal.records())), len(lines) - 1)

    def test_a_second_resume_keeps_what_the_first_one_finished(self):
        """The torn line has to go, not just be read around. Appending after
        it would leave every later block behind unparseable JSON."""
        self._run_and_journal()
        with open(self.path) as handle:
            lines = handle.readlines()
        with open(self.path, "w") as handle:
            handle.writelines(lines[:-1])
            handle.write('{"block": 99, "order": ["fa')

        first = JsonlBlockJournal(self.path, resume=True)
        first.append({"block": len(lines), "order": [], "latencies": {}})

        second = JsonlBlockJournal(self.path, resume=True)
        self.assertEqual(
            len(list(second.records())),
            len(lines),
            "the block the first resume completed must survive the second",
        )

    def test_resuming_continues_instead_of_starting_over(self):
        truth = {"fast": 100.0, "near": 100.6, "slow": 250.0}
        entrants = [RaceEntrant(label) for label in truth]
        capped = dict(RACE_ARGS, max_blocks=4)
        race(
            entrants,
            constant_timer(truth, seed=3),
            journal=JsonlBlockJournal(self.path),
            **capped,
        )
        journaled = len(list(JsonlBlockJournal(self.path, resume=True).records()))

        measured_blocks = []

        def counting_timer(entrant, count):
            measured_blocks.append(entrant.label)
            return [truth[entrant.label]] * count

        resumed = race(
            entrants,
            counting_timer,
            journal=JsonlBlockJournal(self.path, resume=True),
            resume=True,
            **capped,
        )
        self.assertEqual(resumed.blocks_replayed, journaled)
        self.assertEqual(
            measured_blocks,
            [],
            "a resume that already has every block should measure nothing",
        )

    def test_a_journal_missing_an_active_entrant_is_not_replayed(self):
        """The alternative is worse than losing the blocks: an entrant that
        left the first race early has no latencies in the later journalled
        blocks, so replaying them would pair each survivor's block n against
        a block the entrant never ran."""
        truth = {"fast": 100.0, "near": 100.6, "slow": 250.0}
        entrants = [RaceEntrant(label) for label in truth]
        race(
            entrants,
            constant_timer(truth, seed=3),
            journal=JsonlBlockJournal(self.path),
            **dict(RACE_ARGS, max_blocks=6),
        )
        with open(self.path) as handle:
            records = [json.loads(line) for line in handle]
        dropped = records[-1]["order"][0]
        for record in records[len(records) // 2 :]:
            record["order"] = [n for n in record["order"] if n != dropped]
            record["latencies"].pop(dropped, None)
        with open(self.path, "w") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

        resumed = race(
            entrants,
            constant_timer(truth, seed=11),
            journal=JsonlBlockJournal(self.path, resume=True),
            resume=True,
            **dict(RACE_ARGS, max_blocks=6),
        )
        self.assertEqual(resumed.blocks_replayed, len(records) // 2)
        lengths = {len(samples.block_medians) for samples in resumed.samples.values()}
        self.assertEqual(
            lengths,
            {resumed.blocks_run},
            "every entrant that ran to the end must hold one block per look",
        )


if __name__ == "__main__":
    unittest.main()
