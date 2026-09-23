# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for how the MHA forward tuner chooses what to measure and
what to publish: the candidate sample, the backend restriction, and the
gate that refuses to displace a configuration it cannot beat."""

import json
import os
import tempfile
import types
import unittest
from typing import ClassVar

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.
import pandas as pd

from aiter.ops.mha_fwd_policy import (
    MHA_FWD_RUNTIME_CSV_FIELDS,
    MhaFwdCandidate,
    MhaFwdProblem,
    enumerate_mha_fwd_candidates,
)
from aiter.utility.block_race import RaceEntrant
from op_tests.tuners.tune_mha_fwd import MhaFwdTuner


class TestIncumbentGate(unittest.TestCase):
    """A tuning run must be able to tell an improvement from a regression."""

    KEY = ("gfx950", 256, "mi355x")
    INCUMBENT_CONFIG = '{"BLOCK_M":128,"BLOCK_N":64}'
    CHALLENGER_CONFIG = '{"BLOCK_M":256,"BLOCK_N":64}'

    def _tuner(self):
        tuner = MhaFwdTuner.__new__(MhaFwdTuner)
        tuner._incumbents_by_key = {self.KEY: {("gluon", self.INCUMBENT_CONFIG)}}
        tuner._autoselect_by_key = {}
        tuner._incumbent_probe_by_key = {}
        tuner._published_by_key = {}
        tuner._promotions = []
        tuner._race_winner_by_key = {}
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

    def test_a_race_that_certified_nobody_refuses_to_publish(self):
        """Screening has no recorded winner and ranks by latency, so its
        fastest row is the pick. A race that recorded no winner is a different
        situation: taking the fastest row there could ship a candidate the
        race eliminated."""
        tuner = self._tuner()
        tuner._race_winner_by_key = {self.KEY: None}
        with self.assertRaises(RuntimeError):
            tuner._gate_against_incumbent(
                self.KEY, self._frame(challenger_us=800.0, incumbent_us=1000.0)
            )

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

    def test_a_winner_slower_than_auto_select_publishes_no_row(self):
        """The incumbent is whatever dispatch resolves, which for some shapes
        is a backend the catalogue cannot name -- asm_v3 leaves its split
        count to C++. Gating only against the tile backends published rows
        1.5x slower than the GPU already ran, so an unnameable incumbent has
        to be gated on its probe latency instead."""
        tuner = self._tuner()
        tuner._incumbents_by_key = {}
        tuner._autoselect_by_key = {
            self.KEY: {"identity": ("asm_v3", 0, ""), "latency_us": 1108.0}
        }
        winner = tuner._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=1628.0, incumbent_us=1900.0)
        )
        self.assertIsNone(winner, "a slower winner must not reach the table")
        self.assertEqual(tuner._promotions[0]["decision"], "autoselect_retained")

    def test_a_winner_that_beats_auto_select_is_published(self):
        tuner = self._tuner()
        tuner._incumbents_by_key = {}
        tuner._autoselect_by_key = {
            self.KEY: {"identity": ("opus", 0, ""), "latency_us": 1108.0}
        }
        winner = tuner._gate_against_incumbent(
            self.KEY, self._frame(challenger_us=800.0, incumbent_us=1900.0)
        )
        self.assertEqual(winner["backend_config"], self.CHALLENGER_CONFIG)
        self.assertIn("beat auto-select opus", winner["detail"])
        self.assertEqual(tuner._promotions[0]["decision"], "promoted")

    def test_an_auto_select_win_inside_delta_still_publishes_no_row(self):
        """1% against a 2% indifference delta is not a result. Publishing it
        would pin a configuration that is indistinguishable from the one the
        shape already reaches without any row at all."""
        tuner = self._tuner()
        tuner._incumbents_by_key = {}
        tuner._autoselect_by_key = {
            self.KEY: {"identity": ("opus", 0, ""), "latency_us": 1000.0}
        }
        self.assertIsNone(
            tuner._gate_against_incumbent(
                self.KEY, self._frame(challenger_us=990.0, incumbent_us=1900.0)
            )
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


class TestIncumbentIdentity(unittest.TestCase):
    """The incumbent has to be what dispatch resolves, not a fixed backend."""

    def _tuner(self):
        return MhaFwdTuner.__new__(MhaFwdTuner)

    @staticmethod
    def _row(hdim_q, hdim_v):
        return pd.Series(
            {
                "dtype": "bfloat16",
                "dropout_p": 0.0,
                "hdim_q": hdim_q,
                "hdim_v": hdim_v,
            }
        )

    def test_a_resolvable_selection_becomes_the_measured_incumbent(self):
        incumbents = self._tuner()._incumbent_candidates(
            {"identity": ("opus", 0, ""), "latency_us": 1108.0}
        )
        self.assertEqual([c.identity for c in incumbents], [("opus", 0, "")])

    def test_an_unnameable_selection_yields_no_candidate(self):
        """asm_v3 with no forced split is what dispatch reports today, and 0
        is not a legal candidate split. Inventing a split here would measure
        a configuration the shape does not currently run."""
        self.assertEqual(
            self._tuner()._incumbent_candidates(
                {"identity": ("asm_v3", 0, ""), "latency_us": 1108.0}
            ),
            [],
        )

    def test_no_selection_leaves_the_field_alone(self):
        self.assertEqual(self._tuner()._incumbent_candidates(None), [])

    def test_the_shipped_tile_default_follows_the_head_dims(self):
        """The kernel wrappers resolve tiles with has_pe taken from the head
        dims, so an asymmetric shape resolves a different tile. Hardcoding it
        injected a default that hd192/128 never launches."""
        from aiter.ops.triton._gluon_kernels.gfx950.attention.mha import _get_config

        with_pe = _get_config(is_fp8=False, has_pe=True)
        without_pe = _get_config(is_fp8=False, has_pe=False)
        self.assertNotEqual(with_pe, without_pe, "the shapes must differ to test")

        tuner = self._tuner()

        def gluon_default(row):
            return next(
                candidate.backend_config
                for candidate in tuner._shipped_tile_candidates(row)
                if candidate.backend == "gluon"
            )

        self.assertEqual(gluon_default(self._row(192, 128)), with_pe)
        self.assertEqual(gluon_default(self._row(128, 128)), without_pe)


class TestRetunedIncumbent(unittest.TestCase):
    """Re-tuning a shape has to defend the row it already published.

    Gating a re-tune against auto-select instead compares the field against a
    bar the published row already cleared, so anything a contended sweep ranks
    a fraction ahead of that row replaces it.
    """

    ROW: ClassVar[dict] = {
        "gfx": "gfx950",
        "gpu_model": "mi355x",
        "cu_num": 256,
        "mode": "varlen",
        "batch": 1,
        "total_q": 4096,
        "total_k": 42700,
        "max_seqlen_q": 4096,
        "max_seqlen_k": 42700,
        "min_seqlen_q": 1,
        "nhead_q": 12,
        "nhead_k": 12,
        "hdim_q": 192,
        "hdim_v": 128,
        "dtype": "bf16",
        "causal": False,
        "window_left": -1,
        "window_right": -1,
        "sink_size": 0,
        "dropout_p": 0.0,
        "logits_soft_cap": 0.0,
        "how_v3_bf16_cvt": 0,
        "return_lse": False,
        "return_attn_probs": False,
        "has_bias": False,
        "has_alibi": False,
        "has_sink": False,
        "has_block_table": False,
        "has_q_descale": False,
        "has_physical_padding": False,
        "is_grad": False,
    }
    AUTOSELECT: ClassVar[dict] = {"identity": ("ck", 0, ""), "latency_us": 1900.0}
    PUBLISHED: ClassVar[dict] = {
        "identity": ("gluon", 0, '{"BLOCK_M":128,"BLOCK_N":64}'),
        "latency_us": 900.0,
    }

    def _tuner(self, probe=None):
        tuner = MhaFwdTuner.__new__(MhaFwdTuner)
        tuner._published_by_key = {}
        tuner._probe_calls = []

        def record(row, config_file):
            tuner._probe_calls.append(config_file)
            return probe

        tuner._run_fresh_probe = record
        return tuner

    def _key(self):
        return MhaFwdProblem.from_mapping(self.ROW).key()

    def _write_table(self, **overrides):
        row = {
            **self.ROW,
            "backend": "gluon",
            "num_splits": 0,
            "backend_config": "",
            "us": 42.0,
        }
        row.update(overrides)
        handle, path = tempfile.mkstemp(prefix="tuned-", suffix=".csv")
        os.close(handle)
        self.addCleanup(os.unlink, path)
        pd.DataFrame([row])[list(MHA_FWD_RUNTIME_CSV_FIELDS)].to_csv(path, index=False)
        return types.SimpleNamespace(tune_file=path)

    def test_a_shape_with_no_published_row_reuses_the_auto_select_probe(self):
        """The two answers are the same when the table has no row for the
        shape, and probing again would spend a second process to learn it."""
        tuner = self._tuner()
        resolved = tuner._resolve_incumbent(
            self.ROW, types.SimpleNamespace(tune_file=""), self._key(), self.AUTOSELECT
        )
        self.assertIs(resolved, self.AUTOSELECT)
        self.assertEqual(tuner._probe_calls, [])

    def test_a_published_row_becomes_the_incumbent(self):
        probe = {
            "status": "verified",
            "observed": {
                "backend": "gluon",
                "num_splits": 0,
                "backend_config": '{"BLOCK_M":128,"BLOCK_N":64}',
            },
            "latency_us": 900.0,
        }
        args = self._write_table(backend_config='{"BLOCK_M":128,"BLOCK_N":64}')
        tuner = self._tuner(probe)
        tuner._published_by_key = tuner._load_published_table(args)
        resolved = tuner._resolve_incumbent(
            self.ROW, args, self._key(), self.AUTOSELECT
        )
        self.assertEqual(resolved["identity"], self.PUBLISHED["identity"])
        self.assertEqual(resolved["latency_us"], 900.0)
        self.assertEqual(tuner._probe_calls, [args.tune_file])

    def test_a_published_row_that_does_not_resolve_falls_back_to_auto_select(self):
        """A row dispatch cannot honor is a warning and a fallback at runtime,
        so the tuner has nothing to defend and gates against auto-select."""
        args = self._write_table()
        tuner = self._tuner({"status": "failed", "stderr": "no kernel gate matched"})
        tuner._published_by_key = tuner._load_published_table(args)
        self.assertIs(
            tuner._resolve_incumbent(self.ROW, args, self._key(), self.AUTOSELECT),
            self.AUTOSELECT,
        )

    def test_the_published_table_is_keyed_the_way_the_sweep_is(self):
        """The CSV spells the dtype bf16 and the sweep spells it bfloat16, so
        keying on raw cells would miss the row it is meant to defend."""
        args = self._write_table()
        published = MhaFwdTuner.__new__(MhaFwdTuner)._load_published_table(args)
        self.assertEqual(list(published), [self._key()])
        self.assertEqual(published[self._key()], ("gluon", 0, ""))

    def test_a_re_tune_is_gated_on_the_published_row_not_auto_select(self):
        """1000 us beats the 1900 us auto-select handily and loses to the
        900 us row already published, and losing is the answer that counts."""
        tuner = self._tuner()
        key = self._key()
        tuner._published_by_key = {key: ("gluon", 0, "")}
        tuner._incumbents_by_key = {}
        tuner._incumbent_probe_by_key = {key: self.PUBLISHED}
        tuner._autoselect_by_key = {key: self.AUTOSELECT}
        tuner._promotions = []
        tuner._race_winner_by_key = {}
        frame = pd.DataFrame(
            [
                {
                    "backend": "opus",
                    "backend_config": "",
                    "us": 1000.0,
                    "samples_us": "[1000.0,1001.0]",
                }
            ]
        )
        self.assertIsNone(tuner._gate_against_incumbent(key, frame))
        self.assertEqual(tuner._promotions[0]["decision"], "incumbent_retained")


class TestRaceJournalBinding(unittest.TestCase):
    """Replay is only meaningful for the race that produced the blocks."""

    ARGS = types.SimpleNamespace(
        delta=0.02,
        race_alpha=0.05,
        race_block_calls=20,
        race_min_blocks=3,
        race_max_blocks=30,
    )
    KEY = ("gfx950", "mi355x", 256)

    def _path(self, candidates, args=None, key=None):
        tuner = MhaFwdTuner.__new__(MhaFwdTuner)
        tuner._journal_path = "/tmp/run.journal.jsonl"
        return tuner._race_block_journal(args or self.ARGS, key or self.KEY, candidates)

    @staticmethod
    def _candidates(*names, protected=()):
        return [
            RaceEntrant(
                label=name,
                payload=MhaFwdCandidate(name),
                protected=name in protected,
            )
            for name in names
        ]

    def test_the_same_race_resumes_the_same_journal(self):
        first = self._path(self._candidates("ck", "opus"))
        second = self._path(self._candidates("opus", "ck"))
        self.assertEqual(first, second, "candidate order must not split the journal")

    def test_a_changed_catalogue_lands_on_a_different_journal(self):
        """The finding this closes: the name digested only the shape, so
        editing the candidate list and resuming replayed blocks from a race
        over a different field."""
        self.assertNotEqual(
            self._path(self._candidates("ck", "opus")),
            self._path(self._candidates("ck", "opus", "triton")),
        )

    def test_a_changed_threshold_lands_on_a_different_journal(self):
        for field, value in (
            ("delta", 0.05),
            ("race_alpha", 0.01),
            ("race_block_calls", 40),
            ("race_min_blocks", 8),
            ("race_max_blocks", 10),
        ):
            with self.subTest(field=field):
                changed = types.SimpleNamespace(**vars(self.ARGS))
                setattr(changed, field, value)
                self.assertNotEqual(
                    self._path(self._candidates("ck")),
                    self._path(self._candidates("ck"), args=changed),
                )

    def test_a_changed_incumbent_lands_on_a_different_journal(self):
        """Who is protected decides who leaves the field, and who leaves
        decides which blocks the survivors are paired over. A re-tune after
        the published row moved is a different race."""
        self.assertNotEqual(
            self._path(self._candidates("ck", "opus")),
            self._path(self._candidates("ck", "opus", protected=("ck",))),
        )

    def test_two_shapes_do_not_share_a_journal(self):
        # The shape key opens with arch, SKU and CU count, so a journal
        # carried to another GPU lands on a different file too.
        self.assertNotEqual(
            self._path(self._candidates("ck")),
            self._path(self._candidates("ck"), key=("gfx942", "mi325x", 304)),
        )


class TestCandidateSample(unittest.TestCase):
    """--candidate-sample is how a run shrinks its field, so two runs given
    the same sample have to measure the same candidates."""

    ROW = types.SimpleNamespace(gfx="gfx950")
    KEY = ("gfx950", "mi355x", 256)

    def _field(self, sample, backends=""):
        tuner = MhaFwdTuner.__new__(MhaFwdTuner)
        tuner._args = types.SimpleNamespace(backends=backends)
        tuner._autoselect_by_key = {}
        tuner._incumbent_probe_by_key = {}
        tuner._incumbents_by_key = {}
        tuner._resolve_autoselect = lambda row: None
        tuner._resolve_incumbent = lambda row, args, key, autoselect: None
        tuner._shipped_tile_candidates = lambda row: []
        tuner._incumbent_candidates = lambda selection: []
        args = types.SimpleNamespace(candidate_sample=sample)
        return [c.identity for c in tuner.candidate_field(self.ROW, self.KEY, args)]

    def test_a_sample_is_drawn_from_the_catalogue_and_is_reproducible(self):
        full = {c.identity for c in enumerate_mha_fwd_candidates("gfx950")}
        first = self._field(5)
        self.assertEqual(len(first), 5)
        self.assertTrue(set(first) <= full)
        self.assertEqual(first, self._field(5))

    def test_the_sample_composes_with_the_backend_restriction(self):
        field = self._field(3, backends="triton")
        self.assertEqual(len(field), 3)
        self.assertEqual({identity[0] for identity in field}, {"triton"})


class TestBackendRestriction(unittest.TestCase):
    """The restriction is a control for comparing contracts, so it has to
    actually restrict and has to be visible when it does."""

    def test_only_the_named_backends_are_measured(self):
        candidates = enumerate_mha_fwd_candidates("gfx950", ["triton", "gluon"])
        self.assertEqual(sorted({c.backend for c in candidates}), ["gluon", "triton"])

    def test_the_restricted_catalogue_is_a_subset_of_the_full_one(self):
        full = {c.identity for c in enumerate_mha_fwd_candidates("gfx950")}
        restricted = {
            c.identity for c in enumerate_mha_fwd_candidates("gfx950", ["gluon"])
        }
        self.assertTrue(restricted <= full)
        self.assertLess(len(restricted), len(full))

    def test_no_restriction_is_the_full_catalogue(self):
        self.assertEqual(
            [c.identity for c in enumerate_mha_fwd_candidates("gfx950")],
            [c.identity for c in enumerate_mha_fwd_candidates("gfx950", None)],
        )

    def test_an_unknown_backend_is_rejected_rather_than_silently_empty(self):
        with self.assertRaisesRegex(ValueError, "unknown MHA backends"):
            enumerate_mha_fwd_candidates("gfx950", ["trition"])


if __name__ == "__main__":
    unittest.main()
