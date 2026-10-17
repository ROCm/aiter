# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Level 1: what GatedTuner lets into a tuned CSV (no GPU required).

A stub family scripts what auto-select picks, how fast each candidate ran
and which proofs fail, so each test pins one publication rule.
"""

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from aiter.utility.gated_tuner import GatedTuner
from aiter.utility.selection_proof import SelectionProof

KEYS = ["gfx", "gpu_model", "cu_num", "m"]
HARDWARE = ("gfx950", "mi355x", "256")


class StubTuner(GatedTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GatedTuner.ARG_DEFAULTS,
        "config_env_name": "STUB_CONFIG",
    }
    SELECTION_PROBE_MODULE = "unused"
    RUNTIME_FIELDS = (*KEYS, "backend", "us")
    CANDIDATE_FIELDS = ("backend",)

    def __init__(self, autoselect="ck"):
        super().__init__(
            "stub", list(KEYS), ["backend", "us", "errRatio", "status", "detail"]
        )
        self.autoselect = autoselect
        self.failing_proofs: set[str] = set()
        self.proven: list[tuple[str, str]] = []

    def _setup_specific_arguments(self):
        pass

    def get_gfx(self):
        return HARDWARE[0]

    def get_gpu_model(self):
        return HARDWARE[1]

    def get_cu_num(self):
        return int(HARDWARE[2])

    def candidate_from_selection(self, selection):
        backend = selection.get("backend")
        return None if backend == "unnamed" else {"backend": backend}

    def expected_selection(self, row):
        backend = row.get("backend")
        return None if backend is None else {"backend": str(backend)}

    def prove_selection(self, row, config_file, args):
        expected = self.expected_selection(row)
        if expected is None:
            if self.autoselect is None:
                return SelectionProof("failed", None, [], None, "probe crashed", 0.0)
            return SelectionProof(
                "unchecked", None, [{"backend": self.autoselect}], 1.0, "", 0.0
            )
        self.proven.append((str(row["m"]), expected["backend"]))
        if str(row["m"]) in self.failing_proofs:
            return SelectionProof(
                "wrong_selection", expected, [{"backend": "ck"}], 1.0, "stray", 0.0
            )
        return SelectionProof("verified", expected, [expected], 1.0, "", 0.0)

    def result_to_df(self, rets):
        rows = []
        for (key, backend), us, err, status, detail in rets:
            rows.append(
                {
                    **dict(zip(KEYS, key)),
                    "backend": backend,
                    "us": us,
                    "errRatio": err,
                    "status": status,
                    "detail": detail,
                }
            )
        return pd.DataFrame(rows, columns=self.columns)


def problem(m):
    return pd.Series(dict(zip(KEYS, (*HARDWARE, str(m)))))


def measured(m, **timings):
    """mp_tuner-shaped results; a negative time is a failed candidate."""
    key = (*HARDWARE, str(m))
    return [
        ((key, backend), us, 0.0 if us > 0 else 1.0, "ok" if us > 0 else "crash", "")
        for backend, us in timings.items()
    ]


class _Run(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tune_file = os.path.join(directory.name, "tuned_stub.csv")
        Path(self.tune_file).write_text(",".join(StubTuner.RUNTIME_FIELDS) + "\n")

    def args(self, **overrides):
        values = {
            "errRatio": 0.0,
            "profile_file": "",
            "min_improvement_pct": 3.0,
            "all": False,
            "tune_file": self.tune_file,
            "warmup": 1,
            "iters": 1,
            "timeout": 60,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def run_tuner(self, tuner, results_by_m, **overrides):
        args = self.args(**overrides)
        tuner.untunedf = pd.DataFrame([problem(m) for m in results_by_m])
        tuner.load_published(args)
        rets = []
        for m, timings in results_by_m.items():
            tuner.incumbents(problem(m))
            rets.extend(measured(m, **timings))
        winners = tuner.post_process(rets, args)
        tuner.result_to_csv(winners, args.tune_file)
        return tuner

    def table(self):
        return pd.read_csv(self.tune_file, dtype=str)

    def statuses(self, tuner):
        frame = pd.concat([tuner.success, tuner.failed])
        return dict(zip(frame["m"].astype(str), frame["status"]))


class TestAgainstAutoSelect(_Run):
    def test_a_challenger_past_the_bar_is_written_and_proven(self):
        tuner = self.run_tuner(StubTuner(), {128: {"ck": 10.0, "asm_v3": 9.0}})
        self.assertEqual(self.table()["backend"].tolist(), ["asm_v3"])
        self.assertEqual(tuner.proven, [("128", "asm_v3")])
        self.assertEqual(self.statuses(tuner), {"128": "promoted"})

    def test_a_challenger_under_the_bar_writes_nothing(self):
        tuner = self.run_tuner(StubTuner(), {128: {"ck": 10.0, "asm_v3": 9.8}})
        self.assertTrue(self.table().empty)
        self.assertEqual(self.statuses(tuner), {"128": "retain"})

    def test_auto_select_fastest_writes_nothing(self):
        tuner = self.run_tuner(StubTuner(), {128: {"ck": 8.0, "asm_v3": 9.0}})
        self.assertTrue(self.table().empty)
        self.assertEqual(tuner.proven, [])

    def test_an_unmeasured_incumbent_publishes_nothing(self):
        tuner = self.run_tuner(StubTuner(autoselect="unnamed"), {128: {"ck": 9.0}})
        self.assertTrue(self.table().empty)
        self.assertEqual(self.statuses(tuner), {"128": "unverified"})
        self.assertIn("outside the catalogue", tuner.failed.iloc[0]["detail"])

    def test_a_crashed_auto_select_probe_publishes_nothing(self):
        tuner = self.run_tuner(StubTuner(autoselect=None), {128: {"ck": 9.0}})
        self.assertTrue(self.table().empty)
        self.assertEqual(self.statuses(tuner), {"128": "unverified"})


class TestProofGatesTheWrite(_Run):
    def test_a_row_failing_its_proof_is_not_left_in_the_table(self):
        tuner = StubTuner()
        tuner.failing_proofs = {"128"}
        self.run_tuner(
            tuner,
            {128: {"ck": 10.0, "asm_v3": 9.0}, 256: {"ck": 10.0, "asm_v3": 9.0}},
        )
        self.assertEqual(self.table()["m"].tolist(), ["256"])
        self.assertEqual(
            self.statuses(tuner), {"128": "proof_failed", "256": "promoted"}
        )

    def test_a_failed_replacement_keeps_the_previous_row(self):
        pd.DataFrame([{**problem(128).to_dict(), "backend": "opus", "us": 9.5}]).to_csv(
            self.tune_file, index=False
        )
        tuner = StubTuner()
        tuner.failing_proofs = {"128"}
        self.run_tuner(tuner, {128: {"ck": 10.0, "opus": 9.5, "asm_v3": 8.0}}, all=True)
        self.assertEqual(self.table()["backend"].tolist(), ["opus"])


class TestPublishedRows(_Run):
    def publish(self, backend, us=9.0):
        frame = pd.DataFrame([{**problem(128).to_dict(), "backend": backend, "us": us}])
        frame["cu_num"] = frame["cu_num"].astype(int)
        frame["m"] = frame["m"].astype(int)
        frame.to_csv(self.tune_file, index=False)

    def test_retuning_replaces_a_row_written_with_numeric_cells(self):
        self.publish("opus", 9.5)
        self.run_tuner(
            StubTuner(), {128: {"ck": 10.0, "opus": 9.5, "asm_v3": 8.0}}, all=True
        )
        table = self.table()
        self.assertEqual(len(table), 1)
        self.assertEqual(table["backend"].tolist(), ["asm_v3"])

    def test_without_all_a_covered_shape_is_skipped(self):
        self.publish("opus")
        tuner = StubTuner()
        tuner.untunedf = pd.DataFrame([problem(128)])
        tuner.load_published(self.args())
        self.assertTrue(tuner.untunedf.empty)

    def test_a_published_row_is_defended_by_the_bar(self):
        self.publish("opus")
        tuner = self.run_tuner(
            StubTuner(), {128: {"ck": 10.0, "opus": 9.0, "asm_v3": 8.9}}, all=True
        )
        self.assertEqual(self.table()["backend"].tolist(), ["opus"])
        self.assertEqual(self.statuses(tuner), {"128": "retain"})

    def test_a_published_row_that_no_longer_runs_is_dropped(self):
        self.publish("opus")
        tuner = self.run_tuner(
            StubTuner(), {128: {"ck": 10.0, "opus": -1.0, "asm_v3": 9.9}}, all=True
        )
        self.assertTrue(self.table().empty)
        self.assertEqual(self.statuses(tuner), {"128": "drop"})

    def test_auto_select_beating_a_published_row_drops_it(self):
        self.publish("opus")
        self.run_tuner(StubTuner(), {128: {"ck": 8.0, "opus": 9.0}}, all=True)
        self.assertTrue(self.table().empty)

    def test_a_published_row_whose_candidate_never_ran_is_kept(self):
        # An earlier candidate's fault stops the group (not_run), or the group
        # times out: neither says the published candidate cannot serve.
        key = (*HARDWARE, "128")
        for status in ("not_run", "timeout"):
            with self.subTest(status=status):
                self.publish("opus")
                tuner = StubTuner()
                args = self.args(all=True)
                tuner.untunedf = pd.DataFrame([problem(128)])
                tuner.load_published(args)
                tuner.incumbents(problem(128))
                rets = [
                    ((key, "ck"), 10.0, 0.0, "ok", ""),
                    ((key, "asm_v3"), float("inf"), 1.0, "crash", "worker died"),
                    ((key, "opus"), float("inf"), 1.0, status, "not reached"),
                ]
                tuner.result_to_csv(tuner.post_process(rets, args), self.tune_file)
                self.assertEqual(self.table()["backend"].tolist(), ["opus"])
                self.assertEqual(self.statuses(tuner), {"128": "unverified"})


class TestTheTableFile(_Run):
    def test_columns_the_family_does_not_know_survive_a_rewrite(self):
        other = {**problem(64).to_dict(), "backend": "ck", "us": "7.0"}
        other.update(kernelName1="moe_a", block_m="32")
        pd.DataFrame([other]).to_csv(self.tune_file, index=False)
        self.run_tuner(StubTuner(), {128: {"ck": 10.0, "asm_v3": 9.0}})
        table = pd.read_csv(self.tune_file, dtype=str, keep_default_na=False)
        rows = {row["m"]: row for _, row in table.iterrows()}
        self.assertEqual(rows["64"]["kernelName1"], "moe_a")
        self.assertEqual(rows["64"]["block_m"], "32")
        self.assertEqual(rows["128"]["backend"], "asm_v3")
        self.assertEqual(rows["128"]["kernelName1"], "")

    def test_a_rewrite_keeps_the_file_mode_and_writes_through_a_symlink(self):
        os.chmod(self.tune_file, 0o644)
        link = os.path.join(os.path.dirname(self.tune_file), "linked.csv")
        os.symlink(self.tune_file, link)
        self.run_tuner(StubTuner(), {128: {"ck": 10.0, "asm_v3": 9.0}}, tune_file=link)
        self.assertTrue(os.path.islink(link))
        self.assertEqual(os.stat(self.tune_file).st_mode & 0o777, 0o644)
        self.assertEqual(self.table()["backend"].tolist(), ["asm_v3"])


class TestArguments(unittest.TestCase):
    def test_an_out_of_range_bar_is_refused_when_parsing(self):
        tuner = StubTuner()
        for value in ("-1", "100"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                tuner.parser.parse_args(["--min_improvement_pct", value])
        args = tuner.parser.parse_args(["--min_improvement_pct", "5"])
        self.assertEqual(args.min_improvement_pct, 5.0)


if __name__ == "__main__":
    unittest.main()
