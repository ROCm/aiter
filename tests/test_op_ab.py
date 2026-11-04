"""`aiter/utility/op_ab.py` end to end on fake aiter checkouts: each side's
tuner runs in a subprocess importing that side's `aiter`, the sides' rounds
are reduced to their fastest, base's spread is the noise floor, and the
weighted totals use each shape's call count."""

import csv
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "aiter/utility/op_ab.py"
SPEC = importlib.util.spec_from_file_location("op_ab", SCRIPT)
op_ab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(op_ab)

_BASE_TUNER = """
class TunerCommon:
    keys = ["gfx", "cu_num", "M", "N", "K"]

    def _print_benchmark_results(self, label, results, report_file=None, shapes_df=None):
        print(label, results)
"""

# Times a shape at M + N/1000 us; FAKE_SLOW_M makes that M 30% slower.
_TUNER = """
import argparse, csv, os
import pandas as pd
from aiter.utility.base_tuner import TunerCommon

class Tuner(TunerCommon):
    def run(self, args):
        assert args.run_config and args.o and not os.path.exists(args.o)
        df = pd.read_csv(args.i)
        df["gfx"], df["cu_num"] = "gfx950", 256
        self.untunedf = df[self.keys]
        slow = int(os.environ.get("FAKE_SLOW_M", "-1"))
        results = []
        for _, row in df.iterrows():
            us = row["M"] + row["N"] / 1000
            if row["M"] == slow:
                us *= 1.3
            results.append({"shape": str(tuple(row)), "e2e_us": us, "status": "ok"})
        self._print_benchmark_results("Benchmark (default)", results)

p = argparse.ArgumentParser()
p.add_argument("-i"); p.add_argument("-o"); p.add_argument("--run_config", action="store_true")
Tuner().run(p.parse_args())
"""


def _fake_aiter(root: Path) -> Path:
    (root / "aiter/utility").mkdir(parents=True)
    (root / "aiter/__init__.py").write_text("")
    (root / "aiter/utility/__init__.py").write_text("")
    (root / "aiter/utility/base_tuner.py").write_text(textwrap.dedent(_BASE_TUNER))
    (root / "csrc").mkdir()
    (root / "csrc/fake_tune.py").write_text(textwrap.dedent(_TUNER))
    return root


def _shapes(path: Path) -> Path:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["M", "N", "K", "count"])
        w.writerows([[1, 2048, 512, 900], [64, 2048, 512, 90], [3000, 2048, 512, 2]])
    return path


def test_regression_on_one_shape_is_flagged_and_weighted(tmp_path, monkeypatch):
    base = _fake_aiter(tmp_path / "base")
    new = _fake_aiter(tmp_path / "new")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"steps": 100, "step_time_us_mean": 20000.0}))
    monkeypatch.setattr(sys, "argv", ["op_ab.py"])
    out = tmp_path / "ab"
    rc = op_ab.main(
        [
            "--csv",
            str(_shapes(tmp_path / "shapes.csv")),
            "--tuner",
            "csrc/fake_tune.py",
            "--summary",
            str(summary),
            "--base-aiter",
            str(base),
            "--new-aiter",
            str(new),
            "--new-env",
            "FAKE_SLOW_M=64",
            "--repeat",
            "3",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    verdicts = {r["keys"]["M"]: r["verdict"] for r in report["rows"]}
    assert verdicts == {"1": "same", "64": "regressed", "3000": "same"}
    assert [r["side"] for r in report["runs"]] == [
        "base",
        "new",
        "new",
        "base",
        "base",
        "new",
    ]
    assert all(
        r["package_file"].startswith(str(tmp_path / r["side"])) for r in report["runs"]
    )
    t = report["totals"]
    slow_row = 90 * (64 + 2.048)
    assert abs(t["weighted_delta_us"] - 0.3 * slow_row) < 1e-6
    assert abs(t["delta_us_per_step"] - 0.3 * slow_row / 100) < 1e-6
    assert "regressed" in (out / "report.md").read_text()


def _rounds(base, new):
    def side(times):
        return [{("8",): {"us": t, "status": "ok"}} for t in times]

    return {"base": side(base), "new": side(new)}


def test_only_a_complete_separation_of_rounds_is_a_change():
    shapes = [{"keys": {"M": "8"}, "count": 10}]
    # 5% slower at the best round, but one new round beats a base round.
    (row,) = op_ab.compare(
        ["M"], shapes, _rounds([10, 10.2, 10.9], [10.5, 10.6, 10.8]), 0.01
    )
    assert row["verdict"] == "same"
    (row,) = op_ab.compare(
        ["M"], shapes, _rounds([10, 10.2, 10.3], [10.5, 10.6, 10.8]), 0.01
    )
    assert row["verdict"] == "regressed"
    assert row["false_alarm_p"] == 1 / 20
    # Separated but inside the floor.
    (row,) = op_ab.compare(
        ["M"], shapes, _rounds([10, 10.01, 10.02], [10.05, 10.06, 10.07]), 0.01
    )
    assert row["verdict"] == "same"
    (row,) = op_ab.compare(
        ["M"], shapes, _rounds([10.5, 10.6, 10.8], [10, 10.2, 10.3]), 0.01
    )
    assert row["verdict"] == "improved"


def test_failed_side_is_an_error_not_a_time():
    rounds = {
        "base": [{("8",): {"us": 10.0, "status": "ok"}}] * 2,
        "new": [{("8",): {"us": None, "status": "mismatch:err_ratio=0.5"}}] * 2,
    }
    (row,) = op_ab.compare(["M"], [{"keys": {"M": "8"}, "count": 3}], rounds, 0.01)
    assert row["verdict"] == "error"
    assert row["new"]["failed"] == ["mismatch:err_ratio=0.5"]
    assert op_ab.totals([row], None)["errors"] == 1


def test_key_spellings_match_across_csv_and_pandas():
    assert op_ab._norm("64") == op_ab._norm(64.0) == "64"
    assert op_ab._norm("False") == op_ab._norm("0") == "0"
    assert op_ab._norm("torch.bfloat16") == "torch.bfloat16"


_REPLAY = """
import csv, os

def replay(rows_csv, extra_args):
    with open(rows_csv) as f:
        rows = list(csv.DictReader(f))
    slow = os.environ.get("FAKE_SLOW_LABEL")
    results = []
    for row in rows:
        us = float(row["ctx"]) / 100
        if row["label"] == slow:
            us *= 1.5
        results.append({"shape": row["label"], "e2e_us": us, "status": "ok"})
    return rows, results
"""


def test_replay_driver_times_rows_through_the_harness(tmp_path, monkeypatch):
    """Driver ``replay``: one harness, given by path, times both sides; the
    report shows a row's label rather than its recorded call."""
    base = _fake_aiter(tmp_path / "base")
    new = _fake_aiter(tmp_path / "new")
    script = tmp_path / "harness" / "fake_replay.py"
    script.parent.mkdir()
    script.write_text(textwrap.dedent(_REPLAY))
    rows = tmp_path / "attn.csv"
    with open(rows, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "op", "sig", "ctx", "count"])
        w.writerow(["short", "pa", '{"args":[1]}', "1000", 50])
        w.writerow(["long", "pa", '{"args":[1]}', "9000", 5])
    monkeypatch.setattr(sys, "argv", ["op_ab.py"])
    out = tmp_path / "ab"
    op_ab.main(
        [
            "--csv",
            str(rows),
            "--replay",
            str(script),
            "--base-aiter",
            str(base),
            "--new-aiter",
            str(new),
            "--new-env",
            "FAKE_SLOW_LABEL=long",
            "--repeat",
            "3",
            "--out",
            str(out),
        ]
    )
    report = json.loads((out / "report.json").read_text())
    assert report["driver"] == "replay"
    assert {r["keys"]["label"]: r["verdict"] for r in report["rows"]} == {
        "short": "same",
        "long": "regressed",
    }
    assert abs(report["totals"]["weighted_delta_us"] - 5 * 90 * 0.5) < 1e-6
    md = (out / "report.md").read_text()
    assert "| label | count |" in md and '{"args"' not in md
