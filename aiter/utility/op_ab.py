#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time a tuner family's production op on a set of weighted shapes under two
aiter checkouts, base and new, on one GPU, and report which shapes regressed
and what the change is worth per weighted call.

The shapes are a tuner's key columns plus ``count``: how many times a real
run called the op with that key (ATOM's ``tools/op_record/export.py`` writes
them). Each side runs the tuner's ``--run_config`` -- the production op, with
the side's own tuned configs and dispatch, so a shape that falls back to a
default config is timed as it would run -- in a subprocess whose ``aiter`` is
that side's checkout. The sides alternate, ``--repeat`` rounds each, on the
same GPU. A shape's time is its fastest round: contention only adds time.
It regressed when every new round is slower than every base round and its
time grew by more than ``--floor``; improved the other way round. If the
rounds of the two sides are exchangeable -- the same code -- a complete
separation happens with probability 1/C(2R, R) per shape and direction
(1/252 at the default 5 rounds), which is the test's false-alarm rate; a
threshold on the rounds' spread is not used, since three rounds' spread
under-reads the noise (it flagged unchanged shapes at 1.6 % and 2.1 %).

    python aiter/utility/op_ab.py --summary EXPORT/summary.json \\
        --family a8w8_blockscale_bpreshuffle \\
        --base-aiter /path/aiter-base --new-aiter /path/aiter-new \\
        --gpu 0 --repeat 3 --out AB_DIR

writes ``report.json`` and ``report.md`` to ``--out``.
"""

import argparse
import csv
import json
import math
import os
import runpy
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

SIDES = ("base", "new")
# The checkout this file is in: a replay script's path is relative to it.
_AITER_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


@dataclass
class Side:
    name: str
    root: str
    env: dict[str, str] = field(default_factory=dict)


def _parse_env(pairs: list[str] | None) -> dict[str, str]:
    env = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--*-env takes KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def read_shapes(path: str) -> tuple[list[str], list[dict]]:
    """(key columns, rows with their ``count``)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = [c for c in reader.fieldnames or [] if c != "count"]
        rows = []
        for row in reader:
            rows.append(
                {
                    "keys": {c: row[c] for c in columns},
                    "count": int(float(row.get("count") or 1)),
                }
            )
    return columns, rows


def _key(values: dict, columns: list[str]) -> tuple:
    return tuple(_norm(values.get(c)) for c in columns)


def _norm(value) -> str:
    """One spelling per value across CSV text and pandas' parsed values."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return {"true": "1", "false": "0"}.get(text.lower(), text)
    return str(int(number)) if number.is_integer() else repr(number)


# ---------------------------------------------------------------- worker side


def worker(
    driver: str,
    package: str,
    tuner: str,
    keys_csv: str,
    result: str,
    tuner_args: list[str],
) -> int:
    """Time ``keys_csv`` in this process, whose ``package`` (aiter, or ATOM for
    ATOM's own kernels) is the side's checkout, and dump what was timed to
    ``result``: through ``tuner --run_config`` (driver ``tuner``), or a replay
    script's ``replay(csv, args)`` (driver ``replay``:
    `aiter/utility/op_replay.py`, ATOM's ``tools/op_record/replay.py``)."""
    import importlib

    package_file = importlib.import_module(package).__file__

    captured = []
    code, error = 0, None
    if driver == "replay":
        try:
            ns = runpy.run_path(tuner, run_name="op_ab_replay")
            rows, results = ns["replay"](keys_csv, tuner_args)
            captured.append(
                {
                    "label": "replay",
                    "rows": [{k: str(v) for k, v in r.items()} for r in rows],
                    "results": results,
                }
            )
        except Exception:  # noqa: BLE001  reported to the driver, which stops
            code, error = 1, traceback.format_exc()
        # Under torchrun every rank replays; the ranks agree on the times.
        if int(os.environ.get("RANK", "0")) == 0:
            _dump(result, package_file, code, error, captured)
        return 0

    from aiter.utility import base_tuner

    original = base_tuner.TunerCommon._print_benchmark_results

    def capture(self, label, results, report_file=None, shapes_df=None):
        frame = shapes_df if shapes_df is not None else self.untunedf
        rows = []
        if frame is not None:
            columns = [c for c in self.keys if c in frame.columns]
            rows = frame[columns].astype(str).to_dict("records")
        captured.append(
            {
                "label": label,
                "rows": rows,
                "results": [
                    {k: v for k, v in r.items() if isinstance(v, (int, float, str))}
                    for r in results
                ],
            }
        )
        return original(
            self, label, results, report_file=report_file, shapes_df=shapes_df
        )

    base_tuner.TunerCommon._print_benchmark_results = capture
    # An absent tuned file: a tuner skips the shapes its tuned file already
    # has, and the tuned shapes are the ones a production run calls most.
    empty_tuned = os.path.join(os.path.dirname(result), "no_tuned_file.csv")
    sys.argv = [
        tuner,
        "-i",
        keys_csv,
        "-o",
        empty_tuned,
        "--run_config",
        *tuner_args,
    ]
    # As `python tuner.py` would: the tuner imports its own directory's modules.
    sys.path.insert(0, os.path.dirname(os.path.abspath(tuner)))
    try:
        runpy.run_path(tuner, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    except Exception:  # noqa: BLE001  reported to the driver, which stops
        code, error = 1, traceback.format_exc()
    _dump(result, package_file, code, error, captured)
    return 0


def _dump(result: str, package_file: str, code: int, error, captured: list) -> None:
    with open(result, "w") as f:
        json.dump(
            {
                "package_file": package_file,
                "exit_code": code,
                "error": error,
                "captured": captured,
            },
            f,
        )


# ---------------------------------------------------------------- driver side


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def run_side(
    side: Side,
    driver: str,
    package: str,
    tuner: str,
    tuner_args: list[str],
    keys_csv: str,
    gpu: str,
    out_dir: str,
    round_: int,
    ranks: int = 1,
) -> dict:
    result = os.path.join(out_dir, f"{side.name}_r{round_}.json")
    log = os.path.join(out_dir, f"{side.name}_r{round_}.log")
    env = os.environ.copy()
    env.update(side.env)
    env["PYTHONPATH"] = os.pathsep.join(
        [side.root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["HIP_VISIBLE_DEVICES"] = gpu
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A tuner is the side's own; a replay script is the harness, the same for
    # both sides, and times whichever package the side imports.
    tuner_path = os.path.join(side.root, tuner) if driver == "tuner" else tuner
    launcher = [sys.executable]
    if ranks > 1:  # a collective: one process per rank, on the GPUs given
        launcher += [
            "-m",
            "torch.distributed.run",
            "--nproc-per-node",
            str(ranks),
            "--master-port",
            str(_free_port()),
        ]
    cmd = [
        *launcher,
        os.path.abspath(__file__),
        "--_worker",
        "--driver",
        driver,
        "--package",
        package,
        "--tuner",
        tuner_path,
        "--keys-csv",
        keys_csv,
        "--result",
        result,
        "--",
        *tuner_args,
    ]
    started = time.time()
    with open(log, "w") as f:
        proc = subprocess.run(
            cmd,
            cwd=side.root,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not os.path.exists(result):
        raise SystemExit(
            f"{side.name} round {round_}: worker exited {proc.returncode} "
            f"without a result; see {log}"
        )
    with open(result) as f:
        data = json.load(f)
    data["seconds"] = time.time() - started
    data["log"] = log
    if data["error"] or not data["captured"]:
        raise SystemExit(
            f"{side.name} round {round_}: the {driver} timed nothing "
            f"(exit {data['exit_code']}); see {log}\n{data['error'] or ''}"
        )
    expected = os.path.realpath(side.root)
    if not os.path.realpath(data["package_file"]).startswith(expected + os.sep):
        raise SystemExit(
            f"{side.name} round {round_} imported {package} from "
            f"{data['package_file']}, not {side.root}"
        )
    return data


def timings(data: dict, columns: list[str]) -> dict[tuple, dict]:
    """key -> {"us": float or None, "status": str} from one round."""
    out = {}
    for block in data["captured"]:
        rows, results = block["rows"], block["results"]
        if len(rows) != len(results):
            raise SystemExit(
                f"{block['label']}: {len(rows)} shapes but {len(results)} results"
            )
        for row, result in zip(rows, results):
            us = result.get("e2e_us", result.get("us", -1))
            status = str(result.get("status", "ok"))
            ok = status == "ok" and us is not None and us > 0
            out[_key(row, columns)] = {
                "us": float(us) if ok else None,
                "status": status,
            }
    return out


def compare(
    columns: list[str],
    shapes: list[dict],
    rounds: dict[str, list[dict[tuple, dict]]],
    floor: float,
) -> list[dict]:
    rows = []
    for shape in shapes:
        key = _key(shape["keys"], columns)
        row = {"keys": shape["keys"], "count": shape["count"]}
        for name in SIDES:
            seen = [r.get(key) for r in rounds[name]]
            times = [s["us"] for s in seen if s is not None and s["us"] is not None]
            bad = sorted({s["status"] for s in seen if s and s["us"] is None})
            missing = sum(1 for s in seen if s is None)
            row[name] = {
                "us": min(times) if times else None,
                "rounds_us": times,
                "failed": bad,
                "missing_rounds": missing,
            }
        base, new = row["base"], row["new"]
        for side in (base, new):
            times = side["rounds_us"]
            side["spread"] = (max(times) - min(times)) / min(times) if times else None
        row["noise"] = max(
            (s["spread"] for s in (base, new) if s["spread"] is not None), default=None
        )
        if base["us"] is None or new["us"] is None:
            row["verdict"] = "error"
            row["ratio"] = None
            row["false_alarm_p"] = None
        else:
            b, n = base["rounds_us"], new["rounds_us"]
            ratio = new["us"] / base["us"]
            row["ratio"] = ratio
            row["false_alarm_p"] = 1 / math.comb(len(b) + len(n), len(b))
            if min(n) > max(b) and ratio - 1 > floor:
                row["verdict"] = "regressed"
            elif max(n) < min(b) and 1 - ratio > floor:
                row["verdict"] = "improved"
            else:
                row["verdict"] = "same"
        rows.append(row)
    return rows


def totals(rows: list[dict], summary: dict | None) -> dict:
    timed = [r for r in rows if r["verdict"] != "error"]
    base = sum(r["count"] * r["base"]["us"] for r in timed)
    new = sum(r["count"] * r["new"]["us"] for r in timed)
    out = {
        "shapes": len(rows),
        "shapes_timed": len(timed),
        "calls": sum(r["count"] for r in rows),
        "calls_timed": sum(r["count"] for r in timed),
        "weighted_base_us": base,
        "weighted_new_us": new,
        "weighted_delta_us": new - base,
        "weighted_ratio": new / base if base else None,
        "regressed": sum(1 for r in rows if r["verdict"] == "regressed"),
        "improved": sum(1 for r in rows if r["verdict"] == "improved"),
        "errors": len(rows) - len(timed),
    }
    if summary and summary.get("steps"):
        steps = summary["steps"]
        out["window_steps"] = steps
        out["delta_us_per_step"] = (new - base) / steps
        out["base_us_per_step"] = base / steps
        step_us = summary.get("step_time_us_mean")
        if step_us:
            out["step_time_us_mean"] = step_us
            out["delta_pct_of_step"] = 100 * (new - base) / steps / step_us
    return out


def render_md(report: dict) -> str:
    t = report["totals"]
    # A replay row's key is its recorded call; its label is what a reader needs.
    columns = ["label"] if "label" in report["columns"] else report["columns"]
    lines = [
        f"# op A/B: {report['family'] or report['tuner']}",
        "",
        (
            f"- base: `{report['sides']['base']['root']}` "
            f"{report['sides']['base']['env'] or ''}"
        ),
        (
            f"- new: `{report['sides']['new']['root']}` "
            f"{report['sides']['new']['env'] or ''}"
        ),
        (
            f"- GPU {report['gpu']}, {report['repeat']} rounds per side, "
            f"alternating; shape time = fastest round; regressed/improved = every "
            f"round of one side beyond every round of the other (false alarm "
            f"1/{math.comb(2 * report['repeat'], report['repeat'])} per shape and "
            f"direction) and beyond the {report['floor']:.1%} floor; noise = the "
            f"larger side's spread, for reading only"
        ),
        (
            f"- shapes timed {t['shapes_timed']}/{t['shapes']}, "
            f"calls timed {t['calls_timed']}/{t['calls']}"
        ),
        (
            f"- weighted (count x us): base {t['weighted_base_us']:.1f}, "
            f"new {t['weighted_new_us']:.1f}, delta {t['weighted_delta_us']:+.1f} us "
            f"(ratio {t['weighted_ratio'] or float('nan'):.4f})"
        ),
    ]
    if "delta_us_per_step" in t:
        lines.append(
            f"- per step over {t['window_steps']} steps: "
            f"{t['delta_us_per_step']:+.2f} us"
            + (
                f" = {t['delta_pct_of_step']:+.3f}% of the mean step "
                f"({t['step_time_us_mean']:.0f} us)"
                if "delta_pct_of_step" in t
                else ""
            )
        )
    lines += [
        f"- regressed {t['regressed']}, improved {t['improved']}, errors {t['errors']}",
        "",
        (
            "Isolated timings: interactions inside a forward (stream overlap, "
            "cache, communication) are not measured."
        ),
        "",
        "| "
        + " | ".join(columns)
        + " | count | base us | new us | ratio | noise | verdict |",
        "|" + "---|" * (len(columns) + 6),
    ]
    order = {"regressed": 0, "error": 1, "improved": 2, "same": 3}

    def us(side: dict) -> str:
        if side["us"] is not None:
            return f"{side['us']:.2f}"
        return "/".join(side["failed"]) or "-"

    for r in sorted(report["rows"], key=lambda r: (order[r["verdict"]], -r["count"])):
        lines.append(
            "| "
            + " | ".join(str(r["keys"][c]) for c in columns)
            + f" | {r['count']} | {us(r['base'])} | {us(r['new'])} | "
            + (f"{r['ratio']:.3f}" if r["ratio"] is not None else "-")
            + " | "
            + (f"{r['noise']:.1%}" if r["noise"] is not None else "-")
            + f" | {r['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--_worker":
        p = argparse.ArgumentParser()
        p.add_argument("--_worker", action="store_true")
        p.add_argument("--driver", choices=["tuner", "replay"], default="tuner")
        p.add_argument("--package", default="aiter")
        p.add_argument("--tuner", required=True)
        p.add_argument("--keys-csv", required=True)
        p.add_argument("--result", required=True)
        p.add_argument("tuner_args", nargs="*")
        a = p.parse_args(argv)
        return worker(a.driver, a.package, a.tuner, a.keys_csv, a.result, a.tuner_args)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--summary", help="export summary.json; with --family, picks csv/tuner"
    )
    p.add_argument("--family", help="family name in the summary")
    p.add_argument("--csv", help="tuner key columns + count")
    p.add_argument("--tuner", help="tuner script, relative to an aiter checkout")
    p.add_argument(
        "--replay",
        help="replay script (driver replay), e.g. aiter/utility/op_replay.py; "
        "the same file times both sides",
    )
    p.add_argument("--tuner-arg", action="append", default=[], help="repeatable")
    p.add_argument(
        "--base",
        "--base-aiter",
        dest="base",
        required=True,
        help="base checkout: aiter's, or ATOM's for a family ATOM owns",
    )
    p.add_argument("--new", "--new-aiter", dest="new", required=True)
    p.add_argument(
        "--package",
        choices=["aiter", "atom"],
        help="the package each side's checkout provides (default: the "
        "family's owner, else aiter)",
    )
    p.add_argument("--base-env", action="append", help="KEY=VALUE, repeatable")
    p.add_argument("--new-env", action="append", help="KEY=VALUE, repeatable")
    p.add_argument("--gpu", default="0", help="HIP_VISIBLE_DEVICES for both sides")
    p.add_argument(
        "--gpus",
        help="comma-separated GPUs, one per rank, for a family that needs the "
        "recorded TP group (a collective's ``ranks``)",
    )
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--iters", type=int, help="tuner --iters")
    p.add_argument("--warmup", type=int, help="tuner --warmup")
    p.add_argument("--floor", type=float, default=0.01)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    summary = None
    if a.summary:
        with open(a.summary) as f:
            summary = json.load(f)
    csv_path, tuner, tuner_args = a.csv, a.tuner, list(a.tuner_arg)
    driver = "replay" if a.replay else "tuner"
    package = a.package or "aiter"
    if a.replay:
        tuner = os.path.abspath(a.replay)
    if a.family:
        if summary is None:
            raise SystemExit("--family needs --summary")
        fam = summary["families"].get(a.family)
        if fam is None:
            raise SystemExit(
                f"no family {a.family!r} in the summary: {sorted(summary['families'])}"
            )
        csv_path = csv_path or os.path.join(os.path.dirname(a.summary), fam["csv"])
        driver = fam.get("driver", "tuner")
        package = a.package or fam.get("owner", "aiter")
        if driver == "replay":
            # aiter's harness is this checkout's; ATOM's is the new side's.
            harness_root = _AITER_ROOT if package == "aiter" else a.new
            tuner = tuner or os.path.join(os.path.abspath(harness_root), fam["replay"])
        else:
            tuner = tuner or fam["tuner"]
        tuner_args = fam.get("tuner_args", []) + tuner_args
        ranks = int(fam.get("ranks", 1))
    else:
        ranks = len(a.gpus.split(",")) if a.gpus else 1
    gpu = a.gpu
    if ranks > 1:
        if not a.gpus or len(a.gpus.split(",")) != ranks:
            raise SystemExit(
                f"this family runs on {ranks} ranks: give --gpus with {ranks}"
            )
        gpu = a.gpus
    if not csv_path or not tuner:
        raise SystemExit("give --csv and --tuner, or --summary and --family")
    if a.repeat < 3:
        raise SystemExit(
            "--repeat must be at least 3: with fewer rounds a complete separation "
            "of the sides happens by chance too often to call"
        )
    if a.iters:
        tuner_args += ["--iters", str(a.iters)]
    if a.warmup:
        tuner_args += ["--warmup", str(a.warmup)]

    # The workers run in each side's checkout: every path they get is absolute.
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    columns, shapes = read_shapes(csv_path)
    keys_csv = os.path.join(a.out, "shapes.csv")
    with open(keys_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for s in shapes:
            writer.writerow([s["keys"][c] for c in columns])

    sides = {
        "base": Side("base", os.path.abspath(a.base), _parse_env(a.base_env)),
        "new": Side("new", os.path.abspath(a.new), _parse_env(a.new_env)),
    }
    rounds: dict[str, list[dict]] = {name: [] for name in SIDES}
    runs = []
    for i in range(a.repeat):
        order = SIDES if i % 2 == 0 else SIDES[::-1]
        for name in order:
            data = run_side(
                sides[name],
                driver,
                package,
                tuner,
                tuner_args,
                keys_csv,
                gpu,
                a.out,
                i,
                ranks,
            )
            rounds[name].append(timings(data, columns))
            runs.append(
                {
                    "side": name,
                    "round": i,
                    "seconds": data["seconds"],
                    "exit_code": data["exit_code"],
                    "package_file": data["package_file"],
                    "log": data["log"],
                }
            )
            print(f"{name} round {i}: {data['seconds']:.0f}s", flush=True)

    rows = compare(columns, shapes, rounds, a.floor)
    report = {
        "family": a.family,
        "driver": driver,
        "package": package,
        "tuner": tuner,
        "tuner_args": tuner_args,
        "csv": os.path.abspath(csv_path),
        "columns": columns,
        "gpu": gpu,
        "ranks": ranks,
        "repeat": a.repeat,
        "floor": a.floor,
        "sides": {n: {"root": s.root, "env": s.env} for n, s in sides.items()},
        "runs": runs,
        "totals": totals(rows, summary),
        "rows": rows,
    }
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    md = render_md(report)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(md)
    print("\n".join(md.splitlines()[:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
