#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Head-to-head: the elimination race against screening plus finalist rounds.

Two questions, and they need different measurements.

Cost is the easy one: run both paths over the same field and compare wall time
and calls.

Whether the race is *better* is not answered by which path reports the lower
number, because each path reports its own measurement of its own winner, taken
under its own conditions. The only fair comparison re-measures every selected
configuration together, in one session, under identical conditions, and asks
how fast the thing each path chose actually is.

Repeatability needs the same care. The race is deliberately indifferent inside
delta, so it may legitimately select different configurations across sessions
while every one of them is equally fast. Counting distinct winners would score
that as instability when nothing is wrong. So both are reported: how often each
path picked the same configuration, and -- the one that matters -- how much the
*speed of what it picked* moved between sessions.

Each tuning run is a fresh subprocess, which is what makes the sessions
independent, and the order of the two paths alternates so machine drift over
the course of the experiment cannot favour either.

    HIP_VISIBLE_DEVICES=1 PYTHONPATH=$PWD python -m \
        op_tests.tuners.compare_race_vs_screening --sample 200 --sessions 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import triton  # noqa: F401  # ROCm environments may require Triton before torch.
import torch

from aiter.utility.block_race import RaceEntrant, cuda_event_timer, measure_blocks

REPOSITORY_ROOT = Path(__file__).parents[2]


def run_tuner(shape_csv, strategy, sample, workdir, extra=()):
    """One tuning run in its own process, timed from the outside.

    Timed externally on purpose: a path's own report of how long it took
    excludes whatever it does before and after measuring, and process startup
    is part of what the screening path spends.
    """
    output = os.path.join(workdir, f"{strategy}.csv")
    evidence = os.path.join(workdir, f"{strategy}.evidence.json")
    command = [
        sys.executable,
        "-m",
        "op_tests.tuners.tune_mha_fwd",
        "-i",
        shape_csv,
        "-o",
        output,
        "--strategy",
        strategy,
        "--candidate-sample",
        str(sample),
        "--mp",
        "1",
        "--journal-file",
        os.path.join(workdir, f"{strategy}.journal.jsonl"),
        "--evidence-file",
        evidence,
        *extra,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started

    record = {
        "strategy": strategy,
        "wall_seconds": wall,
        "returncode": completed.returncode,
        "winner": None,
        "reported_us": None,
        "calls_spent": None,
        "blocks_run": None,
        "certified": None,
    }
    if completed.returncode != 0:
        record["stderr_tail"] = completed.stderr[-2000:]
        return record

    if os.path.isfile(output):
        frame = pd.read_csv(output)
        if not frame.empty:
            row = frame.iloc[0]
            record["winner"] = {
                "backend": str(row["backend"]),
                "num_splits": int(row["num_splits"]),
                "backend_config": (
                    str(row["backend_config"])
                    if pd.notna(row["backend_config"])
                    else ""
                ),
            }
    if os.path.isfile(evidence):
        with open(evidence) as handle:
            payload = json.load(handle)
        races = payload.get("races") or []
        if races:
            record["calls_spent"] = sum(r["calls_spent"] for r in races)
            record["blocks_run"] = races[0]["blocks_run"]
            record["certified"] = all(r["certified"] for r in races)
        record["promotions"] = payload.get("promotions")
    return record


def remeasure(shape_row, configs, block_calls, blocks, seed):
    """Measure every selected configuration together, under one set of
    conditions, so the numbers can actually be compared.

    Interleaved rather than one after another: whatever the machine is doing
    during the re-measurement moves all of them together inside a block, and
    cancels in the comparison.
    """
    import zlib

    from op_tests.tuners.tune_mha_fwd import _run_candidate, generate_data

    # The problem CSV carries the workload only; the hardware half of the
    # tuning key is filled in by the tuner. Everything needed to rebuild the
    # inputs is here, so do not go looking for gfx or cu_num.
    seed = zlib.crc32(
        ",".join(str(shape_row[field]) for field in sorted(shape_row.index)).encode()
    )
    data = generate_data(
        int(shape_row["batch"]),
        int(shape_row["total_q"]),
        int(shape_row["total_k"]),
        int(shape_row["max_seqlen_q"]),
        int(shape_row["max_seqlen_k"]),
        int(shape_row["nhead_q"]),
        int(shape_row["nhead_k"]),
        int(shape_row["hdim_q"]),
        int(shape_row["hdim_v"]),
        str(shape_row["dtype"]),
        seed,
    )
    tensors = (data["q"], data["k"], data["v"], data["cu_q"], data["cu_k"])
    tail = (
        int(shape_row["max_seqlen_q"]),
        int(shape_row["max_seqlen_k"]),
        int(shape_row["min_seqlen_q"]),
        float(shape_row["dropout_p"]),
        int(shape_row["hdim_q"]) ** -0.5,
        float(shape_row["logits_soft_cap"]),
        int(shape_row["how_v3_bf16_cvt"]),
        bool(int(shape_row["causal"])),
        int(shape_row["window_left"]),
        int(shape_row["window_right"]),
        bool(int(shape_row["return_lse"])),
    )

    def launch(entrant):
        config = entrant.payload["backend_config"]
        return _run_candidate(
            *tensors,
            entrant.payload["backend"],
            int(entrant.payload["num_splits"]),
            json.loads(config) if config else None,
            *tail,
        )

    entrants = [
        RaceEntrant(label=json.dumps(config, sort_keys=True), payload=config)
        for config in configs
    ]
    for entrant in entrants:
        for _ in range(5):
            launch(entrant)
    torch.cuda.synchronize()

    samples = measure_blocks(
        entrants, cuda_event_timer(launch), block_calls, blocks, seed=seed
    )
    return {entrant.label: samples[entrant.label].estimate for entrant in entrants}


def summarize(sessions, remeasured):
    """Turn the runs into the two claims a reviewer is being asked to accept."""
    summary = {}
    for strategy in ("race", "exhaustive"):
        runs = [r for session in sessions for r in session if r["strategy"] == strategy]
        ok = [r for r in runs if r["returncode"] == 0 and r["winner"]]
        picks = [json.dumps(r["winner"], sort_keys=True) for r in ok]
        realized = [
            remeasured[pick] for pick in picks if pick in remeasured
        ]
        summary[strategy] = {
            "runs": len(runs),
            "succeeded": len(ok),
            "wall_seconds_median": (
                statistics.median([r["wall_seconds"] for r in ok]) if ok else None
            ),
            "distinct_winners": len(set(picks)),
            "realized_us": realized,
            "realized_median_us": statistics.median(realized) if realized else None,
            "realized_spread": (
                (max(realized) - min(realized)) / min(realized)
                if len(realized) > 1
                else 0.0
            ),
        }
    if summary["race"]["wall_seconds_median"] and summary["exhaustive"][
        "wall_seconds_median"
    ]:
        summary["speedup"] = (
            summary["exhaustive"]["wall_seconds_median"]
            / summary["race"]["wall_seconds_median"]
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        default=str(REPOSITORY_ROOT / "op_tests/tuners/_full_kimi_mha.csv"),
        help="problem CSV; every row is compared separately",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="give both paths the identical field of this many candidates",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=3,
        help="independent repeats, so the reported spread is between sessions",
    )
    parser.add_argument("--block-calls", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20240917)
    parser.add_argument("--out", default="race_vs_screening.json")
    args = parser.parse_args()

    shapes = pd.read_csv(args.shapes)
    results = {"sample": args.sample, "sessions": args.sessions, "shapes": []}

    for shape_index, shape in shapes.iterrows():
        with tempfile.TemporaryDirectory() as shape_dir:
            one_shape = os.path.join(shape_dir, "shape.csv")
            shape.to_frame().T.to_csv(one_shape, index=False)

            sessions = []
            for session in range(args.sessions):
                # Alternate which path runs first: if the machine drifts over
                # the experiment, a fixed order would hand the advantage to
                # whichever path always ran on the quieter side of it.
                order = (
                    ("race", "exhaustive")
                    if session % 2 == 0
                    else ("exhaustive", "race")
                )
                runs = []
                for strategy in order:
                    workdir = os.path.join(shape_dir, f"s{session}-{strategy}")
                    os.makedirs(workdir, exist_ok=True)
                    print(
                        f"shape {shape_index} session {session}: {strategy}",
                        flush=True,
                    )
                    run = run_tuner(one_shape, strategy, args.sample, workdir)
                    print(
                        f"  {run['wall_seconds']:.0f} s, rc={run['returncode']}, "
                        f"winner={(run['winner'] or {}).get('backend')}",
                        flush=True,
                    )
                    runs.append(run)
                sessions.append(runs)

            chosen = []
            seen = set()
            for run in (r for session in sessions for r in session):
                if not run["winner"]:
                    continue
                token = json.dumps(run["winner"], sort_keys=True)
                if token not in seen:
                    seen.add(token)
                    chosen.append(run["winner"])

            remeasured = {}
            failure = None
            if chosen:
                print(
                    f"  re-measuring {len(chosen)} selected configs together",
                    flush=True,
                )
                try:
                    remeasured = remeasure(
                        shape, chosen, args.block_calls, args.blocks, args.seed
                    )
                except Exception as error:  # noqa: BLE001
                    # Hours of tuning runs are already in hand; losing them
                    # because the comparison step tripped would be the worse
                    # outcome by far.
                    failure = f"{type(error).__name__}: {error}"
                    print(f"  re-measurement failed: {failure}", flush=True)

            results["shapes"].append(
                {
                    "shape_index": int(shape_index),
                    "shape": {k: str(v) for k, v in shape.items()},
                    "sessions": sessions,
                    "remeasured_us": remeasured,
                    "remeasure_error": failure,
                    "summary": summarize(sessions, remeasured),
                }
            )
            # Written after every shape, not once at the end.
            with open(args.out, "w") as handle:
                json.dump(results, handle, indent=2)

    print(f"\nwrote {args.out}")

    for entry in results["shapes"]:
        summary = entry["summary"]
        print(f"\nshape {entry['shape_index']}")
        for strategy in ("race", "exhaustive"):
            item = summary[strategy]
            median = item["realized_median_us"]
            wall = item["wall_seconds_median"]
            print(
                f"  {strategy:11s} "
                f"{'n/a' if wall is None else f'{wall:7.0f}'} s median wall, "
                f"{item['distinct_winners']} distinct winner(s), "
                f"realized {'n/a' if median is None else f'{median:.1f}'} us, "
                f"spread {item['realized_spread']:.2%}"
            )
        if "speedup" in summary:
            print(f"  race is {summary['speedup']:.1f}x cheaper to run")


if __name__ == "__main__":
    main()
