# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regenerate the current-main GPU 1 comparison from checked raw results."""

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MATCH_METADATA = (
    "gpu",
    "torch",
    "flydsl",
    "hip",
    "seed",
    "timing",
    "workgroup_budget",
    "environment",
    "harness_sha256",
)
MATCH_INPUT = (
    "table_width",
    "kv_pool_pages",
    "max_context_length",
    "query_shape",
    "checked",
    "logical_io_bytes",
    "query_heads",
    "kv_heads",
    "head_dim",
    "batch",
    "query_length",
    "page",
    "context_min",
    "context_max",
)


def key(row):
    return row["case"], row["profile"], row["kv_dtype"]


def read_run(name, complete=True):
    run = json.loads((ROOT / f"{name}.json").read_text())
    meta, rows = run["metadata"], run["results"]
    assert not meta["dirty"], name
    if complete:
        assert meta["status"] == "complete", name
        assert len(rows) == meta["expected_results"], name
    else:
        assert meta["status"] == "running", name
        assert len(rows) < meta["expected_results"], name
    assert len({key(r) for r in rows}) == len(rows), name
    assert all(r["checked"] and r["graph_changed_query_checked"] for r in rows), name
    assert all(r["median_us"] > 0 for r in rows), name
    return run


def aggregate(previous, current, reference):
    runs = previous + current
    for run in runs:
        for field in MATCH_METADATA:
            assert run["metadata"][field] == reference[field], field
    indices = [{key(r): r for r in run["results"]} for run in runs]
    assert all(set(i) == set(indices[0]) for i in indices)
    rows = []
    for r in current[0]["results"]:
        k = key(r)
        old = [i[k] for i in indices[: len(previous)]]
        new = [i[k] for i in indices[len(previous) :]]
        for other in old + new:
            for field in MATCH_INPUT:
                assert r[field] == other[field], (k, field)
        old_us = [x["median_us"] for x in old]
        new_us = [x["median_us"] for x in new]
        a, b = map(statistics.median, (old_us, new_us))
        rows.append(
            {
                "case": r["case"],
                "suite": r["suite"],
                "profile": r["profile"],
                "kv_dtype": r["kv_dtype"],
                "previous_us": a,
                "main_us": b,
                "latency_change_pct": 100 * (b / a - 1),
                "speedup": a / b,
                "logical_io_bytes": r["logical_io_bytes"],
                "previous_logical_io_throughput_tb_s": r["logical_io_bytes"] / a / 1e6,
                "main_logical_io_throughput_tb_s": r["logical_io_bytes"] / b / 1e6,
                "previous_run_medians_us": old_us,
                "main_run_medians_us": new_us,
                "previous_run_spread_pct": 100 * (max(old_us) - min(old_us)) / a,
                "main_run_spread_pct": 100 * (max(new_us) - min(new_us)) / b,
                "previous_output_modes": sorted({x["output_mode"] for x in old}),
                "main_output_modes": sorted({x["output_mode"] for x in new}),
                "previous_plan_caps": sorted({x["plan_cap"] for x in old}),
                "main_plan_caps": sorted({x["plan_cap"] for x in new}),
                "previous_plan_capacities": sorted({x["plan_capacity"] for x in old}),
                "main_plan_capacities": sorted({x["plan_capacity"] for x in new}),
            }
        )
    return rows


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def main():
    names = {
        label: [f"{label}-{i}" for i in range(1, 4)] for label in ("previous", "main")
    }
    previous, current = (
        [read_run(n) for n in names[label]] for label in ("previous", "main")
    )
    reference = previous[0]["metadata"]
    assert all(len(r["results"]) == 84 for r in previous + current)
    info = json.loads((ROOT / "run-info.json").read_text())
    assert info["physical_gpu"] == 1
    assert reference["environment"]["HIP_VISIBLE_DEVICES"] == "1"
    assert info["started_sources"] == info["ended_sources"]
    assert info["flydsl_before"] == info["flydsl_after"]
    assert info["harness_sha256"] == reference["harness_sha256"]
    for label, runs in (("previous", previous), ("main", current)):
        commit = info["started_sources"][label]["commit"]
        assert all(r["metadata"]["commit"] == commit for r in runs)
    rows = aggregate(previous, current, reference)
    recheck_names = sorted(
        p.stem
        for label in ("previous", "main")
        for p in ROOT.glob(f"recheck-{label}-[123].json")
    )
    recheck_commands = json.loads((ROOT / "recheck-commands.json").read_text())
    exits = {c["label"]: c["exit_code"] for c in recheck_commands}
    assert set(exits) == set(recheck_names)
    rechecks = {n: read_run(n, complete=exits[n] == 0) for n in recheck_names}
    targeted = []
    if rechecks:
        old = [
            rechecks[n]
            for n in recheck_names
            if n.startswith("recheck-previous-") and exits[n] == 0
        ]
        new = [
            rechecks[n]
            for n in recheck_names
            if n.startswith("recheck-main-") and exits[n] == 0
        ]
        for label, group in (("previous", old), ("main", new)):
            assert all(
                r["metadata"]["commit"] == info["started_sources"][label]["commit"]
                for r in group
            )
        if old and new:
            targeted = aggregate(old, new, reference)
            for r in targeted:
                r["previous_completed_processes"] = len(old)
                r["main_completed_processes"] = len(new)
    diagnostic_commands = json.loads((ROOT / "diagnostic-commands.json").read_text())
    diagnostics = [
        read_run(c["label"], complete=c["exit_code"] == 0) for c in diagnostic_commands
    ]
    all_runs = previous + current + list(rechecks.values()) + diagnostics
    for run in all_runs:
        for field in MATCH_METADATA:
            assert run["metadata"][field] == reference[field], field
    initial = []
    for path in sorted((ROOT / "initial-attempt").glob("*.json")):
        data = json.loads(path.read_text())
        if "metadata" not in data or "results" not in data:
            continue
        for field in MATCH_METADATA:
            assert data["metadata"][field] == reference[field], (path.name, field)
        assert all(
            r["checked"] and r["graph_changed_query_checked"] for r in data["results"]
        ), path.name
        initial.append(data)
    initial_commands = json.loads((ROOT / "initial-attempt/commands.json").read_text())
    failures = [c for c in initial_commands if c["exit_code"] != 0]
    regular = [r for r in rows if r["suite"] != "cluster-boundary"]
    groups = []
    for profile in ("atom", "pr"):
        for dtype in ("bf16", "fp8"):
            selected = [
                r for r in regular if r["profile"] == profile and r["kv_dtype"] == dtype
            ]
            groups.append(
                {
                    "profile": profile,
                    "kv_dtype": dtype,
                    "shapes": len(selected),
                    "geomean_latency_change_pct": 100
                    * (geomean(r["main_us"] / r["previous_us"] for r in selected) - 1),
                    "speedup_min": min(r["speedup"] for r in selected),
                    "speedup_max": max(r["speedup"] for r in selected),
                }
            )
    metadata = {
        "previous_commit": previous[0]["metadata"]["commit"],
        "main_commit": current[0]["metadata"]["commit"],
        "physical_gpu": 1,
        "previous_runs": [n + ".json" for n in names["previous"]],
        "main_runs": [n + ".json" for n in names["main"]],
        "targeted_recheck_runs": [n + ".json" for n in recheck_names],
        "targeted_aggregate_statistic": "median of completed processes only; counts recorded per row; failed runs retained separately",
        "targeted_recheck_failures": [
            {k: c[k] for k in ("label", "exit_code", "started_utc", "ended_utc")}
            for c in recheck_commands
            if c["exit_code"] != 0
        ],
        "diagnostic_processes": [
            {
                "label": c["label"],
                "exit_code": c["exit_code"],
                "commit": run["metadata"]["commit"],
                "completed_measurements": len(run["results"]),
            }
            for c, run in zip(diagnostic_commands, diagnostics)
        ],
        "aggregate_statistic": "median of 3 process medians; each process uses median of 5 round minima",
        "regression_screen_pct": 5,
        "regressions_above_screen": [
            {k: r[k] for k in ("case", "profile", "kv_dtype", "latency_change_pct")}
            for r in rows
            if r["latency_change_pct"] > 5
        ],
        "geomean_latency_change_pct_all_84": 100
        * (geomean(r["main_us"] / r["previous_us"] for r in rows) - 1),
        "geomean_latency_change_pct_regular_72": 100
        * (geomean(r["main_us"] / r["previous_us"] for r in regular) - 1),
        "regular_groups": groups,
        "checked_measurements": sum(len(r["results"]) for r in all_runs),
        "initial_attempt_completed_measurements": sum(
            len(r["results"]) for r in initial
        ),
        "initial_attempt_failures": [
            {k: c[k] for k in ("label", "exit_code", "started_utc", "ended_utc")}
            for c in failures
        ],
        "max_fp8_row_relative_l2": max(
            r["max_row_relative_l2_error"]
            for run in all_runs
            for r in run["results"]
            if r["kv_dtype"] == "fp8"
        ),
    }
    (ROOT / "summary.json").write_text(
        json.dumps(
            {"metadata": metadata, "results": rows, "targeted_recheck": targeted},
            indent=2,
        )
        + "\n"
    )
    fields = [k for k in rows[0] if not isinstance(rows[0][k], list)]
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metadata, indent=2))
    print("Largest aggregate slowdowns:")
    for r in sorted(rows, key=lambda r: r["latency_change_pct"], reverse=True)[:10]:
        print(
            r["profile"],
            r["kv_dtype"],
            r["case"],
            r["latency_change_pct"],
            r["previous_run_medians_us"],
            r["main_run_medians_us"],
        )


if __name__ == "__main__":
    main()
