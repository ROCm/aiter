# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_run(name):
    data = json.loads((ROOT / f"{name}.json").read_text())
    assert data["metadata"]["status"] == "complete", name
    assert not data["metadata"]["dirty"], name
    assert len(data["results"]) == data["metadata"]["expected_results"], name
    assert all(
        row["checked"] and row["graph_changed_query_checked"] for row in data["results"]
    ), name
    return data


def key(row):
    return row["case"], row["profile"], row["kv_dtype"]


def index(data):
    return {key(row): row for row in data["results"]}


previous_names = ["previous", "repeat-previous-2", "repeat-previous-3"]
head_names = ["head", "repeat-head-2", "repeat-head-3"]
previous = [read_run(name) for name in previous_names]
head = [read_run(name) for name in head_names]
main = read_run("main")
all_runs = previous + head + [main]
reference_metadata = previous[0]["metadata"]
for run in all_runs:
    for field in (
        "gpu",
        "torch",
        "flydsl",
        "hip",
        "seed",
        "timing",
        "workgroup_budget",
        "environment",
        "harness_sha256",
    ):
        assert run["metadata"][field] == reference_metadata[field], field
previous_index = [index(run) for run in previous]
head_index = [index(run) for run in head]
main_index = index(main)
summary = []
for row in head[0]["results"]:
    k = key(row)
    old_rows = [run[k] for run in previous_index]
    new_rows = [run[k] for run in head_index]
    for other in old_rows + new_rows + [main_index[k]]:
        for field in (
            "table_width",
            "kv_pool_pages",
            "max_context_length",
            "query_shape",
            "checked",
            "logical_io_bytes",
        ):
            assert other[field] == row[field], (k, field)
    old_us = [r["median_us"] for r in old_rows]
    new_us = [r["median_us"] for r in new_rows]
    old_median, new_median = map(statistics.median, (old_us, new_us))
    summary.append(
        {
            "case": row["case"],
            "suite": row["suite"],
            "profile": row["profile"],
            "kv_dtype": row["kv_dtype"],
            "previous_us": old_median,
            "head_us": new_median,
            "latency_change_pct": 100 * (new_median / old_median - 1),
            "main_us": main_index[k]["median_us"],
            "speedup_vs_main": main_index[k]["median_us"] / new_median,
            "previous_run_medians_us": old_us,
            "head_run_medians_us": new_us,
            "previous_run_spread_pct": 100 * (max(old_us) - min(old_us)) / old_median,
            "head_run_spread_pct": 100 * (max(new_us) - min(new_us)) / new_median,
            "logical_io_bytes": row["logical_io_bytes"],
            "logical_io_throughput_tb_s": row["logical_io_bytes"] / new_median / 1e6,
            "output_mode": row["output_mode"],
            "previous_output_modes": sorted({r["output_mode"] for r in old_rows}),
            "head_output_modes": sorted({r["output_mode"] for r in new_rows}),
            "previous_plan_cap": old_rows[0]["plan_cap"],
            "head_plan_cap": row["plan_cap"],
        }
    )

recheck_names = [
    f"recheck-{label}-{i}" for label in ("previous", "head") for i in range(1, 4)
]
rechecks = {name: read_run(name) for name in recheck_names}
verify_names = [
    f"verify-recheck-{label}-{i}" for label in ("previous", "head") for i in range(1, 4)
]
verify_runs = {name: read_run(name) for name in verify_names}
for run in list(rechecks.values()) + list(verify_runs.values()):
    for field in (
        "gpu",
        "torch",
        "flydsl",
        "hip",
        "seed",
        "timing",
        "workgroup_budget",
        "environment",
        "harness_sha256",
    ):
        assert run["metadata"][field] == reference_metadata[field], field
recheck_summary = []
for r in rechecks["recheck-head-1"]["results"]:
    k = key(r)
    old_us = [
        index(rechecks[f"recheck-previous-{i}"])[k]["median_us"] for i in range(1, 4)
    ]
    new_us = [index(rechecks[f"recheck-head-{i}"])[k]["median_us"] for i in range(1, 4)]
    a, b = map(statistics.median, (old_us, new_us))
    recheck_summary.append(
        {
            "case": r["case"],
            "profile": r["profile"],
            "kv_dtype": r["kv_dtype"],
            "previous_us": a,
            "head_us": b,
            "latency_change_pct": 100 * (b / a - 1),
            "previous_run_medians_us": old_us,
            "head_run_medians_us": new_us,
        }
    )

verify_summary = []
for r in verify_runs["verify-recheck-head-1"]["results"]:
    k = key(r)
    old_us = [
        index(verify_runs[f"verify-recheck-previous-{i}"])[k]["median_us"]
        for i in range(1, 4)
    ]
    new_us = [
        index(verify_runs[f"verify-recheck-head-{i}"])[k]["median_us"]
        for i in range(1, 4)
    ]
    a, b = map(statistics.median, (old_us, new_us))
    verify_summary.append(
        {
            "case": r["case"],
            "profile": r["profile"],
            "kv_dtype": r["kv_dtype"],
            "previous_us": a,
            "head_us": b,
            "latency_change_pct": 100 * (b / a - 1),
            "previous_run_medians_us": old_us,
            "head_run_medians_us": new_us,
        }
    )

geomean_ratio = math.exp(
    statistics.mean(math.log(r["head_us"] / r["previous_us"]) for r in summary)
)
geomean_main_speedup = math.exp(
    statistics.mean(math.log(r["speedup_vs_main"]) for r in summary)
)
all_runs_with_rechecks = all_runs + list(rechecks.values()) + list(verify_runs.values())
totals = sum(len(r["results"]) for r in all_runs_with_rechecks)
metadata = {
    "previous_commit": previous[0]["metadata"]["commit"],
    "head_commit": head[0]["metadata"]["commit"],
    "main_commit": main["metadata"]["commit"],
    "previous_runs": [f"{name}.json" for name in previous_names],
    "head_runs": [f"{name}.json" for name in head_names],
    "main_runs": ["main.json"],
    "recheck_runs": [f"{name}.json" for name in recheck_names],
    "verify_recheck_runs": [f"{name}.json" for name in verify_names],
    "aggregate_statistic": "median of 3 process medians; each process median is median of 5 round minima",
    "regression_screen_pct": 5,
    "regressions_above_screen": [
        {k: r[k] for k in ("case", "profile", "kv_dtype", "latency_change_pct")}
        for r in summary
        if r["latency_change_pct"] > 5
    ],
    "geomean_latency_change_pct": 100 * (geomean_ratio - 1),
    "geomean_speedup_vs_main": geomean_main_speedup,
    "checked_measurements": totals,
    "max_fp8_row_relative_l2": max(
        r["max_row_relative_l2_error"]
        for run in all_runs_with_rechecks
        for r in run["results"]
        if r["kv_dtype"] == "fp8"
    ),
}
(ROOT / "summary.json").write_text(
    json.dumps(
        {
            "metadata": metadata,
            "results": summary,
            "targeted_recheck": recheck_summary,
            "verify_recheck": verify_summary,
        },
        indent=2,
    )
    + "\n"
)
fields = [k for k in summary[0] if not isinstance(summary[0][k], list)]
with (ROOT / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(summary)

print(json.dumps(metadata, indent=2))
print("Largest aggregate slowdowns:")
for r in sorted(summary, key=lambda r: r["latency_change_pct"], reverse=True)[:10]:
    print(
        r["profile"],
        r["kv_dtype"],
        r["case"],
        r["latency_change_pct"],
        r["previous_run_medians_us"],
        r["head_run_medians_us"],
    )
