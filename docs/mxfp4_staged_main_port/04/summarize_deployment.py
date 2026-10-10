#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Verify and summarize completed calibration-winner deployment artifacts offline."""

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path


def main() -> "None":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    base = args.root / "docs/mxfp4_staged_main_port/04"
    jobs = json.loads((base / "manifest.json").read_text())
    assert len(jobs) == 27
    records = []
    tasks = []
    by_fixture = {}
    for job in jobs:
        directory = base / "gpu" / job["id"]
        search = json.loads((directory / "search.json").read_text())
        actions = {}
        for action in ("search", "replay", "aot", "precompiled"):
            task = json.loads(
                (directory / (action + "-task") / "task.json").read_text()
            )
            identity = json.loads(
                (directory / (action + "-task") / "runtime-device.json").read_text()
            )
            assert (
                task["exit_code"] == 0
                and identity["physical_uuid_selector"] == "GPU-5d8a72a332aa124c"
                and identity["pci"] == "0000:f5:00.0"
            )
            tasks.append(
                {
                    "id": job["id"],
                    "action": action,
                    "exit_code": 0,
                    "start_utc": task["process_start_utc"],
                    "exit_utc": task["process_exit_utc"],
                    "implementation_commit": task["implementation_commit"],
                    "physical_uuid": identity["physical_uuid_selector"],
                    "pci": identity["pci"],
                    "task_artifact": str(directory / (action + "-task")),
                }
            )
            if action == "search":
                continue
            result = json.loads((directory / (action + ".json")).read_text())
            if action == "aot":
                assert all(r["compile_time"] is not None for r in result["results"])
                assert {r["stage"] for r in result["results"]} == {1, 2}
                assert all(
                    r["kernel_name"]
                    == search["winner"][
                        "kernelName1" if r["stage"] == 1 else "kernelName2"
                    ]
                    for r in result["results"]
                )
                actions[action] = {
                    "passed": True,
                    "compile_jobs": len(result["jobs"]),
                    "jobs": result["jobs"],
                    "results": result["results"],
                    "cache": result["environment"]["FLYDSL_RUNTIME_CACHE_DIR"],
                    "wall_seconds": result["wall_seconds"],
                    "artifact": str(directory / (action + ".json")),
                }
                continue
            assert (
                result["finite"]
                and math.isfinite(result["cosine_diff"])
                and result["cosine_diff"] <= 0.1
            )
            assert (
                result["kernelName1"] == search["winner"]["kernelName1"]
                and result["kernelName2"] == search["winner"]["kernelName2"]
            )
            assert result["preserve_generated_route_weights"]
            assert (
                result["explicit_q_dtype_a"] == "torch.float4_e2m1fn_x2"
                and result["fuse_quant"] == "fp4"
            )
            if action == "precompiled":
                assert result["environment"]["FLYDSL_RUNTIME_RUN_ONLY"] == "1"
            actions[action] = {
                "passed": True,
                "cosine_diff": result["cosine_diff"],
                "finite": result["finite"],
                "kernelName1": result["kernelName1"],
                "kernelName2": result["kernelName2"],
                "operations": result["operations"],
                "input_route_fingerprints": result["fingerprints"],
                "matched_synchronized_elapsed_median_us": result[
                    "matched_synchronized_elapsed_median_us"
                ],
                "matched_synchronized_real_kernel_trace_span_us": result[
                    "matched_synchronized_real_kernel_trace_span_us"
                ],
                "synchronized_diagnostic_kernel_sum_us": result[
                    "synchronized_diagnostic_kernel_sum_with_main_aggregation_us"
                ],
                "cache": result["environment"]["FLYDSL_RUNTIME_CACHE_DIR"],
                "wall_seconds": result["wall_seconds"],
                "artifact": str(directory / (action + ".json")),
            }
        assert actions["aot"]["cache"] == actions["precompiled"]["cache"]
        assert (
            actions["replay"]["input_route_fingerprints"]
            == actions["precompiled"]["input_route_fingerprints"]
        )
        fixture = (job["model"], job["token"])
        fingerprints = actions["replay"]["input_route_fingerprints"]
        if fixture in by_fixture:
            assert by_fixture[fixture] == fingerprints
        by_fixture[fixture] = fingerprints
        records.append(
            {
                "id": job["id"],
                "model": job["model"],
                "token": job["token"],
                "config": job["config"],
                "mode": job["mode"],
                "fp8": job["fp8"],
                "fp4": job["fp4"],
                "limit": job["limit"],
                "winner_kernel_sum_us": search["winner_us"],
                "actions": actions,
            }
        )
    tasks.sort(key=lambda r: r["start_utc"])
    assert len(tasks) == 108 and all(
        left["exit_utc"] <= right["start_utc"] for left, right in pairwise(tasks)
    )
    summary = {
        "winner_count": 27,
        "deployment_tasks": 81,
        "passed_deployment_tasks": 81,
        "failed_deployment_tasks": 0,
        "overlapping_search_and_winner_deployment_tasks": 0,
        "search_and_deployment_tasks": 108,
        "fingerprints_match_between_modes_and_gpt_configurations": True,
        "kernel_sum_metric": "Main normal continuous device kernel-duration sum; elapsed/span diagnostics below use separate synchronized calls",
        "records": records,
    }
    (base / "deployment-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (base / "search-and-deployment-task-sequence.json").write_text(
        json.dumps({"count": 108, "overlaps": 0, "tasks": tasks}, indent=2) + "\n"
    )
    lines = [
        "# Exact calibration-winner production and precompiled acceptance",
        "",
        "All27 calibration winners passed separate public production replay, matching AOT compilation and later run-only replay:81 independent tasks, all exit0. The27 initial search plus81 winner-deployment processes have0 overlaps on physicalGPU7 UUID`GPU-5d8a72a332aa124c`/PCI`0000:f5:00.0`. The earlier failed Situv2 harness-assertion task is retained separately and is not counted among these81 successes.",
        "",
        "Every replay selected the exact CSVG1/G2 names with explicit A4W4 input, the intended complete stage2 path and finite unchanged Torch cosine metric≤0.1. Every AOT job compiled and run-only replay used its matching cache. Input/routes/weight fingerprints match between ordinary and run-only replay, between search modes, and across allGPT configurations.",
        "",
        "| Winner | Search kernel sum μs | Production error | Run-only error | AOT jobs | Production matched elapsed μs | Production matched trace span μs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        a = row["actions"]
        lines.append(
            f"| {row['id']} | {row['winner_kernel_sum_us']:.4f} | {a['replay']['cosine_diff']:.8f} | {a['precompiled']['cosine_diff']:.8f} | {a['aot']['compile_jobs']} | {a['replay']['matched_synchronized_elapsed_median_us']:.3f} | {a['replay']['matched_synchronized_real_kernel_trace_span_us']:.3f} |"
        )
    lines += [
        "",
        "Search kernel-duration sum is the CSV selection metric. The matched elapsed and real-kernel trace-span columns come from synchronized production calls; they are companion diagnostics, not replacements for the search metric. Synthetic GPU annotations are excluded by the replay harness. AOT compile jobs retain complete effective dtype, persistence, activation parameters and buffer/scale keys in deployment-summary.json and each original aot.json.",
        "",
        "ConfiguredFP8 reduce winners record unweighted packedFP8 routes and final exactly-once route weighting; configuredFP4 scatter winners record packed payload/scales and complete quantized scatter. Both-flags reduce winners retainFP8 output. Required scatter/atomic combinations and active limit5 clamp fixtures remain separately covered by ticket03; winner replay supplies the same configured limit as its search.",
        "",
        "Corrected omitted-argumentSituv2 production/run-only retries also passed with actual4/25 defaults, retained separately in situv2-public-defaults.md. All10 targeted alternating pair checks passed60 finite normal kernel-sum profiles, with median/dispersion/rank analysis in targeted-summary.md. FinalStandards/Spec review is the remaining closing gate.",
    ]
    (base / "deployment-summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
