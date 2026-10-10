#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Root coordinator: prescribed search/deployment tasks, sequential on physical GPU7."""

import argparse
import datetime
import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_search_console(console):
    text = console.read_text()
    staged = re.findall(r"\[mxfp4-staged\] .+ \+ .+ (?:us=|validation failed:)", text)
    ordinary = re.findall(r"\[mxfp4-port\] token=.+ \+ .+ us=", text)
    failures = re.findall(r"\[mxfp4-port\] candidate failed:", text)
    summary = re.findall(r"total tuning time is ([0-9.]+) seconds", text)
    fallback_count = len(re.findall(r"\[mxfp4-staged\] exhaustive fallback", text))
    if staged:
        summary_path = Path(__file__).with_name("summarize_calibration.py")
        spec = importlib.util.spec_from_file_location(
            "calibration_summary", summary_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fallback_count = module.staged_fallback_groups(text)
    return {
        "validation_attempts": len(staged) + len(ordinary) + len(failures),
        "g1_screens": len(re.findall(r"\[mxfp4-staged\] GEMM1 .+ screening", text)),
        "g2_screens": len(re.findall(r"\[mxfp4-staged\] GEMM2 .+ screening", text)),
        "fallback_groups": fallback_count,
        "finite_reference_errors": re.findall(r"cosine_diff=([0-9.eE+-]+)", text),
        "tuning_summary_seconds": float(summary[-1]) if summary else None,
    }


def finalize_search(job, job_dir, counts):
    """Capture an exited controller task; this function never launches a search."""
    artifact = job_dir / "search-task"
    task = json.loads((artifact / "task.json").read_text())
    if "process_exit_utc" not in task:
        raise RuntimeError(f"Cannot finalize a running task: {artifact}")
    winner = job_dir / "winner.csv"
    result_path = job_dir / "search.json"
    import csv

    rows = list(csv.DictReader(winner.open())) if winner.exists() else []
    task = json.loads((artifact / "task.json").read_text())
    start = datetime.datetime.fromisoformat(task["process_start_utc"])
    end = datetime.datetime.fromisoformat(task["process_exit_utc"])
    text = (artifact / "console.log.txt").read_text()
    outcome = "valid"
    if task.get("exit_code") != 0 or len(rows) != 1 or float(rows[0]["us"]) <= 0:
        if task.get("exit_code", 0) < 0 or re.search(
            r"worker .* (?:failed|died)|device-side|memory access fault", text
        ):
            outcome = "worker-death"
        elif re.search(
            r"all candidates failed|no valid staged pair|Tuning Finished", text
        ):
            outcome = "normal-pair"
        else:
            outcome = "incomplete-run"
    result = dict(
        job,
        outcome=outcome,
        winner=rows[0] if len(rows) == 1 else None,
        winner_us=float(rows[0]["us"]) if len(rows) == 1 else None,
        search_wall_seconds=(end - start).total_seconds(),
        implementation_commit=task["implementation_commit"],
        environment=task["environment"],
        static_counts=counts,
        **parse_search_console(artifact / "console.log.txt"),
    )
    write_json(result_path, result)
    console = artifact / "console.log.txt"
    with gzip.open(str(console) + ".gz", "wb") as stream:
        stream.write(console.read_bytes())
    # Preserve complete compressed console; avoid unbounded tracked raw logs.
    console.unlink()
    if job["mode"] == "full":
        write_json(
            job_dir / "cost-after-full.json",
            {
                "actual_search_wall_seconds": result["search_wall_seconds"],
                "validation_attempts": result["validation_attempts"],
                "static_pairs": counts["total_pairs"],
                "original_estimate": json.loads(
                    (job_dir / "cost-before-full.json").read_text()
                ),
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("/root/workspace/tmp_dev/mxfp4-port-runtime"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Record exited search tasks without launching GPU work",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Run selected manifest ids; omitted means all 27",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=("search", "replay", "aot", "precompiled"),
        default=["search"],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "docs/mxfp4_staged_main_port/04"
    jobs = json.loads((base / "manifest.json").read_text())
    counts = {
        row["id"]: row
        for row in json.loads((base / "candidate-counts.json").read_text())["records"]
    }
    if args.only and set(args.only) - {job["id"] for job in jobs}:
        parser.error("--only includes an unknown search id")
    controller = root / "docs/mxfp4_staged_main_port/scripts/run_gpu_task.py"
    for job in jobs:
        if args.only and job["id"] not in args.only:
            continue
        job_dir = base / "gpu" / job["id"]
        winner = job_dir / "winner.csv"
        if args.finalize_only:
            if not (job_dir / "search.json").exists():
                finalize_search(job, job_dir, counts[job["id"]])
            continue
        for action in args.actions:
            artifact = job_dir / (action + "-task")
            result_path = job_dir / (
                "search.json" if action == "search" else action + ".json"
            )
            if (artifact / "task.json").exists():
                prior = json.loads((artifact / "task.json").read_text())
                if prior.get("exit_code") == 0 and result_path.exists():
                    print(f"EXISTING PASS {job['id']} {action}", flush=True)
                    continue
                raise RuntimeError(
                    f"Existing failed/incomplete task requires diagnosis: {artifact}"
                )
            if action == "search" and job["mode"] == "full":
                estimates = json.loads(
                    (base / "full-anchor-cost-estimates.json").read_text()
                )
                estimate = next(
                    row for row in estimates["anchors"] if row["id"] == job["id"]
                )
                write_json(
                    job_dir / "cost-before-full.json",
                    dict(
                        estimate,
                        recorded_before_start_utc=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                        compilation_state="Fresh per-search FlyDSL cache; identical initially empty cache condition across modes/configurations; warmed common AITER auxiliary cache.",
                        admitted_counts=counts[job["id"]],
                    ),
                )
                print(
                    f"FULL COST {job['id']}: {estimate['projected_seconds_range']} seconds; {estimate['uncertainty']}",
                    flush=True,
                )
            env = os.environ.copy()
            env.update(
                {
                    "PATH": str(Path(args.python).parent)
                    + os.pathsep
                    + env.get("PATH", ""),
                    "PYTHONPATH": str(root),
                    "GPU_ARCHS": "gfx950",
                    "FLYDSL_GPU_ARCH": "gfx950",
                    "AITER_USE_SYSTEM_TRITON": "1",
                    "MAX_JOBS": "8",
                    "CK_DIR": os.environ.get(
                        "CK_DIR",
                        "/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel",
                    ),
                    "AITER_JIT_DIR": str(args.runtime / "jit-default"),
                    "AITER_FLYDSL_STAGE2_FP8": "1" if job["fp8"] else "0",
                    "AITER_MXFP4_INTERMEDIATE": "1" if job["fp4"] else "0",
                    "AITER_MXFP4_TUNE_SWIGLU_LIMIT": (
                        str(job["limit"]) if job["limit"] is not None else ""
                    ),
                    "TUNE_MOE_EXPERT_BALANCE": "False",
                    "TUNE_MOE_ROUTING_SEED": "0",
                    "FLYDSL_RUNTIME_CACHE_DIR": str(
                        args.runtime
                        / (
                            "calibration-aot-04-" + job["config"]
                            if action in ("aot", "precompiled")
                            else "calibration-search-04-" + job["id"]
                        )
                    ),
                }
            )
            if action == "search":
                search_cache = Path(env["FLYDSL_RUNTIME_CACHE_DIR"])
                if search_cache.exists() and any(search_cache.iterdir()):
                    raise RuntimeError(
                        f"New search requires initially empty FlyDSL cache: {search_cache}"
                    )
            env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
            env.pop("AITER_CONFIG_FMOE", None)
            if action == "precompiled":
                env["FLYDSL_RUNTIME_RUN_ONLY"] = "1"
            if action == "search":
                command = [
                    "csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py",
                    "--mxfp4-flydsl",
                    "--mxfp4-search-mode",
                    job["mode"],
                    "--mp",
                    "1",
                    "--all",
                    "--sort",
                    "True",
                    "--warmup",
                    "5",
                    "--iters",
                    "101",
                    "--errRatio",
                    "0.1",
                    "-i",
                    job["input"],
                    "-o",
                    str(winner),
                ]
                if job["mode"] == "staged":
                    command += ["--mxfp4-screen-topk", "2"]
            else:
                if not winner.exists():
                    raise RuntimeError(
                        f"Deployment requires completed search winner: {winner}"
                    )
                command = [
                    "docs/mxfp4_staged_main_port/scripts/configured_replay.py",
                    "--csv",
                    str(winner),
                    "--result",
                    str(result_path),
                    "--preserve-generated-route-weights",
                    "--action",
                    "replay" if action == "precompiled" else action,
                    "--warmup",
                    "5",
                    "--iters",
                    "101",
                ]
                if job["limit"] is not None:
                    command += ["--swiglu-limit", str(job["limit"])]
            print(f"RUN {job['id']} {action}", flush=True)
            completed = subprocess.run(
                [
                    args.python,
                    str(controller),
                    "--artifact",
                    str(artifact),
                    "--cwd",
                    str(root),
                    "--",
                    *command,
                ],
                cwd=root,
                env=env,
                check=False,
            )
            if action == "search":
                finalize_search(job, job_dir, counts[job["id"]])
            if completed.returncode:
                raise SystemExit(
                    f"Task failed {job['id']} {action} exit={completed.returncode}; diagnose before continuation"
                )
    print("Selected prescribed calibration actions completed", flush=True)


if __name__ == "__main__":
    main()
