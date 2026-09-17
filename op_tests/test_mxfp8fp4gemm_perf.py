# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Repeat the native gfx1250 ASM GEMM-only benchmark for six F8GEMM cases.

Run from an aiter checkout in its installed environment:
    python -m op_tests.test_mxfp8fp4gemm_perf

Each case/input pair gets a fresh process, with six complete native test calls
inside it. Input generation, rotation, Torch/AP0 prebenchmarks, GEMM timing and
correctness checks all belong to op_tests.test_mxfp8fp4gemm.
"""

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CASES = {
    "wqkv_a": (512, 2048, 7168),
    "wo_b": (512, 7168, 16384),
    "gate_up_proj": (512, 6144, 7168),
    "w2": (512, 7168, 3072),
    "wq_b": (512, 65536, 1536),
    "indexer_wq_b": (512, 8192, 1536),
}
INPUTS = {"constant": "constant", "uniform": "auto"}
ROOT = Path(__file__).resolve().parents[1]


def native_command(case, data_init, repeat, json_path):
    # Inherit AP1, per-shape split-K, formal 2/100 and prebenchmark 2/100
    # from the native entry. No separate timing or input implementation here.
    return [
        sys.executable,
        "-u",
        "-m",
        "op_tests.test_mxfp8fp4gemm",
        "--mode",
        "perf",
        "--intype",
        "a8w8",
        "--shape",
        ",".join(map(str, CASES[case])),
        "--pre-benchmark",
        "--no-reduce",
        "--repeat",
        str(repeat),
        "--data-init",
        data_init,
        "--scale-init",
        INPUTS[data_init],
        "--json",
        str(json_path),
    ]


def complete_profile(row):
    kernels = row.get("profile_gpu_kernels", {})
    return (
        row.get("timing_scope") == "gemm_only"
        and len(kernels) == 1
        and all("f8gemm_" in name for name in kernels)
        and sum(kernels.values()) == row["num_iters"]
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument(
        "--data-init", nargs="+", choices=list(INPUTS), default=list(INPUTS)
    )
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=15,
        help="Retry the whole case/input group only if formal profiler events are incomplete",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "f8gemm_perf_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        ),
        help="New directory for native JSON/logs, all attempts and summaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print native commands without running GPU work",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.max_attempts < 1:
        parser.error("--repeat and --max-attempts must be positive")
    output = args.output_dir.resolve()
    if args.dry_run:
        for case in args.cases:
            for data_init in args.data_init:
                print(
                    shlex.join(
                        native_command(
                            case,
                            data_init,
                            args.repeat,
                            output / f"{case}_{data_init}_attempt1.json",
                        )
                    )
                )
        return
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", flush=True)
    attempts, records, summaries = [], [], []
    for case in args.cases:
        for data_init in args.data_init:
            for attempt in range(1, args.max_attempts + 1):
                stem = f"{case}_{data_init}_attempt{attempt}"
                json_path, log_path = output / f"{stem}.json", output / f"{stem}.log"
                command = native_command(case, data_init, args.repeat, json_path)
                print(shlex.join(command), flush=True)
                entry = {
                    "case": case,
                    "data_init": data_init,
                    "attempt": attempt,
                    "command": command,
                    "cwd": str(ROOT),
                    "log": log_path.name,
                    "json": json_path.name,
                }
                attempts.append(entry)
                write_json(output / "attempts.json", attempts)
                with log_path.open("w") as log:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                entry["exit_code"] = result.returncode
                write_json(output / "attempts.json", attempts)
                if result.returncode:
                    raise RuntimeError(f"Native test failed; see {log_path}")
                rows = json.loads(json_path.read_text())
                if len(rows) != args.repeat or any("asm us" not in row for row in rows):
                    raise RuntimeError(f"Missing native results; see {json_path}")
                counts = [
                    sum(row.get("profile_gpu_kernels", {}).values()) for row in rows
                ]
                accepted = all(complete_profile(row) for row in rows)
                entry.update(accepted=accepted, raw_formal_event_counts=counts)
                write_json(output / "attempts.json", attempts)
                if not accepted:
                    print(
                        f"Incomplete formal profiler records: {stem}, counts={counts}; retained",
                        flush=True,
                    )
                    continue
                values = [row["asm us"] for row in rows]
                for row in rows:
                    records.append(dict(row, case=case, attempt=attempt))
                summary = {
                    "case": case,
                    "data_init": data_init,
                    "splitk": rows[0]["splitk"],
                    "values_us": values,
                    "mean_us": statistics.mean(values),
                    "correctness": [row["asm result"] for row in rows],
                }
                summaries.append(summary)
                write_json(output / "results.json", records)
                write_json(output / "summary.json", summaries)
                with (output / "perf.csv").open("w", newline="") as stream:
                    columns = (
                        ["case", "data_init", "splitk"]
                        + [f"us_{i}" for i in range(1, args.repeat + 1)]
                        + ["mean_us"]
                    )
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    for item in summaries:
                        row = {
                            key: item[key]
                            for key in ("case", "data_init", "splitk", "mean_us")
                        }
                        row.update(
                            {
                                f"us_{i}": value
                                for i, value in enumerate(item["values_us"], 1)
                            }
                        )
                        writer.writerow(row)
                print(
                    f"{case} {data_init}: {values} us; mean={summary['mean_us']:.4f} us; "
                    f"correctness={summary['correctness']}",
                    flush=True,
                )
                break
            else:
                raise RuntimeError(
                    f"No complete profiler group after {args.max_attempts} attempts: {case}/{data_init}; all attempts retained in {output}"
                )
    print(f"Summary: {output / 'perf.csv'}", flush=True)


if __name__ == "__main__":
    main()
