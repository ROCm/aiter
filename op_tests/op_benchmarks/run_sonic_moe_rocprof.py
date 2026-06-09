# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DRIVER = Path(__file__).resolve().parent / "profile_sonic_moe_stage_kernel.py"
SUMMARY_DRIVER = (
    Path(__file__).resolve().parent / "summarize_sonic_moe_rocprof.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect rocprofv3 counters for best SonicMoE/AITER stage kernels."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/app/yifehuan_temp/data/sonic_moe_mi355_latest/rocprof"),
    )
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=Path(
            "/app/yifehuan_temp/data/sonic_moe_mi355_latest/"
            "sonic_moe_mi355_stage_sweep.csv"
        ),
        help=(
            "Optional non-profiling sweep CSV used to annotate the summary "
            "with stable stage timings."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    out_dir = (
        args.output_root
        / f"rocprof_stages_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{commit}"
    )
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROFILE_DRIVER, out_dir / PROFILE_DRIVER.name)

    cases = [
        {
            "case_id": "s0_i512_e128",
            "shape": "32768,4096,512,128,8",
            "routing": "balanced",
            "block_m": "128",
        },
        {
            "case_id": "s1_i1024_e128",
            "shape": "32768,4096,1024,128,8",
            "routing": "balanced",
            "block_m": "128",
        },
        {
            "case_id": "s2_i1024_e256",
            "shape": "32768,4096,1024,256,8",
            "routing": "balanced",
            "block_m": "128",
        },
    ]
    counters = [
        "MfmaUtil",
        "VALUUtilization",
        "FetchSize",
        "WriteSize",
        "MemUnitStalled",
        "LdsUtil",
        "LDSBankConflict",
        "MeanOccupancyPerCU",
    ]

    (out_dir / "README.md").write_text(
        "# ROCprof Stage Profiles\n\n"
        f"Branch: yifehuan/sonicmoe\nCommit: {commit}\n\n"
        "Each raw subdirectory contains rocprofv3 CSV output for one "
        "shape/stage/counter run. The driver loops only the selected "
        "AITER CK MoE stage kernel.\n"
    )
    with (out_dir / "cases.json").open("w") as handle:
        json.dump(
            {
                "commit": commit,
                "driver": str(PROFILE_DRIVER),
                "iters": args.iters,
                "warmup": args.warmup,
                "cases": cases,
                "counters": counters,
            },
            handle,
            indent=2,
        )

    status_fields = [
        "case_id",
        "shape",
        "routing",
        "block_m",
        "stage",
        "counter",
        "returncode",
        "raw_dir",
    ]
    with (out_dir / "run_status.csv").open("w", newline="") as status_file, (
        out_dir / "commands.log"
    ).open("w") as command_log:
        writer = csv.DictWriter(status_file, fieldnames=status_fields)
        writer.writeheader()
        for case in cases:
            for stage in ("stage1", "stage2"):
                for counter in counters:
                    name = f"{case['case_id']}_{stage}_{counter}"
                    run_dir = raw_dir / name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        "rocprofv3",
                        "--pmc",
                        counter,
                        "--kernel-include-regex",
                        "kernel_moe_gemm|ck_moe_stage",
                        "-d",
                        str(run_dir),
                        "-o",
                        name,
                        "-f",
                        "csv",
                        "--",
                        sys.executable,
                        str(PROFILE_DRIVER),
                        "--shape",
                        case["shape"],
                        "--routing",
                        case["routing"],
                        "--block-size-m",
                        case["block_m"],
                        "--stage",
                        stage,
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                    ]
                    print(f"RUN {name}", flush=True)
                    command_log.write(" ".join(cmd) + "\n")
                    command_log.flush()
                    with (run_dir / "stdout.log").open("w") as stdout, (
                        run_dir / "stderr.log"
                    ).open("w") as stderr:
                        proc = subprocess.run(
                            cmd,
                            cwd=REPO_ROOT,
                            stdout=stdout,
                            stderr=stderr,
                            text=True,
                            check=False,
                        )
                    writer.writerow(
                        {
                            "case_id": case["case_id"],
                            "shape": case["shape"],
                            "routing": case["routing"],
                            "block_m": case["block_m"],
                            "stage": stage,
                            "counter": counter,
                            "returncode": proc.returncode,
                            "raw_dir": str(run_dir),
                        }
                    )
                    status_file.flush()
                    if proc.returncode != 0:
                        print(f"FAILED {name} rc={proc.returncode}", flush=True)

    if args.sweep_csv.exists():
        subprocess.run(
            [
                sys.executable,
                str(SUMMARY_DRIVER),
                str(out_dir),
                "--sweep-csv",
                str(args.sweep_csv),
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    print(out_dir)


if __name__ == "__main__":
    main()
