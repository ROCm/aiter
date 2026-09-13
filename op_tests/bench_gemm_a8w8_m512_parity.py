# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check the five M512 shapes against PR #5406's separate-reduction profiles.

Each mode uses 100,000 timed GEMMs by default. The child benchmark verifies
GPU occupancy before importing the runtime and monitors it throughout the run.
Use --dry-run to inspect the selected and reference kernels without GPU access.
"""

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from bench_gemm_a8w8_splitk_gfx1250 import _nonnegative_float, _positive_int

_SHAPES = ((6144, 7168), (7168, 3072), (8192, 1536), (2048, 7168), (7168, 16384))
_PR_HEAD = "56f693b7c7144407fded94266805490007178b4b"


def _csv_kernels(contents):
    kernels = {}
    for row in csv.DictReader(io.StringIO(contents)):
        shape = tuple(int(row[field]) for field in ("M", "N", "K"))
        if row["gfx"] != "gfx1250" or row["cu_num"] != "256" or shape[0] != 512:
            continue
        if shape[1:] not in _SHAPES:
            continue
        if row["libtype"] != "flydsl" or shape in kernels:
            raise ValueError(f"Expected one FlyDSL CSV row for {shape}")
        kernels[shape] = row["kernelName"]
    if len(kernels) != len(_SHAPES):
        raise ValueError("CSV does not cover all five M512 shapes on gfx1250/256 CU")
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-ref", default=_PR_HEAD)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apre", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--graph-iters", type=_positive_int, default=1000)
    parser.add_argument("--replays", type=_positive_int, default=10)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--max-regression-pct", type=_nonnegative_float, default=0.0)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    preshuffle = "abpreshuffle" if args.apre else "bpreshuffle"
    csv_path = (
        Path("aiter/configs/model_configs")
        / f"dsv4_a8w8_blockscale_{preshuffle}_tuned_gemm.csv"
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.reference_ref}^{{commit}}"],
        cwd=repo,
        text=True,
    ).strip()
    original = _csv_kernels(
        subprocess.check_output(
            ["git", "show", f"{revision}:{csv_path}"], cwd=repo, text=True
        )
    )
    selected = _csv_kernels((repo / csv_path).read_text())
    plan = [
        {
            "shape": [512, n, k],
            "reference": original[512, n, k],
            "selected": selected[512, n, k],
        }
        for n, k in _SHAPES
    ]
    if args.dry_run:
        print(json.dumps({"reference_ref": revision, "comparisons": plan}, indent=2))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --dry-run is used")
    # A fresh directory prevents stale results from being accepted if a child
    # exits before reaching timing or correctness validation.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, ENABLE_CK="0")
    reports = []
    for case in plan:
        _, n, k = case["shape"]
        result_file = args.output_dir.resolve() / f"512x{n}x{k}.json"
        command = [
            sys.executable,
            str(repo / "op_tests/bench_gemm_a8w8_splitk_gfx1250.py"),
            "--kernel",
            case["selected"],
            "--reference-kernel",
            case["reference"],
            "-m",
            "512",
            "-n",
            str(n),
            "-k",
            str(k),
            "--graph-iters",
            str(args.graph_iters),
            "--replays",
            str(args.replays),
            "--repeats",
            str(args.repeats),
            "--max-regression-pct",
            str(args.max_regression_pct),
            "--json-output",
            str(result_file),
        ]
        if args.apre:
            command.append("--apre")
        if args.fp16:
            command.append("--fp16")
        result = subprocess.run(command, cwd=repo, env=environment, check=False)
        if not result_file.exists():
            raise SystemExit(f"No valid result for {case['shape']}; stopping the suite")
        report = json.loads(result_file.read_text())
        if result.returncode != (0 if report["parity"]["pass"] else 1):
            raise SystemExit(f"Unexpected benchmark failure for {case['shape']}")
        reports.append(report)
    summary = {
        "reference_ref": revision,
        "reference_mode": "original CSV configuration with separate reduction",
        "pass": all(report["parity"]["pass"] for report in reports),
        "shapes": [
            {"shape": report["shape"], **report["parity"]} for report in reports
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit("M512 performance parity failed for one or more shapes")


if __name__ == "__main__":
    main()
