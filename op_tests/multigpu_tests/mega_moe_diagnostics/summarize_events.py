# SPDX-License-Identifier: MIT
"""Summarize successful HIP-event runs without mixing E2E and stage timings."""

import argparse
import json
from pathlib import Path
import statistics


def summarize(run: Path) -> list[dict]:
    """Return one row per GPU; stage medians pool layers and replays."""
    if (run / "exitcode").read_text().strip() != "0":
        raise ValueError(f"Unsuccessful run: {run}")
    if (run / "boot_before.txt").read_text() != (run / "boot_after.txt").read_text():
        raise ValueError(f"Boot changed during run: {run}")
    records = []
    for rank in range(4):
        data = json.loads((run / f"events_rank{rank}.json").read_text())
        kernels = data["kernels"]
        layers = sum(kernel["stage"] == "gemm1" for kernel in kernels)
        row = {"run": run.name, "rank": rank}
        for stage in ("gemm1", "gemm2", "plan", "ready"):
            values = [
                value
                for kernel in kernels
                if kernel["stage"] == stage
                for value in kernel["samples_us"]
            ]
            if not values or min(values) <= 0:
                raise ValueError(f"Invalid {stage} samples in {run}, GPU{rank}")
            row[f"{stage}_median_us"] = statistics.median(values)
        row["native_e2e_median_us_per_layer"] = data["native_e2e_us"]["median"] / layers
        row["symbols"] = {
            stage: sorted({k["symbol"] for k in kernels if k["stage"] == stage})
            for stage in ("gemm1", "gemm2")
        }
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    records = [row for run in args.runs for row in summarize(run)]
    print("| Run | GPU | GEMM1 us | GEMM2 us | Plan us | Ready us | E2E/layer us |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in records:
        values = [
            row[f"{stage}_median_us"] for stage in ("gemm1", "gemm2", "plan", "ready")
        ]
        values.append(row["native_e2e_median_us_per_layer"])
        fields = " | ".join(f"{value:.2f}" for value in values)
        print(f"| {row['run']} | {row['rank']} | {fields} |")
    if args.json:
        args.json.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
