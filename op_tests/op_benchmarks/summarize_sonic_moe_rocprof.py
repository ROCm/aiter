# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


COUNTER_COLUMNS = {
    "MfmaUtil": "mfma_util_pct",
    "VALUUtilization": "valu_util_pct",
    "FetchSize": "fetch_kib",
    "WriteSize": "write_kib",
    "MemUnitStalled": "mem_stalled_pct",
    "LdsUtil": "lds_util_pct",
    "LDSBankConflict": "lds_bank_conflict_pct",
    "MeanOccupancyPerCU": "occupancy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofv3 CSV counters from SonicMoE stage profiles."
    )
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=None,
        help=(
            "Optional stage sweep CSV. When provided, summary timing uses the "
            "best non-profiling balanced/block_m row instead of rocprof-inflated "
            "kernel timing."
        ),
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--md", type=Path, default=None)
    return parser.parse_args()


def _read_profile_json(raw_dir: Path) -> dict[str, Any]:
    stdout = raw_dir / "stdout.log"
    for line in stdout.read_text().splitlines():
        if line.startswith("profile_stage_json="):
            return json.loads(line.split("=", 1)[1])
    raise RuntimeError(f"missing profile_stage_json in {stdout}")


def _read_counter(raw_dir: Path, counter: str, stage: str) -> dict[str, float]:
    csv_files = list(raw_dir.glob("*_counter_collection.csv"))
    if len(csv_files) != 1:
        raise RuntimeError(f"expected one counter CSV under {raw_dir}, got {csv_files}")

    kernel_rows: list[dict[str, str]] = []
    with csv_files[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kernel_name = row.get("Kernel_Name", "")
            if (
                "kernel_moe_gemm" not in kernel_name
                and "ck_moe_stage" not in kernel_name
            ):
                continue
            if row.get("Counter_Name") != counter:
                continue
            kernel_rows.append(row)

    if not kernel_rows:
        raise RuntimeError(f"missing counter {counter} in {csv_files[0]}")

    # The stage2 profile driver runs stage1 once to fill A2 before the timed
    # stage2 loop. rocprof sees that warmup GEMM, so drop the first matching CK
    # MoE GEMM row and summarize only the stage2 kernel rows.
    if stage == "stage2" and len(kernel_rows) > 1:
        kernel_rows = kernel_rows[1:]

    values = [float(row["Counter_Value"]) for row in kernel_rows]
    lds_bytes = [float(row["LDS_Block_Size"]) for row in kernel_rows]
    vgprs = [float(row["VGPR_Count"]) for row in kernel_rows]
    return {
        "counter_value": mean(values),
        "lds_bytes": max(lds_bytes),
        "vgpr": max(vgprs),
    }


def _best_sweep_times(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    best_by_shape: dict[str, dict[str, Any]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("returncode") != "0":
                continue
            if row.get("routing") != "balanced":
                continue
            if row.get("block_m") != "128":
                continue
            shape = row["shape_str"]
            current = best_by_shape.get(shape)
            if current is None or float(row["direct_fused_moe_ms"]) < float(
                current["direct_fused_moe_ms"]
            ):
                best_by_shape[shape] = row

    times: dict[tuple[str, str], dict[str, float]] = {}
    for shape, row in best_by_shape.items():
        times[(shape, "stage1")] = {"bench_ms": float(row["stage1_ms"])}
        times[(shape, "stage2")] = {"bench_ms": float(row["stage2_ms"])}
    return times


def collect_rows(
    profile_dir: Path, sweep_csv: Path | None
) -> list[dict[str, str | float]]:
    status = profile_dir / "run_status.csv"
    if not status.exists():
        raise FileNotFoundError(status)

    sweep_times = _best_sweep_times(sweep_csv) if sweep_csv and sweep_csv.exists() else {}
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)

    with status.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["returncode"] != "0":
                continue
            raw_dir = Path(row["raw_dir"])
            if not raw_dir.is_absolute():
                raw_dir = profile_dir / raw_dir
            profile = _read_profile_json(raw_dir)
            counter = row["counter"]
            counter_result = _read_counter(raw_dir, counter, profile["stage"])
            key = (profile["shape_str"], profile["stage"])
            record = grouped[key]
            record.update(
                {
                    "shape": profile["shape_str"],
                    "stage": profile["stage"],
                    "profile_ms": profile["stage_ms"],
                    "a2_mib": profile["a2_mib"],
                    "block_m": profile["block_m"],
                    "routing": profile["routing"],
                    "tile_efficiency": profile["tile_efficiency"],
                }
            )
            record[COUNTER_COLUMNS[counter]] = counter_result["counter_value"]
            record["lds_bytes"] = counter_result["lds_bytes"]
            record["vgpr"] = counter_result["vgpr"]

    rows: list[dict[str, str | float]] = []
    for key in sorted(grouped):
        record = grouped[key]
        timing = sweep_times.get(key, {})
        bench_ms = timing.get("bench_ms", record["profile_ms"])
        fetch_kib = float(record.get("fetch_kib", 0.0))
        write_kib = float(record.get("write_kib", 0.0))
        rw_mib = (fetch_kib + write_kib) / 1024.0
        approx_hbm = rw_mib * 1.048576 / float(bench_ms)
        rows.append(
            {
                "shape": record["shape"],
                "stage": record["stage"],
                "routing": record["routing"],
                "block_m": record["block_m"],
                "bench_ms": bench_ms,
                "profile_ms": record["profile_ms"],
                "mfma_util_pct": record.get("mfma_util_pct", ""),
                "valu_util_pct": record.get("valu_util_pct", ""),
                "mem_stalled_pct": record.get("mem_stalled_pct", ""),
                "lds_util_pct": record.get("lds_util_pct", ""),
                "lds_bank_conflict_pct": record.get("lds_bank_conflict_pct", ""),
                "occupancy": record.get("occupancy", ""),
                "fetch_kib": fetch_kib,
                "write_kib": write_kib,
                "read_write_mib": rw_mib,
                "approx_hbm_gb_s": approx_hbm,
                "logical_a2_mib": record["a2_mib"],
                "logical_a2_read_write_mib": float(record["a2_mib"]) * 2.0,
                "tile_efficiency": record["tile_efficiency"],
                "vgpr": record.get("vgpr", ""),
                "lds_bytes": record.get("lds_bytes", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    fields = [
        "shape",
        "stage",
        "routing",
        "block_m",
        "bench_ms",
        "profile_ms",
        "mfma_util_pct",
        "valu_util_pct",
        "mem_stalled_pct",
        "lds_util_pct",
        "lds_bank_conflict_pct",
        "occupancy",
        "fetch_kib",
        "write_kib",
        "read_write_mib",
        "approx_hbm_gb_s",
        "logical_a2_mib",
        "logical_a2_read_write_mib",
        "tile_efficiency",
        "vgpr",
        "lds_bytes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: str | float, digits: int = 2) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def write_md(path: Path, rows: list[dict[str, str | float]], profile_dir: Path) -> None:
    lines = [
        "# ROCprof Stage Summary",
        "",
        f"Profile directory: {profile_dir}",
        "",
        (
            "Counter notes: FetchSize and WriteSize are rocprof derived KiB "
            "counters. Approximate HBM GB/s uses non-profiling stage timing "
            "from the stage sweep when available, because counter collection "
            "inflates kernel runtime."
        ),
        "",
        (
            "| Shape | Stage | Bench ms | MFMA util % | VALU util % | "
            "Mem stalled % | LDS util % | LDS bank conflict % | Occupancy | "
            "R+W MiB | Approx HBM GB/s | Logical A2 R+W MiB | VGPR | LDS bytes |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["shape"]),
                    str(row["stage"]),
                    _fmt(row["bench_ms"], 4),
                    _fmt(row["mfma_util_pct"]),
                    _fmt(row["valu_util_pct"]),
                    _fmt(row["mem_stalled_pct"]),
                    _fmt(row["lds_util_pct"]),
                    _fmt(row["lds_bank_conflict_pct"]),
                    _fmt(row["occupancy"]),
                    _fmt(row["read_write_mib"], 1),
                    _fmt(row["approx_hbm_gb_s"], 1),
                    _fmt(row["logical_a2_read_write_mib"], 1),
                    _fmt(row["vgpr"], 0),
                    _fmt(row["lds_bytes"], 0),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Initial Interpretation",
            "",
            "- Stage1 is the largest stage for I=1024; stage2 is the largest stage for I=512.",
            "- Stage1 reaches about 40-42% MFMA utilization. Stage2 is lower, about 14-26%, and has higher measured read/write traffic.",
            "- LDS bank conflicts are low, so LDS banking is not the main limiter for these CK kernels.",
            "- Logical A2 materialization is 512 MiB per forward at I=512 and 1024 MiB per forward at I=1024, before counting normal input and weight traffic.",
            "- The near-ideal balanced routing tile efficiency means the next optimization should target stage dataflow/A2 traffic instead of only token rounding.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    csv_path = args.csv or profile_dir / "rocprof_stage_summary.csv"
    md_path = args.md or profile_dir / "rocprof_stage_summary.md"
    rows = collect_rows(profile_dir, args.sweep_csv)
    write_csv(csv_path, rows)
    write_md(md_path, rows, profile_dir)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
