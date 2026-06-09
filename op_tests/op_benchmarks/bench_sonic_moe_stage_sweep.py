# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
STAGE_BREAKDOWN = THIS_DIR / "bench_sonic_moe_stage_breakdown.py"

RESULT_FIELDS = [
    "shape_str",
    "dtype",
    "activation",
    "routing",
    "rounding_tile",
    "block_m",
    "dispatch_policy",
    "ksplit_env",
    "use_nt_env",
    "ck_sorting_env",
    "returncode",
    "direct_fused_moe_ms",
    "direct_fused_moe_expert_tflops",
    "stage_sum_ms",
    "stage_sum_expert_tflops",
    "stage_over_direct",
    "sorting_plus_stage_gap_ms",
    "moe_sorting_ms",
    "stage1_ms",
    "stage2_ms",
    "one_stage_ms",
    "bottleneck",
    "bottleneck_share",
    "tile_efficiency",
    "padding_overhead",
    "route_moved",
    "a2_mib",
    "a2_read_write_mib",
    "run_1stage",
    "stage1_backend",
    "stage2_backend",
    "kernelName1",
    "kernelName2",
    "metadata_block_m",
    "ksplit",
    "use_non_temporal_load",
    "cu_num",
    "gfx",
    "log_path",
]

TUNED_FMOE_FIELDS = [
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "block_m",
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "tflops",
    "bw",
    "_tag",
]


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_values(values: str) -> list[str | None]:
    if not values:
        return [None]
    return _csv(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep SonicMoE-on-AITER stage breakdown knobs and emit benchmark "
            "CSV plus optional tuned_fmoe.csv candidate rows."
        )
    )
    parser.add_argument("--shape", action="append", default=None, help="T,H,I,E,K")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--activation", default="swiglu")
    parser.add_argument("--routing", default="topk,rounded,balanced")
    parser.add_argument("--block-m", default="auto,32,64,128")
    parser.add_argument("--dispatch-policy", default="0")
    parser.add_argument("--ksplit", default="", help="Comma-separated AITER_KSPLIT")
    parser.add_argument("--use-nt", default="", help="Comma-separated AITER_USE_NT")
    parser.add_argument(
        "--ck-sorting",
        default="",
        help="Comma-separated AITER_USE_CK_MOE_SORTING; default uses Opus sorting.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounding-tile", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--online-tune",
        action="store_true",
        help="Set AITER_ONLINE_TUNE=1 in child processes.",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="Run direct-vs-stage correctness in every child run.",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Skip direct fused_moe timing in every child run.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write normalized sweep rows to this CSV path.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Write one result_json object per line.",
    )
    parser.add_argument(
        "--tuned-csv",
        type=Path,
        default=None,
        help="Write tuned_fmoe.csv-shaped candidate rows for the best configs.",
    )
    parser.add_argument(
        "--active-tuned-rows",
        action="store_true",
        help="Leave _tag blank in tuned rows. Default tags rows so AITER ignores them.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Keep child stdout for every run under this directory.",
    )
    parser.add_argument("--keep-log", action="store_true")
    return parser.parse_args()


def _run_one(
    shape: str,
    routing: str,
    block_m: str,
    dispatch_policy: str,
    ksplit: str | None,
    use_nt: str | None,
    ck_sorting: str | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    env = os.environ.copy()
    if ksplit is not None:
        env["AITER_KSPLIT"] = ksplit
    if use_nt is not None:
        env["AITER_USE_NT"] = use_nt
    if ck_sorting is not None:
        env["AITER_USE_CK_MOE_SORTING"] = ck_sorting
    if args.online_tune:
        env["AITER_ONLINE_TUNE"] = "1"

    cmd = [
        sys.executable,
        str(STAGE_BREAKDOWN),
        "--shape",
        shape,
        "--dtype",
        args.dtype,
        "--activation",
        args.activation,
        "--routing",
        routing,
        "--block-size-m",
        block_m,
        "--dispatch-policy",
        dispatch_policy,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--rounding-tile",
        str(args.rounding_tile),
        "--seed",
        str(args.seed),
    ]
    if not args.check_correctness:
        cmd.append("--skip-correctness")
    if args.skip_direct:
        cmd.append("--skip-direct")

    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    result: dict[str, Any] = {
        "shape_str": shape,
        "dtype": args.dtype,
        "activation": args.activation,
        "routing": routing,
        "block_m": block_m,
        "dispatch_policy": dispatch_policy,
        "ksplit_env": ksplit or "default",
        "use_nt_env": use_nt or "default",
        "ck_sorting_env": ck_sorting or "default",
        "returncode": proc.returncode,
    }
    for line in proc.stdout.splitlines():
        if line.startswith("result_json="):
            result.update(json.loads(line.split("=", 1)[1]))
            break
    return result, proc.stdout


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [row for row in rows if row.get("returncode") == 0 and row.get("stage_sum_ms")]
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in ok:
        key = (
            row.get("padded_token", row.get("shape", [""])[0]),
            tuple(row.get("shape", [])),
            row.get("dtype"),
            row.get("activation"),
            row.get("routing"),
        )
        prev = best.get(key)
        if prev is None or float(row["stage_sum_ms"]) < float(prev["stage_sum_ms"]):
            best[key] = row
    return sorted(best.values(), key=lambda row: float(row["stage_sum_ms"]))


def _tuned_row(row: dict[str, Any], active: bool) -> dict[str, Any]:
    token, model_dim, inter_dim, expert, topk = row["shape"]
    run_1stage = bool(row.get("run_1stage"))
    us1 = row.get("one_stage_ms") if run_1stage else row.get("stage1_ms")
    us2 = 0.0 if run_1stage else row.get("stage2_ms")
    us = row.get("stage_sum_ms")
    tag = "" if active else "sonic_moe_stage_sweep"
    return {
        "cu_num": row.get("cu_num", ""),
        "token": row.get("padded_token", token),
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert": expert,
        "topk": topk,
        "act_type": row.get("aiter_activation", ""),
        "dtype": row.get("torch_dtype", ""),
        "q_dtype_a": row.get("q_dtype_a", ""),
        "q_dtype_w": row.get("q_dtype_w", ""),
        "q_type": row.get("q_type", ""),
        "use_g1u1": row.get("use_g1u1", ""),
        "doweight_stage1": row.get("doweight_stage1", ""),
        "block_m": row.get("block_m", row.get("metadata_block_m", "")),
        "ksplit": row.get("ksplit", ""),
        "us1": "" if us1 is None else float(us1) * 1000.0,
        "kernelName1": row.get("kernelName1", ""),
        "err1": "" if row.get("direct_stage_max_abs") is None else "0.0%",
        "us2": "" if us2 is None else float(us2) * 1000.0,
        "kernelName2": row.get("kernelName2", ""),
        "err2": "" if row.get("direct_stage_max_abs") is None else "0.0%",
        "us": "" if us is None else float(us) * 1000.0,
        "run_1stage": int(run_1stage),
        "tflops": row.get("stage_sum_expert_tflops", ""),
        "bw": "",
        "_tag": tag,
    }


def _print_best(rows: list[dict[str, Any]]) -> None:
    ok = [row for row in rows if row.get("returncode") == 0 and row.get("stage_sum_ms")]
    ok.sort(key=lambda row: float(row["stage_sum_ms"]))
    print("\nBEST")
    print(
        "rank shape routing block_m policy ksplit_env use_nt ck_sorting "
        "direct_ms stage_ms tflops bottleneck share tile_eff a2_rw_mib"
    )
    for idx, row in enumerate(ok[:20], start=1):
        print(
            f"{idx} "
            f"{row.get('shape_str')} "
            f"{row.get('routing')} "
            f"{row.get('block_m')} "
            f"{row.get('dispatch_policy')} "
            f"{row.get('ksplit_env')} "
            f"{row.get('use_nt_env')} "
            f"{row.get('ck_sorting_env')} "
            f"{_fmt(row.get('direct_fused_moe_ms'))} "
            f"{_fmt(row.get('stage_sum_ms'))} "
            f"{_fmt(row.get('stage_sum_expert_tflops'), 2)} "
            f"{row.get('bottleneck', 'NA')} "
            f"{_fmt(row.get('bottleneck_share'), 3)} "
            f"{_fmt(row.get('tile_efficiency'), 4)} "
            f"{_fmt(row.get('a2_read_write_mib'), 1)}"
        )


def main() -> None:
    args = parse_args()
    shapes = args.shape or [
        "32768,4096,512,128,8",
        "32768,4096,1024,128,8",
    ]
    routings = _csv(args.routing)
    block_ms = _csv(args.block_m)
    dispatch_policies = _csv(args.dispatch_policy)
    ksplit_values = _env_values(args.ksplit)
    use_nt_values = _env_values(args.use_nt)
    ck_sorting_values = _env_values(args.ck_sorting)

    if args.log_dir is not None:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for idx, combo in enumerate(
        product(
            shapes,
            routings,
            block_ms,
            dispatch_policies,
            ksplit_values,
            use_nt_values,
            ck_sorting_values,
        ),
        start=1,
    ):
        shape, routing, block_m, dispatch_policy, ksplit, use_nt, ck_sorting = combo
        print(
            "RUN "
            f"{idx} shape={shape} routing={routing} block_m={block_m} "
            f"policy={dispatch_policy} ksplit={ksplit or 'default'} "
            f"use_nt={use_nt or 'default'} ck_sorting={ck_sorting or 'default'}",
            flush=True,
        )
        result, log = _run_one(
            shape,
            routing,
            block_m,
            dispatch_policy,
            ksplit,
            use_nt,
            ck_sorting,
            args,
        )
        if args.log_dir is not None:
            log_name = (
                f"{idx:04d}_{shape.replace(',', 'x')}_{routing}_"
                f"bm{block_m}_p{dispatch_policy}_ks{ksplit or 'd'}_"
                f"nt{use_nt or 'd'}_cks{ck_sorting or 'd'}.log"
            )
            log_path = args.log_dir / log_name
            log_path.write_text(log)
            result["log_path"] = str(log_path)
        results.append(result)
        if args.keep_log or result.get("returncode") != 0 or "stage_sum_ms" not in result:
            print(log, flush=True)
        else:
            print(
                "  "
                f"direct_ms={_fmt(result.get('direct_fused_moe_ms'))} "
                f"stage_ms={_fmt(result.get('stage_sum_ms'))} "
                f"tflops={_fmt(result.get('stage_sum_expert_tflops'), 2)} "
                f"bottleneck={result.get('bottleneck')} "
                f"share={_fmt(result.get('bottleneck_share'), 3)} "
                f"tile_eff={_fmt(result.get('tile_efficiency'), 4)}",
                flush=True,
            )

    if args.csv is not None:
        _write_csv(args.csv, results, RESULT_FIELDS)
        print(f"\nwrote_csv={args.csv}")
    if args.jsonl is not None:
        _write_jsonl(args.jsonl, results)
        print(f"wrote_jsonl={args.jsonl}")
    if args.tuned_csv is not None:
        tuned_rows = [
            _tuned_row(row, args.active_tuned_rows)
            for row in _best_rows(results)
            if row.get("shape")
        ]
        _write_csv(args.tuned_csv, tuned_rows, TUNED_FMOE_FIELDS)
        print(f"wrote_tuned_csv={args.tuned_csv}")

    _print_best(results)


if __name__ == "__main__":
    main()
