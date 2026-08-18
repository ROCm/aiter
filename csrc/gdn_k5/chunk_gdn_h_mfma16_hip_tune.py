# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep BV for the FlyDSL K5 mfma16_hip kernel and maintain its tuned CSV.

Runtime dispatch consults ``aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv``
(via ``AITER_CONFIG_GDN_K5_MFMA16_HIP``) before falling back to the CU/LDS
rule. AOT compile coverage lives in the companion untuned table.

Usage
-----
Tune shapes declared in the untuned table (matched to K5 prefill test cases):

    python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \\
      -i aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv \\
      -o /tmp/chunk_gdn_h_mfma16_hip_tuned.candidate.csv

Replay the checked-in tuned table and compare measured ``us``:

    python3 csrc/gdn_k5/chunk_gdn_h_mfma16_hip_tune.py \\
      --run_config aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from aiter.jit.core import AITER_ROOT_DIR
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    _GFX_ARCH,
    _hipeq_select_bv,
)
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip as k5,
)

CHUNK_SIZE = 64
BV_CANDIDATES = (16, 32, 64)
_DEFAULT_UNTUNED = (
    f"{AITER_ROOT_DIR}/aiter/configs/chunk_gdn_h_mfma16_hip_untuned.csv"
)
_DEFAULT_TUNED = f"{AITER_ROOT_DIR}/aiter/configs/chunk_gdn_h_mfma16_hip_tuned.csv"
_K5_TEST_PATH = (
    Path(AITER_ROOT_DIR)
    / "op_tests"
    / "flydsl_tests"
    / "test_flydsl_linear_attention_prefill.py"
)
_TUNED_HEADER = (
    "model,arch,dtype,K,V,BT,H,Hg,is_varlen,use_h0,store_fs,snapshot_bf16,"
    "state_bf16,T_flat,N,total_chunks,max_seq_chunks,BV,us"
)
_LOOKUP_KEYS = (
    "arch",
    "H",
    "Hg",
    "V",
    "is_varlen",
    "use_h0",
    "store_fs",
    "snapshot_bf16",
    "state_bf16",
    "total_chunks",
    "max_seq_chunks",
)


def _load_k5_cases():
    spec = importlib.util.spec_from_file_location(
        "_gdn_k5_prefill_cases", _K5_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(zip(module.PREFILL_TEST_IDS, module.PREFILL_PARAMS, strict=True))


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(line for line in f if not line.lstrip().startswith("#")))


def _load_untuned_models(path: str) -> set[str]:
    return {row["model"].strip() for row in _read_csv_rows(path) if row.get("model")}


def _chunk_counts(context_lens, batch):
    per_seq = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in context_lens]
    return sum(per_seq) * batch, max(per_seq)


def _bool_cell(value: Any) -> bool:
    return str(value).strip() == "True"


def _build_k5_inputs(case, snapshot_dtype, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    context_lens = case.resolve_context_lens()
    T_flat = sum(context_lens)
    H, Hg = case.H, case.Hg
    batch = 1 if case.is_varlen else case.dense_batch
    num_states = len(context_lens) if case.is_varlen else batch

    cu_seqlens = None
    if case.is_varlen:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(context_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=dev,
        )

    args = {
        "k": torch.randn((batch, T_flat, Hg, case.K), device=dev, dtype=case.dtype),
        "w": torch.randn((batch, H, T_flat, case.K), device=dev, dtype=case.dtype),
        "u": torch.randn((batch, H, T_flat, case.V), device=dev, dtype=case.dtype),
        "g": torch.randn((batch, H, T_flat), device=dev, dtype=torch.float32) * -0.1,
        "initial_state": torch.zeros(
            (num_states, H, case.V, case.K), device=dev, dtype=case.ssm_state_dtype
        ),
        "output_final_state": case.output_final_state,
        "cu_seqlens": cu_seqlens,
        "state_dtype": case.ssm_state_dtype,
        "snapshot_dtype": snapshot_dtype,
        "g_head_major": True,
    }
    total_chunks, max_seq_chunks = _chunk_counts(context_lens, batch)
    return args, T_flat, num_states, total_chunks, max_seq_chunks


def _bench_us(args, bv: int, warmup: int, iters: int) -> float:
    os.environ["FLYDSL_K5_MFMA16HIP_BV"] = str(bv)
    try:
        for _ in range(warmup):
            k5(**args)
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            k5(**args)
            ends[i].record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    finally:
        os.environ.pop("FLYDSL_K5_MFMA16HIP_BV", None)


def _case_snapshot_dtype(case):
    return case.snapshot_dtype or case.dtype


def _case_matches_row(case, row: dict[str, str]) -> bool:
    snapshot_dtype = _case_snapshot_dtype(case)
    batch = 1 if case.is_varlen else case.dense_batch
    total_chunks, max_seq_chunks = _chunk_counts(case.resolve_context_lens(), batch)
    return (
        case.model_name == row["model"]
        and case.H == int(row["H"])
        and case.Hg == int(row["Hg"])
        and case.K == int(row["K"])
        and case.V == int(row["V"])
        and case.is_varlen == _bool_cell(row["is_varlen"])
        and case.output_final_state == _bool_cell(row["store_fs"])
        and (snapshot_dtype is torch.bfloat16) == _bool_cell(row["snapshot_bf16"])
        and (case.ssm_state_dtype is torch.bfloat16) == _bool_cell(row["state_bf16"])
        and total_chunks == int(row["total_chunks"])
        and max_seq_chunks == int(row["max_seq_chunks"])
    )


def _find_case_for_row(cases, row: dict[str, str]):
    for _, case in cases:
        if _case_matches_row(case, row):
            return case
    return None


def _lookup_key_from_row(row: dict[str, str]) -> tuple:
    bool_cols = {"is_varlen", "use_h0", "store_fs", "snapshot_bf16", "state_bf16"}
    int_cols = {"H", "Hg", "V", "total_chunks", "max_seq_chunks"}
    out = []
    for key in _LOOKUP_KEYS:
        value = row[key]
        if key in bool_cols:
            out.append(_bool_cell(value))
        elif key in int_cols:
            out.append(int(value))
        else:
            out.append(str(value).strip())
    return tuple(out)


def _lookup_key_from_case(case, snapshot_dtype, total_chunks, max_seq_chunks) -> tuple:
    return (
        _GFX_ARCH,
        case.H,
        case.Hg,
        case.V,
        case.is_varlen,
        True,
        case.output_final_state,
        snapshot_dtype is torch.bfloat16,
        case.ssm_state_dtype is torch.bfloat16,
        total_chunks,
        max_seq_chunks,
    )


def _format_tuned_row(
    case,
    snapshot_dtype,
    T_flat,
    N,
    total_chunks,
    max_seq_chunks,
    bv,
    us,
) -> str:
    return (
        f"{case.model_name},{_GFX_ARCH},{case.dtype},{case.K},{case.V},"
        f"{case.BT},{case.H},{case.Hg},{case.is_varlen},True,"
        f"{case.output_final_state},"
        f"{snapshot_dtype is torch.bfloat16},"
        f"{case.ssm_state_dtype is torch.bfloat16},{T_flat},{N},"
        f"{total_chunks},{max_seq_chunks},{bv},{us:.1f}"
    )


def _select_cases(cases, untuned_models: set[str], case_patterns: list[str]):
    patterns = [re.compile(p) for p in case_patterns] if case_patterns else []
    selected = []
    for case_id, case in cases:
        if case.model_name not in untuned_models:
            continue
        if patterns and not any(p.search(case_id) for p in patterns):
            continue
        selected.append((case_id, case))
    return selected


def _sweep_case(case_id, case, warmup, iters, only_improvements):
    if case.K != 128 or case.V != 128 or case.BT != 64:
        print(f"{case_id:58s} skipped (kernel supports K=V=128, BT=64 only)")
        return None

    snapshot_dtype = _case_snapshot_dtype(case)
    inputs, T_flat, N, total_chunks, max_seq_chunks = _build_k5_inputs(
        case, snapshot_dtype
    )

    times = {}
    for bv in BV_CANDIDATES:
        if case.V % bv:
            continue
        times[bv] = _bench_us(inputs, bv, warmup, iters)

    best = min(times, key=times.get)
    rule = _hipeq_select_bv(
        torch.device("cuda:0"), case.H, total_chunks, max_seq_chunks
    )
    gain = (times[rule] - times[best]) / times[rule] * 100 if rule in times else 0.0
    cells = " ".join(f"{times.get(bv, float('nan')):8.1f}" for bv in BV_CANDIDATES)
    print(
        f"{case_id:58s} {total_chunks:7d} {cells} {best:5d} {rule:5d} {gain:6.1f}"
    )

    if only_improvements and best == rule:
        return None

    return _format_tuned_row(
        case,
        snapshot_dtype,
        T_flat,
        N,
        total_chunks,
        max_seq_chunks,
        best,
        times[best],
    )


def _write_tuned_rows(path: Path, rows: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_TUNED_HEADER + "\n")
        for line in rows:
            f.write(line + "\n")


def _run_tune(args):
    untuned_models = _load_untuned_models(args.untune_file)
    cases = _load_k5_cases()
    selected = _select_cases(cases, untuned_models, args.case)
    if not selected:
        print(f"no case matches untuned models in {args.untune_file}")
        sys.exit(1)

    header = (
        f"{'case':58s} {'chunks':>7s} "
        + " ".join(f"BV{bv:<7d}" for bv in BV_CANDIDATES)
    ) + f" {'best':>5s} {'rule':>5s} {'gain%':>6s}"
    print(header)
    print("-" * len(header))

    emitted: dict[tuple, str] = {}
    for case_id, case in selected:
        line = _sweep_case(case_id, case, args.warmup, args.iters, args.only_improvements)
        if line is None:
            continue
        snapshot_dtype = _case_snapshot_dtype(case)
        batch = 1 if case.is_varlen else case.dense_batch
        total_chunks, max_seq_chunks = _chunk_counts(
            case.resolve_context_lens(), batch
        )
        key = _lookup_key_from_case(
            case, snapshot_dtype, total_chunks, max_seq_chunks
        )
        emitted[key] = line
        torch.cuda.empty_cache()

    rows = list(emitted.values())
    print(f"\n{len(rows)} tuned rows")
    out_path = Path(args.tune_file)
    if rows:
        _write_tuned_rows(out_path, rows)
        print(f"wrote {out_path}")
    return rows


def _run_config(args, config_file: str):
    rows = _read_csv_rows(config_file)
    if not rows:
        print(f"no rows in {config_file}")
        return []

    cases = _load_k5_cases()
    results = []
    header = f"{'model':40s} {'BV':>4s} {'csv_us':>8s} {'live_us':>8s} {'delta%':>7s} {'status':>8s}"
    print(header)
    print("-" * len(header))

    for row in rows:
        case = _find_case_for_row(cases, row)
        shape = f"{row['model']} tc={row['total_chunks']}"
        if case is None:
            results.append({"shape": shape, "us": -1, "status": "no_case"})
            print(f"{shape:40s} {'':4s} {float(row['us']):8.1f} {'':8s} {'':7s} {'no_case':>8s}")
            continue

        snapshot_dtype = _case_snapshot_dtype(case)
        inputs, *_rest = _build_k5_inputs(case, snapshot_dtype)
        bv = int(row["BV"])
        csv_us = float(row["us"])
        live_us = _bench_us(inputs, bv, args.warmup, args.iters)
        delta = (live_us - csv_us) / csv_us * 100 if csv_us > 0 else 0.0
        status = "ok" if abs(delta) <= max(args.run_config_tol_pct, 0.0) else "drift"
        results.append({"shape": shape, "us": live_us, "status": status})
        print(
            f"{shape:40s} {bv:4d} {csv_us:8.1f} {live_us:8.1f} {delta:7.1f} {status:>8s}"
        )
        del inputs
        torch.cuda.empty_cache()
    return results


def _merge_improved(existing_path: Path, candidate_rows: list[str], min_pct: float):
    existing = {}
    if existing_path.is_file():
        for row in _read_csv_rows(str(existing_path)):
            existing[_lookup_key_from_row(row)] = row

    candidates = {}
    for line in candidate_rows:
        row = next(csv.DictReader([_TUNED_HEADER, line]))
        candidates[_lookup_key_from_row(row)] = row

    merged = dict(existing)
    updated = 0
    for key, row in candidates.items():
        new_us = float(row["us"])
        old = existing.get(key)
        if old is None:
            merged[key] = row
            updated += 1
            continue
        old_us = float(old["us"])
        if old_us <= 0 or (old_us - new_us) / old_us * 100 >= min_pct:
            merged[key] = row
            updated += 1

    out_rows = []
    for row in merged.values():
        out_rows.append(
            ",".join(str(row[col]) for col in _TUNED_HEADER.split(","))
        )
    _write_tuned_rows(existing_path, out_rows)
    print(f"updated {updated} rows in {existing_path}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--untune_file",
        "--input_file",
        default=_DEFAULT_UNTUNED,
        dest="untune_file",
        help="untuned shape list (model names select K5 prefill cases)",
    )
    parser.add_argument(
        "-o",
        "--tune_file",
        "--tuned_file",
        default=_DEFAULT_TUNED,
        dest="tune_file",
        help="tuned BV table output (use /tmp/... for candidate runs)",
    )
    parser.add_argument(
        "--run_config",
        nargs="?",
        const=True,
        default=False,
        help="benchmark rows in the tuned csv (optional path; default: -o)",
    )
    parser.add_argument(
        "--run_config_tol_pct",
        type=float,
        default=5.0,
        help="run_config pass threshold for live_us vs csv us drift (percent)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="after tuning, compare candidate us against the existing -o file",
    )
    parser.add_argument(
        "--update_improved",
        action="store_true",
        help="with --compare, merge rows improved by at least --min_improvement_pct",
    )
    parser.add_argument(
        "--min_improvement_pct",
        type=float,
        default=3.0,
        help="minimum us improvement required for --update_improved",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=[],
        help="optional regex filters on pytest case ids (after untuned model filter)",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--only-improvements",
        action="store_true",
        help="emit a row only when measured BV beats the rule's choice",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if args.update_improved and not args.compare:
        print("--update_improved requires --compare", file=sys.stderr)
        sys.exit(2)

    if args.list_cases:
        for case_id, _ in _load_k5_cases():
            print(case_id)
        return

    if args.run_config:
        config_file = args.tune_file if args.run_config is True else args.run_config
        results = _run_config(args, config_file)
        bad = [r for r in results if r["status"] not in {"ok"}]
        if bad:
            sys.exit(1)
        return

    rows = _run_tune(args)
    if args.compare and rows:
        existing = Path(args.tune_file)
        candidate_path = Path(str(args.tune_file) + ".candidate")
        _write_tuned_rows(candidate_path, rows)
        print(f"\ncompare candidate: {candidate_path}")
        if args.update_improved:
            _merge_improved(existing, rows, args.min_improvement_pct)


if __name__ == "__main__":
    main()
