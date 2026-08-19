# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared BV sweep helpers for FlyDSL K5 mfma16_hip tuning."""

from __future__ import annotations

import importlib.util
import math
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from aiter.jit.core import AITER_ROOT_DIR
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    _GFX_ARCH,
    _hipeq_select_bv,
)
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip as k5,
)
from aiter.utility.base_tuner import _read_csv

CHUNK_SIZE = 64
BV_CANDIDATES = (16, 32, 64)
_K5_TEST_PATH = (
    Path(AITER_ROOT_DIR)
    / "op_tests"
    / "flydsl_tests"
    / "test_flydsl_linear_attention_prefill.py"
)
LOOKUP_KEYS = (
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
TUNED_EXTRA_COLS = ("dtype", "K", "V", "BT", "T_flat", "N", "BV", "us")
TUNED_COLUMNS = LOOKUP_KEYS + TUNED_EXTRA_COLS


def load_k5_cases():
    spec = importlib.util.spec_from_file_location(
        "_gdn_k5_prefill_cases", _K5_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(zip(module.PREFILL_TEST_IDS, module.PREFILL_PARAMS, strict=True))


def read_csv_rows(path: str) -> list[dict[str, str]]:
    df = _read_csv(path, comment="#")
    rows = []
    for row in df.to_dict("records"):
        rows.append(
            {
                key: (
                    ""
                    if val is None or (isinstance(val, float) and math.isnan(val))
                    else str(val)
                )
                for key, val in row.items()
            }
        )
    return rows


def bool_cell(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip() == "True"


def chunk_counts(context_lens, batch):
    per_seq = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in context_lens]
    return sum(per_seq) * batch, max(per_seq)


def case_snapshot_dtype(case):
    return case.snapshot_dtype or case.dtype


def untuned_has_model(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and "model" in rows[0]


def case_matches_untuned_shape(case, row: dict[str, str]) -> bool:
    bt = int(row.get("BT") or case.BT or 64)
    use_h0 = bool_cell(row.get("use_h0") or "True")
    return (
        case.K == int(row["K"])
        and case.V == int(row["V"])
        and case.BT == bt
        and case.H == int(row["H"])
        and case.Hg == int(row["Hg"])
        and case.is_varlen == bool_cell(row["is_varlen"])
        and use_h0
        and case.output_final_state == bool_cell(row["store_fs"])
    )


def select_cases(cases, untuned_rows, case_patterns: list[str]):
    patterns = [re.compile(p) for p in case_patterns] if case_patterns else []
    use_model = untuned_has_model(untuned_rows)
    untuned_models = (
        {row["model"].strip() for row in untuned_rows if row.get("model")}
        if use_model
        else set()
    )
    selected = []
    for case_id, case in cases:
        if use_model:
            if case.model_name not in untuned_models:
                continue
        elif not any(case_matches_untuned_shape(case, row) for row in untuned_rows):
            continue
        if patterns and not any(p.search(case_id) for p in patterns):
            continue
        selected.append((case_id, case))
    return selected


def build_k5_inputs(case, snapshot_dtype, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    context_lens = case.resolve_context_lens()
    t_flat = sum(context_lens)
    h, hg = case.H, case.Hg
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
        "k": torch.randn((batch, t_flat, hg, case.K), device=dev, dtype=case.dtype),
        "w": torch.randn((batch, h, t_flat, case.K), device=dev, dtype=case.dtype),
        "u": torch.randn((batch, h, t_flat, case.V), device=dev, dtype=case.dtype),
        "g": torch.randn((batch, h, t_flat), device=dev, dtype=torch.float32) * -0.1,
        "initial_state": torch.zeros(
            (num_states, h, case.V, case.K), device=dev, dtype=case.ssm_state_dtype
        ),
        "output_final_state": case.output_final_state,
        "cu_seqlens": cu_seqlens,
        "state_dtype": case.ssm_state_dtype,
        "snapshot_dtype": snapshot_dtype,
        "g_head_major": True,
    }
    total_chunks, max_seq_chunks = chunk_counts(context_lens, batch)
    return args, t_flat, num_states, total_chunks, max_seq_chunks


def bench_us(args, bv: int, warmup: int, iters: int) -> float:
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


def lookup_key_from_case(case, snapshot_dtype, total_chunks, max_seq_chunks) -> tuple:
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


def case_to_tuned_row(
    case,
    snapshot_dtype,
    t_flat,
    n,
    total_chunks,
    max_seq_chunks,
    bv,
    us,
) -> dict[str, Any]:
    return {
        "arch": _GFX_ARCH,
        "dtype": str(case.dtype),
        "K": int(case.K),
        "V": int(case.V),
        "BT": int(case.BT),
        "H": int(case.H),
        "Hg": int(case.Hg),
        "is_varlen": case.is_varlen,
        "use_h0": True,
        "store_fs": case.output_final_state,
        "snapshot_bf16": snapshot_dtype is torch.bfloat16,
        "state_bf16": case.ssm_state_dtype is torch.bfloat16,
        "T_flat": int(t_flat),
        "N": int(n),
        "total_chunks": int(total_chunks),
        "max_seq_chunks": int(max_seq_chunks),
        "BV": int(bv),
        "us": float(f"{us:.1f}"),
    }


def sweep_case_row(
    case_id,
    case,
    warmup: int,
    iters: int,
    only_improvements: bool = False,
) -> dict[str, Any] | None:
    if case.K != 128 or case.V != 128 or case.BT != 64:
        print(f"{case_id:58s} skipped (kernel supports K=V=128, BT=64 only)")
        return None

    snapshot_dtype = case_snapshot_dtype(case)
    inputs, t_flat, n, total_chunks, max_seq_chunks = build_k5_inputs(
        case, snapshot_dtype
    )

    times = {}
    for bv in BV_CANDIDATES:
        if case.V % bv:
            continue
        times[bv] = bench_us(inputs, bv, warmup, iters)

    if not times:
        return None

    best = min(times, key=times.get)
    rule = _hipeq_select_bv(
        torch.device("cuda:0"), case.H, total_chunks, max_seq_chunks
    )
    gain = (times[rule] - times[best]) / times[rule] * 100 if rule in times else 0.0
    cells = " ".join(f"{times.get(bv, float('nan')):8.1f}" for bv in BV_CANDIDATES)
    print(f"{case_id:58s} {total_chunks:7d} {cells} {best:5d} {rule:5d} {gain:6.1f}")

    if only_improvements and best == rule:
        return None

    return case_to_tuned_row(
        case,
        snapshot_dtype,
        t_flat,
        n,
        total_chunks,
        max_seq_chunks,
        best,
        times[best],
    )


def case_matches_row(case, row: dict[str, Any]) -> bool:
    snapshot_dtype = case_snapshot_dtype(case)
    batch = 1 if case.is_varlen else case.dense_batch
    total_chunks, max_seq_chunks = chunk_counts(case.resolve_context_lens(), batch)
    if row.get("model") and case.model_name != row["model"]:
        return False
    return (
        case.H == int(row["H"])
        and case.Hg == int(row["Hg"])
        and case.K == int(row["K"])
        and case.V == int(row["V"])
        and case.is_varlen == bool_cell(row["is_varlen"])
        and case.output_final_state == bool_cell(row["store_fs"])
        and (snapshot_dtype is torch.bfloat16) == bool_cell(row["snapshot_bf16"])
        and (case.ssm_state_dtype is torch.bfloat16) == bool_cell(row["state_bf16"])
        and int(row["total_chunks"]) == total_chunks
        and int(row["max_seq_chunks"]) == max_seq_chunks
    )


def find_case_for_row(cases, row: dict[str, Any]):
    for _, case in cases:
        if case_matches_row(case, row):
            return case
    return None


def dataframe_from_cases(selected: list[tuple[str, Any]]) -> pd.DataFrame:
    rows = []
    for case_id, case in selected:
        snapshot_dtype = case_snapshot_dtype(case)
        batch = 1 if case.is_varlen else case.dense_batch
        total_chunks, max_seq_chunks = chunk_counts(case.resolve_context_lens(), batch)
        row = case_to_tuned_row(
            case,
            snapshot_dtype,
            sum(case.resolve_context_lens()),
            len(case.resolve_context_lens()) if case.is_varlen else batch,
            total_chunks,
            max_seq_chunks,
            bv=0,
            us=0.0,
        )
        row["_case_id"] = case_id
        row["BV"] = pd.NA
        row["us"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)
