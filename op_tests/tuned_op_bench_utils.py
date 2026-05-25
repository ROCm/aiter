# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

_DERIVED_METRIC_SUFFIXES = (" us",)
_DROP_SUFFIXES = (" err", " TFLOPS", " TB/s")
_DROP_COLS = {"logits_diff"}


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _cell(value):
    if _is_missing(value):
        return ""
    return str(value)


def _metric_columns(row: Mapping[str, object], metric_cols: Iterable[str] | None):
    if metric_cols is not None:
        return [col for col in metric_cols if col in row]
    if "us" in row:
        return ["us"]
    return [
        col
        for col in row
        if any(col.endswith(suffix) for suffix in _DERIVED_METRIC_SUFFIXES)
    ]


def _impl_from_metric(metric_col: str, default_impl: str) -> str:
    if metric_col == "us":
        return default_impl
    for suffix in _DERIVED_METRIC_SUFFIXES:
        if metric_col.endswith(suffix):
            return metric_col[: -len(suffix)]
    return metric_col


def _base_columns(row: Mapping[str, object], metric_cols: set[str]):
    base = {}
    for col, value in row.items():
        if col in metric_cols or col in _DROP_COLS:
            continue
        if any(col.endswith(suffix) for suffix in _DROP_SUFFIXES):
            continue
        if any(col.endswith(suffix) for suffix in _DERIVED_METRIC_SUFFIXES):
            continue
        base[col] = _cell(value)
    return base


def append_tuned_op_bench_rows(
    csv_path: str | Path,
    rows: Iterable[Mapping[str, object]],
    *,
    op_name: str,
    metric_cols: Iterable[str] | None = None,
    default_impl: str = "",
) -> int:
    """Append benchmark rows to the shared tuned-op CI CSV.

    Input rows are usually wide benchmark dictionaries. This writes a stable
    long-table schema with one `us` metric per row so different operator tests
    can share the same artifact.
    """
    output_rows = []
    for row in rows:
        row_metric_cols = _metric_columns(row, metric_cols)
        metric_col_set = set(row_metric_cols)
        base = _base_columns(row, metric_col_set)
        base["op"] = op_name
        for metric_col in row_metric_cols:
            value = row.get(metric_col)
            if _is_missing(value):
                continue
            out = dict(base)
            impl = _impl_from_metric(metric_col, default_impl)
            if impl:
                out["impl"] = impl
            out["us"] = _cell(value)
            output_rows.append(out)

    if not output_rows:
        return 0

    csv_path = Path(csv_path)
    new_df = pd.DataFrame(output_rows)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        old_df = pd.read_csv(csv_path, dtype=str).fillna("")
        new_df = pd.concat([old_df, new_df.astype(str).fillna("")], ignore_index=True)
    new_df.to_csv(csv_path, index=False)
    return len(output_rows)
