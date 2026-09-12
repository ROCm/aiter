# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structured output helpers for benchmark drivers."""

import json

import pandas as pd


def _json_default(value):
    """Render non-native JSON scalars without expanding their attributes."""
    return str(value).removeprefix("torch.")


def print_json_table(name, rows, keep=None):
    """Print benchmark rows as one record-oriented JSON object.

    A single-line object is intentional: parent benchmark drivers can validate
    and forward it without parsing pandas' human-readable table formats.
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame([row for row in rows if row is not None])
    if not df.empty:
        df = df.replace("", pd.NA).dropna(axis=1, how="all")
        if keep is not None:
            cols = [column for column in keep if column in df.columns]
            cols += [
                column
                for column in df.columns
                if "err_msg" in column and column not in cols
            ]
            df = df[cols]
    records = json.loads(
        df.to_json(orient="records", default_handler=_json_default)
    )
    print(json.dumps({"name": name, "rows": records}), flush=True)
