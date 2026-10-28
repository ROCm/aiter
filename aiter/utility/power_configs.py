# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Load power-tuned tables without putting that policy in the JIT.

Latency dispatch keeps using the tables ``jit/core.py`` already merges.
``AITER_SELECT=power`` asks this module to overlay ``aiter/configs/power_configs/``
onto that result. An explicit ``AITER_CONFIG_*`` path never reaches here.
"""

import logging
import os

logger = logging.getLogger("aiter")


def power_merge_conflict(dup_rows, dedup_keys) -> str | None:
    """Why duplicate shapes must not be auto-resolved, or None.

    A power row and a latency row for one shape measure different things.
    Two power rows tuned under different perf levels, caps, or clocks have
    watts that cannot be compared.
    """
    from aiter.utility.tune_result import (
        POWER_CONDITION_COLUMNS,
        SELECT_COLUMN,
        row_select_mode,
    )

    if SELECT_COLUMN not in dup_rows.columns:
        return None
    modes = dup_rows[SELECT_COLUMN].map(row_select_mode)
    condition_cols = [c for c in POWER_CONDITION_COLUMNS if c in dup_rows.columns]
    reasons = []
    for _, group in dup_rows.assign(_mode=modes).groupby(
        dedup_keys, dropna=False, sort=False
    ):
        group_modes = set(group["_mode"])
        if len(group_modes) > 1:
            reasons.append("a power-tuned row and a latency row share a shape")
            continue
        if group_modes == {"power"} and condition_cols:
            states = {
                tuple(str(v) for v in row)
                for row in group[condition_cols].itertuples(index=False)
            }
            if len(states) > 1:
                reasons.append(
                    "power-tuned rows for one shape were tuned under "
                    f"different {', '.join(condition_cols)}"
                )
    if not reasons:
        return None
    return (
        "Not auto-resolved: "
        + "; ".join(sorted(set(reasons)))
        + ". Keep power tables in aiter/configs/power_configs/, or load "
        "one model's table through its AITER_CONFIG_* variable.\n"
    )


def _shape_key_strings(frame, keys):
    """Shape keys as comparable strings; integral floats read back from a merge match ints."""
    import pandas as pd

    normalized = {}
    for key in keys:
        column = frame[key]
        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.notna().all() and (numeric == numeric.round()).all():
            normalized[key] = numeric.astype("int64").astype(str)
        else:
            normalized[key] = column.astype(str)
    return pd.DataFrame(normalized).apply(tuple, axis=1)


def apply_power_overlay(config, config_file, tuned_file_name, env_name):
    """With AITER_SELECT=power, overlay aiter/configs/power_configs/ tables.

    ``config`` is the JIT config object: it still owns shape keys and the
    generic duplicate-shape check. This function only decides which power
    file to lay on top. An explicit AITER_CONFIG_* variable never calls it.
    """
    from pathlib import Path

    from aiter.jit.core import AITER_ROOT_DIR
    from aiter.utility.tune_result import POWER_CONFIG_DIRNAME, power_selection_enabled

    if not power_selection_enabled():
        return config_file
    power_dir = Path(f"{AITER_ROOT_DIR}/aiter/configs/{POWER_CONFIG_DIRNAME}/")
    power_files = sorted(
        p
        for p in power_dir.glob(f"*{tuned_file_name}*.csv")
        if p.is_file() and "untuned" not in p.name
    )
    if not power_files:
        return config_file
    return _overlay_power_tables(
        config, config_file, power_files, tuned_file_name, env_name
    )


def _overlay_power_tables(config, latency_file, power_files, merge_name, env_name):
    """Latency table with each power-tuned shape replaced by its power row.

    Shapes without a power row keep their latency row, so AITER_SELECT=power
    never leaves a shape untuned. No source file is rewritten for a
    power/latency overlap.
    """
    import tempfile
    from pathlib import Path

    import pandas as pd

    from aiter.jit.core import mp_lock
    from aiter.utility.tune_result import SELECT_COLUMN

    pairs = []
    for path in power_files:
        df = pd.read_csv(path)
        if not df.empty:
            pairs.append((str(path), df))
    if not pairs:
        return latency_file
    power_df = pd.concat([df for _, df in pairs], ignore_index=True)
    # Same-condition duplicates inside power_configs/ auto-resolve like any
    # other merge; different conditions raise.
    config._raise_on_duplicate_shapes(power_df, pairs, merge_name, env_name)
    keys = config._shape_dedup_keys(power_df, merge_name)
    if not keys or not os.path.exists(latency_file):
        return latency_file
    latency_df = pd.read_csv(latency_file)
    if not all(k in latency_df.columns for k in keys):
        return latency_file
    columns = list(latency_df.columns) + [
        c for c in power_df.columns if c not in latency_df.columns
    ]
    # Fill a column one side lacks the way the model_configs/ merge does:
    # a blank cell turns an int column into floats, which the AOT parser
    # rejects.
    fill = {"xbf16": 0, "run_1stage": 0, "ksplit": 0, SELECT_COLUMN: "latency"}
    for frame in (latency_df, power_df):
        for column in columns:
            if column not in frame.columns:
                frame[column] = fill.get(column, 0)
    latency_df = latency_df[columns]
    power_df = power_df[columns]
    covered = set(_shape_key_strings(power_df, keys))
    keep = ~_shape_key_strings(latency_df, keys).isin(covered)
    merged = pd.concat([latency_df[keep], power_df], ignore_index=True)

    config_path = Path(tempfile.gettempdir()) / "aiter_configs"
    config_path.mkdir(parents=True, exist_ok=True)
    new_file_path = str(config_path / f"{merge_name}.power.csv")
    lock_path = f"{new_file_path}.lock"
    tmp_file_path = f"{new_file_path}.tmp"

    def write_config():
        merged.to_csv(tmp_file_path, index=False)
        os.replace(tmp_file_path, new_file_path)

    mp_lock(lock_path, write_config)
    logger.info(
        f"AITER_SELECT=power: {int((~keep).sum())} shapes of '{merge_name}' "
        f"use power rows from {[p for p, _ in pairs]}"
    )
    return new_file_path
