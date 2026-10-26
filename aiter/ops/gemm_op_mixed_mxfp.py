# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared runtime tuning helpers for mixed MXFP GEMMs."""

import functools
import os

import pandas as pd
import torch

_TUNED_CONFIG_KEY_COLUMNS = ("gfx", "cu_num", "M", "N", "K")
_TUNED_CONFIG_NUMERIC_COLUMNS = ("cu_num", "M", "N", "K", "splitK")
_TUNED_CONFIG_COLUMNS = frozenset(
    {
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "kernelName",
        "splitK",
    }
)


@functools.lru_cache(maxsize=16)
def _get_device_gfx_cu(device_index: int) -> tuple[str, int]:
    """Return runtime architecture and CU count for one concrete device."""
    properties = torch.cuda.get_device_properties(device_index)
    gfx = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if not gfx:
        raise RuntimeError(f"device {device_index} does not report a gfx architecture")
    cu_override = int((os.getenv("CU_NUM") or "0").strip())
    cu_num = cu_override if cu_override != 0 else int(properties.multi_processor_count)
    return gfx, cu_num


@functools.lru_cache(maxsize=16)
def _load_mixed_mxfp_configs(
    tuned_file: str, family: str
) -> dict[tuple[str, int, int, int, int], dict[str, object]]:
    """Load and strictly validate one mixed-MXFP shape-tuning table."""
    family = family.upper()
    if not os.path.exists(tuned_file):
        return {}
    try:
        configs = pd.read_csv(tuned_file)
    except pd.errors.EmptyDataError:
        return {}
    missing = _TUNED_CONFIG_COLUMNS - set(configs.columns)
    if missing:
        raise ValueError(
            f"{tuned_file} is missing required {family} columns: {sorted(missing)}"
        )
    if configs.empty:
        return {}

    configs = configs.copy()
    for column in _TUNED_CONFIG_NUMERIC_COLUMNS:
        numeric = pd.to_numeric(configs[column], errors="raise")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{tuned_file} contains a non-integral {column} value")
        configs[column] = numeric.astype(int)
    for column in ("cu_num", "M", "N", "K"):
        if (configs[column] <= 0).any():
            raise ValueError(f"{tuned_file} contains a non-positive {column} value")
    if configs["gfx"].isna().any() or configs["kernelName"].isna().any():
        raise ValueError(f"{tuned_file} contains an empty gfx or kernelName")
    configs["gfx"] = configs["gfx"].astype(str).str.strip()
    configs["kernelName"] = configs["kernelName"].astype(str).str.strip()

    if (configs["gfx"] == "").any() or (configs["kernelName"] == "").any():
        raise ValueError(f"{tuned_file} contains an empty gfx or kernelName")
    if (configs["splitK"] != 0).any():
        raise ValueError(f"{family} tuned configs must use splitK=0")

    key_columns = list(_TUNED_CONFIG_KEY_COLUMNS)
    duplicate_rows = configs[configs.duplicated(key_columns, keep=False)]
    if not duplicate_rows.empty:
        raise ValueError(
            f"{tuned_file} contains duplicate {family} shape keys:\n"
            f"{duplicate_rows[key_columns + ['kernelName']].to_string(index=False)}"
        )
    return configs.set_index(key_columns).to_dict("index")


def _find_mixed_mxfp_config(
    configs: dict[tuple[str, int, int, int, int], dict[str, object]],
    gfx: str,
    cu_num: int,
    M: int,
    N: int,
    K: int,
    padM: int,
    padN: int,
    padK: int,
) -> tuple[dict[str, object], str, tuple[int, int, int]] | None:
    """Find an exact tuning record before trying its physical padded shape."""
    candidates = [((M, N, K), "exact")]
    padded = (padM, padN, padK)
    if padded != (M, N, K):
        candidates.append((padded, "padded"))

    for (candidate_M, candidate_N, candidate_K), match_kind in candidates:
        config = configs.get((gfx, cu_num, candidate_M, candidate_N, candidate_K))
        if config is not None:
            return (
                config,
                match_kind,
                (candidate_M, candidate_N, candidate_K),
            )
    return None
