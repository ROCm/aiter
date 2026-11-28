# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared runtime tuning helpers for mixed MXFP GEMMs."""

import functools
import os

import pandas as pd
import torch

from ..jit.utils.chip_info import get_tuned_cu_nums, log_tuned_cu_fallback

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
    device: int | None = None,
) -> tuple[dict[str, object], str, tuple[int, int, int]] | None:
    """Find an exact tuning record before trying its physical padded shape.

    At each shape, the CU counts of get_tuned_cu_nums(cu_num, device) are tried
    in turn.
    """
    candidates = [((M, N, K), "exact")]
    padded = (padM, padN, padK)
    if padded != (M, N, K):
        candidates.append((padded, "padded"))

    cu_nums = get_tuned_cu_nums(cu_num, device)
    for (candidate_M, candidate_N, candidate_K), match_kind in candidates:
        for tuned_cu in cu_nums:
            config = configs.get((gfx, tuned_cu, candidate_M, candidate_N, candidate_K))
            if config is not None:
                if tuned_cu != cu_num:
                    log_tuned_cu_fallback("mixed MXFP GEMM", gfx, cu_num, tuned_cu)
                return (
                    config,
                    match_kind,
                    (candidate_M, candidate_N, candidate_K),
                )
    return None


def _tuned_rows_snapshot(configs, gfx: str, cu_num: int, device: int):
    """(M, N, K, kernelName) of the rows a lookup for gfx/cu_num on device may
    use, in the order of get_tuned_cu_nums(cu_num, device).

    A compiled caller takes the first row of the exact shape, then the first
    row of the padded shape, so it keeps the order of _find_mixed_mxfp_config.
    """
    return tuple(
        (M, N, K, str(config["kernelName"]))
        for tuned_cu in get_tuned_cu_nums(cu_num, device)
        for (config_gfx, config_cu, M, N, K), config in configs.items()
        if config_gfx == gfx and config_cu == tuned_cu
    )
