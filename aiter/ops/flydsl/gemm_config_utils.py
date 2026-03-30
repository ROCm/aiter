# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GEMM configuration loading utilities for FlyDSL kernels.

Mirrors the aiter triton config system: JSON files with M_LEQ_x / M_GEQ_x / "any"
keys, optional specialized N=K files, and LRU-cached lookups.
"""

import copy
import functools
import json
import os

from flydsl.runtime.device import get_rocm_arch

FLYDSL_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "configs")

STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _load_json(fpath: str) -> dict | None:
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            return json.load(f)
    return None


@functools.lru_cache(maxsize=256)
def _get_gemm_config_cached(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
) -> tuple[dict, bool]:
    """Internal cached lookup. Use get_gemm_config() for a safe deep-copy."""
    arch = str(get_rocm_arch(timeout_s=300))

    # Load default config (required)
    default_path = os.path.join(FLYDSL_CONFIGS_PATH, f"{arch}-{config_name}.json")
    default_cfg = _load_json(default_path)
    if default_cfg is None:
        raise FileNotFoundError(f"Default config not found: {default_path}")

    # Try specialized N=K config
    config_dict = default_cfg
    is_tuned = False
    if N is not None and K is not None:
        spec_path = os.path.join(
            FLYDSL_CONFIGS_PATH, f"{arch}-{config_name}-N={N}-K={K}.json"
        )
        spec_cfg = _load_json(spec_path)
        if spec_cfg is not None:
            config_dict = spec_cfg
            is_tuned = True

    search_bounds = bounds if bounds is not None else STANDARD_M_BOUNDS

    # Search M_LEQ_x in ascending order
    for bound in search_bounds:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    # Search M_GEQ_x in descending order
    for bound in reversed(search_bounds):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    if "any" in config_dict:
        return dict(config_dict["any"]), is_tuned

    raise KeyError(
        f"No matching config for M={M}, N={N}, K={K} in '{config_name}'"
    )


def get_gemm_config(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
) -> tuple[dict, bool]:
    """Load a GEMM configuration by M/N/K dimensions.

    Lookup order:
      1. Load default: {arch}-{config_name}.json
      2. If N, K given, try specialized: {arch}-{config_name}-N={N}-K={K}.json
      3. Match M_LEQ_x keys (ascending), then M_GEQ_x (descending), then "any"

    Args:
        config_name: Config family name, e.g. "GEMM-A8W8".
        M: M dimension.
        N: N dimension (optional, for specialized lookup).
        K: K dimension (optional, for specialized lookup).
        bounds: Custom M bounds tuple (default: STANDARD_M_BOUNDS).

    Returns:
        (config_dict, is_tuned): A fresh dict of kernel params, and whether
        a specialized N=K config was used.
    """
    config, is_tuned = _get_gemm_config_cached(config_name, M, N, K, bounds)
    return copy.deepcopy(config), is_tuned
