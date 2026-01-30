# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.gemm_config_utils import _load_config_file, USE_LRU_CACHE


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_mhc_config(
    config_name: str,
    M: int,
    C: int | None = None,
    mode: str | None = None,
) -> tuple[dict, bool]:
    """
    Load MHC configuration with matching of M_LEQ_x keys, C, and mode.

    Selection finds the smallest threshold >= input M value.

    Config file naming convention:
    - For MHC_FUSED: mode is required ("lite" or "sinkhorn")
      - e.g., gfx942-MHC_FUSED_LITE.json, gfx942-MHC_FUSED_SINKHORN-C=128.json
    - For other configs (e.g., MHC_SINKHORN): mode is not used
      - e.g., gfx942-MHC_SINKHORN.json

    Args:
        config_name: Base name of the config (e.g., "MHC_FUSED", "MHC_SINKHORN")
        M: M dimension (batch/sequence size)
        C: C dimension (hidden dim per stream, optional for specialized configs)
        mode: H_res mode for MHC_FUSED - "lite" or "sinkhorn" (required for MHC_FUSED)

    Returns:
        Tuple of (config dict, bool indicating if C-specialized config was used)

    Raises:
        ValueError: If mode is invalid or missing when required
        KeyError: If no matching config found
    """
    if not hasattr(get_mhc_config, "_config_cache"):
        get_mhc_config._config_cache = {}

    dev = arch_info.get_arch()
    
    # Determine the actual config name based on mode
    if mode is not None:
        if mode not in ("lite", "sinkhorn"):
            raise ValueError(f"mode must be 'lite' or 'sinkhorn', got '{mode}'")
        actual_config_name = f"{config_name}_{mode.upper()}"
    else:
        # No mode suffix - for standalone configs like MHC_SINKHORN
        actual_config_name = config_name

    cache_key = f"{dev}_{actual_config_name}"

    # Load default config (required)
    if cache_key not in get_mhc_config._config_cache:
        get_mhc_config._config_cache[cache_key] = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-{actual_config_name}.json"
        _load_config_file(
            get_mhc_config._config_cache, cache_key, fpath, "default", fpath_should_exist=True
        )

    config_dict_key = "default"
    used_specialized = False

    # Try C-specific config if C is provided
    if C is not None:
        c_key = f"C_{C}"
        if c_key not in get_mhc_config._config_cache[cache_key]:
            fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-{actual_config_name}-C={C}.json"
            if _load_config_file(
                get_mhc_config._config_cache, cache_key, fpath, c_key
            ):
                config_dict_key = c_key
                used_specialized = True
        elif c_key in get_mhc_config._config_cache[cache_key]:
            config_dict_key = c_key
            used_specialized = True

    config_dict = get_mhc_config._config_cache[cache_key][config_dict_key]

    # Extract M_LEQ_x keys and their thresholds, sorted ascending
    m_leq_keys = []
    for key in config_dict.keys():
        if key.startswith("M_LEQ_"):
            try:
                threshold = int(key[6:])  # Extract number after "M_LEQ_"
                m_leq_keys.append((threshold, key))
            except ValueError:
                continue
    m_leq_keys.sort()  # Sort by threshold value

    # Find smallest threshold >= M (up to or equal to matching)
    for threshold, key in m_leq_keys:
        if M <= threshold:
            return dict(config_dict[key]), used_specialized

    # Fallback to "any" if no matching key found
    if "any" in config_dict:
        return dict(config_dict["any"]), used_specialized

    raise KeyError(f"No matching config for M={M}, C={C}, mode={mode} in '{config_name}'")
