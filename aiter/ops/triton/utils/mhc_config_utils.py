# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import glob
import os
import re

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.gemm_config_utils import _load_config_file, USE_LRU_CACHE


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_mhc_config(
    config_name: str,
    M: int,
    C: int,
    mode: str | None = None,
) -> tuple[dict, bool]:
    """
    Load MHC configuration with threshold matching of M_LEQ_x keys, C, and mode.

    Selection logic:
    - C: Finds the largest C-specific config file threshold <= input C value.
      Available C configs are discovered from files named {arch}-{config}-C={value}.json.
    - M: Within the selected config, finds the largest M_LEQ_x threshold <= input M value.
    
    Architecture fallback:
    - If configs for the current GPU architecture don't exist, falls back to gfx942 configs.
    - This allows MHC operations to work on GPUs without tuned configs (may be suboptimal).

    Config file naming convention:
    - For MHC_FUSED: mode is required ("lite" or "sinkhorn")
      - e.g., gfx942-MHC_FUSED_LITE.json, gfx942-MHC_FUSED_SINKHORN-C=128.json
    - For other configs (e.g., MHC_SINKHORN): mode is not used
      - e.g., gfx942-MHC_SINKHORN.json

    Args:
        config_name: Base name of the config (e.g., "MHC_FUSED", "MHC_SINKHORN")
        M: M dimension (batch/sequence size)
        C: C dimension (hidden dim per stream). Uses threshold matching
            to find the largest available C config <= input C.
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
    fallback_dev = "gfx942"
    
    # Determine the actual config name based on mode
    if mode is not None:
        if mode not in ("lite", "sinkhorn"):
            raise ValueError(f"mode must be 'lite' or 'sinkhorn', got '{mode}'")
        actual_config_name = f"{config_name}_{mode.upper()}"
    else:
        # No mode suffix - for standalone configs like MHC_SINKHORN
        actual_config_name = config_name

    cache_key = f"{dev}_{actual_config_name}"

    # Load default config with fallback for unsupported architectures
    if cache_key not in get_mhc_config._config_cache:
        get_mhc_config._config_cache[cache_key] = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-{actual_config_name}.json"
        
        # Try loading architecture-specific config first
        if not _load_config_file(
            get_mhc_config._config_cache, cache_key, fpath, "default", fpath_should_exist=False
        ):
            # Fallback to gfx942 configs if architecture-specific config doesn't exist
            fpath_fallback = f"{AITER_TRITON_CONFIGS_PATH}/{fallback_dev}-{actual_config_name}.json"
            _load_config_file(
                get_mhc_config._config_cache, cache_key, fpath_fallback, "default", fpath_should_exist=True
            )

    config_dict_key = "default"
    used_specialized = False

    # Try C-specific config (threshold matching: largest C <= input C)
    c_thresholds_key = f"{cache_key}_c_thresholds"

    # Discover available C-specific config files once per cache_key
    if c_thresholds_key not in get_mhc_config._config_cache:
        c_thresholds = []
        
        # Check architecture-specific C configs
        pattern = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-{actual_config_name}-C=*.json"
        for fpath in glob.glob(pattern):
            basename = os.path.basename(fpath)
            match = re.search(r"-C=(\d+)\.json$", basename)
            if match:
                c_thresholds.append(int(match.group(1)))
        
        # Also check fallback architecture C configs
        if dev != fallback_dev:
            pattern_fallback = f"{AITER_TRITON_CONFIGS_PATH}/{fallback_dev}-{actual_config_name}-C=*.json"
            for fpath in glob.glob(pattern_fallback):
                basename = os.path.basename(fpath)
                match = re.search(r"-C=(\d+)\.json$", basename)
                if match:
                    c_val = int(match.group(1))
                    if c_val not in c_thresholds:
                        c_thresholds.append(c_val)
        
        c_thresholds.sort()
        get_mhc_config._config_cache[c_thresholds_key] = c_thresholds

    # Find largest C threshold <= input C
    for c_threshold in reversed(get_mhc_config._config_cache[c_thresholds_key]):
        if C >= c_threshold:
            c_key = f"C_{c_threshold}"
            if c_key not in get_mhc_config._config_cache[cache_key]:
                fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-{actual_config_name}-C={c_threshold}.json"
                # Try architecture-specific C config first, fallback to gfx942 if needed
                if not _load_config_file(
                    get_mhc_config._config_cache, cache_key, fpath, c_key, fpath_should_exist=False
                ):
                    fpath_fallback = f"{AITER_TRITON_CONFIGS_PATH}/{fallback_dev}-{actual_config_name}-C={c_threshold}.json"
                    _load_config_file(
                        get_mhc_config._config_cache, cache_key, fpath_fallback, c_key, fpath_should_exist=False
                    )
            if c_key in get_mhc_config._config_cache[cache_key]:
                config_dict_key = c_key
                used_specialized = True
                break

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

    # Find largest threshold <= M
    matched_key = None
    for threshold, key in m_leq_keys:
        if M >= threshold:
            matched_key = key
        else:
            break

    if matched_key is not None:
        return dict(config_dict[matched_key]), used_specialized

    # Fallback to "any" if no matching key found
    if "any" in config_dict:
        return dict(config_dict["any"]), used_specialized

    raise KeyError(f"No matching config for M={M}, C={C}, mode={mode} in '{config_name}'")
