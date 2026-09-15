# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHC config loading: ``get_mhc_config()`` / ``get_mhc_post_config()``,
with the documented gfx942 arch fallback, on top of the shared core in
``config_utils``.
"""

import functools
import glob
import os
import re

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

logger = AiterTritonLogger()

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)

_FALLBACK_DEV = "gfx942"


def _mhc_config_dir(dev: str, config_name: str) -> str:
    """``dev``'s nested config directory for the ``config_name`` family."""
    return resolve_config_dir("mhc", config_name, backend="triton", arch=dev)


def _load_with_fallback(
    dev: str, config_name: str, fname: str, required: bool = False
) -> dict | None:
    """Load ``fname`` from ``dev``'s ``config_name`` directory, falling back to
    the gfx942 copy for arches without tuned MHC configs (may be suboptimal)."""
    config = load_config_json(
        f"{_mhc_config_dir(dev, config_name)}/{fname}", required=False
    )
    if config is None:
        config = load_config_json(
            f"{_mhc_config_dir(_FALLBACK_DEV, config_name)}/{fname}", required=required
        )
    return config


@functools.lru_cache(maxsize=None if USE_LRU_CACHE else 0)
def _c_thresholds(dev: str, actual_config_name: str) -> tuple[int, ...]:
    """C values that have a specialized config file (arch-specific plus the
    gfx942 fallback), sorted ascending."""
    thresholds = set()
    for d in {dev, _FALLBACK_DEV}:
        cfg_dir = _mhc_config_dir(d, actual_config_name)
        pattern = f"{cfg_dir}/{actual_config_name}-C=*.json"
        for fpath in glob.glob(pattern):
            match = re.search(r"-C=(\d+)\.json$", os.path.basename(fpath))
            if match:
                thresholds.add(int(match.group(1)))
    return tuple(sorted(thresholds))


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
      Available C configs are discovered from the files named
      {config}-C={value}.json in the arch's config directory.
    - M: Within the selected config, finds the largest M_LEQ_x threshold <= input M value.

    Architecture fallback:
    - If configs for the current GPU architecture don't exist, falls back to gfx942 configs.
    - This allows MHC operations to work on GPUs without tuned configs (may be suboptimal).

    Config file naming convention:
    - For MHC_FUSED: mode is required ("sinkhorn")
      - e.g., gfx942/triton/mhc/mhc_fused_sinkhorn/MHC_FUSED_SINKHORN-C=128.json

    Args:
        config_name: Base name of the config (e.g., "MHC_FUSED")
        M: M dimension (batch/sequence size)
        C: C dimension (hidden dim per stream). Uses threshold matching
            to find the largest available C config <= input C.
        mode: H_res mode for MHC_FUSED - "sinkhorn" (required for MHC_FUSED)

    Returns:
        Tuple of (config dict, bool indicating if C-specialized config was used)

    Raises:
        ValueError: If mode is invalid or missing when required
        KeyError: If no matching config found
    """
    dev = arch_info.get_arch()

    if mode is None or mode != "sinkhorn":
        raise ValueError(f"mode must be 'sinkhorn', got '{mode}'")
    actual_config_name = f"{config_name}_{mode.upper()}"

    # Default config (must exist for the arch or the gfx942 fallback)
    config_dict = _load_with_fallback(
        dev, actual_config_name, "DEFAULT.json", required=True
    )
    used_specialized = False

    # C-specific config: largest discovered threshold <= input C wins
    for c_threshold in reversed(_c_thresholds(dev, actual_config_name)):
        if C >= c_threshold:
            specialized = _load_with_fallback(
                dev, actual_config_name, f"{actual_config_name}-C={c_threshold}.json"
            )
            if specialized is not None:
                config_dict = specialized
                used_specialized = True
                break

    # Extract M_LEQ_x keys and their thresholds, sorted ascending
    m_leq_keys = []
    for key in config_dict:
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

    raise KeyError(
        f"No matching config for M={M}, C={C}, mode={mode} in '{config_name}'"
    )


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_mhc_post_config(M: int, C: int) -> dict:
    """Pick the mhc_post config for ``(M, C)`` from the arch's ``mhc_post``
    ``DEFAULT.json``.

    Picks the largest ``C_<value> <= C``, else ``"default"``.
    """
    dev = arch_info.get_arch()
    cfg = load_config_json(f"{_mhc_config_dir(dev, 'MHC_POST')}/DEFAULT.json")

    c_thresholds = sorted(
        int(k[2:]) for k in cfg if k.startswith("C_") and k[2:].isdigit()
    )
    for c_threshold in reversed(c_thresholds):
        if C >= c_threshold:
            return dict(cfg[f"C_{c_threshold}"])

    if "default" in cfg:
        return dict(cfg["default"])

    raise KeyError(f"No matching config for M={M}, C={C} in 'MHC_POST'")


def hip_post_dispatch_block(C: int, arch_id: str) -> int | None:
    """Return the ``residual_block`` ``aiter.mhc_post`` selects for this C.

    Mirrors ``MHC_POST_KERNEL_DISPATCH`` in
    ``csrc/kernels/mhc_kernels.cu``:

        non-gfx942 + C % 1024 == 0 -> 1024
        C % 512 == 0               -> 512
        C % 256 == 0               -> 256
        else                       -> None  (unsupported, caller should skip)

    The HIP kernel additionally enforces ``C >= 2 * residual_block`` via
    ``TORCH_CHECK``, so callers should reject shapes where
    ``C < 2 * hip_post_dispatch_block(C, arch_id)``.
    """
    if arch_id != "gfx942" and C % 1024 == 0:
        return 1024
    if C % 512 == 0:
        return 512
    if C % 256 == 0:
        return 256
    return None
