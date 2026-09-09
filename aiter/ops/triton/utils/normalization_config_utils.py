# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Normalization kernel config loading: ``get_normalization_config()``."""

import functools

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)


@functools.lru_cache(maxsize=32 if USE_LRU_CACHE else 0)
def get_normalization_config(config_name: str, arch: str) -> dict:
    """Per-arch launch config for a normalization kernel family.

    Returns ``{}`` when no tuned file is shipped for this arch; callers
    fall back to their safe defaults.
    """
    cfg_dir = resolve_config_dir("normalization", config_name)
    config = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    return config if config is not None else {}
