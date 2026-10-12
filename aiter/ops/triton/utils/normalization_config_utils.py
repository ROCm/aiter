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

    Raises ``FileNotFoundError`` when no config file is shipped for this arch,
    naming the missing path. Seed a new arch directory from the nearest
    measured arch before adding a new target.
    """
    cfg_dir = resolve_config_dir("normalization", config_name)
    return load_config_json(f"{cfg_dir}/DEFAULT.json", required=True)
