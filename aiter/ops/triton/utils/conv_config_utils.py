# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import functools
import json
import os
from typing import Optional, Tuple

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

USE_LRU_CACHE = True

STANDARD_M_BOUNDS: Tuple[int, ...] = (
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
)


def format_shape_key(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    R: int,
    S: int,
    sh: int,
    sw: int,
    ph: int,
    pw: int,
    dh: int,
    dw: int,
) -> str:
    """Canonical string key for a user-visible conv2d call. Same format used by
    the loader and the kernel-side _get_config helpers.
    """
    return (
        f"N={N},C={C},H={H},W={W},K={K},R={R},S={S},"
        f"sh={sh},sw={sw},ph={ph},pw={pw},dh={dh},dw={dw}"
    )


def _load_config_file(
    cache_dict: dict,
    cache_key: str,
    fpath: str,
    fpath_should_exist: bool = True,
) -> bool:
    if os.path.exists(fpath):
        with open(fpath, "r") as file:
            cache_dict[cache_key] = json.load(file)
        return True
    elif fpath_should_exist:
        raise AssertionError(f"Required config file doesn't exist: {fpath}")
    return False


@functools.lru_cache(maxsize=512 if USE_LRU_CACHE else 0)
def _get_conv_config_cached(
    config_name: str,
    shape_key: Optional[str],
    M: Optional[int],
) -> dict:
    """Three-tier walk: literal shape entry -> M_LEQ bucket -> 'any'."""
    if not hasattr(_get_conv_config_cached, "_file_cache"):
        _get_conv_config_cached._file_cache = {}

    dev = arch_info.get_arch()
    file_cache_key = f"{dev}_{config_name}"

    if file_cache_key not in _get_conv_config_cached._file_cache:
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/conv/{dev}-{config_name}.json"
        _load_config_file(
            _get_conv_config_cached._file_cache,
            file_cache_key,
            fpath,
            fpath_should_exist=True,
        )

    config_dict = _get_conv_config_cached._file_cache[file_cache_key]

    # Tier 1: literal shape key.
    shapes = config_dict.get("shapes", {})
    if shape_key is not None and shape_key in shapes:
        return shapes[shape_key]

    # Tier 2: M-bucket walk.
    if M is not None and M >= 0:
        for bound in STANDARD_M_BOUNDS:
            key = f"M_LEQ_{bound}"
            if M <= bound and key in config_dict:
                return config_dict[key]

    # Tier 3: any fallback.
    if "any" in config_dict:
        return config_dict["any"]

    raise KeyError(
        f"No matching config in '{config_name}' for shape_key={shape_key!r}, "
        f"M={M} on arch {dev} (no literal shape, no bucket, no 'any' fallback)."
    )


def get_conv_config(
    config_name: str,
    shape_key: Optional[str] = None,
    M: Optional[int] = None,
) -> dict:
    """Load a conv kernel config for the running GPU arch.

    Walk order (first hit wins):
        1. ``shapes[shape_key]`` — exact-shape pin from the offline sweep.
        2. ``M_LEQ_<n>`` — row-count bucket walk (M_total for GEMM-like
           kernels, T for Winograd).
        3. ``"any"`` — global fallback.

    Returns a fresh deep-copy of the config dict; safe to mutate.

    Modeled on :func:`get_gemm_config` but with conv-native (shape-key first)
    dispatch and no splitk / N=K= specialization.
    """
    config = _get_conv_config_cached(config_name, shape_key, M)
    return copy.deepcopy(config)
