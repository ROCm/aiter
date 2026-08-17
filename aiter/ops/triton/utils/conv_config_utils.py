# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import (
    AITER_TRITON_CONFIGS_PATH,
    USE_LRU_CACHE,
    load_config_json,
)

STANDARD_M_BOUNDS: tuple[int, ...] = (
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


def format_prepack_shape_key(N: int, C: int, H: int, W: int, CB: int) -> str:
    """Canonical key for an NCHW-to-NCHWc activation pack."""
    return f"N={N},C={C},H={H},W={W},CB={CB}"


@functools.lru_cache(maxsize=512 if USE_LRU_CACHE else 0)
def _get_conv_config_cached(
    config_name: str,
    shape_key: str | None,
    M: int | None,
    variants: tuple[str, ...],
) -> dict:
    """Config walk: variant shape entries, generic shape, M bucket, any."""
    dev = arch_info.get_arch()
    config_dict = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/conv/{dev}-{config_name}.json"
    )

    # Tier 1: optional layout/dtype-specific exact-shape pins.
    if shape_key is not None:
        for variant in variants:
            shapes = config_dict.get(f"shapes_{variant}", {})
            if shape_key in shapes:
                return shapes[shape_key]

    # Tier 2: generic exact-shape pin.
    shapes = config_dict.get("shapes", {})
    if shape_key is not None and shape_key in shapes:
        return shapes[shape_key]

    # Tier 3: M-bucket walk.
    if M is not None and M >= 0:
        for bound in STANDARD_M_BOUNDS:
            key = f"M_LEQ_{bound}"
            if M <= bound and key in config_dict:
                return config_dict[key]

    # Tier 4: any fallback.
    if "any" in config_dict:
        return config_dict["any"]

    raise KeyError(
        f"No matching config in '{config_name}' for shape_key={shape_key!r}, "
        f"M={M} on arch {dev} (no literal shape, no bucket, no 'any' fallback)."
    )


@functools.lru_cache(maxsize=64 if USE_LRU_CACHE else 0)
def has_conv_config(config_name: str) -> bool:
    """Return whether the running architecture ships this optional table."""
    dev = arch_info.get_arch()
    config = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/conv/{dev}-{config_name}.json",
        required=False,
    )
    return config is not None


def get_conv_config(
    config_name: str,
    shape_key: str | None = None,
    M: int | None = None,
    variants: tuple[str, ...] = (),
) -> dict:
    """Load a conv kernel config for the running GPU arch.

    Walk order (first hit wins):
        1. ``shapes_<variant>[shape_key]`` — optional layout/dtype pin.
        2. ``shapes[shape_key]`` — generic exact-shape pin.
        3. ``M_LEQ_<n>`` — row-count bucket walk (M_total for GEMM-like
           kernels, T for Winograd).
        4. ``"any"`` — global fallback.

    Returns a fresh shallow copy of the config dict; safe to mutate. Conv
    entries are flat mappings of scalar tuning values, so a deep copy only
    adds hot-path overhead.

    Modeled on :func:`get_gemm_config` but with conv-native (shape-key first)
    dispatch and no splitk / N=K= specialization.
    """
    config = _get_conv_config_cached(config_name, shape_key, M, variants)
    return dict(config)
