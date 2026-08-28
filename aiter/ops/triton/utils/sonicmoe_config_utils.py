# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load SonicMoE nested DEFAULT.json and pick a tile bucket at launch.

Path: ``configs/<arch>/triton/moe/sonicmoe_bf16/DEFAULT.json``. This is
intentionally not fused_moe's ``get_moe_configs()`` path — that resolver
expects small_M/medium_M/large_M, not SonicMoE kernel sections.
"""

from __future__ import annotations

from typing import Any

from .core import load_config_json
from .gemm_config_utils import resolve_config_dir

_LAUNCH_META = frozenset({"num_warps", "num_stages"})


def load_sonicmoe_configs() -> dict[str, Any] | None:
    cfg_dir, name_prefix = resolve_config_dir("moe", "SONICMOE-BF16", backend="triton")
    stem = f"{name_prefix}SONICMOE-BF16" if name_prefix else "DEFAULT"
    return load_config_json(f"{cfg_dir}/{stem}.json", required=False)


def _clean_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in bucket.items() if not k.startswith("_")}


def split_launch_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split constexpr kwargs from ``num_warps`` / ``num_stages`` launch args."""
    constexprs = {}
    launch = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        if k in _LAUNCH_META:
            launch[k] = v
        else:
            constexprs[k] = v
    return constexprs, launch


def _pick_nk_bucket(
    section: dict[str, Any],
    N: int,
    K: int,
    E: int,
    small: tuple[int, int, int],
    medium: tuple[int, int, int],
) -> dict[str, Any] | None:
    n_s, k_s, e_s = small
    n_m, k_m, e_m = medium
    if N <= n_s and K <= k_s and E <= e_s:
        name = "small_NK"
    elif N <= n_m and K <= k_m and E <= e_m:
        name = "medium_NK"
    else:
        name = "large_NK"
    bucket = section.get(name)
    return _clean_bucket(bucket) if bucket else None


def get_grouped_gemm_fwd_config(N: int, K: int, E: int) -> dict[str, Any] | None:
    configs = load_sonicmoe_configs()
    if not configs or "_grouped_gemm_kernel" not in configs:
        return None
    return _pick_nk_bucket(
        configs["_grouped_gemm_kernel"],
        N,
        K,
        E,
        small=(256, 256, 8),
        medium=(2048, 2048, 32),
    )


def get_grouped_gemm_dw_config(N: int, K: int, E: int) -> dict[str, Any] | None:
    configs = load_sonicmoe_configs()
    if not configs or "_grouped_gemm_dw_kernel" not in configs:
        return None
    return _pick_nk_bucket(
        configs["_grouped_gemm_dw_kernel"],
        N,
        K,
        E,
        small=(128, 128, 4),
        medium=(2048, 2048, 32),
    )


def get_token_gather_config(H: int) -> dict[str, Any] | None:
    configs = load_sonicmoe_configs()
    if not configs or "token_gather_sum_kernel" not in configs:
        return None
    section = configs["token_gather_sum_kernel"]
    if H <= 256:
        name = "small_H"
    elif H <= 1024:
        name = "medium_H"
    else:
        name = "large_H"
    bucket = section.get(name)
    return _clean_bucket(bucket) if bucket else None
