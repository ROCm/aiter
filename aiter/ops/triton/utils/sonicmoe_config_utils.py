# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load SonicMoE kernel configs from the nested tuning tree."""

from __future__ import annotations

from typing import Any

from .config_utils import load_config_json, resolve_config_dir

_LAUNCH_META = frozenset({"num_warps", "num_stages"})


def load_sonicmoe_configs() -> dict[str, Any]:
    cfg_dir = resolve_config_dir("moe", "SONICMOE-BF16", backend="triton")
    config = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    if config is None:
        fallback_dir = resolve_config_dir(
            "moe", "SONICMOE-BF16", backend="triton", arch="gfx942"
        )
        config = load_config_json(f"{fallback_dir}/DEFAULT.json")
    return config


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
) -> dict[str, Any] | None:
    small = section["_bounds"]["small"]
    medium = section["_bounds"]["medium"]
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


def _specialized_key(N: int, K: int, E: int, has_gather: bool) -> str:
    return f"N={N},K={K},E={E},gather={int(has_gather)}"


def get_grouped_gemm_fwd_config(
    N: int,
    K: int,
    E: int,
    has_gather: bool = False,
    use_specialized: bool = False,
) -> dict[str, Any]:
    configs = load_sonicmoe_configs()
    section = configs["_grouped_gemm_kernel"]
    specialized = (
        section.get("_specialized", {}).get(_specialized_key(N, K, E, has_gather))
        if use_specialized
        else None
    )
    return (
        _clean_bucket(specialized) if specialized else _pick_nk_bucket(section, N, K, E)
    )


def get_grouped_gemm_dw_config(
    N: int,
    K: int,
    E: int,
    has_gather: bool = False,
    use_specialized: bool = False,
) -> dict[str, Any]:
    configs = load_sonicmoe_configs()
    section = configs["_grouped_gemm_dw_kernel"]
    specialized = (
        section.get("_specialized", {}).get(_specialized_key(N, K, E, has_gather))
        if use_specialized
        else None
    )
    return (
        _clean_bucket(specialized) if specialized else _pick_nk_bucket(section, N, K, E)
    )


def get_token_gather_config(H: int) -> dict[str, Any]:
    configs = load_sonicmoe_configs()
    section = configs["token_gather_sum_kernel"]
    small_h, medium_h = section["_bounds"]
    if H <= small_h:
        name = "small_H"
    elif H <= medium_h:
        name = "medium_H"
    else:
        name = "large_H"
    bucket = section.get(name)
    return _clean_bucket(bucket)


def get_sonicmoe_kernel_config(name: str) -> dict[str, Any]:
    return _clean_bucket(load_sonicmoe_configs()[name])
