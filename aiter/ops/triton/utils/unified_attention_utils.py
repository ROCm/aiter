# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import functools
import itertools

import torch

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)
from aiter.ops.triton.utils.types import e4m3_dtype

_DEFAULT_BS_BOUNDS = (16, 32, 64, 128, 256)


def get_dtype_str(dtype: torch.dtype):
    if dtype == torch.uint8:
        return "nvfp4"
    if dtype == e4m3_dtype:
        return "fp8"
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return "bf16"
    raise ValueError(f"No unified attention config tag for dtype: {dtype}")


def _dtype_keys(q_tag: str, kv_tag: str):
    return (f"{q_tag}_{kv_tag}", f"{q_tag}_any", f"any_{kv_tag}", "any")


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_unified_attention_config_cached(
    op: str,  # "attn_2d" | "attn_3d" | "reduce" | "kv_split"
    key: str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
    num_queries_per_kv: int,
    block_size: int,
    bounds: tuple[int, ...] | None = None,
    backend: str = "triton",  # "gluon" | "triton"
) -> tuple[dict, bool]:
    """
    Internal cached implementation. Do NOT use this directly. Use
    `get_unified_attention_config()` instead, which returns a deep-copy
    so callers can freely mutate the returned dict without polluting
    the cache.

    Resolves from `<arch>/<backend>/attention/unified_attention/` (
    default named `DEFAULT.json`).
    """
    # input validation
    assert head_size > 0, "head_size must be positive"
    assert num_queries_per_kv > 0, "num_queries_per_kv must be positive"
    assert block_size > 0, "block_size must be positive"
    assert bounds is None or (
        len(bounds) > 0
        and all(x > 0 for x in bounds)
        and all(x < y for x, y in itertools.pairwise(bounds))
    ), "When provided, bounds must be a non-empty tuple of strictly increasing positive numbers"

    config_name = "UNIFIED-ATTENTION"
    q_tag, kv_tag = get_dtype_str(q_dtype), get_dtype_str(kv_dtype)

    # load default config
    cfg_dir = resolve_config_dir("attention", config_name, backend=backend)
    default_stem = "DEFAULT"
    default_fpath = f"{cfg_dir}/{default_stem}.json"
    config_dict = load_config_json(default_fpath, required=False)
    if config_dict is None:
        raise AssertionError(f"Required config file doesn't exist: {default_fpath}")

    is_tuned = False
    for suffix in (
        f"D={head_size}-QPKV={num_queries_per_kv}",
        f"D={head_size}",
    ):
        specialized = load_config_json(
            f"{cfg_dir}/{config_name}-{suffix}.json", required=False
        )
        if specialized is not None:
            config_dict, is_tuned = specialized, True
            break

    where = f"{config_name}[{op}.{key}] in {cfg_dir}"

    # get op configs
    if op not in config_dict:
        raise KeyError(f"{where}: file has no {op}")
    op_configs = config_dict[op]

    # get specific config entry
    entry = op_configs.get(key)
    if entry is None:
        if "any" not in op_configs:
            raise KeyError(f"{where}: key {key} and 'any' not found")
        entry = op_configs["any"]

    # q/kv dtype
    dtype_keys = _dtype_keys(q_tag, kv_tag)
    dtype_configs = None
    for dtype_key in dtype_keys:
        if dtype_key in entry:
            dtype_configs = entry[dtype_key]
            break
    if dtype_configs is None:
        raise KeyError(f"{where}: no dtype entry found for q={q_tag} kv={kv_tag}")

    # bounds
    search_bounds = bounds if bounds is not None else _DEFAULT_BS_BOUNDS

    # search for BS_LEQ_x keys
    for bound in search_bounds:
        candidate = f"BS_LEQ_{bound}"
        if block_size <= bound and candidate in dtype_configs:
            return dict(dtype_configs[candidate]), is_tuned

    # search for BS_GEQ_x keys
    for bound in reversed(search_bounds):
        candidate = f"BS_GEQ_{bound}"
        if block_size >= bound and candidate in dtype_configs:
            return dict(dtype_configs[candidate]), is_tuned

    if "any" in dtype_configs:
        return dict(dtype_configs["any"]), is_tuned

    raise KeyError(
        f"{where}: no matching configuration found for "
        f"head_size={head_size} block_size={block_size} num_queries_per_kv={num_queries_per_kv}"
    )


def get_unified_attention_config(
    op: str,  # "attn_2d" | "attn_3d" | "reduce" | "kv_split"
    key: str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
    num_queries_per_kv: int,
    block_size: int,
    bounds: tuple[int, ...] | None = None,
    backend: str = "triton",  # "gluon" | "triton"
) -> tuple[dict, bool]:
    """
    Load a unified attention configuration using the BS_LEQ_x/BS_GEQ_x/any format.

    This function provides a unified way to load unified attention configs across all kernels.
    It uses the following logic:
    1. Load default config file: <arch>/<backend>/attention/unified_attention/DEFAULT.json
    2. Try the specialized config, which overrides the default:
       UNIFIED-ATTENTION-D={head_size}-QPKV={num_queries_per_kv}.json
    3. If it is absent, try UNIFIED-ATTENTION-D={head_size}.json; the first
       specialized file that exists wins and marks the config as tuned
    4. Look up the op section (attn_2d|attn_3d|reduce|kv_split) in the resulting file
    5. Look up key inside that op, falling back to "any"
    6. Look up the q/kv dtype entry, trying "{q}_{kv}", "{q}_any", "any_{kv}",
       then "any", with the tags (nvfp4|fp8|bf16) coming from get_dtype_str()
    7. Search for BS_LEQ_x keys in order of bounds (default: _DEFAULT_BS_BOUNDS)
    8. If no BS_LEQ_x matches, search for BS_GEQ_x keys in reverse order
    9. Fall back to "any" if no bounds match

    BS_LEQ_x only covers block_size up to the largest bound it names and BS_GEQ_x
    only from the smallest one, so a table bucketed 16/32/(64+) has nothing for a
    48-token page. Every bucketed entry therefore carries an "any" holding the
    next bucket up, and that is what makes the lookup total -- drop it and pages
    between buckets raise KeyError again.

    Args:
        op (str): Unified attention operation (attn_2d|attn_3d|reduce|kv_split)
        bounds (tuple[int, ...] | None, optional): Custom bounds to use instead of
            _DEFAULT_BS_BOUNDS. Defaults to None.

    Returns:
        tuple[dict, bool]:
            Dictionary with the config params (a fresh deep-copy safe to mutate),
            bool indicating if the config is tuned.
    """
    config, is_tuned = _get_unified_attention_config_cached(
        op,
        key,
        q_dtype,
        kv_dtype,
        head_size,
        num_queries_per_kv,
        block_size,
        bounds=bounds,
        backend=backend,
    )
    return copy.deepcopy(config), is_tuned
