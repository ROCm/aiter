# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import functools

import torch

from aiter.ops.triton.utils.core import USE_LRU_CACHE, load_config_json
from aiter.ops.triton.utils.gemm_config_utils import resolve_config_dir
from aiter.ops.triton.utils.types import e4m3_dtype

_DEFAULT_PREFIX_BOUNDS = {
    "SQ": (1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192),  # max_seqlen_q
    "SK": (
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
    ),  # max_seqlen_k
}


def get_dtype_str(dtype: torch.dtype):
    if dtype == torch.uint8:
        return "nvfp4"
    if dtype == e4m3_dtype:
        return "fp8"
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return "bf16"
    raise ValueError(f"No unified attention config tag for dtype: {dtype}")


def _dtype_keys(q_tag: str, kv_tag: str) -> str:
    return (f"{q_tag}_{kv_tag}", f"{q_tag}_any", f"any_{kv_tag}", "any")


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_unified_attention_config_cached(
    op: str,  # "attn_2d" | "attn_3d" | "reduce"
    key: str,
    bucket_prefix: str,  # "SQ" | "SK"
    bucket_value: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
    block_size: int,
    num_queries_per_kv: int,
    backend: str | None = None,
) -> tuple[dict, bool]:
    config_name = "UNIFIED-ATTENTION"
    q_tag, kv_tag = get_dtype_str(q_dtype), get_dtype_str(kv_dtype)

    # load default config
    cfg_dir, _ = resolve_config_dir("attention", config_name, backend=backend)
    default_stem = "DEFAULT"
    default_fpath = f"{cfg_dir}/{default_stem}.json"
    config_dict = load_config_json(default_fpath, required=False)
    if config_dict is None:
        raise AssertionError(f"Required config file doesn't exist: {default_fpath}")

    is_tuned = False
    for suffix in (
        f"D={head_size}-QPKV={num_queries_per_kv}-PAGE={block_size}",
        f"D={head_size}-PAGE={block_size}",
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
    bounds = _DEFAULT_PREFIX_BOUNDS[bucket_prefix]

    # search for {bucket_prefix}_LEQ_x keys
    for bound in bounds:
        candidate = f"{bucket_prefix}_LEQ_{bound}"
        if bucket_value <= bound and candidate in dtype_configs:
            return dict(dtype_configs[candidate]), is_tuned

    # search for {bucket_prefix}_GEQ_x keys
    for bound in bounds:
        candidate = f"{bucket_prefix}_GEQ_{bound}"
        if bucket_value >= bound and candidate in dtype_configs:
            return dict(dtype_configs[candidate]), is_tuned

    if "any" in dtype_configs:
        return dict(dtype_configs[candidate]), False

    raise KeyError(
        f"{where}: no matching configuration found for "
        f"head_size={head_size} block_size={block_size} num_queries_per_kv={num_queries_per_kv}"
    )


def get_unified_attention_config(
    op: str,  # "attn_2d" | "attn_3d" | "reduce"
    key: str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_size: int,
    block_size: int,
    num_queries_per_kv: int,
    backend: str | None = None,
) -> tuple[dict, bool]:
    config, is_tuned = _get_unified_attention_config_cached(
        op, key, q_dtype, kv_dtype, head_size, block_size, num_queries_per_kv, backend
    )
    return copy.deepcopy(config), is_tuned
