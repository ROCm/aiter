# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Any, Dict, Optional
import os
import json
import functools
from ._triton import arch_info
from .core import AITER_TRITON_CONFIGS_PATH
import warnings
import triton

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_mxfp4: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "FP8_W8A8"
    elif use_int8_w8a16:
        return "INT8_W8A16"
    elif use_int8_w8a8:
        return "INT8_W8A8"
    elif use_int4_w4a16:
        return "INT4_W4A16"
    elif use_mxfp4:
        return "MX_FP4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


@functools.lru_cache
def get_moe_configs(dtype: Optional[str]) -> Optional[Dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """
    # First look up if an optimized configuration is available in the configs
    # directory
    dtype_str = "DEFAULT" if dtype is None else dtype
    dev = arch_info.get_device()
    config_file_path = f"{AITER_TRITON_CONFIGS_PATH}/moe/{dev}-MOE-{dtype_str}.json"

    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # If a configuration has been found, return it
            return {key: val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    warnings.warn(
        f"No MoE configuration found for device '{dev}' with dtype '{dtype_str}'. Using default configuration."
    )
    return None


def get_optimal_moe_config(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_mxfp4: Optional[bool] = False,
    M: int = 1,
):
    dtype_str = get_config_dtype_str(
        dtype, use_int8_w8a16, use_int8_w8a8, use_fp8_w8a8, use_int4_w4a16, use_mxfp4
    )
    # print(f"dtype_str={dtype_str}")
    configs = get_moe_configs(dtype_str)
    if configs is not None:
        if configs:
            if M < M_THRESHOLD_SMALL:
                config = configs["small_M"]
            elif M < M_THRESHOLD_MEDIUM:
                config = configs["medium_M"]
            else:
                config = configs["large_M"]
    else:
        # default config
        config = {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }

    # print(f"config={config}")
    return config


def get_optimal_moe_config_func(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    use_mxfp4: Optional[bool] = False,
):
    return functools.partial(
        get_optimal_moe_config,
        dtype,
        use_int8_w8a16,
        use_int8_w8a8,
        use_fp8_w8a8,
        use_int4_w4a16,
        use_mxfp4,
    )


@functools.lru_cache
def get_e2e_moe_configs(
    N: Optional[int], dtype: Optional[str]
) -> Optional[Dict[int, Any]]:
    # First look up if an optimized configuration is available in the configs
    # directory for the specific N and dtype
    dtype_str = "DEFAULT" if dtype is None else dtype
    device = arch_info.get_device()
    config_file_path = (
        f"{AITER_TRITON_CONFIGS_PATH}/moe/{device}-MOE-e2e-N={N}-{dtype_str}.json"
    )


    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # If a configuration has been found, return it
            return {key: val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration for the dtype
    warnings.warn(
        f"No finetuned end-to-end MoE configuration found for N={N} and dtype={dtype_str}. Tried searching at: {config_file_path}."
    )

    default_config_file_path = (
        f"{AITER_TRITON_CONFIGS_PATH}/moe/{device}-MOE-e2e-{dtype_str}.json"
    )

    if os.path.exists(default_config_file_path):
        with open(default_config_file_path) as f:
            # If a configuration has been found, return it
            return {key: val for key, val in json.load(f).items()}

    warnings.warn(
        f"No end-to-end MoE configuration found for dtype={dtype_str}. Tried searching at: {default_config_file_path}. Using hardcoded default configuration."
    )

    return None


def get_optimal_moe_e2e_config(
    N: int,
    dtype: torch.dtype,
    # blockscale fp8
    use_fp8_w8a8: Optional[bool] = False,
    M: int = 1,
):
    N = triton.next_power_of_2(N//2)
    dtype_str = get_config_dtype_str(dtype, use_fp8_w8a8=use_fp8_w8a8)
    configs = get_e2e_moe_configs(N, dtype_str)
    if configs is not None:
        if configs:
            if M < M_THRESHOLD_SMALL:
                config = configs["small_M"]
            elif M < M_THRESHOLD_MEDIUM:
                config = configs["medium_M"]
            else:
                config = configs["large_M"]
    else:
        if dtype == torch.float32:
            config = {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K1": 32,
                "BLOCK_SIZE_K2": 32,
                "GROUP_SIZE_M": 1,
            }
        else:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K1": 128,
                "BLOCK_SIZE_K2": 128,
                "GROUP_SIZE_M": 1,
            }

    return config


def get_optimal_moe_e2e_config_func(
    N: int,
    dtype: torch.dtype,
    use_fp8_w8a8: Optional[bool] = False,
):
    return functools.partial(get_optimal_moe_e2e_config, N, dtype, use_fp8_w8a8)
