# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Any, Dict, Optional
import os
import json
import functools

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
):
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def get_config_file_name(dtype: Optional[str]) -> str:
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    return f"device_name={device_name}{dtype_selector}.json"


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
    json_file_name = get_config_file_name(dtype)

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../moe_configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            # If a configuration has been found, return it
            return {key: val for key, val in json.load(f).items()}

    # If no optimized configuration is available, we will use the default
    # configuration
    return None


def get_optimal_moe_config(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
    M: int = 1,
):
    dtype_str = get_config_dtype_str(
        dtype, use_int8_w8a16, use_int8_w8a8, use_fp8_w8a8, use_int4_w4a16
    )
    configs = get_moe_configs(dtype_str)
    configs = None
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

    return config


def get_optimal_moe_config_func(
    dtype: torch.dtype,
    use_int8_w8a16: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int4_w4a16: Optional[bool] = False,
):
    return functools.partial(
        get_optimal_moe_config,
        dtype,
        use_int8_w8a16,
        use_int8_w8a8,
        use_fp8_w8a8,
        use_int4_w4a16,
    )
