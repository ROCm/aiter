# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import warnings
from typing import Any

import torch

from ._triton import arch_info
from .core import AITER_TRITON_CONFIGS_PATH, USE_LRU_CACHE, load_config_json

M_THRESHOLD_SMALL = 256
M_THRESHOLD_MEDIUM = 1024


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
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


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def get_moe_configs(dtype: str | None) -> dict[int, Any] | None:
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
    dev = arch_info.get_arch()
    configs = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/moe/{dev}-MOE-{dtype_str}.json"
    )
    if configs is not None:
        return configs

    # If no optimized configuration is available, we will use the default
    # configuration
    warnings.warn(
        f"No MoE configuration found for device '{dev}' with dtype '{dtype_str}'. Using default configuration."
    )
    return None


def get_optimal_moe_config(
    dtype: torch.dtype,
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
    M: int = 1,
):
    dtype_str = get_config_dtype_str(
        dtype, use_int8_w8a16, use_int8_w8a8, use_fp8_w8a8, use_int4_w4a16, use_mxfp4
    )
    # print(f"dtype_str={dtype_str}")
    configs = get_moe_configs(dtype_str)
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
    use_int8_w8a16: bool | None = False,
    use_int8_w8a8: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    use_int4_w4a16: bool | None = False,
    use_mxfp4: bool | None = False,
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
