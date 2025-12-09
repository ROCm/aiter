import functools
import json
import os

import triton

from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH


# Standard bounds for M_LEQ_x keys
STANDARD_M_BOUNDS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


@functools.lru_cache(maxsize=1024)
def get_gemm_config(
    config_name,
    M,
    N=None,
    K=None,
    bounds=None,
    specialized_filename=None,
):
    """
    Load a GEMM configuration using the standardized M_LEQ_x/M_GEQ_y/any format.

    This function provides a unified way to load GEMM configs across all kernels.
    It uses the following logic:
    1. Load default config file: {arch}-{config_name}.json
    2. If N and K are provided, try to load specialized config: {arch}-{config_name}-N={N}-K={K}.json
       Or if specialized_filename is provided, use: {arch}-{config_name}-{specialized_filename}.json
    3. Search for M_LEQ_x keys in order of bounds (default: STANDARD_M_BOUNDS)
    4. If no M_LEQ_x matches, search for M_GEQ_x keys in reverse order
    5. Fall back to "any" if no bounds match

    Args:
        config_name: Name of the config (example - "GEMM-A16W16")
        M: M dimension of the GEMM
        N: N dimension of the GEMM (optional)
        K: K dimension of the GEMM (optional)
        bounds: Custom bounds to use instead of STANDARD_M_BOUNDS (optional)
        specialized_filename: Custom specialized filename suffix (optional)

    Returns:
        Dictionary with the config params
    """
    if not hasattr(get_gemm_config, "_config_cache"):
        get_gemm_config._config_cache = {}

    dev = arch_info.get_arch()
    cache_key = f"{dev}_{config_name}"

    if cache_key not in get_gemm_config._config_cache:
        get_gemm_config._config_cache[cache_key] = {}

        # Load default config
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}.json"

        with open(fpath, "r") as file:
            config = json.load(file)
        get_gemm_config._config_cache[cache_key]["default"] = config

    # Determine which config dict to use (default or specialized)
    config_dict_key = "default"
    
    # Handle custom specialized filename (for fused kernels with multiple N dims)
    if specialized_filename is not None:
        spec_key = specialized_filename
        if spec_key not in get_gemm_config._config_cache[cache_key]:

            fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}-{specialized_filename}.json"
            if os.path.exists(fpath):
                with open(fpath, "r") as file:
                    config = json.load(file)
                get_gemm_config._config_cache[cache_key][spec_key] = config
                config_dict_key = spec_key
        else:
            config_dict_key = spec_key

    elif N is not None and K is not None:
        nk_key = f"{N}_{K}"
        if nk_key not in get_gemm_config._config_cache[cache_key]:
            # load specialized config
            fpath = (
                f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-{config_name}-N={N}-K={K}.json"
            )
            if os.path.exists(fpath):
                with open(fpath, "r") as file:
                    config = json.load(file)
                get_gemm_config._config_cache[cache_key][nk_key] = config
                config_dict_key = nk_key
        else:
            config_dict_key = nk_key

    config_dict = get_gemm_config._config_cache[cache_key][config_dict_key]

    # use standard bounds unless custom bounds are passed
    search_bounds = bounds if bounds is not None else STANDARD_M_BOUNDS

    # Search for M_LEQ_x keys
    for bound in search_bounds:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_dict:
            return dict(config_dict[key])

    # Search for M_GEQ_x keys (looking for largest threshold that M exceeds)
    for bound in reversed(search_bounds):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_dict:
            return dict(config_dict[key])

    if "any" in config_dict:
        return dict(config_dict["any"])


def add_default_gemm_config_params(config):
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1

    # adding default cache_modifier if not present as some kernels need this
    if "cache_modifier" not in config and "BLOCK_SIZE_K" in config:
        config["cache_modifier"] = None

    return config


def compute_splitk_params(config, K):
    """
    Compute split-K parameters for a GEMM config.
    """
    add_default_gemm_config_params(config)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if (
        "BLOCK_SIZE_K" in config
        and config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]
    ):
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])

        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4

        config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

    return config
