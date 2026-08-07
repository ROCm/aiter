import copy
import functools
import itertools
import json
import os

import triton

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import (
    AITER_TRITON_CONFIGS_PATH,
    USE_LRU_CACHE,
    load_config_json,
)

# Standard bounds for M_LEQ_x keys (tuple for hashability with LRU cache)
STANDARD_M_BOUNDS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


def _dtype_dir(config_name: str) -> str:
    """Nested-layout directory for a config family:
    ``GEMM-AFP4WFP4`` -> ``gemm_afp4wfp4``."""
    return config_name.lower().replace("-", "_")


def _load_config_file(
    cache_dict: dict,
    cache_key: str,
    fpath: str,
    config_key: str,
    fpath_should_exist: bool = False,
) -> bool:
    """
    Helper function to load a config file and cache it.

    Retained for mhc_config_utils.py — the GEMM resolver uses _load_json().
    """
    if os.path.exists(fpath):
        with open(fpath, "r") as file:
            config = json.load(file)
        cache_dict[cache_key][config_key] = config
        return True
    elif fpath_should_exist:
        raise AssertionError(f"Required config file doesn't exist: {fpath}")
    return False


@functools.lru_cache(maxsize=1024 if USE_LRU_CACHE else 0)
def _get_gemm_config_cached(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
    specialized_filename: str | None = None,
    backend: str | None = None,
    B: int | None = None,
) -> tuple[dict, bool]:
    """
    Internal cached implementation. Do NOT use this directly — use
    ``get_gemm_config()`` instead, which returns a defensive deep-copy so
    callers can freely mutate the returned dict without polluting the cache.

    Resolves from ``<arch>/<backend>/gemm/<d_type>/`` (prefix-less filenames,
    default named ``DEFAULT.json``) first; ``backend=None`` tries triton then
    gluon. Falls back to the legacy flat ``gemm/`` layout (arch-prefixed) for unmigrated configs.
    """
    # Input validation
    assert M >= 0, "M must be positive."
    assert N is None or N > 0, "N must be positive when provided."
    assert K is None or K > 0, "K must be positive when provided."
    assert bounds is None or (
        len(bounds) > 0
        and all(x > 0 for x in bounds)
        and all(x < y for x, y in itertools.pairwise(bounds))
    ), "When provided, bounds must be a non-empty tuple of strictly increasing positive numbers."

    dev = arch_info.get_arch()

    # Nested layout <arch>/<backend>/gemm/<d_type>/ (no arch prefix, default
    # named DEFAULT.json) first, then legacy flat gemm/ (arch-prefixed) for unmigrated configs.
    # TODO(satya): drop the legacy dirs once all configs are migrated.
    dtype_dir = _dtype_dir(config_name)
    arch_prefix = f"{dev}-"
    if backend is None:
        candidate_dirs = [
            (f"{AITER_TRITON_CONFIGS_PATH}/{dev}/triton/gemm/{dtype_dir}", ""),
            (f"{AITER_TRITON_CONFIGS_PATH}/{dev}/gluon/gemm/{dtype_dir}", ""),
            (
                f"{AITER_TRITON_CONFIGS_PATH}/gemm",
                arch_prefix,
            ),  # TODO(satya): legacy, remove
        ]
    else:
        candidate_dirs = [
            (f"{AITER_TRITON_CONFIGS_PATH}/{dev}/{backend}/gemm/{dtype_dir}", ""),
            (
                f"{AITER_TRITON_CONFIGS_PATH}/gemm/{backend}",
                arch_prefix,
            ),  # TODO(satya): legacy, remove
            (
                f"{AITER_TRITON_CONFIGS_PATH}/gemm",
                arch_prefix,
            ),  # TODO(satya): legacy, remove
        ]
    cfg_dir, name_prefix = candidate_dirs[-1]
    default_stem = f"{name_prefix}{config_name}"
    for _dir, _prefix in candidate_dirs:
        # Nested dirs (empty prefix) name their default DEFAULT.json.
        _stem = f"{_prefix}{config_name}" if _prefix else "DEFAULT"
        if os.path.exists(f"{_dir}/{_stem}.json"):
            cfg_dir, name_prefix, default_stem = _dir, _prefix, _stem
            break

    # Load default config (must exist)
    default_fpath = f"{cfg_dir}/{default_stem}.json"
    config_dict = load_config_json(default_fpath)
    if config_dict is None:
        raise AssertionError(f"Required config file doesn't exist: {default_fpath}")

    # Specialized configs override the default; the first existing file wins.
    # A custom specialized_filename (fused kernels with multiple N dims)
    # bypasses the B/N/K candidates.
    specialized_suffixes = []
    if specialized_filename is not None:
        specialized_suffixes = [specialized_filename]
    elif N is not None and K is not None:
        if B is not None:
            specialized_suffixes.append(f"B={B}-N={N}-K={K}")
        specialized_suffixes.append(f"N={N}-K={K}")

    is_tuned = False
    for suffix in specialized_suffixes:
        specialized_config = load_config_json(
            f"{cfg_dir}/{name_prefix}{config_name}-{suffix}.json"
        )
        if specialized_config is not None:
            config_dict, is_tuned = specialized_config, True
            break

    # use standard bounds unless custom bounds are passed
    search_bounds = bounds if bounds is not None else STANDARD_M_BOUNDS

    # Search for M_LEQ_x keys
    for bound in search_bounds:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    # Search for M_GEQ_x keys
    for bound in reversed(search_bounds):
        key = f"M_GEQ_{bound}"
        if M >= bound and key in config_dict:
            return dict(config_dict[key]), is_tuned

    if "any" in config_dict:
        return dict(config_dict["any"]), False

    raise KeyError(
        f"No matching configuration found for M={M}, N={N}, K={K}, B={B}, "
        f"specialized_filename={specialized_filename!r} in config '{config_name}'."
    )


def get_gemm_config(
    config_name: str,
    M: int,
    N: int | None = None,
    K: int | None = None,
    bounds: tuple[int, ...] | None = None,
    specialized_filename: str | None = None,
    backend: str | None = None,
    B: int | None = None,
) -> tuple[dict, bool]:
    """
    Load a GEMM configuration using the standardized M_LEQ_x/M_GEQ_y/any format.

    This function provides a unified way to load GEMM configs across all kernels.
    It uses the following logic:
    1. Load default config file: <d_type>/DEFAULT.json (legacy: {arch}-{config_name}.json)
    2. If B, N and K are provided, try B-specialized config: {config_name}-B={B}-N={N}-K={K}.json
    3. If N and K are provided, try to load specialized config: {config_name}-N={N}-K={K}.json
       Or if specialized_filename is provided, use: {config_name}-{specialized_filename}.json
    4. Search for M_LEQ_x keys in order of bounds (default: STANDARD_M_BOUNDS)
    5. If no M_LEQ_x matches, search for M_GEQ_x keys in reverse order
    6. Fall back to "any" if no bounds match

    Args:
        config_name: Name of the config (example - "GEMM-A16W16")
        M: M dimension of the GEMM
        N: N dimension of the GEMM (optional)
        K: K dimension of the GEMM (optional)
        bounds: Custom bounds to use instead of STANDARD_M_BOUNDS (optional)
        specialized_filename: Custom specialized filename suffix (optional)
        backend: Backend name for per-backend config subdirectory (optional)
        B: Batch dimension for batched GEMM (optional)

    Returns:
        Dictionary with the config params (a fresh deep-copy safe to mutate),
        bool indicating if the config is tuned.(True if tuned, False otherwise)
    """
    config, is_tuned = _get_gemm_config_cached(
        config_name, M, N, K, bounds, specialized_filename, backend, B
    )
    return copy.deepcopy(config), is_tuned


def add_default_gemm_config_params(config: dict) -> dict:
    """
    this fn ensures that all configs have required default values.

    Args:
        config: Dictionary containing GEMM configuration parameters.

    Returns:
        same object as input
    """
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1

    # adding default cache_modifier if not present as some kernels need this
    if "cache_modifier" not in config and "BLOCK_SIZE_K" in config:
        config["cache_modifier"] = None

    return config


def compute_splitk_params(config: dict, K: int) -> dict:
    """
    this fn calculates the SPLITK_BLOCK_SIZE and adjusts BLOCK_SIZE_K
    if necessary based on the NUM_KSPLIT value in the config.

    Args:
        config: Dictionary containing GEMM configuration parameters.
        K: K dimension of the GEMM operation (must be positive)

    Returns:
        same object as input
    """
    assert K > 0, "K must be positive"

    add_default_gemm_config_params(config)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

    if "BLOCK_SIZE_K" in config:
        # If NUM_KSPLIT makes K too small, then BLOCK_K will decrease to be smaller than
        # GROUP_K.
        while (
            config["NUM_KSPLIT"] > 1
            and config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]
        ):
            config["NUM_KSPLIT"] = max(config["NUM_KSPLIT"] // 2, 1)
            config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])

        # If BLOCK_SIZE_K is still too large with NUM_KSPLIT=1, fix it to equal K dim.
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])

            if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
                config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 2

        config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 16)

        # Round the SPLITK_BLOCK_SIZE to multiple of BLOCK_SIZE_K and update NUM_KSPLIT to again.
        if config["NUM_KSPLIT"] > 1 and (
            config["SPLITK_BLOCK_SIZE"] % config["BLOCK_SIZE_K"] != 0
        ):
            config["SPLITK_BLOCK_SIZE"] = (
                triton.cdiv(config["SPLITK_BLOCK_SIZE"], config["BLOCK_SIZE_K"])
                * config["BLOCK_SIZE_K"]
            )
            config["NUM_KSPLIT"] = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

    return config


def _padded_size_32_4(n):
    pad = (n >> 5) << 2
    if (n & 31) == 0 and pad >= 4:
        pad -= 4
    return n + pad


def _padded_size_pow2(n, interval, padding):
    log2_i = (interval - 1).bit_length()
    log2_p = (padding - 1).bit_length() if padding else 0
    pad = (n >> log2_i) << log2_p
    if n % interval == 0 and pad >= padding:
        pad -= padding
    return n + pad


def _gemm_lds_bytes(
    block_m, block_n, block_k, bits_a, bits_b, num_stages, use_async_padding
):
    elem_a = block_m * block_k
    elem_b = block_k * block_n
    if use_async_padding:
        # Padded shared encoding + N buffers (matches TensorAtlas
        # _estimate_triton_lds_async_copy / tritonBLAS origami).
        pa = _padded_size_32_4(elem_a)
        pb = _padded_size_32_4(elem_b)
        if block_k & (block_k - 1) == 0:
            pa = max(pa, _padded_size_pow2(elem_a, block_k, 8))
        if block_n & (block_n - 1) == 0:
            pb = max(pb, _padded_size_pow2(elem_b, block_n, 8))
        return num_stages * (pa * bits_a + pb * bits_b) // 8
    # Non-async: (N-1) extra buffer pairs beyond the active stage.
    LDSA = elem_a * bits_a
    LDSB = elem_b * bits_b
    if num_stages <= 1:
        return max(LDSA, LDSB) // 8
    return (LDSA + LDSB) * (num_stages - 1) // 8


def pick_gemm_num_stages(
    arch, block_m, block_n, block_k, bits_a, bits_b, use_async_padding=False
):
    assert min(block_m, block_n, block_k, bits_a, bits_b) > 0
    # bits_a / bits_b: element bit-widths (8 for fp8, 4 for mxfp4).
    # use_async_padding: True when the kernel lowers to async direct-to-LDS
    # with padded shared encoding (e.g. a4w4 on gfx950).
    cap = arch_info._LDS_CAP_BYTES.get(arch)
    if cap is None:
        return 2
    lds = _gemm_lds_bytes(
        block_m, block_n, block_k, bits_a, bits_b, 2, use_async_padding
    )
    return 2 if lds <= cap else 1
