# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    USE_LRU_CACHE,
    load_config_json,
)


@functools.lru_cache(maxsize=256 if USE_LRU_CACHE else 0)
def _get_quant_config_cached(config_name: str, backend: str, key: str) -> dict:
    dev = arch_info.get_arch()
    d_type = config_name.lower().replace("-", "_")
    config_dict = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/{dev}/{backend}/quant/{d_type}/DEFAULT.json"
    )
    if key not in config_dict:
        raise KeyError(
            f"No matching config in '{config_name}' for key={key!r} on arch "
            f"{dev} backend={backend!r} (keys present: {sorted(config_dict)})."
        )
    return config_dict[key]


def get_quant_config(config_name: str, key: str, backend: str = "gluon") -> dict:
    """Load a tuned quant kernel block-config bucket for the running GPU arch.

    Follows the nested config layout
    ``configs/<arch>/<backend>/quant/<d_type>/DEFAULT.json``, where ``<d_type>``
    is ``config_name.lower().replace("-", "_")`` (mirrors
    ``gemm_config_utils._dtype_dir()``).

    Parameters:
    - config_name: e.g. "QUANT-MXFP4", "QUANT-MXFP8".
    - key: bucket key inside DEFAULT.json, e.g. "M_LEQ_512_N_LEQ_3072".
      Bucket boundaries are the caller's concern; this loader does a single
      exact-key lookup, no bucket-walk.
    - backend: "triton" or "gluon".

    Returns a shallow copy of the bucket dict (safe to mutate).

    Raises KeyError if the bucket key isn't present in the file, or
    FileNotFoundError if DEFAULT.json itself is missing.
    """
    return _get_quant_config_cached(config_name, backend, key).copy()
