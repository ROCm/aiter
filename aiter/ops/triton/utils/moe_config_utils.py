# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MOE dispatch-table loading: ``get_moe_dispatch()`` for the per-arch tuned
tables the MOE GEMM wrappers key by block size and shape, on top of the
shared core in ``config_utils``.
"""

import functools

from aiter.ops.triton.utils.config_utils import (
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)


@functools.lru_cache(maxsize=64 if USE_LRU_CACHE else 0)
def get_moe_dispatch(config_name: str, arch: str, backend: str) -> dict:
    """Tuned dispatch table for one MOE GEMM family on one dispatch path.

    ``backend`` is declared by the dispatch path that consumes the table --
    the two paths key the same family differently (triton by
    ``bm<block_m>_n<N>_k<K>``, gluon with a bucket suffix), so each arch
    ships the file for the path it actually runs. Returns ``{}`` when no
    tuned file is shipped for this arch and backend; callers fall back to
    their safe defaults.

    ``arch`` keys the cache for callers that already resolved it;
    ``resolve_config_dir()`` reads the same value from ``arch_info``. The
    returned dict is the shared cached object -- treat it as read-only.

    Args:
        config_name: MOE family name, e.g. ``"A8W4"`` or ``"A4W4"``.
        arch: the running architecture, as the caller resolved it.
        backend: ``"triton"`` or ``"gluon"`` -- the consuming dispatch path.
    """
    cfg_dir = resolve_config_dir("moe", config_name, backend=backend)
    dispatch = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
    return dispatch if dispatch is not None else {}
