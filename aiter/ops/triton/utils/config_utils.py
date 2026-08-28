# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config-file infrastructure for the AITER Triton ops -- the public surface.

Every tuned Triton config ships as JSON under ``aiter/ops/triton/configs`` in
the nested layout ``<arch>/<backend>/<op>/<d_type>/``. This module is the one
place to import config machinery from; the implementations live in the private
``utils/_config/`` package (mirroring ``utils/_triton/``), one small module per
family, and nothing outside this facade should import them directly.

Submodules re-exported here
---------------------------
_config.core
    ``AITER_TRITON_CONFIGS_PATH``, ``load_config_json()``,
    ``resolve_config_dir()`` -- the path constants, the cached JSON parse and
    the validated, deterministic directory builder every loader goes through.
_config.gemm
    ``get_gemm_config()`` plus the splitk / num-stages helpers.
_config.conv
    ``get_conv_config()`` with the variant-aware four-tier walk, the
    shape-key formatters, and the optional-table probes.
_config.mhc
    ``get_mhc_config()`` / ``get_mhc_post_config()``, with the gfx942 arch
    fallback.
_config.tuned
    ``get_tuned_kernel_config()``, for kernels whose autotune search space
    lives in Python and only need one pinned tile per device.
"""

# Test-facing internals, re-exported so white-box tests can reach the shared
# cache objects through the public module.
from aiter.ops.triton.utils._config.conv import (  # noqa: F401
    CONV_STANDARD_M_BOUNDS,
    _get_conv_config_cached,
    conv_config_uses_exact_routes,
    format_prepack_shape_key,
    format_shape_key,
    get_conv_config,
    has_conv_config,
    has_exact_conv_config,
)
from aiter.ops.triton.utils._config.core import (
    AITER_TRITON_CONFIGS_PATH,
    AITER_TRITON_OPS_PATH,
    USE_LRU_CACHE,
    load_config_json,
    resolve_config_dir,
)
from aiter.ops.triton.utils._config.gemm import (
    STANDARD_M_BOUNDS,
    add_default_gemm_config_params,
    compute_splitk_params,
    get_gemm_config,
    pick_gemm_num_stages,
)
from aiter.ops.triton.utils._config.mhc import (
    get_mhc_config,
    get_mhc_post_config,
    hip_post_dispatch_block,
)
from aiter.ops.triton.utils._config.tuned import (
    get_tuned_kernel_config,
)
from aiter.ops.triton.utils._triton import arch_info  # noqa: F401

__all__ = [
    "AITER_TRITON_CONFIGS_PATH",
    "AITER_TRITON_OPS_PATH",
    "CONV_STANDARD_M_BOUNDS",
    "STANDARD_M_BOUNDS",
    "USE_LRU_CACHE",
    "add_default_gemm_config_params",
    "compute_splitk_params",
    "conv_config_uses_exact_routes",
    "format_prepack_shape_key",
    "format_shape_key",
    "get_conv_config",
    "get_gemm_config",
    "get_mhc_config",
    "get_mhc_post_config",
    "get_tuned_kernel_config",
    "has_conv_config",
    "has_exact_conv_config",
    "hip_post_dispatch_block",
    "load_config_json",
    "pick_gemm_num_stages",
    "resolve_config_dir",
]
