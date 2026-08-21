# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Moved to aiter.ops.triton.utils.arch_info; this module is kept so the old
# import path keeps working for out-of-tree callers.
import warnings

from aiter.ops.triton.utils.arch_info import (  # noqa: F401
    _LDS_CAP_BYTES,
    get_arch,
    is_fp4_avail,
    is_fp8_avail,
    is_mx_scale_preshuffling_avail,
    is_tdm_avail,
)

warnings.warn(
    "aiter.ops.triton.utils._triton.arch_info has moved to "
    "aiter.ops.triton.utils.arch_info; this shim will be removed in a future "
    "AITER release.",
    DeprecationWarning,
    stacklevel=2,
)
