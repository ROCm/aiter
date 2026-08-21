# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Merged into aiter.ops.triton.utils.arch_info; this module is kept so the old
# import path keeps working for out-of-tree callers.
import warnings

from aiter.ops.triton.utils.arch_info import (  # noqa: F401
    get_num_sms,
    get_num_xcds,
)

warnings.warn(
    "aiter.ops.triton.utils.device_info has merged into "
    "aiter.ops.triton.utils.arch_info; this shim will be removed in a future "
    "AITER release.",
    DeprecationWarning,
    stacklevel=2,
)
