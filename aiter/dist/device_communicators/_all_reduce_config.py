# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import warnings
from collections.abc import Mapping


def get_ar_1stage_max_bytes(environ: Mapping[str, str]) -> int:
    """Read the byte limit, preserving the deprecated KB name's byte semantics.

    An explicit BYTES value takes precedence, including a negative value that
    selects the caller's default limit.
    """
    if "AITER_AR_1STAGE_MAX_BYTES" in environ:
        return int(environ["AITER_AR_1STAGE_MAX_BYTES"])
    if "AITER_AR_1STAGE_MAX_KB" in environ:
        value = int(environ["AITER_AR_1STAGE_MAX_KB"])
        warnings.warn(
            "AITER_AR_1STAGE_MAX_KB is deprecated and is interpreted in bytes, "
            "not KiB. Use AITER_AR_1STAGE_MAX_BYTES with the same value to "
            "preserve the limit.",
            FutureWarning,
            stacklevel=2,
        )
        return value
    return -1
