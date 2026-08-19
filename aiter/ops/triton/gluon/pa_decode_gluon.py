# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deprecated alias for the relocated pa_decode gluon kernel.

Now specialized per arch under ``_gluon_kernels/<arch>/attention/pa_decode.py``;
this module forwards to the copy for the live arch. See this package's __init__.
"""

import sys

from aiter.ops.triton.utils._triton import arch_info

if arch_info.get_arch() == "gfx942":
    from aiter.ops.triton._gluon_kernels.gfx942.attention import pa_decode as _impl
else:
    from aiter.ops.triton._gluon_kernels.gfx950.attention import pa_decode as _impl

sys.modules[__name__] = _impl
