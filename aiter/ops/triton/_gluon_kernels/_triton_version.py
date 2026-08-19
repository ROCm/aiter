# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Minimum-triton check for the gluon kernels that need triton >= 3.6.

This used to live in ``aiter/ops/triton/gluon/__init__.py``, so importing any
kernel in that package ran it. Those kernels now sit under
``_gluon_kernels/<arch>/`` next to kernels that were never gated (the gfx1250
MoE/GEMM/quant kernels and friends reach ``_gluon_kernels`` without it), so the
check cannot live in a package ``__init__`` any more without widening its reach.
The modules that were behind it call ``require_gluon_triton()`` explicitly
instead, which keeps the gate on exactly the kernels it always applied to.
"""

import functools
import os
import warnings

from packaging.version import Version

_MIN_TRITON = Version("3.6.0")


@functools.lru_cache(maxsize=1)
def require_gluon_triton():
    """Raise unless triton is new enough for these kernels.

    A no-op when triton is absent, and downgraded to a warning under
    AITER_USE_SYSTEM_TRITON=1 -- both matching the original guard.
    """
    try:
        import triton
    except ImportError:
        return

    if Version(triton.__version__.split("+")[0]) >= _MIN_TRITON:
        return

    msg = (
        f"aiter gluon kernels require triton>={_MIN_TRITON}, found {triton.__version__}"
    )
    if int(os.environ.get("AITER_USE_SYSTEM_TRITON", "0")):
        warnings.warn(
            f"[aiter] AITER_USE_SYSTEM_TRITON=1: {msg}. "
            "Please install a compatible version via .github/scripts/install_triton.sh, "
            "otherwise unexpected errors may occur.",
        )
    else:
        raise RuntimeError(msg)
