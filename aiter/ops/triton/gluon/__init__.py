# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from packaging.version import Version

try:
    import triton

    _triton_version = Version(triton.__version__.split("+")[0])
    _min_version = Version("3.6.0")
    if _triton_version < _min_version:
        raise RuntimeError(
            f"aiter gluon kernels require triton>=3.6.0, found {triton.__version__}"
        )
except ImportError:
    pass
