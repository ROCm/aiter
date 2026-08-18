# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""JIT-compiled sampling kernels."""

import os
import re

import torch


def _get_rocm_major_version() -> int | None:
    version = os.environ.get("AITER_ROCM_VERSION") or getattr(
        torch.version, "hip", None
    )
    match = re.match(r"\s*(\d+)", version or "")
    return int(match.group(1)) if match else None


_rocm_major_version = _get_rocm_major_version()
if _rocm_major_version is not None and _rocm_major_version >= 10:
    raise ImportError(
        "AITER cpp_itfs sampling is disabled on ROCm 10 and later because "
        "csrc/cpp_itfs/sampling/sampling.cuh is not compatible with this toolchain."
    )
