# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Thin access to the loaded HIP runtime. ctypes and stdlib only, no torch and no
# aiter imports, so aiter.jit.utils and aiter.dist can both use it.

import ctypes
import functools

# The unversioned symlink ships in the -devel package, which need not be on
# LD_LIBRARY_PATH at runtime; split rocm-sdk pip layouts carry only the versioned
# soname.
_SONAMES = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")


@functools.lru_cache(maxsize=1)
def load_hip_runtime() -> ctypes.CDLL:
    """Load libamdhip64, or raise OSError reporting why each candidate failed."""
    errors = []
    for soname in _SONAMES:
        try:
            return ctypes.CDLL(soname)
        except OSError as e:
            errors.append(f"  {soname}: {e}")
    raise OSError("Could not load the HIP runtime:\n" + "\n".join(errors))


@functools.lru_cache(maxsize=1)
def get_hip_runtime_version() -> tuple[int, int, int] | None:
    """(major, minor, patch) of the loaded runtime, or None if unavailable.

    The loaded library, not the ROCm installed on disk or the one torch was built
    against; only it can say how an attribute ordinal will be read.
    """
    try:
        libhip = load_hip_runtime()
        val = ctypes.c_int(0)
        if libhip.hipRuntimeGetVersion(ctypes.byref(val)) != 0:
            return None
    except Exception:  # noqa: BLE001
        return None

    # HIP_VERSION = major * 10000000 + minor * 100000 + patch
    raw = val.value
    return raw // 10_000_000, (raw // 100_000) % 100, raw % 100_000


def get_current_hip_device() -> int:
    """Ordinal of the device this thread is bound to, or 0 if unavailable.

    Uncached on purpose: torch.cuda.set_device() moves it between calls.
    """
    try:
        libhip = load_hip_runtime()
        val = ctypes.c_int(0)
        if libhip.hipGetDevice(ctypes.byref(val)) != 0:
            return 0
    except Exception:  # noqa: BLE001
        return 0
    return val.value
