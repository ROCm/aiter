# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Thin access to the loaded HIP runtime. ctypes and stdlib only, no torch and no
# aiter imports, so aiter.jit.utils and aiter.dist can both use it. The one
# exception is the Windows DLL search, which reuses cpp_extension's ROCm lookup.

import ctypes
import functools
import glob
import os
import sys


def _library_names() -> tuple[str, ...]:
    if sys.platform != "win32":
        # The unversioned symlink ships in the -devel package, which need not be
        # on LD_LIBRARY_PATH at runtime; split rocm-sdk pip layouts carry only the
        # versioned soname.
        names = ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")
        torch_hip = _torch_hip_version()
        if torch_hip:
            torch_soname = f"libamdhip64.so.{torch_hip.split('.')[0]}"
            if torch_soname not in names:
                names += (torch_soname,)
        return names

    # Windows ROCm ships a version-suffixed DLL, e.g. `amdhip64_7.dll`.
    from cpp_extension import _find_rocm_home

    rocm_home = _find_rocm_home()
    if not rocm_home:
        return ("amdhip64.dll",)

    dlls = sorted(glob.glob(os.path.join(rocm_home, "bin", "amdhip64*.dll")))
    return (*dlls, "amdhip64.dll")


def _torch_hip_version() -> str | None:
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    version = getattr(torch, "version", None)
    if version is None:
        return None
    return getattr(version, "hip", None)


def _loaded_hip_runtime() -> ctypes.CDLL | None:
    """The HIP runtime already loaded into this process, e.g. torch's."""
    if sys.platform == "win32":
        return None
    for name in _library_names():
        try:
            return ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LAZY)
        except OSError:
            continue
    return None


@functools.lru_cache(maxsize=1)
def load_hip_runtime() -> ctypes.CDLL:
    """Load the HIP runtime, or raise OSError reporting why each candidate failed.

    Binds to the copy already loaded into the process when there is one (Linux
    only), so the handle shares torch's runtime instead of loading a second one
    by name. The choice is cached, so call this after torch has initialized HIP.
    """
    loaded = _loaded_hip_runtime()
    if loaded is not None:
        return loaded
    errors = []
    for soname in _library_names():
        try:
            return ctypes.CDLL(soname)
        except OSError as e:
            errors.append(f"  {soname}: {e}")
    raise OSError("Could not load the HIP runtime:\n" + "\n".join(errors))


@functools.lru_cache(maxsize=1)
def _query_hip_runtime_version() -> tuple[int, int, int]:
    """Cache successful version reads; exceptions are not cached."""
    libhip = load_hip_runtime()
    val = ctypes.c_int(0)
    err = libhip.hipRuntimeGetVersion(ctypes.byref(val))
    if err != 0:
        raise RuntimeError(f"hipRuntimeGetVersion failed with error {err}")

    # HIP_VERSION = major * 10000000 + minor * 100000 + patch
    raw = val.value
    return raw // 10_000_000, (raw // 100_000) % 100, raw % 100_000


def get_hip_runtime_version() -> tuple[int, int, int] | None:
    """(major, minor, patch) of the loaded runtime, or None if unavailable.

    The loaded library, not the ROCm installed on disk or the one torch was built
    against; only it can say how an attribute ordinal will be read. Successful
    reads are cached; failures return None and can be retried.
    """
    try:
        return _query_hip_runtime_version()
    except Exception:  # noqa: BLE001
        return None


def get_current_hip_device() -> int | None:
    """Ordinal of the device this thread is bound to, or None if HIP cannot say.

    None rather than 0, which is a real ordinal a caller could act on.

    Uncached on purpose: torch.cuda.set_device() moves it between calls.
    """
    try:
        libhip = load_hip_runtime()
        val = ctypes.c_int(0)
        if libhip.hipGetDevice(ctypes.byref(val)) != 0:
            return None
    except Exception:  # noqa: BLE001
        return None
    return val.value
