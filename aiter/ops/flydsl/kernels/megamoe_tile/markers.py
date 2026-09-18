# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os


def _load_roctx():
    candidates = [
        ctypes.util.find_library("rocprofiler-sdk-roctx"),
        ctypes.util.find_library("roctx64"),
        "/opt/rocm/lib/librocprofiler-sdk-roctx.so",
        "/opt/rocm/lib/libroctx64.so",
    ]
    for path in candidates:
        if not path or not os.path.exists(path) and "/" in path:
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            lib.roctxRangePushA.restype = ctypes.c_int
            lib.roctxRangePop.argtypes = []
            lib.roctxRangePop.restype = ctypes.c_int
            if hasattr(lib, "roctxProfilerPause"):
                lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
                lib.roctxProfilerPause.restype = ctypes.c_int
            if hasattr(lib, "roctxProfilerResume"):
                lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
                lib.roctxProfilerResume.restype = ctypes.c_int
            return lib
        except (OSError, AttributeError):
            continue
    return None


_ROCTX = _load_roctx()
_PROFILER_RUNNING = (
    os.environ.get("MEGAMOE_TILE_PROFILER_STARTS_PAUSED", "0") != "1"
)


def _profile_regions_enabled() -> bool:
    return os.environ.get("MEGAMOE_TILE_PROFILE_REGIONS", "0") == "1"


def _require_profiler_control(name: str):
    if _ROCTX is None or not hasattr(_ROCTX, name):
        raise RuntimeError(f"ROCTx profiler control {name} is unavailable")
    return getattr(_ROCTX, name)


@contextlib.contextmanager
def roctx_range(name: str):
    """ROCtx range used by rocprofv3 marker/ATT collection."""

    if _ROCTX is None:
        yield
        return
    _ROCTX.roctxRangePushA(name.encode("utf-8"))
    try:
        yield
    finally:
        _ROCTX.roctxRangePop()


def profiler_pause() -> None:
    global _PROFILER_RUNNING
    if not _profile_regions_enabled() or not _PROFILER_RUNNING:
        return
    result = _require_profiler_control("roctxProfilerPause")(0)
    if result != 0:
        raise RuntimeError(f"roctxProfilerPause failed with status {result}")
    _PROFILER_RUNNING = False


def profiler_resume() -> None:
    global _PROFILER_RUNNING
    if not _profile_regions_enabled() or _PROFILER_RUNNING:
        return
    result = _require_profiler_control("roctxProfilerResume")(0)
    if result != 0:
        raise RuntimeError(f"roctxProfilerResume failed with status {result}")
    _PROFILER_RUNNING = True
