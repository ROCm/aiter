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
            return lib
        except (OSError, AttributeError):
            continue
    return None


_ROCTX = _load_roctx()


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
    if (
        os.environ.get("MEGAMOE_TILE_PROFILE_REGIONS", "0") == "1"
        and _ROCTX is not None
        and hasattr(_ROCTX, "roctxProfilerPause")
    ):
        _ROCTX.roctxProfilerPause(0)


def profiler_resume() -> None:
    if (
        os.environ.get("MEGAMOE_TILE_PROFILE_REGIONS", "0") == "1"
        and _ROCTX is not None
        and hasattr(_ROCTX, "roctxProfilerResume")
    ):
        _ROCTX.roctxProfilerResume(0)
