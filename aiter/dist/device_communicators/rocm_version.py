# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Best-effort ROCm version detection.

Used to gate the cross-device buffer transport for custom communication ops:
on gfx1250, ``hipIpcGetMemHandle`` is unusable on early ROCm releases, so we
fall back to the HIP VMM path (see :mod:`vmm_allocator`). Once the runtime is
recent enough that IPC works, the IPC path is preferred again. This module tells
the caller which ROCm it is running on so that decision can be made.

The version is detected from several sources, tried in priority order, so a
missing ``/opt/rocm/.info/version`` (e.g. a versioned install under
``/opt/rocm-<ver>`` or a container with a non-standard layout) does not defeat
detection:

    1. ``AITER_ROCM_VERSION`` env override (for testing the gate itself).
    2. ``.info/version*`` files under candidate ROCm roots.
    3. ``torch.version.hip`` (the HIP version torch was built against).
    4. ``hipRuntimeGetVersion`` from the loaded ``libamdhip64.so``.

Returns a ``(major, minor, patch)`` tuple that compares naturally against a
threshold like ``(7, 15)``, or ``None`` if every source failed.
"""

import ctypes
import functools
import glob
import os
import re

from aiter import logger


def _parse(s):
    """Extract the leading ``major.minor[.patch]`` from an arbitrary string."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", s or "")
    if not m:
        return None
    return tuple(int(x) for x in m.groups(default="0"))


def _candidate_roots():
    """ROCm install roots to probe, highest priority first."""
    roots = []
    # Explicit env pointers win — they name the ROCm the user actually wired up.
    for env in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        p = os.environ.get(env)
        if p:
            roots.append(p)
    # Default unversioned symlink.
    roots.append("/opt/rocm")
    # Versioned installs — probe the highest version first.
    roots.extend(sorted(glob.glob("/opt/rocm-*"), reverse=True))
    # De-dup while preserving order.
    seen = set()
    ordered = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            ordered.append(r)
    return ordered


def _from_info_files():
    """Read ``.info/version*`` under each candidate root."""
    filenames = (".info/version", ".info/version-dev", ".info/version-hip")
    for root in _candidate_roots():
        for fn in filenames:
            path = os.path.join(root, fn)
            try:
                with open(path) as f:
                    v = _parse(f.read())
            except OSError:
                continue
            if v:
                logger.info("ROCm version %s from %s", v, path)
                return v
    return None


def _from_torch():
    """``torch.version.hip`` — HIP version torch was built against.

    On modern ROCm (>= 6.0) the HIP major.minor tracks the ROCm release, so
    ``"7.15.xxxxx"`` -> ``(7, 15, xxxxx)``.
    """
    try:
        import torch

        hip = getattr(torch.version, "hip", None)
    except Exception:
        return None
    v = _parse(hip)
    if v:
        logger.info("ROCm version %s from torch.version.hip=%r", v, hip)
    return v


def _from_hip_runtime():
    """``hipRuntimeGetVersion`` from the live ``libamdhip64.so``.

    HIP packs the version as ``major*1e7 + minor*1e5 + patch``. This is the HIP
    *runtime* version; it aligns with the ROCm release from ROCm 6.0 onward,
    which is the range that matters for the IPC gate.
    """
    hip = None
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            hip = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if hip is None:
        return None
    try:
        raw = ctypes.c_int()
        if hip.hipRuntimeGetVersion(ctypes.byref(raw)) != 0:
            return None
        val = raw.value
    except Exception:
        return None
    v = (val // 10_000_000, (val // 100_000) % 100, val % 100_000)
    logger.info("ROCm version %s from hipRuntimeGetVersion=%d", v, val)
    return v


@functools.lru_cache(maxsize=1)
def get_rocm_version():
    """Best-effort ``(major, minor, patch)`` of the active ROCm.

    Returns ``None`` if no source could determine a version. Result is cached
    for the process lifetime.
    """
    override = os.environ.get("AITER_ROCM_VERSION")
    if override:
        v = _parse(override)
        logger.info("ROCm version %s from AITER_ROCM_VERSION=%r", v, override)
        return v

    for source in (_from_info_files, _from_torch, _from_hip_runtime):
        v = source()
        if v:
            return v

    logger.warning("Could not detect ROCm version from any source")
    return None
