# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reusable GPU architecture probe for opus subpackages.

opus subpackages (a16w16, future a8w8, ...) need to know whether the
current GPU is one they support, so they can either expose the real
implementation or fall back to a stub that raises a clear error at
*invocation* time (vs. at import time, which would break callers'
``from aiter import ...`` because :mod:`aiter.__init__` imports opus
inside a try/except that swallows ImportError and stops loading the
following 30+ unrelated ops).

The split between :func:`_detect_arch` (never raises) and
:func:`_check_arch` (raises) lets each caller pick its preferred
failure mode:

* ``__init__`` of an opus subpackage uses :func:`_detect_arch` so that
  importing it on a non-supported GPU is silent (warning + stub) and
  does not cascade into the surrounding ``aiter/__init__.py``.
* Anyone who *wants* a hard ImportError on the wrong arch (e.g. an
  external script that should refuse to start) can call
  :func:`_check_arch` directly.

Detection order (shared by both helpers):

1. ``GPU_ARCHS`` env var (split on ``';'``). Skips the special
   ``'native'`` token. This path covers build-only hosts (no GPU) and
   CI workflows that pin GPU_ARCHS explicitly.
2. ``GPU_ARCHS=native`` (default) -> probe ``rocminfo`` via
   ``aiter.jit.utils.chip_info.get_gfx_runtime``.
3. ``rocminfo`` unavailable (no GPU / CPU host) -> log debug and treat
   as "unknown"; the host-side dispatcher in ``opus_gemm.cu`` catches
   the unsupported device at call time.
"""

import logging
import os
from typing import Iterable, Optional, Tuple

logger = logging.getLogger("aiter.ops.opus._arch")


def _detect_arch(
    supported: Iterable[str],
) -> Tuple[bool, Optional[str]]:
    """Probe the active GPU arch against ``supported`` without raising.

    Parameters
    ----------
    supported : iterable of str
        GPU architecture names this feature accepts (e.g. ``{"gfx950"}``).
        Comparison is case-insensitive.

    Returns
    -------
    (ok, detected) : tuple
        ``ok`` is ``True`` iff a supported arch was detected. ``detected``
        is a human-readable string describing what we saw (the matching
        arch when ``ok`` is True; the full ``GPU_ARCHS`` env value, the
        rocminfo arch, or ``None`` for "unknown / probe failed" otherwise).
        Never raises.
    """
    supported_set = {a.lower() for a in supported}

    gpu_archs_env = os.getenv("GPU_ARCHS", "native").strip()
    explicit_archs = [
        a.strip().lower()
        for a in gpu_archs_env.split(";")
        if a.strip() and a.strip() != "native"
    ]
    # Path 1: GPU_ARCHS lists explicit arch(es). Use that as the source of
    # truth -- handles build-only hosts and multi-arch wheel scenarios where
    # ``rocminfo`` cannot tell us which arch the wheel was built for.
    if explicit_archs:
        match = next((a for a in explicit_archs if a in supported_set), None)
        if match is not None:
            return True, match
        return False, gpu_archs_env

    # Path 2: GPU_ARCHS='native' (default). Probe rocminfo.
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        gfx = get_gfx_runtime().lower()
    except Exception as e:
        logger.debug(
            "opus: arch probe could not query rocminfo (%s). "
            "Treating as unknown; downstream host dispatcher will catch "
            "unsupported devices at call time.",
            e,
        )
        return False, None

    return (gfx in supported_set), gfx


def _check_arch(
    supported: Iterable[str],
    *,
    feature: str,
    hint: Optional[str] = None,
) -> None:
    """Raise ImportError if :func:`_detect_arch` reports an unsupported arch.

    Kept for callers that prefer hard failure over a stub. Most
    in-tree callers should prefer :func:`_detect_arch` so they can
    install a stub and avoid cascading import failures.

    Parameters
    ----------
    supported : iterable of str
        GPU architecture names this feature accepts.
    feature : str
        Human-readable name of the feature being guarded; included in
        the error message (e.g. ``"aiter.ops.opus (a16w16)"``).
    hint : str, optional
        Extra guidance appended to the error message.

    Raises
    ------
    ImportError
        If detection succeeded and no detected arch is in ``supported``.
        A "probe failed / unknown" outcome is treated as pass (the
        dispatcher will catch it at call time).
    """
    ok, detected = _detect_arch(supported)
    if ok or detected is None:
        return
    supported_set = {a.lower() for a in supported}
    msg = (
        f"{feature} only supports GPU arches {sorted(supported_set)}; "
        f"detected {detected!r}."
    )
    if hint:
        msg = f"{msg} {hint}"
    raise ImportError(msg)
