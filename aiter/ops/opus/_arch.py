# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reusable GPU architecture guard for opus submodules.

Each opus subpackage (a16w16, future a8w8, ...) calls ``_check_arch`` at
import time with its own supported architecture set. The check raises
``ImportError`` when the current GPU is not in that set, so ``from
aiter.ops.opus.<sub> import ...`` fails fast with a clear message instead
of dropping the user into a JIT compile failure or, worse, a silent
runtime miscompare against gfx950-only intrinsics.

Detection order in ``_check_arch``:

1. ``GPU_ARCHS`` env var (split on ';'). Skips the special ``'native'``
   token. This path covers build-only hosts (no GPU) and CI workflows
   that pin GPU_ARCHS explicitly.
2. ``GPU_ARCHS=native`` (default) -> probe ``rocminfo`` via
   ``aiter.jit.utils.chip_info.get_gfx_runtime``.
3. ``rocminfo`` unavailable (no GPU / CPU host) -> log debug and pass;
   the host-side dispatcher in ``opus_gemm.cu`` catches the unsupported
   device at call time.
"""

import logging
import os
from typing import Iterable, Optional

logger = logging.getLogger("aiter.ops.opus._arch")


def _check_arch(
    supported: Iterable[str],
    *,
    feature: str,
    hint: Optional[str] = None,
) -> None:
    """Raise ImportError if the current GPU arch is not in ``supported``.

    Parameters
    ----------
    supported : iterable of str
        GPU architecture names this feature accepts (e.g. ``{"gfx950"}``).
        Comparison is case-insensitive.
    feature : str
        Human-readable name of the feature being guarded; included in the
        error message (e.g. ``"aiter.ops.opus (a16w16)"``).
    hint : str, optional
        Extra guidance appended to the error message (e.g. instructions on
        how to set GPU_ARCHS).

    Raises
    ------
    ImportError
        If detection succeeded and no detected arch is in ``supported``.
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
        if any(a in supported_set for a in explicit_archs):
            return
        msg = (
            f"{feature} only supports GPU arches {sorted(supported_set)}; "
            f"GPU_ARCHS={gpu_archs_env!r} requests {explicit_archs} "
            f"(none supported)."
        )
        if hint:
            msg = f"{msg} {hint}"
        raise ImportError(msg)

    # Path 2: GPU_ARCHS='native' (default). Probe rocminfo.
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        gfx = get_gfx_runtime().lower()
    except Exception as e:
        logger.debug(
            "opus: skipping arch guard for %s (GPU_ARCHS=native and "
            "rocminfo unavailable: %s). Downstream host dispatcher will "
            "catch unsupported devices.",
            feature,
            e,
        )
        return

    if gfx in supported_set:
        return
    msg = (
        f"{feature} only supports GPU arches {sorted(supported_set)}; "
        f"rocminfo detected gfx={gfx!r}."
    )
    if hint:
        msg = f"{msg} {hint}"
    raise ImportError(msg)
