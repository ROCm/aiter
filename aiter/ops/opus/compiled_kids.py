# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which OPUS kernel ids the loaded ``module_deepgemm_opus`` really contains.

``module_deepgemm_opus`` is a subset build: the heuristic floor, the kids that
the in-tree tuned CSVs name at build time and ``compiled_kids_opus.json``. A
tuned row can therefore pass the Python registry checks and still name a kid
that this binary lacks, and the C++ launcher then raises ``kid N is not
compiled in the current OPUS module`` inside the serving process.

The codegen embeds the exact compiled ``family/gfx/cu_num/kid`` set in the
binary (``opus_compiled_kids()``), so a prebuilt module carries its own
manifest. Callers that select a *tuned* kid ask :func:`tuned_kid_is_compiled`
and fall back to their next candidate when it answers ``False``. An exact
caller-chosen kid is not filtered: it still reaches the launcher and its error.

The C++ dispatcher looks a kid up under the *physical* device (``gcnArchName``
and ``multiProcessorCount``), not under the ``CU_NUM`` / ``GPU_ARCHS`` overrides
that tuned-row lookup honours. :func:`tuned_kid_is_compiled` therefore asks the
device itself, never the row's ``cu_num``.
"""

from __future__ import annotations

import threading

import torch

from aiter import logger

from ...jit.core import compile_ops
from ...jit.utils.build_targets import base_gfx_arch
from ._arch import _device_arch_and_cu


@compile_ops("module_deepgemm_opus", fc_name="opus_compiled_kids")
def opus_compiled_kids() -> str: ...


# One lock guards the manifest cache and the once-per-kid warnings: without it,
# threads that miss together each log the same warning.
_LOCK = threading.Lock()
_UNSET = object()
_manifest: frozenset[str] | None | object = _UNSET
_warned: set[str] = set()


def _load_manifest() -> frozenset[str] | None:
    try:
        raw = opus_compiled_kids()
    except AttributeError:
        logger.warning(
            "module_deepgemm_opus does not export opus_compiled_kids(); it was "
            "built before the compiled-kid manifest, so tuned OPUS kids cannot "
            "be checked against it. Rebuild it (AITER_REBUILD=1) to get the "
            "uncompiled-kid fallback."
        )
        return None
    if raw is None:
        return None
    return frozenset(key for key in raw.split(";") if key)


def compiled_opus_kids() -> frozenset[str] | None:
    """The module's ``family/gfx/cu_num/kid`` keys, or ``None`` if unknowable.

    ``None`` means the loaded binary predates the embedded manifest. Callers
    then keep their previous behaviour, because dropping every non-floor kid
    would silently slow down tuned rows that are compiled.
    """
    global _manifest
    manifest = _manifest
    if manifest is _UNSET:
        with _LOCK:
            if _manifest is _UNSET:
                _manifest = _load_manifest()
            manifest = _manifest
    return manifest


def _reset_cache() -> None:
    """Forget the manifest and the warned kids (tests)."""
    global _manifest
    with _LOCK:
        _manifest = _UNSET
        _warned.clear()


def _warn_uncompiled_kid(key: str) -> None:
    family, gfx, cu_num, kid = key.split("/")
    # The only tuner of this family; opus_gemm_a8w8_tune.py tunes other families.
    tuner = (
        "opus_gemm/opus_gemm_tune.py"
        if family == "a16w16"
        else "gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    )
    with _LOCK:
        if key in _warned:
            return
        _warned.add(key)
        logger.warning(
            f"OPUS kid {kid} ({family} on {gfx}, {cu_num} CUs) named by a tuned "
            "row is not compiled into the loaded module_deepgemm_opus; ignoring "
            "the row and using the next candidate or the default backend. To use "
            "the row, rebuild the module with that kid in its compile set: put "
            "the row in an in-tree tuned CSV and run once with AITER_REBUILD=1, "
            f"or benchmark it with csrc/{tuner}, which builds the kids it tunes."
        )


def tuned_kid_is_compiled(
    family: str, kid: int, device: torch.device | str | int | None = None
) -> bool:
    """Whether a tuned OPUS kid can be launched on ``device``; warns once if not.

    The key is built from the device's own arch and CU count, which is what the
    C++ dispatcher keys on, so a ``CU_NUM`` override in the tuned-row lookup
    cannot make an uncompiled (family, gfx, cu, kid) look compiled.
    """
    compiled = compiled_opus_kids()
    if compiled is None:
        return True
    gfx, cu_num = _device_arch_and_cu(
        torch.device("cuda") if device is None else device
    )
    key = f"{family}/{base_gfx_arch(gfx.lower())}/{int(cu_num)}/{int(kid)}"
    if key in compiled:
        return True
    _warn_uncompiled_kid(key)
    return False


__all__ = ["compiled_opus_kids", "opus_compiled_kids", "tuned_kid_is_compiled"]
