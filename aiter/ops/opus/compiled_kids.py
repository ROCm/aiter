# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which OPUS kernel ids the loaded ``module_deepgemm_opus`` really contains.

``module_deepgemm_opus`` is a subset build: the heuristic floor, the kids that
the in-tree tuned CSVs name at build time and ``compiled_kids_opus.json``. A
tuned row can therefore pass the Python registry checks and still name a kid
that this binary lacks, and the C++ launcher then raises ``kid N is not
compiled in the current OPUS module`` inside the serving process.

The codegen embeds a per-device launchability manifest of ``family/gfx/cu_num/kid``
keys in the binary (``opus_compiled_kids()``), so a prebuilt module carries its
own. It is derived from the selected routes (the sidecar), with a separate
``gfx1250-strict`` namespace and without kids that have no device code. Callers
that select a *tuned* kid ask :func:`tuned_kid_is_compiled`
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
from ...jit.utils.chip_info import get_gfx_runtime
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


def _device_namespace(device) -> tuple[str, int]:
    """The (arch, CU count) the manifest is keyed by for ``device``.

    The CU count is the physical one. The arch is the device's, except that a
    gfx1250-strict (MI450 A0) device is reported by torch as plain ``gfx1250``:
    it takes the ``gfx1250-strict`` namespace, whose manifest lists only the kids
    that have device code for a strict device (see ``gen_compiled_kids_header``).
    """
    gfx, cu_num = _device_arch_and_cu(
        torch.device("cuda") if device is None else device
    )
    gfx = gfx.lower()
    if base_gfx_arch(gfx) == "gfx1250" and gfx != "gfx1250-strict":
        try:
            live = get_gfx_runtime().lower()
        except Exception:  # noqa: BLE001 -- no rocminfo: keep the torch arch
            live = gfx
        if live == "gfx1250-strict":
            gfx = live
    return gfx, int(cu_num)


def tuned_kid_is_compiled(
    family: str, kid: int, device: torch.device | str | int | None = None
) -> bool:
    """Whether a tuned OPUS kid can be launched on ``device``; warns once if not.

    The key is built from the device's own arch and CU count, which is what the
    C++ dispatcher keys on, so a ``CU_NUM`` override in the tuned-row lookup
    cannot make an uncompiled (family, gfx, cu, kid) look compiled. Pass the
    device the GEMM will run on; ``None`` means the current device, which is only
    right for a caller that has no tensor.
    """
    compiled = compiled_opus_kids()
    if compiled is None:
        return True
    gfx, cu_num = _device_namespace(device)
    # TODO(#8152, #8237): those PRs key the C++ tables by ``opus_namespace_cu``,
    # so a compute partition (an unlisted CU count) resolves through the
    # full-chip table. Once they merge, ``cu_num`` here must go through the same
    # mapping, or a partition would be judged against a table C++ never reads.
    # Coordinate with that change; not changed here.
    key = f"{family}/{gfx}/{cu_num}/{int(kid)}"
    if key in compiled:
        return True
    _warn_uncompiled_kid(key)
    return False


__all__ = ["compiled_opus_kids", "opus_compiled_kids", "tuned_kid_is_compiled"]
