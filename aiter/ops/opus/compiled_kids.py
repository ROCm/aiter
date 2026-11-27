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
"""

from __future__ import annotations

import functools

from aiter import logger

from ...jit.core import compile_ops
from ...jit.utils.build_targets import base_gfx_arch


@compile_ops("module_deepgemm_opus", fc_name="opus_compiled_kids")
def opus_compiled_kids() -> str: ...


@functools.cache
def compiled_opus_kids() -> frozenset[str] | None:
    """The module's ``family/gfx/cu_num/kid`` keys, or ``None`` if unknowable.

    ``None`` means the loaded binary predates the embedded manifest. Callers
    then keep their previous behaviour, because dropping every non-floor kid
    would silently slow down tuned rows that are compiled.
    """
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


@functools.cache
def _warn_uncompiled_kid(key: str) -> None:
    family, gfx, cu_num, kid = key.split("/")
    tuner = "opus_gemm_tune.py" if family == "a16w16" else "opus_gemm_a8w8_tune.py"
    logger.warning(
        f"OPUS kid {kid} ({family} on {gfx}, {cu_num} CUs) named by a tuned row "
        "is not compiled into the loaded module_deepgemm_opus; ignoring the row "
        "and using the next candidate or the default backend. To use the row, "
        "rebuild the module with that kid in its compile set: put the row in an "
        "in-tree tuned CSV and run once with AITER_REBUILD=1, or benchmark it "
        f"with csrc/opus_gemm/{tuner}, which builds the kids it tunes."
    )


def tuned_kid_is_compiled(family: str, gfx: str, cu_num: int, kid: int) -> bool:
    """Whether a tuned OPUS kid can be launched; warns once when it cannot."""
    compiled = compiled_opus_kids()
    if compiled is None:
        return True
    key = f"{family}/{base_gfx_arch(str(gfx).lower())}/{int(cu_num)}/{int(kid)}"
    if key in compiled:
        return True
    _warn_uncompiled_kid(key)
    return False


__all__ = ["compiled_opus_kids", "opus_compiled_kids", "tuned_kid_is_compiled"]
