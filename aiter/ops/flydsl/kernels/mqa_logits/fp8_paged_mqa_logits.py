# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged FP8 MQA logits (decode), dispatched by architecture.

gfx950 runs the H32/D128/KVB64 ragged LDS kernel in
``fp8_paged_mqa_logits_gfx950``; gfx1250 the WMMA kernel in
``fp8_paged_mqa_logits_gfx1250`` (any 16-multiple head count, 8- to 256-token
pages, the packed cache ``deepgemm_fp8_paged_mqa_logits`` reads). Only the
running arch's module is imported.
"""

import functools
import importlib

from aiter.jit.utils.chip_info import get_gfx

# gfx -> (module, entry point); any other arch keeps the gfx950 kernel.
_KERNELS = {
    "gfx1250": (
        "fp8_paged_mqa_logits_gfx1250",
        "flydsl_fp8_paged_mqa_logits_gfx1250",
    ),
}
_DEFAULT = ("fp8_paged_mqa_logits_gfx950", "flydsl_fp8_paged_mqa_logits")


@functools.cache
def _kernel(gfx):
    module, name = _KERNELS.get(gfx, _DEFAULT)
    return getattr(importlib.import_module(f".{module}", __package__), name)


def flydsl_fp8_paged_mqa_logits(*args, **kwargs):
    """Paged FP8 MQA logits; see the per-architecture entry points."""
    return _kernel(get_gfx())(*args, **kwargs)


def __getattr__(name):
    # The per-arch entry points, imported on first access.
    if name == "flydsl_fp8_paged_mqa_logits_gfx950":
        from .fp8_paged_mqa_logits_gfx950 import flydsl_fp8_paged_mqa_logits_gfx950

        return flydsl_fp8_paged_mqa_logits_gfx950
    if name == "flydsl_fp8_paged_mqa_logits_gfx1250":
        from .fp8_paged_mqa_logits_gfx1250 import flydsl_fp8_paged_mqa_logits_gfx1250

        return flydsl_fp8_paged_mqa_logits_gfx1250
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The per-arch entry points stay out of __all__: they only exist via
# __getattr__, so a star import would load every arch's module.
__all__ = ["flydsl_fp8_paged_mqa_logits"]
