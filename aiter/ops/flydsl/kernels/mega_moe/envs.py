# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Every environment variable MegaMoEV2 reads, in one place.

Read through this module (``envs.AITER_MEGA_...``) rather than ``os.environ``.
A value is read when it is accessed, so the caller decides when it takes effect:
at import, when a kernel is compiled, or when an instance is built.
"""

import os
from collections.abc import Callable
from typing import Any


def _int(name: str, default: str) -> Callable[[], int]:
    return lambda: int(os.environ.get(name, default))


def _flag(name: str, default: str) -> Callable[[], bool]:
    return lambda: os.environ.get(name, default) == "1"


environment_variables: dict[str, Callable[[], Any]] = {
    # Largest MTPR still dispatched fixed-slot (direct expert slots, no count
    # exchange). 511 / 1023 / 2047 admit the 256 / 512 / 1024-token capacities,
    # validated on the EP8 v4_pro layout. Read once, at import.
    "AITER_MEGA_FIXED_SLOT_MAX_MTPR": _int("AITER_MEGA_FIXED_SLOT_MAX_MTPR", "255"),
    # Default of forward(mask_invalid_slots=...): combine skips top-k slots with
    # id -1. Off because the id check costs prefill time; callers that pad with
    # -1 ask for it per call.
    "AITER_MEGA_COMBINE_MASK": _flag("AITER_MEGA_COMBINE_MASK", "0"),
    # Load the AOT Stage1/Stage2 bundles when an instance is built.
    "AITER_MEGA_MOE_PRELOAD": _flag("AITER_MEGA_MOE_PRELOAD", "0"),
    # Fallbacks for optimizations that are on by default (read per Stage1 compile):
    # GEMM1 quant epilogue columns per lane, 8 or 2 (the previous pass).
    "AITER_MEGA_S1_EPI_EVEC": _int("AITER_MEGA_S1_EPI_EVEC", "8"),
    # Paired K loop for fixed-slot GEMM1 tiles with LDS-DMA A copies.
    "AITER_MEGA_S1_FIXED_KPAIR": _flag("AITER_MEGA_S1_FIXED_KPAIR", "1"),
    # Compact dispatch producers keep two rows in flight.
    "AITER_MEGA_DISPATCH_FAST_COPY": _flag("AITER_MEGA_DISPATCH_FAST_COPY", "1"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)
