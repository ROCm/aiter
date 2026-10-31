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


def _flag(value: str) -> bool:
    return value == "1"


# name -> (parser, default)
environment_variables: dict[str, tuple[Callable[[str], Any], str]] = {
    # Largest MTPR still dispatched fixed-slot (direct expert slots, no count
    # exchange). 511 / 1023 / 2047 admit the 256 / 512 / 1024-token capacities,
    # validated on the EP8 v4_pro layout. Read once, at import.
    "AITER_MEGA_FIXED_SLOT_MAX_MTPR": (int, "255"),
    # Default of forward(mask_invalid_slots=...): combine skips top-k slots with
    # id -1. Off because the id check costs prefill time; callers that pad with
    # -1 ask for it per call.
    "AITER_MEGA_COMBINE_MASK": (_flag, "0"),
    # Load the AOT Stage1/Stage2 bundles when an instance is built.
    "AITER_MEGA_MOE_PRELOAD": (_flag, "0"),
    # Fallbacks for optimizations that are on by default (read per Stage1 compile):
    # GEMM1 quant epilogue columns per lane, 8 or 2.
    "AITER_MEGA_S1_EPI_EVEC": (int, "8"),
    # Paired K loop for fixed-slot GEMM1 tiles with LDS-DMA A copies.
    "AITER_MEGA_S1_FIXED_KPAIR": (_flag, "1"),
    # Compact dispatch producers keep two rows in flight.
    "AITER_MEGA_DISPATCH_FAST_COPY": (_flag, "1"),
    # Stage2 + combine through the SDMA copy engines (stage2_sdma) when an instance is
    # built with stage2_sdma=None. Needs mori with cco SDMA and MORI_ENABLE_SDMA=1;
    # unsupported configurations keep the default Stage2. Per instance, not per call
    # (every rank must take the same path): meant for prefill instances, as it is
    # faster from ~2048 tokens per rank and slower below ~1024 (fixed plan / SDMA
    # latency and full 256-row GEMM2 tiles).
    "AITER_MEGA_STAGE2_SDMA": (_flag, "0"),
    # SDMA Stage2: sort tiles per shipped chunk.
    "AITER_MEGA_SDMA_CHUNK_TILES": (int, "16"),
    # SDMA Stage2: rows reserved per (src, dst) pair, as a multiple of the even share
    # max_tok_per_rank * topk / world_size. Rows past it take the direct P2P path.
    "AITER_MEGA_SDMA_ROW_SLACK": (float, "2.0"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        parse, default = environment_variables[name]
        return parse(os.environ.get(name, default))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)
