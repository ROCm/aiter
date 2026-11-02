# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Every environment variable MegaMoEV2 reads, in one place.

The gfx1250 stage-1 (S1F) switches and tuning overrides of mega_moe_gfx1250
are registered here too.

Read through this module (``envs.AITER_MEGA_...``) rather than ``os.environ``.
A value is read when it is accessed, so the caller decides when it takes effect:
at import, when a kernel is compiled, or when an instance is built.
"""

import os
from collections.abc import Callable
from typing import Any


def _flag(value: str) -> bool:
    return value == "1"


def _int_or(default: int) -> Callable[[str], int]:
    """An int; blank means ``default``."""
    return lambda value: int(value.strip() or default)


def _ints_or(default: tuple) -> Callable[[str], tuple]:
    """Comma-separated ints; blank means ``default``."""

    def parse(value: str) -> tuple:
        value = value.strip()
        return tuple(int(v) for v in value.split(",")) if value else default

    return parse


def _s1f_mode(value: str) -> str:
    # "" (unset: on by default), "0" (off) or "1" (explicitly on).
    value = value.strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"AITER_MEGA_S1F must be 0 or 1, got {value!r}")
    return value


def _s1f_buckets(value: str) -> tuple | None:
    # "N" sets all four tokens-per-rank buckets, "a,b,c,d" sets them in table
    # order (<=128 / <=1024 / <=2048 / above); blank (None) keeps the table.
    value = value.strip()
    if not value:
        return None
    vals = tuple(int(v) for v in value.split(","))
    return vals * 4 if len(vals) == 1 else vals


def _route_parallel(value: str) -> bool | None:
    # None (unset) keeps the token-major walk of the BLOCKS/WARPS override.
    return None if value == "" else value == "1"


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
    # gfx1250 (mega_moe_gfx1250). The S1F defaults come from DSV4.1-Flash DEP4
    # on MI455X, MLPerf Endpoints agentic c512: +2.1% gen tok/s/GPU over S1F
    # off. Each variable only overrides its default.
    # Dispatch and GEMM1 as one persistent stage-1 kernel: on by default, 0
    # turns it off, an explicit 1 also warns when a step falls back.
    "AITER_MEGA_S1F": (_s1f_mode, ""),
    # Compile every stage-1 plan variant before serving; 0 skips it.
    "AITER_S1F_PRECOMPILE": (lambda value: value != "0", "1"),
    # A step whose receive bound exceeds this runs the whole S1F=0 path, so
    # the widest prefill steps keep dispatch + GEMM1; 0 means no limit.
    # Read once, at import.
    "AITER_S1F_MAX_RECV": (_int_or(4096), "4096"),
    # Fuse whenever the tuned GEMM1's tile_m matches (the stage-1 kernel runs
    # its own 256x256 n/k tiles); 0 requires tile_m, tile_n and tile_k to all
    # match. Read once, at import.
    "AITER_S1F_ANY_TILE": (lambda value: value.strip() != "0", "1"),
    # Overrides of the S1F_PRODUCERS / S1F_BLOCK_RUNS tables in s1f_stage1.py
    # (blank keeps the table), e.g. AITER_S1F_BLOCK_RUNS=4,8,8,8. Read once,
    # at import.
    "AITER_S1F_PRODUCERS": (_s1f_buckets, ""),
    "AITER_S1F_BLOCK_RUNS": (_s1f_buckets, ""),
    # Compact dispatch schedule (see _DISPATCH_COMPACT in config.py), read
    # when the dispatch is built. Up to ROUTE_UPTO tokens per rank, the steps
    # past 256 walk routes with (block, warp) = ROUTE_SPEC; above it the tail
    # entry applies. ROUTE_UPTO=0 runs _DISPATCH_COMPACT unchanged.
    "AITER_TDM_COMPACT_ROUTE_UPTO": (_int_or(1024), "1024"),
    "AITER_TDM_COMPACT_ROUTE_SPEC": (_ints_or((384, 16)), "384,16"),
    # 0|1 picks the walk for the AITER_TDM_COMPACT_BLOCKS/WARPS override
    # (unset: token-major).
    "AITER_TDM_COMPACT_ROUTE_PARALLEL": (_route_parallel, ""),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        parse, default = environment_variables[name]
        return parse(os.environ.get(name, default))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)
