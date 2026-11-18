# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
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
    # Fixed-slot Stage1 starts each GEMM1 tile once its expert's rows have
    # landed instead of after every source finishes dispatch: sources send
    # per-expert route counts ahead of the payload so the plan is built early,
    # and count their published rows per (expert, source) (read per Stage1
    # compile).
    "AITER_MEGA_S1_FIXED_EARLY_TILES": (_flag, "0"),
    # Fixed-slot Stage1 also runs GEMM2 and the combine scatter: once a CTA
    # finds no GEMM1 work left it takes Stage2's output tiles, each as soon as
    # its M tile's GEMM1 work items are done, and the separate Stage2 launch is
    # skipped (read when an instance is built and per Stage1 compile).
    "AITER_MEGA_S1_FIXED_FUSED_STAGE2": (_flag, "0"),
    # On mori's uncached heap, fixed-slot producers publish their counts and
    # done flags after their stores drain instead of after a system release
    # (an L2 writeback under the running GEMM).  Read per Stage1 compile.
    "AITER_MEGA_S1_FIXED_LIGHT_RELEASE": (_flag, "0"),
    # Fixed-slot Stage1 launches num_cu * grid_mult CTAs instead of adding the
    # owner and producers on top: they are the first resident CTAs (arrival
    # tickets) and run GEMM work after their role, so no consumer waits for a
    # producer to exit before it becomes resident (read per Stage1 compile).
    "AITER_MEGA_S1_FIXED_RESIDENT_GRID": (_flag, "0"),
    # V4.1-Flash EP8 A4W4 fixed-slot decode GEMM1 on the fused-MoE recipe's tuned
    # 2048-token tile (flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w3_fp4): sort_block_m
    # 64, waves_per_eu 3, b_nt 2, instead of t32x128x256 at waves_per_eu 2, b_nt 3.
    # Read when the config is selected.
    "AITER_MEGA_S1_FIXED_TUNED_GEMM1": (_flag, "0"),
    # With TUNED_GEMM1, GEMM1 takes the fused-MoE kernel's whole 64 x 256
    # gate+up workgroup tile on two consumer CTAs per CU (tile_n 256,
    # grid_mult 2, waves_per_eu 2) and runs its K loop fully unrolled.
    # Read when the config is selected and per Stage1 compile.
    "AITER_MEGA_S1_FIXED_TUNED_GEMM1_WIDE": (_flag, "0"),
    # With TUNED_GEMM1_WIDE and the fused GEMM2, GEMM1 items past one round of
    # consumer CTAs leave a partial last round of whole 64 x 256 items; the last
    # 192 of them (at most that overflow) run as two 64 x 128 items each, so the
    # tail is spread over every CTA. Read per Stage1 compile.
    "AITER_MEGA_S1_FIXED_TUNED_GEMM1_V2": (_flag, "0"),
    # With AITER_MEGA_S1_FIXED_TUNED_GEMM1, the fused GEMM2 takes Stage1's 64-row
    # sort blocks whole (block_m 64, the fused-MoE recipe's t64x128x256 Stage2)
    # instead of two 32-row blocks each streaming the expert's W2.
    # Read when the config is selected.
    "AITER_MEGA_S2_FIXED_TUNED_GEMM2": (_flag, "0"),
    # The fused GEMM2's combine scatter keeps every lane busy (four rows a wave
    # at once, each lane group with its own peer address) instead of one row a
    # wave on 16 lanes. Read per Stage1 compile.
    "AITER_MEGA_S2_FIXED_TUNED_GEMM2_V2": (_flag, "0"),
    # mori's symmetric heap type.  Unset (mori's default) or "uncached" keeps
    # the dispatch payload out of L2, so an early tile only drops stale L1
    # lines before it reads its rows; any other value takes a system acquire
    # per tile instead.
    "MORI_SHMEM_HEAP_TYPE": (str, "uncached"),
    # Compact dispatch producers keep two rows in flight.
    "AITER_MEGA_DISPATCH_FAST_COPY": (_flag, "1"),
    # Small- and large-input paths for the EP8 DeepSeek-V4 a8w4 layout (gfx950), read when an
    # instance is built. Small: single-kernel runtimes next to the main path (forward_small_m).
    "AITER_MEGA_SMALL_M": (_flag, "0"),
    # Large: quantization plus one persistent kernel for large per-rank inputs.
    "AITER_MEGA_LARGE_M": (_flag, "0"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        parse, default = environment_variables[name]
        return parse(os.environ.get(name, default))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)
