# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""`MOEMetadata`: the stage callables and flags one dispatch decision produces.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
This module moved here from aiter/fused_moe.py and is shared with the legacy MoE path:
aiter.fused_moe re-exports it and legacy MoE calls run it. Changes must keep legacy
behavior (`python tests/moe_dispatch/snapshot.py --check`) and should loop in the moev2
owners (@qli88). See aiter/moev2/README.md.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from aiter.fused_moe_registry import BoundFusedMoeImpl


class AsmMoeMode(IntEnum):
    """ASM manifest modes; integer values are the CSV contract."""

    SORTED = 0
    NO_SORT = 1
    EMSORT = 2
    POST_REDUCE = 3

    @property
    def requires_host_sort(self) -> bool:
        return self is AsmMoeMode.SORTED or self is AsmMoeMode.POST_REDUCE


@dataclass
class MOEMetadata:
    stage1: Callable | None
    stage2: Callable | None
    block_m: int
    ksplit: int
    run_1stage: bool = False
    has_bias: bool = False
    use_non_temporal_load: bool = True
    fuse_quant: str = ""
    stage2_has_bias: bool = False
    flat: AsmMoeMode | int = AsmMoeMode.SORTED
    output_aux: bool | str = False
    prequant: bool = True
    skip_inter_quant: bool = False
    route_bucket: str = ""
    expected_sorted_blocks: int | None = None
    min_sorted_blocks: int | None = None
    max_sorted_blocks: int | None = None
    full_impl: BoundFusedMoeImpl | None = None
