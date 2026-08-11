# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Production code-generation metadata for Opus MoE backward."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class OpusMoeBackwardFamily(str, Enum):
    DOWN_BWD = "down_bwd"
    ROUTE_DX = "route_dx"
    ROUTE_REDUCE = "route_reduce"
    DW1 = "dw1"
    DW2 = "dw2"
    ROUTER_BWD = "router_bwd"
    BIAS_BWD = "bias_bwd"

    @property
    def macro_name(self) -> str:
        return self.value.upper()


class OpusMoeBackwardRouteLayout(str, Enum):
    """Layout used by a kernel's route-indexed inputs."""

    SORTED_ROUTE_MAJOR = "sorted_route_major"
    TOKEN_SLOT_MAJOR = "token_slot_major"
    COMPACT_ROUTE_MAJOR = "compact_route_major"


@dataclass(frozen=True)
class OpusMoeBackwardInstance:
    kid: int
    name: str
    family: OpusMoeBackwardFamily
    arch: str
    dtype: str
    route_layout: OpusMoeBackwardRouteLayout
    block_m: int
    block_n: int
    block_k: int
    block_threads: int
    min_blocks_per_cu: int
    has_oob: bool
    split_k: int
    trait: str
    launcher: str


# Keep only production kernels here.  Failed tuning candidates belong in
# benchmark logs, not the JIT manifest: every registered entry increases HIP
# compile time and can accidentally become an auto-dispatch target.
OPUS_MOE_BACKWARD_INSTANCES: tuple[OpusMoeBackwardInstance, ...] = (
    OpusMoeBackwardInstance(
        kid=2,
        name="down_bwd_bf16_gfx950_bm32_bn128_bk64_padded_wide",
        family=OpusMoeBackwardFamily.DOWN_BWD,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        block_m=32,
        block_n=128,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "DownBwdBf16Gfx950Bm32Bn128Bk64Padded"
        ),
        launcher="opus_moe_backward::gfx950::down_bwd_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=5,
        name="route_dx_bf16_gfx950_bm32_bn128_bk64_padded_wide",
        family=OpusMoeBackwardFamily.ROUTE_DX,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        block_m=32,
        block_n=128,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "RouteDxBf16Gfx950Bm32Bn128Bk64WideStore"
        ),
        launcher="opus_moe_backward::gfx950::route_dx_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=0,
        name="route_reduce_bf16_gfx950_bm16_bn128",
        family=OpusMoeBackwardFamily.ROUTE_REDUCE,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.TOKEN_SLOT_MAJOR,
        block_m=16,
        block_n=128,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "RouteReduceBf16Gfx950Bm16Bn128"
        ),
        launcher="opus_moe_backward::gfx950::route_reduce_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=5,
        name="dw1_bf16_gfx950_bm64_bn128_bk32_swizzled_wide",
        family=OpusMoeBackwardFamily.DW1,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        block_m=64,
        block_n=128,
        block_k=32,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "Dw1Bf16Gfx950Bm64Bn128Bk32Swizzled"
        ),
        launcher="opus_moe_backward::gfx950::dw1_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=3,
        name="dw2_bf16_gfx950_bm64_bn64_bk64_swizzled_wide",
        family=OpusMoeBackwardFamily.DW2,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        block_m=64,
        block_n=64,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled"
        ),
        launcher="opus_moe_backward::gfx950::dw2_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=0,
        name="router_bwd_fp32_gfx950_bm32_bn8",
        family=OpusMoeBackwardFamily.ROUTER_BWD,
        arch="gfx950",
        dtype="fp32",
        route_layout=OpusMoeBackwardRouteLayout.TOKEN_SLOT_MAJOR,
        block_m=32,
        block_n=8,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=4,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "RouterBwdF32Gfx950Bm32Bn8"
        ),
        launcher="opus_moe_backward::gfx950::router_bwd_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=0,
        name="bias_bwd_bf16_gfx950_bm32_bn16_r16",
        family=OpusMoeBackwardFamily.BIAS_BWD,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        block_m=32,
        block_n=16,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "BiasBwdBf16Gfx950Bm32Bn16R16"
        ),
        launcher="opus_moe_backward::gfx950::bias_bwd_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="down_bwd_varlen_bf16_gfx950_bm32_bn128_bk64_padded",
        family=OpusMoeBackwardFamily.DOWN_BWD,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=32,
        block_n=128,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "DownBwdVarlenBf16Gfx950Bm32Bn128Bk64Padded"
        ),
        launcher="opus_moe_backward::gfx950::down_bwd_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="route_dx_varlen_bf16_gfx950_bm32_bn128_bk64_wide",
        family=OpusMoeBackwardFamily.ROUTE_DX,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=32,
        block_n=128,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "RouteDxVarlenBf16Gfx950Bm32Bn128Bk64WideStore"
        ),
        launcher="opus_moe_backward::gfx950::route_dx_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="route_reduce_varlen_bf16_gfx950_bm16_bn128",
        family=OpusMoeBackwardFamily.ROUTE_REDUCE,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=16,
        block_n=128,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "RouteReduceVarlenBf16Gfx950Bm16Bn128"
        ),
        launcher="opus_moe_backward::gfx950::route_reduce_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="dw1_varlen_bf16_gfx950_bm64_bn128_bk32_swizzled",
        family=OpusMoeBackwardFamily.DW1,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=64,
        block_n=128,
        block_k=32,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "Dw1VarlenBf16Gfx950Bm64Bn128Bk32Swizzled"
        ),
        launcher="opus_moe_backward::gfx950::dw1_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="dw2_varlen_bf16_gfx950_bm64_bn64_bk64_swizzled",
        family=OpusMoeBackwardFamily.DW2,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=64,
        block_n=64,
        block_k=64,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=False,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::"
            "Dw2VarlenBf16Gfx950Bm64Bn64Bk64Swizzled"
        ),
        launcher="opus_moe_backward::gfx950::dw2_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="router_bwd_varlen_fp32_gfx950_bm32_bn8",
        family=OpusMoeBackwardFamily.ROUTER_BWD,
        arch="gfx950",
        dtype="fp32",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=32,
        block_n=8,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=4,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::RouterBwdVarlenF32Gfx950Bm32Bn8"
        ),
        launcher="opus_moe_backward::gfx950::router_bwd_launch_gfx950",
    ),
    OpusMoeBackwardInstance(
        kid=100,
        name="bias_bwd_varlen_bf16_gfx950_bm32_bn16_r16",
        family=OpusMoeBackwardFamily.BIAS_BWD,
        arch="gfx950",
        dtype="bf16",
        route_layout=OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        block_m=32,
        block_n=16,
        block_k=1,
        block_threads=256,
        min_blocks_per_cu=2,
        has_oob=True,
        split_k=1,
        trait=(
            "opus_moe_backward::gfx950::BiasBwdVarlenBf16Gfx950Bm32Bn16R16"
        ),
        launcher="opus_moe_backward::gfx950::bias_bwd_launch_gfx950",
    ),
)


_CPP_QUALIFIED_NAME = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*$"
)
_SUPPORTED_ARCHES = frozenset({"gfx950"})
_SUPPORTED_DTYPES = frozenset({"bf16", "fp32"})


def validate_instances(
    instances: Iterable[OpusMoeBackwardInstance],
) -> tuple[OpusMoeBackwardInstance, ...]:
    """Validate and return a deterministic tuple of kernel instances."""

    materialized = tuple(instances)
    seen_kids: dict[tuple[OpusMoeBackwardFamily, int], str] = {}
    seen_names: set[str] = set()

    for inst in materialized:
        if not isinstance(inst.family, OpusMoeBackwardFamily):
            raise ValueError(f"invalid family for {inst.name!r}: {inst.family!r}")
        if inst.kid < 0:
            raise ValueError(f"kid must be non-negative for {inst.name!r}")
        kid_key = (inst.family, inst.kid)
        if kid_key in seen_kids:
            raise ValueError(
                f"duplicate kid {inst.kid} in family {inst.family.value!r}: "
                f"{seen_kids[kid_key]!r} and {inst.name!r}"
            )
        if not inst.name:
            raise ValueError("kernel name must not be empty")
        if inst.name in seen_names:
            raise ValueError(f"duplicate kernel name {inst.name!r}")
        if inst.arch not in _SUPPORTED_ARCHES:
            raise ValueError(f"unsupported arch {inst.arch!r} for {inst.name!r}")
        if inst.dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported dtype {inst.dtype!r} for {inst.name!r}")
        if not isinstance(inst.route_layout, OpusMoeBackwardRouteLayout):
            raise ValueError(
                f"invalid route layout for {inst.name!r}: {inst.route_layout!r}"
            )
        if min(inst.block_m, inst.block_n, inst.block_k) <= 0:
            raise ValueError(f"tile sizes must be positive for {inst.name!r}")
        if inst.block_threads <= 0 or inst.block_threads > 1024:
            raise ValueError(
                f"block_threads must be in [1, 1024] for {inst.name!r}"
            )
        if inst.block_threads % 64 != 0:
            raise ValueError(
                f"gfx950 block_threads must be wave64-aligned for {inst.name!r}"
            )
        if inst.min_blocks_per_cu <= 0:
            raise ValueError(f"min_blocks_per_cu must be positive for {inst.name!r}")
        if inst.split_k <= 0:
            raise ValueError(f"split_k must be positive for {inst.name!r}")
        if not _CPP_QUALIFIED_NAME.fullmatch(inst.trait):
            raise ValueError(f"invalid C++ trait name {inst.trait!r}")
        if not _CPP_QUALIFIED_NAME.fullmatch(inst.launcher):
            raise ValueError(f"invalid C++ launcher name {inst.launcher!r}")

        seen_kids[kid_key] = inst.name
        seen_names.add(inst.name)

    return tuple(sorted(materialized, key=lambda x: (x.family.value, x.kid, x.name)))
