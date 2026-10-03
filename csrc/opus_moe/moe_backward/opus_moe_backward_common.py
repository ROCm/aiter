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


# Only production auto targets and legal layout/geometry fallbacks are compiled.
# Retained IDs stay stable; retired comparison IDs fail dispatch.
def _instance(kid, name, family, layout, tile, threads, residency, oob, trait, dtype="bf16"):
    return OpusMoeBackwardInstance(
        kid=kid, name=name, family=family, arch="gfx950", dtype=dtype,
        route_layout=layout, block_m=tile[0], block_n=tile[1], block_k=tile[2],
        block_threads=threads, min_blocks_per_cu=residency, has_oob=oob, split_k=1,
        trait=f"opus_moe_backward::gfx950::{trait}",
        launcher=f"opus_moe_backward::gfx950::{family.value}_launch_gfx950",
    )


OPUS_MOE_BACKWARD_INSTANCES: tuple[OpusMoeBackwardInstance, ...] = (
    _instance(
        2, "down_bwd_bf16_gfx950_bm32_bn128_bk64_padded_wide",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 64), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn128Bk64Padded",
    ),
    _instance(
        9, "down_bwd_bf16_gfx950_bm32_bn128_bk32_m5_predecoded",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM5Cohort20Predecoded",
    ),
    _instance(
        10, "down_bwd_bf16_gfx950_bm32_bn128_bk32_m6_predecoded",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM6Cohort24Predecoded",
    ),
    _instance(
        11, "down_bwd_bf16_gfx950_bm32_bn256_bk32_m6_predecoded",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32PaddedM6Cohort24Predecoded",
    ),
    _instance(
        12, (
            "down_bwd_bf16_gfx950_bm32_bn256_bk32_m6_predecoded_"
            "deferred_z_wait"
        ),
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32PaddedM6Cohort24PredecodedDeferredZWait",
    ),
    _instance(
        13, (
            "down_bwd_bf16_gfx950_bm32_bn256_bk32_m6_"
            "saved_a_scaled"
        ),
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32M6SavedAScaled",
    ),
    _instance(
        16, (
            "down_bwd_bf16_gfx950_bm32_bn256_bk32_m6_split_bn64_"
            "pipelined_z_saved_a_scaled"
        ),
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32M6SplitBN64PipelinedZSavedAScaled",
    ),
    _instance(
        18, "down_bwd_bf16_gfx950_bn256_m6_blocked_dz_g2_sparse_owner",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32M6BlockedDzG2SparseOwner",
    ),
    _instance(
        19, "down_bwd_bf16_gfx950_bn256_m6_blocked_dz_g2_mubuf_store",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "DownBwdBf16Gfx950Bm32Bn256Bk32M6BlockedDzG2SparseOwnerMubufStore",
    ),
    _instance(
        5, "route_dx_bf16_gfx950_bm32_bn128_bk64_padded_wide",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 64), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn128Bk64WideStore",
    ),
    _instance(
        7, "route_dx_bf16_gfx950_bm32_bn128_bk32_cohort4",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreCohort4",
    ),
    _instance(
        9, "route_dx_bf16_gfx950_bm32_bn128_bk32_m5_cohort10_aslab_pad",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPad",
    ),
    _instance(
        0, "route_reduce_bf16_gfx950_bm16_bn128",
        OpusMoeBackwardFamily.ROUTE_REDUCE,
        OpusMoeBackwardRouteLayout.TOKEN_SLOT_MAJOR,
        (16, 128, 1), 256, 2, True,
        "RouteReduceBf16Gfx950Bm16Bn128",
    ),
    _instance(
        11, "route_dx_bf16_gfx950_bm32_bn128_bk32_m5_cohort10_aslab_pad_sorted_output",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 128, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPadSortedOutput",
    ),
    _instance(
        13, "route_dx_bf16_gfx950_bm32_bn256_bk32_m3_cohort6",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPad",
    ),
    _instance(
        14, "route_dx_bf16_gfx950_bm32_bn256_bk32_m3_cohort6_sorted",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPadSortedOutput",
    ),
    _instance(
        15, (
            "route_dx_bf16_gfx950_bm32_bn256_bk32_m3_cohort6_"
            "sorted_b_first"
        ),
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPadSortedOutputBFirst",
    ),
    _instance(
        16, "route_dx_bf16_gfx950_bm32_bn256_bk32_m5_cohort10_sorted_b_first",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM5Cohort10ASlabPadSortedOutputBFirst",
    ),
    _instance(
        17, (
            "route_dx_bf16_gfx950_bm32_bn256_bk32_m5_binary_compact_"
            "cohort10_sorted_b_first"
        ),
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 256, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM5BinaryCompactCohort10ASlabPadSortedOutputBFirst",
    ),
    _instance(
        18, (
            "route_dx_bf16_gfx950_bm32_bn512_bk32_m3_binary_compact_"
            "cohort6_sorted_b_first"
        ),
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 512, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn512Bk32WideStoreM3BinaryCompactCohort6ASlabPadSortedOutputBFirst",
    ),
    _instance(
        19, (
            "route_dx_bf16_gfx950_bm32_bn512_bk32_m3_binary_compact_"
            "cohort6_sorted_b_first_n_fast"
        ),
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 512, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn512Bk32WideStoreM3BinaryCompactCohort6ASlabPadSortedOutputBFirstNFast",
    ),
    _instance(
        20, "route_dx_bf16_gfx950_bn512_m3_n_fast_blocked_dz_g2",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 512, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn512Bk32WideStoreM3BinaryCompactCohort6ASlabPadSortedOutputBFirstNFastBlockedDzG2",
    ),
    _instance(
        21, "route_dx_bf16_gfx950_bn512_m3_blocked_g2_mubuf_store",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 512, 32), 256, 2, False,
        "RouteDxBf16Gfx950Bm32Bn512Bk32WideStoreM3BinaryCompactCohort6ASlabPadSortedOutputBFirstNFastBlockedDzG2MubufStore",
    ),
    _instance(
        1, "route_reduce_bf16_gfx950_bm16_bn128_sorted_input",
        OpusMoeBackwardFamily.ROUTE_REDUCE,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (16, 128, 1), 256, 2, True,
        "RouteReduceBf16Gfx950Bm16Bn128SortedInput",
    ),
    _instance(
        3, (
            "route_reduce_bf16_gfx950_bm1_bn2048_sorted_input_"
            "distributed_ids"
        ),
        OpusMoeBackwardFamily.ROUTE_REDUCE,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (1, 2048, 1), 256, 2, True,
        "RouteReduceBf16Gfx950Bm1Bn2048SortedInputDistributedIds",
    ),
    _instance(
        5, "dw1_bf16_gfx950_bm64_bn128_bk32_swizzled_wide",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 128, 32), 256, 2, False,
        "Dw1Bf16Gfx950Bm64Bn128Bk32Swizzled",
    ),
    _instance(
        25, "dw1_bf16_gfx950_bm256_blocked_g2_factored_soffset_grid3d",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4BlockedG2FactoredSoffsetGrid3D",
    ),
    _instance(
        18, (
            "dw1_bf16_gfx950_bm256_bn128_bk32_wave4_reverse_cohort4_"
            "prefetch_ab_eager_sorted_x_b_first_triple_lds"
        ),
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4PrefetchABEagerSortedXBFirstTripleLds",
    ),
    _instance(
        15, (
            "dw1_bf16_gfx950_bm256_bn128_bk32_wave4_reverse_cohort4_"
            "prefetch_ab_sorted_x_b_first_double_lds"
        ),
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4PrefetchABSortedXBFirstDoubleLds",
    ),
    _instance(
        14, (
            "dw1_bf16_gfx950_bm256_bn128_bk32_wave4_reverse_cohort4_"
            "prefetch_a_sorted_x_double_lds"
        ),
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4PrefetchASortedXDoubleLds",
    ),
    _instance(
        11, "dw1_bf16_gfx950_bm256_bn128_bk32_wave4_cohort2_double_lds",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4Cohort2DoubleLds",
    ),
    _instance(
        13, (
            "dw1_bf16_gfx950_bm256_bn128_bk32_wave4_reverse_cohort4_"
            "prefetch_a_double_lds"
        ),
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4PrefetchADoubleLds",
    ),
    _instance(
        8, "dw1_bf16_gfx950_bm64_bn128_bk32_swizzled_cohort2_direct_lds",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 128, 32), 256, 2, False,
        "Dw1Bf16Gfx950Bm64Bn128Bk32SwizzledCohort2DirectLds",
    ),
    _instance(
        9, "dw1_bf16_gfx950_bm128_bn128_bk32_cohort2_double_lds",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (128, 128, 32), 256, 2, False,
        "Dw1Bf16Gfx950Bm128Bn128Bk32SwizzledCohort2DoubleLds",
    ),
    _instance(
        3, "dw2_bf16_gfx950_bm64_bn64_bk64_swizzled_wide",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 64, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled",
    ),
    _instance(
        18, "dw2_bf16_gfx950_bm128_bn256_bk32_cohort4_grid3d",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (128, 256, 32), 256, 1, False,
        "Dw2Bf16Gfx950Bm128Bn256Bk32Cohort4Grid3D",
    ),
    _instance(
        12, (
            "dw2_bf16_gfx950_bm256_bn128_bk32_"
            "cohort4_triple_lds"
        ),
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw2Bf16Gfx950Bm256Bn128Bk32Cohort4TripleLds",
    ),
    _instance(
        13, (
            "dw2_bf16_gfx950_bm256_bn128_bk32_"
            "cohort4_prefetch_ab_triple_lds"
        ),
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw2Bf16Gfx950Bm256Bn128Bk32Cohort4PrefetchABTripleLds",
    ),
    _instance(
        16, (
            "dw2_bf16_gfx950_bm256_bn128_bk32_"
            "cohort4_native_b32_zero_pad_triple_lds"
        ),
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 32), 256, 1, False,
        "Dw2Bf16Gfx950Bm256Bn128Bk32Cohort4NativeB32ZeroPadTripleLds",
    ),
    _instance(
        11, (
            "dw2_bf16_gfx950_bm256_bn128_bk64_swizzled_"
            "cohort4_direct_lds"
        ),
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (256, 128, 64), 256, 1, False,
        "Dw2Bf16Gfx950Bm256Bn128Bk64SwizzledCohort4DualLdsWave2x2",
    ),
    _instance(
        10, (
            "dw2_bf16_gfx950_bm128_bn128_bk64_swizzled_"
            "adaptive_routes"
        ),
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (128, 128, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledAdaptiveRoutes",
    ),
    _instance(
        5, "dw2_bf16_gfx950_bm64_bn64_bk64_swizzled_cohort4_direct_lds",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 64, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort4DirectLds",
    ),
    _instance(
        6, "dw2_bf16_gfx950_bm64_bn64_bk64_swizzled_cohort1_direct_lds",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 64, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort1DirectLds",
    ),
    _instance(
        7, "dw2_bf16_gfx950_bm64_bn64_bk64_swizzled_cohort2_direct_lds",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (64, 64, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort2DirectLds",
    ),
    _instance(
        9, "dw2_bf16_gfx950_bm128_bn64_bk64_swizzled_cohort1_dual_lds",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (128, 64, 64), 256, 2, False,
        "Dw2Bf16Gfx950Bm128Bn64Bk64SwizzledCohort1DualLds",
    ),
    _instance(
        0, "router_bwd_fp32_gfx950_bm32_bn8",
        OpusMoeBackwardFamily.ROUTER_BWD,
        OpusMoeBackwardRouteLayout.TOKEN_SLOT_MAJOR,
        (32, 8, 1), 256, 4, True,
        "RouterBwdF32Gfx950Bm32Bn8", dtype="fp32",
    ),
    _instance(
        0, "bias_bwd_bf16_gfx950_bm32_bn16_r16",
        OpusMoeBackwardFamily.BIAS_BWD,
        OpusMoeBackwardRouteLayout.SORTED_ROUTE_MAJOR,
        (32, 16, 1), 256, 2, True,
        "BiasBwdBf16Gfx950Bm32Bn16R16",
    ),
    _instance(
        100, "down_bwd_varlen_bf16_gfx950_bm32_bn128_bk64_padded",
        OpusMoeBackwardFamily.DOWN_BWD,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (32, 128, 64), 256, 2, False,
        "DownBwdVarlenBf16Gfx950Bm32Bn128Bk64Padded",
    ),
    _instance(
        100, "route_dx_varlen_bf16_gfx950_bm32_bn128_bk64_wide",
        OpusMoeBackwardFamily.ROUTE_DX,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (32, 128, 64), 256, 2, False,
        "RouteDxVarlenBf16Gfx950Bm32Bn128Bk64WideStore",
    ),
    _instance(
        100, "route_reduce_varlen_bf16_gfx950_bm16_bn128",
        OpusMoeBackwardFamily.ROUTE_REDUCE,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (16, 128, 1), 256, 2, True,
        "RouteReduceVarlenBf16Gfx950Bm16Bn128",
    ),
    _instance(
        100, "dw1_varlen_bf16_gfx950_bm64_bn128_bk32_swizzled",
        OpusMoeBackwardFamily.DW1,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (64, 128, 32), 256, 2, False,
        "Dw1VarlenBf16Gfx950Bm64Bn128Bk32Swizzled",
    ),
    _instance(
        100, "dw2_varlen_bf16_gfx950_bm64_bn64_bk64_swizzled",
        OpusMoeBackwardFamily.DW2,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (64, 64, 64), 256, 2, False,
        "Dw2VarlenBf16Gfx950Bm64Bn64Bk64Swizzled",
    ),
    _instance(
        100, "router_bwd_varlen_fp32_gfx950_bm32_bn8",
        OpusMoeBackwardFamily.ROUTER_BWD,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (32, 8, 1), 256, 4, True,
        "RouterBwdVarlenF32Gfx950Bm32Bn8", dtype="fp32",
    ),
    _instance(
        100, "bias_bwd_varlen_bf16_gfx950_bm32_bn16_r16",
        OpusMoeBackwardFamily.BIAS_BWD,
        OpusMoeBackwardRouteLayout.COMPACT_ROUTE_MAJOR,
        (32, 16, 1), 256, 2, True,
        "BiasBwdVarlenBf16Gfx950Bm32Bn16R16",
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
