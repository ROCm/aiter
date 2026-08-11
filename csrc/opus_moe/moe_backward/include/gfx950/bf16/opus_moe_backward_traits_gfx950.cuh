// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_backward_common.cuh"
#include "opus/opus.hpp"

namespace opus_moe_backward::gfx950
{

template<Family FamilyValue,
         int BlockM,
         int BlockN,
         int BlockK,
         int BlockThreads,
         int MinBlocksPerCu,
         bool HasOob,
         RouteLayout Layout = RouteLayout::SortedRouteMajor>
struct Bf16Traits
{
    static constexpr Family FAMILY = FamilyValue;
    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K = BlockK;
    static constexpr int BLOCK_SIZE = BlockThreads;
    static constexpr int MIN_BLOCKS_PER_CU = MinBlocksPerCu;
    static constexpr bool HAS_OOB = HasOob;
    static constexpr int SPLIT_K = 1;
    static constexpr RouteLayout ROUTE_LAYOUT = Layout;
    static_assert(B_M > 0 && B_N > 0 && B_K > 0);
    static_assert(BLOCK_SIZE > 0 && BLOCK_SIZE <= 1024);
    static_assert(BLOCK_SIZE % 64 == 0, "gfx950 executes wave64");
    static_assert(MIN_BLOCKS_PER_CU > 0);
};

// K1: 32 sorted routes x 128 intermediate columns x K64.  Two asynchronous
// stages contain a 4 KiB dO tile and a W2 tile padded by 64 bytes per four
// rows, matching the gfx950 padded_shared transpose-read lowering.
struct DownBwdBf16Gfx950Bm32Bn128Bk64Padded
    : Bf16Traits<Family::DownBwd, 32, 128, 64, 256, 2, false>
{
    static constexpr int ROUTE_M = 32;
    static constexpr int T_M = 1;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 1;
    static constexpr int E_N = 1;
    static constexpr int E_K = 4;
    static constexpr int VEC_A = 16 / sizeof(D_A);
    static constexpr int VEC_B = 16 / sizeof(D_B);
    static constexpr int VEC_TR_B = 8 / sizeof(D_B);
    static constexpr int VEC_C = 4;

    static constexpr int SMEM_B_GROUP_ROWS = 4;
    static constexpr int SMEM_B_ROW_BYTES = B_N * sizeof(D_B);
    static constexpr int SMEM_B_GROUP_DATA_BYTES =
        SMEM_B_GROUP_ROWS * SMEM_B_ROW_BYTES;
    static constexpr int SMEM_B_GROUP_PAD_BYTES = 64;
    static constexpr int SMEM_B_GROUP_BYTES =
        SMEM_B_GROUP_DATA_BYTES + SMEM_B_GROUP_PAD_BYTES;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES = SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = 0;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
    static_assert(SMEM_B_GROUP_DATA_BYTES == 1024);
    static_assert(SMEM_B_GROUPS == 16);
};

// K2: gathered dZ x W1, retaining Triton's 32x128x64 two-stage geometry.
struct RouteDxBf16Gfx950Bm32Bn128Bk64WideStore
    : Bf16Traits<Family::RouteDx, 32, 128, 64, 256, 2, false>
{
    static constexpr int ROUTE_M = 32;
    static constexpr int T_M = 1;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 1;
    static constexpr int E_N = 1;
    static constexpr int E_K = 4;
    static constexpr int VEC_A = 16 / sizeof(D_A);
    static constexpr int VEC_B = 16 / sizeof(D_B);
    static constexpr int VEC_TR_B = 8 / sizeof(D_B);
    static constexpr int VEC_C = 4;

    static constexpr int SMEM_B_GROUP_ROWS = 4;
    static constexpr int SMEM_B_ROW_BYTES = B_N * sizeof(D_B);
    static constexpr int SMEM_B_GROUP_DATA_BYTES =
        SMEM_B_GROUP_ROWS * SMEM_B_ROW_BYTES;
    static constexpr int SMEM_B_GROUP_PAD_BYTES = 64;
    static constexpr int SMEM_B_GROUP_BYTES =
        SMEM_B_GROUP_DATA_BYTES + SMEM_B_GROUP_PAD_BYTES;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES = SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = 0;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
    static_assert(SMEM_B_GROUP_DATA_BYTES == 1024);
    static_assert(SMEM_B_GROUPS == 16);
};

struct RouteReduceBf16Gfx950Bm16Bn128
    : Bf16Traits<Family::RouteReduce,
                 16,
                 128,
                 1,
                 256,
                 2,
                 true,
                 RouteLayout::TokenSlotMajor>
{
    static constexpr int VEC = 8;
    static constexpr int MAX_TOPK = 8;
};

// Router scores and their gradients remain FP32 even though the surrounding
// expert path is BF16.  Eight expert columns per token exactly match the first
// production E=8 shape; larger E values use a second grid dimension.
struct RouterBwdF32Gfx950Bm32Bn8
    : Bf16Traits<Family::RouterBwd,
                 32,
                 8,
                 1,
                 256,
                 4,
                 true,
                 RouteLayout::TokenSlotMajor>
{
    static constexpr int MAX_TOPK = 8;
};

struct BiasBwdBf16Gfx950Bm32Bn16R16
    : Bf16Traits<Family::BiasBwd,
                 32,
                 16,
                 1,
                 256,
                 2,
                 true>
{
    static constexpr int DSCORE_GROUP_SIZE = 8;
    static constexpr int DB_ROUTE_GROUP_SIZE = 16;
    static constexpr int MAX_TOPK = 8;
    static_assert(B_N * DB_ROUTE_GROUP_SIZE == BLOCK_SIZE);
};

// Compact variable-routing variants reuse the production math pipeline while
// selecting an unambiguous route-id decoder at compile time.
struct DownBwdVarlenBf16Gfx950Bm32Bn128Bk64Padded
    : DownBwdBf16Gfx950Bm32Bn128Bk64Padded
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

struct RouteDxVarlenBf16Gfx950Bm32Bn128Bk64WideStore
    : RouteDxBf16Gfx950Bm32Bn128Bk64WideStore
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

struct RouteReduceVarlenBf16Gfx950Bm16Bn128
    : RouteReduceBf16Gfx950Bm16Bn128
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

struct RouterBwdVarlenF32Gfx950Bm32Bn8 : RouterBwdF32Gfx950Bm32Bn8
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

struct BiasBwdVarlenBf16Gfx950Bm32Bn16R16
    : BiasBwdBf16Gfx950Bm32Bn16R16
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

// K4: dZ^T x X, 64x128 output with K32 and one swizzled LDS allocation.
struct Dw1Bf16Gfx950Bm64Bn128Bk32Swizzled
    : Bf16Traits<Family::Dw1, 64, 128, 32, 256, 2, false>
{
    static constexpr int T_M = 1;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 2;
    static constexpr int E_N = 1;
    static constexpr int E_K = 2;
    static constexpr int VEC_A = 16 / sizeof(D_A);
    static constexpr int VEC_B = 16 / sizeof(D_B);
    static constexpr int VEC_TR_B = 8 / sizeof(D_B);
    static constexpr int VEC_C = 4;

    static constexpr int SMEM_B_GROUP_ROWS = 0;
    static constexpr int SMEM_B_ROW_BYTES = 0;
    static constexpr int SMEM_B_GROUP_DATA_BYTES = 0;
    static constexpr int SMEM_B_GROUP_PAD_BYTES = 0;
    static constexpr int SMEM_B_GROUP_BYTES = 0;
    static constexpr int SMEM_B_GROUPS = 0;
    static constexpr int SMEM_B_BYTES = B_N * B_K * sizeof(D_B);

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = 0;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
};

// K5: dO^T x (S*A), 64x64 output with K64 and swizzled LDS reuse.
struct Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled
    : Bf16Traits<Family::Dw2, 64, 64, 64, 256, 2, false>
{
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 1;
    static constexpr int E_N = 1;
    static constexpr int E_K = 4;
    static constexpr int VEC_A = 16 / sizeof(D_A);
    static constexpr int VEC_B = 16 / sizeof(D_B);
    static constexpr int VEC_TR_B = 8 / sizeof(D_B);
    static constexpr int VEC_C = 4;

    static constexpr int SMEM_B_GROUP_ROWS = 4;
    static constexpr int SMEM_B_ROW_BYTES = B_N * sizeof(D_B);
    static constexpr int SMEM_B_GROUP_DATA_BYTES = 0;
    static constexpr int SMEM_B_GROUP_PAD_BYTES = 0;
    static constexpr int SMEM_B_GROUP_BYTES = 0;
    static constexpr int SMEM_B_GROUPS = 0;
    static constexpr int SMEM_B_BYTES = B_N * B_K * sizeof(D_B);

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = 0;
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
};

struct Dw1VarlenBf16Gfx950Bm64Bn128Bk32Swizzled
    : Dw1Bf16Gfx950Bm64Bn128Bk32Swizzled
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

struct Dw2VarlenBf16Gfx950Bm64Bn64Bk64Swizzled
    : Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled
{
    static constexpr RouteLayout ROUTE_LAYOUT =
        RouteLayout::CompactRouteMajor;
};

} // namespace opus_moe_backward::gfx950
