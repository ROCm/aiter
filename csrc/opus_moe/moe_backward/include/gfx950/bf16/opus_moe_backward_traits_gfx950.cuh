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
    static constexpr int ROUTE_COHORT_TILES = 0;
    static constexpr int ROUTE_M_TILES = 1;
    static constexpr bool COMPACT_ROUTE_GROUP_GRID = false;
    static constexpr bool PREDECODE_ROUTE_METADATA = false;
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

struct DownBwdBf16Gfx950Bm32Bn128Bk64PaddedCohort2
    : DownBwdBf16Gfx950Bm32Bn128Bk64Padded
{
    static constexpr int ROUTE_COHORT_TILES = 2;
};

// Reuse one W2 tile across two adjacent 32-route tiles of the same expert.
// The physical sorter granularity remains 32 rows, so this works with the
// existing forward metadata and masks the final half when an expert owns an
// odd number of padded tiles.
struct DownBwdBf16Gfx950Bm32Bn128Bk64PaddedM2Cohort2
    : DownBwdBf16Gfx950Bm32Bn128Bk64PaddedCohort2
{
    static constexpr int ROUTE_M_TILES = 2;
};

// The largest reuse factor that keeps the two-stage dO/W2 payload below the
// 64 KiB per-workgroup LDS boundary: 2 * (3 * 4 KiB + 17 KiB) ~= 58 KiB.
struct DownBwdBf16Gfx950Bm32Bn128Bk64PaddedM3Cohort2
    : DownBwdBf16Gfx950Bm32Bn128Bk64PaddedCohort2
{
    static constexpr int ROUTE_M_TILES = 3;
};

// Halving BK leaves enough LDS to keep five adjacent route tiles live while
// reusing one W2 tile.  A twenty-tile launch cohort aligns four such groups,
// keeping W2 reuse close without giving up dO locality.  The extra K-loop
// boundaries are amortized once the routed activation footprint is beyond L2.
struct DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM5Cohort20
    : DownBwdBf16Gfx950Bm32Bn128Bk64PaddedCohort2
{
    static constexpr int B_K = 32;
    static constexpr int E_K = 2;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES = SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;
    static constexpr int ROUTE_M_TILES = 5;
    static constexpr int ROUTE_COHORT_TILES = 20;
    static constexpr bool COMPACT_ROUTE_GROUP_GRID = true;
};

struct DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM5Cohort20Predecoded
    : DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM5Cohort20
{
    static constexpr bool PREDECODE_ROUTE_METADATA = true;
};

struct DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM6Cohort24Predecoded
    : DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM5Cohort20Predecoded
{
    static constexpr int ROUTE_M_TILES = 6;
    static constexpr int ROUTE_COHORT_TILES = 24;
};

// Double the intermediate-column tile while retaining M6 W2 reuse.  Four
// waves each own two 32-column MFMA repeats, halving gathered dO traffic and
// dScore workspace/finalize traffic for sufficiently large launch grids.
struct DownBwdBf16Gfx950Bm32Bn256Bk32PaddedM6Cohort24Predecoded
    : DownBwdBf16Gfx950Bm32Bn128Bk32PaddedM6Cohort24Predecoded
{
    static constexpr int B_N = 256;
    static constexpr int E_N = 2;
    static constexpr bool BATCH_SIGMOID = true;
    // Each stage moves a wider W2 slab than the gathered dO slab.  Issue W2
    // first so the transfer that gates vmcnt(0) starts as early as possible;
    // this wins across the BN256 shape family without changing resources.
    static constexpr bool ISSUE_B_FIRST = true;
    // Queue three route-tile A reads behind the wider B transpose reads before
    // waiting on LDS.  This overlaps their latency without crossing the live-
    // range cliff observed when all four leading A fragments stay resident.
    static constexpr int PREFETCH_A_TILES = 3;
    // After the GEMM mainloop, reuse its dead LDS allocation to coalesce the
    // otherwise lane-scattered Z reads.  One direct 16-byte global-to-LDS
    // issue per lane fills two adjacent route rows; a padded row-pair layout
    // limits the subsequent epilogue reads to two-way same-bank sharing.
    static constexpr bool STAGE_Z_IN_LDS = true;
    static constexpr int Z_LDS_PAIR_PAD = 16; // BF16 elements = 32 bytes.
    static constexpr int SMEM_B_ROW_BYTES = B_N * sizeof(D_B);
    static constexpr int SMEM_B_GROUP_DATA_BYTES =
        SMEM_B_GROUP_ROWS * SMEM_B_ROW_BYTES;
    static constexpr int SMEM_B_GROUP_BYTES =
        SMEM_B_GROUP_DATA_BYTES + SMEM_B_GROUP_PAD_BYTES;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES = SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;
    static_assert(B_N == T_N * W_N * E_N);
};

struct DownBwdBf16Gfx950Bm32Bn256Bk32PaddedM6Cohort24PredecodedDeferredZWait
    : DownBwdBf16Gfx950Bm32Bn256Bk32PaddedM6Cohort24Predecoded
{
    static constexpr bool DEFER_Z_LDS_WAIT = true;
};

// K2: gathered dZ x W1, retaining Triton's 32x128x64 two-stage geometry.
struct RouteDxBf16Gfx950Bm32Bn128Bk64WideStore
    : Bf16Traits<Family::RouteDx, 32, 128, 64, 256, 2, false>
{
    // The legacy two-dimensional launch walks all route tiles before the
    // next output-N tile.  Derived instances can bound that reuse domain to
    // a small route cohort: route tiles remain fastest within each N tile so
    // they share W1, while dZ is revisited after only one cohort.
    static constexpr int ROUTE_COHORT_TILES = 0;
    static constexpr int ROUTE_M_TILES = 2;
    static constexpr int ROUTE_M = 32;
    static constexpr int SMEM_A_SLAB_PAD = 0;
    static constexpr bool WRITE_SORTED_ROUTES = false;
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

struct RouteDxBf16Gfx950Bm32Bn128Bk64WideStoreCohort4
    : RouteDxBf16Gfx950Bm32Bn128Bk64WideStore
{
    static constexpr int ROUTE_COHORT_TILES = 4;
};

struct RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreCohort4
    : RouteDxBf16Gfx950Bm32Bn128Bk64WideStoreCohort4
{
    // Preserve the two-stage pipeline while halving each stage's LDS
    // footprint.  The extra K-loop boundaries trade more barriers for higher
    // residency; auto-dispatch can select it only if the measured trade wins.
    static constexpr int B_K = 32;
    static constexpr int E_K = 2;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES =
        SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;
};

struct RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10
    : RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreCohort4
{
    static constexpr int ROUTE_COHORT_TILES = 10;
    static constexpr int ROUTE_M_TILES = 5;
};

// Match the forward GEMM-style LDS policy: add 32 bytes after each 16-row dZ
// slab.  Direct global-to-LDS loads remain lane-linear inside a slab while
// successive slabs rotate by eight banks.  The full double buffer still fits
// four workgroups per CU.
struct RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPad
    : RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10
{
    static constexpr int SMEM_A_SLAB_PAD = 16;
};

// Keep the grouped GEMM output in its natural expert-sorted order.  The
// paired K3 instance uses reverse_sorted to gather the K routes for a token,
// avoiding K2's random scatter without changing the routing ABI or GEMM.
struct RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPadSortedOutput
    : RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPad
{
    static constexpr bool WRITE_SORTED_ROUTES = true;
};

// Double the output-N tile so each dZ slab feeds twice as many columns.  Four
// waves retain the 32-column MFMA mapping and each compute two N repeats; the
// B transfer uses two direct-to-LDS vectors per lane to fill each padded
// 4x256 row group.  Three route tiles balance W1 reuse against the extra N
// accumulators without binding the geometry to a model tuple.
struct RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPad
    : RouteDxBf16Gfx950Bm32Bn128Bk32WideStoreM5Cohort10ASlabPad
{
    static constexpr int B_N = 256;
    static constexpr int E_N = 2;
    static constexpr int ROUTE_COHORT_TILES = 6;
    static constexpr int ROUTE_M_TILES = 3;
    // The wide-N M3 tile uses six consecutive 16-row A slabs.  A 96-byte
    // inter-slab rotation lowers gfx950 LDS dependency waits and compiler
    // address-register pressure while retaining two resident workgroups.
    static constexpr int SMEM_A_SLAB_PAD = 48;
    static constexpr int SMEM_B_ROW_BYTES = B_N * sizeof(D_B);
    static constexpr int SMEM_B_GROUP_DATA_BYTES =
        SMEM_B_GROUP_ROWS * SMEM_B_ROW_BYTES;
    static constexpr int SMEM_B_GROUP_BYTES =
        SMEM_B_GROUP_DATA_BYTES + SMEM_B_GROUP_PAD_BYTES;
    static constexpr int SMEM_B_GROUPS = B_K / SMEM_B_GROUP_ROWS;
    static constexpr int SMEM_B_BYTES = SMEM_B_GROUPS * SMEM_B_GROUP_BYTES;
    static_assert(B_N == T_N * W_N * E_N);
};

struct RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPadSortedOutput
    : RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPad
{
    static constexpr bool WRITE_SORTED_ROUTES = true;
};

struct RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPadSortedOutputBFirst
    : RouteDxBf16Gfx950Bm32Bn256Bk32WideStoreM3Cohort6ASlabPadSortedOutput
{
    static constexpr bool ISSUE_B_FIRST = true;
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
    static constexpr bool READ_SORTED_ROUTES = false;
    static constexpr bool BROADCAST_ROUTE_ID = false;
};

struct RouteReduceBf16Gfx950Bm16Bn128SortedInput
    : RouteReduceBf16Gfx950Bm16Bn128
{
    static constexpr bool READ_SORTED_ROUTES = true;
};

// When one 2048-column tile covers the complete token row, all four waves in
// the CTA consume adjacent slices of the same reverse-gathered routes.  Each
// wave broadcasts one route id instead of issuing 64 duplicate metadata
// loads.  This keeps the same 2048 output elements per CTA as BM16xBN128 but
// lowers both VGPR pressure and random-memory scheduling overhead.
struct RouteReduceBf16Gfx950Bm1Bn2048SortedInput
    : RouteReduceBf16Gfx950Bm16Bn128SortedInput
{
    static constexpr int B_M = 1;
    static constexpr int B_N = 2048;
    static constexpr bool BROADCAST_ROUTE_ID = true;
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
    // Zero preserves the original (expert-fastest) 3-D grid.  Positive
    // values group a small expert cohort while walking all output tiles,
    // shortening the L2 reuse distance without serializing the whole grid by
    // expert.  Tuning candidates inherit this trait and override the policy.
    static constexpr int EXPERT_COHORT = 0;
    static constexpr bool DIRECT_GMEM_TO_LDS = false;
    static constexpr bool DOUBLE_BUFFER = false;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = false;
    static constexpr bool SPLIT_B_N64_SWIZZLE = false;
    static constexpr int EMPTY_M_TILES_PER_CTA = 1;
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

struct Dw1Bf16Gfx950Bm64Bn128Bk32SwizzledCohort2DirectLds
    : Dw1Bf16Gfx950Bm64Bn128Bk32Swizzled
{
    static constexpr int EXPERT_COHORT = 2;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DOUBLE_BUFFER = false;
};

// Keep the four-wave workgroup used by the BM64 kernel, but let every lane
// load two 64-row dZ slabs.  A 2x2 wave grid balances dZ and gathered-X
// transpose reads while sharing each X tile across twice as many dW1 rows.
// Two LDS stages overlap the next gathered reduction tile with the current
// MFMA, without the occupancy cost of a 512-thread BM128 workgroup.
struct Dw1Bf16Gfx950Bm128Bn128Bk32SwizzledCohort2DoubleLds
    : Bf16Traits<Family::Dw1, 128, 128, 32, 256, 2, false>
{
    static constexpr int EXPERT_COHORT = 2;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DOUBLE_BUFFER = true;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = true;
    static constexpr int EMPTY_M_TILES_PER_CTA = 2;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 2;
    static constexpr int E_N = 2;
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

// Eight waves cover a 256x128 output tile while keeping the same four native
// 32x32 accumulator tiles per wave as the BM128 kernel.  Four-wave groups load
// one 64-row dZ slab each; every wave loads two slabs in total.  The full
// workgroup cooperatively loads one gathered-X tile, halving X/metadata loads
// relative to two independent BM128 CTAs.  Two 24-KiB LDS stages retain the
// existing reduction pipeline at one 512-thread workgroup per CU.
struct Dw1Bf16Gfx950Bm256Bn128Bk32SwizzledCohort2DoubleLds
    : Bf16Traits<Family::Dw1, 256, 128, 32, 512, 1, false>
{
    static constexpr int EXPERT_COHORT = 2;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DOUBLE_BUFFER = true;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = true;
    static constexpr int EMPTY_M_TILES_PER_CTA = 8;
    static constexpr int T_M = 4;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 2;
    static constexpr int E_N = 2;
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

// Cover the same BM256xBN128 tile with four waves.  Each wave owns twice as
// many independent C fragments, trading VGPRs for fewer scheduled waves and
// more MFMA chains between the transpose-LDS waits.
struct Dw1Bf16Gfx950Bm256Bn128Bk32Wave4Cohort2DoubleLds
    : Bf16Traits<Family::Dw1, 256, 128, 32, 256, 1, false>
{
    static constexpr int EXPERT_COHORT = 2;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DOUBLE_BUFFER = true;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = true;
    static constexpr bool SPLIT_B_N64_SWIZZLE = true;
    static constexpr int EMPTY_M_TILES_PER_CTA = 8;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 4;
    static constexpr int E_N = 2;
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

struct Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4DoubleLds
    : Dw1Bf16Gfx950Bm256Bn128Bk32Wave4Cohort2DoubleLds
{
    static constexpr int EXPERT_COHORT = 4;
    static constexpr bool REVERSE_EXPERT_ORDER = true;
};

// Keep the same output and LDS geometry as the production wave4 kernel, but
// issue the second K16 dZ fragment before the first fragment's MFMA.  X is
// read after that MFMA so only the larger A fragment remains live early.
struct Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4PrefetchADoubleLds
    : Dw1Bf16Gfx950Bm256Bn128Bk32Wave4ReverseCohort4DoubleLds
{
    static constexpr bool PREFETCH_REDUCTION_A = true;
};

// K5: dO^T x (S*A), 64x64 output with K64 and swizzled LDS reuse.
struct Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled
    : Bf16Traits<Family::Dw2, 64, 64, 64, 256, 2, false>
{
    static constexpr int EXPERT_COHORT = 0;
    static constexpr bool DIRECT_GMEM_TO_LDS = false;
    static constexpr bool DUAL_OPERAND_LDS = false;
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

struct Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort4DirectLds
    : Dw2Bf16Gfx950Bm64Bn64Bk64Swizzled
{
    static constexpr int EXPERT_COHORT = 4;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
};

struct Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort1DirectLds
    : Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort4DirectLds
{
    static constexpr int EXPERT_COHORT = 1;
};

struct Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort2DirectLds
    : Dw2Bf16Gfx950Bm64Bn64Bk64SwizzledCohort4DirectLds
{
    static constexpr int EXPERT_COHORT = 2;
};

// Accumulate two native output-M tiles per workgroup while keeping both
// operands resident in LDS.  This halves the output grid and shares one
// a_scaled tile across two D slabs for large routed working sets.
struct Dw2Bf16Gfx950Bm128Bn64Bk64SwizzledCohort1DualLds
    : Bf16Traits<Family::Dw2, 128, 64, 64, 256, 2, false>
{
    static constexpr int EXPERT_COHORT = 1;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DUAL_OPERAND_LDS = true;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 2;
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

// Eight waves share a 128x128 output tile.  Four loader waves cover each
// 64-column slab of both operands; the complete 32 KiB LDS tile remains
// resident while every wave accumulates two native 32x32 outputs.
struct Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledCohort1DualLdsWave4x2
    : Bf16Traits<Family::Dw2, 128, 128, 64, 512, 2, false>
{
    static constexpr int EXPERT_COHORT = 1;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DUAL_OPERAND_LDS = true;
    static constexpr int T_M = 4;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 1;
    static constexpr int E_N = 2;
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

struct Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledCohort1DualLdsWave2x2
    : Bf16Traits<Family::Dw2, 128, 128, 64, 256, 2, false>
{
    static constexpr int EXPERT_COHORT = 1;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DUAL_OPERAND_LDS = true;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 2;
    static constexpr int E_N = 2;
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

// Four waves share one 128-column a_scaled slab across a 256-row dO output
// tile.  Each wave owns eight native C tiles and pipelines one K16 fragment
// at a time; compared with the two-wave BM128 candidate this halves the grid
// and each thread's direct-to-LDS vector count while retaining four resident
// waves in the 48 KiB workgroup.
struct Dw2Bf16Gfx950Bm256Bn128Bk64SwizzledCohort1DualLdsWave2x2
    : Bf16Traits<Family::Dw2, 256, 128, 64, 256, 1, false>
{
    static constexpr int EXPERT_COHORT = 1;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DUAL_OPERAND_LDS = true;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = true;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 4;
    static constexpr int E_N = 2;
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

struct Dw2Bf16Gfx950Bm256Bn128Bk64SwizzledCohort4DualLdsWave2x2
    : Dw2Bf16Gfx950Bm256Bn128Bk64SwizzledCohort1DualLdsWave2x2
{
    static constexpr int EXPERT_COHORT = 4;
    // Issue the next, larger dO fragment before the current MFMA chain, then
    // read the matching a_scaled fragment.  The 48-KiB LDS footprint already
    // fixes residency at one CTA/CU, so the extra 16 VGPRs hide transpose-read
    // latency without reducing occupancy.
    static constexpr bool PREFETCH_REDUCTION_A = true;
};

// Reuse the K32 double-stage weight-gradient pipeline for dW2.  The route
// source direction is the inverse of dW1: dO gathers token rows while
// a_scaled stays in expert-sorted rows.  Two 24-KiB stages preserve the
// production BM256xBN128 output tile while overlapping the next K32 load.
struct Dw2Bf16Gfx950Bm256Bn128Bk32Cohort4DoubleLds
    : Bf16Traits<Family::Dw2, 256, 128, 32, 256, 1, false>
{
    static constexpr int EXPERT_COHORT = 4;
    static constexpr bool DIRECT_GMEM_TO_LDS = true;
    static constexpr bool DOUBLE_BUFFER = true;
    static constexpr bool PIPELINE_REDUCTION_FRAGMENTS = true;
    static constexpr bool PREFETCH_REDUCTION_A = true;
    static constexpr bool SPLIT_B_N64_SWIZZLE = true;
    static constexpr bool SWAP_ROUTE_SOURCES = true;
    static constexpr int EMPTY_M_TILES_PER_CTA = 8;
    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static constexpr int W_M = 32;
    static constexpr int W_N = 32;
    static constexpr int W_K = 16;

    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int E_M = 4;
    static constexpr int E_N = 2;
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

struct Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledRouteLe30720
    : Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledCohort1DualLdsWave2x2
{
    static constexpr int MAX_ROUTE_COUNT = 30720;
};

struct Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledRouteGt30720
    : Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledCohort1DualLdsWave4x2
{
    static constexpr int MIN_ROUTE_COUNT = 30721;
};

// Dispatch marker: the launcher emits the mutually exclusive short- and
// long-reduction kernels above.  The route-count predicate is read on device,
// so no host synchronization or routing-distribution assumption is needed.
struct Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledAdaptiveRoutes
    : Dw2Bf16Gfx950Bm128Bn128Bk64SwizzledRouteLe30720
{
    static constexpr bool ADAPTIVE_ROUTE_SPLIT = true;
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
