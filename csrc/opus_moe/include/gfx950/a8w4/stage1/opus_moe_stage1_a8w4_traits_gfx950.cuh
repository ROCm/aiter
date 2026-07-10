// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_common.cuh"

#include "opus/opus.hpp"

#include <cstdint>

namespace opus_moe
{
namespace stage1_a8w4
{

struct OpusMoeStage1A8W4Kargs
{
    const uint8_t* __restrict__ hidden_fp8;
    const uint8_t* __restrict__ w1_fp4;
    const uint8_t* __restrict__ hidden_scale_e8m0;
    const uint8_t* __restrict__ w1_scale_e8m0;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    uint8_t* __restrict__ inter_states_fp8;
    uint8_t* __restrict__ inter_states_scale_e8m0;

    int64_t stride_hidden_t;
    int64_t stride_w1_e;
    int64_t stride_w1_n;
    int64_t stride_hidden_scale_route;
    int64_t stride_w1_scale_row;
    int64_t stride_out_t;
    int64_t stride_out_k;
    int64_t stride_out_scale_route;

    int token_num;
    int topk;
    int sorted_blocks;
    int kernel_id;
    int inter_dim_pad;
    int model_dim_pad;
};

constexpr int kTopK = 6;
constexpr int kModelDim = 7168;
constexpr int kLogicalInterDim = 512;
constexpr int kInterDimPad = 128;
constexpr int kEffectiveInterDim = kLogicalInterDim - kInterDimPad;
constexpr int kGateUpLogicalDim = 2 * kLogicalInterDim;
constexpr int kGateUpEffectiveDim = 2 * kEffectiveInterDim;
constexpr int kExperts = 384;

constexpr int kScaleGroupLogicalK = 32;
constexpr int kFp4ValuesPerByte = 2;
constexpr int kVectorBytes = 16;
constexpr int kMfmaM = 16;
constexpr int kMfmaN = 16;
constexpr int kMfmaK = 128;
constexpr int kDefaultCtaThreads = 256;
constexpr int kMinBlocksPerCu = 2;

// Shape aliases pass tuning flags positionally. Keep the trailing flag order as:
// MinBlocksPerCuOverride, SkipInvalidAScaleGuard, AssumeRouteTileHasRoute,
// PairGateUpSingleGroup, AsyncARegLds, FullNextARegPrefetch.
template<int BlockM,
         int BlockN,
         int BlockK,
         int SortBlockM = BlockM,
         int EpilogueRowSplit = 1,
         bool GateUpGroupSplit = false,
         int OutputScaleGroupsOverride = 0,
         int KWave = 1,
         bool CapRouteTilesToRoutedBlocks = false,
         bool ClampOutputFp8 = true,
         bool SplitSelectorB = false,
         int MinBlocksPerCuOverride = 0,
         bool SkipInvalidAScaleGuard = false,
         bool AssumeRouteTileHasRoute = false,
         bool PairGateUpSingleGroup = false,
         bool AsyncARegLds = false,
         bool FullNextARegPrefetch = false>
struct OpusMoeStage1A8W4Shape
{
    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K_LOGICAL = BlockK;
    static constexpr int SORT_BLOCK_M = SortBlockM;
    static constexpr int T_M = 1;
    static constexpr int T_N = 1;
    static constexpr int MMA_M = kMfmaM;
    static constexpr int MMA_N = kMfmaN;
    static constexpr int MMA_K = kMfmaK;
    static constexpr int M_MFMA_PER_WAVE = B_M / kMfmaM;
    static constexpr int N_MFMA_PER_WAVE = 1;
    static constexpr int THREADS_K = 4;
    static constexpr int K_STEP_PACKED = B_K_LOGICAL / kFp4ValuesPerByte;
    static constexpr int BYTES_PER_VEC = kVectorBytes;
    static constexpr int VEC_A = BYTES_PER_VEC;
    static constexpr int B_BYTES_PER_VEC = BYTES_PER_VEC;
    static constexpr int A_LDS_STAGES = 3;
    static constexpr int A_LDS_STAGE_ELEMS = B_M * B_K_LOGICAL;
    static constexpr int WAVE_SIZE = opus::get_warp_size();
    static constexpr int K_WAVE = KWave;
    static constexpr bool PAIR_GATE_UP_SINGLE_GROUP = PairGateUpSingleGroup;
    static constexpr bool ASYNC_A_REG_LDS = AsyncARegLds || B_M >= 128;
    static constexpr int KWAVE_BASE_WAVES =
        PAIR_GATE_UP_SINGLE_GROUP ? 2 : 4;
    static constexpr int BLOCK_SIZE =
        K_WAVE == 1 ? kDefaultCtaThreads : KWAVE_BASE_WAVES * K_WAVE * WAVE_SIZE;
    static constexpr int MIN_BLOCKS_PER_CU =
        MinBlocksPerCuOverride > 0 ? MinBlocksPerCuOverride :
        K_WAVE == 1 ? kMinBlocksPerCu :
        1;

    static constexpr int TOPK = kTopK;
    static constexpr int H = kModelDim;
    static constexpr int MFMA_K_STEPS = H / kMfmaK;
    static constexpr int K_TILES = MFMA_K_STEPS;
    static constexpr int HIDDEN_VECS_PER_ROW = H / BYTES_PER_VEC;
    static constexpr int PACKED_H = H / kFp4ValuesPerByte;
    static constexpr int W1_VECS_PER_ROW = PACKED_H / BYTES_PER_VEC;
    static constexpr int LOGICAL_INTER_DIM = kLogicalInterDim;
    static constexpr int INTER_DIM_PAD = kInterDimPad;
    static constexpr int EFFECTIVE_INTER_DIM = kEffectiveInterDim;
    static constexpr int GATE_UP_LOGICAL_DIM = kGateUpLogicalDim;
    static constexpr int GATE_UP_EFFECTIVE_DIM = kGateUpEffectiveDim;
    static constexpr int EXPERTS = kExperts;
    static constexpr int DEFAULT_OUTPUT_COLS_PER_TILE = B_N / 2;
    static constexpr int DEFAULT_OUTPUT_SCALE_GROUPS_PER_TILE =
        DEFAULT_OUTPUT_COLS_PER_TILE / kScaleGroupLogicalK;
    static constexpr int OUTPUT_SCALE_GROUPS_PER_TILE =
        OutputScaleGroupsOverride > 0 ?
            OutputScaleGroupsOverride :
            DEFAULT_OUTPUT_SCALE_GROUPS_PER_TILE;
    static constexpr int ACC_SCALE_GROUPS_PER_TILE =
        PAIR_GATE_UP_SINGLE_GROUP ? 2 : OUTPUT_SCALE_GROUPS_PER_TILE;
    static constexpr int OUTPUT_COLS_PER_TILE =
        OUTPUT_SCALE_GROUPS_PER_TILE * kScaleGroupLogicalK;
    static constexpr int SCALE_MN_PACK = 2;
    static constexpr int SCALE_K_PACK = 2;
    static constexpr int EPILOGUE_ROW_SPLIT = EpilogueRowSplit;
    static constexpr int EPILOGUE_ROWS_PER_PASS = B_M / EPILOGUE_ROW_SPLIT;
    static constexpr int EPILOGUE_SMEM_COLS =
        OUTPUT_SCALE_GROUPS_PER_TILE * kScaleGroupLogicalK * 2;
    static constexpr int EPILOGUE_SMEM_ROWS = EPILOGUE_ROWS_PER_PASS;
    static constexpr int EPILOGUE_THREADS = EPILOGUE_ROWS_PER_PASS * 2;
    static constexpr int EPILOGUE_SMEM_BYTES =
        EPILOGUE_SMEM_ROWS * EPILOGUE_SMEM_COLS *
        static_cast<int>(sizeof(float));
    static constexpr int A_REG_LDS_STAGE_BYTES = B_M * MMA_K;
    static constexpr int A_REG_LDS_BYTES =
        2 * SCALE_K_PACK * A_REG_LDS_STAGE_BYTES;
    static constexpr int KWAVE_REDUCE_BYTES =
        K_WAVE == 1 ? 0 : K_WAVE * KWAVE_BASE_WAVES *
                              M_MFMA_PER_WAVE *
                              ACC_SCALE_GROUPS_PER_TILE * WAVE_SIZE *
                              static_cast<int>(sizeof(float)) * 4;
    static constexpr int MAINLOOP_SCRATCH_BYTES =
        A_REG_LDS_BYTES > KWAVE_REDUCE_BYTES ? A_REG_LDS_BYTES :
                                               KWAVE_REDUCE_BYTES;
    static constexpr int SHARED_SCRATCH_BYTES =
        EPILOGUE_SMEM_BYTES > MAINLOOP_SCRATCH_BYTES ?
            EPILOGUE_SMEM_BYTES :
            MAINLOOP_SCRATCH_BYTES;
    static constexpr bool GATE_UP_GROUP_SPLIT = GateUpGroupSplit;
    static constexpr bool CAP_ROUTE_TILES_TO_ROUTED_BLOCKS =
        CapRouteTilesToRoutedBlocks;
    static constexpr bool CLAMP_OUTPUT_FP8 = ClampOutputFp8;
    static constexpr bool SPLIT_SELECTOR_B = SplitSelectorB;
    static constexpr bool SKIP_INVALID_A_SCALE_GUARD =
        SkipInvalidAScaleGuard;
    static constexpr bool ASSUME_ROUTE_TILE_HAS_ROUTE =
        AssumeRouteTileHasRoute;
    static constexpr bool FULL_NEXT_A_REG_PREFETCH =
        FullNextARegPrefetch;
    static constexpr int GATE_UP_GROUP_SPLIT_GROUPS =
        OUTPUT_SCALE_GROUPS_PER_TILE / 2;

    static constexpr int TILE2_PAD =
        (B_K_LOGICAL / 2 - EFFECTIVE_INTER_DIM % (B_K_LOGICAL / 2)) %
        (B_K_LOGICAL / 2);
    static constexpr int STAGE1_COL_TILES =
        (EFFECTIVE_INTER_DIM + OUTPUT_COLS_PER_TILE - 1) /
        OUTPUT_COLS_PER_TILE;
    static constexpr int SCALE_GROUP_LOGICAL_K = kScaleGroupLogicalK;
    static constexpr int SCALE_GROUPS = LOGICAL_INTER_DIM / SCALE_GROUP_LOGICAL_K;
    static constexpr int EFFECTIVE_SCALE_GROUPS =
        (EFFECTIVE_INTER_DIM + SCALE_GROUP_LOGICAL_K - 1) / SCALE_GROUP_LOGICAL_K;
    static constexpr int HIDDEN_SCALE_GROUPS = H / SCALE_GROUP_LOGICAL_K;
    static constexpr int HIDDEN_SCALE_WORDS_PER_ROW =
        HIDDEN_SCALE_GROUPS * static_cast<int>(sizeof(uint8_t)) /
        static_cast<int>(sizeof(uint32_t));
    static constexpr int M_SCALE_PACKS =
        (M_MFMA_PER_WAVE + SCALE_MN_PACK - 1) / SCALE_MN_PACK;
    static constexpr int SCALE_WORDS_PER_K_TILE =
        (HIDDEN_SCALE_GROUPS / 4) / SCALE_K_PACK;
    static constexpr int SCALE_LAYOUT_STRIDE_K0 = 4 * MMA_M;
    static constexpr int SCALE_LAYOUT_STRIDE_N0 =
        SCALE_WORDS_PER_K_TILE * SCALE_LAYOUT_STRIDE_K0;

    using D_A = uint8_t;
    using D_W = uint8_t;

    static_assert(B_M > 0);
    static_assert(B_N > 0);
    static_assert(B_K_LOGICAL > 0);
    static_assert(SORT_BLOCK_M > 0);
    static_assert(SORT_BLOCK_M % B_M == 0);
    static_assert(H % MMA_K == 0);
    static_assert(H % B_K_LOGICAL == 0);
    static_assert(B_K_LOGICAL % MMA_K == 0);
    static_assert(GATE_UP_EFFECTIVE_DIM % B_N == 0);
    static_assert(LOGICAL_INTER_DIM % SCALE_GROUP_LOGICAL_K == 0);
    static_assert(B_K_LOGICAL % kFp4ValuesPerByte == 0);
    static_assert(B_N % 2 == 0);
    static_assert(B_N % (2 * MMA_N) == 0);
    static_assert(B_M % MMA_M == 0);
    static_assert(M_MFMA_PER_WAVE > 0);
    static_assert(MFMA_K_STEPS % SCALE_K_PACK == 0);
    static_assert(K_WAVE > 0);
    static_assert(K_WAVE == 1 || OUTPUT_SCALE_GROUPS_PER_TILE == 1);
    static_assert(MFMA_K_STEPS % K_WAVE == 0);
    static_assert((MFMA_K_STEPS / K_WAVE) % SCALE_K_PACK == 0);
    static_assert(K_WAVE == 1 || BLOCK_SIZE <= 1024);
    static_assert(EPILOGUE_ROW_SPLIT > 0);
    static_assert(B_M % EPILOGUE_ROW_SPLIT == 0);
    static_assert(EPILOGUE_ROWS_PER_PASS > 0);
    static_assert(EPILOGUE_SMEM_COLS > 0);
    static_assert(EPILOGUE_SMEM_ROWS > 0);
    static_assert(!GATE_UP_GROUP_SPLIT ||
                  OUTPUT_SCALE_GROUPS_PER_TILE % 2 == 0);
    static_assert(H % BYTES_PER_VEC == 0);
    static_assert(PACKED_H % BYTES_PER_VEC == 0);
    static_assert(OUTPUT_COLS_PER_TILE % SCALE_GROUP_LOGICAL_K == 0);
    static_assert(HIDDEN_SCALE_GROUPS % static_cast<int>(sizeof(uint32_t)) == 0);
    static_assert(!CAP_ROUTE_TILES_TO_ROUTED_BLOCKS || SORT_BLOCK_M % B_M == 0);
    static_assert(!ASSUME_ROUTE_TILE_HAS_ROUTE ||
                  CAP_ROUTE_TILES_TO_ROUTED_BLOCKS);
    static_assert(!PAIR_GATE_UP_SINGLE_GROUP ||
                  (!GATE_UP_GROUP_SPLIT &&
                   OUTPUT_SCALE_GROUPS_PER_TILE == 1));
};

using OpusMoeStage1A8W4P0Bm32Bn384GateUpGroupSplitNoClamp =
    OpusMoeStage1A8W4Shape<32, 384, 256, 32, 2, true, 0, 1, false, false>;
using OpusMoeStage1A8W4P0Bm16Bn384G6KWave1NoClampGroupSplitMin1AReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, true, 6, 1, true, false, false, 1>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave2CapRoutesNoClampSplitSelectorBMin2PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 2, true, false, true, 2, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin1PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, true, 1, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin2PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, true, 2, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampSplitSelectorBMin3PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, true, 3, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin1PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, false, 1, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin2PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, false, 2, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin3PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, false, 3, false, false, true>;
using OpusMoeStage1A8W4P0Bm16Bn384G1KWave4CapRoutesNoClampMin4PairGateUpAReuse =
    OpusMoeStage1A8W4Shape<16, 384, 256, 16, 1, false, 1, 4, true, false, false, 4, false, false, true>;
using OpusMoeStage1A8W4P0Bm64Bn384GateUpGroupSplitT4096NoClampMin1AsyncACapRoutesAssumeRouteSplitB =
    OpusMoeStage1A8W4Shape<64, 384, 256, 64, 2, true, 0, 1, true, false, true, 1, false, true, false, true>;
using OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin3NoScaleGuardSplitB =
    OpusMoeStage1A8W4Shape<128, 384, 256, 128, 2, true, 0, 1, false, false, true, 3, true>;
using OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin2NoScaleGuardFullNextA =
    OpusMoeStage1A8W4Shape<128, 384, 256, 128, 2, true, 0, 1, false, false, false, 0, true, false, false, false, true>;
using OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin3NoScaleGuardFullNextA =
    OpusMoeStage1A8W4Shape<128, 384, 256, 128, 2, true, 0, 1, false, false, false, 3, true, false, false, false, true>;
using OpusMoeStage1A8W4P0Bm128Bn384GateUpGroupSplitNoClampMin4NoScaleGuardFullNextA =
    OpusMoeStage1A8W4Shape<128, 384, 256, 128, 2, true, 0, 1, false, false, false, 4, true, false, false, false, true>;
} // namespace stage1_a8w4
} // namespace opus_moe
