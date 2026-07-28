// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_common.cuh"
#include "opus/opus.hpp"

template<typename Contract = opus_moe::OpusMoeStage2A8W4DefaultContract,
         int BlockM = opus_moe::kStage2A8W4DecodeDefaultBlockM,
         int BlockN = opus_moe::kStage2A8W4DecodeDefaultBlockN,
         int SortBlockM = BlockM,
         bool DirectAtomicOut = true,
         bool PaceRouteBlocksToPow2 = false,
         int BlockThreadsOverride = 0,
         int MinBlocksPerCuOverride = 0,
         int CachectlBOverride = 0,
         int CachectlWScaleOverride = 0>
struct OpusMoeStage2A8W4DecodeShape
{
    // Atomic vs MXFP8 route-out is a structural compile-time choice.
    static constexpr bool DIRECT_ATOMIC_OUT = DirectAtomicOut;
    static constexpr bool IS_BM16 = BlockM == opus_moe::kStage2A8W4DecodeBlockM16;
    static constexpr bool IS_BM32_BN256 =
        BlockM == opus_moe::kStage2A8W4DecodeBlockM32 && BlockN == opus_moe::kStage2A8W4DecodeBlockN256;
    static constexpr bool IS_BM64_BN256 =
        BlockM == opus_moe::kStage2A8W4DecodeBlockM64 && BlockN == opus_moe::kStage2A8W4DecodeBlockN256;
    static constexpr int B_M = BlockM, B_N = BlockN;
    static constexpr int B_K_LOGICAL = opus_moe::kStage2A8W4DecodeBKLogical;
    static constexpr int K_STEP_PACKED = B_K_LOGICAL / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte;

    using D_A = uint8_t;

    static constexpr int BYTES_PER_VEC = opus_moe::kStage2A8W4DecodeVectorBytes;
    static constexpr int VEC_A = BYTES_PER_VEC / sizeof(D_A), B_BYTES_PER_VEC = BYTES_PER_VEC;

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = CachectlBOverride;
    static constexpr int CACHECTL_W_SCALE = CachectlWScaleOverride;

    static constexpr int MMA_M = opus_moe::kStage2A8W4DecodeMfmaM;
    static constexpr int MMA_N = opus_moe::kStage2A8W4DecodeMfmaN;
    static constexpr int MMA_K = opus_moe::kStage2A8W4DecodeMfmaK;
    static constexpr int THREADS_K = opus::get_warp_size() / MMA_M;
    static constexpr int T_M = IS_BM32_BN256 ? 2 : 1;

    static constexpr int DECODE_LOGICAL_INTER_DIM = Contract::DECODE_LOGICAL_INTER_DIM;
    static constexpr int DECODE_INTER_DIM_PAD = Contract::DECODE_INTER_DIM_PAD;
    static constexpr int DECODE_EFFECTIVE_INTER_DIM = Contract::DECODE_EFFECTIVE_INTER_DIM;
    static constexpr int SORT_BLOCK_M = SortBlockM;
    static constexpr int ROUTE_M_STRIDE = B_M;
    // route_out XCD swizzle (gfx950=8 XCDs).
    static constexpr int NUM_XCD = DIRECT_ATOMIC_OUT ? 1 : 8;
    static constexpr int SWIZZLE_W = 2;
    static constexpr int SWIZZLE_C =
        (!DIRECT_ATOMIC_OUT && (IS_BM32_BN256 || IS_BM64_BN256))
            ? (IS_BM64_BN256 && DECODE_EFFECTIVE_INTER_DIM / K_STEP_PACKED >= 4 ? 24 : 32)
            : 0;
    static constexpr bool DECODE_PACE_ROUTE_BLOCKS_TO_POW2 = PaceRouteBlocksToPow2;
    static constexpr int K_TILES = DECODE_EFFECTIVE_INTER_DIM / K_STEP_PACKED;
    static constexpr int DEFAULT_BLOCK_SIZE =
        !DIRECT_ATOMIC_OUT && IS_BM64_BN256 && K_TILES >= 4
            ? 128
            : opus_moe::kStage2A8W4DecodeDefaultCtaThreads;
    static constexpr int BLOCK_SIZE =
        BlockThreadsOverride > 0 ? BlockThreadsOverride : DEFAULT_BLOCK_SIZE;
    static constexpr int T_N = (BLOCK_SIZE / opus::get_warp_size()) / T_M;
    // Route-out K>=4 BM64 variants need looser launch resources to avoid
    // pressure from the wider decode mainloop; K3 keeps the tuned resources.
    static constexpr int ROUTE_OUT_MAX_MIN_BLOCKS_PER_CU =
        IS_BM64_BN256 && K_TILES >= 4 ? 2 : 4;
    static constexpr int DEFAULT_ROUTE_OUT_MIN_BLOCKS_PER_CU =
        ROUTE_OUT_MAX_MIN_BLOCKS_PER_CU;
    static constexpr int DEFAULT_MIN_BLOCKS_PER_CU =
        !DIRECT_ATOMIC_OUT ? DEFAULT_ROUTE_OUT_MIN_BLOCKS_PER_CU : (IS_BM16 ? 4 : 2);
    static constexpr int REQUESTED_MIN_BLOCKS_PER_CU =
        MinBlocksPerCuOverride > 0 ? MinBlocksPerCuOverride : DEFAULT_MIN_BLOCKS_PER_CU;
    static constexpr int MIN_BLOCKS_PER_CU =
        !DIRECT_ATOMIC_OUT && REQUESTED_MIN_BLOCKS_PER_CU > ROUTE_OUT_MAX_MIN_BLOCKS_PER_CU
            ? ROUTE_OUT_MAX_MIN_BLOCKS_PER_CU
            : REQUESTED_MIN_BLOCKS_PER_CU;
    static constexpr int A_LDS_STAGE_ELEMS = B_M * K_STEP_PACKED;
    static constexpr int A_LDS_FULL_BYTES =
        K_TILES * A_LDS_STAGE_ELEMS * static_cast<int>(sizeof(D_A));
    static constexpr int A_LDS_FULL_RESERVED_BYTES =
        B_M * (2 * static_cast<int>(sizeof(int32_t)) + static_cast<int>(sizeof(float)));
    static constexpr int A_LDS_STREAM_STAGES = 2;
    // Keep the A staging buffer and route metadata within gfx950's 160 KiB
    // workgroup LDS budget.
    static constexpr int A_LDS_STAGES =
        A_LDS_FULL_BYTES <= 160 * 1024 - A_LDS_FULL_RESERVED_BYTES
            ? K_TILES
            : A_LDS_STREAM_STAGES;
    static constexpr int SCALE_GROUP_LOGICAL_K = opus_moe::kStage2A8W4DecodeScaleGroupLogicalK;
    static constexpr int DECODE_SCALE_GROUPS = DECODE_LOGICAL_INTER_DIM / SCALE_GROUP_LOGICAL_K;
    static constexpr int SCALE_GROUPS_PER_ROW_PACK = (K_TILES + 1) / 2;
    static constexpr int SCALE_WORDS_PER_GROUP_PACK = opus_moe::kStage2A8W4DecodeScaleWordsPerGroupPack;
    static constexpr int SCALE_WORDS_PER_ROW_PACK = SCALE_GROUPS_PER_ROW_PACK * SCALE_WORDS_PER_GROUP_PACK;
    static constexpr int SCALE_ROWS_PER_ROW_PACK = 2 * MMA_M;
    static constexpr int B_PAYLOAD_KLANE_STRIDE_BYTES = B_K_LOGICAL;
    static constexpr int B_PAYLOAD_K_STRIDE_BYTES = BYTES_PER_VEC / opus_moe::kStage2A8W4DecodeFp4ValuesPerByte;
    static constexpr int B_THREADGROUP_STRIDE_BYTES = THREADS_K * B_PAYLOAD_KLANE_STRIDE_BYTES;

    static constexpr int M_MFMA_PER_WAVE = B_M / (T_M * MMA_M);
    static constexpr int N_MFMA_PER_WAVE = B_N / (T_N * MMA_N);
    static constexpr int HALF_N_MFMA_PER_WAVE = N_MFMA_PER_WAVE / 2;
    static constexpr int A_LDS_BUFFER_LOAD_INSTS = (K_TILES - 1) * M_MFMA_PER_WAVE;
    static constexpr int C_LDS_N = B_N, VEC_C = opus_moe::kStage2A8W4DecodeCVec;
    static constexpr int ELEM_PER_ATOMIC = opus_moe::kStage2A8W4DecodeCValuesPerAtomic;

    using D_ACC = opus::fp32_t;
    using D_MFMA_A = opus::fp8_t;
    using D_MFMA_B = opus::fp4_t;

    static_assert(BYTES_PER_VEC == opus_moe::kStage2A8W4DecodeVectorBytes);
    static_assert(VEC_A * static_cast<int>(sizeof(D_A)) == BYTES_PER_VEC);
    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N);
    static_assert(THREADS_K == K_STEP_PACKED / (2 * VEC_A));
    static_assert(THREADS_K == K_STEP_PACKED / (2 * B_BYTES_PER_VEC));
    static_assert(DECODE_LOGICAL_INTER_DIM % opus_moe::kStage2A8W4DecodeFp4ValuesPerByte == 0);
    static_assert((B_PAYLOAD_KLANE_STRIDE_BYTES & (B_PAYLOAD_KLANE_STRIDE_BYTES - 1)) == 0);
    static_assert((B_THREADGROUP_STRIDE_BYTES & (B_THREADGROUP_STRIDE_BYTES - 1)) == 0);
    static_assert(B_M % (T_M * MMA_M) == 0);
    static_assert(B_N % (T_N * MMA_N) == 0);
    static_assert(SORT_BLOCK_M > 0);
    static_assert(SORT_BLOCK_M % B_M == 0);
    static_assert(DECODE_EFFECTIVE_INTER_DIM == K_TILES * K_STEP_PACKED);
    static_assert(K_TILES > 0);
    static_assert(DECODE_LOGICAL_INTER_DIM % SCALE_GROUP_LOGICAL_K == 0);
    static_assert(DECODE_SCALE_GROUPS <=
                  SCALE_GROUPS_PER_ROW_PACK *
                      opus_moe::kStage2A8W4DecodeScaleGroupsPerRowPack);
};
