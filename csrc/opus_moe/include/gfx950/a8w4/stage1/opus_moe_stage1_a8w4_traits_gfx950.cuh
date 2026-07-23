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
    const float* __restrict__ w1_bias;
    const int32_t* __restrict__ sorted_token_ids;
    const int32_t* __restrict__ sorted_expert_ids;
    const int32_t* __restrict__ num_valid_ids;
    uint8_t* __restrict__ inter_states_fp8;
    uint8_t* __restrict__ inter_states_scale_e8m0;

    int64_t stride_hidden_t;
    int64_t stride_w1_e;
    int64_t stride_w1_bias_e;
    int64_t stride_out_t;
    int64_t stride_out_k;
    int64_t stride_out_scale_route;

    int token_num;
    int topk;
    int num_experts;
    int inter_dim;
    int hidden_scale_cols;
    int k_steps;
};

constexpr int kScaleGroupLogicalK =
    opus_moe::kOpusMoeStage1A8W4ScaleGroupLogicalK;
constexpr int kFp4ValuesPerByte = 2, kVectorBytes = 16;
constexpr int kMfmaM = 16, kMfmaN = 16, kMfmaK = 128;
constexpr int kScaleMnPack = 2, kScaleKPack = 2;
constexpr int kDefaultCtaThreads = 256;
constexpr int kMinBlocksPerCu = 2;
constexpr int kGfx950LdsBytes = 163840;
constexpr int kDedicatedEpilogueScratchLdsLimit = 48 * 1024;

enum class Stage1Activation
{
    Silu,
    Swiglu,
};

template<int BlockM,
         int BlockN,
         typename Policy>
struct OpusMoeStage1A8W4Shape
{
    static constexpr int B_M = BlockM, B_N = BlockN;
    static constexpr int MMA_M = kMfmaM, MMA_N = kMfmaN, MMA_K = kMfmaK;
    static constexpr int M_MFMA_PER_WAVE = B_M / kMfmaM;
    static constexpr int SCALE_MN_PACK = kScaleMnPack;
    static constexpr int SCALE_K_PACK = kScaleKPack;
    static constexpr int BYTES_PER_VEC = kVectorBytes;
    static constexpr int WAVE_SIZE = opus::get_warp_size();
    static constexpr int K_WAVE = Policy::K_WAVE;
    static constexpr bool GATE_UP_GROUP_SPLIT = Policy::GATE_UP_GROUP_SPLIT;
    static constexpr bool WEIGHT_LOAD_STREAM = Policy::WEIGHT_LOAD_STREAM;
    static constexpr int XCD_SWIZZLE = Policy::XCD_SWIZZLE;
    static constexpr int KWAVE_BASE_WAVES =
        GATE_UP_GROUP_SPLIT ? Policy::BLOCK_THREADS / WAVE_SIZE : 2;
    static constexpr int BLOCK_SIZE =
        K_WAVE == 1 ? Policy::BLOCK_THREADS :
            KWAVE_BASE_WAVES * K_WAVE * WAVE_SIZE;
    static constexpr int MIN_BLOCKS_PER_CU =
        Policy::MIN_BLOCKS_PER_CU_OVERRIDE > 0 ?
        Policy::MIN_BLOCKS_PER_CU_OVERRIDE :
        K_WAVE == 1 ? kMinBlocksPerCu :
        1;
    static constexpr Stage1Activation ACTIVATION = Policy::ACTIVATION;

    static constexpr int SCALE_GROUP_LOGICAL_K = kScaleGroupLogicalK;
    static constexpr int OUTPUT_SCALE_GROUPS_PER_TILE =
        GATE_UP_GROUP_SPLIT ? (B_N / 2) / SCALE_GROUP_LOGICAL_K : 1;
    static constexpr int ACC_SCALE_GROUPS_PER_TILE =
        GATE_UP_GROUP_SPLIT ? OUTPUT_SCALE_GROUPS_PER_TILE : 2;
    static constexpr int OUTPUT_COLS_PER_TILE =
        OUTPUT_SCALE_GROUPS_PER_TILE * SCALE_GROUP_LOGICAL_K;
    static constexpr int EPILOGUE_ROW_SPLIT = GATE_UP_GROUP_SPLIT ? 2 : 1;
    static constexpr int EPILOGUE_ROWS_PER_PASS = B_M / EPILOGUE_ROW_SPLIT;
    static constexpr int EPILOGUE_SMEM_COLS =
        OUTPUT_SCALE_GROUPS_PER_TILE * SCALE_GROUP_LOGICAL_K * 2;
    static constexpr int QUANT_GROUP_BLOCKS = Policy::QUANT_GROUP_BLOCKS;
    static constexpr int QUANT_ACTIVE_THREADS = EPILOGUE_ROWS_PER_PASS * 2 * QUANT_GROUP_BLOCKS;
    static constexpr int QUANT_GROUPS_PER_THREAD = OUTPUT_SCALE_GROUPS_PER_TILE / QUANT_GROUP_BLOCKS;
    static constexpr int EPILOGUE_SMEM_BYTES =
        EPILOGUE_ROWS_PER_PASS * EPILOGUE_SMEM_COLS *
        static_cast<int>(sizeof(float));
    static constexpr int A_REG_LDS_STAGE_BYTES = B_M * MMA_K;
    static constexpr int KWAVE_REDUCE_BYTES =
        K_WAVE == 1 ? 0 : K_WAVE * KWAVE_BASE_WAVES *
                              M_MFMA_PER_WAVE *
                              ACC_SCALE_GROUPS_PER_TILE * WAVE_SIZE *
                              static_cast<int>(sizeof(float)) * 4;
    static constexpr int MAINLOOP_SCRATCH_BYTES =
        2 * SCALE_K_PACK * A_REG_LDS_STAGE_BYTES > KWAVE_REDUCE_BYTES ?
            2 * SCALE_K_PACK * A_REG_LDS_STAGE_BYTES :
            KWAVE_REDUCE_BYTES;
    static constexpr bool DEDICATED_EPILOGUE_SCRATCH =
        GATE_UP_GROUP_SPLIT &&
        MAINLOOP_SCRATCH_BYTES + EPILOGUE_SMEM_BYTES <=
            kDedicatedEpilogueScratchLdsLimit;
    static constexpr int EPILOGUE_SCRATCH_OFFSET =
        DEDICATED_EPILOGUE_SCRATCH ? MAINLOOP_SCRATCH_BYTES : 0;
    static constexpr int SHARED_SCRATCH_BYTES =
        DEDICATED_EPILOGUE_SCRATCH ?
            MAINLOOP_SCRATCH_BYTES + EPILOGUE_SMEM_BYTES :
        EPILOGUE_SMEM_BYTES > MAINLOOP_SCRATCH_BYTES ?
            EPILOGUE_SMEM_BYTES :
            MAINLOOP_SCRATCH_BYTES;
    static constexpr bool SKIP_INVALID_A_SCALE_GUARD =
        Policy::SKIP_INVALID_A_SCALE_GUARD;
    static constexpr int B_GROUPS_PER_WAVE =
        GATE_UP_GROUP_SPLIT ?
            OUTPUT_SCALE_GROUPS_PER_TILE / (KWAVE_BASE_WAVES / 2) : 1;
    static constexpr int B_ITEMS_PER_GROUP = 2;
    static constexpr int B_ITEMS_PER_WAVE = B_GROUPS_PER_WAVE * B_ITEMS_PER_GROUP;

    static constexpr int HALF_SCALE_GROUP = SCALE_GROUP_LOGICAL_K / 2;
    static constexpr int M_SCALE_PACKS =
        (M_MFMA_PER_WAVE + SCALE_MN_PACK - 1) / SCALE_MN_PACK;
    static constexpr int SCALE_LAYOUT_STRIDE_K0 = 4 * MMA_M;
    static constexpr int W1_PAYLOAD_K_STEP_STRIDE_BYTES = (64 / MMA_N) * MMA_N * BYTES_PER_VEC;

    using D_ACC = opus::fp32_t;
    using D_MFMA_A = opus::fp8_t;
    using D_MFMA_B = opus::fp4_t;

    static_assert(K_WAVE == 1 || OUTPUT_SCALE_GROUPS_PER_TILE == 1);
    static_assert(!GATE_UP_GROUP_SPLIT || B_M >= 32);
    static_assert(!GATE_UP_GROUP_SPLIT || (B_GROUPS_PER_WAVE > 0 && OUTPUT_SCALE_GROUPS_PER_TILE % (KWAVE_BASE_WAVES / 2) == 0));
    static_assert(SCALE_GROUP_LOGICAL_K % (2 * sizeof(uint32_t)) == 0);
    static_assert(B_M % EPILOGUE_ROW_SPLIT == 0);
    static_assert(EPILOGUE_ROWS_PER_PASS % MMA_M == 0);
    static_assert(OUTPUT_SCALE_GROUPS_PER_TILE % QUANT_GROUP_BLOCKS == 0);
    static_assert(QUANT_ACTIVE_THREADS <= BLOCK_SIZE);
    static_assert(SHARED_SCRATCH_BYTES <= kGfx950LdsBytes);
    static_assert(GATE_UP_GROUP_SPLIT || M_MFMA_PER_WAVE <= 2);
};

template<bool GateUpGroupSplit = false,
         int KWave = 1,
         int MinBlocksPerCuOverride = 0,
         bool SkipInvalidAScaleGuard = false,
         int QuantGroupBlocks = 1,
         Stage1Activation Activation = Stage1Activation::Silu,
         int BlockThreads = kDefaultCtaThreads,
         bool WeightLoadStream = false,
         int XcdSwizzle = 0>
struct OpusMoeStage1A8W4Policy
{
    static constexpr bool GATE_UP_GROUP_SPLIT = GateUpGroupSplit;
    static constexpr int K_WAVE = KWave;
    static constexpr int MIN_BLOCKS_PER_CU_OVERRIDE = MinBlocksPerCuOverride;
    static constexpr bool SKIP_INVALID_A_SCALE_GUARD = SkipInvalidAScaleGuard;
    static constexpr int QUANT_GROUP_BLOCKS = QuantGroupBlocks;
    static constexpr Stage1Activation ACTIVATION = Activation;
    static constexpr int BLOCK_THREADS = BlockThreads;
    static constexpr bool WEIGHT_LOAD_STREAM = WeightLoadStream;
    static constexpr int XCD_SWIZZLE = XcdSwizzle;
};

} // namespace stage1_a8w4
} // namespace opus_moe
