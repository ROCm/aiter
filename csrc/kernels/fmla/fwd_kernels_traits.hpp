// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// =====================================================================================================================
// Kernel traits
//
enum class XqaStrategy
{
    Disable,  // disable xqa
    External, // enable xqa by torch.Tensor transform
    Internal  // enable xqa by tensor view transform
};

/// TODO: combine it with decode trait.
template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN0_,
          int32_t kBlockN1_,
          int32_t kNumWarps_,
          int32_t kWaveOccupancy_,
          bool    kKVLoadOnce_      = false,
          XqaStrategy kXqaStrategy_ = XqaStrategy::Internal>
struct FlashMlaPrefillKernelTrait
{
    static constexpr int32_t kSizeD                     = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                    = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kSizeNope                  = 512;  //tmp force 512
    static constexpr int32_t kSizeRope                  = kSizeD - kSizeNope;
    static constexpr int32_t kNumWarps                  = kNumWarps_;
    static constexpr int32_t kPageBlockSize             = 16;
    static constexpr int32_t kNumThreads                = kNumWarps * ck_tile::get_warp_size();
    static constexpr int32_t kWaveOccupancy             = kWaveOccupancy_;
    static constexpr int32_t kNumWarpsSoftmax           = 4;
    static constexpr int32_t kNumThreadsSoftmax         = kNumWarpsSoftmax * ck_tile::get_warp_size();
    static constexpr int32_t kNumWarpsCombine           = 4;
    static constexpr int32_t kNumThreadsCombine         = kNumWarpsCombine * ck_tile::get_warp_size();
    static constexpr int32_t kBlockM                    = kBlockM_;
    static constexpr int32_t kBlockN0                   = kBlockN0_;
    static constexpr int32_t kBlockK0                   = kKVLoadOnce_ ? kSizeD : 32;
    static constexpr int32_t kBlockN1                   = kBlockN1_;
    static constexpr int32_t kBlockK1                   = (kNumWarps == 8) ? kBlockN0 : 16; // TODO: make it tunable once slice tile issue is fixed.
    static constexpr int32_t kFixedOverheadNumBlocks    = 5;
    static constexpr int32_t kMaxBatchSize              = 4096;
    static constexpr int32_t kCuReuse                   = 2;
    static constexpr int32_t kMaxSplits                 = 128;
    static constexpr int32_t kKNumRepeat                = kSizeNope / kMaxSplits;
    static constexpr bool    kPadHeadDimQ               = false;
    static constexpr bool    kPadHeadDimV               = false;
    static constexpr bool    kPadSeqLenQ                = true;
    static constexpr bool    kPadSeqLenK                = true;
    static constexpr bool    kKVLoadOnce                = kKVLoadOnce_;
    static constexpr XqaStrategy   kXqaStrategy         = kXqaStrategy_;

    static constexpr int32_t kNumPrefetchK  = kKVLoadOnce_ ? 2 : 1;
    static constexpr int32_t kNumPrefetchV  = 1;
    static constexpr int32_t kNumPrefetchKV = ck_tile::max(kNumPrefetchK, kNumPrefetchV);

    using QKWarpTile = std::conditional_t<kKVLoadOnce, ck_tile::sequence<16, 16, 32>,
                                                       ck_tile::sequence<16, 16, 16>>;
    using KVWarpTile = ck_tile::sequence<16, 16, 16>;

    // Special settings for GEMM_0/QK_GEMM especially for the 8 waves since half of waves are idle
    static constexpr int32_t kWarpGemm0M = QKWarpTile::at(ck_tile::number<0>{});
    static constexpr int32_t kWarpGemm0N = QKWarpTile::at(ck_tile::number<1>{});
    static_assert((kWarpGemm0M == 4) || (kWarpGemm0M == 16) || (kWarpGemm0M == 32));
    static_assert((kWarpGemm0N == 4) || (kWarpGemm0N == 16) || (kWarpGemm0N == 32));

    static constexpr int32_t Gemm0ColWarps    = kBlockM  / kWarpGemm0M;    // #warp in a column
    static constexpr int32_t Gemm0MaxRowWarps = kBlockN0 / kWarpGemm0N;    // #warp in a row based on block size
    static constexpr int32_t Gemm0ResRowWarps = kNumWarps / Gemm0ColWarps; // residual #warp can be used to handle a row
    static constexpr int32_t Gemm0RowWarps    = ck_tile::min(Gemm0ResRowWarps, Gemm0MaxRowWarps);
    static_assert((kNumWarps >= Gemm0ColWarps) && ((kNumWarps % Gemm0ColWarps) == 0));

    static constexpr int32_t kNumWarpsGemm0   = Gemm0RowWarps * Gemm0ColWarps;
    static constexpr int32_t kNumThreadsGemm0 = kNumWarpsGemm0 * ck_tile::get_warp_size();

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

