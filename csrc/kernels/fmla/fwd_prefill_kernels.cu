// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_scatter_gather.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/ops/reduce.hpp>
#include "block_gemm_areg_bsmem_creg.hpp"

// =====================================================================================================================
// Utils
//
// #define FMLA_FWD_FAST_EXP2 1
#define DEBUG_ONE_KERNEL 0
#define HEADV 512

enum class XqaStrategy
{
    Disable,  // disable xqa
    External, // enable xqa by torch.Tensor transform
    Internal  // enable xqa by tensor view transform
};

CK_TILE_DEVICE bool IsDebugThreadBlock(const int x = 0, const int y = 0, const int z = 0)
{
    return blockIdx.x == x && blockIdx.y == y && blockIdx.z == z;
}

// Returns count of warps which don't contain any idle thread.
template <int32_t NumWarps, int32_t M, int32_t N>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
{
    static_assert((NumWarps > 0) && ((NumWarps & (NumWarps - 1)) == 0)); // NumWarps should be power of 2
    constexpr int32_t ElemPerThread = (M * N) / (NumWarps * ck_tile::get_warp_size());
    if constexpr(0 < ElemPerThread)
    {
        return NumWarps;
    }
    else
    {
        return GetMaxNumWarpsForTile<NumWarps / 2, M, N>();
    }
}

// Returns vector size for given warp count for handing the specified matrix.
template <int32_t NumWarps, int32_t M, int32_t N, typename DataType>
CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
{
    constexpr int32_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N>();
    constexpr int32_t ElemPerThread = (M * N) / (MaxNumWarps * ck_tile::get_warp_size());
    constexpr int32_t MaxNPerThread = 16 / sizeof(DataType);
    return ck_tile::min(MaxNPerThread, ElemPerThread);
}

// =====================================================================================================================
// Definitions and helper structures
//

/// TODO: combine it with decode trait.
template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN0_,
          int32_t kBlockN1_,
          int32_t kNumWarps_,
          int32_t kWaveOccupancy_,
          bool    kKVLoadOnce_,
          XqaStrategy kXqaStrategy_>
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

template<typename Traits_, typename scalar_t, typename acc_t>
struct FlashMlaPrefillPolicy
{
public:
    using Traits = Traits_;
    using InOutType = scalar_t;
    using AccType   = acc_t;

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentQ()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreadsGemm0;
        constexpr int32_t kMPerBlock = Traits::kBlockM;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(InOutType);

        // this should align with MakeQDramTileDistribution()
        constexpr int32_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return ck_tile::min(ElemPerThread, MaxVectorSize);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentK()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreadsGemm0;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(InOutType);
        constexpr int32_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        return ck_tile::min(MaxVectorSize, ElemPerThread);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentV()
    {
        // Assuming Value is row-major just like Key.
        constexpr int32_t kBlockSize   = Traits::kNumThreads;
        constexpr int32_t kNPerBlock   = Traits::kBlockN1;
        constexpr int32_t kKPerBlock   = Traits::kBlockK1;
        constexpr int32_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        constexpr int32_t kMaxVecLoad =
            ck_tile::min(total_pixels, static_cast<int32_t>(16 / sizeof(InOutType)));
        constexpr int32_t kMinVecLoad = 4 / sizeof(InOutType);

        constexpr int32_t kVecLoad =
            ((total_pixels / kMaxVecLoad) >= kMinVecLoad)
                ? kMaxVecLoad
                : (total_pixels / kMinVecLoad);

        return kVecLoad;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentOacc()
    {
        int32_t result = 1;

        if constexpr (Traits::kPadHeadDimV == false)
        {
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kMPerBlock = Traits::kBlockM;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;

            constexpr int32_t M1 = kBlockSize / ck_tile::get_warp_size();
            constexpr int32_t M2 = ck_tile::min(kMPerBlock / M1, ck_tile::get_warp_size());
            constexpr int32_t N0 = ck_tile::get_warp_size() / M2;
            constexpr int32_t N1 = kNPerBlock / N0;

            // Each thread cannot handle more than 16 bytes
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(InOutType)));
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentO()
    {
        int32_t result = 1;

        if constexpr (Traits::kPadHeadDimV == false)
        {
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kMPerBlock = Traits::kBlockM;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;

            constexpr int32_t M1 = kBlockSize / ck_tile::get_warp_size();
            constexpr int32_t M2 = ck_tile::min(kMPerBlock / M1, ck_tile::get_warp_size());
            constexpr int32_t N0 = ck_tile::get_warp_size() / M2;
            constexpr int32_t N1 = kNPerBlock / N0;

            // Each thread cannot handle more than 16 bytes
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(AccType)));
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLse()
    {
        return GetVectorSizeForTile<Traits::kNumWarps,
                                    Traits::kMaxSplits,
                                    Traits::kBlockM,
                                    AccType>();
    }

    template<bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        constexpr int32_t BlockTileK = !Traits::kKVLoadOnce ? Traits::kBlockK0 :
                                       IsLoadOnceRope ? Traits::kSizeRope : Traits::kSizeNope;
        using BlockTile     = ck_tile::sequence<Traits::kBlockM, Traits::kBlockN0, BlockTileK>;
        using BlockWarps    = ck_tile::sequence<Traits::Gemm0ColWarps, Traits::Gemm0RowWarps, 1>;
        using TileGemmShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, typename Traits::QKWarpTile>;

        using GemmProblem =
            ck_tile::BlockGemmProblem<InOutType, InOutType, AccType, Traits::kNumThreadsGemm0, TileGemmShape>;

        constexpr auto warp_gemm = []()
        {
            if constexpr (std::is_same_v<InOutType, ck_tile::fp16_t> && std::is_same_v<AccType, float>)
            {
                if constexpr(Traits::kWarpGemm0M == 32)
                    return ck_tile::WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(Traits::kWarpGemm0M == 16)
                    return ck_tile::WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // kWarpGemmM == 4
                    return ck_tile::WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr (std::is_same_v<InOutType, ck_tile::bf16_t> && std::is_same_v<AccType, float>)
            {
                if constexpr (Traits::kWarpGemm0M == 32)
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr (Traits::kWarpGemm0M == 16)
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // kWarpGemmM == 4
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
        }();

        using WarpGemm = ck_tile::remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegPolicy<InOutType, InOutType, AccType, BlockWarps, WarpGemm>;

        return BlockGemmARegBSmemCReg<GemmProblem, BlockGemmPolicy>{};
    }

    CK_TILE_DEVICE static bool HandleGemm0()
    {
        bool result = true;

        if constexpr (Traits::Gemm0ResRowWarps > Traits::Gemm0MaxRowWarps)
        {
            constexpr int32_t MaxWarpIdx = Traits::kNumWarpsGemm0 - 1;
            if (ck_tile::get_warp_id() > MaxWarpIdx)
            {
                result = false;
            }
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        constexpr int32_t kWarpGemmM = Traits::KVWarpTile::at(ck_tile::number<0>{});
        static_assert((kWarpGemmM == 4) || (kWarpGemmM == 16) || (kWarpGemmM == 32));

        constexpr int32_t ColWarps = Traits::kBlockM / kWarpGemmM;
        constexpr int32_t RowWarps = Traits::kNumWarps / ColWarps;
        static_assert((Traits::kNumWarps >= ColWarps) && ((Traits::kNumWarps % ColWarps) == 0));

        using BlockTile     = ck_tile::sequence<Traits::kBlockM, Traits::kBlockN1, Traits::kBlockK1>;
        using BlockWarps    = ck_tile::sequence<ColWarps, RowWarps, 1>;
        using TileGemmShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, typename Traits::KVWarpTile>;

        using GemmProblem = ck_tile::BlockGemmProblem<InOutType, InOutType, AccType, Traits::kNumThreads, TileGemmShape>;

        auto warp_gemm = []()
        {
            if constexpr(std::is_same_v<InOutType, ck_tile::fp8_t> && std::is_same_v<AccType, float>)
            {
                return ck_tile::WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
            }
            else
            {
                return ck_tile::WarpGemmMfmaDispatcher<
                    InOutType, InOutType, AccType,
                    Traits::KVWarpTile::at(ck_tile::number<0>{}),
                    Traits::KVWarpTile::at(ck_tile::number<1>{}),
                    Traits::KVWarpTile::at(ck_tile::number<2>{}),
                    true>{};
            }
        }();

        using WarpGemm = ck_tile::remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegPolicy<InOutType, InOutType, AccType, BlockWarps, WarpGemm>;

        return BlockGemmARegBSmemCReg<GemmProblem, BlockGemmPolicy>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;
        constexpr int32_t kKPack     = 16 / sizeof(InOutType);

        constexpr auto k_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kNPerBlock>{}, ck_tile::number<kKPack>{}),
            ck_tile::make_tuple(ck_tile::number<(kNPerBlock + 1) * kKPack>{}, ck_tile::number<kKPack>{}, ck_tile::number<1>{}),
            ck_tile::number<kKPack>{},
            ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
            k_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_pass_through_transform(ck_tile::number<kNPerBlock>{}),
                ck_tile::make_merge_transform(make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0, 2>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetKNopeSingleRepeatSize()
    {
        return (Traits::kSizeRope * 2 + 8) * sizeof(scalar_t); // 2 = lane group 8 = padsize
    }

    template <bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleKElementSpaceSize()
    {
        constexpr ck_tile::index_t kNPerBlock = Traits::kBlockN0;
        constexpr ck_tile::index_t kKPerBlock = IsLoadOnceRope ? Traits::kSizeRope : Traits::kMaxSplits;
        constexpr ck_tile::index_t kRepeatK = IsLoadOnceRope ? 1 : Traits::kKNumRepeat;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t kKPack  = 16 / sizeof(scalar_t);
        constexpr ck_tile::index_t KVector = 4  / sizeof(scalar_t);
        constexpr ck_tile::index_t kPad    = kKPack;

        static_assert(warpSize * KVector >= kKPerBlock &&
                      warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK  = kKPerBlock / KVector;
        constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK;
        constexpr ck_tile::index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

        return NumIssues * NumWarps * kRepeatK * (warpSize * KVector + kPad);
    }

    template <bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleKSpaceSize()
    {
        return GetSingleKElementSpaceSize<IsLoadOnceRope>() * sizeof(scalar_t);
    }

    template <bool IsLoadOnceRope = false,
              ck_tile::index_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
        MakeKLdsStoreBlockDescriptor(ck_tile::number<IBuf> = ck_tile::number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr ck_tile::index_t kNPerBlock = Traits::kBlockN0;
        constexpr ck_tile::index_t kKPerBlock = IsLoadOnceRope ? Traits::kSizeRope : Traits::kMaxSplits;
        constexpr ck_tile::index_t kRepeatK = IsLoadOnceRope ? 1 : Traits::kKNumRepeat;
        constexpr ck_tile::index_t kBlockSize = Traits::kNumThreads;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t kKPack  = 16 / sizeof(scalar_t);
        constexpr ck_tile::index_t KVector = 4  / sizeof(scalar_t);
        constexpr ck_tile::index_t kPad = 
            kKPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK =
            kKPerBlock / KVector; // 
        constexpr ck_tile::index_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr ck_tile::index_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        if constexpr(IsLoadOnceRope)
        {
            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
                ck_tile::make_tuple(ck_tile::number<NumIssues>{},  // n0
                           ck_tile::number<LaneGroups>{}, // n1
                           ck_tile::number<NumWarps>{},   // n2
                           ck_tile::number<LanesPerK>{},  // k0
                           ck_tile::number<KVector>{}),   // k1
                ck_tile::make_tuple(ck_tile::number<NumWarps * (warpSize * KVector + kPad)>{},
                           ck_tile::number<kKPerBlock>{},
                           ck_tile::number<warpSize * KVector + kPad>{},
                           ck_tile::number<KVector>{},
                           ck_tile::number<1>{}),
                ck_tile::number<IBuf * GetSingleKElementSpaceSize<true>()>{},
                ck_tile::number<KVector>{},
                ck_tile::number<1>{});

            // TODO this layout is hard coded, and will be used in async copy buffer view load
            // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
            constexpr auto k_lds_block_desc_issues_warps_lanes = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(make_pass_through_transform(ck_tile::number<NumIssues>{}),
                           ck_tile::make_pass_through_transform(ck_tile::number<NumWarps>{}),
                           ck_tile::make_merge_transform(ck_tile::make_tuple(
                               ck_tile::number<LaneGroups>{},
                               ck_tile::number<LanesPerK>{}, 
                               ck_tile::number<KVector>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{}, ck_tile::sequence<1, 3, 4>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));

            return k_lds_block_desc_issues_warps_lanes;
        }
        else
        {
            constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor_with_offset(
                ck_tile::make_tuple(ck_tile::number<NumIssues>{},  // n0
                           ck_tile::number<LaneGroups>{}, // n1
                           ck_tile::number<NumWarps>{},   // n2
                           ck_tile::number<kRepeatK>{},  // krepeatK
                           ck_tile::number<LanesPerK>{},  // k0
                           ck_tile::number<KVector>{}),   // k1
                ck_tile::make_tuple(ck_tile::number<NumWarps* kRepeatK * (warpSize * KVector + kPad)>{},
                           ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                           ck_tile::number<(warpSize * KVector + kPad) * kRepeatK>{},
                           ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                           ck_tile::number<KVector>{},
                           ck_tile::number<1>{}),
                ck_tile::number<IBuf * GetSingleKElementSpaceSize<false>()>{},
                ck_tile::number<KVector>{},
                ck_tile::number<1>{});

            // TODO this layout is hard coded, and will be used in async copy buffer view load
            // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
            constexpr auto k_lds_block_desc_issues_warps_lanes = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(make_pass_through_transform(ck_tile::number<NumIssues>{}),
                           ck_tile::make_pass_through_transform(ck_tile::number<NumWarps>{}),
                           ck_tile::make_merge_transform(ck_tile::make_tuple(
                               ck_tile::number<LaneGroups>{},
                               ck_tile::number<kRepeatK>{},
                               ck_tile::number<LanesPerK>{}, 
                               ck_tile::number<KVector>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<2>{}, ck_tile::sequence<1, 3, 4, 5>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}, ck_tile::sequence<2>{}));

            return k_lds_block_desc_issues_warps_lanes;
        }
    }

    template <bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr ck_tile::index_t kNPerBlock = Traits::kBlockN0;
        constexpr ck_tile::index_t kKPerBlock = IsLoadOnceRope ? Traits::kSizeRope : Traits::kMaxSplits;
        constexpr ck_tile::index_t kRepeatK = IsLoadOnceRope ? 1 : Traits::kKNumRepeat;
        constexpr ck_tile::index_t kBlockSize = Traits::kNumThreads;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t kKPack  = 16 / sizeof(scalar_t);
        constexpr ck_tile::index_t KVector = 4 / sizeof(scalar_t);
        constexpr ck_tile::index_t kPad    = kKPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr ck_tile::index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));
        // constexpr ck_tile::index_t SingleKSize = NumIssues * NumWarps * (warpSize * KVector + kPad);
        // constexpr ck_tile::index_t SingleVSize =
        // MakeVLdsBlockDescriptor<Problem>().get_element_space_size();
        constexpr ck_tile::index_t BufferSize =
            GetSingleKElementSpaceSize<IsLoadOnceRope>();

        if constexpr(IsLoadOnceRope)
        {
            constexpr auto k_lds_block_desc_0 =
                ck_tile::make_naive_tensor_descriptor(ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchK>{},    // num_buffers
                                                        ck_tile::number<NumIssues>{},          // n0
                                                        ck_tile::number<NumWarps>{},           // n2
                                                        ck_tile::number<LaneGroups>{},         // n1
                                                        ck_tile::number<kKPerBlock / kKPack>{}, // k0
                                                        ck_tile::number<kKPack>{}),             // k1
                                             ck_tile::make_tuple(ck_tile::number<BufferSize>{},
                                                        ck_tile::number<NumWarps*(warpSize * KVector + kPad)>{},
                                                        ck_tile::number<(warpSize * KVector + kPad)>{},
                                                        ck_tile::number<kKPerBlock>{},
                                                        ck_tile::number<kKPack>{},
                                                        ck_tile::number<1>{}),
                                             ck_tile::number<kKPack>{},
                                             ck_tile::number<1>{});

            constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchK>{},
                                                    ck_tile::number<NumIssues>{},
                                                    ck_tile::number<LaneGroups>{},
                                                    ck_tile::number<NumWarps>{})),
                    ck_tile::make_merge_transform(ck_tile::make_tuple(
                                                  ck_tile::number<kKPerBlock / kKPack>{}, 
                                                  ck_tile::number<kKPack>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0, 1, 3, 2>{}, ck_tile::sequence<4, 5>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
            return k_lds_block_desc;
        }
        else
        {
            constexpr auto k_lds_block_desc_0 =
                ck_tile::make_naive_tensor_descriptor(ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchK>{},    // num_buffers
                                                        ck_tile::number<NumIssues>{},          // n0
                                                        ck_tile::number<NumWarps>{},           // n2
                                                        ck_tile::number<LaneGroups>{},         // n1
                                                        ck_tile::number<kRepeatK>{}, 
                                                        ck_tile::number<kKPerBlock / kKPack>{}, // k0
                                                        ck_tile::number<kKPack>{}),             // k1
                                                     ck_tile::make_tuple(ck_tile::number<BufferSize>{},
                                                        ck_tile::number<NumWarps*(warpSize * KVector + kPad) * kRepeatK>{},
                                                        ck_tile::number<(warpSize * KVector + kPad) * kRepeatK>{},
                                                        ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                                                        // ck_tile::number<Traits::kSizeD>{}, 
                                                        ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                                                        ck_tile::number<kKPack>{},
                                                        ck_tile::number<1>{}),
                                             ck_tile::number<kKPack>{},
                                             ck_tile::number<1>{});

            constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchK>{},
                                                    ck_tile::number<NumIssues>{},
                                                    ck_tile::number<LaneGroups>{},
                                                    ck_tile::number<NumWarps>{})),
                    ck_tile::make_merge_transform(ck_tile::make_tuple(
                                                  ck_tile::number<kRepeatK>{}, 
                                                  ck_tile::number<kKPerBlock / kKPack>{}, 
                                                  ck_tile::number<kKPack>{}))),
                ck_tile::make_tuple(ck_tile::sequence<0, 1, 3, 2>{}, ck_tile::sequence<4, 5, 6>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
            return k_lds_block_desc;
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr int32_t Banks        = 32; /// TODO: need change based on arch
        constexpr int32_t PixelsPerRow = Banks * 4 / sizeof(InOutType);
        constexpr int32_t kKPack       = 16 / sizeof(InOutType);
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr int32_t NPerRow    = PixelsPerRow / kKPack;
        constexpr int32_t kNPerBlock = Traits::kBlockN1;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);
        constexpr int32_t BufferSize = Traits::kKVLoadOnce ? GetSmemSizeSingleV() : GetSmemSizeSingleKV();

        constexpr auto v_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchV>{},
                       ck_tile::number<kKPerBlock / kKPack>{},
                       ck_tile::number<kNPerBlock / NPerRow>{},
                       ck_tile::number<NPerRow>{},
                       ck_tile::number<kKPack>{}),
            ck_tile::make_tuple(ck_tile::number<BufferSize>{},
                       ck_tile::number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       ck_tile::number<PixelsPerRow + kKPack>{},
                       ck_tile::number<kKPack>{},
                       ck_tile::number<1>{}),
            ck_tile::number<kKPack>{},
            ck_tile::number<1>{});

        constexpr auto v_lds_block_desc = ck_tile::transform_tensor_descriptor(
            v_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_merge_transform(ck_tile::make_tuple(
                    ck_tile::number<Traits::kNumPrefetchV>{}, ck_tile::number<kNPerBlock / NPerRow>{}, ck_tile::number<NPerRow>{})),
                ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0, 2, 3>{}, ck_tile::sequence<1, 4>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return v_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsStoreBlockDescriptor()
    {
        constexpr ck_tile::index_t kNPerBlock = 512;
        constexpr ck_tile::index_t kKPerBlock = 16;

        constexpr ck_tile::index_t kKPack  = 16 / sizeof(scalar_t);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kNPerBlock>{}, ck_tile::number<kKPack>{}),
            ck_tile::make_tuple(ck_tile::number<(kNPerBlock + 1) * kKPack>{}, ck_tile::number<kKPack>{}, ck_tile::number<1>{}),
            ck_tile::number<8>{},
            ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            ck_tile::make_tuple(
                make_pass_through_transform(ck_tile::number<kNPerBlock>{}),
                make_merge_transform(ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0, 2>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsLoadBlockDescriptor()
    {
        constexpr ck_tile::index_t kNPerBlock = 512;
        constexpr ck_tile::index_t kKPerBlock = 16;
        constexpr ck_tile::index_t kBlockSize = Traits::kNumThreads;
        constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
        constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

        constexpr ck_tile::index_t kKPack  = 16 / sizeof(scalar_t);
        constexpr ck_tile::index_t KVector = 8 / sizeof(scalar_t); // 4
        constexpr ck_tile::index_t kPad    = kKPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr ck_tile::index_t LanesPerK  = kKPerBlock / KVector; // 16 / 4 = 4
        constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK; // 64 / 4 = 16
        constexpr ck_tile::index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps); // 1
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr auto v_lds_block_desc_0 =
            ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<NumIssues>{},          // n0
                                    ck_tile::number<NumWarps>{},           // n2
                                    ck_tile::number<LaneGroups>{},         // n1
                                    ck_tile::number<kKPerBlock / kKPack>{}, // k0
                                    ck_tile::number<kKPack>{}),             // k1
                ck_tile::make_tuple(ck_tile::number<NumWarps*(warpSize * KVector + kPad)>{},
                                    ck_tile::number<(warpSize * KVector + kPad)>{},
                                    ck_tile::number<kKPerBlock>{},
                                    ck_tile::number<kKPack>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<kKPack>{},
                ck_tile::number<1>{});

        constexpr auto k_lds_block_desc = ck_tile::transform_tensor_descriptor(
            v_lds_block_desc_0,
            ck_tile::make_tuple(
                ck_tile::make_merge_transform(
                    ck_tile::make_tuple(ck_tile::number<NumIssues>{},
                                        ck_tile::number<LaneGroups>{},
                                        ck_tile::number<NumWarps>{})),
                ck_tile::make_merge_transform(
                    ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, 
                                        ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0, 2, 1>{}, ck_tile::sequence<3, 4>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        return k_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakePShuffleLdsBlockDescriptor()
    {
        constexpr int32_t kPackSize = 16 / sizeof(AccType);

        constexpr auto p_lds_block_desc =
            ck_tile::make_naive_tensor_descriptor(
                ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0 / kPackSize>{},
                                    ck_tile::number<Traits::kBlockM>{},
                                    ck_tile::number<kPackSize>{}),
                ck_tile::make_tuple(ck_tile::number<(Traits::kBlockM + 1) * kPackSize>{},
                                    ck_tile::number<kPackSize>{},
                                    ck_tile::number<1>{}),
                ck_tile::number<kPackSize>{},
                ck_tile::number<1>{});

        constexpr auto p_lds_block_desc_merge =
            ck_tile::transform_tensor_descriptor(
                p_lds_block_desc,
                ck_tile::make_tuple(
                    ck_tile::make_pass_through_transform(ck_tile::number<Traits::kBlockM>{}),
                    ck_tile::make_merge_transform(
                        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0 / kPackSize>{},
                                            ck_tile::number<kPackSize>{}))),
                ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0, 2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return p_lds_block_desc_merge;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeSingleKV()
    {
        constexpr int32_t SingleKSize = MakeKLdsBlockDescriptor().get_element_space_size();
        constexpr int32_t SingleVSize =[&]() {
            constexpr int32_t Banks        = 32; /// TODO: need change based on arch
            constexpr int32_t PixelsPerRow = Banks * 4 / sizeof(InOutType);
            constexpr int32_t kKPack       = 16 / sizeof(InOutType);
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr int32_t NPerRow    = PixelsPerRow / kKPack;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;
            constexpr int32_t kKPerBlock = Traits::kBlockK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return ck_tile::max(SingleKSize, SingleVSize) * sizeof(InOutType);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeSingleV()
    {
        constexpr int32_t Banks        = 32; /// TODO: need change based on arch
        constexpr int32_t PixelsPerRow = Banks * 4 / sizeof(scalar_t);
        constexpr int32_t kKPack       = 16 / sizeof(scalar_t);
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr int32_t NPerRow    = PixelsPerRow / kKPack;
        constexpr int32_t kNPerBlock = Traits::kBlockN1;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack) * sizeof(scalar_t);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeK()
    {
        return Traits::kKVLoadOnce ?  
            (GetSingleKElementSpaceSize<true>() + GetSingleKElementSpaceSize<false>()) * sizeof(scalar_t) :
            Traits::kNumPrefetchKV * GetSmemSizeSingleKV();
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetPSmemStart()
    {
        return Traits::kKVLoadOnce ?  
            Traits::kNumPrefetchK * GetSmemSizeK() + Traits::kNumPrefetchV * GetSmemSizeSingleV() :
            Traits::kNumPrefetchKV * GetSmemSizeSingleKV();
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSize()
    {
        return MakePShuffleLdsBlockDescriptor().get_element_space_size() * sizeof(AccType) + GetPSmemStart();
    }


    template<bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = ck_tile::remove_cvref_t<decltype(GetQKBlockGemm<IsLoadOnceRope>())>;

        constexpr int32_t BlockTileK = !Traits::kKVLoadOnce ? Traits::kBlockK0 :
                                       IsLoadOnceRope ? Traits::kSizeRope : Traits::kSizeNope;
        return BlockGemm::template MakeABlockTileDistribution<
            Traits::kBlockM,
            BlockTileK>();
    }

    template<bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr (!Traits::kKVLoadOnce)
        {
            constexpr int32_t kBlockSize = Traits::kNumThreadsGemm0;
            constexpr int32_t kNPerBlock = Traits::kBlockN0;
            constexpr int32_t kKPerBlock = Traits::kBlockK0;

            constexpr int32_t MaxVectorSize = 16 / sizeof(InOutType);
            constexpr int32_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
            constexpr int32_t K1 = ck_tile::min(MaxVectorSize, ElemPerThread);
            constexpr int32_t K0 = kKPerBlock / K1;
            constexpr int32_t N2 = ck_tile::get_warp_size() / K0;
            constexpr int32_t N1 = kBlockSize / ck_tile::get_warp_size();
            constexpr int32_t N0 = kNPerBlock / (N2 * N1);

            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                                    ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
                                                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                                                    ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<2, 0>>,
                                                    ck_tile::sequence<1, 2>,
                                                    ck_tile::sequence<0, 1>>{});
        }
        else
        {
            // TODO: add distribution calculate
            // return ck_tile::make_static_tile_distribution(
            //     ck_tile::tile_distribution_encoding<
            //         ck_tile::sequence<>,
            //         ck_tile::tuple<ck_tile::sequence<1, 1, 8, 2>, ck_tile::sequence<9, 4, 8, 2>>,
            //         ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
            //         ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
            //         ck_tile::sequence<1, 1, 2, 2>,
            //         ck_tile::sequence<0, 3, 0, 3>>{});
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kNPerBlock = Traits::kBlockN0;
            constexpr ck_tile::index_t kKPerBlock = IsLoadOnceRope ? Traits::kSizeRope : Traits::kMaxSplits;
            constexpr ck_tile::index_t NumWarps   = Traits::kNumWarps;
            constexpr ck_tile::index_t warpSize   = ck_tile::get_warp_size();

            constexpr ck_tile::index_t KVector = 4 / sizeof(scalar_t);

            static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
            constexpr ck_tile::index_t LanesPerK  = kKPerBlock / KVector; // within a wave
            constexpr ck_tile::index_t LaneGroups = warpSize / LanesPerK; // within a wave
            constexpr ck_tile::index_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr ck_tile::index_t N0 = NumIssues;
            constexpr ck_tile::index_t N1 = LaneGroups;
            constexpr ck_tile::index_t N2 = NumWarps;
            constexpr ck_tile::index_t K0 = LanesPerK;
            constexpr ck_tile::index_t K1 = KVector;

            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                           ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
                                           ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                                           ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
                                           ck_tile::sequence<1, 2>,
                                           ck_tile::sequence<0, 1>>{});
        }
    }

    template<bool IsLoadOnceRope = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetNumRepeatOfKDramTileDistribution()
    {
        using KDstrEncode = typename decltype(MakeKDramTileDistribution<IsLoadOnceRope>())::DstrEncode;
        return KDstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVTileDistributionStride()
    {
        using VDstrEncode = typename decltype(MakeVTileDistribution())::DstrEncode;

        constexpr ck_tile::index_t VKVectors = VDstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<3>{}];
        // constexpr ck_tile::index_t VNVectors = VDstrEncode::hs_lengthss_[ck_tile::number<1>{}][ck_tile::number<3>{}];
        constexpr ck_tile::index_t VKRepeats = VKVectors / 2;
        return VKRepeats;
    }


    CK_TILE_HOST_DEVICE static constexpr auto MakeVTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                    // ck_tile::sequence<1, 1, 8, 2>,
                    // ck_tile::sequence<8, 4, 8, 2>,
                ck_tile::tuple<ck_tile::sequence<1, 1, 4, 4>, ck_tile::sequence<1, 4, 16, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVTTileDistribution()
    {
        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<
                    // ck_tile::sequence<8, 4, 8, 2>,
                    // ck_tile::sequence<1, 1, 8, 2>>,
                    ck_tile::sequence<1, 4, 16, 2>,
                    ck_tile::sequence<1, 1, 4, 4>>,
                ck_tile::tuple<ck_tile::sequence<2, 1>, ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                // ck_tile::sequence<2, 2, 1, 1>,
                // ck_tile::sequence<0, 3, 0, 3>>{});
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        // Assuming layout of V is always row-major
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN1;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;

        constexpr int32_t N1 = GetAlignmentV();
        constexpr int32_t N0 = kNPerBlock / N1; // P

        constexpr int32_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(ElemPerThread % N1 == 0);
        constexpr int32_t K3     = ElemPerThread / N1;
        constexpr int32_t kKPack = 16 / sizeof(InOutType);
        static_assert(kKPack % K3 == 0);
        constexpr int32_t K2 = kKPack / K3;

        if constexpr(ck_tile::get_warp_size() % (K2 * N0) == 0)
        {
            constexpr int32_t K1 = ck_tile::get_warp_size() / (K2 * N0);
            constexpr int32_t K0 = kBlockSize / ck_tile::get_warp_size();
            static_assert(kKPerBlock == K0 * K1 * K2 * K3);
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                                    ck_tile::tuple<ck_tile::sequence<N0, N1>, ck_tile::sequence<K0, K1, K2, K3>>,
                                                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1, 2>>,
                                                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0, 2>>,
                                                    ck_tile::sequence<2, 1>,
                                                    ck_tile::sequence<3, 1>>{});
        }
        else
        {
            constexpr int32_t K1   = (K2 * N0) / ck_tile::get_warp_size();
            constexpr int32_t K2_m = K2 / K1;
            constexpr int32_t K0   = kBlockSize / ck_tile::get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                                    ck_tile::tuple<ck_tile::sequence<N0, N1>, ck_tile::sequence<K0, K1, K2_m, K3>>,
                                                    ck_tile::tuple<ck_tile::sequence<2, 2>, ck_tile::sequence<1, 2>>,
                                                    ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<0, 2>>,
                                                    ck_tile::sequence<2, 1>,
                                                    ck_tile::sequence<3, 1>>{});
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetNumRepeatOfVDramTileDistribution()
    {
        using VDstrEncode = typename decltype(MakeVDramTileDistribution())::DstrEncode;
        return VDstrEncode::hs_lengthss_[ck_tile::number<1>{}][ck_tile::number<3>{}];
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        // Only called when V is row-major
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN1;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;

        constexpr int32_t N1 = GetAlignmentV();
        constexpr int32_t N0 = kNPerBlock / N1;

        constexpr int32_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(ElemPerThread % N1 == 0);
        constexpr int32_t K3     = ElemPerThread / N1;
        constexpr int32_t kKPack = 16 / sizeof(InOutType);
        static_assert(kKPack % K3 == 0);
        constexpr int32_t K2 = kKPack / K3;

        if constexpr(ck_tile::get_warp_size() % (K2 * N0) == 0)
        {
            constexpr int32_t K1 = ck_tile::get_warp_size() / (K2 * N0);
            constexpr int32_t K0 = kBlockSize / ck_tile::get_warp_size();
            static_assert(kKPerBlock == K0 * K1 * K2 * K3);
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                                    ck_tile::tuple<ck_tile::sequence<N0, N1>, ck_tile::sequence<K0, K1, K2, K3>>,
                                                    ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2, 1, 2>>,
                                                    ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1, 0, 2>>,
                                                    ck_tile::sequence<1, 2>,
                                                    ck_tile::sequence<1, 3>>{});
        }
        else
        {
            constexpr int32_t K1   = (K2 * N0) / ck_tile::get_warp_size();
            constexpr int32_t K2_m = K2 / K1;
            constexpr int32_t K0   = kBlockSize / ck_tile::get_warp_size() / K1;
            static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                                    ck_tile::tuple<ck_tile::sequence<N0, N1>, ck_tile::sequence<K0, K1, K2_m, K3>>,
                                                    ck_tile::tuple<ck_tile::sequence<2, 2>, ck_tile::sequence<1, 2>>,
                                                    ck_tile::tuple<ck_tile::sequence<0, 1>, ck_tile::sequence<0, 2>>,
                                                    ck_tile::sequence<1, 2>,
                                                    ck_tile::sequence<1, 3>>{});
        }
    }
};

template <typename Traits, typename scalar_t, typename acc_t>
struct FlashMlaCombineKernelPolicy
{
private:
    template <typename DataType>
    CK_TILE_DEVICE static constexpr auto MakeOutputTileDistribution()
    {
        constexpr int32_t kVectorN     = GetVectorSizeForTile<Traits::kNumWarpsCombine, 1, Traits::kSizeDV, DataType>();
        constexpr int32_t kThrPerWarpN = ck_tile::get_warp_size();
        constexpr int32_t kNumWarpN    = Traits::kNumWarpsCombine;

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,    // no replicate
                ck_tile::tuple<ck_tile::sequence<1>,
                               ck_tile::sequence<kNumWarpN, kThrPerWarpN, kVectorN>>,
                ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<2>>,
                ck_tile::tuple<ck_tile::sequence<0>, ck_tile::sequence<1>>,
                ck_tile::sequence<1, 2>,
                ck_tile::sequence<0, 2>>{});
    }

public:
    CK_TILE_DEVICE static auto MakeOaccuTileWindow(
        void* p_output_accum,
        const int32_t hsidx,
        const int32_t size_hs,
        const int32_t split_offset,
        const int32_t num_splits)
    {
        const int32_t offset_oaccum = split_offset * size_hs * Traits::kSizeDV;

        // Shape of tensor for a block: [num_splits, Traits::kSizeDV]
        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                reinterpret_cast<acc_t*>(p_output_accum) + offset_oaccum,
                ck_tile::make_tuple(num_splits * size_hs, Traits::kSizeDV), // lengths
                ck_tile::make_tuple(Traits::kSizeDV, 1),                    // strides
                ck_tile::number<Traits::kSizeDV>{},                         // last dim alignment
                ck_tile::number<1>{});                                      // last dim stride

        // Each thread group handles tile whose shape is [1, Traits::kSizeDV]
        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {hsidx, 0});                          // origin

        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<acc_t>());
    }

    CK_TILE_DEVICE static auto MakeOutputTileWindow(
        scalar_t* p_output)
    {
        const auto naive_view =
            ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_output,
                ck_tile::make_tuple(1, Traits::kSizeDV),    // lengths
                ck_tile::make_tuple(Traits::kSizeDV, 1),    // strides
                ck_tile::number<Traits::kSizeDV>{},         // last dim alignment
                ck_tile::number<1>{});                      // last dim stride

        const auto tile_window = ck_tile::make_tile_window(
            naive_view,
            ck_tile::make_tuple(ck_tile::number<1>{},               // window size
                                ck_tile::number<Traits::kSizeDV>{}),
            {0, 0});                                                // origin

        return ck_tile::make_tile_window(tile_window, MakeOutputTileDistribution<scalar_t>());
    }
};

union TileSchedulerMetaData
{
    struct Core
    {
        int32_t begin_batch_idx;
        int32_t begin_seqlen_idx;
        int32_t end_batch_idx;
        int32_t end_seqlen_idx;
        int32_t begin_n_split_idx;
    };
    uint32_t data[8];
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);

struct FlashMlaPrefillFwdParams
{
    int32_t* __restrict__ p_seqlens_k;      // [b]
    int32_t* __restrict__ p_block_table;    // [b, max_seqlen_pad // block_size]
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;      // batch count
    int32_t size_s_pk;   // seqlen of q after XQA pack
    int32_t size_s_ori;  // seqlen of q original
    int32_t size_s_tr;   // seqlen of q for tensor
    int32_t size_h_pk;   // head count of q after XQA pack
    int32_t size_h_ori;  // head count of q original
    int32_t size_h_tr;   // head count of q for tensor
    int32_t hq_hk_ratio; // head count of q / head count of kv
    int32_t num_splits;
    int64_t block_table_batch_stride;
    int32_t num_page_blocks;
    int32_t page_block_size;
    float scale_softmax;
    ck_tile::mdiv mask_y_ratio_mdiv;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;         // stride in batch of query
    index_t stride_s_q;         //    ... in sequence ...
    index_t stride_h_q;         //    ... in head ...
    index_t stride_b_k;         // stride in batch of key
    index_t stride_s_k;         //    ... in sequence ...
    index_t stride_h_k;         //    ... in head ...
    index_t stride_b_v;         // stride in batch of value
    index_t stride_s_v;         //    ... in sequence ...
    index_t stride_h_v;         //    ... in head ...
    index_t stride_b_o;         // stride in batch of output
    index_t stride_s_o;         //    ... in sequence ...
    index_t stride_h_o;         //    ... in head ...
    index_t stride_h_lse;
    index_t stride_b_lseacc;
    index_t stride_h_lseacc;
    index_t stride_sp_lseacc;   //    ... in split ...
    index_t stride_b_oacc;
    index_t stride_h_oacc;
    index_t stride_sp_oacc;     //    ... in split ...
    index_t stride_s_oacc;
};

// =====================================================================================================================
// Kernel Functions
//

template <typename Traits>
CK_TILE_DEVICE static auto GetTileIndex(const int32_t num_splits)
{
    const auto f = [](int32_t dividend, int32_t divisor) {
        int32_t quotient = dividend / divisor;
        int32_t modulus  = dividend - quotient * divisor;
        return ck_tile::make_tuple(quotient, modulus);
    };

    const auto [mid, split_id] = f(blockIdx.x, num_splits);
    const int32_t hid          = blockIdx.y;
    const int32_t bid          = blockIdx.z;

    return ck_tile::make_tuple(mid, split_id, hid, bid);
}

// This function get the range of seqlen for the specified `split_idx`. `granularity` is the granularity of group of
// workload which cannot be further subdivded.
// The workload is divided as evenly as possible. When the workload cannot be evenly divided by num_splits, the
// high-ranking splits will get 1 additional `granularity` of tasks.
// E.g. when `num_seqlen` is `28`, `granularity` is `2` and `num_splits` is `3`, the 3 splits will be assigned the
// following tasks:
// split.0: [0, 10)  // 10 workloads
// split.1: [10, 20) // 10 workloads
// split.2: [20, 28) //  8 workloads
// split.3: [28, 36) // Note that this may not be what you're expecting. upper_bound may be helpful in this case.
CK_TILE_DEVICE static auto GetSeqlenRange(
    const int32_t num_seqlen,
    const int32_t granularity,
    const int32_t num_splits,
    const int32_t split_idx,
    const int32_t lower_bound,
    const int32_t upper_bound)
{
    const int32_t num_workload = ck_tile::integer_divide_ceil(num_seqlen, granularity);
    const int32_t base_workload = ck_tile::integer_divide_floor(num_workload, num_splits);
    const int32_t addition_threshold = num_workload % num_splits;
    const int32_t start = base_workload * split_idx + ck_tile::min(addition_threshold, split_idx);
    const int32_t count = base_workload + ((split_idx < addition_threshold) ? 1 : 0);
    const int32_t end = start + count;

    return ck_tile::make_tuple(ck_tile::max(lower_bound, start * granularity),
                               ck_tile::min(upper_bound, end * granularity));
}

template <typename Policy, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeQDram(const scalar_t* p_data,
                                     const int32_t size_s_ori,
                                     const int32_t stride_s,
                                     const int32_t hq_hk_ratio,
                                     const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto q_dram_naive = [&] {
        if constexpr(Traits::kXqaStrategy == XqaStrategy::Internal)
        {
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, hq_hk_ratio, Traits::kSizeD),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentQ()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s_ori, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(Traits::kSizeD)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, Traits::kSizeD),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentQ()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        q_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockK0>{}),
        ck_tile::sequence<false, Traits::kPadHeadDimQ>{});
}

template <typename Policy, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeKDram(
    const scalar_t* p_data,
    const int32_t   height,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    const auto k_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data, // will update this pointer if using paged-kvcache
        ck_tile::make_tuple(height, Traits::kSizeD),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentK()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        k_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{}),
        ck_tile::sequence<true, Traits::kPadHeadDimQ>{});
}

template <typename Policy, typename scalar_t = typename Policy::InOutType>
CK_TILE_DEVICE static auto MakeVDram(
    const scalar_t* p_data,
    const int32_t   length,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    // Assuming Value is row-major just like Key.
    const auto v_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data, // will update this pointer if using paged-kvcache
        ck_tile::make_tuple(length, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentV()>{},
        ck_tile::number<1>{});

    const auto v_dram_transposed = ck_tile::transform_tensor_view(
        v_dram_naive,
        ck_tile::make_tuple(ck_tile::make_pass_through_transform(Traits::kSizeDV),
                            ck_tile::make_pass_through_transform(length)),
        ck_tile::make_tuple(ck_tile::sequence<1>{}, ck_tile::sequence<0>{}),
        ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

    return ck_tile::pad_tensor_view(
        v_dram_transposed,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{},
                            ck_tile::number<Traits::kBlockK1>{}),
        ck_tile::sequence<Traits::kPadHeadDimV, Traits::kPadSeqLenK>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseAccDram(scalar_t* p_data,
                                          const Lengths& window_lengths,
                                          const int32_t size_s_ori,
                                          const int32_t hq_hk_ratio,
                                          const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto lse_acc_dram_naive = [&] {
        if constexpr(Traits::kXqaStrategy == XqaStrategy::Internal)
        {
            // transpose + merge: (hq_hk_ratio, seqlen_q) -> (seqlenq * hq_hk_ratio)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(hq_hk_ratio, size_s_ori),
                ck_tile::make_tuple(stride_h, 1),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s_ori, hq_hk_ratio))),
                ck_tile::make_tuple(ck_tile::sequence<1, 0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori),
                ck_tile::make_tuple(1),
                ck_tile::number<1>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        lse_acc_dram_naive,
        window_lengths,
        ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutAccDram(scalar_t* p_data,
                                          const int32_t size_s_ori,
                                          const int32_t stride_s,
                                          const int32_t hq_hk_ratio,
                                          const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto o_acc_dram_naive = [&] {
        if constexpr(Traits::kXqaStrategy == XqaStrategy::Internal)
        {
            // merge: (seqlen_q, hq_hk_ratio, headdim) -> (seqlen_q*hq_hk_ratio, headdim)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, hq_hk_ratio, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentOacc()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s_ori, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(Traits::kSizeDV)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentOacc()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        o_acc_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseDram(scalar_t* p_data,
                                       const Lengths& window_lenghts,
                                       const int32_t size_s_ori,
                                       const int32_t hq_hk_ratio,
                                       const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto lse_dram_naive = [&] {
        if constexpr(Traits::kXqaStrategy == XqaStrategy::Internal)
        {
            // transpose + merge: (hq_hk_ratio, seqlen_q) -> (seqlenq * hq_hk_ratio)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(hq_hk_ratio, size_s_ori),
                ck_tile::make_tuple(stride_h, 1),
                ck_tile::number<Policy::GetAlignmentLse()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s_ori, hq_hk_ratio))),
                ck_tile::make_tuple(ck_tile::sequence<1, 0>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}));
        }
        else
        {

            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori),
                ck_tile::make_tuple(1),
                ck_tile::number<Policy::GetAlignmentLse()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        lse_dram_naive, window_lenghts, ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutDram(scalar_t* p_data,
                                       const int32_t size_s_ori,
                                       const int32_t stride_s,
                                       const int32_t hq_hk_ratio,
                                       const int32_t stride_h)
{
    using Traits = typename Policy::Traits;

    const auto o_dram_naive = [&] {
        if constexpr(Traits::kXqaStrategy == XqaStrategy::Internal)
        {
            // merge: (seqlen_q, hq_hk_ratio, headdim) -> (seqlen_q * hq_hk_ratio, headdim)
            const auto view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, hq_hk_ratio, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, stride_h, 1),
                ck_tile::number<Policy::GetAlignmentO()>{},
                ck_tile::number<1>{});
            return ck_tile::transform_tensor_view(
                view,
                ck_tile::make_tuple(
                    ck_tile::make_merge_transform(ck_tile::make_tuple(size_s_ori, hq_hk_ratio)),
                    ck_tile::make_pass_through_transform(Traits::kSizeDV)),
                ck_tile::make_tuple(ck_tile::sequence<0, 1>{}, ck_tile::sequence<2>{}),
                ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));
        }
        else
        {
            return ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
                p_data,
                ck_tile::make_tuple(size_s_ori, Traits::kSizeDV),
                ck_tile::make_tuple(stride_s, 1),
                ck_tile::number<Policy::GetAlignmentO()>{},
                ck_tile::number<1>{});
        }
    }();

    return ck_tile::pad_tensor_view(
        o_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <bool IsMasking, typename acc_t>
CK_TILE_DEVICE static auto GetValidatedMax(acc_t raw_m)
{
    if constexpr (IsMasking)
    {
        return (raw_m == -ck_tile::numeric<acc_t>::infinity()) ? ck_tile::type_convert<acc_t>(0.f) : raw_m;
    }
    else
    {
        return raw_m;
    }
}

template <typename Policy, typename SaccType>
CK_TILE_DEVICE static auto ShuffleSacc(
    const SaccType& s_acc,
    uint8_t*        p_smem)
{
    using Traits = typename Policy::Traits;

    constexpr int32_t kWarpGemmM = Traits::QKWarpTile::at(ck_tile::number<0>{});
    constexpr int32_t ColWarps = Traits::kBlockM / kWarpGemmM;
    constexpr int32_t RowWarps = Traits::kNumWarps / ColWarps;

    if constexpr (RowWarps > 1)
    {
        // P shuffle LDS windows
        auto p_shuffle_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            reinterpret_cast<SaccType::DataType*>(p_smem), Policy::MakePShuffleLdsBlockDescriptor());
        auto p_st_lds_window =
            ck_tile::make_tile_window(
                p_shuffle_lds,
                ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN0>{}),
                {0, 0});
        auto p_ld_lds_window =
            ck_tile::make_tile_window(
                p_shuffle_lds,
                ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN0>{}),
                {0, 0},
                Policy::GetKVBlockGemm().MakeABlockTileDistribution());

        if (Policy::HandleGemm0())
        {
            ck_tile::store_tile(p_st_lds_window, s_acc);
        }
        ck_tile::block_sync_lds();
        return ck_tile::load_tile(p_ld_lds_window);
    }
    else
    {
        return s_acc;
    }
}

template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename OaccBlockTileType,
         typename MLBlockTileType,
         typename Mask>
CK_TILE_DEVICE static void kn_fmla_fwd_splitkv_prefill_tile_epilogue(
    LseDramBlockWindow& lse_dram_window_,
    OutDramBlockWindow& out_dram_window_,
    OaccBlockTileType*  o_acc,
    MLBlockTileType&    m,
    MLBlockTileType&    l,
    Mask&               mask)
{
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;

    // 7. Store LSE
    //
    auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
    constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
    ck_tile::sweep_tile_span(lse_acc_spans[ck_tile::number<0>{}], [&, m_ = m, l_ = l](auto id0) {
        constexpr auto i = make_tuple(id0);
#if FMLA_FWD_FAST_EXP2
#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif
        lse_acc(i) = m_[i] / C_LOG2E  + log(l_[i]);
#else
        lse_acc(i) = m_[i] + log(l_[i]);
#endif
    });
    ck_tile::store_tile(lse_dram_window_, lse_acc);

    // 8. Adjust and output
    //
    constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();;
    ck_tile::sweep_tile_span(o_spans[ck_tile::number<0>{}], [&](auto id0) {
        constexpr auto i = ck_tile::make_tuple(id0);
        const auto tmp   = [&]() {
            if constexpr (Mask::IsMasking)
            {
                return l[i] == 0.f ? 0.f : 1 / l[i];
            }
            else
            {
                return 1 / l[i];
            }
        }();
        ck_tile::sweep_tile_span(o_spans[ck_tile::number<1>{}], [&](auto id1) {
            constexpr auto ij = ck_tile::make_tuple(id0, id1);
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
            {
#if 0
                acc_t o_acc_v = o_acc[n1_id](ij);
                asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                             : [v_o_acc] "+v"(o_acc_v)
                             : [v_tmp] "v"(tmp));
                o_acc[n1_id](ij) = o_acc_v;
#else
                o_acc[n1_id](ij) *= tmp;
#endif
            });
        });
    });

    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
    {
        ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
        if constexpr (n1_id < (n1_loops - 1))
        {
            ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
        }
    }
    );
}

template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename QDramBlockWindow,
         typename KDramBlockWindow,
         typename VDramBlockWindow,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename Mask>
CK_TILE_DEVICE static void kn_fmla_fwd_splitkv_prefill_tile(
    const QDramBlockWindow& q_dram_window_,
    const KDramBlockWindow& k_dram_window_raw,
    const VDramBlockWindow& v_dram_window_raw,
    LseDramBlockWindow&     lse_dram_window_,
    OutDramBlockWindow&     out_dram_window_,
    const int32_t*          p_block_table,
    const int32_t           page_block_size,
    const int32_t           stride_s_k,
    const int32_t           stride_s_v,
    int32_t                 seqlen_k,
    int32_t                 num_splits,
    int32_t                 split_id,
    Mask                    mask,
    float                   scale_s,
    uint8_t*                p_smem)
{
    using Policy = FlashMlaPrefillPolicy<Traits, scalar_t, acc_t>;


    // 1. Allocate LDS
    //
    auto k_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        reinterpret_cast<scalar_t*>(p_smem), Policy::MakeKLdsBlockDescriptor());
    auto v_lds = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        reinterpret_cast<scalar_t*>(p_smem), Policy::MakeVLdsBlockDescriptor());

    auto k_lds_window = ck_tile::make_tile_window(
        k_lds, ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{}), {0, 0});
    auto v_lds_window = ck_tile::make_tile_window(
        v_lds, Policy::MakeVLdsBlockDescriptor().get_lengths(), {0, 0});


    // 2. Misc. preparation
    //

    // Loop counts
    constexpr int32_t k0_loops = Traits::kSizeD / Traits::kBlockK0;      // #loop for Q in reg
    constexpr int32_t k1_loops = Traits::kBlockN0 / Traits::kBlockK1;
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;
    static_assert(k0_loops >= 2);
    static_assert(k1_loops >= 1);
    static_assert(n1_loops >= 1);
    static_assert((Traits::kSizeD   % Traits::kBlockK0) == 0);
    static_assert((Traits::kBlockN0 % Traits::kBlockK1) == 0);
    static_assert((Traits::kSizeDV  % Traits::kBlockN1) == 0);
    // TODO: make it tunable once slice tile issue is fixed.
    static_assert(!((k1_loops > 1) && (Traits::kNumWarps == 8)),
                  "k1_loops > 1 is not support by wave 8 for now due to the issue in tile_slice.");

    // Block GEMMs
    constexpr auto gemm_0 = Policy::GetQKBlockGemm();
    constexpr auto gemm_1 = Policy::GetKVBlockGemm();

    // Reduction funtions for softmax
    using BlockShape           = ck_tile::remove_cvref_t<decltype(gemm_0.GetCBlockShape())>;
    using BlockReduce2dProblem = ck_tile::BlockReduce2dProblem<acc_t, acc_t, BlockShape>;
    auto block_reduce_2d       = ck_tile::BlockReduce2d<BlockReduce2dProblem>();        // In-thread
    auto block_reduce_2d_sync  = ck_tile::BlockReduce2dSync<BlockReduce2dProblem>();    // In-warp
    auto reduce_sum_func = ck_tile::ReduceOp::Add{};
    auto reduce_max_func = ck_tile::ReduceOp::Max{};

    // sacc, S, P, M, L, Oacc
    using SaccBlockTileType    = decltype(gemm_0.MakeCBlockTile());
    auto s_acc                 = SaccBlockTileType{};
    using SaccShuffledTileType = decltype(ShuffleSacc<Policy>(s_acc, nullptr));
    auto s_acc_shuffled        = SaccShuffledTileType{};
    using MLBlockTileType      = decltype(block_reduce_2d(s_acc_shuffled,
                                                          ck_tile::numeric<acc_t>::min(), 
                                                          reduce_max_func));
    using OaccBlockTileType    = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc[n1_loops];
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id){ck_tile::clear_tile(o_acc[n1_id]);});
    ck_tile::set_tile(m, -ck_tile::numeric<acc_t>::infinity());
    ck_tile::clear_tile(l);

    const auto q_origin = q_dram_window_.get_window_origin();
    auto [origin_start, origin_end] =
        mask.GetTileRangeAlongX(q_origin.at(ck_tile::number<0>{}),
                                ck_tile::number<Traits::kBlockM>{},
                                ck_tile::number<Traits::kBlockN0>{});
    auto [seqlen_k_start, seqlen_k_end] =
        GetSeqlenRange(
            seqlen_k,
            Traits::kBlockN0,
            num_splits,
            split_id,
            origin_start,
            ck_tile::min(origin_end, seqlen_k));


    // 3. Quick exit if no work to do
    //
    const int32_t num_total_loop =
        ck_tile::integer_divide_ceil(seqlen_k_end - seqlen_k_start, Traits::kBlockN0);

    if (num_total_loop <= 0)
    {
        auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
        ck_tile::set_tile(lse_acc, -ck_tile::numeric<acc_t>::infinity());
        ck_tile::store_tile(lse_dram_window_, lse_acc);
        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
        {
            ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
            if constexpr (n1_id < (n1_loops - 1))
            {
                ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
            }
        });
        return;
    }


    // 4. Load Q to lds and reg
    //
    auto q_dram_window =
        ck_tile::make_tile_window(q_dram_window_.get_bottom_tensor_view(),
                                  q_dram_window_.get_window_lengths(),
                                  q_dram_window_.get_window_origin(),
                                  Policy::MakeQRegTileDistribution());
    using QTile = decltype(ck_tile::load_tile(q_dram_window));
    QTile q_regs[2];


    // 5. Prepare KV
    //
    auto k_dram_window_origin = ck_tile::make_tile_window(
        k_dram_window_raw.get_bottom_tensor_view(),
        k_dram_window_raw.get_window_lengths(),
        {seqlen_k_start, 0});
    auto k_dist = Policy::MakeKDramTileDistribution();
    auto k_coord = k_dist.calculate_index();
    constexpr auto kKNumRepeat = Policy::GetNumRepeatOfKDramTileDistribution();
    constexpr auto kKPageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_base_idx = k_coord[kKPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKNumRepeat> k_offsets;
    ck_tile::statically_indexed_array<bool, kKNumRepeat> k_valids;
    ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_base_idx + Traits::kBlockN0 / kKNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
        k_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_window_origin.get_bottom_tensor_view(),
        k_dram_window_origin.get_window_lengths(),
        k_dram_window_origin.get_window_origin(),
        k_dist,
        k_offsets,
        k_valids,
        kKPageIdxDim);
    k_dram_window.init_raw();

    auto v_dram_window_origin = ck_tile::make_tile_window(
        v_dram_window_raw.get_bottom_tensor_view(),
        v_dram_window_raw.get_window_lengths(),
        {0, seqlen_k_start});
    auto v_dist = Policy::MakeVDramTileDistribution();
    auto v_coord = v_dist.calculate_index();
    constexpr auto kVNumRepeat = Policy::GetNumRepeatOfVDramTileDistribution();
    constexpr auto kVPageIdxDim = ck_tile::number<1>{};
    const int32_t seqlen_v_base_idx = v_coord[kVPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kVNumRepeat> v_offsets;
    ck_tile::statically_indexed_array<bool, kVNumRepeat> v_valids;
    ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_v_base_idx + rid.value;
        const int32_t page_idx   = seqlen_idx / page_block_size;
        const int32_t inside_idx = seqlen_idx % page_block_size;
        v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
        v_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    using VDramWindow = decltype(ck_tile::make_tile_scatter_gather(
        v_dram_window_origin.get_bottom_tensor_view(),
        v_dram_window_origin.get_window_lengths(),
        v_dram_window_origin.get_window_origin(),
        v_dist,
        v_offsets,
        v_valids,
        kVPageIdxDim));
    VDramWindow v_dram_windows[n1_loops];
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
    {
        constexpr ck_tile::array<int32_t, 2> origin_col = {n1_id * Traits::kBlockN1, 0};
        v_dram_windows[n1_id] = ck_tile::make_tile_scatter_gather(
            v_dram_window_origin.get_bottom_tensor_view(),
            v_dram_window_origin.get_window_lengths(),
            v_dram_window_origin.get_window_origin() + origin_col,
            v_dist,
            v_offsets,
            v_valids,
            kVPageIdxDim);
        v_dram_windows[n1_id].init_raw();
    });


    // 6. Main loop
    //
    for(int32_t loop_idx = 0; loop_idx < num_total_loop; loop_idx += 1)
    {
        const bool is_even_loop = (loop_idx % 2 == 0);

        if (Policy::HandleGemm0())
        {
            ck_tile::clear_tile(s_acc);

            // I. QK GEMM
            //
            ck_tile::array<int32_t, 2> qk_origin =
                {0, is_even_loop ? 0 : Traits::kBlockK0 * (k0_loops - 1)};
            ck_tile::array<int32_t, 2> qk_direction =
                {0, is_even_loop ? Traits::kBlockK0 : -Traits::kBlockK0};
            auto q_dram_window = ck_tile::make_tile_window(
                q_dram_window_.get_bottom_tensor_view(),
                q_dram_window_.get_window_lengths(),
                q_dram_window_.get_window_origin() + qk_origin,
                Policy::MakeQRegTileDistribution());
            q_regs[0] = ck_tile::load_tile(q_dram_window);
            ck_tile::move_tile_window(q_dram_window, qk_direction);
            q_regs[1] = ck_tile::load_tile(q_dram_window);

            // Load 1st K tile from DRAM to SMEM and start loading the 2nd
            // k_dram_window moves along K0 and only moves within page block.
            auto k_block_tile = ck_tile::load_tile(k_dram_window);

            ck_tile::move_tile_window(k_dram_window, qk_direction);
            ck_tile::store_tile(k_lds_window, k_block_tile);
            k_block_tile = ck_tile::load_tile(k_dram_window);

            // Main part of QK GEMM_0: conduct GEMM and load K tiles 
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    ck_tile::block_sync_lds();
                    gemm_0(s_acc, q_regs[k0_id % 2], k_lds_window);
                    ck_tile::block_sync_lds();
                    // pre-load q
                    ck_tile::move_tile_window(q_dram_window, qk_direction);
                    q_regs[k0_id % 2] = ck_tile::load_tile(q_dram_window);
                    // pre-load k
                    ck_tile::move_tile_window(k_dram_window, qk_direction);
                    ck_tile::store_tile(k_lds_window, k_block_tile);
                    k_block_tile = ck_tile::load_tile(k_dram_window);
                });
            }

            // Tailing 2 tiles of QK GEMM_0
            ck_tile::block_sync_lds();
            gemm_0(s_acc, q_regs[(k0_loops - 2) % 2], k_lds_window);

            ck_tile::block_sync_lds();
            ck_tile::store_tile(k_lds_window, k_block_tile);

            ck_tile::block_sync_lds();
            gemm_0(s_acc, q_regs[(k0_loops - 1) % 2], k_lds_window);

            ck_tile::tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
        }
        else
        {
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_s_barrier();
                });
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
        }

        // prefetch load V tile
        auto v_prefetch = ck_tile::load_tile(v_dram_windows[0]);
        
        if (Policy::HandleGemm0())
        {
            // II. scale_s, mask, softmax
            //

            // Masking
            // Note that masking is also required when k is padded
            const auto k_origin = k_dram_window_origin.get_window_origin();
            const bool need_perpixel_check = mask.IsEdgeTile(
                __builtin_amdgcn_readfirstlane(q_origin.at(ck_tile::number<0>{})),
                __builtin_amdgcn_readfirstlane(k_origin.at(ck_tile::number<0>{})),
                ck_tile::number<Traits::kBlockM>{},
                ck_tile::number<Traits::kBlockN0>{});

            if (need_perpixel_check)
            {
                ck_tile::set_tile_if(
                    s_acc, -ck_tile::numeric<acc_t>::infinity(),
                    [&](auto ids)
                    {
                        const auto row = q_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<0>{});
                        const auto col = k_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<1>{});
                        return mask.IsOutOfBound(row, col);
                    });
            }
        }

        s_acc_shuffled = ShuffleSacc<Policy>(
            s_acc, p_smem + Policy::GetPSmemStart());

        // Get max of row
        auto m_local = block_reduce_2d(s_acc_shuffled, reduce_max_func.GetIdentityValue<acc_t>(), reduce_max_func);
        block_reduce_2d_sync(m_local, reduce_max_func);

        const auto m_old = m;
        ck_tile::tile_elementwise_inout(
            [](auto& e0, auto e1, auto e2) { e0 = ck_tile::max(e1, e2); }, m, m_old, m_local);

        // Compute exp(x_i - m)
        auto p_intermedia = ck_tile::make_static_distributed_tensor<acc_t>(s_acc_shuffled.get_tile_distribution());
        const auto p_spans = decltype(p_intermedia)::get_distributed_spans();
        ck_tile::sweep_tile_span(
            p_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                auto row_max  = GetValidatedMax<Mask::IsMasking>(m[i]);
                ck_tile::sweep_tile_span(
                    p_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
#if FMLA_FWD_FAST_EXP2
                        p_intermedia(ij) = ck_tile::exp2(s_acc_shuffled[ij] - row_max);
#else
                        p_intermedia(ij) = ck_tile::exp(s_acc_shuffled[ij] - row_max);
#endif
                    });
            });

        // Compute row sum of exp(x_i - m)
        auto rowsum_p = block_reduce_2d(p_intermedia, reduce_sum_func.GetIdentityValue<acc_t>(), reduce_sum_func);
        block_reduce_2d_sync(rowsum_p, reduce_sum_func);

        // Calculate new l and adjust old output acc
        constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();
        ck_tile::sweep_tile_span(
            o_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                const auto row_max = GetValidatedMax<Mask::IsMasking>(m[i]);
#if FMLA_FWD_FAST_EXP2
                const auto temp_i  = ck_tile::exp2(m_old[i] - row_max);
#else
                const auto temp_i  = ck_tile::exp(m_old[i] - row_max);
#endif
                l(i) = temp_i * l[i] + rowsum_p[i];
                ck_tile::sweep_tile_span(
                    o_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
                        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
                        {
#if 0
                            acc_t o_acc_v = o_acc[n1_id](ij);
                            asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                                        : [v_o_acc] "+v"(o_acc_v)
                                        : [v_tmp] "v"(temp_i));
                            o_acc[n1_id](ij) = o_acc_v;
#else
                            o_acc[n1_id](ij) *= temp_i;
#endif
                        });
                    });
            });


        // III. GEMM for PV
        //

        // Store V tile to LDS. V is expected as row-major so it needs to be shuffled before store.
        ck_tile::block_sync_lds();
        const auto p = ck_tile::cast_tile<scalar_t>(p_intermedia);
        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
        {
            auto v_shuffled = ck_tile::make_static_distributed_tensor<scalar_t>(Policy::MakeShuffledVRegBlockDescriptor());
            ck_tile::shuffle_tile(v_shuffled, v_prefetch);
            ck_tile::store_tile(v_lds_window, v_shuffled);
            __builtin_amdgcn_sched_barrier(0); // This barrier decrease the useage of vgprs, but
                                               // negatively affects performance for small seqlen.
            if constexpr(k1_loops > 1)
            {
                ck_tile::static_for<0, k1_loops - 1, 1>{}([&](auto k1_id)
                {
                    // ck_tile::move_tile_window(v_dram_windows[n1_id], {0, Traits::kBlockK1});
                    ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
                    {
                        const int32_t seqlen_idx = seqlen_v_base_idx + rid.value +
                            loop_idx * Traits::kBlockN0 + (k1_id + 1) * Traits::kBlockK1;
                        const int32_t page_idx   = seqlen_idx / page_block_size;
                        const int32_t inside_idx = seqlen_idx % page_block_size;
                        v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
                        v_valids[rid] = (seqlen_idx < seqlen_k_end);
                    });
                    v_dram_windows[n1_id].update_page_idx_and_valids(v_offsets, v_valids);

                    auto v = ck_tile::load_tile(v_dram_windows[n1_id]); // load next v
                    ck_tile::block_sync_lds();

                    gemm_1(o_acc[n1_id],
                           ck_tile::get_slice_tile(
                               p,
                               ck_tile::sequence<0, k1_id * Traits::kBlockK1>{},
                               ck_tile::sequence<Traits::kBlockM, (k1_id + 1) * Traits::kBlockK1>{}),
                           v_lds_window);
                    ck_tile::block_sync_lds();

                    auto v_shuffled = ck_tile::make_static_distributed_tensor<scalar_t>(
                        Policy::MakeShuffledVRegBlockDescriptor());
                        ck_tile::shuffle_tile(v_shuffled, v);
                        ck_tile::store_tile(v_lds_window, v_shuffled); // store the prefetch
                });
            }
            // Output tail
            ck_tile::block_sync_lds();

            if constexpr (n1_id < (n1_loops-1))
            {
                v_prefetch = ck_tile::load_tile(v_dram_windows[n1_id + 1]);
                // TODO: The following code is not necessary but it positively affects performance
                // for unknown reason. Remove the following code once it no longer affects.
                ck_tile::set_tile_if(
                    v_prefetch, ck_tile::numeric<scalar_t>::zero(),
                    [&](auto ids)
                    {
                        const auto col = ids.at(kVPageIdxDim);
                        return (loop_idx * Traits::kBlockN0 + col) >= seqlen_k;
                    });
            }
            if constexpr (k1_loops > 1)
            {
                gemm_1(o_acc[n1_id],
                   ck_tile::get_slice_tile(
                        p,
                        ck_tile::sequence<0, (k1_loops - 1) * Traits::kBlockK1>{},
                        ck_tile::sequence<Traits::kBlockM, Traits::kBlockN0>{}),
                   v_lds_window);
            }
            else
            {
                gemm_1(o_acc[n1_id], p, v_lds_window);
            }
            ck_tile::block_sync_lds();
        });

        if ((loop_idx + 1) < num_total_loop)
        {
            // Move K to next block of column
            ck_tile::array<int32_t, 2> next_qk_origin =
                {0, is_even_loop ? Traits::kBlockK0 * (k0_loops - 1) : 0};
            ck_tile::move_tile_window(k_dram_window_origin, {Traits::kBlockN0, 0});
            k_dram_window.set_window_origin(k_dram_window_origin.get_window_origin() + next_qk_origin);
            // Recalculate offsets
            ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / page_block_size;
                const int32_t inside_idx = seqlen_idx % page_block_size;
                k_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_k;
                k_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_dram_window.update_page_idx_and_valids(k_offsets, k_valids);
            // Move to next V block
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
            {
                // ck_tile::move_tile_window(v_dram_windows[n1_id], {0, -Traits::kBlockK1 * (k1_loops - 1)});
                ck_tile::static_for<0, kVNumRepeat, 1>{}([&](auto rid)
                {
                    const int32_t seqlen_idx = seqlen_v_base_idx + rid.value + (loop_idx + 1) * Traits::kBlockN0;
                    const int32_t page_idx   = seqlen_idx / page_block_size;
                    const int32_t inside_idx = seqlen_idx % page_block_size;
                    v_offsets[rid] = (p_block_table[page_idx] * page_block_size + inside_idx) * stride_s_v;
                    v_valids[rid] = (seqlen_idx < seqlen_k_end);
                });
                v_dram_windows[n1_id].update_page_idx_and_valids(v_offsets, v_valids);
            });
        }
    }

    // 7. tile epilogue: store LSE, Adjust and output
    kn_fmla_fwd_splitkv_prefill_tile_epilogue<Traits, scalar_t, acc_t, out_t>(
        lse_dram_window_,
        out_dram_window_,
        o_acc,
        m,
        l,
        mask);
}

// this function work for mla which load key&value tensor once, 
// transpose the k-nope tensor into v tensor, and never load v tensor from dram again.
// TODO: 1. async load from dram to lds. use double lds buffer to load two kv blocks.
//       2. transpose v value while gemm_0.
template<typename Traits,
         typename scalar_t,
         typename acc_t,
         typename out_t,
         typename QNopeDramRegBlockWindow,
         typename QRopeDramRegBlockWindow,
         typename KVDramTensorView,
         typename LseDramBlockWindow,
         typename OutDramBlockWindow,
         typename Mask>
CK_TILE_DEVICE static void kn_fmla_fwd_splitkv_prefill_load_once_tile(
    const QNopeDramRegBlockWindow&  q_nope_dram_window_,
    const QRopeDramRegBlockWindow&  q_rope_dram_window_,
    const KVDramTensorView&         k_dram_tensor_view_,
    LseDramBlockWindow&             lse_dram_window_,
    OutDramBlockWindow&             out_dram_window_,
    const int32_t*                  p_block_table,
    const int32_t                   stride_s_k,
    const int32_t                   stride_s_v,
    int32_t                         seqlen_k,
    int32_t                         num_splits,
    int32_t                         split_id,
    Mask                            mask,
    float                           scale_s,
    uint8_t*                        p_smem
    )
{
    using Policy = FlashMlaPrefillPolicy<Traits, scalar_t, acc_t>;

    // 1. Allocate LDS
    //
    const auto k_rope_smem_offset = (Traits::kNumPrefetchK * Policy::template GetSingleKSpaceSize<false>());

    const auto k_nope_repeat_smem_offset = Policy::GetKNopeSingleRepeatSize();
    constexpr auto k_oob_ck = ck_tile::bool_constant<true>{};
    constexpr auto k_pre_np = ck_tile::bool_constant<false>{};

    auto k_nope_lds_ptr = reinterpret_cast<scalar_t*>(p_smem);
    auto k_nope_st_lds_windows = ck_tile::generate_tuple(
        [&](auto i_buf) {
            return ck_tile::make_tile_window(
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    k_nope_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<false>(i_buf)),
                Policy::template MakeKLdsStoreBlockDescriptor<false>(i_buf).get_lengths(),
                // ck_tile::make_tuple(ck_tile::number<4>{}, ck_tile::number<4>{}, ck_tile::number<128>{}),
                {0, 0, 0});
        },
        ck_tile::number<Traits::kNumPrefetchK>{});
    auto k_nope_ld_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        k_nope_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<false>());
    auto k_nope_ld_lds_window = ck_tile::make_tile_window(
        k_nope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{}),
        {0, 0});

    auto k_rope_lds_ptr = reinterpret_cast<scalar_t*>(p_smem + k_rope_smem_offset);
    auto k_rope_st_lds_windows = ck_tile::generate_tuple(
        [&](auto i_buf) {
            return ck_tile::make_tile_window(
                ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
                    k_rope_lds_ptr, Policy::template MakeKLdsStoreBlockDescriptor<true>(i_buf)),
                Policy::template MakeKLdsStoreBlockDescriptor<true>(i_buf).get_lengths(),
                // ck_tile::make_tuple(ck_tile::number<2>{}, ck_tile::number<4>{}, ck_tile::number<128>{}),
                {0, 0, 0});
        },
        ck_tile::number<Traits::kNumPrefetchK>{});
    auto k_rope_ld_lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
        k_rope_lds_ptr, Policy::template MakeKLdsLoadBlockDescriptor<true>());
    auto k_rope_ld_lds_window = ck_tile::make_tile_window(
        k_rope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeRope>{}),
        {0, 0});

    auto vt_lds_ptr = reinterpret_cast<scalar_t*>(p_smem + Traits::kNumPrefetchK *
        Policy::GetSmemSizeK());
    auto v_ld_lds_window = ck_tile::make_tile_window(
        k_nope_ld_lds_view,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{}),
        {0, 0},
        Policy::MakeVTileDistribution());
    auto vt_st_lds_window = ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            vt_lds_ptr,
            Policy::MakeVLdsStoreBlockDescriptor()),
        Policy::MakeVLdsStoreBlockDescriptor().get_lengths(), {0, 0});
    auto vt_ld_lds_window = ck_tile::make_tile_window(
        ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(
            vt_lds_ptr,
            Policy::MakeVLdsStoreBlockDescriptor()),
        Policy::MakeVLdsStoreBlockDescriptor().get_lengths(), {0, 0});


    // 2. Misc. preparation
    //

    // Loop counts
    constexpr int32_t k0_loops = Traits::kSizeD / Traits::kBlockK0;      // #loop for Q in reg
    constexpr int32_t k1_loops = Traits::kBlockN0 / Traits::kBlockK1;
    constexpr int32_t n1_loops = Traits::kSizeDV / Traits::kBlockN1;
    static_assert(k0_loops >= 1);
    static_assert(k1_loops >= 1);
    static_assert(n1_loops >= 1);
    // static_assert((Traits::kSizeD   % Traits::kBlockK0) == 0);
    static_assert((Traits::kBlockN0 % Traits::kBlockK1) == 0);
    static_assert((Traits::kSizeDV  % Traits::kBlockN1) == 0);

    // Block GEMMs
    constexpr auto gemm_0      = Policy::template GetQKBlockGemm<false>();
    constexpr auto gemm_0_rope = Policy::template GetQKBlockGemm<true>();
    constexpr auto gemm_1      = Policy::GetKVBlockGemm();

    // Reduction funtions for softmax
    const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    // Reduction funtions for softmax
    using BlockShape           = ck_tile::remove_cvref_t<decltype(gemm_0.GetCBlockShape())>;
    using BlockReduce2dProblem = ck_tile::BlockReduce2dProblem<acc_t, acc_t, BlockShape>;
    auto block_reduce_2d       = ck_tile::BlockReduce2d<BlockReduce2dProblem>();        // In-thread
    auto block_reduce_2d_sync  = ck_tile::BlockReduce2dSync<BlockReduce2dProblem>();    // In-warp
    auto reduce_sum_func = ck_tile::ReduceOp::Add{};
    auto reduce_max_func = ck_tile::ReduceOp::Max{};

    // sacc, S, P, M, L, Oacc
    using SaccBlockTileType    = decltype(gemm_0.MakeCBlockTile());
    auto s_acc                 = SaccBlockTileType{};
    using SaccShuffledTileType = decltype(ShuffleSacc<Policy>(s_acc, nullptr));
    auto s_acc_shuffled        = SaccShuffledTileType{};
    using MLBlockTileType      = decltype(block_reduce_2d(s_acc_shuffled,
                                                          ck_tile::numeric<acc_t>::min(), 
                                                          reduce_max_func));
    using OaccBlockTileType    = decltype(gemm_1.MakeCBlockTile());
    OaccBlockTileType o_acc[n1_loops];
    auto m = MLBlockTileType{};
    auto l = MLBlockTileType{};
    ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id){ck_tile::clear_tile(o_acc[n1_id]);});
    ck_tile::set_tile(m, -ck_tile::numeric<acc_t>::infinity());
    ck_tile::clear_tile(l);


    const auto q_origin = q_nope_dram_window_.get_window_origin();
    auto [origin_start, origin_end] =
        mask.GetTileRangeAlongX(q_origin.at(ck_tile::number<0>{}),
                                ck_tile::number<Traits::kBlockM>{},
                                ck_tile::number<Traits::kBlockN0>{});
    auto [seqlen_k_start, seqlen_k_end] =
        GetSeqlenRange(seqlen_k, Traits::kBlockN0, num_splits, split_id, origin_start, origin_end);


    // 3. Quick exit if no work to do
    //
    const int32_t num_total_loop =
        ck_tile::integer_divide_ceil(seqlen_k_end - seqlen_k_start, Traits::kBlockN0);

    if (num_total_loop <= 0)
    {
        auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
        ck_tile::set_tile(lse_acc, -ck_tile::numeric<acc_t>::infinity());
        ck_tile::store_tile(lse_dram_window_, lse_acc);
        ck_tile::static_for<0, n1_loops, 1>{}(
            [&](auto n1_id){
                ck_tile::store_tile(out_dram_window_, ck_tile::cast_tile<out_t>(o_acc[n1_id]));
                if constexpr (n1_id < (n1_loops - 1))
                {
                    ck_tile::move_tile_window(out_dram_window_, {0, Traits::kBlockN1});
                }
            }
        );
        return;
    }

    // 4. Load Q to lds and reg
    //
    auto q_dram_window_nope = ck_tile::make_tile_window(
        q_nope_dram_window_.get_bottom_tensor_view(),
        q_nope_dram_window_.get_window_lengths(),
        q_nope_dram_window_.get_window_origin(),
        Policy::template MakeQRegTileDistribution<false>());
    auto q_dram_window_rope = ck_tile::make_tile_window(
        q_rope_dram_window_.get_bottom_tensor_view(),
        q_rope_dram_window_.get_window_lengths(),
        q_rope_dram_window_.get_window_origin(),
        Policy::template MakeQRegTileDistribution<true>());

    auto q_reg_nope = ck_tile::load_tile(q_dram_window_nope);
    auto q_reg_rope = ck_tile::load_tile(q_dram_window_rope);

    // 5. Prepare KV
    //
    constexpr auto k_nope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeNope>{});
    constexpr auto k_rope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kSizeRope>{});

    auto k_dist = Policy::template MakeKDramTileDistribution<false>();
    auto k_coord = k_dist.calculate_index();
    constexpr auto kKNumRepeat = Policy::template GetNumRepeatOfKDramTileDistribution<false>();
    constexpr auto kKPageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_base_idx = k_coord[kKPageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKNumRepeat> k_offsets;
    ck_tile::statically_indexed_array<bool, kKNumRepeat> k_valids;
    ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_base_idx + Traits::kBlockN0 / kKNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / Traits::kPageBlockSize;
        const int32_t inside_idx = seqlen_idx % Traits::kPageBlockSize;
        k_offsets[rid] = (p_block_table[page_idx] * Traits::kPageBlockSize + inside_idx) * stride_s_k;
        k_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_tensor_view_,
        k_nope_dram_window_lengths,
        {seqlen_k_start, 0},
        k_dist,
        k_offsets,
        k_valids,
        kKPageIdxDim);
    k_dram_window.init_raw();

    ck_tile::static_for<0, Traits::kKNumRepeat, 1>{}([&](auto rid)
    {
        ck_tile::async_load_tile_raw(
            k_nope_st_lds_windows.at(ck_tile::number<0>{}), k_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_nope_repeat_smem_offset * rid);
        __builtin_amdgcn_sched_barrier(0);
        ck_tile::async_load_fence(k_dram_window.get_num_of_access());
        __builtin_amdgcn_s_barrier();
        ck_tile::move_tile_window(k_dram_window, {0, Traits::kMaxSplits});
    });
    ck_tile::move_tile_window(k_dram_window, {0, -Traits::kSizeNope});

    auto k_rope_dist = Policy::template MakeKDramTileDistribution<true>();
    auto k_rope_coord = k_rope_dist.calculate_index();
    constexpr auto kKRopeNumRepeat = Policy::template GetNumRepeatOfKDramTileDistribution<true>();
    constexpr auto kKRopePageIdxDim = ck_tile::number<0>{};
    const int32_t seqlen_k_rope_base_idx = k_rope_coord[kKRopePageIdxDim] + seqlen_k_start;
    ck_tile::statically_indexed_array<int32_t, kKRopeNumRepeat> k_rope_offsets;
    ck_tile::statically_indexed_array<bool, kKRopeNumRepeat> k_rope_valids;
    ck_tile::static_for<0, kKRopeNumRepeat, 1>{}([&](auto rid)
    {
        const int32_t seqlen_idx = seqlen_k_rope_base_idx + Traits::kBlockN0 / kKRopeNumRepeat * rid.value;
        const int32_t page_idx   = seqlen_idx / Traits::kPageBlockSize;
        const int32_t inside_idx = seqlen_idx % Traits::kPageBlockSize;
        k_rope_offsets[rid] = (p_block_table[page_idx] * Traits::kPageBlockSize + inside_idx) * stride_s_k;
        k_rope_valids[rid] = (seqlen_idx < seqlen_k_end);
    });
    auto k_rope_dram_window = ck_tile::make_tile_scatter_gather(
        k_dram_tensor_view_,
        k_rope_dram_window_lengths,
        {seqlen_k_start, Traits::kSizeNope},
        k_rope_dist,
        k_rope_offsets,
        k_rope_valids,
        kKRopePageIdxDim);
    k_rope_dram_window.init_raw();

    ck_tile::async_load_tile_raw(
        k_rope_st_lds_windows(ck_tile::number<0>{}), k_rope_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_rope_smem_offset);
    __builtin_amdgcn_sched_barrier(0);
    ck_tile::async_load_fence(k_rope_dram_window.get_num_of_access());
    __builtin_amdgcn_s_barrier();

    static constexpr auto I0    = ck_tile::number<0>{};
    static constexpr auto I1    = ck_tile::number<1>{};
    static constexpr auto I3    = ck_tile::number<3>{};

    static constexpr uint32_t s_perm0 = 0x07060302;
    static constexpr uint32_t s_perm1 = 0x05040100;
    static constexpr uint32_t VKRepeats = Policy::GetVTileDistributionStride();
    // 6. Main loop
    //
    // Define main loop
    auto main_loop = [&]<bool IsEvenLoop>(int32_t loop_idx)
    {
        ck_tile::clear_tile(s_acc);
        ck_tile::block_sync_lds();

        using k_st_idx   = std::conditional_t<IsEvenLoop, ck_tile::number<1>, ck_tile::number<0>>;
        constexpr auto k_ld_begin = IsEvenLoop ? ck_tile::number<0>{} : ck_tile::number<Traits::kBlockN0>{};
        constexpr auto k_ld_end   = ck_tile::number<k_ld_begin + Traits::kBlockN0>{};
        constexpr ck_tile::array<int32_t, 2> v_lds_direction = { IsEvenLoop ? Traits::kBlockN0 : -Traits::kBlockN0, -Traits::kSizeNope };

        auto vt_tile = ck_tile::make_static_distributed_tensor<scalar_t>(Policy::MakeVTTileDistribution());
#pragma unroll 2
        for (int32_t vidx = 0; vidx < Traits::kKNumRepeat; ++vidx)
        {
            auto v_tile  = ck_tile::load_tile(v_ld_lds_window);
            ck_tile::move_tile_window(v_ld_lds_window, {0, Traits::kMaxSplits});
            auto vt_thread_buffer = vt_tile.get_thread_buffer().get();

            auto k_thread_buffer = v_tile.get_thread_buffer().get();
            ck_tile::static_for<0, VKRepeats, 1>{}([&](auto rid){
                constexpr auto dw_offset   = ck_tile::number<rid * 2>{};
                constexpr auto dw_offset_1 = ck_tile::number<rid * 2 + 1>{};
                constexpr auto vt_dw_offset = rid + VKRepeats;
                reinterpret_cast<uint32_t*>(vt_thread_buffer)[rid] = __builtin_amdgcn_perm(
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset_1],
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset],
                    s_perm1
                );
                reinterpret_cast<uint32_t*>(vt_thread_buffer)[vt_dw_offset] = __builtin_amdgcn_perm(
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset_1],
                    reinterpret_cast<uint32_t*>(k_thread_buffer)[dw_offset],
                    s_perm0
                );
            });
            // ck_tile::block_sync_lds();
            ck_tile::store_tile(vt_st_lds_window, vt_tile);
            ck_tile::move_tile_window(vt_st_lds_window, {Traits::kMaxSplits, 0});
            ck_tile::block_sync_lds();
        }
        ck_tile::move_tile_window(vt_st_lds_window, {-Traits::kSizeNope, 0});
        ck_tile::move_tile_window(v_ld_lds_window, v_lds_direction);

        // I. QK GEMM
        if (Policy::HandleGemm0())
        {
            ck_tile::clear_tile(s_acc);

            // I. QK GEMM
            //
            if constexpr (k0_loops > 1)
            {
                constexpr ck_tile::array<int32_t, 2> qk_direction = {0, Traits::kBlockK0};
                ck_tile::static_for<0, k0_loops, 1>{}(
                    [&](auto k0_id)
                    {
                        ck_tile::block_sync_lds();
                        gemm_0(s_acc,
                               ck_tile::get_slice_tile(q_reg_nope,
                                   ck_tile::sequence<0, k0_id * Traits::kBlockK0>{},
                                   ck_tile::sequence<Traits::kBlockM, (k0_id + 1) * Traits::kBlockK0>{}),
                               k_nope_ld_lds_window);

                        // pre-load k
                        ck_tile::move_tile_window(k_nope_ld_lds_window, qk_direction);
                    });
            }
            if constexpr (k0_loops == 1)
            {
                ck_tile::block_sync_lds();
                __builtin_amdgcn_s_waitcnt(3952);
                gemm_0(s_acc,
                       q_reg_nope,
                       ck_tile::get_slice_tile(k_nope_ld_lds_window,
                            ck_tile::sequence<k_ld_begin, 0>{},
                            ck_tile::sequence<k_ld_end, Traits::kSizeNope>{}));
                ck_tile::block_sync_lds();
            }

            // QK rope tail
            ck_tile::block_sync_lds();
            // __builtin_amdgcn_s_waitcnt(3952);
            gemm_0_rope(s_acc,
                        q_reg_rope,
                        ck_tile::get_slice_tile(k_rope_ld_lds_window,
                            ck_tile::sequence<k_ld_begin, 0>{},
                            ck_tile::sequence<k_ld_end, Traits::kSizeRope>{}));

            ck_tile::tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
        }
        else
        {
            if constexpr (k0_loops > 2)
            {
                ck_tile::static_for<0, k0_loops - 2, 1>{}([&](auto k0_id)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_s_barrier();
                });
            }
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_s_barrier();
        }

        // II. scale_s, mask, softmax
        //

        // Masking
        // Note that masking is also required when k is padded
        const auto k_origin = k_dram_window.get_window_origin();
        const bool need_perpixel_check = mask.IsEdgeTile(
            __builtin_amdgcn_readfirstlane(q_origin.at(ck_tile::number<0>{})),
            __builtin_amdgcn_readfirstlane(k_origin.at(ck_tile::number<0>{})),
            ck_tile::number<Traits::kBlockM>{},
            ck_tile::number<Traits::kBlockN0>{});

        if (need_perpixel_check)
        {
            ck_tile::set_tile_if(
                s_acc, -ck_tile::numeric<acc_t>::infinity(),
                [&](auto ids)
                {
                    const auto row = q_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<0>{});
                    const auto col = k_origin.at(ck_tile::number<0>{}) + ids.at(ck_tile::number<1>{});
                    return mask.IsOutOfBound(row, col);
                });
        }

        s_acc_shuffled = ShuffleSacc<Policy>(
            s_acc, p_smem + Policy::GetPSmemStart());

        // Get max of row
        auto m_local = block_reduce_2d(s_acc_shuffled, reduce_max_func.GetIdentityValue<acc_t>(), reduce_max_func);
        block_reduce_2d_sync(m_local, reduce_max_func);

        const auto m_old = m;
        ck_tile::tile_elementwise_inout(
            [](auto& e0, auto e1, auto e2) { e0 = ck_tile::max(e1, e2); }, m, m_old, m_local);

        // Compute exp(x_i - m)
        auto p_intermedia = ck_tile::make_static_distributed_tensor<acc_t>(s_acc_shuffled.get_tile_distribution());
        const auto p_spans = decltype(p_intermedia)::get_distributed_spans();
        ck_tile::sweep_tile_span(
            p_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                auto row_max  = GetValidatedMax<Mask::IsMasking>(m[i]);
                ck_tile::sweep_tile_span(
                    p_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
#if FMLA_FWD_FAST_EXP2
                        p_intermedia(ij) = ck_tile::exp2(s_acc_shuffled[ij] - row_max);
#else
                        p_intermedia(ij) = ck_tile::exp(s_acc_shuffled[ij] - row_max);
#endif
                    });
            });

        // Compute row sum of exp(x_i - m)
        auto rowsum_p = block_reduce_2d(p_intermedia, reduce_sum_func.GetIdentityValue<acc_t>(), reduce_sum_func);
        block_reduce_2d_sync(rowsum_p, reduce_sum_func);

        // Calculate new l and adjust old output acc
        constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();
        ck_tile::sweep_tile_span(
            o_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                const auto row_max = GetValidatedMax<Mask::IsMasking>(m[i]);
#if FMLA_FWD_FAST_EXP2
                const auto temp_i  = ck_tile::exp2(m_old[i] - row_max);
#else
                const auto temp_i  = ck_tile::exp(m_old[i] - row_max);
#endif
                l(i) = temp_i * l[i] + rowsum_p[i];
                ck_tile::sweep_tile_span(
                    o_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
                        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
                        {
#if 0
                            acc_t o_acc_v = o_acc[n1_id](ij);
                            asm volatile("v_mul_f32 %[v_o_acc], %[v_tmp], %[v_o_acc]\n"
                                        : [v_o_acc] "+v"(o_acc_v)
                                        : [v_tmp] "v"(temp_i));
                            o_acc[n1_id](ij) = o_acc_v;
#else
                            o_acc[n1_id](ij) *= temp_i;
#endif
                        });
                    });
            });


        // III. GEMM for PV and load next k block
        //
        if (loop_idx != num_total_loop - 1) {
            // Load next K
            ck_tile::move_tile_window(k_dram_window, {Traits::kBlockN0, 0});
            ck_tile::move_tile_window(k_rope_dram_window, {Traits::kBlockN0, 0});

            // Recalculate offsets
            ck_tile::static_for<0, kKNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / Traits::kPageBlockSize;
                const int32_t inside_idx = seqlen_idx % Traits::kPageBlockSize;
                k_offsets[rid] = (p_block_table[page_idx] * Traits::kPageBlockSize + inside_idx) * stride_s_k;
                k_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_dram_window.update_page_idx_and_valids(k_offsets, k_valids);

            ck_tile::static_for<0, kKRopeNumRepeat, 1>{}([&](auto rid)
            {
                const int32_t seqlen_idx = seqlen_k_rope_base_idx + (loop_idx + 1) * Traits::kBlockN0 +
                                           Traits::kBlockN0 / kKRopeNumRepeat * rid.value;
                const int32_t page_idx   = seqlen_idx / Traits::kPageBlockSize;
                const int32_t inside_idx = seqlen_idx % Traits::kPageBlockSize;
                k_rope_offsets[rid] = (p_block_table[page_idx] * Traits::kPageBlockSize + inside_idx) * stride_s_k;
                k_rope_valids[rid] = (seqlen_idx < seqlen_k_end);
            });
            k_rope_dram_window.update_page_idx_and_valids(k_rope_offsets, k_rope_valids);

            ck_tile::static_for<0, 4, 1>{}([&](auto rid)
            {
                ck_tile::async_load_tile_raw(
                    k_nope_st_lds_windows.at(k_st_idx{}), k_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_nope_repeat_smem_offset * rid);
                __builtin_amdgcn_sched_barrier(0);
                ck_tile::async_load_fence(k_dram_window.get_num_of_access());
                __builtin_amdgcn_s_barrier();
                ck_tile::move_tile_window(k_dram_window, {0, Traits::kMaxSplits});
            });
            ck_tile::move_tile_window(k_dram_window, {0, -Traits::kSizeNope});

            ck_tile::async_load_tile_raw(
                k_rope_st_lds_windows.at(k_st_idx{}), k_rope_dram_window, ck_tile::number<-1>{}, k_oob_ck, k_pre_np, k_rope_smem_offset);
            __builtin_amdgcn_sched_barrier(0);
            ck_tile::async_load_fence(k_rope_dram_window.get_num_of_access());
            __builtin_amdgcn_s_barrier();
        }

        const auto p = ck_tile::cast_tile<scalar_t>(p_intermedia);
        if constexpr (k1_loops == 1)
        {

            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id) {
                ck_tile::block_sync_lds();
                gemm_1(o_acc[n1_id],
                       p,
                       vt_ld_lds_window);
            });
        }
        else
        {
            ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id) {
                ck_tile::static_for<0, k1_loops, 1>{}([&](auto k1_id) {
                    ck_tile::block_sync_lds();
                    gemm_1(o_acc[n1_id],
                           ck_tile::get_slice_tile(
                               p,
                               ck_tile::sequence<0, k1_id * Traits::kBlockK1>{},
                               ck_tile::sequence<Traits::kBlockM, (k1_id + 1) * Traits::kBlockK1>{}),
                           vt_ld_lds_window);
                    ck_tile::block_sync_lds();
                    ck_tile::move_tile_window(vt_ld_lds_window, {0, Traits::kBlockK1});
                });
                vt_ld_lds_window.set_window_origin({(n1_id + 1) * Traits::kBlockN1, 0});
            });
        }
    };


    // Execute the loop
    main_loop.template operator()<true>(0);
    for (int32_t loop_idx = 1; (loop_idx + 1) < num_total_loop; (loop_idx += 2))
    {
        main_loop.template operator()<false>(loop_idx);
        main_loop.template operator()<true>(loop_idx + 1);
    }
    if ((num_total_loop % 2) == 0)
    {
        main_loop.template operator()<false>(num_total_loop - 1);
    }

    // 7. tile epilogue: store LSE, Adjust and output
    kn_fmla_fwd_splitkv_prefill_tile_epilogue<Traits, scalar_t, acc_t, out_t>(
        lse_dram_window_,
        out_dram_window_,
        o_acc,
        m,
        l,
        mask);
}

// =====================================================================================================================
// Kernel Entry
//

template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool kIsCausal, bool kDoSplit>
__launch_bounds__(Traits::kNumThreads, Traits::kWaveOccupancy)
__global__ void kn_fmla_fwd_splictkv_prefill(
    const FlashMlaPrefillFwdParams params)
{
    using Policy = FlashMlaPrefillPolicy<Traits, scalar_t, acc_t>;

    // allocate LDS
    __shared__ uint8_t p_smem[Policy::GetSmemSize()];

    const auto [tile_m_id, split_id, hqid, bid] =
        kDoSplit ? GetTileIndex<Traits>(params.num_splits) : GetTileIndex<Traits>(1);
    const auto hqid_xqa =
        (Traits::kXqaStrategy == XqaStrategy::Internal) ? hqid * params.hq_hk_ratio : hqid;
    const auto hkid   = hqid_xqa / params.hq_hk_ratio;
    const int32_t mid = __builtin_amdgcn_readfirstlane(tile_m_id * Traits::kBlockM);

    constexpr bool enableXqa = (Traits::kXqaStrategy != XqaStrategy::Disable);
    // Define causal mask
    using Mask             = ck_tile::SimplifiedGenericAttentionMask<kIsCausal, enableXqa>;
    const int32_t seqlen_k = params.p_seqlens_k[bid];
    Mask mask              = kIsCausal ? Mask{params.size_s_ori,
                                              seqlen_k - params.size_s_ori + 1,
                                              params.size_s_pk,
                                              seqlen_k,
                                              params.mask_y_ratio_mdiv}
                                       : Mask{params.size_s_pk, seqlen_k};

    constexpr auto q_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockK0>{});
    constexpr auto q_nope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kSizeNope>{});
    constexpr auto q_rope_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kSizeRope>{});
    constexpr auto k_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{});
    constexpr auto v_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{}, ck_tile::number<Traits::kBlockK1>{});

    const scalar_t* p_query = reinterpret_cast<const scalar_t*>(params.p_query) +
                              int64_t(hqid_xqa) * params.stride_h_q +   // head offset
                              int64_t(bid) * params.stride_b_q;     // batch offset
    const scalar_t* p_key   = reinterpret_cast<const scalar_t*>(params.p_key) +
                              int64_t(hkid) * params.stride_h_k;    // head offset
    const scalar_t* p_value = reinterpret_cast<const scalar_t*>(params.p_value) +
                              int64_t(hkid) * params.stride_h_v;    // head offset
    const int32_t*  p_block_table = params.p_block_table +
                                    int64_t(bid) * params.block_table_batch_stride; // batch offset

    const int32_t kv_cache_width = params.num_page_blocks * params.page_block_size;

    const auto q_dram = MakeQDram<Policy>(
        p_query, params.size_s_tr, params.stride_s_q, params.hq_hk_ratio, params.stride_h_q);
    const auto k_dram = MakeKDram<Policy>(p_key,   kv_cache_width, params.stride_s_k);
    const auto v_dram = MakeVDram<Policy>(p_value, kv_cache_width, params.stride_s_v);    

    auto q_dram_window = ck_tile::make_tile_window(q_dram, q_dram_window_lengths, {mid, 0});
    auto k_dram_window = ck_tile::make_tile_window(k_dram, k_dram_window_lengths, {0, 0});
    auto v_dram_window = ck_tile::make_tile_window(v_dram, v_dram_window_lengths, {0, 0});

    if constexpr (kDoSplit)
    {
        acc_t* p_lse_acc = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                           int64_t(hqid_xqa) * params.stride_h_lseacc +     // head offset
                           int64_t(bid) * params.stride_b_lseacc +      // batch offset
                           int64_t(split_id) * params.stride_sp_lseacc; // split offset
        out_t* p_out_acc = reinterpret_cast<out_t*>(params.p_output_accum) +
                           int64_t(hqid_xqa) * params.stride_h_oacc +      // head offset
                           int64_t(bid) * params.stride_b_oacc +       // batch offset
                           int64_t(split_id) * params.stride_sp_oacc;  // split offset

        auto lse_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_acc_dram = MakeLseAccDram<Policy>(p_lse_acc,
                                                         lse_acc_dram_window_lengths,
                                                         params.size_s_tr,
                                                         params.hq_hk_ratio,
                                                         params.stride_h_lseacc);
        const auto out_acc_dram = MakeOutAccDram<Policy>(p_out_acc,
                                                         params.size_s_tr,
                                                         params.stride_s_oacc,
                                                         params.hq_hk_ratio,
                                                         params.stride_h_oacc);

        auto lse_acc_dram_window =
            ck_tile::make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {mid});
        auto out_acc_dram_window =
            ck_tile::make_tile_window(out_acc_dram, out_acc_dram_window_lengths, {mid, 0});


        if constexpr (!Traits::kKVLoadOnce) {
            kn_fmla_fwd_splitkv_prefill_tile<Traits, scalar_t, acc_t, out_t>(
                q_dram_window,
                k_dram_window,
                v_dram_window,
                lse_acc_dram_window,
                out_acc_dram_window,
                p_block_table,
                params.page_block_size,
                params.stride_s_k,
                params.stride_s_v,
                seqlen_k,
                params.num_splits,
                split_id,
                mask,
#if FMLA_FWD_FAST_EXP2
            static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
            params.scale_softmax,
#endif
                p_smem);
        }
        else
        {
            const auto q_nope_dram_window = ck_tile::make_tile_window(
                q_dram,
                q_nope_dram_window_lengths,
                {mid, 0});
            const auto q_rope_dram_window = ck_tile::make_tile_window(
                q_dram,
                q_rope_dram_window_lengths,
                {mid, Traits::kSizeNope});
            kn_fmla_fwd_splitkv_prefill_load_once_tile<Traits, scalar_t, acc_t, out_t>(
                q_nope_dram_window,
                q_rope_dram_window,
                k_dram,
                lse_acc_dram_window,
                out_acc_dram_window,
                p_block_table,
                params.stride_s_k,
                params.stride_s_v,
                seqlen_k,
                params.num_splits,
                split_id,
                mask,
#if FMLA_FWD_FAST_EXP2
            static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
            params.scale_softmax,
#endif
                p_smem);
        }
    }
    else
    {
        // Assuming lse is in shape [b, h, s] and is contiguous
        acc_t* p_lse =
            reinterpret_cast<acc_t*>(params.p_softmax_lse) +
            (int64_t(bid) * params.size_h_tr + hqid_xqa) * params.size_s_tr; // batch+head offset
        out_t* p_out = reinterpret_cast<out_t*>(params.p_output) +
                       int64_t(hqid_xqa) * params.stride_h_o + // head offset
                       int64_t(bid) * params.stride_b_o;       // batch offset

        auto lse_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_dram = MakeLseDram<Policy>(p_lse,
                                                  lse_dram_window_lengths,
                                                  params.size_s_tr,
                                                  params.hq_hk_ratio,
                                                  params.stride_h_lse);
        const auto out_dram = MakeOutDram<Policy>(
            p_out, params.size_s_tr, params.stride_s_o, params.hq_hk_ratio, params.stride_h_o);

        auto lse_dram_window =
            ck_tile::make_tile_window(lse_dram, lse_dram_window_lengths, {mid});
        auto out_dram_window =
            ck_tile::make_tile_window(out_dram, out_dram_window_lengths, {mid, 0});

        if constexpr (!Traits::kKVLoadOnce)
        {
            kn_fmla_fwd_splitkv_prefill_tile<Traits, scalar_t, acc_t, out_t>(
                q_dram_window,
                k_dram_window,
                v_dram_window,
                lse_dram_window,
                out_dram_window,
                p_block_table,
                params.page_block_size,
                params.stride_s_k,
                params.stride_s_v,
                seqlen_k,
                1, // num_splits
                0, // split_id
                mask,
#if FMLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
        else
        {
            const auto q_nope_dram_window = ck_tile::make_tile_window(
                q_dram,
                q_nope_dram_window_lengths,
                {mid, 0});
            const auto q_rope_dram_window = ck_tile::make_tile_window(
                q_dram,
                q_rope_dram_window_lengths,
                {mid, Traits::kSizeNope});
            kn_fmla_fwd_splitkv_prefill_load_once_tile<Traits, scalar_t, acc_t, out_t>(
                q_nope_dram_window,
                q_rope_dram_window,
                k_dram,
                lse_dram_window,
                out_dram_window,
                p_block_table,
                params.stride_s_k,
                params.stride_s_v,
                seqlen_k,
                params.num_splits,
                split_id,
                mask,
#if FMLA_FWD_FAST_EXP2
                static_cast<float>(params.scale_softmax * ck_tile::log2e_v<>),
#else
                params.scale_softmax,
#endif
                p_smem);
        }
    }
}

template <typename Traits, int32_t kMaxSplits, typename out_t, typename in_t>
__global__ void kn_fmla_fwd_splictkv_prefill_combine(
    const FlashMlaPrefillFwdParams params)
{
    using Policy  = FlashMlaCombineKernelPolicy<Traits, out_t, in_t>;
    using index_t = int64_t;

    __shared__ in_t lds_lse_scale[kMaxSplits];

    const int32_t bidx = blockIdx.z;

    const int32_t num_splits   = params.num_splits;
    const int32_t split_offset = bidx * params.num_splits;
    assert((num_splits > 1) && (num_splits <= kMaxSplits));

    const int32_t lane_id          = ck_tile::get_lane_id();
    const int32_t hidx             = blockIdx.y;
    const int32_t sidx             = blockIdx.x;
    const int32_t hsidx            = hidx * params.size_s_tr + sidx;
    const int32_t shidx            = hidx + sidx * params.size_h_tr;
    const int32_t size_hs          = params.size_h_tr * params.size_s_tr;
    const index_t offset_lse_accum = split_offset * size_hs + hsidx; // offset to split 0
    const index_t offset_lse       = bidx * size_hs + hsidx;

    if (ck_tile::get_warp_id() == 0)
    {
        const in_t* p_lse_accum = reinterpret_cast<in_t*>(params.p_softmax_lseaccum) + offset_lse_accum;
        in_t* p_lse             = reinterpret_cast<in_t*>(params.p_softmax_lse) + offset_lse;

        constexpr int32_t kNumLsePerThr = ck_tile::integer_divide_ceil(kMaxSplits, ck_tile::get_warp_size());
        in_t local_lse[kNumLsePerThr];

        // Load thread local LSE and get local max LSE
        in_t max_lse = -ck_tile::numeric<in_t>::infinity();
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
            const in_t lse =
                (split_idx < num_splits) ? p_lse_accum[split_idx * size_hs] : -ck_tile::numeric<in_t>::infinity();
            local_lse[i] = lse;
            max_lse = ck_tile::max(max_lse, lse);
        }

        // Get global max LSE
        #pragma unroll
        for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
        {
            max_lse = ck_tile::max(max_lse, __shfl_xor(max_lse, offset));
        }

        // Get sum of LSE
        in_t sum_lse = 0.f;
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
#ifdef FMLA_FWD_FAST_EXP2
            static_assert(0, "have not figured out if need exp2 here");
#endif
            sum_lse += ck_tile::exp(local_lse[i] - max_lse);
        }
        #pragma unroll
        for (int32_t offset = ck_tile::get_warp_size() / 2; offset > 0; offset /= 2)
        {
            sum_lse += __shfl_xor(sum_lse, offset);
        }

        // Get global LSE
        const auto [global_lse, output_lse] = [&]() {
            if ((sum_lse == 0.f) || (sum_lse != sum_lse))
            {
                return ck_tile::make_tuple(ck_tile::numeric<in_t>::infinity(), -ck_tile::numeric<in_t>::infinity());
            }
            else
            {
                const in_t lse = ck_tile::log(sum_lse) + max_lse;
                return ck_tile::make_tuple(lse, lse);
            }
        } ();

        if (lane_id == 0)
        {
            *p_lse = output_lse;
        }

        // Write LSE to LDS
        #pragma unroll
        for (int32_t i = 0; i < kNumLsePerThr; ++i)
        {
            const int32_t split_idx = i * ck_tile::get_warp_size() + lane_id;
            if (split_idx < num_splits)
            {
                lds_lse_scale[split_idx] = ck_tile::exp(local_lse[i] - global_lse);
            }
        }
    }

    __builtin_amdgcn_sched_barrier(0);
    ck_tile::block_sync_lds();

    static_assert(Traits::kSizeDV % Traits::kNumThreadsCombine == 0);

    auto oaccu_window =
        Policy::MakeOaccuTileWindow(params.p_output_accum, shidx, size_hs, split_offset, num_splits);

    auto reg_out = ck_tile::make_static_distributed_tensor<in_t>(
        decltype(ck_tile::load_tile(oaccu_window))::get_tile_distribution());
    ck_tile::set_tile(reg_out, 0.f);

    for (int32_t split_idx = 0; split_idx < num_splits; ++split_idx)
    {
        const in_t lse_scale = lds_lse_scale[split_idx];
        auto oaccu = ck_tile::load_tile(oaccu_window);
        ck_tile::sweep_tile(oaccu, [&](auto idx) {
            reg_out(idx) += lse_scale * oaccu(idx);
        });
        ck_tile::move_tile_window(oaccu_window, {size_hs, 0});
    }

    auto dram_out = Policy::MakeOutputTileWindow(
        static_cast<out_t*>(params.p_output) +
        bidx * params.stride_b_o + hidx * params.stride_h_o + sidx * params.stride_s_o);
    ck_tile::store_tile(dram_out, ck_tile::cast_tile<out_t>(reg_out));
}

// =====================================================================================================================
// Dispatch
//

template <typename Traits, typename scalar_t, typename acc_t, typename out_t, bool kIsCausal>
void dispatch_fmla_fwd_splictkv_prefill(
    const FlashMlaPrefillFwdParams& params)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int32_t num_blk =
        ck_tile::integer_divide_ceil(params.size_s_pk, Traits::kBlockM) * params.num_splits;
    const dim3 grid_attn = dim3(num_blk, params.size_h_pk, params.size_b);
    const dim3 grid_comb = dim3(params.size_s_tr, params.size_h_tr, params.size_b);


    if (params.num_splits > 1)
    {
        // out_t is not take into consideration when doing splits because combine shader is always expected to do
        // the final output type conversion.
        auto kn_attn = &kn_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, acc_t, kIsCausal, true>;
        auto kn_comb =
            (params.num_splits <= 32)  ? &kn_fmla_fwd_splictkv_prefill_combine<Traits, 32,  scalar_t, acc_t> :
            // (params.num_splits <= 64)  ? &kn_fmla_fwd_splictkv_prefill_combine<Traits, 64,  scalar_t, acc_t> :
            // (params.num_splits <= 96)  ? &kn_fmla_fwd_splictkv_prefill_combine<Traits, 96,  scalar_t, acc_t> :
            // (params.num_splits <= 128) ? &kn_fmla_fwd_splictkv_prefill_combine<Traits, 128, scalar_t, acc_t> :
            static_cast<decltype(kn_fmla_fwd_splictkv_prefill_combine<Traits, 32, scalar_t, acc_t>)*>(nullptr);
        TORCH_CHECK(kn_comb != nullptr, "num_splits is larger than expected (<=128) !");
        kn_attn<<<grid_attn, Traits::kNumThreads, 0, stream>>>(params);
        kn_comb<<<grid_comb, Traits::kNumThreadsCombine, 0, stream>>>(params);
    }
    else
    {
        auto kn_attn = &kn_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, kIsCausal, false>;
        kn_attn<<<grid_attn, Traits::kNumThreads, 0, stream>>>(params);
    }
}

// =====================================================================================================================
// Interfaces
//
#define DISPATCH_FMLA_TYPES(TYPE, IS_CAUSAL, NAME, ...)                      \
    switch ((TYPE))                                                          \
    {                                                                        \
        case at::ScalarType::BFloat16:                                       \
        {                                                                    \
            using scalar_t = ck_tile::bf16_t;                                \
            using out_t = std::conditional_t<kForceOutAcc, acc_t, scalar_t>; \
            if ((IS_CAUSAL))                                                 \
            {                                                                \
                constexpr bool Is_causal = true;                             \
                __VA_ARGS__;                                                 \
            }                                                                \
            else                                                             \
            {                                                                \
                constexpr bool Is_causal = false;                            \
                __VA_ARGS__;                                                 \
            }                                                                \
            break;                                                           \
        }                                                                    \
        case at::ScalarType::Half:                                           \
        {                                                                    \
            using scalar_t = ck_tile::fp16_t;                                \
            using out_t = std::conditional_t<kForceOutAcc, acc_t, scalar_t>; \
            if ((IS_CAUSAL))                                                 \
            {                                                                \
                constexpr bool Is_causal = true;                             \
                __VA_ARGS__;                                                 \
            }                                                                \
            else                                                             \
            {                                                                \
                constexpr bool Is_causal = false;                            \
                __VA_ARGS__;                                                 \
            }                                                                \
            break;                                                           \
        }                                                                    \
        default:                                                             \
            TORCH_CHECK(false, NAME " does't support ",                      \
                        toString((TYPE)), ".");                              \
    }

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    int32_t result = 1;

    if (batch_nhead_mblocks < 0.8f * num_SMs)
    {
        max_splits = std::min(max_splits, std::min(num_SMs, num_n_blocks));
        float max_efficiency = 0.f;
        std::vector<float> efficiency;
        efficiency.reserve(max_splits);

        // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
        // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
        // (i.e. it's 11 splits anyway).
        // So we check if the number of blocks per split is the same as the previous num_splits.
        auto is_split_eligible = [&num_n_blocks](int num_splits) {
            return (num_splits == 1) ||
                (ck_tile::integer_divide_ceil(num_n_blocks, num_splits) !=
                 ck_tile::integer_divide_ceil(num_n_blocks, num_splits - 1));
        };

        for(int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
            if(!is_split_eligible(num_splits))
            {
                efficiency.push_back(0.f);
            }
            else
            {
                float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
                float eff     = n_waves / ceil(n_waves);
                if(eff > max_efficiency)
                {
                    max_efficiency = eff;
                }
                efficiency.push_back(eff);
            }
        }

        for(int num_splits = 1; num_splits <= max_splits; num_splits++)
        {
            if(!is_split_eligible(num_splits))
            {
                continue;
            }

            if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
            {
                result = num_splits;
                break;
            }
        }
    }

    return result;
}

template <typename Traits>
int32_t calculate_num_splits(
    const int32_t size_b,
    const int32_t size_h,
    const int32_t size_s)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    ck_tile::hip_check_error(hipGetDevice(&dev));
    ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
    const int32_t cu_count = dev_prop.multiProcessorCount;

    const int32_t num_m_blocks = ck_tile::integer_divide_ceil(size_s, Traits::kBlockM);
    const int32_t num_n_blocks = ck_tile::integer_divide_ceil(Traits::kSizeDV, Traits::kBlockN1);

    return num_splits_heuristic(size_b * size_h * num_m_blocks, cu_count * Traits::kCuReuse, num_n_blocks, 128);
}

std::vector<torch::Tensor> flash_mla_fwd_prefill_with_kvcache_impl(
    torch::Tensor&       query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const int32_t        head_size_v,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const float          softmax_scale,
    const bool           is_causal)
{
    constexpr bool kKVLoadOnce        = false;
    constexpr XqaStrategy kXqaStrategy = XqaStrategy::Internal;
    //                             dqk  dv   m0  n0  n1   #warp  wave_occu
    using Traits = std::conditional_t<kKVLoadOnce,
        FlashMlaPrefillKernelTrait<576, 512, 64, 16, 512, 8,     1,   kKVLoadOnce, kXqaStrategy>,
        FlashMlaPrefillKernelTrait<576, 512, 64, 64, 256, 4,     2,   kKVLoadOnce, kXqaStrategy>>;
    constexpr bool kForceOutAcc = false;
    using acc_t                 = float;

    torch::Tensor vcache = value_cache.data_ptr() ? value_cache : key_cache;

    auto opts = query.options();
    static_assert(std::is_same_v<acc_t, float>);
    auto opts_acc = opts.dtype(torch::kFloat32);

    const int32_t batch_size      = query.size(0);
    const int32_t seqlen_q_ori    = query.size(1);
    const int32_t num_heads_q_ori = query.size(2);
    int32_t seqlen_q              = seqlen_q_ori;
    int32_t num_heads_q           = num_heads_q_ori;

    const int32_t head_size = query.size(3);
    TORCH_CHECK((head_size == 576) && (head_size_v == HEADV), "Only support QK head dim 576 and V head dim HEADV!");

    const int32_t num_blocks      = key_cache.size(0);
    const int32_t page_block_size = key_cache.size(1);
    const int32_t num_heads_k     = key_cache.size(2);

    TORCH_CHECK(num_heads_q % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    const int32_t hq_hk_ratio_ori = num_heads_q_ori / num_heads_k;
    int32_t hq_hk_ratio = hq_hk_ratio_ori;
    int32_t mask_y_ratio      = 1;

    if constexpr(Traits::kXqaStrategy != XqaStrategy::Disable)
    {
        seqlen_q     = seqlen_q_ori * hq_hk_ratio_ori;
        num_heads_q  = num_heads_k;
        mask_y_ratio = hq_hk_ratio_ori;
        if constexpr(Traits::kXqaStrategy == XqaStrategy::External) {
            hq_hk_ratio = 1;
            if(num_heads_k == 1)
            {
                query = query.reshape({batch_size, seqlen_q, num_heads_q, head_size});
            }
            else
            {
                query = query.view({batch_size, seqlen_q_ori, num_heads_q, hq_hk_ratio_ori, head_size})
                            .transpose(2, 3)
                            .reshape({batch_size, seqlen_q, num_heads_q, head_size});
            }
        }
    }

    const int32_t num_splits = calculate_num_splits<Traits>(batch_size, num_heads_q, seqlen_q);
    const bool    do_splits = num_splits > 1;

    int32_t seqlen_q_tr = Traits::kXqaStrategy == XqaStrategy::Internal ? seqlen_q_ori : seqlen_q;
    int32_t num_heads_q_tr = Traits::kXqaStrategy == XqaStrategy::Internal ? num_heads_q_ori : num_heads_q;
    // Combine shader, which only exists when num_splits > 1, will conduct type convert by default and force.
    // Thus, kForceOutAcc doesn't work in this case.
    auto output = torch::empty({batch_size, seqlen_q_tr, num_heads_q_tr, head_size_v},
                               (kForceOutAcc && !do_splits) ? opts_acc : opts);
    auto softmax_lse = torch::empty({batch_size, num_heads_q_tr, seqlen_q_tr}, opts_acc);

    FlashMlaPrefillFwdParams params = {};

    params.num_splits    = num_splits;
    params.p_seqlens_k   = cache_seqlens.data_ptr<int32_t>();
    params.p_block_table = block_table.data_ptr<int32_t>();

    params.p_query            = query.data_ptr();
    params.p_key              = key_cache.data_ptr();
    params.p_value            = vcache.data_ptr();
    params.p_output           = output.data_ptr();
    params.p_softmax_lse      = softmax_lse.data_ptr();

    params.size_b                   = batch_size;
    params.size_s_pk                = seqlen_q;
    params.size_s_ori               = seqlen_q_ori;
    params.size_s_tr                = seqlen_q_tr;
    params.size_h_pk                = num_heads_q;
    params.size_h_ori               = num_heads_q_ori;
    params.size_h_tr                = num_heads_q_tr;
    params.hq_hk_ratio              = hq_hk_ratio;
    params.block_table_batch_stride = block_table.stride(0);
    params.num_page_blocks          = num_blocks;
    params.page_block_size          = page_block_size;
    params.scale_softmax            = softmax_scale;

    params.mask_y_ratio_mdiv = ck_tile::mdiv{static_cast<uint32_t>(mask_y_ratio)};

    params.stride_b_q   = query.stride(0);
    params.stride_s_q   = query.stride(1);
    params.stride_h_q   = query.stride(2);
    params.stride_b_k   = key_cache.stride(0);
    params.stride_s_k   = key_cache.stride(1); // size_hk * size_d
    params.stride_h_k   = key_cache.stride(2);
    params.stride_b_v   = vcache.stride(0);
    params.stride_s_v   = vcache.stride(1); // size_hk * size_d
    params.stride_h_v   = vcache.stride(2);
    params.stride_b_o   = output.stride(0);
    params.stride_s_o   = output.stride(1);
    params.stride_h_o   = output.stride(2);
    params.stride_h_lse = softmax_lse.stride(1);

    if(num_splits > 1)
    {
        auto output_accum =
            torch::empty({batch_size, num_splits, seqlen_q_tr, num_heads_q_tr, head_size_v}, opts_acc);
        auto softmax_lseaccum =
            torch::empty({batch_size, num_splits, num_heads_q_tr, seqlen_q_tr}, opts_acc);

        params.p_softmax_lseaccum = softmax_lseaccum.data_ptr();
        params.p_output_accum     = output_accum.data_ptr();
        params.stride_b_oacc      = output_accum.stride(0);
        params.stride_h_oacc      = output_accum.stride(3);
        params.stride_sp_oacc     = output_accum.stride(1);
        params.stride_s_oacc      = output_accum.stride(2);
        params.stride_b_lseacc    = softmax_lseaccum.stride(0);
        params.stride_h_lseacc    = softmax_lseaccum.stride(2);
        params.stride_sp_lseacc   = softmax_lseaccum.stride(1);
    }

    DISPATCH_FMLA_TYPES(
        query.scalar_type(),
        is_causal,
        "fmla_fwd",
        [&](){
            dispatch_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, Is_causal>(params);
        }();
    );
    // assert(is_causal == false);
    // assert(query.scalar_type() == at::ScalarType::BFloat16);
    // using scalar_t = ck_tile::bf16_t;
    // using out_t = std::conditional_t<kForceOutAcc, acc_t, scalar_t>;
    // dispatch_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, false>(params);

    if constexpr(Traits::kXqaStrategy == XqaStrategy::External)
    {
        // post process for out and softmax_lse
        if(num_heads_k == 1)
        {
            output = output.reshape({batch_size, seqlen_q_ori, num_heads_q_ori, head_size_v});
        }
        else
        {
            output = output.view({batch_size, seqlen_q_ori, hq_hk_ratio_ori, num_heads_q, head_size_v})
                         .transpose(2, 3)
                         .reshape({batch_size, seqlen_q_ori, num_heads_q_ori, head_size_v});
        }
        softmax_lse = softmax_lse.view({batch_size, num_heads_q, seqlen_q_ori, hq_hk_ratio_ori})
                          .transpose(2, 3)
                          .reshape({batch_size, num_heads_q_ori, seqlen_q_ori});
    }

    return {output.to(opts), softmax_lse};
}
