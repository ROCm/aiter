// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/core/tensor/tile_scatter_gather.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/fmha.hpp>
#include <ck_tile/ops/gemm.hpp>

// =====================================================================================================================
// Utils
//

CK_TILE_DEVICE bool IsDebugThreadBlock(const int x = 0, const int y = 0, const int z = 0)
{
    return blockIdx.x == x && blockIdx.y == y && blockIdx.z == z;
}

// Returns count of warps which don't contain any idle thread.
template <int32_t NumWarps, int32_t M, int32_t N>
CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
{
    static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);
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
          bool    kEnableXQA_>
struct FlashMlaPrefillKernelTrait
{
    static constexpr int32_t kSizeD                     = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                    = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps                  = kNumWarps_;
    static constexpr int32_t kNumThreads                = kNumWarps * ck_tile::get_warp_size();
    static constexpr int32_t kWaveOccupancy             = 2;
    static constexpr int32_t kNumWarpsSoftmax           = 4;
    static constexpr int32_t kNumThreadsSoftmax         = kNumWarpsSoftmax * ck_tile::get_warp_size();
    static constexpr int32_t kNumWarpsCombine           = 4;
    static constexpr int32_t kNumThreadsCombine         = kNumWarpsCombine * ck_tile::get_warp_size();
    static constexpr int32_t kBlockM                    = kBlockM_;
    static constexpr int32_t kBlockN0                   = kBlockN0_;
    static constexpr int32_t kBlockN1                   = kBlockN1_;
    static constexpr int32_t kBlockK0                   = 32;
    static constexpr int32_t kBlockK1                   = 16;
    static constexpr int32_t kFixedOverheadNumBlocks    = 5;
    static constexpr int32_t kMaxBatchSize              = 4096;
    static constexpr int32_t kCuReuse                   = 2;
    static constexpr int32_t kMaxSplits                 = 128;
    static constexpr bool    kPadHeadDimQ               = false;
    static constexpr bool    kPadHeadDimV               = false;
    static constexpr bool    kPadSeqLenQ                = true;
    static constexpr bool    kPadSeqLenK                = true;
    static constexpr bool    kEnableXQA                 = kEnableXQA_;

    static constexpr int32_t kNumPrefetchK  = 1;
    static constexpr int32_t kNumPrefetchV  = 1;
    static constexpr int32_t kNumPrefetchKV = ck_tile::max(kNumPrefetchK, kNumPrefetchV);

    using QKWarpTile = ck_tile::sequence<16, 16, 16>;
    using KVWarpTile = ck_tile::sequence<16, 16, 16>;

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
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kMPerBlock = Traits::kBlockM;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(scalar_t);

        // this should align with MakeQDramTileDistribution()
        constexpr int32_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return ck_tile::min(ElemPerThread, MaxVectorSize);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetAlignmentK()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(scalar_t);
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
            ck_tile::min(total_pixels, static_cast<int32_t>(16 / sizeof(scalar_t)));
        constexpr int32_t kMinVecLoad = 4 / sizeof(scalar_t);

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
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(scalar_t)));
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
            result = ck_tile::min(N1, static_cast<int32_t>(16 / sizeof(acc_t)));
        }

        return result;
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLse()
    {
        return GetVectorSizeForTile<Traits::kNumWarps,
                                    Traits::kMaxSplits,
                                    Traits::kBlockM,
                                    acc_t>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;
        constexpr int32_t kKPack     = 16 / sizeof(scalar_t);

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

    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
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

        constexpr auto v_lds_block_desc_0 = ck_tile::make_naive_tensor_descriptor(
            ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchKV>{},
                       ck_tile::number<kKPerBlock / kKPack>{},
                       ck_tile::number<kNPerBlock / NPerRow>{},
                       ck_tile::number<NPerRow>{},
                       ck_tile::number<kKPack>{}),
            ck_tile::make_tuple(ck_tile::number<GetSmemSizeSingleKV()>{},
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
                    ck_tile::number<Traits::kNumPrefetchKV>{}, ck_tile::number<kNPerBlock / NPerRow>{}, ck_tile::number<NPerRow>{})),
                ck_tile::make_merge_transform(ck_tile::make_tuple(ck_tile::number<kKPerBlock / kKPack>{}, ck_tile::number<kKPack>{}))),
            ck_tile::make_tuple(ck_tile::sequence<0, 2, 3>{}, ck_tile::sequence<1, 4>{}),
            ck_tile::make_tuple(ck_tile::sequence<0>{}, ck_tile::sequence<1>{}));

        return v_lds_block_desc;
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSizeSingleKV()
    {
        constexpr int32_t SingleKSize = MakeKLdsBlockDescriptor().get_element_space_size();
        constexpr int32_t SingleVSize =[&]() {
            constexpr int32_t Banks        = 32; /// TODO: need change based on arch
            constexpr int32_t PixelsPerRow = Banks * 4 / sizeof(scalar_t);
            constexpr int32_t kKPack       = 16 / sizeof(scalar_t);
            static_assert(PixelsPerRow % kKPack == 0);
            constexpr int32_t NPerRow    = PixelsPerRow / kKPack;
            constexpr int32_t kNPerBlock = Traits::kBlockN1;
            constexpr int32_t kKPerBlock = Traits::kBlockK1;
            static_assert(kNPerBlock % NPerRow == 0);
            static_assert(kKPerBlock % kKPack == 0);

            return (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);
        }();

        return ck_tile::max(SingleKSize, SingleVSize) * sizeof(scalar_t);
    }

    CK_TILE_HOST_DEVICE static constexpr int32_t GetSmemSize()
    {
        return Traits::kNumPrefetchKV * GetSmemSizeSingleKV();
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using BlockTile     = ck_tile::sequence<Traits::kBlockM, Traits::kBlockN0, Traits::kBlockK0>;
        using BlockWarps    = ck_tile::sequence<Traits::kNumWarps, 1, 1>;
        using TileGemmShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, typename Traits::QKWarpTile>;

        using GemmProblem = ck_tile::BlockGemmProblem<scalar_t, scalar_t, acc_t, Traits::kNumThreads, TileGemmShape>;

        constexpr int32_t kWarpGemmM = Traits::QKWarpTile::at(ck_tile::number<0>{});
        static_assert((kWarpGemmM == 4) || (kWarpGemmM == 16) || (kWarpGemmM == 32));

        constexpr auto warp_gemm = []()
        {
            if constexpr (std::is_same_v<scalar_t, ck_tile::fp16_t> && std::is_same_v<acc_t, float>)
            {
                if constexpr(kWarpGemmM == 32)
                    return ck_tile::WarpGemmMfmaF16F16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr(kWarpGemmM == 16)
                    return ck_tile::WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{};
                else // kWarpGemmM == 4
                    return ck_tile::WarpGemmMfmaF16F16F32M4N64K16{};
            }
            else if constexpr (std::is_same_v<scalar_t, ck_tile::bf16_t> && std::is_same_v<acc_t, float>)
            {
                if constexpr (kWarpGemmM == 32)
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M32N32K16SwizzleBTransposedCDistribution{};
                else if constexpr (kWarpGemmM == 16)
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution{};
                else // kWarpGemmM == 4
                    return ck_tile::WarpGemmMfmaBf16Bf16F32M4N64K16{};
            }
        }();

        using BlockGemmPolicy =
            ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<scalar_t, scalar_t, acc_t, BlockWarps, decltype(warp_gemm)>;

        return ck_tile::BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetKVWarpGemm()
    {
        if constexpr(std::is_same_v<scalar_t, ck_tile::fp8_t> && std::is_same_v<acc_t, float>)
        {
            return ck_tile::WarpGemmMfmaFp8Fp8F32M32N32K16SwizzleBTransposedCDistribution<>{};
        }
        else
        {
            return ck_tile::WarpGemmMfmaDispatcher<
                scalar_t, scalar_t, acc_t,
                Traits::KVWarpTile::at(ck_tile::number<0>{}),
                Traits::KVWarpTile::at(ck_tile::number<1>{}),
                Traits::KVWarpTile::at(ck_tile::number<2>{}),
                true>{};
        }
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetKVBlockGemm()
    {
        using BlockTile     = ck_tile::sequence<Traits::kBlockM, Traits::kBlockN1, Traits::kBlockK1>;
        using BlockWarps    = ck_tile::sequence<Traits::kNumWarps, 1, 1>;
        using TileGemmShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, typename Traits::KVWarpTile>;

        constexpr int32_t kNumWarps   = Traits::kNumWarps;
        constexpr int32_t kNumThreads = kNumWarps * ck_tile::get_warp_size();

        using GemmProblem = ck_tile::BlockGemmProblem<scalar_t, scalar_t, acc_t, kNumThreads, TileGemmShape>;

        auto warp_gemm = GetKVWarpGemm();

        using WarpGemm = ck_tile::remove_cvref_t<decltype(warp_gemm)>;

        using BlockGemmPolicy =
            ck_tile::BlockGemmARegBSmemCRegV2CustomPolicy<scalar_t, scalar_t, acc_t, BlockWarps, WarpGemm>;

        return ck_tile::BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = ck_tile::remove_cvref_t<decltype(GetQKBlockGemm())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Traits::kBlockM,
            Traits::kBlockK0>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kBlockK0;

        constexpr int32_t MaxVectorSize = 16 / sizeof(scalar_t);
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

    CK_TILE_HOST_DEVICE static constexpr auto GetNumRepeatOfKDramTileDistribution()
    {
        using KDstrEncode = typename decltype(MakeKDramTileDistribution())::DstrEncode;
        return KDstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
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
        constexpr int32_t kKPack = 16 / sizeof(scalar_t);
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
        constexpr int32_t kKPack = 16 / sizeof(scalar_t);
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

    int32_t size_b;         // batch count
    int32_t size_s;         // seqlen of q
    int32_t size_h;         // head count of q
    int32_t hq_hk_ratio;    // head count of q / head count of kv
    int32_t num_splits;
    int64_t block_table_batch_stride;
    int32_t num_page_blocks;
    int32_t page_block_size;
    float   scale_softmax;
    int32_t mask_y_ratio;

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
CK_TILE_DEVICE static auto MakeQDram(
    const scalar_t* p_data,
    const int32_t   height,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    const auto q_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(height, Traits::kSizeD),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentQ()>{},
        ck_tile::number<1>{});

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
CK_TILE_DEVICE static auto MakeLseAccDram(
    scalar_t* p_data,
    const Lengths&  window_lengths,
    const int32_t   size_s)
{
    using Traits = typename Policy::Traits;

    const auto lse_acc_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s),
        ck_tile::make_tuple(1),
        ck_tile::number<1>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        lse_acc_dram_naive,
        window_lengths,
        ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutAccDram(
    scalar_t* p_data,
    const int32_t   size_s,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    const auto o_acc_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentOacc()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        o_acc_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <typename Policy, typename Lengths, typename scalar_t>
CK_TILE_DEVICE static auto MakeLseDram(
    scalar_t* p_data,
    const Lengths&  window_lenghts,
    const int32_t   size_s)
{
    using Traits = typename Policy::Traits;

    const auto lse_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s),
        ck_tile::make_tuple(1),
        ck_tile::number<Policy::GetAlignmentLse()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        lse_dram_naive, window_lenghts, ck_tile::sequence<Traits::kPadSeqLenQ>{});
}

template <typename Policy, typename scalar_t>
CK_TILE_DEVICE static auto MakeOutDram(
    scalar_t* p_data,
    const int32_t   size_s,
    const int32_t   stride_s)
{
    using Traits = typename Policy::Traits;

    const auto o_dram_naive = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
        p_data,
        ck_tile::make_tuple(size_s, Traits::kSizeDV),
        ck_tile::make_tuple(stride_s, 1),
        ck_tile::number<Policy::GetAlignmentO()>{},
        ck_tile::number<1>{});

    return ck_tile::pad_tensor_view(
        o_dram_naive,
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{}),
        ck_tile::sequence<Traits::kPadSeqLenQ, Traits::kPadHeadDimV>{});
}

template <int32_t VirtualDim, typename scalar_t, typename Dram>
CK_TILE_DEVICE static auto MakePageBlockNavigator(
    const scalar_t* p_data,
    const Dram&     dram_complete,
    const Dram&     dram_last,
    const int32_t   bid,
    const int32_t   hid,
    const int32_t   seqlen_k,
    const int32_t   stride_b,
    const int32_t   stride_h,
    const int32_t*  p_block_table,
    const int32_t   stride_b_block_table,
    const int32_t   page_block_size)
{
    const auto* p_block_indices = p_block_table + int64_t(bid) * stride_b_block_table;
    const int32_t num_blocks = ck_tile::integer_divide_ceil(seqlen_k, page_block_size);

    const int64_t fixed_offset = static_cast<int64_t>(hid) * stride_h;

    return ck_tile::make_page_block_navigator<const scalar_t, VirtualDim>(
        p_data,
        stride_b, // vcache page-block stride/size
        fixed_offset,
        p_block_indices,
        num_blocks,
        page_block_size,
        dram_complete,
        dram_last);
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

    // Block GEMMs
    constexpr auto gemm_0 = Policy::GetQKBlockGemm();
    constexpr auto gemm_1 = Policy::GetKVBlockGemm();

    // Reduction funtions for softmax
    const auto f_max = [](auto e0, auto e1) { return ck_tile::max(e0, e1); };
    const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

    // sacc, S, P, M, L, Oacc
    using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
    auto s_acc              = SaccBlockTileType{};
    using SBlockTileType    = decltype(ck_tile::cast_tile<acc_t>(s_acc));
    using MLBlockTileType   = decltype(ck_tile::block_tile_reduce<acc_t>(
        SBlockTileType{}, ck_tile::sequence<1>{}, f_max, acc_t{0}));
    using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
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

    // Define main loop
    auto main_loop = [&]<bool IsEvenLoop>(int32_t loop_idx)
    {
        ck_tile::clear_tile(s_acc);

        // I. QK GEMM
        //
        constexpr ck_tile::array<int32_t, 2> qk_origin =
            {0, IsEvenLoop ? 0 : Traits::kBlockK0 * (k0_loops - 1)};
        constexpr ck_tile::array<int32_t, 2> qk_direction =
            {0, IsEvenLoop ? Traits::kBlockK0 : -Traits::kBlockK0};
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

        // prefetch load V tile
        auto v_prefetch = ck_tile::load_tile(v_dram_windows[0]);
        

        // II. scale_s, mask, softmax
        //

        // Masking
        // Note that masking is also required when k is padded
        const auto k_origin = k_dram_window_origin.get_window_origin();
        const bool need_perpixel_check = mask.IsEdgeTile(
            q_origin.at(ck_tile::number<0>{}),
            k_origin.at(ck_tile::number<0>{}),
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

        // Get max of row
        auto m_local = ck_tile::block_tile_reduce<acc_t>(
            s_acc, ck_tile::sequence<1>{}, f_max, -ck_tile::numeric<acc_t>::infinity());
        ck_tile::block_tile_reduce_sync(m_local, f_max, ck_tile::bool_constant<false>{});
        const auto m_old = m;
        ck_tile::tile_elementwise_inout(
            [](auto& e0, auto e1, auto e2) { e0 = ck_tile::max(e1, e2); }, m, m_old, m_local);

        // Compute exp(x_i - m)
        auto p_intermedia = ck_tile::make_static_distributed_tensor<acc_t>(s_acc.get_tile_distribution());
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
                        p_intermedia(ij) = ck_tile::exp(s_acc[ij] - row_max);
                    });
            });

        // Compute row sum of exp(x_i - m)
        auto rowsum_p = ck_tile::block_tile_reduce<acc_t>(p_intermedia, ck_tile::sequence<1>{}, f_sum, acc_t(0));
        ck_tile::block_tile_reduce_sync(rowsum_p, f_sum, ck_tile::bool_constant<false>{});

        // Calculate new l and adjust old output acc
        constexpr auto o_spans = ck_tile::remove_cvref_t<decltype(o_acc[0])>::get_distributed_spans();
        ck_tile::sweep_tile_span(
            o_spans[ck_tile::number<0>{}],
            [&](auto id0)
            {
                constexpr auto i = ck_tile::make_tuple(id0);
                const auto row_max = GetValidatedMax<Mask::IsMasking>(m[i]);
                const auto temp_i  = ck_tile::exp(m_old[i] - row_max);
                l(i) = temp_i * l[i] + rowsum_p[i];
                ck_tile::sweep_tile_span(
                    o_spans[ck_tile::number<1>{}],
                    [&](auto id1)
                    {
                        constexpr auto ij = ck_tile::make_tuple(id0, id1);
                        ck_tile::static_for<0, n1_loops, 1>{}([&](auto n1_id)
                        {
#if 1
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

            if constexpr (k1_loops > 1)
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
                // TODO: The following code is not necessary but it positively affects performance for unknown reason.
                // Remove the following code once it no longer affects.
                ck_tile::set_tile_if(
                    v_prefetch, ck_tile::numeric<scalar_t>::zero(),
                    [&](auto ids)
                    {
                        const auto col = ids.at(kVPageIdxDim);
                        return (loop_idx * Traits::kBlockN0 + col) >= seqlen_k;
                    });
            }

            gemm_1(o_acc[n1_id],
                   ck_tile::get_slice_tile(
                        p,
                        ck_tile::sequence<0, (k1_loops - 1) * Traits::kBlockK1>{},
                        ck_tile::sequence<Traits::kBlockM, Traits::kBlockN0>{}),
                   v_lds_window);
            ck_tile::block_sync_lds();
        });

        if ((loop_idx + 1) < num_total_loop)
        {
            // Move K to next block of column
            constexpr ck_tile::array<int32_t, 2> next_qk_origin =
                {0, IsEvenLoop ? Traits::kBlockK0 * (k0_loops - 1) : 0};
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


    // 7. Store LSE
    //
    auto lse_acc = ck_tile::make_static_distributed_tensor<acc_t>(m.get_tile_distribution());
    constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
    ck_tile::sweep_tile_span(lse_acc_spans[ck_tile::number<0>{}], [&, m_ = m, l_ = l](auto id0) {
        constexpr auto i = make_tuple(id0);
        lse_acc(i) = m_[i] + log(l_[i]);
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
#if 1
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
    const auto hkid = hqid / params.hq_hk_ratio;
    const int32_t mid = __builtin_amdgcn_readfirstlane(tile_m_id * Traits::kBlockM);

    // Define causal mask
    using Mask             = ck_tile::SimplifiedGenericAttentionMask<kIsCausal, Traits::kEnableXQA>;
    const int32_t seqlen_k = params.p_seqlens_k[bid];
    Mask mask              = kIsCausal ? Mask{params.size_s,
                                 seqlen_k - params.size_s / params.mask_y_ratio + 1,
                                 params.size_s,
                                 seqlen_k,
                                 params.mask_y_ratio}
                                       : Mask{params.size_s, seqlen_k};

    auto q_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockK0>{});
    auto k_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN0>{}, ck_tile::number<Traits::kBlockK0>{});
    auto v_dram_window_lengths =
        ck_tile::make_tuple(ck_tile::number<Traits::kBlockN1>{}, ck_tile::number<Traits::kBlockK1>{});

    const scalar_t* p_query = reinterpret_cast<const scalar_t*>(params.p_query) +
                              int64_t(hqid) * params.stride_h_q +   // head offset
                              int64_t(bid) * params.stride_b_q;     // batch offset
    const scalar_t* p_key   = reinterpret_cast<const scalar_t*>(params.p_key) +
                              int64_t(hkid) * params.stride_h_k;    // head offset
    const scalar_t* p_value = reinterpret_cast<const scalar_t*>(params.p_value) +
                              int64_t(hkid) * params.stride_h_v;    // head offset
    const int32_t*  p_block_table = params.p_block_table +
                                    int64_t(bid) * params.block_table_batch_stride; // batch offset

    const int32_t kv_cache_width = params.num_page_blocks * params.page_block_size;

    const auto q_dram = MakeQDram<Policy>(p_query, params.size_s,  params.stride_s_q);
    const auto k_dram = MakeKDram<Policy>(p_key,   kv_cache_width, params.stride_s_k);
    const auto v_dram = MakeVDram<Policy>(p_value, kv_cache_width, params.stride_s_v);    

    auto q_dram_window = ck_tile::make_tile_window(q_dram, q_dram_window_lengths, {mid, 0});
    auto k_dram_window = ck_tile::make_tile_window(k_dram, k_dram_window_lengths, {0, 0});
    auto v_dram_window = ck_tile::make_tile_window(v_dram, v_dram_window_lengths, {0, 0});

    if constexpr (kDoSplit)
    {
        acc_t* p_lse_acc = reinterpret_cast<acc_t*>(params.p_softmax_lseaccum) +
                           int64_t(hqid) * params.stride_h_lseacc +     // head offset
                           int64_t(bid) * params.stride_b_lseacc +      // batch offset
                           int64_t(split_id) * params.stride_sp_lseacc; // split offset
        out_t* p_out_acc = reinterpret_cast<out_t*>(params.p_output_accum) +
                           int64_t(hqid) * params.stride_h_oacc +      // head offset
                           int64_t(bid) * params.stride_b_oacc +       // batch offset
                           int64_t(split_id) * params.stride_sp_oacc;  // split offset

        auto lse_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_acc_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});

        const auto lse_acc_dram = MakeLseAccDram<Policy>(p_lse_acc, lse_acc_dram_window_lengths, params.size_s);
        const auto out_acc_dram = MakeOutAccDram<Policy>(p_out_acc, params.size_s, params.stride_s_oacc);

        auto lse_acc_dram_window =
            ck_tile::make_tile_window(lse_acc_dram, lse_acc_dram_window_lengths, {mid});
        auto out_acc_dram_window =
            ck_tile::make_tile_window(out_acc_dram, out_acc_dram_window_lengths, {mid, 0});

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
            params.scale_softmax,
            p_smem);
    }
    else
    {
        // Assuming lse is in shape [b, h, s] and is contiguous
        acc_t* p_lse = reinterpret_cast<acc_t*>(params.p_softmax_lse) +
                       (int64_t(bid) * params.size_h + hqid) * params.size_s; // batch+head offset
        out_t* p_out = reinterpret_cast<out_t*>(params.p_output) +
                       int64_t(hqid) * params.stride_h_o +   // head offset
                       int64_t(bid) * params.stride_b_o;     // batch offset

        auto lse_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{});
        auto out_dram_window_lengths =
            ck_tile::make_tuple(ck_tile::number<Traits::kBlockM>{}, ck_tile::number<Traits::kBlockN1>{});
        
        const auto lse_dram = MakeLseDram<Policy>(p_lse, lse_dram_window_lengths, params.size_s);
        const auto out_dram = MakeOutDram<Policy>(p_out, params.size_s, params.stride_s_o);

        auto lse_dram_window =
            ck_tile::make_tile_window(lse_dram, lse_dram_window_lengths, {mid});
        auto out_dram_window =
            ck_tile::make_tile_window(out_dram, out_dram_window_lengths, {mid, 0});

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
            params.scale_softmax,
            p_smem);
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
    const int32_t hsidx            = hidx * params.size_s + sidx;
    const int32_t shidx            = hidx + sidx * params.size_h;
    const int32_t size_hs          = params.size_h * params.size_s;
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

    const int32_t num_blk   = ck_tile::integer_divide_ceil(params.size_s,  Traits::kBlockM) * params.num_splits;
    const dim3    grid_attn = dim3(num_blk, params.size_h, params.size_b);
    const dim3    grid_comb = dim3(params.size_s, params.size_h, params.size_b);

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
    constexpr bool ENABLE_PACK_QK_RATIO = true;
    //                                        dqk  dv   m0  n0  n1  #warp
    using Traits = FlashMlaPrefillKernelTrait<576, 512, 64, 64, 256, 4, ENABLE_PACK_QK_RATIO>;
    constexpr bool kForceOutAcc = false;
    using acc_t                 = float;

    torch::Tensor vcache = value_cache.data_ptr() ? value_cache : key_cache;

    auto opts = query.options();
    static_assert(std::is_same_v<acc_t, float>);
    auto opts_acc = opts.dtype(torch::kFloat32);

    const int32_t batch_size = query.size(0);
    const int32_t seqlen_q_ori = query.size(1);
    const int32_t num_heads_q_ori = query.size(2);
    int32_t seqlen_q = seqlen_q_ori;
    int32_t num_heads_q = num_heads_q_ori;

    const int32_t head_size = query.size(3);
    TORCH_CHECK((head_size == 576) && (head_size_v == 512), "Only support QK head dim 576 and V head dim 512!");

    const int32_t num_blocks = key_cache.size(0);
    const int32_t page_block_size = key_cache.size(1);
    const int32_t num_heads_k = key_cache.size(2);

    TORCH_CHECK(num_heads_q % num_heads_k == 0,
                "Number of heads in key/value must divide number of heads in query");

    const int32_t hq_hk_ratio = num_heads_q / num_heads_k;
    int32_t mask_y_ratio      = 1;

    if constexpr(Traits::kEnableXQA)
    {
        seqlen_q     = seqlen_q_ori * hq_hk_ratio;
        num_heads_q  = num_heads_k;
        mask_y_ratio = hq_hk_ratio;
        if(num_heads_k == 1)
        {
            query = query.reshape({batch_size, seqlen_q, num_heads_q, head_size});
        }
        else
        {
            query = query.view({batch_size, seqlen_q_ori, num_heads_q, hq_hk_ratio, head_size})
                        .transpose(2, 3)
                        .reshape({batch_size, seqlen_q, num_heads_q, head_size});
        }
    }

    const int32_t num_splits = calculate_num_splits<Traits>(batch_size, num_heads_q, seqlen_q);
    const bool    do_splits = num_splits > 1;

    // Combine shader, which only exists when num_splits > 1, will conduct type convert by default and force.
    // Thus, kForceOutAcc doesn't work in this case.
    auto output = torch::empty({batch_size, seqlen_q, num_heads_q, head_size_v},
                               (kForceOutAcc && !do_splits) ? opts_acc : opts);
    auto softmax_lse = torch::empty({batch_size, num_heads_q, seqlen_q}, opts_acc);

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
    params.size_s                   = seqlen_q;
    params.size_h                   = num_heads_q;
    params.hq_hk_ratio              = num_heads_q / num_heads_k;
    params.block_table_batch_stride = block_table.stride(0);
    params.num_page_blocks          = num_blocks;
    params.page_block_size          = page_block_size;
    params.scale_softmax            = softmax_scale;

    params.mask_y_ratio = mask_y_ratio;

    params.stride_b_q = query.stride(0);
    params.stride_s_q = query.stride(1);
    params.stride_h_q = query.stride(2);
    params.stride_b_k = key_cache.stride(0);
    params.stride_s_k = key_cache.stride(1); // size_hk * size_d
    params.stride_h_k = key_cache.stride(2);
    params.stride_b_v = vcache.stride(0);
    params.stride_s_v = vcache.stride(1);    // size_hk * size_d
    params.stride_h_v = vcache.stride(2);
    params.stride_b_o = output.stride(0);
    params.stride_s_o = output.stride(1);
    params.stride_h_o = output.stride(2);

    if(num_splits > 1)
    {
        auto output_accum =
            torch::empty({batch_size, num_splits, seqlen_q, num_heads_q, head_size_v}, opts_acc);
        auto softmax_lseaccum =
            torch::empty({batch_size, num_splits, num_heads_q, seqlen_q}, opts_acc);

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
    // assert(is_causal == true);
    // assert(query.scalar_type() == at::ScalarType::BFloat16);
    // using scalar_t = ck_tile::bf16_t;
    // using out_t = std::conditional_t<kForceOutAcc, acc_t, scalar_t>;
    // dispatch_fmla_fwd_splictkv_prefill<Traits, scalar_t, acc_t, out_t, true>(params);

    if constexpr(Traits::kEnableXQA)
    {
        // post process for out and softmax_lse
        if(num_heads_k == 1)
        {
            output = output.reshape({batch_size, seqlen_q_ori, num_heads_q_ori, head_size_v});
        }
        else
        {
            output = output.view({batch_size, seqlen_q_ori, hq_hk_ratio, num_heads_q, head_size_v})
                         .transpose(2, 3)
                         .reshape({batch_size, seqlen_q_ori, num_heads_q_ori, head_size_v});
        }
        softmax_lse = softmax_lse.view({batch_size, num_heads_q, seqlen_q_ori, hq_hk_ratio})
                          .transpose(2, 3)
                          .reshape({batch_size, num_heads_q_ori, seqlen_q_ori});
    }

    return {output.to(opts), softmax_lse};
}
