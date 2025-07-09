// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/reduce.hpp>
#include "fwd_kernels_traits.hpp"
#include "block_gemm_areg_bsmem_creg.hpp"

// =====================================================================================================================
// Utils
//
// #define FMLA_FWD_FAST_EXP2 1

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
                                    Traits::kKIterations,
                                    Traits::kBlockM,
                                    AccType>();
    }

    template<int32_t KPerBlock = Traits::kBlockK0>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        constexpr int32_t BlockTileK = (Traits::kKVLoadOnce == false) ? Traits::kBlockK0 : KPerBlock;
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
        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);
        return (Traits::kKIterations + kKPack) * sizeof(scalar_t);
    }

    template<int32_t KPerBlock, int32_t RepeatK>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleKElementSpaceSize()
    {
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = KPerBlock;
        constexpr int32_t kRepeatK   = RepeatK;
        constexpr int32_t NumWarps   = Traits::kNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        // 16 means max load elemtments number in an instrution: b128->16bytes
        // 4  means max load elemtments number to lds in gfx942: b32 ->4bytes
        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);
        constexpr int32_t KVector = 4  / sizeof(scalar_t);
        constexpr int32_t kPad    = kKPack;

        static_assert((warpSize * KVector >= kKPerBlock) &&
                      (warpSize * KVector % kKPerBlock == 0));
        constexpr int32_t LanesPerK  = kKPerBlock / KVector;
        constexpr int32_t LaneGroups = warpSize / LanesPerK;
        constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);

        return NumIssues * NumWarps * kRepeatK * (warpSize * KVector + kPad);
    }

    template<int32_t KPerBlock, int32_t RepeatK>
    CK_TILE_HOST_DEVICE static constexpr auto GetSingleKSpaceSize()
    {
        return GetSingleKElementSpaceSize<KPerBlock, RepeatK>() * sizeof(scalar_t);
    }

    template <int32_t KPerBlock,
              int32_t RepeatK,
              int32_t IBuf = 0>
    CK_TILE_HOST_DEVICE static constexpr auto
        MakeKLdsStoreBlockDescriptor(ck_tile::number<IBuf> = ck_tile::number<0>{})
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = KPerBlock;
        constexpr int32_t kRepeatK   = RepeatK;
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t NumWarps   = Traits::kNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);
        constexpr int32_t KVector = 4  / sizeof(scalar_t);
        constexpr int32_t kPad = 
            kKPack; // for async-copy, this pad is between warps. Optimize this for lds_read speed

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr int32_t LanesPerK =
            kKPerBlock / KVector; // 
        constexpr int32_t LaneGroups =
            warpSize /
            LanesPerK; // how many groups (within a wave), they may load different N, but same K
        constexpr int32_t NumIssues = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr int32_t BufferSize = GetSingleKElementSpaceSize<KPerBlock, kRepeatK>();

        if constexpr (KPerBlock == Traits::kSizeRope)
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
                ck_tile::number<IBuf * BufferSize>{},
                ck_tile::number<KVector>{},
                ck_tile::number<1>{});

            constexpr auto k_lds_block_desc_issues_warps_lanes = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(ck_tile::make_pass_through_transform(ck_tile::number<NumIssues>{}),
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
                           ck_tile::number<kRepeatK>{},   // krepeatK
                           ck_tile::number<LanesPerK>{},  // k0
                           ck_tile::number<KVector>{}),   // k1
                ck_tile::make_tuple(ck_tile::number<NumWarps* kRepeatK * (warpSize * KVector + kPad)>{},
                           ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                           ck_tile::number<(warpSize * KVector + kPad) * kRepeatK>{},
                           ck_tile::number<(warpSize * KVector + kPad) / LaneGroups>{},
                           ck_tile::number<KVector>{},
                           ck_tile::number<1>{}),
                ck_tile::number<IBuf * BufferSize>{},
                ck_tile::number<KVector>{},
                ck_tile::number<1>{});

            // TODO this layout is hard coded, and will be used in async copy buffer view load
            // in LDS the real layout is (bufs, N0, N2, N1*K0*K1)
            constexpr auto k_lds_block_desc_issues_warps_lanes = ck_tile::transform_tensor_descriptor(
                k_lds_block_desc_0,
                ck_tile::make_tuple(ck_tile::make_pass_through_transform(ck_tile::number<NumIssues>{}),
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

    template<int32_t KPerBlock, int32_t RepeatK>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsLoadBlockDescriptor()
    {
        // K is always k-major, we use async-copy to load into LDS
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = KPerBlock;
        constexpr int32_t kRepeatK   = RepeatK;
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t NumWarps   = Traits::kNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);
        constexpr int32_t KVector = 4 / sizeof(scalar_t);
        constexpr int32_t kPad    = kKPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr int32_t LanesPerK  = kKPerBlock / KVector; // within a wave
        constexpr int32_t LaneGroups = warpSize / LanesPerK; // within a wave
        constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
        static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

        constexpr int32_t BufferSize = GetSingleKElementSpaceSize<KPerBlock, kRepeatK>();

        if constexpr (KPerBlock == Traits::kSizeRope)
        {
            constexpr auto k_lds_block_desc_0 =
                ck_tile::make_naive_tensor_descriptor(ck_tile::make_tuple(ck_tile::number<Traits::kNumPrefetchK>{},    // num_buffers
                                                                          ck_tile::number<NumIssues>{},           // n0
                                                                          ck_tile::number<NumWarps>{},            // n2
                                                                          ck_tile::number<LaneGroups>{},          // n1
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
                                                                          ck_tile::number<(warpSize * KVector + kPad) * kRepeatK / LaneGroups>{},
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
        constexpr int32_t kNPerBlock = Traits::kSizeDV;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;

        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);

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
        constexpr int32_t kNPerBlock = Traits::kSizeDV;
        constexpr int32_t kKPerBlock = Traits::kBlockK1;
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t NumWarps   = Traits::kNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        constexpr int32_t kKPack  = 16 / sizeof(scalar_t);
        constexpr int32_t KVector = 8 / sizeof(scalar_t); // 4
        constexpr int32_t kPad    = kKPack; // for async-copy, this pad is between warps

        static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
        constexpr int32_t LanesPerK  = kKPerBlock / KVector; // 16 / 4 = 4
        constexpr int32_t LaneGroups = warpSize / LanesPerK; // 64 / 4 = 16
        constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps); // 1
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
            (GetSingleKElementSpaceSize<Traits::kKIterations, Traits::kKNopeLdsRepeat>() +
             GetSingleKElementSpaceSize<Traits::kSizeRope, 1>()) * sizeof(scalar_t) :
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


    template<int32_t KPerBlock = Traits::kBlockK0>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = ck_tile::remove_cvref_t<decltype(GetQKBlockGemm<KPerBlock>())>;

        constexpr int32_t BlockTileK = (Traits::kKVLoadOnce == false) ? Traits::kBlockK0 : KPerBlock;
        return BlockGemm::template MakeABlockTileDistribution<
            Traits::kBlockM,
            BlockTileK>();
    }

    template<int32_t KPerBlock = Traits::kBlockK0>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        if constexpr (Traits::kKVLoadOnce == false)
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
            constexpr int32_t kBlockSize = Traits::kNumThreads;
            constexpr int32_t kNPerBlock = Traits::kBlockN0;
            constexpr int32_t kKPerBlock = KPerBlock;
            constexpr int32_t NumWarps   = Traits::kNumWarps;
            constexpr int32_t warpSize   = ck_tile::get_warp_size();

            constexpr int32_t KVector = 4 / sizeof(scalar_t);

            static_assert(warpSize * KVector >= kKPerBlock && warpSize * KVector % kKPerBlock == 0);
            constexpr int32_t LanesPerK  = kKPerBlock / KVector; // within a wave
            constexpr int32_t LaneGroups = warpSize / LanesPerK; // within a wave
            constexpr int32_t NumIssues  = kNPerBlock / (LaneGroups * NumWarps);
            static_assert(NumIssues == kNPerBlock * kKPerBlock / (kBlockSize * KVector));

            constexpr int32_t N0 = NumIssues;
            constexpr int32_t N1 = LaneGroups;
            constexpr int32_t N2 = NumWarps;
            constexpr int32_t K0 = LanesPerK;
            constexpr int32_t K1 = KVector;

            return ck_tile::make_static_tile_distribution(
                ck_tile::tile_distribution_encoding<ck_tile::sequence<1>,
                                           ck_tile::tuple<ck_tile::sequence<N0, N1, N2>, ck_tile::sequence<K0, K1>>,
                                           ck_tile::tuple<ck_tile::sequence<1>, ck_tile::sequence<1, 2>>,
                                           ck_tile::tuple<ck_tile::sequence<2>, ck_tile::sequence<1, 0>>,
                                           ck_tile::sequence<1, 2>,
                                           ck_tile::sequence<0, 1>>{});
        }
    }

    template<int32_t KPerBlock = Traits::kBlockK0>
    CK_TILE_HOST_DEVICE static constexpr auto GetNumRepeatOfKDramTileDistribution()
    {
        using KDstrEncode = typename decltype(MakeKDramTileDistribution<KPerBlock>())::DstrEncode;
        return KDstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<0>{}];
    }

    CK_TILE_HOST_DEVICE static constexpr auto GetVTileDistributionStride()
    {
        using VDstrEncode = typename decltype(MakeVTileDistribution())::DstrEncode;

        constexpr int32_t VKVectors = VDstrEncode::hs_lengthss_[ck_tile::number<0>{}][ck_tile::number<3>{}];
        constexpr int32_t VKRepeats = VKVectors / 2;
        return VKRepeats;
    }


    CK_TILE_HOST_DEVICE static constexpr auto MakeVTileDistribution()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kKIterations;
        constexpr int32_t kKNumWarps = Traits::kNumWarps;
        constexpr int32_t kNNumWarps = Traits::kNumWarps / kKNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        constexpr int32_t kKVector = 4 / sizeof(scalar_t);
        constexpr int32_t kNVector = 4 * Traits::kWaveOccupancy / sizeof(scalar_t);

        constexpr int32_t kKLanes      = kKPerBlock / (kKNumWarps * kKVector);
        constexpr int32_t kLaneGroups  = warpSize / kKLanes;

        constexpr int32_t kNRepeart = kNPerBlock / (kLaneGroups * kNVector);
        constexpr int32_t kKRepeart = kKPerBlock / (kKNumWarps * kKLanes * kKVector);

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<
                    ck_tile::sequence<kNRepeart, kNNumWarps, kLaneGroups, kNVector>,
                    ck_tile::sequence<kKRepeart, kKNumWarps, kKLanes, kKVector>>,
                ck_tile::tuple<ck_tile::sequence<1, 2>, ck_tile::sequence<1, 2>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
                ck_tile::sequence<1, 1, 2, 2>,
                ck_tile::sequence<0, 3, 0, 3>>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeVTTileDistribution()
    {
        constexpr int32_t kBlockSize = Traits::kNumThreads;
        constexpr int32_t kNPerBlock = Traits::kBlockN0;
        constexpr int32_t kKPerBlock = Traits::kKIterations;
        constexpr int32_t kKNumWarps = Traits::kNumWarps;
        constexpr int32_t kNNumWarps = Traits::kNumWarps / kKNumWarps;
        constexpr int32_t warpSize   = ck_tile::get_warp_size();

        constexpr int32_t kKVector = 4 / sizeof(scalar_t);
        constexpr int32_t kNVector = 4 * Traits::kWaveOccupancy / sizeof(scalar_t);

        constexpr int32_t kKLanes      = kKPerBlock / (kKNumWarps * kKVector);
        constexpr int32_t kLaneGroups  = warpSize / kKLanes;

        constexpr int32_t kNRepeart = kNPerBlock / (kLaneGroups * kNVector);
        constexpr int32_t kKRepeart = kKPerBlock / (kKNumWarps * kKLanes * kKVector);

        return ck_tile::make_static_tile_distribution(
            ck_tile::tile_distribution_encoding<
                ck_tile::sequence<>,
                ck_tile::tuple<
                    ck_tile::sequence<kKRepeart, kKNumWarps, kKLanes, kKVector>,
                    ck_tile::sequence<kNRepeart, kNNumWarps, kLaneGroups, kNVector>>,
                ck_tile::tuple<ck_tile::sequence<2, 1>, ck_tile::sequence<2, 1>>,
                ck_tile::tuple<ck_tile::sequence<1, 1>, ck_tile::sequence<2, 2>>,
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
