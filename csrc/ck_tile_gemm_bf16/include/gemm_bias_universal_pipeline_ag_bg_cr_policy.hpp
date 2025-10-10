// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
namespace ck_tile {

template <typename Derived>
struct UniversalGemmBasePolicy_bias
{
#if defined(__gfx950__)
    template <typename Problem>
    static constexpr bool is_a_load_tr =
        std::is_same_v<remove_cvref_t<typename Problem::ALayout>, tensor_layout::gemm::ColumnMajor>;
    template <typename Problem>
    static constexpr bool is_b_load_tr =
        std::is_same_v<remove_cvref_t<typename Problem::BLayout>, tensor_layout::gemm::RowMajor>;
#else
    template <typename Problem>
    static constexpr bool is_a_load_tr = false;
    template <typename Problem>
    static constexpr bool is_b_load_tr = false;
#endif

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr auto ATileAccessPattern = tile_distribution_pattern::thread_raked;
    static constexpr auto BTileAccessPattern = tile_distribution_pattern::thread_raked;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {

        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(is_a_load_tr<Problem>)
        {
            // TODO: better lds descriptor for performance
            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<MPerBlock>{}),
                make_tuple(number<MPerBlock>{}, number<1>{}),
                number<MPerBlock>{},
                number<1>{});
            return a_lds_block_desc_0;
        }
        else
        {
            constexpr index_t KPack = GetSmemPackA<Problem>();

            constexpr auto DataTypeSize = sizeof(ADataType);
            constexpr auto MLdsLayer =
                (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

            constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<KPerBlock / KPack * MLdsLayer>{},
                           number<MPerBlock / MLdsLayer>{},
                           number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * MLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<MPerBlock / MLdsLayer>{},
                                                         number<KPerBlock / KPack * MLdsLayer>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto a_lds_block_desc_xk0_mnldslayer_mn_xk1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(
                               make_tuple(number<MLdsLayer>{}, number<KPerBlock / KPack>{})),
                           make_pass_through_transform(number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto a_lds_block_desc = transform_tensor_descriptor(
                a_lds_block_desc_xk0_mnldslayer_mn_xk1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<MPerBlock / MLdsLayer>{}, number<MLdsLayer>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<KPerBlock / KPack>{}, number<KPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return a_lds_block_desc;
        }
    }

    /**
     * @brief Create LDS block descriptor for B tensor.
     *
     * @tparam Problem  Gemm pipeline problem.
     * @return B tensor LDS block descriptor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

#if 1
        if constexpr(is_b_load_tr<Problem>)
        {
            // TODO: better lds descriptor for performance
            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor( //
                make_tuple(number<KPerBlock>{}, number<NPerBlock>{}),
                make_tuple(number<NPerBlock>{}, number<1>{}),
                number<NPerBlock>{},
                number<1>{});
            return b_lds_block_desc_0;
        }
        else
        // else if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
        {
            constexpr index_t KPack     = GetSmemPackB<Problem>();
            constexpr auto BK0          = number<KPerBlock / KPack>{};
            constexpr auto DataTypeSize = sizeof(BDataType);
            constexpr auto NLdsLayer =
                (32 * 4 / KPerBlock / DataTypeSize) < 1 ? 1 : (32 * 4 / KPerBlock / DataTypeSize);

            constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(
                    BK0 * number<NLdsLayer>{}, number<NPerBlock / NLdsLayer>{}, number<KPack>{}),
                make_tuple(number<KPack>{}, number<KPerBlock * NLdsLayer>{}, number<1>{}),
                number<KPack>{},
                number<1>{});

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc_0,
                make_tuple(make_xor_transform(make_tuple(number<NPerBlock / NLdsLayer>{},
                                                         BK0 * number<NLdsLayer>{})),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<1, 0>{}, sequence<2>{}),
                make_tuple(sequence<1, 0>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(number<NLdsLayer>{}, BK0)),
                           make_pass_through_transform(number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(number<KPack>{})),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

            constexpr auto b_lds_block_desc = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<NPerBlock / NLdsLayer>{}, number<NLdsLayer>{})),
                           make_merge_transform_v3_division_mod(make_tuple(BK0, number<KPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return b_lds_block_desc;
        }
#else
        else // B is Row Major
        {
            constexpr index_t BlockSize   = Problem::kBlockSize;
            constexpr index_t VecLoadSize = GetVectorSizeB<Problem>();
            using TileEncodingPattern     = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          KPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern>;

            constexpr auto BK0 = number<TileEncodingPattern::X1>{};
            constexpr auto BK1 = number<TileEncodingPattern::Y0>{};
            // constexpr auto N0 = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N0 = TileEncodingPattern::X0;
            constexpr auto N1 = NPerBlock / N0;

            using WarpTile         = typename Problem::BlockGemmShape::WarpTile;
            constexpr auto NPerXdl = number<WarpTile::at(I1)>{};

            // constexpr auto KThreadWrite     =
            // BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto KThreadWrite     = TileEncodingPattern::Y2;
            constexpr auto K0PerThreadWrite = BK0 / KThreadWrite;
            constexpr auto KThreadRead      = 64 / NPerXdl;
            constexpr auto K0PerThreadRead  = BK0 / KThreadRead;

            constexpr auto kfold =
                (BK1 * N0 * sizeof(BDataType) > 128) ? 1 : 128 / (BK1 * N0 * sizeof(BDataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair = (BK1 * NPerXdl * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (BK1 * NPerXdl * sizeof(BDataType))) > N0
                                              ? N0
                                              : 128 / (BK1 * NPerXdl * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(number<KThreadWrite / kfold / KThreadReadPerm>{},
                           number<K0PerThreadWrite>{},
                           number<KThreadReadPerm * N1>{},
                           number<kfold * N0 / npair>{},
                           number<npair>{},
                           BK1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_xor_transform(
                        make_tuple(number<KThreadReadPerm * N1>{}, number<kfold * N0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(BK1)),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(
                    sequence<0>{}, sequence<1>{}, sequence<2, 3>{}, sequence<4>{}, sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(number<KThreadReadPerm>{}, number<N1>{})),
                    make_unmerge_transform(make_tuple(number<kfold>{}, number<N0 / npair>{})),
                    make_pass_through_transform(number<npair>{}),
                    make_pass_through_transform(BK1)),
                make_tuple(sequence<0>{},
                           sequence<1>{},
                           sequence<2>{},
                           sequence<3>{},
                           sequence<4>{},
                           sequence<5>{}),
                make_tuple(sequence<1>{},
                           sequence<2>{},
                           sequence<0, 3>{},
                           sequence<4, 5>{},
                           sequence<6>{},
                           sequence<7>{}));

            // constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
            //     b_lds_block_desc_unmerged,
            //     make_tuple(make_merge_transform_v3_division_mod(
            //                    make_tuple(number<KThreadReadPerm>{},
            //                               number<KThreadWrite / kfold / KThreadReadPerm>{},
            //                               number<kfold>{},
            //                               number<K0PerThreadWrite>{})),
            //                make_merge_transform_v3_division_mod(
            //                    make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{})),
            //                make_pass_through_transform(BK1)),
            //     make_tuple(sequence<0, 1, 4, 2>{}, sequence<5, 6, 3>{}, sequence<7>{}),
            //     make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            constexpr auto b_lds_block_desc_kn = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(number<KThreadReadPerm>{},
                                          number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          number<kfold>{},
                                          number<K0PerThreadWrite>{},
                                          BK1)),
                           make_merge_transform_v3_division_mod(
                               make_tuple(number<N0 / npair>{}, number<npair>{}, number<N1>{}))),
                make_tuple(sequence<0, 1, 4, 2, 7>{}, sequence<5, 6, 3>{}),
                make_tuple(sequence<1>{}, sequence<0>{}));

            // return b_lds_block_desc_bk0_n_bk1;
            return b_lds_block_desc_kn;

            // constexpr auto b_lds_block_desc_bk0_n_bk1 = make_naive_tensor_descriptor(
            //     make_tuple(BK0, number<NPerBlock>{}, number<KPack>{}),
            //     make_tuple(number<KPack>{}, number<KPerBlock>{}, number<1>{}),
            //     number<KPack>{},
            //     number<1>{});

            // constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            //     b_lds_block_desc_bk0_n_bk1,
            //     make_tuple(make_pass_through_transform(number<NPerBlock>{}),
            //                make_merge_transform_v3_division_mod(make_tuple(BK0,
            //                number<KPack>{}))),
            //     make_tuple(sequence<1>{}, sequence<0, 2>{}),
            //     make_tuple(sequence<0>{}, sequence<1>{}));

            // return b_lds_block_desc;
        }
#endif
    }

    /**
     * @brief Get the maximum global memory vector load size.
     *
     * @tparam Problem      The UniversalGemmPipelineProblem object.
     * @tparam DataType     The tensor data type we're considering.
     * @tparam MNPerBlock   The MPerBlock or NPerBlock value depending on tensor (A/B).
     * @tparam XPerTile     The contiguous Tile dimension size.
     * @return Maximum DRAM vector load size.
     */
    template <typename Problem, typename DataType, index_t MNPerBlock, index_t XPerTile>
    CK_TILE_HOST_DEVICE static constexpr auto GetGlobalVectorLoadSize()
    {
        constexpr index_t BlockSize           = Problem::kBlockSize;
        constexpr index_t KPerBlock           = Problem::BlockGemmShape::kK;
        constexpr index_t elements_per_thread = MNPerBlock * KPerBlock / BlockSize;
        constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;

        // Assume DataType is even!
        if constexpr(XPerTile % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                     elements_per_thread % (PackedSize * 32 / sizeof(DataType)) == 0 &&
                     PackedSize == 2)
        {
            return (PackedSize * 32 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 16 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 16 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 16 / sizeof(DataType));
        }
        else if constexpr(XPerTile % (PackedSize * 8 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 8 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 8 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 4 &&
                          XPerTile % (PackedSize * 4 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 4 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 4 / sizeof(DataType));
        }
        else if constexpr(sizeof(DataType) >= PackedSize * 2 &&
                          XPerTile % (PackedSize * 2 / sizeof(DataType)) == 0 &&
                          elements_per_thread % (PackedSize * 2 / sizeof(DataType)) == 0)
        {
            return (PackedSize * 2 / sizeof(DataType));
        }
        else
        {
            return PackedSize;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeA()
    {
        using ALayout               = remove_cvref_t<typename Problem::ALayout>;
        using ADataType             = remove_cvref_t<typename Problem::ADataType>;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, KPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, ADataType, MPerBlock, MPerBlock>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeB()
    {
        using BLayout               = remove_cvref_t<typename Problem::BLayout>;
        using BDataType             = remove_cvref_t<typename Problem::BDataType>;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, NPerBlock>();
        }
        else
        {
            return GetGlobalVectorLoadSize<Problem, BDataType, NPerBlock, KPerBlock>();
        }
    }

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @tparam Problem - Gemm pipeline problem class.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeC()
    {
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;
        using WG        = typename BlockGemm::WarpGemm;

        constexpr bool TransposeC = Problem::TransposeC;
        using CLayout             = typename Problem::CLayout;
        using CWarpDstr           = typename WG::CWarpDstr;

        // N is contiguous dimension
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has multiple consecutive elements in
                // N dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
            else
            {
                // In this case each thread has just a single item in Ndim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
        }
        // M is contiguous dimension
        else if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::ColumnMajor>)
        {
            if constexpr(TransposeC)
            {
                // In this case each thread has just a single item in Mdim
                return WG::WarpGemmAttribute::Impl::kCNLane / WG::kN;
            }
            else
            {
                // In this case each thread has multiple consecutive elements in
                // M dimension, however consecutive threads' elements have stride.
                constexpr index_t NDimY = CWarpDstr::NDimY;
                constexpr auto c_warp_y_lengths =
                    CWarpDstr{}.get_ys_to_d_descriptor().get_lengths();
                static_assert(WG::WarpGemmAttribute::Impl::kCM1PerLane ==
                              c_warp_y_lengths.get(number<NDimY - 1>{}));
                return c_warp_y_lengths.get(number<NDimY - 1>{});
            }
        }
        else
        {
            static_assert(false, "Unsupported CLayout!");
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Problem::TransposeC;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeA : GetVectorSizeA<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        // Tile: MPerBlock X KPerBlock
        if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          MPerBlock,
                                                                          KPerBlock,
                                                                          VecLoadSize,
                                                                          ATileAccessPattern,
                                                                          NumWaveGroups>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
        // Tile: KPerBlock X MPerBlock
        else
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          KPerBlock,
                                                                          MPerBlock,
                                                                          VecLoadSize,
                                                                          ATileAccessPattern,
                                                                          NumWaveGroups>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        // Tile: KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          KPerBlock,
                                                                          NPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern,
                                                                          NumWaveGroups>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
        // Tile: NPerBlock X KPerBlock
        else
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                          NPerBlock,
                                                                          KPerBlock,
                                                                          VecLoadSize,
                                                                          BTileAccessPattern,
                                                                          NumWaveGroups>;
            return TileEncodingPattern::Make2DStaticTileDistribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledARegTileDistribution()
    {
        using ALayout = remove_cvref_t<typename Problem::ALayout>;
        static_assert(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        constexpr index_t BlockSize     = Problem::kBlockSize;
        constexpr index_t MPerBlock     = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock     = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize   = GetVectorSizeA<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                      KPerBlock,
                                                                      MPerBlock,
                                                                      VecLoadSize,
                                                                      ATileAccessPattern,
                                                                      NumWaveGroups>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeShuffledBRegTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;
        static_assert(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>);
        constexpr index_t BlockSize     = Problem::kBlockSize;
        constexpr index_t NPerBlock     = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock     = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize   = GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
                                                                      KPerBlock,
                                                                      NPerBlock,
                                                                      VecLoadSize,
                                                                      BTileAccessPattern,
                                                                      NumWaveGroups>;
        return TileEncodingPattern::MakeShuffled2DStaticTileDistribution();
    }

     template <typename Problem>
   __host__ __device__ static constexpr auto MakeBiasDramTileDistribution()
   {
   
    using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;
    
    using WG        = typename BlockGemm::WarpGemm;
    constexpr index_t MWarp =  Problem::BlockGemmShape::w_kM;
    constexpr index_t NWarp =  Problem::BlockGemmShape::w_kN;
    // constexpr index_t kMPerBlock =  Problem::BlockGemmShape::kM;
    constexpr index_t kNPerBlock =  Problem::BlockGemmShape::kN;
    //  constexpr index_t kBlockSize = Problem::kBlockSize;
        // Gemm0 should be transposed
       
        
        constexpr index_t M1 = WG::WarpGemmAttribute::Impl::kCMLane;
        constexpr index_t M0 = MWarp;
        // constexpr index_t M2 = kMPerBlock/(WG::WarpGemmAttribute::Impl::kM*M0); 
        constexpr index_t N4 = WG::WarpGemmAttribute::Impl::kCM0PerLane;
        constexpr index_t N3 = WG::WarpGemmAttribute::Impl::kCNLane;
        // constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kCM1PerLane;
        constexpr index_t N1 = NWarp;
        constexpr index_t N0 = kNPerBlock / (N1 * WG::WarpGemmAttribute::Impl::kN);
    //    //if(get_thread_global_1d_id() ==64)
    //    //printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,\n",MWarp,NWarp,kNPerBlock,M1,M0,N4,N3,N2,N1,N0);
    //     return make_static_tile_distribution(
    //         tile_distribution_encoding<sequence<M2,M0, M1>,
    //                                    tuple<sequence<N0, N1,N2, N3, N4>>,
    //                                    tuple<sequence<0, 1>, sequence<0, 1>>,
    //                                    tuple<sequence<1, 1>, sequence<2, 3>>,
    //                                    sequence<1, 1, 1>,
    //                                    sequence<2, 0, 4>>{});
   
     return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<M0,  M1>,
                tuple<sequence<N0, N1, N3, N4>>,
                tuple<sequence<0, 1>, sequence<0, 1>>,
                tuple<sequence<0, 1>, sequence<1, 2>>,
                sequence<1, 1>,
                sequence<0, 3>>{});
    }


    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackA()
    {
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;
        constexpr index_t KPack = BlockGemm::Traits::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemPackB()
    {
        using BlockGemm = remove_cvref_t<decltype(Derived::template GetBlockGemm<Problem>())>;
        constexpr index_t KPack = BlockGemm::Traits::KPack;
        return KPack;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeA()
    {
        constexpr auto a_lds_desc     = MakeALdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_a = integer_least_multiple(
            sizeof(typename Problem::ADataType) * a_lds_desc.get_element_space_size(), 16);
        return smem_size_a;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeB()
    {
        constexpr auto b_lds_desc     = MakeBLdsBlockDescriptor<Problem>();
        constexpr index_t smem_size_b = integer_least_multiple(
            sizeof(typename Problem::BDataType) * b_lds_desc.get_element_space_size(), 16);
        return smem_size_b;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();

        return smem_size_a + smem_size_b;
    }
};

// UniversalGemm Policy
struct UniversalGemmPipelineAgBgCrPolicy_bias
    : public UniversalGemmBasePolicy_bias<UniversalGemmPipelineAgBgCrPolicy_bias>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        constexpr index_t vector_size =
            DS_READ_TR_SIZE() / sizeof(typename Problem::ComputeDataType);
        constexpr index_t thread_elements = WarpTile::at(I1) * WarpTile::at(I2) / get_warp_size();
        constexpr auto wg_attr_num_access =
            !(is_a_load_tr<Problem> || is_b_load_tr<Problem>) ? WGAttrNumAccessEnum::Single
            : vector_size == thread_elements                  ? WGAttrNumAccessEnum::Single
            : vector_size * 2 == thread_elements              ? WGAttrNumAccessEnum::Double
            : vector_size * 4 == thread_elements              ? WGAttrNumAccessEnum::Quad
                                                              : WGAttrNumAccessEnum::Invalid;

        using WarpGemm        = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                                typename Problem::ComputeDataType,
                                                typename Problem::CDataType,
                                                WarpTile::at(I0),
                                                WarpTile::at(I1),
                                                WarpTile::at(I2),
                                                Problem::TransposeC,
                                                false,
                                                Problem::UseStructuredSparsity,
                                                wg_attr_num_access>;
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
