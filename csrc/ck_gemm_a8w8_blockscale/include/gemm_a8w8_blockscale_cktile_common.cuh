#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

using TILE_FP32 = float;
using TILE_FP16 = ck_tile::half_t;
using TILE_BF16 = ck_tile::bf16_t;
using TILE_FP8  = ck_tile::fp8_t;

using ADataType       = TILE_FP8;
using BDataType       = TILE_FP8;
using AccDataType     = TILE_FP32;
using ComputeDataType = ADataType;

using ALayout         = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout        = ck_tile::tensor_layout::gemm::RowMajor;
using AQLayout_8Warps = ck_tile::tensor_layout::gemm::ColumnMajor;
using BLayout         = ck_tile::tensor_layout::gemm::ColumnMajor;
using BQLayout        = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout         = ck_tile::tensor_layout::gemm::RowMajor;

using CDEElementWise = ck_tile::element_wise::PassThrough;

using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;

template <ck_tile::index_t M_Tile,
          ck_tile::index_t N_Tile,
          ck_tile::index_t K_Tile,
          ck_tile::index_t M_Warp,
          ck_tile::index_t N_Warp,
          ck_tile::index_t K_Warp,
          ck_tile::index_t M_Warp_Tile,
          ck_tile::index_t N_Warp_Tile,
          ck_tile::index_t K_Warp_Tile,
          bool TiledMMAPermuteN                    = false,
          bool TransposeC                          = false,
          bool UsePersistentKernel                 = false,
          ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave,
          int BlockPerCu                           = 1>
struct CreateTileGemmConfig
{
    static constexpr ck_tile::index_t M_Tile_v                  = M_Tile;
    static constexpr ck_tile::index_t N_Tile_v                  = N_Tile;
    static constexpr ck_tile::index_t K_Tile_v                  = K_Tile;
    static constexpr ck_tile::index_t M_Warp_v                  = M_Warp;
    static constexpr ck_tile::index_t N_Warp_v                  = N_Warp;
    static constexpr ck_tile::index_t K_Warp_v                  = K_Warp;
    static constexpr ck_tile::index_t M_Warp_Tile_v             = M_Warp_Tile;
    static constexpr ck_tile::index_t N_Warp_Tile_v             = N_Warp_Tile;
    static constexpr ck_tile::index_t K_Warp_Tile_v             = K_Warp_Tile;
    static constexpr bool TiledMMAPermuteN_v                    = TiledMMAPermuteN;
    static constexpr bool TransposeC_v                          = TransposeC;
    static constexpr bool UsePersistentKernel_v                 = UsePersistentKernel;
    static constexpr ck_tile::GemmPipelineScheduler Scheduler_v = Scheduler;
    static constexpr int BlockPerCu_v                           = BlockPerCu;
};

template <ck_tile::index_t M_Tile,
          ck_tile::index_t N_Tile,
          ck_tile::index_t K_Tile,
          ck_tile::index_t M_Warp,
          ck_tile::index_t N_Warp,
          ck_tile::index_t K_Warp,
          ck_tile::index_t M_Warp_Tile,
          ck_tile::index_t N_Warp_Tile,
          ck_tile::index_t K_Warp_Tile,
          bool TiledMMAPermuteN                    = false,
          bool TransposeC                          = false,
          bool UsePersistentKernel                 = false,
          ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave,
          int BlockPerCu                           = 1>
using TileGemmConfig = CreateTileGemmConfig<M_Tile,
                                            N_Tile,
                                            K_Tile,
                                            M_Warp,
                                            N_Warp,
                                            K_Warp,
                                            M_Warp_Tile,
                                            N_Warp_Tile,
                                            K_Warp_Tile,
                                            TiledMMAPermuteN,
                                            TransposeC,
                                            UsePersistentKernel,
                                            Scheduler,
                                            BlockPerCu>;

template <typename QDataType,
          typename OutDataType,
          typename GemmConfig,
          bool PadN,
          bool PadK,
          bool PreshuffleB,
          bool UseDoubleSmemBuffer = PreshuffleB>
void TileGemmComputeImpl(ck_tile::QuantGemmHostArgs& args)
{

    static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped;
    static constexpr bool transpose_c             = BQuantGroupSize::kN == 128;
    static constexpr bool eight_warps =
        BQuantGroupSize::kN == 128 &&
        (GemmConfig::M_Warp_v * GemmConfig::N_Warp_v * GemmConfig::K_Warp_v == 8) &&
        GemmConfig::K_Warp_Tile_v == 128;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile_v, GemmConfig::N_Tile_v, GemmConfig::K_Tile_v>,
        ck_tile::sequence<GemmConfig::M_Warp_v, GemmConfig::N_Warp_v, GemmConfig::K_Warp_v>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile_v,
                          GemmConfig::N_Warp_Tile_v,
                          GemmConfig::K_Warp_Tile_v>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits =
        ck_tile::TileGemmQuantTraits<false, // PadM
                                     PadN,
                                     PadK,
                                     false, // PreshuffleQuant for A, not supported yet
                                     false, // PreshuffleQuant for B, not supported yet (distinct from PreshuffleB below)
                                     PreshuffleB, // PreshuffleB (weight/B matrix preshuffle), supported
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     QuantMode,
                                     std::conditional_t<eight_warps, AQLayout_8Warps, AQLayout>,
                                     BQLayout,
                                     transpose_c,
                                     UseDoubleSmemBuffer>;

    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<ADataType,
                                                                 BDataType,
                                                                 AccDataType,
                                                                 GemmShape,
                                                                 GemmTraits,
                                                                 ComputeDataType>;

    using BaseGemmPipeline = std::conditional_t<
        eight_warps,
        ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>;

    //const ck_tile::index_t K_split =
    //    (args.K + GemmConfig::K_Tile_v - 1) / GemmConfig::K_Tile_v * GemmConfig::K_Tile_v;
    //const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    //const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::index_t K_split  = ck_tile::integer_least_multiple(args.K, GemmConfig::K_Tile_v);
    const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop         = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;

        using PipelineProblem = ck_tile::GemmABQuantPipelineProblem<ADataType,
                                                                    QDataType, // AQDataType
                                                                    BDataType,
                                                                    QDataType, // BQDataType
                                                                    AccDataType,
                                                                    GemmShape,
                                                                    GemmTraits,
                                                                    AQuantGroupSize,
                                                                    BQuantGroupSize,
                                                                    transpose_c,
                                                                    ComputeDataType,
                                                                    GemmConfig::Scheduler_v,
                                                                    has_hot_loop_v,
                                                                    tail_number_v>;

        using GemmPipeline = std::conditional_t<
            eight_warps,
            ck_tile::ABQuantGemmPipelineAgBgCrAsync<PipelineProblem>,
            std::conditional_t<UseDoubleSmemBuffer && PreshuffleB,
                               ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                               ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>;
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             ck_tile::tuple<>,
                                             AccDataType,
                                             OutDataType,
                                             ck_tile::tuple<>,
                                             CLayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp_v,
                                             GemmConfig::N_Warp_v * GemmConfig::K_Warp_v,
                                             GemmConfig::M_Warp_Tile_v,
                                             GemmConfig::N_Warp_Tile_v,
                                             GemmConfig::K_Warp_Tile_v,
                                             transpose_c,
                                             1,
                                             false,
                                             1,
                                             GemmConfig::TiledMMAPermuteN_v>>;

        using Kernel =
            ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(args.k_batch != 1)
        {
            throw std::runtime_error("split-k is not supported yet!");
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        ck_tile::launch_kernel(
            ck_tile::stream_config{nullptr /*stream_id*/, false /*time_kernel*/, 1 /*log_level*/},
            ck_tile::make_kernel<GemmConfig::BlockPerCu_v>(Kernel{}, grids, blocks, 0, kargs));
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename QDataType, typename OutDataType, typename GemmConfig, bool PreshuffleB>
void TileGemmCompute(ck_tile::QuantGemmHostArgs& args)
{
    const bool pad_n = (args.N % BQuantGroupSize::kN != 0);
    const bool pad_k = (args.K % AQuantGroupSize::kK != 0);

    if(pad_n && pad_k)
    {
        TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, true, true, PreshuffleB>(args);
    }
    else if(pad_n && !pad_k)
    {
        TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, true, false, PreshuffleB>(args);
    }
    else if(!pad_n && pad_k)
    {
        TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, false, true, PreshuffleB>(args);
    }
    else
    {
        TileGemmComputeImpl<QDataType, OutDataType, GemmConfig, false, false, PreshuffleB>(args);
    }
}

template <typename QDataType, typename OutDataType, typename GemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_blockscale_cktile_impl(torch::Tensor& XQ,
                                                               torch::Tensor& WQ,
                                                               torch::Tensor& x_scale,
                                                               torch::Tensor& w_scale,
                                                               torch::Tensor& Y,
                                                               bool PreshuffleB)
{
    // check
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    // M, N, K
    const int M = XQ.size(0);
    const int N = WQ.size(0);
    const int K = XQ.size(1);

    // prepare args
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr  = XQ.data_ptr();
    args.aq_ptr = x_scale.data_ptr();
    args.b_ptr  = WQ.data_ptr();
    args.bq_ptr = w_scale.data_ptr();
    args.c_ptr  = Y.data_ptr();

    // split-k is not supported yet for tile quant gemm, set k_batch to 1
    args.k_batch = 1;
    args.M       = M;
    args.N       = N;
    args.K       = K;

    const int AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
    const int BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
    const int BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);

    const int stride_A = K;
    const int stride_B = K;
    const int stride_C = N;
    // const int stride_AQ = AQK;
    const int stride_AQ = (GemmInstance::M_Warp_v * GemmInstance::N_Warp_v * GemmInstance::K_Warp_v == 8)
                              ? M    // Col-Major
                              : AQK; // Row-Major
    const int stride_BQ = BQN;

    args.QK_A      = AQK;
    args.QK_B      = BQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_AQ = stride_AQ;
    args.stride_BQ = stride_BQ;

    // do tile GEMM
    if(PreshuffleB)
    {
        TileGemmCompute<QDataType, OutDataType, GemmInstance, true>(args);
    }
    else
    {
        TileGemmCompute<QDataType, OutDataType, GemmInstance, false>(args);
    }

    return Y;
}

#endif // USE_ROCM
