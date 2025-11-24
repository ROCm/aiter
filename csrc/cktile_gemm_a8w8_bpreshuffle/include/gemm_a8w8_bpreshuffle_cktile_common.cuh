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
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>
#include "flatmm_basic.hpp"

using F16         = ck_tile::half_t;
using BF16        = ck_tile::bf16_t;
using FP8         = ck_tile::fp8_t;
using F32         = float;
using B16         = ck_tile::bf16_t;
using ADataType   = typename GemmBasicTypeConfig<ck_tile::fp8_t>::ADataType;
using BDataType   = typename GemmBasicTypeConfig<ck_tile::fp8_t>::BDataType;
using CDataType   = ck_tile::bf16_t;
using AccDataType = typename GemmBasicTypeConfig<ck_tile::fp8_t>::AccDataType;
using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                  const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                               FlatmmConfig::kPadN,
                                                               FlatmmConfig::kPadK,
                                                               FlatmmConfig::DoubleSmemBuffer,
                                                               ALayout,
                                                               BLayout,
                                                               ELayout,
                                                               FlatmmConfig::TransposeC,
                                                               FlatmmConfig::UseStructuredSparsity,
                                                               persistent,
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>;

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, CodegenFlatmmShape, Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = FlatmmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        using CodegenPipelineProblem = ck_tile::FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v>;

        using CodegenFlatmmPipeline =
            ck_tile::FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             false>>;

        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = ck_tile::FlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem.Print();

            auto run_flush_cache = [&]() {
                // flush icache
                ck_tile::flush_icache();
                // rotating mem
                rotating_mem.Next();
                // clear c mem
                if(args.k_batch > 1)
                    hipGetErrorString(hipMemsetAsync(
                        args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
            };
            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                run_flush_cache,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time = ck_tile::launch_kernel(
                s,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };
    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}
template <bool sTransposeC,
          bool sUseStructuredSparsity,
          int sTileParitionerGroupNum,
          int sTileParitionerM01,
          ck_tile::index_t sNumWaveGroups,
          bool sDoubleSmemBuffer,
          bool PadM,
          bool PadN,
          bool PadK,
          int BlockPerCu,
          int MTile,
          int NTile,
          int KTile,
          int MWarp,
          int NWarp,
          int KWarp,
          int MWTile,
          int NWTile,
          int KWTile,
          ck_tile::GemmPipelineScheduler sScheduler = ck_tile::GemmPipelineScheduler::Default>
struct CreateTileConfig
{
    static constexpr bool TransposeC                = sTransposeC;
    static constexpr bool UseStructuredSparsity     = sUseStructuredSparsity;
    static constexpr int TileParitionerGroupNum     = sTileParitionerGroupNum;
    static constexpr int TileParitionerM01          = sTileParitionerM01;
    static constexpr ck_tile::index_t NumWaveGroups = sNumWaveGroups;
    static constexpr bool DoubleSmemBuffer          = sDoubleSmemBuffer;
    static constexpr bool kPadM                     = PadM;
    static constexpr bool kPadN                     = PadN;
    static constexpr bool kPadK                     = PadK;
    static constexpr int kBlockPerCu                = BlockPerCu;
    static constexpr int M_Tile                     = MTile;
    static constexpr int N_Tile                     = NTile;
    static constexpr int K_Tile                     = KTile;
    static constexpr int M_Warp                     = MWarp;
    static constexpr int N_Warp                     = NWarp;
    static constexpr int K_Warp                     = KWarp;
    static constexpr int M_Warp_Tile                = MWTile;
    static constexpr int N_Warp_Tile                = NWTile;
    static constexpr int K_Warp_Tile                = KWTile;
    static constexpr auto Scheduler                 = sScheduler;
};

template <typename AccDataType,
          typename EDataType,
          bool sTransposeC,
          bool sUseStructuredSparsity,
          int sTileParitionerGroupNum,
          int sTileParitionerM01,
          ck_tile::index_t sNumWaveGroups,
          bool sDoubleSmemBuffer,
          bool PadM,
          bool PadN,
          bool PadK,
          int BlockPerCu,
          int MTile,
          int NTile,
          int KTile,
          int MWarp,
          int NWarp,
          int KWarp,
          int MWTile,
          int NWTile,
          int KWTile,
          ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Default>
using CustomConfig = CreateTileConfig<sTransposeC,
                                      sUseStructuredSparsity,
                                      sTileParitionerGroupNum,
                                      sTileParitionerM01,
                                      sNumWaveGroups,
                                      sDoubleSmemBuffer,
                                      PadM,
                                      PadN,
                                      PadK,
                                      BlockPerCu,
                                      MTile,
                                      NTile,
                                      KTile,
                                      MWarp,
                                      NWarp,
                                      KWarp,
                                      MWTile,
                                      NWTile,
                                      KWTile,
                                      Scheduler>;

template <typename DDataType, typename EDataType, typename FlatmmInstance>
__forceinline__ torch::Tensor
gemm_a8w8_bpreshuffle_cktile_impl(torch::Tensor& XQ,
                                  torch::Tensor& WQ,
                                  torch::Tensor& x_scale,
                                  torch::Tensor& w_scale,
                                  torch::Tensor& out // Out:[M, N] fp16
)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
    using ADataType      = typename GemmBasicTypeConfig<ck_tile::fp8_t>::ADataType;
    using BDataType      = typename GemmBasicTypeConfig<ck_tile::fp8_t>::BDataType;
    using CDataType      = ck_tile::bf16_t;;
    using AccDataType    = typename GemmBasicTypeConfig<ck_tile::fp8_t>::AccDataType;
    using DsDataType     = ck_tile::tuple<>;
    using ALayout        = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout        = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout        = ck_tile::tensor_layout::gemm::RowMajor;
    using DsLayout       = ck_tile::tuple<>;
    using CDEElementWise = ck_tile::element_wise::PassThrough;
    int m                = XQ.size(0);
    int n                = out.size(1);
    int k                = XQ.size(1);

    using ScaleM       = typename ck_tile::FlatmmScalePointer<1>;
    using ScaleN       = typename ck_tile::FlatmmScalePointer<1>;


    ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN> args;
    args.a_ptr = (void*)XQ.data_ptr();
    args.b_ptr = (void*)WQ.data_ptr();
    args.scale_m = ck_tile::FlatmmScalePointer<1>{reinterpret_cast<AccDataType*>(x_scale.data_ptr()),m};
    args.scale_n = ck_tile::FlatmmScalePointer<1>{reinterpret_cast<AccDataType*>(w_scale.data_ptr()),n};
    args.e_ptr = (void*)out.data_ptr();

    args.k_batch  = 1;
    args.M        = m;
    args.N        = n;
    args.K        = k;
    args.stride_A = k;
    args.stride_B = k;
    args.stride_C = n;
    args.stride_E = n;

    const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(XQ));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    ck_tile::stream_config naive_config{stream};
    flatmm_calc<FlatmmInstance,
                ADataType,
                BDataType,
                DsDataType,
                AccDataType,
                CDataType,
                ALayout,
                BLayout,
                DsLayout,
                CLayout,
                ScaleM,
                ScaleN,
                false,
                CDEElementWise>(args, naive_config);

    return out;
}

#endif // USE_ROCM
