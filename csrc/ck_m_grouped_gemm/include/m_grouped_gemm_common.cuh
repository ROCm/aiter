// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "m_grouped_gemm.h"
#include <hip/hip_runtime.h>
#include <ATen/ATen.h>

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

template <typename DataType,
          int M_Tile_, 
          int N_Tile_, 
          int K_Tile_, 
          int M_Warp_, 
          int N_Warp_, 
          int M_Warp_Tile_, 
          int N_Warp_Tile_, 
          int K_Warp_Tile_>
struct MGroupedFlatmmConfig
{
    static constexpr ck_tile::index_t M_Tile = M_Tile_;
    static constexpr ck_tile::index_t N_Tile = N_Tile_;
    static constexpr ck_tile::index_t K_Tile = K_Tile_ / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = M_Warp_;
    static constexpr ck_tile::index_t N_Warp = N_Warp_;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = M_Warp_Tile_;
    static constexpr ck_tile::index_t N_Warp_Tile = N_Warp_Tile_;
    // TODO:
    static constexpr ck_tile::index_t K_Warp_Tile = 64 / sizeof(DataType); // sizeof(DataType) == 2 ? 32 : 64;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 1;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

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
            bool persistent,
            typename CDEElementWise,
            typename KernelArguments>
void grouped_flatmm(KernelArguments& args, ck_stream_config& s)
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
                                             CodegenPipelineProblem::kBlockSize,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             memory_operation,
                                             FlatmmConfig::NumWaveGroups>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel =
            ck_tile::GroupedFlatmmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        // if(!Kernel::IsSupportedArgument(kargs))
        // {
        //     throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        // }

        // if(s.flush_cache_)
        // {
        //     std::cout << "Flushing cache..." << std::endl;
        //     static constexpr ck_tile::index_t APackedSize =
        //         std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
        //     static constexpr ck_tile::index_t BPackedSize =
        //         std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

        //     ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
        //         args.group_count * args.M, args.K, args.stride_A, is_row_major(ALayout{})));
        //     ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
        //         args.K, args.group_count * args.N, args.stride_B, is_row_major(BLayout{})));

        //     auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
        //     auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

        //     ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
        //         kargs.a_ptr, kargs.b_shuffle_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
        //     rotating_mem.Print();

        //     auto run_flush_cache = [&]() {
        //         // flush icache
        //         ck_tile::flush_icache();
        //         // rotating mem
        //         rotating_mem.Next();
        //         // clear c mem
        //         if(args.k_batch > 1)
        //             hipGetErrorString(hipMemsetAsync(
        //                 args.e_ptr, 0, args.group_count * args.M * args.N * sizeof(CDataType), s.stream_id_));
        //     };
        //     ave_time = ck_tile::launch_kernel_preprocess(
        //         s,
        //         run_flush_cache,
        //         ck_tile::make_kernel<blocks.x, FlatmmConfig::kBlockPerCu>(
        //             Kernel{}, grids, blocks, 0, kargs));
        // }
        // else
        // {
        ck_tile::launch_kernel(s,
                                ck_tile::make_kernel<blocks.x, FlatmmConfig::kBlockPerCu>(
                                    Kernel{}, grids, blocks, 0, kargs));
        // }

        // return ave_time;
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

    if(tail_num == ck_tile::TailNumber::Odd)
    {
        RunSplitk(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
    }
    else if(tail_num == ck_tile::TailNumber::Even)
    {
        RunSplitk(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
    }
    else
    {
        std::ostringstream err;
        err << "For compute pipeline tail number should always be Full, but have \"" << tail_num
            << "\" which is not supported! PrefetchStages: " << BaseGemmPipeline::PrefetchStages
            << "\n File: " << __FILE__ << ":" << __LINE__ << ", in function: " << __func__;
        throw std::runtime_error(err.str());
    }
}