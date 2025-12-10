// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"
#include "moe_cktile2stages.h"
#include <ATen/ATen.h>
#include <hip/hip_runtime.h>
#include <string>

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

template <typename DataType,
          int M_Tile_,
          int N_Tile_,
          int K_Tile_,
          int M_Warp_,
          int N_Warp_,
          int M_Warp_Tile_,
          int N_Warp_Tile_,
          int K_Warp_Tile_,
          int kBlockPerCu_>
struct MoeFlatmmConfig
{
    static constexpr ck_tile::index_t M_Tile = M_Tile_;
    static constexpr ck_tile::index_t N_Tile = N_Tile_;
    static constexpr ck_tile::index_t K_Tile = K_Tile_;

    static constexpr ck_tile::index_t M_Warp = M_Warp_;
    static constexpr ck_tile::index_t N_Warp = N_Warp_;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = M_Warp_Tile_;
    static constexpr ck_tile::index_t N_Warp_Tile = N_Warp_Tile_;
    static constexpr ck_tile::index_t K_Warp_Tile = K_Warp_Tile_;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = kBlockPerCu_;
    static constexpr int TileParitionerGroupNum     = 1;
    static constexpr int TileParitionerM01          = 1;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = false;
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
          ck_tile::MoeFlatmmKind moe_kind,
          typename CDEElementWise,
          typename MoeFlatmmHostArgs>
void moe_gemm(const MoeFlatmmHostArgs& args, const ck_stream_config& s)
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
                                                               false, // UsePersistentKernel_
                                                               FlatmmConfig::NumWaveGroups,
                                                               true>; // Preshuffle_

    constexpr bool MXFP4_Pipeline = std::is_same_v<BDataType, ck_tile::pk_fp4_t>;

    if constexpr(!MXFP4_Pipeline && moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up)
    {
        static_assert(
            FlatmmConfig::N_Tile % (FlatmmConfig::N_Warp * FlatmmConfig::N_Warp_Tile * 2) == 0,
            "requires NRepeat is multiple of 2 for FFN_gemm1_gate_up");
    }

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_flatmm requires ADataType is a wider type than BDataType");

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    const ck_tile::amd_buffer_coherence_enum b_mem_nt_type =
        BaseGemmPipeline::GetBMemNTType(
            args.NumTokens,
            args.N,
            args.K);

    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_,
                         const auto b_mem_nt_type_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = FlatmmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;
        constexpr auto b_mem_nt_type_v  = b_mem_nt_type_.value;

        using CodegenPipelineProblem =
            std::conditional_t<MXFP4_Pipeline,
                               ck_tile::F16xMXF4FlatmmPipelineProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CodegenFlatmmShape,
                                                                      CodegenGemmTraits,
                                                                      scheduler,
                                                                      has_hot_loop_v,
                                                                      tail_number_v,
                                                                      b_mem_nt_type_v>,
                               ck_tile::FlatmmPipelineProblem<ADataType,
                                                              BDataType,
                                                              AccDataType,
                                                              CodegenFlatmmShape,
                                                              CodegenGemmTraits,
                                                              scheduler,
                                                              has_hot_loop_v,
                                                              tail_number_v,
                                                              b_mem_nt_type_v>>;

        constexpr int BlockedXDLN_PerWarp =
            (MXFP4_Pipeline || (moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up))
                ? 2
                : 1; // determined by scale shuffle pattern

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                             ComputeDataType,
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
                                             FlatmmConfig::TiledMMAPermuteN,
                                             BlockedXDLN_PerWarp>>;

        using CodegenFlatmmPipeline = std::conditional_t<
            MXFP4_Pipeline,
            ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>,
            ck_tile::MoeFlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>>;

        using FusedAct =
            std::conditional_t<MXFP4_Pipeline, ck_tile::moe::Swiglu, ck_tile::moe::MoeSilu>;

        using Kernel = ck_tile::MoeFlatmmKernel<TilePartitioner,
                                                CodegenFlatmmPipeline,
                                                GemmEpilogue,
                                                moe_kind,
                                                FusedAct>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        // if(!Kernel::IsSupportedArgument(kargs))
        // {
        //     throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        // }

        // if(s.log_level_ > 0)
        // {
        //     std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
        //               << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
        //               << "problem: " << CodegenPipelineProblem::GetName() << "\n"
        //               << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
        //               << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
        //               << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
        //               << std::endl;
        // }
        //
        // if(s.flush_cache_)
        // {
        //     std::cout << "Flushing cache..." << std::endl;
        //     static constexpr ck_tile::index_t APackedSize =
        //         std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
        //     static constexpr ck_tile::index_t BPackedSize =
        //         std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;

        //     ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
        //         moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2 ? args.NumTokens * args.TopK
        //                                                        : args.NumTokens,
        //         args.K,
        //         args.stride_A,
        //         is_row_major(ALayout{})));
        //     ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
        //         args.K, args.N * args.NumExperts, args.stride_B, is_row_major(BLayout{})));

        //     const int outputN =
        //         moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm1_gate_up ? args.N / 2 : args.N;

        //     auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
        //     auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

        //     ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
        //         kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
        //     rotating_mem.Print();

        //     auto run_flush_cache = [&]() {
        //         // flush icache
        //         ck_tile::flush_icache();
        //         // rotating mem
        //         rotating_mem.Next();
        //         // clear c mem
        //         if(moe_kind == ck_tile::MoeFlatmmKind::kFFN_gemm2)
        //             hipGetErrorString(hipMemsetAsync(
        //                 args.e_ptr, 0, args.NumTokens * args.N * sizeof(CDataType),
        //                 s.stream_id_));
        //         else if(args.k_batch > 1)
        //             hipGetErrorString(
        //                 hipMemsetAsync(args.e_ptr,
        //                                0,
        //                                args.NumTokens * args.TopK * outputN * sizeof(CDataType),
        //                                s.stream_id_));
        //     };
        //     ave_time = ck_tile::launch_kernel_preprocess(
        //         s,
        //         run_flush_cache,
        //         ck_tile::make_kernel<blocks.x, FlatmmConfig::kBlockPerCu>(
        //             Kernel{}, grids, blocks, 0, kargs));
        // }
        // else
        // {
        ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        // }
        // return ave_time;
    };

    const auto RunBMem = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        if(b_mem_nt_type == ck_tile::amd_buffer_coherence_enum::WAVE_NT1)
        {
            Run(has_hot_loop_,
                tail_number_,
                memory_operation_,
                ck_tile::integral_constant<ck_tile::amd_buffer_coherence_enum,
                                           ck_tile::amd_buffer_coherence_enum::WAVE_NT1>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                memory_operation_,
                ck_tile::integral_constant<ck_tile::amd_buffer_coherence_enum,
                                           ck_tile::amd_buffer_coherence_enum::coherence_default>{});
        }
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            RunBMem(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                               ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            RunBMem(has_hot_loop_,
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
