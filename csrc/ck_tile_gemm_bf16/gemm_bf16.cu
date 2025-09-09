// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <initializer_list>
#include <cstdlib>


#include <THC/THCAtomics.cuh>

#include <sstream>
#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ck_tile/host.hpp"
#include "gemm_utils.hpp"
#include "gemm_bool_switch.hpp"
#include "gemm_setting.hpp"
#include "tile_gemm_shape_bias.hpp"
#include "gemm_bias_kernel.hpp"



using GemmConfig = GemmConfigPreshuffle_2<ck_tile::bf16_t>;


template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ELayout,
          bool Persistent>
float gemm_calc(const ck_tile::GemmHostArgs_bias& args, const ck_tile::stream_config& s);

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ELayout,
          ck_tile::index_t kM,
          ck_tile::index_t kN,
          ck_tile::index_t MaxK,
          bool Persistent>
float gemm_calc_dipatch(const ck_tile::GemmHostArgs_bias& args, const ck_tile::stream_config& s);

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}


torch::Tensor ck_tile_gemm_bf16(
                        torch::Tensor& XQ,
                        torch::Tensor& WQ,
                        torch::Tensor& bias,
                        torch::Tensor& Y
                         )
{
  
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
  

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;
    using ALayout = Row;
    using BLayout = Col;
    using CLayout = Row;
    using ELayout = Row;
    
    
    ck_tile::index_t M        = XQ.size(0);
    ck_tile::index_t N        = WQ.size(0);
    ck_tile::index_t K        = XQ.size(1);
    ck_tile::index_t kbatch   = 1;
    int n_warmup              = 0;
    int n_repeat              = 1;
    bool persistent           = 1;
    ck_tile::index_t stride_A = 0;
    ck_tile::index_t stride_B = 0;
    ck_tile::index_t stride_C = 0;

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(ALayout{}));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(BLayout{}));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));
    
    
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using CDataType   = ck_tile::bf16_t;
    
    using AccDataType = typename GemmTypeConfig<ADataType, BDataType, CDataType>::AccDataType;
    
   
     ck_tile::GemmHostArgs_bias args =  {XQ.data_ptr(),
                                    WQ.data_ptr(),
                                    Y.data_ptr(),
                                    bias.data_ptr(),
                                    kbatch,
                                    M,
                                    N,
                                    K,
                                    stride_A,
                                    stride_B,
                                    stride_C};

    
    
    
    float ave_time;
   
    int device_id = XQ.device().index();
    at::hip::HIPGuard device_guard(device_id);

 
    hipStream_t hip_stream =at::hip::getCurrentHIPStream(device_id);
    auto config = ck_tile::stream_config{nullptr, false, 0, n_warmup, n_repeat, false, false, 1};
    // 2. 配置异步执行
    
    if (1)
    {
        
        if(persistent)
        {
           
            ave_time = gemm_calc<ADataType,
                                BDataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                CLayout,
                                ELayout,
                                true>(
                args, config);
        }
        else
        {
            ave_time = gemm_calc<ADataType,
                                BDataType,
                                AccDataType,
                                CDataType,
                                ALayout,
                                BLayout,
                                CLayout,
                                ELayout,
                                false>(
                args, config);
        }
        
        }
        else
        {
            TORCH_CHECK(false, "Unsupported scales/output dtype!");
        }
    


    

    return Y;
    }


template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ELayout,
          bool Persistent>
float gemm_calc(const ck_tile::GemmHostArgs_bias& args, const ck_tile::stream_config& s)
{
    float ave_time=0.0;
    KM_KN_KK_SWITCH(
    args.M,
    kM,
    args.N,
    kN,
    args.K,
    MaxK,
    [&]{
        ave_time = gemm_calc_dipatch<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout,
                                    ELayout,
                                    kM,
                                    kN,
                                    MaxK,
                                    Persistent>(args, s);
    });
    return ave_time;
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ELayout,
          ck_tile::index_t kM,
          ck_tile::index_t kN,
          ck_tile::index_t MaxK,
          bool Persistent>
float gemm_calc_dipatch(const ck_tile::GemmHostArgs_bias& args, const ck_tile::stream_config& s)
{
    using GemmTileShapeConfig_ = GemmTileShapeConfig<kM, kN, MaxK>;
    // using GemmConfig           = GemmConfigPreshuffle_2<ck_tile::bf16_t>;
  
 
    // GemmConfigPreshufle_2<ck_tile::fp8_t> GemmConfig;
    const bool pad_M = !(args.M % GemmTileShapeConfig_::M_Tile == 0);
    const bool pad_N = !(args.N % GemmTileShapeConfig_::N_Tile == 0);
    const bool pad_K = !(args.K % GemmTileShapeConfig_::K_Tile == 0);
    float ave_time{0};
    
    // printf("%d,%d,%d \n",GemmTileShapeConfig_::M_Tile,GemmTileShapeConfig_::N_Tile,GemmTileShapeConfig_::K_Tile);
    BOOL_SWITCH_3(
        pad_M,
        padM_,
        pad_N,
        padN_,
        pad_K,
        padK_,
        [&]{
            
            // printf("%d,%d,%d,%d,%d,%d\n",GemmTileShapeConfig_::M_Tile, GemmTileShapeConfig_::N_Tile,GemmTileShapeConfig_::K_Tile,GemmTileShapeConfig_::M_Warp,GemmTileShapeConfig_::N_Warp,GemmTileShapeConfig_::K_Warp);
            using GemmShape = ck_tile::TileGemmShape_bias<
            ck_tile::sequence<GemmTileShapeConfig_::M_Tile, GemmTileShapeConfig_::N_Tile, GemmTileShapeConfig_::K_Tile>,
            ck_tile::sequence<GemmTileShapeConfig_::M_Warp, GemmTileShapeConfig_::N_Warp, GemmTileShapeConfig_::K_Warp>,
            ck_tile::
                sequence<GemmTileShapeConfig_::M_Warp_Tile, GemmTileShapeConfig_::N_Warp_Tile, GemmTileShapeConfig_::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;
            
             using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<padM_,
                                           padN_,
                                           padK_,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           GemmConfig::NumWaveGroups>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<padM_,
                                                                 padN_,
                                                                 padK_,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 ELayout,
                                                                 GemmConfig::TransposeC,
                                                                 GemmConfig::UseStructuredSparsity,
                                                                 Persistent,
                                                                 GemmConfig::NumWaveGroups,
                                                                 GemmConfig::Preshuffle>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run =
        [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
            
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                ck_tile::CShuffleEpilogueProblem<ADataType,
                                                 BDataType,
                                                 ck_tile::tuple<>,
                                                 AccDataType,
                                                 CDataType,
                                                 ck_tile::tuple<>,
                                                 CLayout,
                                                 ck_tile::element_wise::PassThrough,
                                                 TilePartitioner::MPerBlock,
                                                 TilePartitioner::NPerBlock,
                                                 GemmTileShapeConfig_::M_Warp,
                                                 GemmTileShapeConfig_::N_Warp,
                                                 GemmTileShapeConfig_::M_Warp_Tile,
                                                 GemmTileShapeConfig_::N_Warp_Tile,
                                                 GemmTileShapeConfig_::K_Warp_Tile,
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation,
                                                 GemmConfig::NumWaveGroups>>;
            using Kernel = ck_tile::GemmKernel_bias<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            dim3 grids;
            if constexpr(Persistent)
            {
                grids = Kernel::MaxOccupancyGridSize(s);
            }
            else
            {
                grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            }
            constexpr dim3 blocks = Kernel::BlockSize();
            // dim3 blocks = Kernel::BlockSize();
            // if(!Kernel::IsSupportedArgument(kargs))
            // {
            //     throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
            // }

                    
                   
                       if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }
        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes();
            auto size_b_buffer = b_n.get_element_space_size_in_bytes();

            ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b_buffer);
            // rotating_mem.Print();

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
               ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
        else
        {
            ave_time =
                ck_tile::launch_kernel(s,
                                       ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
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

        });
     
     
    return ave_time;
}
