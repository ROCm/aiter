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

// using ADataType   = ck_tile::fp8_t;
// using BDataType   = ck_tile::fp8_t;
// using CDataType   = ck_tile::half_t;
// using AccDataType = float;

// using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
// using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
// using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

// using CDEElementWise = ck_tile::element_wise::PassThrough;

// template <typename AB1DataType,
//           typename EDataType,
//           ck::index_t M_Tile,
//           ck::index_t N_Tile,
//           ck::index_t K_Tile,
//           ck::index_t M_Warp,
//           ck::index_t N_Warp,
//           ck::index_t K_Warp,
//           ck::index_t M_Warp_Tile,
//           ck::index_t N_Warp_Tile,
//           ck::index_t K_Warp_Tile,
//           bool kPadM,
//           bool kPadN,
//           bool kPadK,
//           bool TransposeC                          = false,
//           bool DoubleSmemBuffer                    = false,
//           bool UsePersistentKernel                 = false,
//           ck_tile::GemmPipelineScheduler Scheduler = ck_tile::GemmPipelineScheduler::Intrawave>
// struct TileGemmTileHelperF8BlockScale
// {
//     using GemmShape =
//         ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
//                                ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
//                                ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
//     using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;
//     using GemmTraits      = ck_tile::TileGemmQuantTraits<kPadM,
//                                                          kPadN,
//                                                          kPadK,
//                                                          false, // PreshuffleQuant
//                                                          false, // PreshuffleB
//                                                          ALayout,
//                                                          BLayout,
//                                                          CLayout,
//                                                          ck_tile::QuantType::RowColQuant,
//                                                          ALayout, // for AQLayout
//                                                          BLayout, // for BQLayout
//                                                          TransposeC,
//                                                          DoubleSmemBuffer>;
//     using GemmPipelineProblem =
//         ck_tile::GemmPipelineProblemBase<ADataType, BDataType, EDataType, GemmShape, GemmTraits>;
//     using PipelineProblem = ck_tile::GemmRowColTensorQuantPipelineProblem<ADataType,
//                                                                           BDataType,
//                                                                           AccDataType,
//                                                                           AccDataType,
//                                                                           GemmShape,
//                                                                           GemmTraits,
//                                                                           TransposeC,
//                                                                           AB1DataType,
//                                                                           Scheduler>;
//     using GemmPipeline    = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;
//     using GemmEpilogue    = ck_tile::CShuffleEpilogue<
//            ck_tile::CShuffleEpilogueProblem<ADataType,
//                                             BDataType,
//                                             ck_tile::tuple<>,
//                                             AccDataType,
//                                             CDataType,
//                                             ck_tile::tuple<>,
//                                             CLayout,
//                                             CDEElementWise,
//                                             TilePartitioner::MPerBlock,
//                                             TilePartitioner::NPerBlock,
//                                             M_Warp,
//                                             N_Warp,
//                                             M_Warp_Tile,
//                                             N_Warp_Tile,
//                                             K_Warp_Tile,
//                                             TransposeC,
//                                             ck_tile::memory_operation_enum::set>>;
//     using type = ck_tile::QuantGemmKernel<TilePartitioner,
//                                           GemmPipeline,
//                                           GemmEpilogue,
//                                           ck_tile::QuantType::RowColQuant>;
// };

// template <typename DDataType, typename EDataType, typename GemmHelper>
// __forceinline__ torch::Tensor tile_gemm_a8w8_blockscale_impl(torch::Tensor& XQ,
//                                                              torch::Tensor& WQ,
//                                                              torch::Tensor& x_scale,
//                                                              torch::Tensor& w_scale,
//                                                              torch::Tensor& Y)
// {
//     using GemmInstance = typename GemmHelper::type;

//     ck_tile::QuantGemmHostArgs args;
//     args.a_ptr  = XQ.data_ptr();
//     args.aq_ptr = x_scale.data_ptr();
//     args.b_ptr  = WQ.data_ptr();
//     args.bq_ptr = w_scale.data_ptr();
//     args.c_ptr  = Y.data_ptr();

//     // split-k is not supported yet for tile quant gemm, set k_batch to 1
//     args.k_batch = 1;
//     args.M       = XQ.size(0);
//     args.N       = WQ.size(0);
//     args.K       = XQ.size(1);
//     // Row quantization for A
//     args.QK_A = 1;
//     // Column quantization for B
//     args.QK_B      = 1;
//     args.stride_A  = XQ.stride(-2);
//     args.stride_B  = WQ.stride(-2);
//     args.stride_C  = Y.stride(-2);
//     args.stride_AQ = x_scale.stride(-2);
//     args.stride_BQ = w_scale.stride(-2);

//     // do tile GEMM
//     const dim3 grids  = GemmInstance::GridSize(args.M, args.N, args.k_batch);
//     const dim3 blocks = GemmInstance::BlockSize();
//     auto kargs        = GemmInstance::MakeKernelArgs(args);
//     TORCH_CHECK(GemmInstance::IsSupportedArgument(kargs), "Wrong! This GEMM is not supported!");
//     auto ave_time = ck_tile::launch_kernel(
//         ck_tile::stream_config{nullptr /*stream_id*/, false /*time_kernel*/, 1 /*log_level*/},
//         ck_tile::make_kernel<GemmInstance::kBlockPerCu>(GemmInstance{}, grids, blocks, 0,
//         kargs));

//     return Y;
// }

#endif // USE_ROCM
