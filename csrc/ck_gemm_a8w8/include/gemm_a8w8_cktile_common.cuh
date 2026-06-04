#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

using TILE_FP32 = float;
using TILE_I32  = int;
using TILE_FP16 = ck_tile::half_t;
using TILE_BF16 = ck_tile::bf16_t;
using TILE_FP8  = ck_tile::fp8_t;
using TILE_I8   = int8_t;

using ALayout  = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout  = ck_tile::tensor_layout::gemm::ColumnMajor;
using D0Layout = ck_tile::tensor_layout::gemm::RowMajor;
using D1Layout = ck_tile::tensor_layout::gemm::ColumnMajor;
using D2Layout = ck_tile::tensor_layout::gemm::RowMajor;
using ELayout  = ck_tile::tensor_layout::gemm::RowMajor;

using CDEElementWise = ck_tile::element_wise::PassThrough;

using HostArgs = ck_tile::GemmMultiABDHostArgs<1, 1, 2>;

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
          int BlockPerCu                           = 1,
          bool AQRowMajor                          = false>
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
    static constexpr bool AQRowMajor_v                          = AQRowMajor;
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
          int BlockPerCu                           = 1,
          bool AQRowMajor                          = false>
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
                                            BlockPerCu,
                                            AQRowMajor>;

template <typename ABDataType,
          typename DDataType,
          typename EDataType, 
          typename GemmConfig,
          bool PreshuffleB,
          bool UseDoubleSmemBuffer = PreshuffleB>
void TileGemmComputeImpl(const HostArgs& args)
{
    using ComputeDataType = ABDataType;
    using AccDataType = std::conditional_t<std::is_same_v<ABDataType, TILE_I8>, TILE_I32, TILE_FP32>;

    constexpr bool kPadM            = false;
    constexpr bool kPadN            = false;
    constexpr bool kPadK            = false;

    constexpr bool TransposeC = false;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile_v, GemmConfig::N_Tile_v, GemmConfig::K_Tile_v>,
        ck_tile::sequence<GemmConfig::M_Warp_v, GemmConfig::N_Warp_v, GemmConfig::K_Warp_v>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile_v,
                          GemmConfig::N_Warp_Tile_v,
                          GemmConfig::K_Warp_Tile_v>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                        kPadN,
                                                        kPadK,
                                                        UseDoubleSmemBuffer,
                                                        ck_tile::tuple<ALayout>,
                                                        ck_tile::tuple<BLayout>,
                                                        ELayout,
                                                        TransposeC>;

    using PipelineProblem = ck_tile::UniversalGemmPipelineProblem<ck_tile::tuple<ABDataType>,
                                                                  ck_tile::tuple<ABDataType>,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  GemmTraits,
                                                                  GemmConfig::Scheduler_v,
                                                                  ck_tile::element_wise::PassThrough,
                                                                  ck_tile::element_wise::PassThrough>;

    using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;

    // TODO: add optional bias
    using element_op = ck_tile::element_wise::MultiDMultiply;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ck_tile::tuple<ABDataType>,
                                        ck_tile::tuple<ABDataType>,
                                        ck_tile::tuple<DDataType, DDataType>,
                                        AccDataType,
                                        EDataType,
                                        ck_tile::tuple<D0Layout, D1Layout>,
                                        ELayout,
                                        element_op,
                                        TilePartitioner::MPerBlock,
                                        TilePartitioner::NPerBlock,
                                        GemmConfig::M_Warp_v,
                                        GemmConfig::N_Warp_v * GemmConfig::K_Warp_v,
                                        GemmConfig::M_Warp_Tile_v,
                                        GemmConfig::N_Warp_Tile_v,
                                        GemmConfig::K_Warp_Tile_v,
                                        PipelineProblem::TransposeC>>;

    using Kernel = ck_tile::GemmKernelMultiABD<TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs   = Kernel::MakeKernelArgs(args);

    const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    ck_tile::launch_kernel(
        ck_tile::stream_config{at::hip::getCurrentHIPStream() /*stream_id*/, false /*time_kernel*/, 1 /*log_level*/},
        ck_tile::make_kernel<GemmConfig::BlockPerCu_v>(Kernel{}, grids, blocks, 0, kargs));
}

template <typename ABDataType, typename DDataType, typename EDataType, typename GemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_cktile_impl(torch::Tensor& XQ,
                                                    torch::Tensor& WQ,
                                                    torch::Tensor& x_scale,
                                                    torch::Tensor& w_scale,
                                                    torch::Tensor& Y,
                                                    std::optional<torch::Tensor> bias,
                                                    int k_batch = 1)
{
    // check
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");

    TORCH_CHECK(XQ.stride(-1) == 1,
                "CKTile blockscale GEMM: XQ inner dim must be contiguous, "
                "got strides=[",
                XQ.stride(0),
                ",",
                XQ.stride(1),
                "]");
    TORCH_CHECK(WQ.stride(-1) == 1,
                "CKTile blockscale GEMM: WQ inner dim must be contiguous, "
                "got strides=[",
                WQ.stride(0),
                ",",
                WQ.stride(1),
                "]");
    TORCH_CHECK(Y.stride(-1) == 1,
                "CKTile blockscale GEMM: Y inner dim must be contiguous, "
                "got strides=[",
                Y.stride(0),
                ",",
                Y.stride(1),
                "]");

    // Split-K uses atomic_add into C; zero the output buffer first.
    // Use zero_() so all rows are cleared regardless of the leading-dimension
    // stride (e.g. padded tensors produced by vLLM's _maybe_pad_fp8_weight).
    if(k_batch > 1)
    {
        Y.zero_();
    }

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    std::array<int, 1> strideAs({K});
    std::array<int, 1> strideBs({K});
    int strideE = N;

    HostArgs args(
        std::array<const void*, 1>{XQ.data_ptr()},
        std::array<const void*, 1>{WQ.data_ptr()},
        std::array<const void*, 2>{w_scale.data_ptr(), x_scale.data_ptr()},
        Y.data_ptr(),
        k_batch,
        M,
        N,
        K,
        strideAs,
        strideBs,
        std::array<int, 2>({0, 0}),
        strideE);

    // TODO: add Multiply-Multiply-Add operator for bias case
    TileGemmComputeImpl<ABDataType, DDataType, EDataType, GemmInstance, false>(args);

    // if constexpr(has_bias)
    // {
    //     auto cde_element_op = ck_tile::element_wise::MultiDMultiplyAdd<EDataType, AccDataType, DDataType, DDataType>{};

    //     TileGemmComputeImpl<ABDataType, DDataType, EDataType, GemmInstance, false>(args);
    // }
    // else
    // {
    //     auto cde_element_op = ck_tile::element_wise::MultiDMultiply<EDataType, AccDataType, DDataType, DDataType>{};

    //     TileGemmComputeImpl<ABDataType, DDataType, EDataType, GemmInstance, false>(args);
    // }

    return Y;
}

#endif // USE_ROCM
