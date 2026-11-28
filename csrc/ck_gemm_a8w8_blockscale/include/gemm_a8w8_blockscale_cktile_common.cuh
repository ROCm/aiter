#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include "gemm_a8w8_blockscale_splitk_reduce.cuh"

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

// K batch blockIdx.z writes its partial to its own [M, N] slice of C, so every
// element has a single writer.
template <typename Kernel>
struct SplitKPartialsKernel : Kernel
{
    CK_TILE_DEVICE void operator()(typename Kernel::KernelArgs kargs) const
    {
        kargs.c_ptr = static_cast<typename Kernel::CDataType*>(kargs.c_ptr) +
                      static_cast<ck_tile::long_index_t>(blockIdx.z) * kargs.M * kargs.N;
        Kernel::operator()(kargs);
    }
};

template <typename QDataType,
          typename OutDataType,
          typename GemmConfig,
          bool PadN,
          bool PadK,
          bool PreshuffleB,
          bool SplitKPartials,
          bool UseDoubleSmemBuffer = PreshuffleB>
void TileGemmComputeImpl(ck_tile::QuantGemmHostArgs& args, ck_tile::index_t grid_k_batch)
{

    static constexpr ck_tile::QuantType QuantMode = ck_tile::QuantType::ABQuantGrouped;
    static constexpr bool transpose_c             = BQuantGroupSize::kN == 128;
    static constexpr bool eight_waves =
        BQuantGroupSize::kN == 128 &&
        (GemmConfig::M_Warp_v * GemmConfig::N_Warp_v * GemmConfig::K_Warp_v == 8) &&
        GemmConfig::K_Warp_Tile_v == 128;
    // When AQRowMajor is true for an 8-warp config, the kernel reads x_scale
    // in row-major layout natively, avoiding the host-side transpose.
    static constexpr bool aq_col_major = eight_waves && !GemmConfig::AQRowMajor_v;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile_v, GemmConfig::N_Tile_v, GemmConfig::K_Tile_v>,
        ck_tile::sequence<GemmConfig::M_Warp_v, GemmConfig::N_Warp_v, GemmConfig::K_Warp_v>,
        ck_tile::sequence<GemmConfig::M_Warp_Tile_v,
                          GemmConfig::N_Warp_Tile_v,
                          GemmConfig::K_Warp_Tile_v>>;

    using TilePartitioner = ck_tile::GemmTile1DPartitioner<GemmShape>;

    using GemmTraits = ck_tile::TileGemmQuantTraits<
        true, // PadM
        PadN,
        PadK,
        false,       // PreshuffleQuant for A, not supported yet
        false,       // PreshuffleQuant for B, not supported yet (distinct from PreshuffleB below)
        PreshuffleB, // PreshuffleB (weight/B matrix preshuffle), supported
        ALayout,
        BLayout,
        CLayout,
        QuantMode,
        std::conditional_t<aq_col_major, AQLayout_8Warps, AQLayout>,
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
        eight_waves,
        ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
        ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>;

    // Each split runs its own K range, so pick the pipeline from the per-split loop count.
    const ck_tile::index_t k_grain  = args.k_batch * GemmConfig::K_Tile_v;
    const ck_tile::index_t K_split  = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile_v;
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
            eight_waves,
            ck_tile::ABQuantGemmPipelineAgBgCrEightWaves<PipelineProblem>,
            std::conditional_t<UseDoubleSmemBuffer && PreshuffleB,
                               ck_tile::WPABQuantBPipelineAgBgCrV2<PipelineProblem>,
                               ck_tile::ABQuantGemmPipelineAgBgCrCompV3<PipelineProblem>>>;
        static_assert(!GemmConfig::TiledMMAPermuteN_v,
                      "TiledMMAPermuteN=true requires PermuteNEpilogue, not CShuffleEpilogue");
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
                                             1>>;

        using BaseKernel =
            ck_tile::QuantGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue, QuantMode>;
        using Kernel =
            std::conditional_t<SplitKPartials, SplitKPartialsKernel<BaseKernel>, BaseKernel>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, grid_k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }
        using k_attr_t = ck_tile::kernel_attr<eight_waves>;
        ck_tile::launch_kernel(
            ck_tile::stream_config{at::hip::getCurrentHIPStream() /*stream_id*/, false /*time_kernel*/, 1 /*log_level*/},
            ck_tile::make_kernel<GemmConfig::BlockPerCu_v, k_attr_t>(
                Kernel{}, grids, blocks, 0, kargs));
    };

    BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
}

template <typename QDataType,
          typename OutDataType,
          typename GemmConfig,
          bool PreshuffleB,
          bool SplitKPartials = false>
void TileGemmCompute(ck_tile::QuantGemmHostArgs& args, ck_tile::index_t grid_k_batch)
{
    const bool pad_n = (args.N % BQuantGroupSize::kN != 0);
    const bool pad_k = (args.K % AQuantGroupSize::kK != 0);

    if(pad_n && pad_k)
    {
        TileGemmComputeImpl<QDataType,
                            OutDataType,
                            GemmConfig,
                            true,
                            true,
                            PreshuffleB,
                            SplitKPartials>(args, grid_k_batch);
    }
    else if(pad_n && !pad_k)
    {
        TileGemmComputeImpl<QDataType,
                            OutDataType,
                            GemmConfig,
                            true,
                            false,
                            PreshuffleB,
                            SplitKPartials>(args, grid_k_batch);
    }
    else if(!pad_n && pad_k)
    {
        TileGemmComputeImpl<QDataType,
                            OutDataType,
                            GemmConfig,
                            false,
                            true,
                            PreshuffleB,
                            SplitKPartials>(args, grid_k_batch);
    }
    else
    {
        TileGemmComputeImpl<QDataType,
                            OutDataType,
                            GemmConfig,
                            false,
                            false,
                            PreshuffleB,
                            SplitKPartials>(args, grid_k_batch);
    }
}

template <typename QDataType, typename OutDataType, typename GemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_blockscale_cktile_impl(torch::Tensor& XQ,
                                                               torch::Tensor& WQ,
                                                               torch::Tensor& x_scale,
                                                               torch::Tensor& w_scale,
                                                               torch::Tensor& Y,
                                                               bool PreshuffleB,
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

    // M, N, K
    const int M = XQ.size(0);
    const int N = WQ.size(0);
    const int K = XQ.size(1);

    // Whether this kernel configuration uses column-major AQ layout,
    // requiring a host-side transpose of x_scale.
    constexpr bool aq_col_major =
        BQuantGroupSize::kN == 128 &&
        (GemmInstance::M_Warp_v * GemmInstance::N_Warp_v * GemmInstance::K_Warp_v == 8) &&
        GemmInstance::K_Warp_Tile_v == 128 &&
        !GemmInstance::AQRowMajor_v;

    constexpr bool eight_waves =
        BQuantGroupSize::kN == 128 &&
        (GemmInstance::M_Warp_v * GemmInstance::N_Warp_v * GemmInstance::K_Warp_v == 8) &&
        GemmInstance::K_Warp_Tile_v == 128;

    // prepare args
    ck_tile::QuantGemmHostArgs args;
    args.a_ptr = XQ.data_ptr();

    // Declared at function scope so the transposed tensor stays alive
    // through the async kernel launch.
    torch::Tensor x_scale_t;

    if constexpr(aq_col_major)
    {
        // 8-warp ColumnMajor AQ: transpose x_scale to col-major
        if(!PreshuffleB)
        {
            x_scale_t   = x_scale.transpose(0, 1).contiguous().view(x_scale.sizes());
            args.aq_ptr = x_scale_t.data_ptr();
        }
        else
        {
            args.aq_ptr = x_scale.data_ptr();
        }
    }
    else if constexpr(!eight_waves)
    {
        if(PreshuffleB)
        {
            x_scale_t   = x_scale.view({x_scale.size(1), x_scale.size(0)}).transpose(0, 1).contiguous();
            args.aq_ptr = x_scale_t.data_ptr();
        }
        else
        {
            args.aq_ptr = x_scale.data_ptr();
        }
    }
    else
    {
        // 8-warp RowMajor AQ: use x_scale directly, no transpose needed
        args.aq_ptr = x_scale.data_ptr();
    }

    args.b_ptr  = WQ.data_ptr();
    args.bq_ptr = w_scale.data_ptr();
    args.c_ptr  = Y.data_ptr();

    args.k_batch = k_batch;
    args.M       = M;
    args.N       = N;
    args.K       = K;

    const int AQK = ck_tile::integer_divide_ceil(K, AQuantGroupSize::kK);
    const int BQK = ck_tile::integer_divide_ceil(K, BQuantGroupSize::kK);
    const int BQN = ck_tile::integer_divide_ceil(N, BQuantGroupSize::kN);

    // Read leading-dimension strides from tensor metadata instead of
    // assuming dense layout.  vLLM's _maybe_pad_fp8_weight can produce
    // row-major tensors whose leading-dimension stride exceeds the
    // logical column count (e.g. shape [N,K] with stride [K+pad, 1]).
    const int stride_A  = XQ.stride(0);
    const int stride_B  = WQ.stride(0);
    const int stride_C  = Y.stride(0);
    const int stride_AQ = aq_col_major ? M : static_cast<int>(x_scale.stride(0));
    const int stride_BQ = w_scale.stride(0);

    args.QK_A      = AQK;
    args.QK_B      = BQK;
    args.stride_A  = stride_A;
    args.stride_B  = stride_B;
    args.stride_C  = stride_C;
    args.stride_AQ = stride_AQ;
    args.stride_BQ = stride_BQ;

    // More than two K batches on gfx950: each batch writes an FP32 partial to its
    // own workspace slice and the partials are reduced in batch order, so the
    // output does not depend on atomic arrival order.
    if(aiter_blockscale::use_splitk_partials(k_batch))
    {
        // Equal K ranges on plain B: run each batch as its own K=K/k_batch GEMM
        // (k_batch=1 in the kernel, so the epilogue stores instead of adding).
        // The batch index still comes from blockIdx.z. Otherwise the batches add
        // into a zeroed slice, one writer per element.
        const bool store_partials = !PreshuffleB && K % (k_batch * GemmInstance::K_Tile_v) == 0;
        auto ws = aiter_blockscale::splitk_workspace(Y, k_batch, M, N, /*zeroed=*/!store_partials);
        args.c_ptr    = ws.data_ptr();
        args.stride_C = N;
        if(store_partials)
        {
            // QK_A/QK_B stay the full group counts: the kernel uses them as the
            // scale extents and the BQ stride, offset by blockIdx.z * K groups.
            args.K       = K / k_batch;
            args.k_batch = 1;
        }
        if(PreshuffleB)
        {
            TileGemmCompute<QDataType, TILE_FP32, GemmInstance, true, true>(args, k_batch);
        }
        else
        {
            TileGemmCompute<QDataType, TILE_FP32, GemmInstance, false, true>(args, k_batch);
        }
        aiter_blockscale::splitk_reduce(ws, Y, k_batch, M, N);
        return Y;
    }

    // Split-K uses atomic_add into C; zero the output buffer first.
    // Use zero_() so all rows are cleared regardless of the leading-dimension
    // stride (e.g. padded tensors produced by vLLM's _maybe_pad_fp8_weight).
    if(k_batch > 1)
    {
        Y.zero_();
    }

    // do tile GEMM
    if(PreshuffleB)
    {
        TileGemmCompute<QDataType, OutDataType, GemmInstance, true>(args, k_batch);
    }
    else
    {
        TileGemmCompute<QDataType, OutDataType, GemmInstance, false>(args, k_batch);
    }

    return Y;
}

#endif // USE_ROCM
