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

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

#include "gemm_a8w8_blockscale_splitk_reduce.cuh"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using FP32 = float;
using FP16 = ck::half_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using A1DataType       = FP32;
using B0DataType       = FP8;
using B1DataType       = FP32;
using AccDataType      = FP32;
using CShuffleDataType = FP32;
using DsDataType       = ck::Tuple<>;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

// static constexpr ck::index_t Scale_Block_M = 1;
// static constexpr ck::index_t Scale_Block_N = 128;
// static constexpr ck::index_t Scale_Block_K = 128;

enum class GemmSpecialization : unsigned int {
    Default    = 0,
    MPadding   = 1u<<0,
    NPadding   = 1u<<1,
    KPadding   = 1u<<2,
    MNPadding  = MPadding|NPadding,
    MKPadding  = MPadding|KPadding,
    NKPadding  = NPadding|KPadding,
    MNKPadding = MPadding|NPadding|KPadding
};

static GemmSpecialization operator|(GemmSpecialization lhs, GemmSpecialization rhs) {
    return static_cast<GemmSpecialization>(unsigned(lhs) | unsigned(rhs));
}

static GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block)
{
    auto HasPadding = [](int x, int y) {
        return (x % y) != 0;
    };

    GemmSpecialization spec{};
    if (HasPadding(m, m_per_block))
        spec = spec | GemmSpecialization::MPadding;
    if (HasPadding(n, n_per_block))
        spec = spec | GemmSpecialization::NPadding;
    if (HasPadding(k, k_per_block))
        spec = spec | GemmSpecialization::KPadding;

    return spec;
}

template <typename AB1DataType,
          typename EDataType,
          ck::index_t BlockSize,
          ck::index_t Scale_Block_M,
          ck::index_t Scale_Block_N,
          ck::index_t Scale_Block_K,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          ck::index_t CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
          ck::index_t CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched =
              ck::BlockGemmPipelineScheduler::Intrawave,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v1,
          auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default>
using DeviceLegacyGemmHelperF8BlockScale =
    ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    // clang-format off
         <A0Layout, B0Layout, DsLayout, ELayout,
          A0DataType, AB1DataType, B0DataType, AB1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
          AElementOp,  BElementOp, CDEElementOp, GemmSpec,
          BlockSize, Scale_Block_M, Scale_Block_N, Scale_Block_K,
          MPerBlock, NPerBlock, KPerBlock,
          AK1, BK1,
          MPerXDL, NPerXDL,
          MXdlPerWave, NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, AK1, AK1, 0,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, BK1, BK1, 0,
          CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
          CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CDEShuffleBlockTransferScalarPerVectors,
          BlkGemmPipeSched,
          BlkGemmPipelineVer, A0DataType>;
// clang-format on

// The fused split-K kernels store their FP32 partials through CK's float global
// buffer stores, the only float global stores in the modules that include this
// header. On gfx950, where the fused kernel is compiled, make them device-scope
// stores (sc1: write-through, coherent across XCDs), so that the workgroup that
// reduces a tile can read the partials of the others without a release fence
// (buffer_wbl2, which writes back the whole L2).
#if defined(__gfx950__)
namespace ck {
constexpr auto kSplitKStoreCoherence =
    AmdBufferCoherenceEnum::DEVICE_NT0 != AmdBufferCoherenceEnum::DefaultCoherence
        ? AmdBufferCoherenceEnum::DEVICE_NT0
        : AmdBufferCoherenceEnum::GLC;
#define AITER_SPLITK_COHERENT_FLOAT_STORE(N)                                              \
    template <>                                                                           \
    __device__ void amd_buffer_store<float, N, AmdBufferCoherenceEnum::DefaultCoherence>( \
        const typename vector_type_maker<float, N>::type::type src_thread_data,           \
        float* p_dst_wave,                                                                \
        const index_t dst_thread_element_offset,                                          \
        const bool dst_thread_element_valid,                                              \
        const index_t dst_element_space_size)                                             \
    {                                                                                     \
        amd_buffer_store<float, N, kSplitKStoreCoherence>(src_thread_data,                \
                                                          p_dst_wave,                     \
                                                          dst_thread_element_offset,      \
                                                          dst_thread_element_valid,       \
                                                          dst_element_space_size);        \
    }
AITER_SPLITK_COHERENT_FLOAT_STORE(1)
AITER_SPLITK_COHERENT_FLOAT_STORE(2)
AITER_SPLITK_COHERENT_FLOAT_STORE(4)
AITER_SPLITK_COHERENT_FLOAT_STORE(8)
#undef AITER_SPLITK_COHERENT_FLOAT_STORE
} // namespace ck
#endif

// Kernel arguments of the fused deterministic split-K GEMM: the FP32-output
// instance's arguments (C = the [KBatch, M, N] workspace) plus the real output
// and the per-tile arrival counters.
template <typename Argument>
struct SplitKFusedArgument : Argument
{
    explicit SplitKFusedArgument(const Argument& arg) : Argument(arg) {}

    uint16_t* p_out;
    int64_t ld_out;
    uint32_t* p_counters;
};

// kernel_gemm_xdl_cshuffle_v3 with C stored (not added) to the workspace slice
// of K batch blockIdx.z; the last K batch of a tile to finish also reduces it
// (see gemm_a8w8_blockscale_splitk_reduce.cuh).
template <typename Gridwise,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          bool IsBF16,
          bool HasMainKBlockLoop,
          ck::index_t MinimumOccupancy,
          ck::TailNumber TailNum = ck::TailNumber::Full>
__global__ void __launch_bounds__(Gridwise::MaxBlockSize, MinimumOccupancy)
    kernel_gemm_blockscale_splitk_fused(SplitKFusedArgument<typename Gridwise::Argument> karg)
{
#if defined(__gfx950__)
    constexpr auto Set = ck::InMemoryDataOperationEnum::Set;
    if constexpr(Gridwise::template IsValidCompilationParameter<Set>())
    {
        __shared__ char p_shared[Gridwise::GetSharedMemoryNumberOfByte(ck::get_device_arch())];

        const auto block_2_ctile_map = typename Gridwise::Block2CTileMap{karg.M, karg.N, 4};
        const auto tile              = block_2_ctile_map.CalculateBottomIndex(
            ck::make_multi_index(static_cast<ck::index_t>(blockIdx.x)));
        const int m0 = __builtin_amdgcn_readfirstlane(tile[ck::Number<0>{}] * MPerBlock);
        const int n0 = __builtin_amdgcn_readfirstlane(tile[ck::Number<1>{}] * NPerBlock);
        if(m0 < 0 || n0 < 0 || m0 >= karg.M || n0 >= karg.N)
            return;

        const float* ws          = karg.p_c_grid;
        auto splitk_batch_offset = typename Gridwise::SplitKBatchOffset(karg);
        Gridwise::template Run<HasMainKBlockLoop, Set, TailNum>(
            karg.p_a_grid + splitk_batch_offset.a_k_split_offset,
            karg.p_b_grid + splitk_batch_offset.b_k_split_offset,
            karg.p_ds_grid,
            karg.p_c_grid + static_cast<ck::long_index_t>(blockIdx.z) * karg.M * karg.N,
            karg.p_a_scale_grid + splitk_batch_offset.scale_a_k_split_offset,
            karg.p_b_scale_grid + splitk_batch_offset.scale_b_k_split_offset,
            p_shared,
            karg,
            karg.a_element_op,
            karg.b_element_op,
            karg.c_element_op);

        aiter_blockscale::splitk_tile_fixup<MPerBlock, NPerBlock, Gridwise::MaxBlockSize, IsBF16>(
            ws,
            karg.p_out,
            karg.ld_out,
            karg.p_counters + blockIdx.x * aiter_blockscale::kSplitKCounterStride,
            karg.M,
            karg.N,
            m0,
            n0,
            karg.KBatch);
    }
#else
    ck::ignore = karg;
#endif
}

// Launches the fused split-K kernel for PartialsInstance (FP32 output),
// following the kernel selection of DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3.
// Returns false, launching nothing, when that selection has no kernel.
template <typename PartialsInstance,
          bool IsBF16,
          ck::BlockGemmPipelineVersion PipelineVer,
          ck::index_t MinOccupancy,
          ck::index_t KPerBlock,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock>
bool launch_splitk_fused(const typename PartialsInstance::Argument& arg,
                         torch::Tensor& Y,
                         uint32_t* counters,
                         hipStream_t stream)
{
    using Gridwise = typename PartialsInstance::GridwiseGemm64;
    using ck::TailNumber;

    const auto [gdx, gdy, gdz] = Gridwise::CalculateGridSize(arg.M, arg.N, arg.KBatch);
    SplitKFusedArgument<typename Gridwise::Argument> karg(arg);
    karg.p_out      = reinterpret_cast<uint16_t*>(Y.data_ptr());
    karg.ld_out     = Y.stride(0);
    karg.p_counters = counters;

    const ck::index_t k_grain = arg.KBatch * KPerBlock;
    const ck::index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

    const auto launch = [&](auto kernel) {
        kernel<<<dim3(gdx, gdy, gdz), dim3(Gridwise::MaxBlockSize), 0, stream>>>(karg);
        return true;
    };
    constexpr ck::index_t MO = MinOccupancy;
    if(Gridwise::CalculateHasMainKBlockLoop(K_split))
    {
        if constexpr(PipelineVer == ck::BlockGemmPipelineVersion::v1 ||
                     PipelineVer == ck::BlockGemmPipelineVersion::v3)
        {
            return launch(kernel_gemm_blockscale_splitk_fused<Gridwise,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              IsBF16,
                                                              true,
                                                              MO>);
        }
    }
    else if constexpr(PipelineVer == ck::BlockGemmPipelineVersion::v1)
    {
        const auto tail = Gridwise::CalculateKBlockLoopTailNum(K_split);
        if(tail == TailNumber::Full)
        {
            return launch(kernel_gemm_blockscale_splitk_fused<Gridwise,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              IsBF16,
                                                              false,
                                                              MO>);
        }
        if(tail == TailNumber::Odd)
        {
            return launch(kernel_gemm_blockscale_splitk_fused<Gridwise,
                                                              MPerBlock,
                                                              NPerBlock,
                                                              IsBF16,
                                                              false,
                                                              MO,
                                                              TailNumber::Odd>);
        }
    }
    return false;
}

template <typename DeviceInstance, typename DDataType, typename EOutType>
auto make_blockscale_argument(torch::Tensor& XQ,
                              torch::Tensor& WQ,
                              torch::Tensor& x_scale,
                              torch::Tensor& w_scale,
                              EOutType* p_e,
                              int KBatch)
{
    constexpr ck::index_t NumDTensor = DsDataType::Size();

    const int M = XQ.size(0);
    const int N = WQ.size(0);
    const int K = XQ.size(1);

    auto device_gemm = DeviceInstance{};
    auto argument    = device_gemm.MakeArgument(XQ.data_ptr(),
                                                WQ.data_ptr(),
                                                std::array<const void*, NumDTensor>{},
                                                p_e,
                                                M,
                                                N,
                                                K,
                                                XQ.stride(-2),
                                                WQ.stride(-2),
                                                std::array<ck::index_t, NumDTensor>{},
                                                N,
                                                reinterpret_cast<DDataType*>(x_scale.data_ptr()),
                                                reinterpret_cast<DDataType*>(w_scale.data_ptr()),
                                                AElementOp{},
                                                BElementOp{},
                                                CDEElementOp{});
    if(KBatch > 1)
    {
        device_gemm.SetKBatch(&argument, KBatch);
    }
    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");
    return std::make_pair(device_gemm, argument);
}

// On gfx950 with KBatch 4, 8 or 16, each batch stores an FP32 partial and the
// last batch to finish a tile adds the partials in batch order
// (bitwise-reproducible output). Otherwise, and when the fused path has no
// kernel for this shape (wave32, or pipelines other than v1/v3), GemmInstance
// runs as before, which for KBatch > 1 adds the partials atomically into Y.
template <typename DDataType,
          typename EDataType,
          typename GemmInstance,
          typename PartialsInstance                = void,
          ck::BlockGemmPipelineVersion PipelineVer = ck::BlockGemmPipelineVersion::v1,
          ck::index_t MinOccupancy                 = 1,
          ck::index_t KPerBlock                    = 0,
          ck::index_t MPerBlock                    = 0,
          ck::index_t NPerBlock                    = 0>
__forceinline__ torch::Tensor gemm_a8w8_blockscale_impl(torch::Tensor& XQ,
                                                        torch::Tensor& WQ,
                                                        torch::Tensor& x_scale,
                                                        torch::Tensor& w_scale,
                                                        torch::Tensor& Y,
                                                        int KBatch = 1)
{
    TORCH_CHECK(KBatch >= 1, "KBatch must be >= 1, got ", KBatch);

    const int M = XQ.size(0);
    const int N = WQ.size(0);
    auto stream = at::hip::getCurrentHIPStream();

    if constexpr(!std::is_void_v<PartialsInstance>)
    {
        const int64_t ws_elems = static_cast<int64_t>(KBatch) * M * N;
        if(aiter_blockscale::use_splitk_fused(KBatch) && ck::get_warp_size() == 64 &&
           ws_elems * sizeof(float) <= INT32_MAX)
        {
            TORCH_CHECK(Y.stride(-1) == 1, "split-K: Y inner dim must be contiguous");
            using Gridwise        = typename PartialsInstance::GridwiseGemm64;
            const int64_t tiles   = std::get<0>(Gridwise::CalculateGridSize(M, N, KBatch));
            const auto counters   = aiter_blockscale::splitk_counters(Y, tiles, stream);
            auto ws               = aiter_blockscale::splitk_workspace(Y, KBatch, M, N);
            auto [gemm, argument] = make_blockscale_argument<PartialsInstance, DDataType>(
                XQ, WQ, x_scale, w_scale, ws.data_ptr<float>(), KBatch);
            if(launch_splitk_fused<PartialsInstance,
                                   std::is_same_v<EDataType, BF16>,
                                   PipelineVer,
                                   MinOccupancy,
                                   KPerBlock,
                                   MPerBlock,
                                   NPerBlock>(argument, Y, counters.ptr, stream))
            {
                return Y;
            }
        }
    }

    auto [device_gemm, argument] = make_blockscale_argument<GemmInstance, DDataType>(
        XQ, WQ, x_scale, w_scale, reinterpret_cast<EDataType*>(Y.data_ptr()), KBatch);
    auto invoker = device_gemm.MakeInvoker();
    invoker.Run(argument, StreamConfig{stream});
    return Y;
}

#endif // USE_ROCM
