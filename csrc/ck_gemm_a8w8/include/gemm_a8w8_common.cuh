#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8 = int8_t;
using F8 = ck::f8_t;
using I32 = int;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

// using ADataType = I8;
// using BDataType = I8;
// using AccDataType = I32;
// using CShuffleDataType = F32;
// using ComputeDataType = I8;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using D1Layout = Col;
using D2Layout = Row;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using DsLayout2 = ck::Tuple<D0Layout, D1Layout, D2Layout>;
using ELayout = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

template <typename AccDataType, typename DDataType, typename EDataType>
struct RowwiseScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, AccDataType, DDataType, DDataType>(
        EDataType &e, const AccDataType &c, const DDataType &d0, const DDataType &d1) const
    {
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<EDataType>(x0_f);
    }
};

template <typename AccDataType, typename DDataType, typename EDataType>
struct MultiplyMultiplyAdd
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;

    template <>
    __host__ __device__ constexpr void operator()<EDataType, AccDataType, DDataType, DDataType, EDataType>(
        EDataType &e, const AccDataType &c, const DDataType &d0, const DDataType &d1, const EDataType &d2) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1) + ck::type_convert<F32>(d2);

        e = ck::type_convert<EDataType>(x0_f);
    }
};

// using CDEElementOp = RowwiseScale;
// using CDEElementOp2 = MultiplyMultiplyAdd;

template <typename DDataType>
using DsDataType = ck::Tuple<DDataType, DDataType>;

template <typename DDataType, typename D2DataType>
using DsDataType2 = ck::Tuple<DDataType, DDataType, D2DataType>;

#if 0
template <typename DDataType, typename EDataType=DDataType>
using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
// clang-format off
///######|  ALayout|  BLayout| DsLayout| ELayout|      AData|      BData|               DsData|                 EData|     AccData|         CShuffle|           A|           B|          CDE|           GEMM| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
///######|         |         |         |        |       Type|       Type|                 Type|                  Type|        Type|         DataType| Elementwise| Elementwise|  Elementwise| Spacialization|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
///######|         |         |         |        |           |           |                     |                      |            |                 |   Operation|   Operation|    Operation|               |      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
///######|         |         |         |        |           |           |                     |                      |            |                 |            |            |             |               |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |    S<C, D0, D1>|
///###### RRR
///      <      Row,      Row, DsLayout, ELayout, ADataType, BDataType, DsDataType<DDataType>, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,   4,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<16, 16, 1>,    S<0, 2, 1>,     S<0, 2, 1>,             1,               8,              4,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;
///###### RCR
         <      Row,      Col, DsLayout, ELayout, ADataType, BDataType, DsDataType<DDataType>, EDataType, AccDataType, CShuffleDataType,  AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,   256,   128,    64,  16,  16,  32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,     S<4, 64, 1>,    S<1, 0, 2>,     S<1, 0, 2>,             2,              16,             16,          0,          1,           1,               S<1, 32, 1, 8>,      S<8, 8, 1>,  ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1, I8>;
// clang-format on
#endif

template <
    typename ABDataType,
    typename AccDataType,
    typename DDataType, 
    typename EDataType,
    typename CDEElementOp,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    auto GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding>
using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        std::conditional_t<ck::is_same_v<CDEElementOp, RowwiseScale<AccDataType, DDataType, EDataType>>, DsLayout, DsLayout2>,
        ELayout,
        ABDataType,
        ABDataType,
        std::conditional_t<ck::is_same_v<CDEElementOp, RowwiseScale<AccDataType, DDataType, EDataType>>, DsDataType<DDataType>, DsDataType2<DDataType, EDataType>>,
        EDataType,
        AccDataType,
        AccDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        BLOCK_SIZE,                       // Block Size
        MBLOCK,                           // M per Block
        NBLOCK,                           // N per Block
        KBLOCK,                           // K per Block
        KBLOCK / ABLOCK_TRANSFER{}.At(0), // AK1
        16,                               // BK1
        WAVE_TILE_M,                      // M per Xdl
        WAVE_TILE_N,                      // N per Xdl
        WAVE_MAP_M,                       // Mxdl per Wave
        WAVE_MAP_N,                       // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        KBLOCK / ABLOCK_TRANSFER{}.At(0),
        KBLOCK / ABLOCK_TRANSFER{}.At(0),
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ABDataType>;


template <typename ABDataType, typename DDataType, typename EDataType, bool has_bias,
          typename DeviceGemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_rowwise_impl(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{
    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    int StrideA = K;
    int StrideB = K;
    int StrideE = N;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(XQ));
    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    using AccDataType = std::conditional_t<ck::is_same_v<ABDataType, I8>, I32, F32>;
    if constexpr(has_bias)
    {
        auto cde_element_op = MultiplyMultiplyAdd<AccDataType, DDataType, EDataType>{};

        auto argument = device_gemm.MakeArgument(
            reinterpret_cast<ABDataType *>(XQ.data_ptr()),
            reinterpret_cast<ABDataType *>(WQ.data_ptr()),
            std::array<const void *, 3>{reinterpret_cast<DDataType *>(w_scale.data_ptr()),
                                            reinterpret_cast<DDataType *>(x_scale.data_ptr()),
                                            reinterpret_cast<EDataType *>(bias.value().data_ptr())},
            reinterpret_cast<EDataType *>(Y.data_ptr()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            std::array<ck::index_t, 3>{0, 0, 0},
            StrideE,
            KBatch,
            a_element_op,
            b_element_op,
            cde_element_op);
        TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

        invoker.Run(argument, StreamConfig{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()});
    }
    else
    {
        auto cde_element_op = RowwiseScale<AccDataType, DDataType, EDataType>{};

        auto argument = device_gemm.MakeArgument(
            reinterpret_cast<ABDataType *>(XQ.data_ptr()),
            reinterpret_cast<ABDataType *>(WQ.data_ptr()),
            std::array<const void *, 2>{reinterpret_cast<DDataType *>(w_scale.data_ptr()),
                                            reinterpret_cast<DDataType *>(x_scale.data_ptr())},
            reinterpret_cast<EDataType *>(Y.data_ptr()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            std::array<ck::index_t, 2>{0, 0},
            StrideE,
            KBatch,
            a_element_op,
            b_element_op,
            cde_element_op);
        TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

        invoker.Run(argument, StreamConfig{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()});
    }
    return Y;
}

#endif // USE_ROCM
