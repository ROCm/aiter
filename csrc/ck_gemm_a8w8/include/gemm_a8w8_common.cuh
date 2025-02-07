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
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using I8 = int8_t;
using I32 = int;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using FP8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType = FP8;
using BDataType = FP8;
using AccDataType = F32;
using CShuffleDataType = F32;
using ComputeDataType = FP8;

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

struct RowwiseScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, AccDataType, F16, F16>(
        F16 &e, const AccDataType &c, const F16 &d0, const F16 &d1) const
    {
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<F16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, AccDataType, B16, B16>(
        B16 &e, const AccDataType &c, const B16 &d0, const B16 &d1) const
    {
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<B16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<F16, AccDataType, F32, F32>(
        F16 &e, const AccDataType &c, const F32 &d0, const F32 &d1) const
    {
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<F16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<B16, AccDataType, F32, F32>(
        B16 &e, const AccDataType &c, const F32 &d0, const F32 &d1) const
    {
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<B16>(x0_f);
    }
};

struct MultiplyMultiplyAdd
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E &e, const C &c, const D0 &d0, const D1 &d1, const D2 &d2) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float, ck::half_t>(
        ck::half_t &e, const int &c, const float &d0, const float &d1, const ck::half_t &d2) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1) + ck::type_convert<F32>(d2);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float, ck::bhalf_t>(
        ck::bhalf_t &e, const int &c, const float &d0, const float &d1, const ck::bhalf_t &d2) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1) + ck::type_convert<F32>(d2);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, ck::half_t, ck::half_t, ck::half_t>(
        ck::half_t &e, const int &c, const ck::half_t &d0, const ck::half_t &d1, const ck::half_t &d2) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1) + ck::type_convert<F32>(d2);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

        template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float, F16>(F16 &e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1,
                                                                            const F16 &d2) const
    {
        const float x0_f = c * d0 * d1 + ck::type_convert<F32>(d2);

        e = ck::type_convert<F16>(x0_f);
    }
    template <>
    __host__ __device__ constexpr void operator()<F16, float, F16, F16, F16>(F16 &e,
                                                                            const float& c,
                                                                            const F16& d0,
                                                                            const F16& d1,
                                                                            const F16 &d2) const
    {
        const float x0_f = c * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1) + ck::type_convert<F32>(d2);
        e = ck::type_convert<F16>(x0_f);
    }
    
    template <>
    __host__ __device__ constexpr void operator()<B16, float, B16, B16, B16>(B16 &e,
                                                                            const float& c,
                                                                            const B16& d0,
                                                                            const B16& d1,
                                                                            const B16 &d2) const
    {
        const float x0_f = c * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1) + ck::type_convert<F32>(d2);
        e = ck::type_convert<B16>(x0_f);
    }
    template <>
    __host__ __device__ constexpr void operator()<B16, float, float, float, B16>(B16 &e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1,
                                                                            const B16 &d2) const
    {
        const float x0_f = c * d0 * d1 + ck::type_convert<F32>(d2);
        e = ck::type_convert<B16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, ck::bhalf_t, ck::bhalf_t, ck::bhalf_t>(
        ck::bhalf_t &e, const int &c, const ck::bhalf_t &d0, const ck::bhalf_t &d1, const ck::bhalf_t &d2) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1) + ck::type_convert<F32>(d2);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }
};

using CDEElementOp = RowwiseScale;
using CDEElementOp2 = MultiplyMultiplyAdd;

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
    typename DDataType, typename EDataType,
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
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        DsDataType<DDataType>,
        EDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        BLOCK_SIZE,                      // Block Size
        MBLOCK,                          // M per Block
        NBLOCK,                          // N per Block
        KBLOCK,                          // K per Block
        KBLOCK / ABLOCK_TRANSFER{}.At(0), // AK1
        16,                              // BK1
        WAVE_TILE_M,                     // M per Xdl
        WAVE_TILE_N,                     // N per Xdl
        WAVE_MAP_M,                      // Mxdl per Wave
        WAVE_MAP_N,                      // Nxdl per Wave
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
        ComputeDataType>;

template <
    typename DDataType, typename EDataType,
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
using DeviceGemmHelperMMA =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout2,
        ELayout,
        ADataType,
        BDataType,
        DsDataType2<DDataType, EDataType>,
        EDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        BElementOp,
        CDEElementOp2,
        GEMM_SPEC,
        BLOCK_SIZE,                      // Block Size
        MBLOCK,                          // M per Block
        NBLOCK,                          // N per Block
        KBLOCK,                          // K per Block
        KBLOCK / ABLOCK_TRANSFER{}.At(0), // AK1
        16,                              // BK1
        WAVE_TILE_M,                     // M per Xdl
        WAVE_TILE_N,                     // N per Xdl
        WAVE_MAP_M,                      // Mxdl per Wave
        WAVE_MAP_N,                      // Nxdl per Wave
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
        ComputeDataType>;

template <typename DDataType, typename EDataType,
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
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DeviceGemmInstance::NumDTensor;

    auto argument = device_gemm.MakeArgument(
        reinterpret_cast<ADataType *>(XQ.data_ptr()),
        reinterpret_cast<BDataType *>(WQ.data_ptr()),
        std::array<const void *, NumDTensor>{
            reinterpret_cast<DDataType *>(w_scale.data_ptr()),
            reinterpret_cast<DDataType *>(x_scale.data_ptr())},
        reinterpret_cast<EDataType *>(Y.data_ptr()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        std::array<ck::index_t, NumDTensor>{0, 0},
        StrideE,
        KBatch,
        a_element_op,
        b_element_op,
        cde_element_op);
    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

template <typename DDataType, typename EDataType,
          typename DeviceGemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_mma_impl(
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
    auto cde_element_op2 = CDEElementOp2{};

    constexpr ck::index_t NumDTensor = DeviceGemmInstance::NumDTensor;

    auto argument = device_gemm.MakeArgument(
        reinterpret_cast<ADataType *>(XQ.data_ptr()),
        reinterpret_cast<BDataType *>(WQ.data_ptr()),
        std::array<const void *, NumDTensor>{
            reinterpret_cast<DDataType *>(w_scale.data_ptr()),
            reinterpret_cast<DDataType *>(x_scale.data_ptr()),
            reinterpret_cast<EDataType *>(bias.value().data_ptr())},
        reinterpret_cast<EDataType *>(Y.data_ptr()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        std::array<ck::index_t, NumDTensor>{0, 0, 0},
        StrideE,
        KBatch,
        a_element_op,
        b_element_op,
        cde_element_op2);
    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

#endif // USE_ROCM
