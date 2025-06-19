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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
using F32  = float;
using B16  = ck::bhalf_t;
using Row  = ck::tensor_layout::gemm::RowMajor;
using Col  = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using B0DataType       = FP8;
using AccDataType      = F32;
using CShuffleDataType = F32;
using D0DataType       = F32;
using D1DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType>;
using EDataType        = F16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout  = Row;

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const;

    template <>
    __host__ __device__ constexpr void operator()<F16, float, float, float>(F16& e,
                                                                            const float& c,
                                                                            const float& d0,
                                                                            const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<F16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<BF16, float, float, float>(BF16& e,
                                                                             const float& c,
                                                                             const float& d0,
                                                                             const float& d1) const
    {
        const float x0_f = c * d0 * d1;

        e = ck::type_convert<BF16>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, int, float, float>(
        ck::half_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::half_t>(x0_f);
    }

    template <>
    __host__ __device__ constexpr void operator()<ck::bhalf_t, int, float, float>(
        ck::bhalf_t& e, const int& c, const float& d0, const float& d1) const
    {
        const float x0_f =
            ck::type_convert<float>(c) * ck::type_convert<float>(d0) * ck::type_convert<float>(d1);

        e = ck::type_convert<ck::bhalf_t>(x0_f);
    }
};
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = MultiplyMultiply;

template <typename AccDataType,
          typename EDataType,
          ck::index_t BlockSize,
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
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v3,
          auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default>
using DeviceGemmHelperF8Flatmm =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3_BPreshuffle
    // clang-format off
         <A0Layout, B0Layout, DsLayout, ELayout,
          A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType, 
          AElementOp,  BElementOp, CDEElementOp, GemmSpec, BlockSize,
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

template <typename DDataType, typename EDataType, typename DeviceGemmInstance>
__forceinline__ torch::Tensor gemm_a8w8_bpreshuffle_impl(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y)
{
    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    int StrideA = XQ.stride(-2);
    int StrideB = WQ.stride(-2);
    int StrideE = N;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();
    constexpr auto I0 = ck::Number<0>{};

    // do GEMM
    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();
    auto argument = device_gemm.MakeArgument(XQ.data_ptr(),
                                             WQ.data_ptr(),
                                             std::array<const void *, NumDTensor>{
                                                reinterpret_cast<DDataType *>(w_scale.data_ptr()),
                                                reinterpret_cast<DDataType *>(x_scale.data_ptr())},
                                             reinterpret_cast<EDataType *>(Y.data_ptr()),
                                             M,
                                             N,
                                             K,
                                             StrideA,
                                             StrideB,
                                             std::array<ck::index_t, NumDTensor>{I0, I0},
                                             StrideE,
                                             1,
                                             a_element_op,
                                             b_element_op,
                                             cde_element_op);

    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

#endif // USE_ROCM
