#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using MFMA = ck::tensor_layout::gemm::MFMA;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using F4PK = ck::f4x2_pk_t;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using F32 = float;
using E8M0PK = int32_t;

using ADataType = F4PK;
using BDataType = F4PK;
using XPackedDataType = E8M0PK;

using AccDataType = float;

using ALayout = Row;
// using BLayout = MFMA;
using CLayout = Row;

using AElementOp = PassThrough; // elementwise transformation for A matrix
using BElementOp = PassThrough; // elementwise transformation for B matrix
using CElementOp = PassThrough; // elementwise transformation for C matrix

constexpr ck::index_t DataPackedSize = 2;               // Packed representation of data
constexpr ck::index_t ScaleBlockSize = 32;              // scaling block size
constexpr ck::index_t KPerBlock = 256 / DataPackedSize; // 256 f4 = 128 fp4x2

static constexpr auto Intrawave = ck::BlockGemmPipelineScheduler::Intrawave;
static constexpr auto Interwave = ck::BlockGemmPipelineScheduler::Interwave;

template <typename BLayout,
          typename CDataType,
          ck::index_t BlockSize,
          ck::index_t MPerBlock, ck::index_t NPerBlock, ck::index_t KPerBlock,
          ck::index_t AK1, ck::index_t BK1,
          ck::index_t MPerXDL, ck::index_t NPerXDL,
          ck::index_t MXdlPerWave, ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          ck::index_t CShuffleMXdlPerWavePerShuffle,
          ck::index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock,
          ck::BlockGemmPipelineScheduler BlkGemmPipeSched = Intrawave,
          ck::BlockGemmPipelineVersion BlkGemmPipelineVer = ck::BlockGemmPipelineVersion::v3,
          auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default>
using DeviceGemmHelperF4BlockScale = ck::tensor_operation::device::DeviceGemmMX_Xdl_CShuffleV3
    // clang-format off
         <ALayout, BLayout, CLayout,
          ADataType, XPackedDataType, BDataType, XPackedDataType, CDataType, AccDataType, CDataType,
          AElementOp, BElementOp, CElementOp, GemmSpec,
          ScaleBlockSize, BlockSize,
          MPerBlock, NPerBlock, KPerBlock,
          AK1, BK1,
          MPerXDL, NPerXDL,
          MXdlPerWave, NXdlPerWave,
          ABlockTransferThreadClusterLengths_AK0_M_AK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, AK1, AK1,
          true,
          BBlockTransferThreadClusterLengths_BK0_N_BK1,
          S<1, 0, 2>, S<1, 0, 2>,
          2, BK1, BK1,
          true,
          CShuffleMXdlPerWavePerShuffle,
          CShuffleNXdlPerWavePerShuffle,
          CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          CShuffleBlockTransferScalarPerVector_NPerBlock,
          BlkGemmPipeSched,
          BlkGemmPipelineVer,
          ADataType, BDataType>;
// clang-format on

template <typename CDataType, typename DeviceGemmInstance>
__forceinline__ torch::Tensor gemm_a4w4_blockscale_impl(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &a_scale,
    torch::Tensor &b_scale,
    torch::Tensor &C,
    int splitK)
{
    int M = A.size(0);
    int N = B.size(0);
    int K = A.size(1) * 2; // always fp4_x2

    // TODO: support batch gemm
    int KBatch = std::pow(2, splitK);

    int StrideA = A.stride(-2) * 2; // always fp4_x2
    int StrideB = B.stride(-2) * 2; // always fp4_x2
    int StrideC = C.stride(-2);
    int Scale_Stride_A = a_scale.stride(-2);
    int Scale_Stride_B = b_scale.stride(-2);

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};


    // do GEMM
    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();
    auto argument = device_gemm.MakeArgument(static_cast<ADataType*>(A.data_ptr()),
                                             static_cast<XPackedDataType*>(a_scale.data_ptr()),
                                             static_cast<BDataType*>(B.data_ptr()),
                                             static_cast<XPackedDataType*>(b_scale.data_ptr()),
                                             static_cast<CDataType*>(C.data_ptr()),
                                             M,
                                             N,
                                             K,
                                             StrideA,
                                             Scale_Stride_A,
                                             StrideB,
                                             Scale_Stride_B,
                                             StrideC,
                                             KBatch,
                                             a_element_op,
                                             b_element_op,
                                             c_element_op);

    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return C;
}

#endif // USE_ROCM
