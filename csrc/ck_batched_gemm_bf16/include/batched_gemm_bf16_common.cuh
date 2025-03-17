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
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
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

using BF16 = ck::bhalf_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType = BF16;
using BDataType = BF16;
using AccDataType = F32;
using CShuffleDataType = F32;
using ComputeDataType = BF16;
using EDataType = BF16;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using DsLayout = ck::Tuple<>;
using ELayout = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = PassThrough;

using DsDataType = ck::Tuple<>;

template <
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
    ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        DsDataType,
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
        KBLOCK / ABLOCK_TRANSFER{}.At(0),// AK1
        KBLOCK / BBLOCK_TRANSFER{}.At(0),// AK1
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
        KBLOCK / BBLOCK_TRANSFER{}.At(0),
        KBLOCK / BBLOCK_TRANSFER{}.At(0),
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ComputeDataType>;

template <typename DeviceGemmInstance>
__forceinline__ torch::Tensor batched_gemm_bf16_impl(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{
    int B = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = XQ.size(2);

    int StrideA = K;
    int StrideB = K;
    int StrideE = N;

    int BatchStrideA = M * K;
    int BatchStrideB = N * K;
    int BatchStrideE = M * N;

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
        std::array<const void *, NumDTensor>{},
        reinterpret_cast<EDataType *>(Y.data_ptr()),
        M,
        N,
        K,
        B,
        StrideA,
        StrideB,
        std::array<ck::index_t, NumDTensor>{},
        StrideE,
        BatchStrideA,
        BatchStrideB,
        std::array<ck::index_t, NumDTensor>{},
        BatchStrideE,
        a_element_op,
        b_element_op,
        cde_element_op);

    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");

    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

#endif // USE_ROCM
