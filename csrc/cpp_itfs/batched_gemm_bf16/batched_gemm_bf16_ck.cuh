#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free port of csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh.
//
// The CK kernel configuration (DeviceGemmHelper) is byte-for-byte identical to
// the upstream torch build. The only change is the launch wrapper: instead of
// torch::Tensor, batched_gemm_bf16_impl() consumes a POD batched_gemm_bf16_args
// (raw device pointers + caller-supplied stream/device), and uses
// HipDeviceGuard / AITER_CHECK in place of the ATen device guard / TORCH_CHECK.
// No <torch/*> or <ATen/*> headers are pulled in.
#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <array>

#include "aiter_hip_common.h" // HipDeviceGuard, AITER_CHECK
#include "batched_gemm_bf16.h" // batched_gemm_bf16_args

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16 = ck::bhalf_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using ComputeDataType  = BF16;
using EDataType        = BF16;

using ALayout  = Row;
using BLayout  = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;
using DsDataType   = ck::Tuple<>;

template <int BLOCK_SIZE,
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
          auto GEMM_SPEC = ck::tensor_operation::device::GemmSpecialization::MNKPadding>
using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<ALayout,
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
                                                                         BLOCK_SIZE,
                                                                         MBLOCK,
                                                                         NBLOCK,
                                                                         KBLOCK,
                                                                         KBLOCK / ABLOCK_TRANSFER{}.At(0),
                                                                         KBLOCK / BBLOCK_TRANSFER{}.At(0),
                                                                         WAVE_TILE_M,
                                                                         WAVE_TILE_N,
                                                                         WAVE_MAP_M,
                                                                         WAVE_MAP_N,
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
void batched_gemm_bf16_impl(const batched_gemm_bf16_args& a)
{
    const int M = a.M;
    const int N = a.N;
    const int K = a.K;
    const int B = a.B;

    const int StrideA = K;
    const int StrideB = K;
    const int StrideE = N;

    const int BatchStrideA = M * K;
    const int BatchStrideB = N * K;
    const int BatchStrideE = M * N;

    HipDeviceGuard device_guard(a.device_id);
    auto device_gemm = DeviceGemmInstance{};
    auto invoker     = device_gemm.MakeInvoker();

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DeviceGemmInstance::NumDTensor;

    auto argument = device_gemm.MakeArgument(
        reinterpret_cast<const ADataType*>(a.a_ptr),
        reinterpret_cast<const BDataType*>(a.b_ptr),
        std::array<const void*, NumDTensor>{},
        reinterpret_cast<EDataType*>(a.e_ptr),
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

    AITER_CHECK(device_gemm.IsSupportedArgument(argument),
                "batched_gemm_bf16: unsupported GEMM argument for shape B=",
                B,
                " M=",
                M,
                " N=",
                N,
                " K=",
                K);

    invoker.Run(argument, StreamConfig{a.stream});
}

#endif // USE_ROCM
