#pragma once
#include <limits>
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
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

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

    // Derive every stride from the TENSOR, not from the logical shape.
    //
    // The original six expressions (StrideA = K, StrideB = K, StrideE = N,
    // BatchStrideA = M*K, BatchStrideB = N*K, BatchStrideE = M*N) assume all three
    // operands are contiguous in their logical shape. CK takes these as ordinary
    // runtime scalars and supports arbitrary values, so the kernel was never the
    // limitation -- the wrapper simply never asked the caller what it allocated.
    //
    // Two measured consequences of the old form:
    //   * a non-contiguous Y was written AS IF contiguous -- no error, correct
    //     arithmetic, wrong placement, and `return Y` handed back a view whose
    //     strides described a layout the memory did not have;
    //   * WQ in a (B, K, N) orientation made BatchStrideB = N*K read ~4x past the
    //     end of the tensor, surfacing as a HIP illegal memory access rather than
    //     a layout error.
    //
    // ALayout=Row  -> A is M x K, leading dim is the stride between M rows
    // BLayout=Col  -> B is K x N column-major, leading dim is the stride between N columns
    // ELayout=Row  -> E is M x N, leading dim is the stride between M rows
    // In every case the INNERMOST dim must be unit-stride; anything else is a
    // layout these descriptors cannot express and is rejected loudly below.
    TORCH_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3,
                "batched_gemm_bf16: XQ, WQ and Y must all be 3-D, got ",
                XQ.dim(), ", ", WQ.dim(), ", ", Y.dim());
    TORCH_CHECK(WQ.size(0) == B && Y.size(0) == B,
                "batched_gemm_bf16: batch mismatch -- XQ ", B, ", WQ ", WQ.size(0),
                ", Y ", Y.size(0));
    TORCH_CHECK(WQ.size(2) == K,
                "batched_gemm_bf16: contraction mismatch -- XQ K=", K, ", WQ K=",
                WQ.size(2), ". WQ must be (batch, N, K); a (batch, K, N) operand is "
                "the orientation that used to read past the end of the tensor.");
    TORCH_CHECK(Y.size(1) == M && Y.size(2) == N,
                "batched_gemm_bf16: output shape mismatch -- expected (", B, ", ", M,
                ", ", N, "), got (", Y.size(0), ", ", Y.size(1), ", ", Y.size(2), ")");
    TORCH_CHECK(XQ.stride(2) == 1,
                "batched_gemm_bf16: XQ must be unit-stride in its innermost (K) dim, "
                "got stride ", XQ.stride(2), ". CK's row-major A descriptor cannot "
                "express this layout.");
    TORCH_CHECK(WQ.stride(2) == 1,
                "batched_gemm_bf16: WQ must be unit-stride in its innermost (K) dim, "
                "got stride ", WQ.stride(2), ". CK's column-major B descriptor cannot "
                "express this layout.");
    TORCH_CHECK(Y.stride(2) == 1,
                "batched_gemm_bf16: Y must be unit-stride in its innermost (N) dim, "
                "got stride ", Y.stride(2), ". CK's row-major E descriptor cannot "
                "express this layout -- refusing rather than mis-placing the result.");

    // CK takes these as int; a stride that does not fit is a silent truncation.
    auto fits_int = [](int64_t v) {
        return v >= 0 && v <= static_cast<int64_t>(std::numeric_limits<int>::max());
    };
    TORCH_CHECK(fits_int(XQ.stride(0)) && fits_int(XQ.stride(1)) &&
                fits_int(WQ.stride(0)) && fits_int(WQ.stride(1)) &&
                fits_int(Y.stride(0)) && fits_int(Y.stride(1)),
                "batched_gemm_bf16: a stride does not fit in int32 and would be "
                "silently truncated");

    int StrideA = static_cast<int>(XQ.stride(1));
    int StrideB = static_cast<int>(WQ.stride(1));
    int StrideE = static_cast<int>(Y.stride(1));

    int BatchStrideA = static_cast<int>(XQ.stride(0));
    int BatchStrideB = static_cast<int>(WQ.stride(0));
    int BatchStrideE = static_cast<int>(Y.stride(0));

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(XQ));
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

    invoker.Run(argument, StreamConfig{at::hip::getCurrentHIPStream()});
    return Y;
}

#endif // USE_ROCM
