// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AIESW-32176: shared types + device-op instance for the gfx1151 CK W4A16
// b_scale GEMM. Mirrors the layout of csrc/ck_gemm_a4w4_blockscale/include/
// (typedefs hoisted out of the dispatcher .cu).
#pragma once

#ifdef USE_ROCM
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#endif

#include <torch/all.h>
#include <torch/extension.h>
#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_b_scale.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// aiter-style type aliases (match csrc/ck_gemm_a4w4_blockscale/include/...)
using F16 = ck::half_t;
using B16 = ck::bhalf_t;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace ck_w4a16 {

// Weight + accumulator dtypes are fixed; activation / scale / output dtype is
// templated (T = F16 or B16) to mirror the F16/B16 dispatch pattern in
// csrc/ck_gemm_a4w4_blockscale/.
using BDataType   = ck::pk_i4_t;
using AccDataType = float;

using ALayout = Row;
using BLayout = Col;
using CLayout = Row;

inline constexpr auto GemmDefault =
    ck::tensor_operation::device::GemmSpecialization::Default;
inline constexpr bool PermuteA = false;
inline constexpr bool PermuteB = true;
inline constexpr ck::index_t Scale_Block_N = 1;
inline constexpr ck::index_t Scale_Block_K = 128;

// EXP1_FINAL config from gfx1151 sweep (30.0 TFLOPS verified at gate_up_proj
// M=3968 N=19456 K=2560). Holds 28-31 TFLOPS uniformly across M=256-16384 on
// the same column. Same kernel handles all four Qwen3-4B prefill linear
// columns at runtime.
inline constexpr ck::index_t KPerBlock = 32;

// Templated device-op instance — T is fp16 or bf16 (activation = scale =
// shuffle = output dtype). Tile / pipeline params are dtype-independent.
//
// clang-format off
template <typename T>
using DeviceGemmInstance =
    ck::tensor_operation::device::DeviceGemm_BScale_Wmma_CShuffleV3<
        ALayout,   BLayout,  CLayout,
        T,         BDataType, T, T, AccDataType, T,
        PassThrough, PassThrough, PassThrough, GemmDefault,
        256, Scale_Block_N, Scale_Block_K,
        128, 128,
        KPerBlock, 8, 8,
        16,  16,
        4,    2,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 1,
        S<4, 64, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
        2, 8, 8, 1,
        1, 1, S<1, 32, 1, 8>, 8,
        ck::BlockGemmPipelineScheduler::Interwave, ck::BlockGemmPipelineVersion::v1,
        T, T, PermuteA, PermuteB>;
// clang-format on

}  // namespace ck_w4a16
