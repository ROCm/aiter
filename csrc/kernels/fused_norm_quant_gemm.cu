// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include "rmsnorm_quant.h"
#include "hipbsolgemm.cuh"

namespace aiter {

torch::Tensor fused_rmsnorm_quant_gemm(
    const torch::Tensor& input,       // [M, K] BF16
    const torch::Tensor& weight_fp8,  // [N, K] FP8
    const torch::Tensor& norm_w,      // [K] BF16
    double eps,
    const torch::Tensor& scale_a,     // [1] or [1,1] FP32
    const torch::Tensor& scale_w,     // [1] or [1,1] FP32
    torch::Tensor& fp8_workspace,     // [M, K] FP8 pre-allocated
    int solution_index
) {
    // Phase 1: fused RMSNorm + FP8 quantization into pre-allocated workspace.
    // Uses the existing AITER CK kernel on the current HIP stream.
    // rmsnorm_quant signature: (out, input, scale, weight, epsilon)
    // The non-const cast is needed because rmsnorm_quant takes non-const refs.
    auto input_nc = const_cast<torch::Tensor&>(input);
    auto scale_a_nc = const_cast<torch::Tensor&>(scale_a);
    auto norm_w_nc = const_cast<torch::Tensor&>(norm_w);
    rmsnorm_quant(fp8_workspace, input_nc, scale_a_nc, norm_w_nc, eps);

    // Phase 2: hipBLASLt GEMM on the same HIP stream.
    // hipb_mm computes: fp8_workspace @ weight_fp8^T with scaled output.
    auto weight_t = weight_fp8.t();
    auto sa = scale_a.to(torch::kFloat32).reshape({1, 1});
    auto sw = scale_w.to(torch::kFloat32).reshape({1, 1});

    return hipb_mm(
        fp8_workspace,
        weight_t,
        solution_index,
        /*bias=*/std::nullopt,
        /*out_dtype=*/at::kBFloat16,
        /*scaleA=*/sa,
        /*scaleB=*/sw);
}

} // namespace aiter
