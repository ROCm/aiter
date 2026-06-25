// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Sole torch-aware entry point for a4w4 blockscale GEMM tuning.
// See gemm_a4w4_blockscale_pybind.cu for the rationale.
#include "rocm_ops.hpp"
#include "gemm_a4w4_blockscale.h"
#include "pybind_aiter_compat.h"

#include <ATen/hip/HIPContext.h>

namespace {

torch::Tensor gemm_a4w4_blockscale_tune(torch::Tensor& XQ,
                                         torch::Tensor& WQ,
                                         torch::Tensor& x_scale,
                                         torch::Tensor& w_scale,
                                         torch::Tensor& Y,
                                         int kernelId,
                                         int splitK)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
    TORCH_CHECK(Y.scalar_type() == at::ScalarType::Half ||
                Y.scalar_type() == at::ScalarType::BFloat16,
                "Unsupported output dtype!");

    auto xq_at = aiter_pybind::make_aiter_tensor(XQ);
    auto wq_at = aiter_pybind::make_aiter_tensor(WQ);
    auto xs_at = aiter_pybind::make_aiter_tensor(x_scale);
    auto ws_at = aiter_pybind::make_aiter_tensor(w_scale);
    auto y_at  = aiter_pybind::make_aiter_tensor(Y);

    aiter::gemm_a4w4_blockscale_tune(
        xq_at, wq_at, xs_at, ws_at, y_at, kernelId, splitK,
        at::hip::getCurrentHIPStream());
    return Y;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A4W4_BLOCKSCALE_TUNE_PYBIND;
}
