// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free pybind entry point for a4w4 blockscale GEMM tuning.
// See gemm_a4w4_blockscale_pybind.cu for the rationale.
#include "aiter_stream.h"
#include "gemm_a4w4_blockscale.h"
#include "rocm_ops.hpp"

namespace {

void gemm_a4w4_blockscale_tune(aiter_tensor_t& XQ,
                               aiter_tensor_t& WQ,
                               aiter_tensor_t& x_scale,
                               aiter_tensor_t& w_scale,
                               aiter_tensor_t& Y,
                               int kernelId,
                               int splitK)
{
    AITER_CHECK(XQ.dtype_ == WQ.dtype_, "Weights and activations should have the same dtype!");
    AITER_CHECK(x_scale.dtype_ == w_scale.dtype_, "Scales should have the same dtype!");
    AITER_CHECK(Y.dtype_ == AITER_DTYPE_fp16 || Y.dtype_ == AITER_DTYPE_bf16,
                "Unsupported output dtype!");

    aiter::gemm_a4w4_blockscale_tune(
        XQ, WQ, x_scale, w_scale, Y, kernelId, splitK, aiter::getCurrentHIPStream());
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    GEMM_A4W4_BLOCKSCALE_TUNE_PYBIND;
}
