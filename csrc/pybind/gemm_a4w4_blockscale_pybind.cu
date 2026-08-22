// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free pybind entry point for a4w4 blockscale GEMM.  Torch lives only in
// the Python layer: @compile_ops(..., develop=True) makes core.py convert each
// torch.Tensor argument to a pybind aiter_tensor_t via torch_to_aiter_pybind()
// and push the caller's stream through _set_current_hip_stream(), so this TU
// includes no torch or ATen header.
//
// Writes into Out in place and returns void; the Python wrapper in
// aiter/ops/gemm_op_a4w4.py returns Out so the public API still yields a
// torch.Tensor.
#include "aiter_stream.h"
#include "gemm_a4w4_blockscale.h"
#include "rocm_ops.hpp"

namespace {

void gemm_a4w4_blockscale(aiter_tensor_t& XQ,
                          aiter_tensor_t& WQ,
                          aiter_tensor_t& x_scale,
                          aiter_tensor_t& w_scale,
                          aiter_tensor_t& Y,
                          int splitK,
                          std::string kernelName)
{
    AITER_CHECK(XQ.dtype_ == WQ.dtype_, "Weights and activations should have the same dtype!");
    AITER_CHECK(x_scale.dtype_ == w_scale.dtype_, "Scales should have the same dtype!");
    AITER_CHECK(Y.dtype_ == AITER_DTYPE_fp16 || Y.dtype_ == AITER_DTYPE_bf16,
                "Unsupported output dtype!");

    aiter::gemm_a4w4_blockscale(
        XQ, WQ, x_scale, w_scale, Y, splitK, aiter::getCurrentHIPStream(), kernelName);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    GEMM_A4W4_BLOCKSCALE_PYBIND;
}
