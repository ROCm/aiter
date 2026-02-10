// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include "test_opus_mfma.h"

void run_mfma_32x32x8_f16_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(C.dtype() == torch::kFloat16, "C must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "A, B, C must be contiguous");
    const int M = 32, N = 32, K = 8;
    TORCH_CHECK((A.sizes() == torch::IntArrayRef{M, K}), "A must be 32x8");
    TORCH_CHECK((B.sizes() == torch::IntArrayRef{N, K}), "B must be 32x8");
    TORCH_CHECK((C.sizes() == torch::IntArrayRef{M, N}), "C must be 32x32");

    int stride_a = static_cast<int>(A.stride(0));
    int stride_b = static_cast<int>(B.stride(0));
    int stride_c = static_cast<int>(C.stride(0));
    TORCH_CHECK(stride_a >= K && stride_b >= K && stride_c >= N,
                "Strides must be row-major (stride(0) >= inner dim)");

    run_mfma_32x32x8_f16(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        stride_a,
        stride_b,
        stride_c);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_mfma_32x32x8_f16", &run_mfma_32x32x8_f16_torch,
          "OPUS 32x32x8 fp16 MFMA (block_v2, swap_ab): C = B @ A^T (A 32x8, B 32x8, C 32x32)");
}
