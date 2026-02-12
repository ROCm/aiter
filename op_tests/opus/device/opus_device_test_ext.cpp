// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file opus_device_test_ext.cpp
 * @brief Single PyTorch extension binding all OPUS device-test kernels.
 *
 * Exposes:
 *   opus_device_test.run_mfma_32x32x8_f16(A, B, C)
 *   opus_device_test.run_vector_add(A, B, Result)
 */

#include <torch/extension.h>
#include "test_mfma.h"
#include "test_vector_add.h"

// ---------- MFMA wrapper ----------

static void run_mfma_32x32x8_f16_torch(
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
        A.data_ptr(), B.data_ptr(), C.data_ptr(),
        stride_a, stride_b, stride_c);
}

// ---------- Vector-add wrapper ----------

static void run_vector_add_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor Result)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(Result.is_cuda(), "Result must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(Result.dtype() == torch::kFloat32, "Result must be float32");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && Result.is_contiguous(),
                "A, B, Result must be contiguous");
    TORCH_CHECK(A.dim() == 1 && B.dim() == 1 && Result.dim() == 1,
                "A, B, Result must be 1-D");
    int n = static_cast<int>(A.numel());
    TORCH_CHECK(B.numel() == n && Result.numel() == n,
                "A, B, Result must have the same number of elements");

    run_vector_add(A.data_ptr(), B.data_ptr(), Result.data_ptr(), n);
}

// ---------- Module ----------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_mfma_32x32x8_f16", &run_mfma_32x32x8_f16_torch,
          "OPUS 32x32x8 fp16 MFMA (block_v2, swap_ab): C = A @ B^T");
    m.def("run_vector_add", &run_vector_add_torch,
          "OPUS vector addition with gmem load/store: Result = A + B");
}
