// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file opus_device_test_ext.cpp
 * @brief Single PyTorch extension binding all OPUS device-test kernels.
 *
 * Exposes:
 *   opus_device_test.run_mfma(A, B, C, variant)   -- variant: "32x32x8_f16", "32x32x8_bf16",
 *                                                   "16x16x16_f16", "16x16x16_bf16",
 *                                                   "32x32x16_fp8", "32x32x16_bf8",
 *                                                   "16x16x32_fp8", "16x16x32_bf8", ...
 *   opus_device_test.run_vector_add(A, B, Result)
 *   opus_device_test.run_async_load(Src, Dst)
 *   opus_device_test.run_dtype_convert(In, Out, variant)
 */

#include <torch/extension.h>
#include "test_mfma.h"
#include "test_vector_add.h"
#include "test_async_load.h"
#include "test_dtype_convert.h"

// ---------- MFMA wrapper ----------

static void run_mfma_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    const std::string& variant)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "A, B, C must be contiguous");

    // Parse variant to determine expected input/output dtypes.
    // fp8/bf8 variants use fp8/bf8 inputs and fp32 output (raw accumulator).
    torch::Dtype expected_in_dtype, expected_out_dtype;
    std::string in_dtype_name;

    if (variant.find("fp8") != std::string::npos) {
        expected_in_dtype  = torch::kFloat8_e4m3fnuz;
        expected_out_dtype = torch::kFloat32;
        in_dtype_name = "float8_e4m3fnuz";
    } else if (variant.find("bf8") != std::string::npos) {
        expected_in_dtype  = torch::kFloat8_e5m2fnuz;
        expected_out_dtype = torch::kFloat32;
        in_dtype_name = "float8_e5m2fnuz";
    } else if (variant.find("bf16") != std::string::npos) {
        expected_in_dtype  = torch::kBFloat16;
        expected_out_dtype = torch::kBFloat16;
        in_dtype_name = "bfloat16";
    } else {
        expected_in_dtype  = torch::kFloat16;
        expected_out_dtype = torch::kFloat16;
        in_dtype_name = "float16";
    }

    TORCH_CHECK(A.dtype() == expected_in_dtype,  "A must be ", in_dtype_name, " for variant ", variant);
    TORCH_CHECK(B.dtype() == expected_in_dtype,  "B must be ", in_dtype_name, " for variant ", variant);
    TORCH_CHECK(C.dtype() == expected_out_dtype, "C must be ", (expected_out_dtype == torch::kFloat32 ? "float32" : in_dtype_name), " for variant ", variant);

    int stride_a = static_cast<int>(A.stride(0));
    int stride_b = static_cast<int>(B.stride(0));
    int stride_c = static_cast<int>(C.stride(0));

    if (variant == "32x32x8_f16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 8}),  "A must be 32x8 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 8}),  "B must be 32x8 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x8_f16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "32x32x8_bf16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 8}),  "A must be 32x8 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 8}),  "B must be 32x8 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x8_bf16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x16_f16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 16}), "A must be 16x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 16}), "B must be 16x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x16_f16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x16_bf16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 16}), "A must be 16x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 16}), "B must be 16x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x16_bf16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "32x32x16_f16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 16}), "A must be 32x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 16}), "B must be 32x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x16_f16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "32x32x16_bf16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 16}), "A must be 32x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 16}), "B must be 32x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x16_bf16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x32_f16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 32}), "A must be 16x32 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 32}), "B must be 16x32 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x32_f16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x32_bf16") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 32}), "A must be 16x32 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 32}), "B must be 16x32 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x32_bf16(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    // --- FP8 / BF8 variants (fp32 output) ---
    } else if (variant == "32x32x16_fp8") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 16}), "A must be 32x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 16}), "B must be 32x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x16_fp8(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "32x32x16_bf8") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 16}), "A must be 32x16 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 16}), "B must be 32x16 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x16_bf8(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x32_fp8") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 32}), "A must be 16x32 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 32}), "B must be 16x32 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x32_fp8(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x32_bf8") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 32}), "A must be 16x32 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 32}), "B must be 16x32 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x32_bf8(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else {
        TORCH_CHECK(false, "Unknown MFMA variant: ", variant);
    }
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

// ---------- Async-load wrapper ----------

static void run_async_load_torch(
    torch::Tensor Src,
    torch::Tensor Dst)
{
    TORCH_CHECK(Src.is_cuda(), "Src must be a CUDA tensor");
    TORCH_CHECK(Dst.is_cuda(), "Dst must be a CUDA tensor");
    TORCH_CHECK(Src.dtype() == torch::kFloat32, "Src must be float32");
    TORCH_CHECK(Dst.dtype() == torch::kFloat32, "Dst must be float32");
    TORCH_CHECK(Src.is_contiguous() && Dst.is_contiguous(),
                "Src, Dst must be contiguous");
    TORCH_CHECK(Src.dim() == 1 && Dst.dim() == 1,
                "Src, Dst must be 1-D");
    int n = static_cast<int>(Src.numel());
    TORCH_CHECK(Dst.numel() == n,
                "Src and Dst must have the same number of elements");

    run_async_load(Src.data_ptr(), Dst.data_ptr(), n);
}

// ---------- Dtype-convert wrappers ----------

static void run_dtype_convert_torch(
    torch::Tensor In,
    torch::Tensor Out,
    const std::string& variant)
{
    TORCH_CHECK(In.is_cuda(), "In must be a CUDA tensor");
    TORCH_CHECK(Out.is_cuda(), "Out must be a CUDA tensor");
    TORCH_CHECK(In.dtype() == torch::kFloat32, "In must be float32");
    TORCH_CHECK(Out.dtype() == torch::kFloat32, "Out must be float32");
    TORCH_CHECK(In.is_contiguous() && Out.is_contiguous(),
                "In, Out must be contiguous");
    TORCH_CHECK(In.dim() == 1 && Out.dim() == 1,
                "In, Out must be 1-D");
    int n = static_cast<int>(In.numel());
    TORCH_CHECK(Out.numel() == n,
                "In and Out must have the same number of elements");

    if (variant == "fp32_bf16") {
        run_dtype_convert_fp32_bf16(In.data_ptr(), Out.data_ptr(), n);
    } else if (variant == "fp32_fp16") {
        run_dtype_convert_fp32_fp16(In.data_ptr(), Out.data_ptr(), n);
    } else if (variant == "fp32_fp8") {
        TORCH_CHECK(n % 4 == 0,
                     "For fp32_fp8, n must be a multiple of 4 (packed x4 conversion)");
        run_dtype_convert_fp32_fp8(In.data_ptr(), Out.data_ptr(), n);
    } else {
        TORCH_CHECK(false, "Unknown dtype_convert variant: ", variant);
    }
}

// ---------- Module ----------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_mfma", &run_mfma_torch,
          "OPUS MFMA (block_v2, swap_ab): C = A @ B^T. "
          "variant: '32x32x8_f16', '32x32x8_bf16', '16x16x16_f16', '16x16x16_bf16'");
    m.def("run_vector_add", &run_vector_add_torch,
          "OPUS vector addition with gmem load/store: Result = A + B");
    m.def("run_async_load", &run_async_load_torch,
          "OPUS async_load: copy Src -> Dst through LDS (global->LDS->global)");
    m.def("run_dtype_convert", &run_dtype_convert_torch,
          "OPUS dtype round-trip: In(fp32) -> lowp -> Out(fp32). "
          "variant: 'fp32_bf16', 'fp32_fp16', or 'fp32_fp8'");
}
