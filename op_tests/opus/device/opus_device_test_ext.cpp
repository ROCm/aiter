// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file opus_device_test_ext.cpp
 * @brief Single PyTorch extension binding all OPUS device-test kernels.
 *
 * Exposes:
 *   opus_device_test.run_mfma(A, B, C, variant)   -- variant: "32x32x8_f16", "32x32x8_bf16", ...
 *   opus_device_test.run_mxfp(A, B, C, variant)  -- variant: "mxfp8_32x32x64", "mxfp4_16x16x128", ...
 *   opus_device_test.run_vector_add(A, B, Result)
 *   opus_device_test.run_async_load(Src, Dst)
 *   opus_device_test.run_dtype_convert(In, Out, variant)  -- variant: "fp32_bf16", "fp32_fp16",
 *                                                          "fp32_fp8", "fp32_fp4"
 */

#include <torch/extension.h>
#include "test_mfma.h"
#include "test_mxfp.h"
#include "test_vector_add.h"
#include "test_async_load.h"
#include "test_dtype_convert.h"
#include "test_load_store_if.h"

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
    // FP8/BF8 torch dtype is arch-dependent:
    //   gfx942: float8_e4m3fnuz / float8_e5m2fnuz
    //   gfx950: float8_e4m3fn   / float8_e5m2
    torch::Dtype expected_out_dtype;
    bool dtype_ok = false;
    std::string in_dtype_name;

    if (variant.find("fp8") != std::string::npos) {
        dtype_ok = (A.dtype() == torch::kFloat8_e4m3fnuz || A.dtype() == torch::kFloat8_e4m3fn);
        expected_out_dtype = torch::kFloat32;
        in_dtype_name = "float8_e4m3fnuz or float8_e4m3fn";
    } else if (variant.find("bf8") != std::string::npos) {
        dtype_ok = (A.dtype() == torch::kFloat8_e5m2fnuz || A.dtype() == torch::kFloat8_e5m2);
        expected_out_dtype = torch::kFloat32;
        in_dtype_name = "float8_e5m2fnuz or float8_e5m2";
    } else if (variant.find("bf16") != std::string::npos) {
        dtype_ok = (A.dtype() == torch::kBFloat16);
        expected_out_dtype = torch::kBFloat16;
        in_dtype_name = "bfloat16";
    } else if (variant.find("_f32") != std::string::npos) {
        dtype_ok = (A.dtype() == torch::kFloat32);
        expected_out_dtype = torch::kFloat32;
        in_dtype_name = "float32";
    } else {
        dtype_ok = (A.dtype() == torch::kFloat16);
        expected_out_dtype = torch::kFloat16;
        in_dtype_name = "float16";
    }

    TORCH_CHECK(dtype_ok, "A must be ", in_dtype_name, " for variant ", variant);
    TORCH_CHECK(B.dtype() == A.dtype(), "B must have same dtype as A for variant ", variant);
    TORCH_CHECK(C.dtype() == expected_out_dtype, "C must be ", (expected_out_dtype == torch::kFloat32 ? "float32" : in_dtype_name), " for variant ", variant);

    int stride_a = static_cast<int>(A.stride(0));
    int stride_b = static_cast<int>(B.stride(0));
    int stride_c = static_cast<int>(C.stride(0));

    if (variant == "32x32x2_f32") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 2}),  "A must be 32x2 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{32, 2}),  "B must be 32x2 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be 32x32 for variant ", variant);
        run_mfma_32x32x2_f32(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "16x16x4_f32") {
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 4}),  "A must be 16x4 for variant ", variant);
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{16, 4}),  "B must be 16x4 for variant ", variant);
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be 16x16 for variant ", variant);
        run_mfma_16x16x4_f32(A.data_ptr(), B.data_ptr(), C.data_ptr(), stride_a, stride_b, stride_c);
    } else if (variant == "32x32x8_f16") {
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

// ---------- MXFP wrapper ----------

static void run_mxfp_torch(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    const std::string& variant,
    int64_t scale_a,
    int64_t scale_b)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
                "A, B, C must be contiguous");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "C must be float32");

    int sa = static_cast<int>(scale_a);
    int sb = static_cast<int>(scale_b);

    if (variant == "mxfp8_32x32x64") {
        bool dtype_ok = (A.dtype() == torch::kFloat8_e4m3fnuz || A.dtype() == torch::kFloat8_e4m3fn);
        TORCH_CHECK(dtype_ok, "A must be float8_e4m3fnuz or float8_e4m3fn for variant ", variant);
        TORCH_CHECK(B.dtype() == A.dtype(), "B must have same dtype as A");
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 64}), "A must be [32,64]");
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{64, 32}), "B must be [64,32]");
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be [32,32]");
        run_mxfp8_32x32x64(A.data_ptr(), B.data_ptr(), C.data_ptr(), sa, sb);
    } else if (variant == "mxfp8_16x16x128") {
        bool dtype_ok = (A.dtype() == torch::kFloat8_e4m3fnuz || A.dtype() == torch::kFloat8_e4m3fn);
        TORCH_CHECK(dtype_ok, "A must be float8_e4m3fnuz or float8_e4m3fn for variant ", variant);
        TORCH_CHECK(B.dtype() == A.dtype(), "B must have same dtype as A");
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 128}), "A must be [16,128]");
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{128, 16}), "B must be [128,16]");
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be [16,16]");
        run_mxfp8_16x16x128(A.data_ptr(), B.data_ptr(), C.data_ptr(), sa, sb);
    } else if (variant == "mxfp4_32x32x64") {
        TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8 (packed fp4x2) for variant ", variant);
        TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8 (packed fp4x2) for variant ", variant);
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{32, 32}), "A must be [32,32] (packed fp4)");
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{64, 16}), "B must be [64,16] (packed fp4)");
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{32, 32}), "C must be [32,32]");
        run_mxfp4_32x32x64(A.data_ptr(), B.data_ptr(), C.data_ptr(), sa, sb);
    } else if (variant == "mxfp4_16x16x128") {
        TORCH_CHECK(A.dtype() == torch::kUInt8, "A must be uint8 (packed fp4x2) for variant ", variant);
        TORCH_CHECK(B.dtype() == torch::kUInt8, "B must be uint8 (packed fp4x2) for variant ", variant);
        TORCH_CHECK((A.sizes() == torch::IntArrayRef{16, 64}), "A must be [16,64] (packed fp4)");
        TORCH_CHECK((B.sizes() == torch::IntArrayRef{128, 8}), "B must be [128,8] (packed fp4)");
        TORCH_CHECK((C.sizes() == torch::IntArrayRef{16, 16}), "C must be [16,16]");
        run_mxfp4_16x16x128(A.data_ptr(), B.data_ptr(), C.data_ptr(), sa, sb);
    } else {
        TORCH_CHECK(false, "Unknown mxfp variant: ", variant);
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
    } else if (variant == "fp32_fp4") {
        TORCH_CHECK(n % 8 == 0,
                     "For fp32_fp4, n must be a multiple of 8 (packed x8 conversion)");
        run_dtype_convert_fp32_fp4(In.data_ptr(), Out.data_ptr(), n);
    } else {
        TORCH_CHECK(false, "Unknown dtype_convert variant: ", variant);
    }
}

// ---------- Predicated load/store wrappers ----------

static void run_predicated_copy_torch(
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
    TORCH_CHECK(Dst.numel() >= n, "Dst must be at least as large as Src");

    run_predicated_copy(Src.data_ptr(), Dst.data_ptr(), n);
}

static void run_free_func_add_torch(
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

    run_free_func_add(A.data_ptr(), B.data_ptr(), Result.data_ptr(), n);
}

static void run_predicated_async_load_torch(
    torch::Tensor Src,
    torch::Tensor Dst,
    int64_t n_padded)
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
    TORCH_CHECK(Dst.numel() >= n_padded, "Dst must be at least n_padded elements");
    TORCH_CHECK(n_padded % 256 == 0, "n_padded must be a multiple of 256");

    run_predicated_async_load(Src.data_ptr(), Dst.data_ptr(), n, static_cast<int>(n_padded));
}

// ---------- Module ----------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_mfma", &run_mfma_torch,
          "OPUS MFMA (block_v2, swap_ab): C = A @ B^T. "
          "variant: '32x32x8_f16', '32x32x8_bf16', '16x16x16_f16', '16x16x16_bf16'");
    m.def("run_mxfp", &run_mxfp_torch,
          "OPUS MXFP (gfx950 only): C = A @ B with E8M0 block scaling. "
          "variant: 'mxfp8_32x32x64', 'mxfp8_16x16x128', 'mxfp4_32x32x64', 'mxfp4_16x16x128'",
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("variant"),
          py::arg("scale_a") = 127, py::arg("scale_b") = 127);
    m.def("run_vector_add", &run_vector_add_torch,
          "OPUS vector addition with gmem load/store: Result = A + B");
    m.def("run_async_load", &run_async_load_torch,
          "OPUS async_load: copy Src -> Dst through LDS (global->LDS->global)");
    m.def("run_dtype_convert", &run_dtype_convert_torch,
          "OPUS dtype round-trip: In(fp32) -> lowp -> Out(fp32). "
          "variant: 'fp32_bf16', 'fp32_fp16', 'fp32_fp8', or 'fp32_fp4'");
    m.def("run_predicated_copy", &run_predicated_copy_torch,
          "OPUS predicated copy: Dst[i] = Src[i] where i < n, via gmem load_if/store_if");
    m.def("run_free_func_add", &run_free_func_add_torch,
          "OPUS vector add via free function API: Result = A + B");
    m.def("run_predicated_async_load", &run_predicated_async_load_torch,
          "OPUS predicated async_load: copy Src -> LDS -> Dst with bounds predicate",
          py::arg("Src"), py::arg("Dst"), py::arg("n_padded"));
}
