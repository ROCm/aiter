// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below
// just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by
// default in hip_fp16.h #undef __HIP_NO_HALF_OPERATORS__ #undef
// __HIP_NO_HALF_CONVERSIONS__ #endif

#include "aiter_tensor.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#if HIPBLASLT_VERSION_MAJOR < 1
// Compatibility names for internal arguments; older libraries have no scale-mode attributes.
enum hipblasLtMatmulMatrixScale_t
{
    HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F    = 0,
    HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F = 3,
};
#endif

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <sstream>
#include <fstream>
#include <filesystem>

void hipb_create_extension();

void hipb_destroy_extension();

// Torch-free: `result` is caller-allocated (Python side); outDtype is read from
// result.dtype(). See gradlib/csrc/hipbsolgemm.cu.
void hipb_mm(const aiter_tensor_t& mat1,
             const aiter_tensor_t& mat2,
             const int solution_index,
             aiter_tensor_t& result,
             std::optional<aiter_tensor_t> bias     = std::nullopt,
             std::optional<aiter_tensor_t> scaleA   = std::nullopt,
             std::optional<aiter_tensor_t> scaleB   = std::nullopt,
             std::optional<aiter_tensor_t> scaleOut = std::nullopt,
             std::optional<bool> bpreshuffle        = std::nullopt,
             std::optional<bool> use_gelu           = std::nullopt,
             std::optional<int64_t> scaleA_mode     = std::nullopt,
             std::optional<int64_t> scaleB_mode     = std::nullopt);

std::vector<int> hipb_findallsols(const aiter_tensor_t& mat1,
                                  const aiter_tensor_t& mat2,
                                  aiter_tensor_t& result,
                                  std::optional<aiter_tensor_t> bias   = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleA = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleB = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleC = std::nullopt,
                                  bool bpreshuffle                     = false,
                                  bool use_gelu                        = false,
                                  std::optional<int64_t> scaleA_mode   = std::nullopt,
                                  std::optional<int64_t> scaleB_mode   = std::nullopt);

// Set by the JIT (check_hipblaslt_has_w4a16 in aiter/jit/core.py) when the hipBLASLt headers
// declare the w4a16 API (HIP_R_4I A with the VEC*_16{BF,F}[_ZP]_EXT scale modes); without it the
// w4a16 entry points report no support.
#ifndef AITER_HIPBLASLT_HAS_W4A16
#define AITER_HIPBLASLT_HAS_W4A16 0
#endif

// Whether hipBLASLt has a w4a16 kernel on the current device for a [N, K] int4 weight with row
// stride `lda` (in int4 elements), activations with row stride `ldb` and an output with row stride
// `ldd` (in elements), at each batch size in `batch_sizes` exactly. False when aiter was built
// against a hipBLASLt without the w4a16 API.
bool hipb_w4a16_is_supported(bool bf16,
                             int64_t n,
                             int64_t k,
                             int64_t lda,
                             int64_t ldb,
                             int64_t ldd,
                             int64_t group_size,
                             bool has_zp,
                             int64_t int4_encoding,
                             const std::vector<int64_t>& batch_sizes);

// Bytes of workspace hipb_w4a16_mm needs for these tensors (0 when none).
int64_t hipb_w4a16_workspace_size(const aiter_tensor_t& x,
                                  const aiter_tensor_t& w_q,
                                  const aiter_tensor_t& result,
                                  int64_t group_size,
                                  bool has_zp,
                                  int64_t int4_encoding);

void hipb_w4a16_mm(const aiter_tensor_t& x,
                   const aiter_tensor_t& w_q,
                   const aiter_tensor_t& scale,
                   aiter_tensor_t& result,
                   std::optional<aiter_tensor_t> workspace,
                   int64_t group_size,
                   bool has_zp,
                   int64_t int4_encoding);

std::string getHipblasltKernelName(int solution_index);

int get_algoIdx_hip_tuning_csv(
  const std::string filename, hipblasLtHandle_t handle,
  const bool bpreshuffle,
  const hipblasOperation_t trans_a, const hipblasOperation_t trans_b,
  const int32_t m, const int32_t n, const int32_t k,
  const hipDataType A_data_type, const int32_t lda, const int64_t stride_a,
  const hipDataType B_data_type, const int32_t ldb, const int64_t stride_b,
  const hipDataType C_data_type, const int32_t ldc, const int64_t stride_c,
  const hipblasComputeType_t compute_type, const int32_t batch_count,
  hipblasLtMatmulMatrixScale_t scaleA_mode, hipblasLtMatmulMatrixScale_t scaleB_mode);

void append_hip_tuning_csv(
  hipblasLtMatmulAlgo_t& algo, const std::string filename,
  bool bpreshuffle,
  hipblasOperation_t trans_a, hipblasOperation_t trans_b,int m, int n, int k,
  hipDataType A_data_type, int32_t lda, int64_t stride_a,
  hipDataType B_data_type, int32_t ldb, int64_t stride_b,
  hipDataType C_data_type, int32_t ldc, int64_t stride_c,
  hipblasComputeType_t compute_type, int32_t batch_count, bool write_header_if_missing,
  hipblasLtMatmulMatrixScale_t scaleA_mode, hipblasLtMatmulMatrixScale_t scaleB_mode);

hipblasStatus_t hipblasLt_online_tuning(
    hipblasLtHandle_t handle, int m, int n, int k,
    hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t ADesc, hipblasLtMatrixLayout_t BDesc, hipblasLtMatrixLayout_t CDesc,
    const void* A, const void* B, void* C,
    void* workspace, size_t workspaceSize, const void* alpha, const void* beta,
    std::vector<hipblasLtMatmulHeuristicResult_t>& tunedResults,
    size_t size_dA, size_t size_dB, size_t size_dC, int64_t totalRotatingSizeNeeded, hipDataType intype, hipDataType outtype,
    hipStream_t stream);


