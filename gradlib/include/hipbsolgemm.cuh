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
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

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
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

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
             std::optional<bool> use_gelu           = std::nullopt);

std::vector<int> hipb_findallsols(const aiter_tensor_t& mat1,
                                  const aiter_tensor_t& mat2,
                                  aiter_tensor_t& result,
                                  std::optional<aiter_tensor_t> bias   = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleA = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleB = std::nullopt,
                                  std::optional<aiter_tensor_t> scaleC = std::nullopt,
                                  bool bpreshuffle                     = false,
                                  bool use_gelu                        = false);

std::string getHipblasltKernelName(int solution_index);

int get_algoIdx_hip_tuning_csv(
  const std::string filename, hipblasLtHandle_t handle,
  const bool bpreshuffle, const bool use_rowwise,
  const hipblasOperation_t trans_a, const hipblasOperation_t trans_b,
  const int32_t m, const int32_t n, const int32_t k,
  const hipDataType A_data_type, const int32_t lda, const int64_t stride_a,
  const hipDataType B_data_type, const int32_t ldb, const int64_t stride_b,
  const hipDataType C_data_type, const int32_t ldc, const int64_t stride_c,
  const hipblasComputeType_t compute_type, const int32_t batch_count);

void append_hip_tuning_csv(
  hipblasLtMatmulAlgo_t& algo, const std::string filename,
  bool bpreshuffle, bool use_rowwise,
  hipblasOperation_t trans_a, hipblasOperation_t trans_b,int m, int n, int k,
  hipDataType A_data_type, int32_t lda, int64_t stride_a,
  hipDataType B_data_type, int32_t ldb, int64_t stride_b,
  hipDataType C_data_type, int32_t ldc, int64_t stride_c,
  hipblasComputeType_t compute_type, int32_t batch_count, bool write_header_if_missing);

hipblasStatus_t hipblasLt_online_tuning(
    hipblasLtHandle_t handle, int m, int n, int k,
    hipblasLtMatmulDesc_t matmulDesc, hipblasLtMatrixLayout_t ADesc, hipblasLtMatrixLayout_t BDesc, hipblasLtMatrixLayout_t CDesc,
    const void* A, const void* B, void* C,
    void* workspace, size_t workspaceSize, const void* alpha, const void* beta,
    std::vector<hipblasLtMatmulHeuristicResult_t>& tunedResults,
    size_t size_dA, size_t size_dB, size_t size_dC, int64_t totalRotatingSizeNeeded, hipDataType intype, hipDataType outtype,
    hipStream_t stream);


