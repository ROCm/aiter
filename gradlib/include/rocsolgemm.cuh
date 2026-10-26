// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API

#include "aiter_tensor.h"

#include <hip/hip_runtime.h>
// #include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <assert.h>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <rocblas/rocblas.h>

void rocb_create_extension();

void rocb_destroy_extension();

// Torch-free: `result` is caller-allocated (Python side); see gradlib/csrc/rocsolgemm.cu.
void RocSolIdxBlas(const aiter_tensor_t& mat1,
                   const aiter_tensor_t& mat2,
                   aiter_tensor_t& result,
                   const int32_t solution_index = 0);

std::vector<rocblas_int>
RocFindAllSolIdxBlas(const aiter_tensor_t& mat1, const aiter_tensor_t& mat2, aiter_tensor_t& result);
