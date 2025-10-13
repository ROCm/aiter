// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPFunctions.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <c10/hip/HIPStream.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>

#include <hip/hip_runtime.h>
// #include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <assert.h>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>

#include <rocblas/rocblas.h>

void rocb_create_extension();

void rocb_destroy_extension();

torch::Tensor RocSolIdxBlas(const torch::Tensor& mat1,
                            const torch::Tensor& mat2,
                            const int32_t solution_index = 0);

std::vector<rocblas_int> RocFindAllSolIdxBlas(const torch::Tensor& mat1, const torch::Tensor& mat2);
