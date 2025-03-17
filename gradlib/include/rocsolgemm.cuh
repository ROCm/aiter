// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#define ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_BETA_FEATURES_API

#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
// #include <c10/cuda/CUDACachingAllocator.h>
#include <c10/hip/HIPStream.h>
#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <ATen/cuda/CUDAEvent.h>

#include <hip/hip_runtime.h>
// #include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <assert.h>
#include "nvToolsExt.h"

#include <rocblas/rocblas.h>

void rocb_create_extension();

void rocb_destroy_extension();

torch::Tensor RocSolIdxBlas(
    const torch::Tensor &mat1,
    const torch::Tensor &mat2,
    const int32_t solution_index = 0);

std::vector<rocblas_int> RocFindAllSolIdxBlas(
    const torch::Tensor &mat1,
    const torch::Tensor &mat2);
