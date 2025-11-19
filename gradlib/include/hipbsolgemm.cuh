// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef __gfx908__
// // Uncomment ifdef and endif only if you need to undef the HIP_HALF ops below
// just for gfx908 and not for others
// // below lines enable hip float to half conversion which are disabled by
// default in hip_fp16.h #undef __HIP_NO_HALF_OPERATORS__ #undef
// __HIP_NO_HALF_CONVERSIONS__ #endif

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
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <tuple>

void hipb_create_extension();

void hipb_destroy_extension();

torch::Tensor hipb_mm(const torch::Tensor& mat1,
                      const torch::Tensor& mat2,
                      const int solution_index,
                      std::optional<torch::Tensor> bias        = std::nullopt,
                      std::optional<c10::ScalarType> out_dtype = std::nullopt,
                      std::optional<torch::Tensor> scaleA = std::nullopt,
                      std::optional<torch::Tensor> scaleB = std::nullopt,
                      std::optional<torch::Tensor> scaleOut = std::nullopt,
                      std::optional<bool> bpreshuffle = std::nullopt);

std::vector<int> hipb_findallsols(const torch::Tensor& mat1,
                                  const torch::Tensor& mat2,
                                  std::optional<torch::Tensor> bias        = std::nullopt,
                                  std::optional<c10::ScalarType> out_dtype = std::nullopt,
                                  std::optional<torch::Tensor> scaleA      = std::nullopt,
                                  std::optional<torch::Tensor> scaleB      = std::nullopt,
                                  std::optional<torch::Tensor> scaleC      = std::nullopt,
                                  bool bpreshuffle                         = false);

std::string getHipblasltKernelName(int solution_index);
