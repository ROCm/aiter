#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// #include "moe_flatmm.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/moe_flatmm.hpp"
#include "py_itfs_common.h"
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

using MoeKernel        = std::function<torch::Tensor(torch::Tensor&,
                                              torch::Tensor&,
                                              torch::Tensor&,
                                              torch::Tensor&,
                                              torch::Tensor&,
                                              torch::Tensor&,
                                              int,
                                              std::optional<int>,
                                              std::optional<int>,
                                              std::optional<torch::Tensor>,
                                              std::optional<torch::Tensor>,
                                              std::optional<torch::Tensor>,
                                              std::optional<torch::Tensor>,
                                              std::optional<int>)>;
using ck_stream_config = ck_tile::stream_config;
using row_major        = ck_tile::tensor_layout::gemm::RowMajor;
using col_major        = ck_tile::tensor_layout::gemm::ColumnMajor;
using bf16             = ck_tile::bf16_t;
using fp16             = ck_tile::half_t;
using fp8              = ck_tile::fp8_t;
using pk_fp4           = ck_tile::pk_fp4_t;

__attribute__((visibility("default"))) torch::Tensor
cktile_moe_gemm1(torch::Tensor& XQ,
                 torch::Tensor& WQ,
                 torch::Tensor& Y,
                 torch::Tensor& sorted_ids,
                 torch::Tensor& sorted_expert_ids,
                 torch::Tensor& max_token_ids,
                 int topk,
                 std::optional<int> n_padded_zeros,
                 std::optional<int> k_padded_zeros,
                 std::optional<torch::Tensor> topk_weight,
                 std::optional<torch::Tensor> x_scale,
                 std::optional<torch::Tensor> w_scale,
                 std::optional<torch::Tensor> exp_bias,
                 std::optional<int> activation,
                 std::optional<int> block_m);

__attribute__((visibility("default"))) torch::Tensor
cktile_moe_gemm2(torch::Tensor& XQ,
                 torch::Tensor& WQ,
                 torch::Tensor& Y,
                 torch::Tensor& sorted_ids,
                 torch::Tensor& sorted_expert_ids,
                 torch::Tensor& max_token_ids,
                 int topk,
                 std::optional<int> n_padded_zeros,
                 std::optional<int> k_padded_zeros,
                 std::optional<torch::Tensor> topk_weight,
                 std::optional<torch::Tensor> x_scale,
                 std::optional<torch::Tensor> w_scale,
                 std::optional<torch::Tensor> exp_bias,
                 std::optional<int> activation,
                 std::optional<int> block_m);
