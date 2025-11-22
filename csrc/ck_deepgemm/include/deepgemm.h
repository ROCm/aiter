#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "flatmm_basic.hpp"
#include "py_itfs_common.h"
#include <torch/all.h>
#include <torch/extension.h>

template <class scaleM = ck_tile::FlatmmScalePointer<-1>,
          class scaleN = ck_tile::FlatmmScalePointer<-1>>
using m_grouped_flatmm_args = ck_tile::MaskedGroupedFlatmmHostArgs<scaleM, scaleN>;
using ck_stream_config      = ck_tile::stream_config;
using row_major             = ck_tile::tensor_layout::gemm::RowMajor;
using col_major             = ck_tile::tensor_layout::gemm::ColumnMajor;
using bf16                  = ck_tile::bf16_t;
using fp16                  = ck_tile::half_t;
using fp8                   = ck_tile::fp8_t;

torch::Tensor deepgemm(torch::Tensor& XQ,
                       torch::Tensor& WQ,
                       torch::Tensor& Y,
                       torch::Tensor& group_layout,
                       std::optional<torch::Tensor> x_scale,
                       std::optional<torch::Tensor> w_scale);
