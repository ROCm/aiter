#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AIESW-32176: forward declaration for the CK WMMA W4A16 b_scale GEMM op.
// The full implementation lives in csrc/ck_w4a16/gemm_w4a16.cu.
#include <torch/all.h>
#include <torch/extension.h>
#include <optional>

torch::Tensor gemm_w4a16(at::Tensor& in_a,
                         at::Tensor& in_b,
                         at::Tensor& in_s,
                         at::Tensor& Y,
                         int64_t group_size,
                         std::optional<at::Tensor> scaled_zp);
